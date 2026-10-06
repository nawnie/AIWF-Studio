"""ai-toolkit (ostris) config generation for Qwen-Image 2.1 LoRAs.

Schema follows ``config/examples/train_lora_qwen_image_24gb.yaml`` and
``train_lora_qwen_image_edit_2509_32gb.yaml`` from the ai-toolkit repo, with
the Qwen-Image 2.1 extension (``extensions_built_in/diffusion_models/qwen_image_2``,
``arch: qwen_image_2``). Editing is not a separate arch: a dataset with
``control_path`` folders trains reference-guided editing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ARCH = "qwen_image_2"
DEFAULT_NAME_OR_PATH = "Comfy-Org/Qwen-Image-2.1"  # weights; configs/processor are pulled from Qwen/Qwen-Image-2.1


@dataclass
class TrainSpec:
    name: str = "qwen21_lora_v1"
    dataset_dir: str = ""
    control_dirs: list[str] = field(default_factory=list)  # 0-3 folders -> reference-guided edit LoRA
    training_folder: str = "output"
    name_or_path: str = DEFAULT_NAME_OR_PATH  # HF repo id, local repack folder, or single .safetensors DiT file
    trigger_word: str = ""
    default_caption: str = ""
    rank: int = 16
    alpha: int = 16
    learning_rate: float = 1e-4
    steps: int = 2000
    batch_size: int = 1
    gradient_accumulation: int = 1
    resolutions: list[int] = field(default_factory=lambda: [1024])
    optimizer: str = "adamw8bit"
    timestep_type: str = "shift"  # ai-toolkit: sigmoid | linear | shift | weighted ...
    caption_dropout: float = 0.05
    save_every: int = 250
    max_saves: int = 4
    save_dtype: str = "float16"
    # memory
    quantize: bool = True
    qtype: str = "convrot8"
    quantize_te: bool = True
    qtype_te: str = "convrot8"
    low_vram: bool = True
    layer_offloading: bool = True
    layer_offloading_transformer_percent: float = 1.0
    layer_offloading_text_encoder_percent: float = 1.0
    cache_latents_to_disk: bool = True
    cache_text_embeddings: bool = True
    gradient_checkpointing: bool = True
    # model kwargs (qwen_image_2 extension)
    rgba: bool = False
    control_image_max_pixels: int = 1024 * 1024
    match_target_res: bool = True
    # sampling during training (off by default on 16 GB: costs time and VRAM)
    sampling: bool = False
    sample_every: int = 250
    sample_prompts: list[str] = field(default_factory=list)
    sample_control_images: list[str] = field(default_factory=list)
    sample_steps: int = 25
    sample_guidance: float = 1.0
    sample_size: int = 1024
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainSpec":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


PRESET_16GB = dict(quantize=True, qtype="convrot8", quantize_te=True, qtype_te="convrot8", low_vram=True,
                   layer_offloading=True, layer_offloading_transformer_percent=1.0,
                   layer_offloading_text_encoder_percent=1.0, cache_latents_to_disk=True,
                   cache_text_embeddings=True, gradient_checkpointing=True, batch_size=1, rank=16, alpha=16,
                   resolutions=[1024], sampling=False)
PRESET_24GB = dict(PRESET_16GB, layer_offloading=False, resolutions=[768, 1024], rank=32, alpha=32)


def apply_preset(spec: TrainSpec, preset: dict[str, Any]) -> TrainSpec:
    for key, value in preset.items():
        setattr(spec, key, list(value) if isinstance(value, list) else value)
    return spec


def validate_spec(spec: TrainSpec) -> list[str]:
    problems = []
    if not spec.name.strip():
        problems.append("name is required")
    if not spec.dataset_dir or not Path(spec.dataset_dir).is_dir():
        problems.append("dataset_dir must be an existing folder")
    if len(spec.control_dirs) > 3:
        problems.append("ai-toolkit supports at most 3 control folders")
    if spec.rank < 1 or spec.alpha < 1:
        problems.append("rank/alpha must be >= 1")
    if spec.steps < 1:
        problems.append("steps must be >= 1")
    if any(r % 32 for r in spec.resolutions):
        problems.append("resolutions should be multiples of 32 (the 2.1 DiT groups latents in 2x2 blocks)")
    if spec.trigger_word and spec.cache_text_embeddings:
        problems.append("trigger_word is ignored when cache_text_embeddings is on (ai-toolkit); "
                        "bake the trigger into the captions instead (Train tab: 'Prepend trigger')")
    return problems


def build_config(spec: TrainSpec) -> dict[str, Any]:
    """Return the ai-toolkit job dict (dump with yaml.safe_dump)."""
    dataset: dict[str, Any] = {
        "folder_path": spec.dataset_dir,
        "caption_ext": "txt",
        "caption_dropout_rate": float(spec.caption_dropout),
        "shuffle_tokens": False,
        "cache_latents_to_disk": bool(spec.cache_latents_to_disk),
        "resolution": list(spec.resolutions),
    }
    if spec.default_caption:
        dataset["default_caption"] = spec.default_caption
    if spec.control_dirs:
        dataset["control_path"] = list(spec.control_dirs) if len(spec.control_dirs) > 1 else spec.control_dirs[0]

    train: dict[str, Any] = {
        "batch_size": int(spec.batch_size),
        "steps": int(spec.steps),
        "gradient_accumulation": int(spec.gradient_accumulation),
        "train_unet": True,
        "train_text_encoder": False,
        "gradient_checkpointing": bool(spec.gradient_checkpointing),
        "noise_scheduler": "flowmatch",
        "timestep_type": spec.timestep_type,
        "optimizer": spec.optimizer,
        "lr": float(spec.learning_rate),
        "dtype": "bf16",
        "cache_text_embeddings": bool(spec.cache_text_embeddings),
        "unload_text_encoder": bool(spec.cache_text_embeddings),
        "disable_sampling": not spec.sampling,
    }
    model: dict[str, Any] = {
        "name_or_path": spec.name_or_path,
        "arch": ARCH,
        "quantize": bool(spec.quantize),
        "qtype": spec.qtype,
        "quantize_te": bool(spec.quantize_te),
        "qtype_te": spec.qtype_te,
        "low_vram": bool(spec.low_vram),
        "layer_offloading": bool(spec.layer_offloading),
        "layer_offloading_transformer_percent": float(spec.layer_offloading_transformer_percent),
        "layer_offloading_text_encoder_percent": float(spec.layer_offloading_text_encoder_percent),
        "model_kwargs": {
            "rgba": bool(spec.rgba),
            "control_image_max_pixels": int(spec.control_image_max_pixels),
            "match_target_res": bool(spec.match_target_res),
        },
    }
    sample: dict[str, Any] = {
        "sampler": "flowmatch",
        "sample_every": int(spec.sample_every),
        "sample_start_step": 0,
        "width": int(spec.sample_size),
        "height": int(spec.sample_size),
        "neg": "",
        "seed": int(spec.seed),
        "walk_seed": True,
        "guidance_scale": float(spec.sample_guidance),
        "sample_steps": int(spec.sample_steps),
    }
    prompts = list(spec.sample_prompts) or ["[trigger] portrait photo, soft window light, 85mm" if spec.trigger_word
                                              else "portrait photo, soft window light, 85mm"]
    if spec.control_dirs and spec.sample_control_images:
        sample["samples"] = []
        for prompt in prompts:
            entry: dict[str, Any] = {"prompt": prompt}
            for index, img in enumerate(spec.sample_control_images[:3], start=1):
                entry[f"ctrl_img_{index}"] = img
            sample["samples"].append(entry)
    else:
        sample["prompts"] = prompts

    process: dict[str, Any] = {
        "type": "diffusion_trainer",
        "training_folder": spec.training_folder,
        "device": "cuda:0",
        "network": {"type": "lora", "linear": int(spec.rank), "linear_alpha": int(spec.alpha)},
        "save": {"dtype": spec.save_dtype, "save_every": int(spec.save_every),
                 "max_step_saves_to_keep": int(spec.max_saves)},
        "datasets": [dataset],
        "train": train,
        "model": model,
        "sample": sample,
    }
    if spec.trigger_word and not spec.cache_text_embeddings:
        process["trigger_word"] = spec.trigger_word
    return {"job": "extension", "config": {"name": spec.name, "process": [process]},
            "meta": {"name": "[name]", "version": "1.0"}}


def write_config(spec: TrainSpec, path: str | Path) -> Path:
    import yaml  # PyYAML is in the app requirements

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(build_config(spec), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def build_command(python_exe: str | Path, ai_toolkit_dir: str | Path, config_path: str | Path) -> list[str]:
    return [str(python_exe), str(Path(ai_toolkit_dir) / "run.py"), str(config_path)]


def expected_output_lora(spec: TrainSpec, ai_toolkit_dir: str | Path) -> Path:
    folder = Path(spec.training_folder)
    if not folder.is_absolute():
        folder = Path(ai_toolkit_dir) / folder
    return folder / spec.name / f"{spec.name}.safetensors"
