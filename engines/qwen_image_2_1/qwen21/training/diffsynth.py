"""DiffSynth-Studio (ModelScope) training command builder for Qwen-Image 2.1.

Mirrors ``examples/qwen_image_21/model_training/lora/Qwen-Image-2.1.sh`` from
DiffSynth-Studio: ``accelerate launch examples/qwen_image_21/model_training/train.py``
with ``--lora_base_model dit``. Edit training adds ``--data_file_keys image,edit_image``
and ``--extra_inputs edit_image`` with a metadata.json that lists the reference
image(s) per sample.

Memory flags exposed by the script: ``--use_gradient_checkpointing``,
``--use_gradient_checkpointing_offload``, ``--fp8_models <model entry>`` (safe for
the frozen text encoder / VAE), ``--quant_options "<entry>:bitsandbytes_nf4"``,
``--initialize_model_on_cpu``. A bf16 7B DiT alone is 14 GB, so on a 16 GB card
this recipe is the reference path rather than the fast one; ai-toolkit with
int8 ConvRot + layer offloading is what we recommend there.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen-Image-2.1"
DIT_ENTRY = f"{MODEL_ID}:transformer/diffusion_pytorch_model*.safetensors"
TE_ENTRY = f"{MODEL_ID}:text_encoder/model*.safetensors"
VAE_ENTRY = f"{MODEL_ID}:vae/diffusion_pytorch_model*.safetensors"


@dataclass
class DiffSynthSpec:
    name: str = "Qwen-Image-2.1_lora"
    dataset_dir: str = ""
    metadata_path: str = ""  # metadata.csv (t2i) or metadata.json (edit)
    edit_mode: bool = False
    output_path: str = ""
    learning_rate: float = 1e-4
    num_epochs: int = 5
    dataset_repeat: int = 50
    lora_rank: int = 32
    lora_target_modules: str = ""  # "" = the script's default set for this model
    max_pixels: int = 1024 * 1024
    gradient_checkpointing: bool = True
    gradient_checkpointing_offload: bool = True
    fp8_text_encoder: bool = True
    nf4_text_encoder: bool = False
    initialize_on_cpu: bool = True
    save_steps: int | None = None
    extra_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_command(spec: DiffSynthSpec, python_exe: str | Path, diffsynth_dir: str | Path) -> list[str]:
    diffsynth_dir = Path(diffsynth_dir)
    script = diffsynth_dir / "examples" / "qwen_image_21" / "model_training" / "train.py"
    cmd = [str(python_exe), "-m", "accelerate.commands.launch", str(script),
           "--dataset_base_path", spec.dataset_dir,
           "--dataset_metadata_path", spec.metadata_path or str(Path(spec.dataset_dir) / ("metadata.json" if spec.edit_mode else "metadata.csv")),
           "--max_pixels", str(int(spec.max_pixels)),
           "--dataset_repeat", str(int(spec.dataset_repeat)),
           "--model_id_with_origin_paths", ",".join((DIT_ENTRY, TE_ENTRY, VAE_ENTRY)),
           "--learning_rate", repr(float(spec.learning_rate)),
           "--num_epochs", str(int(spec.num_epochs)),
           "--remove_prefix_in_ckpt", "pipe.dit.",
           "--output_path", spec.output_path or f"./models/train/{spec.name}",
           "--lora_base_model", "dit",
           "--lora_target_modules", spec.lora_target_modules,
           "--lora_rank", str(int(spec.lora_rank)),
           "--find_unused_parameters"]
    if spec.edit_mode:
        cmd += ["--data_file_keys", "image,edit_image", "--extra_inputs", "edit_image"]
    if spec.gradient_checkpointing:
        cmd.append("--use_gradient_checkpointing")
    if spec.gradient_checkpointing_offload:
        cmd.append("--use_gradient_checkpointing_offload")
    if spec.fp8_text_encoder:
        cmd += ["--fp8_models", TE_ENTRY]
    if spec.nf4_text_encoder:
        cmd += ["--quant_options", f"{TE_ENTRY}:bitsandbytes_nf4"]
    if spec.initialize_on_cpu:
        cmd.append("--initialize_model_on_cpu")
    if spec.save_steps:
        cmd += ["--save_steps", str(int(spec.save_steps))]
    cmd += list(spec.extra_args)
    return cmd


def command_to_shell(cmd: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(c) if c else '""' for c in cmd)
