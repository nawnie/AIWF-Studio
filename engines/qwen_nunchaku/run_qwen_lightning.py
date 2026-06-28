from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel


ROOT = Path(__file__).resolve().parents[2]
MODEL_BASE = ROOT / "models" / "qwen-image" / "Diffusers" / "Qwen-Image"
DOWNLOAD_BASE = ROOT / "downloads" / "qwen_nunchaku" / "base"
DEFAULT_BASE = MODEL_BASE if (MODEL_BASE / "model_index.json").is_file() else DOWNLOAD_BASE
MODEL_TRANSFORMER = ROOT / "models" / "qwen-image" / "Nunchaku" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
DOWNLOAD_TRANSFORMER = (
    ROOT / "downloads" / "qwen_nunchaku" / "transformer" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
)
DEFAULT_TRANSFORMER = MODEL_TRANSFORMER if MODEL_TRANSFORMER.is_file() else DOWNLOAD_TRANSFORMER
DEFAULT_OUTPUT = ROOT / "outputs" / "qwen_nunchaku_lightning.png"


LIGHTNING_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def _total_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def build_pipeline(base_dir: Path, transformer_path: Path, blocks_on_gpu: int):
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Missing Qwen base folder: {base_dir}")
    if not transformer_path.is_file():
        raise FileNotFoundError(f"Missing Nunchaku transformer: {transformer_path}")

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER_CONFIG)
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(str(transformer_path))
    pipe = QwenImagePipeline.from_pretrained(
        str(base_dir),
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=True)

    if _total_vram_gb() > 18:
        pipe.enable_model_cpu_offload()
    else:
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=max(1, blocks_on_gpu))
        if "transformer" not in pipe._exclude_from_cpu_offload:
            pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Nunchaku Qwen-Image Lightning without touching AIWF's main venv.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default="A small bookstore window display with a sign reading New Arrivals, crisp text, clean composition.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--blocks-on-gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-only", action="store_true", help="Load the local model and exit before generation.")
    args = parser.parse_args()

    pipe = build_pipeline(args.base_dir, args.transformer, args.blocks_on_gpu)
    print(f"Loaded Qwen Nunchaku Lightning from {args.transformer}")
    print(f"Base components: {args.base_dir}")
    print(f"CUDA available: {torch.cuda.is_available()} VRAM: {_total_vram_gb():.2f} GiB")
    if args.load_only:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(int(args.seed))
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
        generator=generator,
    ).images[0]
    image.save(args.output)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
