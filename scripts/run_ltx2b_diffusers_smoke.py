from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aiwf.services.ltx_diffusers import run_ltx2b_diffusers


DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "ltx" / "checkpoints" / "ltx-video-2b-v0.9.5.safetensors"
DEFAULT_T5 = REPO_ROOT / "models" / "flux" / "Textencoder" / "t5xxl_fp16.safetensors"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "ltx-videos" / "ltx2b-diffusers-smoke.mp4"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded local Diffusers LTX 2B smoke test.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--t5-weights", type=Path, default=DEFAULT_T5)
    parser.add_argument("--tokenizer", type=str, default=os.environ.get("AIWF_T5_TOKENIZER", "google/t5-v1_1-xxl"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default="a calm ocean at sunrise, gentle waves, cinematic lighting")
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    started = time.time()
    try:
        result = run(args)
    except Exception as exc:  # noqa: BLE001 - smoke script should return readable failure JSON.
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checkpoint": str(args.checkpoint),
            "t5_weights": str(args.t5_weights),
            "output": str(args.output),
            "seconds": round(time.time() - started, 2),
        }
        print(json.dumps(payload, indent=2) if args.json else f"LTX 2B smoke failed: {payload}")
        return 1

    result["seconds"] = round(time.time() - started, 2)
    print(json.dumps(result, indent=2) if args.json else f"LTX 2B smoke wrote {result['output']}")
    return 0


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = args.checkpoint.resolve()
    t5_weights = args.t5_weights.resolve()
    output = args.output.resolve()
    result = run_ltx2b_diffusers(
        checkpoint=checkpoint,
        t5_weights=t5_weights,
        tokenizer_id=args.tokenizer,
        output=output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        steps=args.steps,
        seed=args.seed,
        max_sequence_length=args.max_sequence_length,
        guidance_scale=args.guidance_scale,
    )

    return {
        "ok": True,
        "route": "ltx-0.9.5-diffusers-local-t5xxl",
        "checkpoint": str(checkpoint),
        "t5_weights": str(t5_weights),
        "output": str(result.output_path),
        "bytes": result.bytes,
        "width": result.width,
        "height": result.height,
        "frames": result.frame_count,
        "fps": result.fps,
        "steps": args.steps,
    }


if __name__ == "__main__":
    raise SystemExit(main())
