from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

from sdcpp_smoke_test import find_sd_cli, repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experimental stable-diffusion.cpp video smoke test.")
    parser.add_argument("--sd-cli", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", default="sdcpp-video-smoke")
    parser.add_argument("--prompt", default="AIWF stable-diffusion.cpp video smoke test, moving camera, sharp details")
    parser.add_argument("--negative-prompt", default="blurry, low quality")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--backend", default=os.environ.get("AIWF_SDCPP_BACKEND", "cuda0"))
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    out_dir = repo_root() / "outputs" / "sdcpp-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
    cmd = [
        find_sd_cli(args.sd_cli),
        "-M", "vid_gen",
        "-m", args.model,
        "-p", args.prompt,
        "-n", args.negative_prompt,
        "-o", str(output),
        "-W", str(args.width),
        "-H", str(args.height),
        "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg_scale),
        "-s", str(args.seed),
        "--video-frames", str(args.frames),
        "--fps", str(args.fps),
        "--backend", args.backend,
        *args.extra,
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_root()))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
