from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_sd_cli(explicit: str = "") -> str:
    if explicit and Path(explicit).is_file():
        return str(Path(explicit).resolve())
    env = os.environ.get("AIWF_SDCPP_BINARY", "")
    if env and Path(env).is_file():
        return str(Path(env).resolve())
    root = repo_root()
    for candidate in (
        root / "tools" / "stable-diffusion.cpp" / "bin" / "sd-cli.exe",
        root / "tools" / "stable-diffusion.cpp" / "build" / "bin" / "Release" / "sd-cli.exe",
        root / "tools" / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli.exe",
        root / "tools" / "stable-diffusion.cpp" / "bin" / "sd-cli",
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    raise SystemExit("sd-cli not found. Pass --sd-cli or run scripts/install_sdcpp.ps1.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a stable-diffusion.cpp image smoke test.")
    parser.add_argument("--sd-cli", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", default="sdcpp-image-smoke")
    parser.add_argument("--prompt", default="AIWF stable-diffusion.cpp smoke test, sharp details")
    parser.add_argument("--negative-prompt", default="blurry, low quality")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default=os.environ.get("AIWF_SDCPP_BACKEND", "cuda0"))
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    out_dir = repo_root() / "outputs" / "sdcpp-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
    cmd = [
        find_sd_cli(args.sd_cli),
        "-m", args.model,
        "-p", args.prompt,
        "-n", args.negative_prompt,
        "-o", str(output),
        "-W", str(args.width),
        "-H", str(args.height),
        "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg_scale),
        "-s", str(args.seed),
        "--backend", args.backend,
        *args.extra,
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_root()))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
