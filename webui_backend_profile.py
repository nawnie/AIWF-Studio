from __future__ import annotations

import argparse
import os
import sys

from aiwf.runtime.bootstrap_env import apply_from_argv


def _normalize_backend(value: str | None) -> str:
    normalized = (value or "diffusers").strip().lower().replace("_", "-")
    aliases = {
        "stable-diffusion.cpp": "sdcpp",
        "stable-diffusion-cpp": "sdcpp",
        "sdcpp": "sdcpp",
        "sd-cpp": "sdcpp",
        "dual": "dual",
        "both": "dual",
        "diffusers": "diffusers",
        "onnx": "onnx",
    }
    return aliases.get(normalized, "diffusers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch AIWF Studio Pro with a selected image backend profile.")
    parser.add_argument(
        "--backend",
        default=os.environ.get("AIWF_PROFILE_BACKEND", "diffusers"),
        help="Image backend profile: diffusers, dual, sdcpp, or onnx.",
    )
    args, passthrough = parser.parse_known_args()
    backend = _normalize_backend(args.backend)

    # Remove this wrapper's backend flag before app_pro/app.py parse the rest of the normal launch args.
    sys.argv = [sys.argv[0], *passthrough]
    apply_from_argv(passthrough)
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ["AIWF_PROFILE_BACKEND"] = backend

    from aiwf import app_pro

    original_resolve_flags = app_pro._resolve_flags

    def resolve_profile_flags():
        flags = original_resolve_flags()
        return flags.model_copy(update={"inference_backend": backend})

    app_pro._resolve_flags = resolve_profile_flags
    app_pro.main()


if __name__ == "__main__":
    main()
