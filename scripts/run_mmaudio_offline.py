#!/usr/bin/env python3
"""Run the upstream MMAudio demo with all runtime downloads disabled."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _require_local_model_assets(config) -> None:  # noqa: ANN001
    required = [config.model_path, config.vae_path, config.synchformer_ckpt]
    if config.bigvgan_16k_path is not None:
        required.append(config.bigvgan_16k_path)
    missing = [str(path) for path in required if not Path(path).is_file() or Path(path).stat().st_size <= 0]
    if missing:
        raise RuntimeError(
            "MMAudio assets are missing or empty; install the selected model from Audio model setup first: "
            + ", ".join(missing)
        )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_mmaudio_offline.py <demo.py> [demo arguments...]")
    demo = Path(sys.argv[1]).expanduser().resolve(strict=True)
    from mmaudio.eval_utils import ModelConfig

    # The upstream demo calls this before loading. Replace the downloader with
    # validation so a render can never fall back to requests.get().
    ModelConfig.download_if_needed = _require_local_model_assets
    sys.argv = [str(demo), *sys.argv[2:]]
    runpy.run_path(str(demo), run_name="__main__")


if __name__ == "__main__":
    main()
