"""Check the standalone engine API package lock without importing Torch."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


def main() -> int:
    ready = sys.version_info[:2] == (3, 12)
    try:
        manifest = Path(sys.argv[1])
        required: dict[str, str] = {}
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, version = line.partition("==")
            if not separator or not name or not version:
                ready = False
                break
            required[name] = version
        if not required:
            ready = False
        if ready:
            ready = all(importlib.metadata.version(name) == version for name, version in required.items())
        if ready:
            import fastapi  # noqa: F401
            import httpx  # noqa: F401
            import pydantic  # noqa: F401
            import pydantic_settings  # noqa: F401
            import uvicorn  # noqa: F401
            from PIL import Image  # noqa: F401
            ready = "torch" not in sys.modules
    except Exception:
        ready = False
    print("READY" if ready else "NOT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
