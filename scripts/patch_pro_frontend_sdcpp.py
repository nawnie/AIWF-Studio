from __future__ import annotations

from pathlib import Path

from ensure_pro_frontend import patch_app_tsx


if __name__ == "__main__":
    patch_app_tsx(Path(__file__).resolve().parents[1])
