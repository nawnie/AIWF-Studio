from __future__ import annotations

from aiwf.core.config.settings import RuntimeFlags
from aiwf.services.model_download import ModelDownloadService


DEFAULT_SD15_CATALOG_KEY = "hf-sd15-pruned"


def main() -> int:
    flags = RuntimeFlags()
    service = ModelDownloadService(flags)
    entry = service.find_catalog(DEFAULT_SD15_CATALOG_KEY)
    if entry is None:
        raise SystemExit(f"Missing model catalog entry: {DEFAULT_SD15_CATALOG_KEY}")
    if service.is_catalog_installed(entry):
        print(f"{entry.title} is already installed.")
        return 0
    path = service.download_catalog(DEFAULT_SD15_CATALOG_KEY)
    print(f"Installed {entry.title}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
