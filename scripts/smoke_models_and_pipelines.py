"""No-GUI smoke validation for AIWF model and pipeline wiring.

This is intentionally light: it imports runtime packages, checks catalog and
pipeline metadata, scans local checkpoints, and reports the defaults the GUI
would apply. It does not load full model weights or generate images; use
scripts/smoke_backend.py for real generation passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class Check:
    name: str
    ok: bool
    message: str
    details: dict[str, Any] | None = None


def _load_runtime_flags():
    from aiwf.core.config.launch import load_launch_settings, merge_launch_settings
    from aiwf.core.config.settings import RuntimeFlags

    flags = RuntimeFlags(data_dir=ROOT)
    saved = load_launch_settings(ROOT / "launch.json")
    if saved is not None:
        flags = merge_launch_settings(flags, saved)
    return flags


def _pipeline_checks(flags) -> list[Check]:
    from aiwf.core.config.settings import UserSettings
    from aiwf.services.pipeline_preflight import (
        preflight_ltx_pipeline,
        preflight_diffusers_pipeline,
        preflight_image_runtime_pipelines,
        preflight_onnx_pipeline,
        preflight_qwen_nunchaku_pipeline,
        preflight_sana_video_pipeline,
        preflight_wan_pipeline,
    )
    from aiwf.services.pipeline_registry import PipelineRegistry

    settings = UserSettings()
    registry = PipelineRegistry(flags, settings)
    checks = [
        _preflight_to_check(preflight_diffusers_pipeline()),
        _preflight_to_check(preflight_image_runtime_pipelines()),
        _preflight_to_check(preflight_qwen_nunchaku_pipeline(flags)),
        _preflight_to_check(preflight_sana_video_pipeline(flags, settings)),
        _preflight_to_check(preflight_wan_pipeline(flags, settings)),
        _preflight_to_check(preflight_ltx_pipeline(flags, settings)),
    ]

    onnx_root = (flags.resolved_models_dir() / "onnx").resolve()
    if onnx_root.exists():
        checks.append(_preflight_to_check(preflight_onnx_pipeline(onnx_root)))

    for pipeline in [*registry.image_pipelines(), *registry.video_pipelines()]:
        checks.append(
            Check(
                f"registry:{pipeline.id}",
                True,
                pipeline.message,
                {
                    "label": pipeline.label,
                    "kind": pipeline.kind,
                    "engine": pipeline.engine,
                    "ready": pipeline.ready,
                    "summary": pipeline.summary,
                },
            )
        )
    return checks


def _preflight_to_check(result) -> Check:
    return Check(
        f"preflight:{result.pipeline}",
        result.ok,
        result.markdown(),
        {
            "items": [asdict(item) for item in result.items],
            "warnings": list(result.warnings),
            "metadata": dict(result.metadata),
        },
    )


def _catalog_checks(flags) -> list[Check]:
    from aiwf.services.model_download import ModelDownloadService
    from aiwf.services.model_download_catalog import QUICK_START_BUNDLES

    service = ModelDownloadService(flags)
    checks: list[Check] = []
    required_bundles = ("flux2", "zimage", "qwen-image", "qwen-nunchaku", "sana", "sana-video", "ltx23")
    for bundle_key in required_bundles:
        keys = QUICK_START_BUNDLES.get(bundle_key, [])
        missing = [key for key in keys if service.find_catalog(key) is None]
        installed = []
        destinations = []
        for key in keys:
            entry = service.find_catalog(key)
            if entry is None:
                continue
            installed.append(service.is_catalog_installed(entry))
            target = (
                service.snapshot_destination_for(entry.category, entry.repo_id)
                if entry.snapshot
                else service.destination_for(entry.category, entry.filename)
            )
            destinations.append(str(target))
        checks.append(
            Check(
                f"catalog:{bundle_key}",
                not missing and bool(keys),
                "ok" if not missing and keys else f"missing catalog entries: {', '.join(missing) or '(empty bundle)'}",
                {"keys": keys, "installed": installed, "destinations": destinations},
            )
        )
    return checks


def _checkpoint_checks(flags) -> list[Check]:
    from aiwf.infrastructure.diffusers.checkpoints import scan_from_flags
    from aiwf.infrastructure.diffusers.model_presets import resolve_model_preset
    from aiwf.infrastructure.model_inventory import scan_and_write_model_inventory

    scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    checks: list[Check] = [
        Check(
            "checkpoint-scan",
            True,
            f"{len(checkpoints)} selectable checkpoint(s) discovered",
            {"models_dir": str(flags.resolved_models_dir()), "ckpt_dir": str(flags.resolved_ckpt_dir())},
        )
    ]
    for checkpoint in checkpoints:
        preset = resolve_model_preset({}, checkpoint.id, getattr(checkpoint, "architecture", None))
        required = {"steps", "cfg_scale", "sampler", "scheduler", "width", "height"}
        missing = sorted(required - set(preset))
        checks.append(
            Check(
                f"checkpoint:{checkpoint.id}",
                not missing,
                "defaults ok" if not missing else f"missing defaults: {', '.join(missing)}",
                {
                    "title": checkpoint.title,
                    "filename": checkpoint.filename,
                    "kind": checkpoint.kind,
                    "architecture": checkpoint.architecture,
                    "preset": preset,
                },
            )
        )
    return checks


def _print_text(checks: list[Check]) -> None:
    for check in checks:
        mark = "OK" if check.ok else "FAIL"
        print(f"[{mark}] {check.name}: {check.message.splitlines()[0]}")
        if check.details and check.name.startswith("checkpoint:"):
            preset = check.details.get("preset") or {}
            print(
                "      "
                f"{check.details.get('architecture')} "
                f"steps={preset.get('steps')} cfg={preset.get('cfg_scale')} "
                f"sampler={preset.get('sampler')} scheduler={preset.get('scheduler')} "
                f"size={preset.get('width')}x{preset.get('height')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    flags = _load_runtime_flags()
    checks = [*_pipeline_checks(flags), *_catalog_checks(flags), *_checkpoint_checks(flags)]
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2, default=str))
    else:
        _print_text(checks)
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
