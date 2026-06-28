"""Report AIWF pipeline/model readiness without running inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_runtime_flags():
    from aiwf.core.config.launch import load_launch_settings, merge_launch_settings
    from aiwf.core.config.settings import RuntimeFlags

    flags = RuntimeFlags(data_dir=ROOT)
    saved = load_launch_settings(ROOT / "launch.json")
    if saved is not None:
        flags = merge_launch_settings(flags, saved)
    return flags


def _print_text(records, summary: dict[str, int]) -> None:  # noqa: ANN001
    print("Pipeline readiness ledger")
    print("Status counts: " + ", ".join(f"{key}={value}" for key, value in summary.items() if value))
    current_family = ""
    for record in records:
        if record.family != current_family:
            current_family = record.family
            print(f"\n[{current_family}]")
        path = f" ({record.path})" if record.path else ""
        print(f"- {record.status:20} {record.id} -> {record.route}{path}")
        print(f"  {record.reason}")
        if record.suggested_action:
            print(f"  next: {record.suggested_action}")


def main() -> int:
    from aiwf.services.pipeline_readiness import collect_pipeline_readiness, readiness_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--no-downloads", action="store_true", help="Skip scanning the user's Downloads folder.")
    parser.add_argument(
        "--downloads-root",
        action="append",
        type=Path,
        help="Additional or replacement Downloads root to scan. Can be passed more than once.",
    )
    parser.add_argument("--output", type=Path, help="Optional path for writing the JSON ledger.")
    parser.add_argument("--force-rescan", action="store_true", help="Force a fresh model inventory scan.")
    args = parser.parse_args()

    flags = _load_runtime_flags()
    download_roots = tuple(args.downloads_root or ()) or None
    records = collect_pipeline_readiness(
        flags,
        include_downloads=not args.no_downloads,
        download_roots=download_roots,
        force_rescan=args.force_rescan,
    )
    summary = readiness_summary(records)
    payload = {"summary": summary, "records": [record.to_dict() for record in records]}
    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_text(records, summary)
        if args.output:
            print(f"\nWrote JSON ledger: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
