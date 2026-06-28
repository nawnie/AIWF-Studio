"""Probe the isolated LTX runtime without running video generation."""
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


def main() -> int:
    from aiwf.core.domain.ltx import LTX_GEMMA_BACKEND_GGUF, LtxVideoRequest
    from aiwf.core.config.settings import UserSettings
    from aiwf.services.ltx import LtxService, LtxUnavailable

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", action="store_true", help="Probe the native Gemma GGUF hidden-state route.")
    parser.add_argument("--gguf-path", type=Path, help="Gemma GGUF path. Defaults to the local Heretic Q3 file.")
    parser.add_argument("--gemma-root", type=Path, help="Gemma tokenizer/processor sidecar folder.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    flags = _load_runtime_flags()
    service = LtxService(flags, UserSettings())
    payload: dict[str, object] = {"ok": False, "mode": "runtime", "events": []}

    try:
        if args.gguf:
            request = LtxVideoRequest(
                gemma_backend=LTX_GEMMA_BACKEND_GGUF,
                gemma_gguf_path=str(args.gguf_path or ""),
                gemma_root=str(args.gemma_root or ""),
            )
            events = service.probe_gemma_gguf(request)
            payload.update({"ok": True, "mode": "gguf", "events": events})
        else:
            status = service.registry.status("ltx")
            payload.update(
                {
                    "ok": status.ready,
                    "mode": "runtime",
                    "enabled": status.enabled,
                    "messages": list(status.messages),
                    "python_exe": str(status.python_exe),
                    "worker_script": str(status.worker_script),
                    "repo_dir": str(status.repo_dir) if status.repo_dir is not None else "",
                }
            )
    except LtxUnavailable as exc:
        payload.update({"ok": False, "mode": "gguf" if args.gguf else "runtime", "error": str(exc)})

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if payload.get("ok"):
            print("LTX probe OK")
        else:
            print("LTX probe blocked")
        if payload.get("error"):
            print(payload["error"])
        elif payload.get("messages"):
            print("; ".join(str(item) for item in payload["messages"]))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
