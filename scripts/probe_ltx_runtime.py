"""Probe the isolated LTX runtime without running video generation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_GGUF_INVENTORY = ROOT / "_local" / "logs" / "ltx_gemma_gguf_inventory_latest.json"
QUANT_RE = re.compile(r"(Q[2-8](?:_[A-Z0-9]+){0,2}|IQ[1-4]_[A-Z0-9]+|F16|BF16|FP16|FP8)", re.IGNORECASE)


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
    parser.add_argument(
        "--gguf-inventory",
        action="store_true",
        help="Scan local Gemma GGUF candidates metadata-only and write an inventory receipt.",
    )
    parser.add_argument("--gguf-path", type=Path, help="Gemma GGUF path. Defaults to the local Heretic Q3 file.")
    parser.add_argument("--gemma-root", type=Path, help="Gemma tokenizer/processor sidecar folder.")
    parser.add_argument(
        "--inventory-root",
        type=Path,
        action="append",
        default=[],
        help="Extra root to scan for Gemma GGUF files. May be passed multiple times.",
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=DEFAULT_GGUF_INVENTORY,
        help="JSON receipt path for --gguf-inventory.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Return exit code 0 when the GGUF route is present but blocked cleanly by the hidden-state contract.",
    )
    args = parser.parse_args()

    flags = _load_runtime_flags()
    service = LtxService(flags, UserSettings())
    payload: dict[str, object] = {"ok": False, "mode": "runtime", "events": []}

    try:
        if args.gguf_inventory:
            payload = _gemma_gguf_inventory(flags, roots=args.inventory_root)
            output_path = args.inventory_output
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                payload["receipt_path"] = str(output_path)
        elif args.gguf:
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
        error = str(exc)
        recent_output = _extract_recent_output(error)
        payload.update(
            {
                "ok": False,
                "mode": "gguf" if args.gguf else "runtime",
                "blocked_cleanly": _is_clean_ltx_gguf_blocker(error),
                "error": error,
            }
        )
        if recent_output:
            payload["recent_output"] = recent_output

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
    if payload.get("ok"):
        return 0
    if args.allow_blocked and payload.get("blocked_cleanly"):
        return 0
    return 1


def _gemma_gguf_inventory(flags, *, roots: list[Path]) -> dict[str, object]:  # noqa: ANN001
    selected_roots = _gguf_inventory_roots(flags, extra_roots=roots)
    assets = [_inspect_gguf_candidate(path) for path in _iter_gemma_gguf_candidates(selected_roots)]
    _mark_smallest_heretic(assets)
    smallest = next((item for item in assets if item.get("smallest_heretic")), None)
    return {
        "ok": True,
        "mode": "gguf-inventory",
        "roots": [str(path) for path in selected_roots],
        "candidate_count": len(assets),
        "smallest_heretic_path": str(smallest.get("path")) if smallest else "",
        "capabilities": _runtime_capability_snapshot(),
        "assets": assets,
    }


def _gguf_inventory_roots(flags, *, extra_roots: list[Path]) -> list[Path]:  # noqa: ANN001
    roots = [
        flags.resolved_models_dir() / "LLM" / "GGUF",
        flags.resolved_models_dir() / "ltx" / "text_encoder",
        Path.home() / "Downloads",
        *extra_roots,
    ]
    seen: set[str] = set()
    result: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _iter_gemma_gguf_candidates(roots: list[Path]) -> list[Path]:
    seen: set[str] = set()
    candidates: list[Path] = []
    for root in roots:
        if root.is_file():
            paths = [root]
        else:
            paths = sorted(root.rglob("*.gguf"), key=lambda item: item.name.lower())
        for path in paths:
            if not _is_gemma_gguf_candidate(path):
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            key = os.path.normcase(str(resolved))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(resolved)
    return candidates


def _is_gemma_gguf_candidate(path: Path) -> bool:
    if path.suffix.lower() != ".gguf":
        return False
    text = path.as_posix().lower()
    return "gemma" in text or "heretic" in text


def _inspect_gguf_candidate(path: Path) -> dict[str, object]:
    stat = path.stat()
    item: dict[str, object] = {
        "path": str(path),
        "filename": path.name,
        "size_gib": round(stat.st_size / (1024**3), 3),
        "quantization": _quant_from_filename(path.name),
        "is_heretic": _is_heretic_candidate(path),
        "selected_default": path.name.lower() == "gemma-3-12b-it-heretic-q3_k_m.gguf",
        "storage": "gguf",
        "runtime_backend": "metadata-only",
        "generation_ready": False,
        "blocker": "No installed quantized GGUF backend exposes the Gemma hidden-state tuple plus attention mask required by LTX.",
    }
    try:
        import gguf

        reader = gguf.GGUFReader(str(path))
        architecture = _gguf_field_text(reader, "general.architecture")
        hidden_size = _gguf_field_int(reader, "gemma3.embedding_length")
        layer_count = _gguf_field_int(reader, "gemma3.block_count")
        expected_hidden_states = layer_count + 1 if layer_count is not None else None
        item.update(
            {
                "ok": True,
                "architecture": architecture,
                "name": _gguf_field_text(reader, "general.name"),
                "hidden_size": hidden_size,
                "layers": layer_count,
                "expected_hidden_states": expected_hidden_states,
                "heads": _gguf_field_text(reader, "gemma3.attention.head_count"),
                "tensor_count": len(reader.tensors),
                "ltx_contract_metadata_ready": bool(
                    architecture.lower() == "gemma3" and hidden_size is not None and layer_count is not None
                ),
            }
        )
    except Exception as exc:
        item.update({"ok": False, "error": str(exc)})
    return item


def _mark_smallest_heretic(assets: list[dict[str, object]]) -> None:
    heretics = [item for item in assets if item.get("is_heretic") and isinstance(item.get("size_gib"), float)]
    if not heretics:
        return
    smallest = min(heretics, key=lambda item: float(item["size_gib"]))
    for item in assets:
        item["smallest_heretic"] = item is smallest


def _is_heretic_candidate(path: Path) -> bool:
    return "heretic" in path.name.lower()


def _quant_from_filename(filename: str) -> str:
    matches = QUANT_RE.findall(filename)
    if not matches:
        return ""
    return matches[-1].upper()


def _runtime_capability_snapshot() -> dict[str, object]:
    ltx_python = ROOT / "engines" / "ltx" / ".venv" / "Scripts" / "python.exe"
    return {
        "app_gguf": _module_available("gguf"),
        "app_transformers": _module_available("transformers"),
        "app_bitsandbytes": _module_available("bitsandbytes"),
        "app_tensorrt_llm": _module_available("tensorrt_llm"),
        "ltx_gguf": _module_available_for_python(ltx_python, "gguf"),
        "ltx_transformers": _module_available_for_python(ltx_python, "transformers"),
        "ltx_bitsandbytes": _module_available_for_python(ltx_python, "bitsandbytes"),
        "ltx_llama_cpp": _module_available_for_python(ltx_python, "llama_cpp"),
        "ltx_tensorrt_llm": _module_available_for_python(ltx_python, "tensorrt_llm"),
        "bitsandbytes_applicability": "Transformers/HF safetensors quantized loading only; not a GGUF hidden-state backend.",
        "tensorrt_llm_applicability": "Future LLM serving or converted-engine lane; not a drop-in LTX GGUF text-encoder backend.",
    }


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _module_available_for_python(python_exe: Path, module: str) -> bool:
    if not python_exe.is_file():
        return False
    code = f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec({module!r}) else 1)"
    try:
        result = subprocess.run(
            [str(python_exe), "-c", code],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _gguf_field_text(reader, key: str) -> str:  # noqa: ANN001
    field = reader.fields.get(key)
    if field is None:
        return ""
    try:
        contents = field.contents()
        if isinstance(contents, bytes):
            return contents.decode("utf-8", errors="replace")
        if hasattr(contents, "tolist"):
            contents = contents.tolist()
        if isinstance(contents, list) and len(contents) == 1:
            value = contents[0]
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)
        return str(contents)
    except Exception:
        try:
            return field.parts[-1].tobytes().decode("utf-8", errors="replace")
        except Exception:
            return ""


def _gguf_field_int(reader, key: str) -> int | None:  # noqa: ANN001
    text = _gguf_field_text(reader, key)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _extract_recent_output(error: str) -> str:
    marker = "Recent LTX output:"
    if marker not in error:
        return ""
    return error.split(marker, 1)[1].strip()


def _is_clean_ltx_gguf_blocker(error: str) -> bool:
    lowered = error.lower()
    return (
        "native gemma gguf" in lowered
        and "generation is blocked" in lowered
        and "hidden" in lowered
        and "attention mask" in lowered
    )


if __name__ == "__main__":
    raise SystemExit(main())
