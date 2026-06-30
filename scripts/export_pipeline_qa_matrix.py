from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_READINESS = Path("_local/logs/pipeline_readiness_with_downloads_latest.json")
DEFAULT_OUTPUT = Path("docs/qa/pipeline_feature_testing_matrix.csv")
LTX_SMOKETEST_LOG = Path("scripts/ltx_smoketest.log")
GGUF_INVENTORY_RECEIPT = "F:\\AIWF_Studio\\_local\\logs\\ltx_gemma_gguf_inventory_latest.json"
HERETIC_Q3_CONVERTED_ROOT = "F:\\AIWF_Studio\\models\\ltx\\text_encoder\\gemma-3-12b-heretic-q3km-converted"
HERETIC_Q3_CONVERTED_OUTPUT = "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx-smoketest.mp4"
HERETIC_Q3_CONVERTED_RECEIPT = "F:\\AIWF_Studio\\_local\\logs\\ltx_heretic_q3_converted_smoke_latest.json"
HERETIC_Q3_CONVERTED_DIRECT_OUTPUT = "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx23-20260628T153424Z-3d2a31.mp4"
HERETIC_Q3_CONVERTED_FULL_SANE_OUTPUT = "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx23-20260628T155411Z-0f0648.mp4"
HERETIC_Q3_CONVERTED_FULL_SANE_RECEIPT = "F:\\AIWF_Studio\\_local\\logs\\ltx_heretic_q3_converted_full_sane_latest.json"
SANA_SPRINT_OUTPUT = "F:\\AIWF_Studio\\outputs\\sana-images\\sana-sprint-smoke.png"
SANA_VIDEO_OUTPUT = "F:\\AIWF_Studio\\outputs\\sana-videos\\sana-video-20260628T204805Z-838c3e.mp4"

FIELDNAMES = [
    "row_id",
    "scope",
    "family",
    "base_type",
    "asset_type",
    "model_or_route",
    "path",
    "storage",
    "quantization",
    "route",
    "current_status",
    "current_result",
    "blocking_path",
    "blocking_reason",
    "suggested_action",
    "smoke_command",
    "smoke_status",
    "smoke_profile",
    "smoke_output_path",
    "smoke_width",
    "smoke_height",
    "smoke_frames",
    "smoke_fps",
    "smoke_steps",
    "smoke_seconds",
    "warm_second_seconds",
    "full_sane_target",
    "full_sane_prompt_profile",
    "full_sane_video_seconds",
    "full_sane_image_passes",
    "full_sane_expected_width",
    "full_sane_expected_height",
    "full_sane_expected_frames",
    "full_sane_expected_fps",
    "full_sane_expected_steps",
    "full_sane_status",
    "full_sane_output_path",
    "full_sane_seconds_cold",
    "full_sane_second_output_path",
    "full_sane_seconds_warm",
    "supports_txt2img",
    "supports_img2img",
    "supports_inpaint",
    "supports_i2v",
    "supports_t2v",
    "supports_video_audio",
    "supports_lora",
    "supports_gguf",
    "supports_control",
    "supports_llm_vl",
    "source",
    "metadata_json",
    "notes",
]

PASS_STATUSES = {"working"}
BLOCKED_STATUSES = {"blocked-cleanly", "broken-runtime", "unsupported-no-route"}

KNOWN_EVIDENCE: dict[str, dict[str, str]] = {
    "route:ltx-0.9.5-diffusers-local-t5xxl": {
        "smoke_status": "pass",
        "smoke_profile": "bounded_1step_9frames",
        "smoke_output_path": "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx2b-diffusers-smoke.mp4",
        "smoke_width": "128",
        "smoke_height": "128",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "1",
        "smoke_seconds": "60.37",
        "warm_second_seconds": "5.78",
        "notes": "Fresh LTX 2B Diffusers smoke passed; prior same-process second service run reused cache.",
    },
    "registry:ltx-2b-diffusers": {
        "smoke_status": "pass",
        "smoke_profile": "bounded_1step_9frames",
        "smoke_output_path": "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx2b-diffusers-smoke.mp4",
        "smoke_width": "128",
        "smoke_height": "128",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "1",
        "smoke_seconds": "60.37",
        "warm_second_seconds": "5.78",
        "notes": "Fresh LTX 2B Diffusers smoke passed; prior same-process second service run reused cache.",
    },
    "preflight:ltx-2b": {
        "smoke_status": "pass",
        "smoke_profile": "bounded_1step_9frames",
        "smoke_output_path": "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx2b-diffusers-smoke.mp4",
        "smoke_width": "128",
        "smoke_height": "128",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "1",
        "smoke_seconds": "60.37",
        "warm_second_seconds": "5.78",
        "notes": "Fresh LTX 2B Diffusers smoke passed; prior same-process second service run reused cache.",
    },
    "preflight:ltx-2-3": {
        "smoke_status": "pass",
        "smoke_profile": "bounded_1step_9frames_fp8_no_offload",
        "smoke_output_path": "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx-smoketest.mp4",
        "smoke_width": "128",
        "smoke_height": "128",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "1",
        "smoke_seconds": "174.35",
        "full_sane_status": "pass",
        "full_sane_output_path": HERETIC_Q3_CONVERTED_FULL_SANE_OUTPUT,
        "full_sane_seconds_cold": "286.38",
        "full_sane_expected_width": "128",
        "full_sane_expected_height": "128",
        "full_sane_expected_frames": "41",
        "full_sane_expected_fps": "8",
        "full_sane_expected_steps": "4",
        "notes": (
            "LTX 2.3 FP8 script smoke passed with converted Heretic Q3 Gemma root. "
            "Heretic full-sane pass produced 41 frames (5.125s) with native AAC audio; no warmed second run yet."
        ),
    },
    "registry:ltx-2.3": {
        "smoke_status": "pass",
        "smoke_profile": "bounded_1step_9frames_fp8_no_offload",
        "smoke_output_path": "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx-smoketest.mp4",
        "smoke_width": "128",
        "smoke_height": "128",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "1",
        "smoke_seconds": "174.35",
        "full_sane_status": "pass",
        "full_sane_output_path": HERETIC_Q3_CONVERTED_FULL_SANE_OUTPUT,
        "full_sane_seconds_cold": "286.38",
        "full_sane_expected_width": "128",
        "full_sane_expected_height": "128",
        "full_sane_expected_frames": "41",
        "full_sane_expected_fps": "8",
        "full_sane_expected_steps": "4",
        "notes": (
            "LTX 2.3 FP8 script smoke passed with converted Heretic Q3 Gemma root. "
            "Heretic full-sane pass produced 41 frames (5.125s) with native AAC audio; no warmed second run yet."
        ),
    },
    "route:ltx-one-stage-hf-gemma": {
        "smoke_status": "pass",
        "smoke_profile": "bounded_1step_9frames_fp8_no_offload",
        "smoke_output_path": "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx-smoketest.mp4",
        "smoke_width": "128",
        "smoke_height": "128",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "1",
        "smoke_seconds": "174.35",
        "full_sane_status": "pass",
        "full_sane_output_path": HERETIC_Q3_CONVERTED_FULL_SANE_OUTPUT,
        "full_sane_seconds_cold": "286.38",
        "full_sane_expected_width": "128",
        "full_sane_expected_height": "128",
        "full_sane_expected_frames": "41",
        "full_sane_expected_fps": "8",
        "full_sane_expected_steps": "4",
        "notes": (
            "LTX 2.3 FP8 script smoke passed with converted Heretic Q3 Gemma root. "
            "Heretic full-sane pass produced 41 frames (5.125s) with native AAC audio; no warmed second run yet."
        ),
    },
    "route:ltx-one-stage-heretic-gguf": {
        "smoke_command": "venv\\Scripts\\python.exe scripts\\probe_ltx_runtime.py --gguf --json --allow-blocked",
        "smoke_status": "blocked-cleanly",
        "smoke_profile": "gguf_contract_probe_no_dequant",
        "smoke_output_path": "F:\\AIWF_Studio\\_local\\logs\\ltx_heretic_q3_gguf_probe_latest.json",
        "smoke_steps": "0",
        "smoke_seconds": "8.7",
        "notes": (
            "Heretic Q3 GGUF metadata probe verified Gemma3 hidden_size=3840, layers=48, "
            "expected_hidden_states=49, tensors=626; generation remains blocked because no installed "
            "quantized GGUF backend exposes the hidden-state tuple plus attention mask LTX requires. "
            f"Gemma GGUF inventory receipt: {GGUF_INVENTORY_RECEIPT}."
        ),
    },
    "registry:sana": {
        "current_status": "working",
        "current_result": "smoke passed; saved artifact verified",
        "smoke_status": "pass",
        "smoke_profile": "sana_sprint_06b_2step_512",
        "smoke_command": "venv\\Scripts\\python.exe scripts\\smoke_image_routes.py --only Sana_Sprint_0.6B_1024px_diffusers --steps 2 --width 512 --height 512 --timeout-seconds 900 --json",
        "smoke_output_path": SANA_SPRINT_OUTPUT,
        "smoke_width": "512",
        "smoke_height": "512",
        "smoke_steps": "2",
        "smoke_seconds": "16.30",
        "notes": "Sana Sprint 0.6B image route passed a bounded 2-step smoke and a saved artifact pass.",
    },
    "preflight:sana-video": {
        "current_status": "working",
        "current_result": "smoke passed; silent video artifact verified",
        "smoke_status": "pass",
        "smoke_profile": "sana_video_480p_2step_9frames_no_audio",
        "smoke_command": "inline SanaVideoService smoke, frames=9 steps=2 fps=8 generate_audio=false",
        "smoke_output_path": SANA_VIDEO_OUTPUT,
        "smoke_width": "832",
        "smoke_height": "480",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "2",
        "smoke_seconds": "21.29",
        "notes": "Sana Video 480p route produced a silent 9-frame H.264 MP4.",
    },
    "registry:sana-video": {
        "current_status": "working",
        "current_result": "smoke passed; silent video artifact verified",
        "smoke_status": "pass",
        "smoke_profile": "sana_video_480p_2step_9frames_no_audio",
        "smoke_command": "inline SanaVideoService smoke, frames=9 steps=2 fps=8 generate_audio=false",
        "smoke_output_path": SANA_VIDEO_OUTPUT,
        "smoke_width": "832",
        "smoke_height": "480",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "2",
        "smoke_seconds": "21.29",
        "notes": "Sana Video 480p route produced a silent 9-frame H.264 MP4.",
    },
}

PATH_EVIDENCE: dict[str, dict[str, str]] = {
    "f:\\aiwf_studio\\models\\ltx\\checkpoints\\ltx-2.3-22b-dev-fp8.safetensors": {
        "smoke_status": "pass",
        "smoke_profile": "bounded_1step_9frames_fp8_no_offload",
        "smoke_output_path": "F:\\AIWF_Studio\\outputs\\ltx-videos\\ltx-smoketest.mp4",
        "smoke_width": "128",
        "smoke_height": "128",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "1",
        "smoke_seconds": "174.35",
        "full_sane_status": "pass",
        "full_sane_output_path": HERETIC_Q3_CONVERTED_FULL_SANE_OUTPUT,
        "full_sane_seconds_cold": "286.38",
        "full_sane_expected_width": "128",
        "full_sane_expected_height": "128",
        "full_sane_expected_frames": "41",
        "full_sane_expected_fps": "8",
        "full_sane_expected_steps": "4",
        "notes": (
            "LTX 2.3 FP8 script smoke passed with converted Heretic Q3 Gemma root. "
            "Heretic full-sane pass produced 41 frames (5.125s) with native AAC audio; no warmed second run yet."
        ),
    },
    "f:\\aiwf_studio\\models\\ltx\\text_encoder\\gemma-3-12b-heretic-q3km-converted\\model.safetensors": {
        "smoke_status": "pass",
        "smoke_profile": "bounded_1step_9frames_fp8_no_offload_heretic_q3_converted",
        "smoke_command": "cmd /c \"set AIWF_NO_PAUSE=1&& scripts\\run_ltx_smoketest.bat\"",
        "smoke_output_path": HERETIC_Q3_CONVERTED_OUTPUT,
        "smoke_width": "128",
        "smoke_height": "128",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "1",
        "smoke_seconds": "174.35",
        "full_sane_status": "pass",
        "full_sane_output_path": HERETIC_Q3_CONVERTED_FULL_SANE_OUTPUT,
        "full_sane_seconds_cold": "286.38",
        "full_sane_expected_width": "128",
        "full_sane_expected_height": "128",
        "full_sane_expected_frames": "41",
        "full_sane_expected_fps": "8",
        "full_sane_expected_steps": "4",
        "notes": (
            "Converted smallest Heretic Q3 GGUF into HF-shaped BF16 Gemma safetensors, with official "
            "Gemma vision/projector sidecar for loader compatibility. Script output verified with 9 video frames "
            f"and AAC audio; full-sane receipt: {HERETIC_Q3_CONVERTED_FULL_SANE_RECEIPT}."
        ),
    },
    "f:\\aiwf_studio\\models\\llm\\gguf\\gemma-3-12b-it-heretic-q3_k_m.gguf": {
        "smoke_status": "metadata-only",
        "smoke_profile": "gemma_gguf_inventory_no_dequant",
        "smoke_command": "venv\\Scripts\\python.exe scripts\\probe_ltx_runtime.py --gguf-inventory --json",
        "smoke_output_path": GGUF_INVENTORY_RECEIPT,
        "smoke_steps": "0",
        "notes": (
            "Smallest local Heretic GGUF candidate; metadata inventory is safe/no-dequant. "
            "LTX generation still needs a hidden-state backend."
        ),
    },
    "f:\\aiwf_studio\\models\\llm\\gguf\\gemma-3-12b-it-heretic-q4_k_m.gguf": {
        "smoke_status": "metadata-only",
        "smoke_profile": "gemma_gguf_inventory_no_dequant",
        "smoke_command": "venv\\Scripts\\python.exe scripts\\probe_ltx_runtime.py --gguf-inventory --json",
        "smoke_output_path": GGUF_INVENTORY_RECEIPT,
        "smoke_steps": "0",
        "notes": "Alternative Heretic GGUF candidate; Q3_K_M remains the preferred smallest local Heretic test asset.",
    },
    "f:\\aiwf_studio\\models\\llm\\gguf\\gemma-3-12b-it-heretic-q4_k_s.gguf": {
        "smoke_status": "metadata-only",
        "smoke_profile": "gemma_gguf_inventory_no_dequant",
        "smoke_command": "venv\\Scripts\\python.exe scripts\\probe_ltx_runtime.py --gguf-inventory --json",
        "smoke_output_path": GGUF_INVENTORY_RECEIPT,
        "smoke_steps": "0",
        "notes": "Alternative Heretic GGUF candidate; Q3_K_M remains the preferred smallest local Heretic test asset.",
    },
    "f:\\aiwf_studio\\models\\sana\\diffusers\\sana_sprint_0.6b_1024px_diffusers": {
        "current_status": "working",
        "current_result": "smoke passed; saved artifact verified",
        "smoke_status": "pass",
        "smoke_profile": "sana_sprint_06b_2step_512",
        "smoke_command": "venv\\Scripts\\python.exe scripts\\smoke_image_routes.py --only Sana_Sprint_0.6B_1024px_diffusers --steps 2 --width 512 --height 512 --timeout-seconds 900 --json",
        "smoke_output_path": SANA_SPRINT_OUTPUT,
        "smoke_width": "512",
        "smoke_height": "512",
        "smoke_steps": "2",
        "smoke_seconds": "16.30",
        "notes": "Sana Sprint 0.6B image route passed a bounded 2-step smoke and a saved artifact pass.",
    },
    "f:\\aiwf_studio\\models\\sana-video\\diffusers\\sana-video_2b_480p_diffusers": {
        "current_status": "working",
        "current_result": "smoke passed; silent video artifact verified",
        "smoke_status": "pass",
        "smoke_profile": "sana_video_480p_2step_9frames_no_audio",
        "smoke_command": "inline SanaVideoService smoke, frames=9 steps=2 fps=8 generate_audio=false",
        "smoke_output_path": SANA_VIDEO_OUTPUT,
        "smoke_width": "832",
        "smoke_height": "480",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "2",
        "smoke_seconds": "21.29",
        "notes": "Sana Video 480p route produced a silent 9-frame H.264 MP4.",
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an Excel-compatible pipeline QA matrix CSV.")
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = json.loads(args.readiness.read_text(encoding="utf-8"))
    records = list(data.get("records") or [])
    rows = [_record_to_row(record) for record in records]
    rows.append(_manual_ltx23_fp8_row())
    rows.append(_manual_ltx23_heretic_q3_converted_row())
    rows.append(_manual_sana_sprint_row())
    rows.append(_manual_sana_video_row())
    _apply_latest_ltx_smoke_evidence(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))


def _apply_latest_ltx_smoke_evidence(rows: list[dict[str, str]]) -> None:
    evidence = _latest_ltx_smoke_evidence()
    if not evidence:
        return
    for row in rows:
        if row.get("smoke_status") == "pass" and row.get("smoke_output_path") == HERETIC_Q3_CONVERTED_OUTPUT:
            row.update(evidence)


def _latest_ltx_smoke_evidence() -> dict[str, str]:
    if not LTX_SMOKETEST_LOG.is_file():
        return {}
    start: datetime | None = None
    end: datetime | None = None
    for raw in LTX_SMOKETEST_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("job_id") != "ltx_smoketest":
            continue
        ts = _parse_iso_datetime(str(event.get("ts") or ""))
        if ts is None:
            continue
        if start is None and event.get("kind") == "status":
            start = ts
        if event.get("kind") == "complete":
            end = ts
    if start is None or end is None or end <= start:
        return {}
    return {"smoke_seconds": f"{(end - start).total_seconds():.2f}"}


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _record_to_row(record: dict[str, Any]) -> dict[str, str]:
    metadata = {str(k): str(v) for k, v in (record.get("metadata") or {}).items()}
    row_id = str(record.get("id") or "")
    path = str(record.get("path") or "")
    route = str(record.get("route") or "")
    status = str(record.get("status") or "")
    family = str(record.get("family") or "")
    asset_type = str(record.get("asset_type") or "")
    reason = str(record.get("reason") or "")
    row = {
        "row_id": row_id,
        "scope": _scope(row_id, asset_type),
        "family": family,
        "base_type": _base_type(family, route, asset_type, metadata),
        "asset_type": asset_type,
        "model_or_route": _model_or_route(row_id, path, route),
        "path": path,
        "storage": str(record.get("storage") or ""),
        "quantization": str(record.get("quantization") or ""),
        "route": route,
        "current_status": status,
        "current_result": _current_result(status),
        "blocking_path": path if status in BLOCKED_STATUSES else "",
        "blocking_reason": reason if status in BLOCKED_STATUSES else "",
        "suggested_action": str(record.get("suggested_action") or ""),
        "smoke_command": str(record.get("smoke_command") or ""),
        "smoke_status": "pass" if status == "working" else "",
        "smoke_output_path": str(record.get("receipt_path") or ""),
        "full_sane_target": _full_sane_target(family, route),
        "full_sane_prompt_profile": _prompt_profile(family, route),
        "full_sane_video_seconds": "5" if _is_video(family, route) else "",
        "full_sane_image_passes": "2" if _is_image(family, route) else "",
        "source": metadata.get("source", ""),
        "metadata_json": json.dumps(metadata, sort_keys=True, ensure_ascii=True),
    }
    row.update(_feature_flags(family, route, path, asset_type))
    row.update(PATH_EVIDENCE.get(path.lower(), {}))
    row.update(KNOWN_EVIDENCE.get(row_id, {}))
    return _complete(row)


def _manual_ltx23_fp8_row() -> dict[str, str]:
    row = {
        "row_id": "manual:ltx-2.3-fp8-no-offload-smoke",
        "scope": "manual_smoke",
        "family": "ltx",
        "base_type": "video:ltx-2.3:fp8",
        "asset_type": "pipeline",
        "model_or_route": "LTX 2.3 FP8 one-stage no-offload",
        "path": "F:\\AIWF_Studio\\models\\ltx\\checkpoints\\ltx-2.3-22b-dev-fp8.safetensors",
        "storage": "safetensors",
        "quantization": "fp8",
        "route": "ltx-one-stage-hf-gemma",
        "current_status": "working",
        "current_result": "smoke passed; full sane 4-step run passed",
        "smoke_command": "cmd /c \"set AIWF_NO_PAUSE=1&& scripts\\run_ltx_smoketest.bat\"",
        "full_sane_target": "video_5_seconds",
        "full_sane_prompt_profile": "default_video_prompt",
        "full_sane_video_seconds": "5",
        "metadata_json": "{}",
    }
    row.update(_feature_flags("ltx", "ltx-one-stage-hf-gemma", row["path"], "pipeline"))
    row.update(PATH_EVIDENCE[row["path"].lower()])
    return _complete(row)


def _manual_ltx23_heretic_q3_converted_row() -> dict[str, str]:
    row = {
        "row_id": "manual:ltx-2.3-heretic-q3-converted-smoke",
        "scope": "manual_smoke",
        "family": "ltx",
        "base_type": "video:ltx-2.3:fp8:gemma-heretic-q3-converted",
        "asset_type": "pipeline",
        "model_or_route": "LTX 2.3 FP8 one-stage + converted Heretic Q3 Gemma",
        "path": HERETIC_Q3_CONVERTED_ROOT,
        "storage": "safetensors",
        "quantization": "source Q3_K_M GGUF converted to BF16 safetensors",
        "route": "ltx-one-stage-hf-gemma",
        "current_status": "working",
        "current_result": "smoke passed; full sane Heretic run passed",
        "smoke_command": "cmd /c \"set AIWF_NO_PAUSE=1&& scripts\\run_ltx_smoketest.bat\"",
        "full_sane_target": "video_5_seconds",
        "full_sane_prompt_profile": "default_video_prompt",
        "full_sane_video_seconds": "5",
        "full_sane_expected_width": "128",
        "full_sane_expected_height": "128",
        "full_sane_expected_frames": "41",
        "full_sane_expected_fps": "8",
        "full_sane_expected_steps": "4",
        "metadata_json": json.dumps(
            {
                "gemma_root": HERETIC_Q3_CONVERTED_ROOT,
                "source_gguf": "F:\\AIWF_Studio\\models\\LLM\\GGUF\\gemma-3-12b-it-heretic-Q3_K_M.gguf",
                "direct_output": HERETIC_Q3_CONVERTED_DIRECT_OUTPUT,
                "smoke_receipt": HERETIC_Q3_CONVERTED_RECEIPT,
                "full_sane_output": HERETIC_Q3_CONVERTED_FULL_SANE_OUTPUT,
                "full_sane_receipt": HERETIC_Q3_CONVERTED_FULL_SANE_RECEIPT,
                "verified_smoke_video_stream": "h264 128x128 9 frames 8 fps duration 1.125s",
                "verified_smoke_audio_stream": "aac duration 1.09s",
                "verified_full_sane_video_stream": "h264 128x128 41 frames 8 fps duration 5.125s",
                "verified_full_sane_audio_stream": "aac duration 5.09s",
            },
            sort_keys=True,
        ),
        "notes": (
            "This is the working practical Heretic route. Native quantized GGUF generation remains blocked, "
            "but the smallest Heretic Q3 GGUF can be converted to an HF-shaped Gemma root and used by LTX."
        ),
    }
    row.update(_feature_flags("ltx", "ltx-one-stage-hf-gemma", row["path"], "pipeline"))
    row.update(
        {
            "smoke_status": "pass",
            "smoke_profile": "bounded_1step_9frames_fp8_no_offload_heretic_q3_converted",
            "smoke_output_path": HERETIC_Q3_CONVERTED_OUTPUT,
            "smoke_width": "128",
            "smoke_height": "128",
            "smoke_frames": "9",
            "smoke_fps": "8",
            "smoke_steps": "1",
            "smoke_seconds": "174.35",
            "full_sane_status": "pass",
            "full_sane_output_path": HERETIC_Q3_CONVERTED_FULL_SANE_OUTPUT,
            "full_sane_seconds_cold": "286.38",
        }
    )
    return _complete(row)


def _manual_sana_sprint_row() -> dict[str, str]:
    row = {
        "row_id": "manual:sana-sprint-06b-image-smoke",
        "scope": "manual_smoke",
        "family": "image",
        "base_type": "image:sana",
        "asset_type": "pipeline",
        "model_or_route": "Sana Sprint 0.6B 1024px Diffusers",
        "path": "F:\\AIWF_Studio\\models\\sana\\Diffusers\\Sana_Sprint_0.6B_1024px_diffusers",
        "storage": "diffusers",
        "route": "sana",
        "current_status": "working",
        "current_result": "smoke passed; saved artifact verified",
        "smoke_command": "venv\\Scripts\\python.exe scripts\\smoke_image_routes.py --only Sana_Sprint_0.6B_1024px_diffusers --steps 2 --width 512 --height 512 --timeout-seconds 900 --json",
        "smoke_status": "pass",
        "smoke_profile": "sana_sprint_06b_2step_512",
        "smoke_output_path": SANA_SPRINT_OUTPUT,
        "smoke_width": "512",
        "smoke_height": "512",
        "smoke_steps": "2",
        "smoke_seconds": "16.30",
        "full_sane_target": "image_default_two_pass_second_timed",
        "full_sane_prompt_profile": "default_image_prompt",
        "full_sane_image_passes": "2",
        "metadata_json": json.dumps(
            {
                "saved_artifact_seconds": "6.13",
                "verified_image_size": "512x512",
                "verified_image_bytes": "393177",
            },
            sort_keys=True,
        ),
        "notes": "Sana Sprint 0.6B image route passed and produced a nonblank saved PNG artifact.",
    }
    row.update(_feature_flags("image", "sana", row["path"], "pipeline"))
    return _complete(row)


def _manual_sana_video_row() -> dict[str, str]:
    row = {
        "row_id": "manual:sana-video-2b-480p-smoke",
        "scope": "manual_smoke",
        "family": "video",
        "base_type": "video:sana",
        "asset_type": "pipeline",
        "model_or_route": "SANA Video 2B 480p Diffusers",
        "path": "F:\\AIWF_Studio\\models\\sana-video\\Diffusers\\SANA-Video_2B_480p_diffusers",
        "storage": "diffusers",
        "route": "sana-video",
        "current_status": "working",
        "current_result": "smoke passed; silent video artifact verified",
        "smoke_command": "inline SanaVideoService smoke, frames=9 steps=2 fps=8 generate_audio=false",
        "smoke_status": "pass",
        "smoke_profile": "sana_video_480p_2step_9frames_no_audio",
        "smoke_output_path": SANA_VIDEO_OUTPUT,
        "smoke_width": "832",
        "smoke_height": "480",
        "smoke_frames": "9",
        "smoke_fps": "8",
        "smoke_steps": "2",
        "smoke_seconds": "21.29",
        "full_sane_target": "video_5_seconds",
        "full_sane_prompt_profile": "default_video_prompt",
        "full_sane_video_seconds": "5",
        "metadata_json": json.dumps(
            {
                "verified_video_stream": "h264 832x480 9 frames 8 fps duration 1.125s",
                "verified_audio": "none",
                "verified_video_bytes": "45174",
            },
            sort_keys=True,
        ),
        "notes": "Sana Video 480p route produced a nonblank silent MP4. Use MMAudio post-process for audio.",
    }
    row.update(_feature_flags("video", "sana-video", row["path"], "pipeline"))
    return _complete(row)


def _complete(row: dict[str, str]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in FIELDNAMES}


def _scope(row_id: str, asset_type: str) -> str:
    if row_id.startswith("registry:"):
        return "registry_route"
    if row_id.startswith("preflight:"):
        return "preflight_route"
    if row_id.startswith("route:"):
        return "runtime_route"
    if row_id.startswith("asset:"):
        return "model_asset"
    return asset_type or "record"


def _base_type(family: str, route: str, asset_type: str, metadata: dict[str, str]) -> str:
    arch = metadata.get("architecture") or metadata.get("header_arch") or metadata.get("modelspec.architecture") or ""
    if "ltx" in route or family == "ltx":
        return "video:ltx"
    if "wan" in route or family == "wan":
        return "video:wan"
    if "sana" in route or family == "sana-video":
        return "video:sana" if "video" in route or family == "sana-video" else "image:sana"
    if "qwen" in route:
        return "image:qwen"
    if "flux" in route or "flux" in arch.lower():
        return "image:flux"
    if family == "image":
        return f"image:{arch or asset_type or 'unknown'}"
    if family in {"llm", "vl", "llm-vl"}:
        return "llm-vl"
    return f"{family}:{arch or asset_type or 'unknown'}"


def _model_or_route(row_id: str, path: str, route: str) -> str:
    if row_id.startswith(("registry:", "preflight:", "route:")):
        return route or row_id
    if path:
        return Path(path).name
    return row_id


def _current_result(status: str) -> str:
    if status in PASS_STATUSES:
        return "smoke passed"
    if status == "metadata-only":
        return "discovered; smoke pending"
    if status == "blocked-cleanly":
        return "blocked before runtime"
    if status == "broken-runtime":
        return "known runtime failure"
    if status == "unsupported-no-route":
        return "asset present; route missing or unsupported"
    return status


def _full_sane_target(family: str, route: str) -> str:
    if _is_video(family, route):
        return "video_5_seconds"
    if _is_image(family, route):
        return "image_default_two_pass_second_timed"
    if family in {"llm", "vl", "llm-vl"}:
        return "llm_vl_default_eval"
    return ""


def _prompt_profile(family: str, route: str) -> str:
    if _is_video(family, route):
        return "default_video_prompt"
    if _is_image(family, route):
        return "default_image_prompt"
    if family in {"llm", "vl", "llm-vl"}:
        return "default_llm_vl_prompt"
    return ""


def _is_video(family: str, route: str) -> bool:
    text = f"{family} {route}".lower()
    return any(token in text for token in ("wan", "ltx", "sana-video", "video"))


def _is_image(family: str, route: str) -> bool:
    text = f"{family} {route}".lower()
    return family == "image" or any(token in text for token in ("qwen", "flux", "sana", "diffusers", "onnx"))


def _feature_flags(family: str, route: str, path: str, asset_type: str) -> dict[str, str]:
    text = f"{family} {route} {path} {asset_type}".lower()
    is_video = _is_video(family, route)
    is_image = _is_image(family, route)
    return {
        "supports_txt2img": _yes(is_image and not any(token in text for token in ("inpaint", "control", "upscale"))),
        "supports_img2img": _yes(is_image and "inpaint" not in text),
        "supports_inpaint": _yes("inpaint" in text),
        "supports_i2v": _yes("i2v" in text or "image-to-video" in text or route.startswith("wan") or "ltx" in text),
        "supports_t2v": _yes(is_video and ("ltx" in text or "sana" in text or "t2v" in text or "ti2v" in text)),
        "supports_video_audio": _yes("ltx-2.3" in text or "ltx-one-stage" in text),
        "supports_lora": _yes("lora" in text),
        "supports_gguf": _yes("gguf" in text),
        "supports_control": _yes("control" in text or "fun" in text),
        "supports_llm_vl": _yes(family in {"llm", "vl", "llm-vl"} or "qwen-vl" in text or "florence" in text),
    }


def _yes(value: bool) -> str:
    return "yes" if value else ""


if __name__ == "__main__":
    main()
