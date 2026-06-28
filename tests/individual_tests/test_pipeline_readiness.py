from __future__ import annotations

import json
import struct
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.ltx import LTX_FULL_CHECKPOINT, LTX_GEMMA_REPO, LTX_HERETIC_Q3_GGUF
from aiwf.services.pipeline_readiness import (
    PipelineReadinessRecord,
    classify_pipeline_asset,
    collect_pipeline_readiness,
    readiness_summary,
)
from aiwf.services.worker_tenant import python_exe_for_venv


def _write_safetensors_header(path: Path, header: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def test_wan_fun_control_header_is_unsupported_no_route(tmp_path: Path):
    path = tmp_path / "models" / "wan" / "Safetensor" / "wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors"
    _write_safetensors_header(
        path,
        {
            "__metadata__": {
                "modelspec.architecture": "wan_2.2_14b_i2v",
                "modelspec.title": "Wan Fun-Control High",
            },
            "diffusion_model.patch_embedding.weight": {
                "dtype": "F8_E4M3",
                "shape": [5120, 52, 2, 2],
                "data_offsets": [0, 0],
            },
            "diffusion_model.blocks.0.weight": {
                "dtype": "F8_E4M3",
                "shape": [1],
                "data_offsets": [0, 0],
            },
        },
    )

    record = classify_pipeline_asset("wan", "wan", path)

    assert record.status == "unsupported-no-route"
    assert record.route == "wan-diffusers"
    assert record.required_vae == "2.1"
    assert "52-channel" in record.reason


def test_ltx_gguf_download_is_unsupported_no_route(tmp_path: Path):
    path = tmp_path / "Downloads" / "ltx23DISTILLEDGGUF_q2k.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF")

    record = classify_pipeline_asset("ltx", "ltx", path, source="downloads")

    assert record.status == "unsupported-no-route"
    assert record.storage == "gguf"
    assert record.route == "ltx-2.3"
    assert "GGUF" in record.reason


def test_gemma_heretic_download_stays_out_of_ltx_runtime(tmp_path: Path):
    path = tmp_path / "Downloads" / "gemma-3-12b-it-heretic-Q3_K_M.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF")

    record = classify_pipeline_asset("ltx", "ltx", path, source="downloads")

    assert record.status == "unsupported-no-route"
    assert record.storage == "gguf"
    assert "not wired into the current LTX worker route" in record.reason


def test_gemma_gguf_under_ltx_models_is_ltx_asset(tmp_path: Path):
    path = tmp_path / "models" / "ltx" / "text_encoder" / "gemma-3-12b-q4_0-gguf" / "gemma-3-12b-it-Q4_0.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF")

    records = collect_pipeline_readiness(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        include_downloads=False,
    )
    record = next(item for item in records if item.path == str(path))

    assert record.family == "ltx"
    assert record.status == "unsupported-no-route"
    assert record.route == "ltx-2.3"


def test_ltx_route_records_include_heretic_gguf_blocker(tmp_path: Path):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")
    models = tmp_path / "models"
    checkpoint = models / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = models / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    gemma.mkdir(parents=True)
    gguf = models / "LLM" / "GGUF" / LTX_HERETIC_Q3_GGUF
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")

    records = collect_pipeline_readiness(
        RuntimeFlags(data_dir=tmp_path, models_dir=models, output_dir=tmp_path / "outputs"),
        include_downloads=False,
    )
    record = next(item for item in records if item.id == "route:ltx-one-stage-heretic-gguf")

    assert record.status == "blocked-cleanly"
    assert record.storage == "gguf"
    assert "hidden-state" in record.reason


def test_ltx2b_diffusers_route_is_wired_when_assets_exist(tmp_path: Path):
    models = tmp_path / "models"
    checkpoint = models / "ltx" / "checkpoints" / "ltx-video-2b-v0.9.5.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    t5 = models / "flux" / "Textencoder" / "t5xxl_fp16.safetensors"
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake")

    records = collect_pipeline_readiness(
        RuntimeFlags(data_dir=tmp_path, models_dir=models, output_dir=tmp_path / "outputs"),
        include_downloads=False,
    )
    record = next(item for item in records if item.id == "route:ltx-0.9.5-diffusers-local-t5xxl")

    assert record.status == "metadata-only"
    assert record.route == "ltx-0.9.5-diffusers-local-t5xxl"
    assert record.required_text_encoder.endswith("t5xxl_fp16.safetensors")


def test_known_bad_image_checkpoint_is_broken_runtime(tmp_path: Path):
    path = tmp_path / "models" / "Stable-diffusion" / "4xBHI_dat2_multiblurjpg.safetensors"
    _write_safetensors_header(path, {"__metadata__": {}})

    record = classify_pipeline_asset("checkpoint", "sd15", path)

    assert record.status == "broken-runtime"
    assert record.route == "diffusers"
    assert "missing the expected CLIP text model" in record.reason


def test_collect_pipeline_readiness_includes_download_assets(tmp_path: Path):
    downloads = tmp_path / "Downloads"
    ltx = downloads / "ltx23_ltx2322bDistilled.safetensors"
    _write_safetensors_header(ltx, {"__metadata__": {"modelspec.architecture": "ltx"}})
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")

    records = collect_pipeline_readiness(flags, include_downloads=True, download_roots=(downloads,))

    assert any(record.path == str(ltx) and record.family == "ltx" for record in records)
    assert readiness_summary(records)["metadata-only"] >= 1


def test_readiness_record_serializes_without_secret_fields():
    record = PipelineReadinessRecord(
        id="asset:test",
        family="image",
        asset_type="checkpoint",
        path="models/example.safetensors",
        status="metadata-only",
        route="diffusers",
        reason="metadata only",
        metadata={"tokenizer": "local tokenizer"},
    )

    payload = record.to_dict()

    assert payload["id"] == "asset:test"
    assert "hf_" not in json.dumps(payload).lower()
