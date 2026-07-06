from __future__ import annotations

import json
import struct
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.ltx import (
    LTX_FULL_CHECKPOINT,
    LTX_FULL_CHECKPOINT_FP8,
    LTX_GEMMA_REPO,
    LTX_HERETIC_Q3_CONVERTED_FOLDER,
    LTX_HERETIC_Q3_GGUF,
)
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


def test_ltx_fp8_checkpoint_uses_one_stage_route(tmp_path: Path):
    path = tmp_path / "models" / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT_FP8
    _write_safetensors_header(path, {"__metadata__": {"modelspec.architecture": "ltx"}})

    record = classify_pipeline_asset("ltx", "ltx", path)

    assert record.status == "metadata-only"
    assert record.route == "ltx-one-stage-hf-gemma"
    assert record.storage == "safetensors"
    assert record.quantization == "FP8"
    assert "offload=none" in record.reason


def test_ltx_fp8_checkpoint_is_working_when_receipt_exists(tmp_path: Path):
    models = tmp_path / "models"
    outputs = tmp_path / "outputs"
    path = models / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT_FP8
    _write_safetensors_header(path, {"__metadata__": {"modelspec.architecture": "ltx"}})
    receipt = outputs / "ltx-videos" / "ltx23-smoke.mp4"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"fake mp4")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models, output_dir=outputs)

    record = classify_pipeline_asset("ltx", "ltx", path, flags=flags)

    assert record.status == "working"
    assert record.route == "ltx-one-stage-hf-gemma"
    assert "runtime smoke receipt" in record.reason


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


def test_converted_heretic_gemma_folder_is_ltx_text_encoder_asset(tmp_path: Path):
    root = tmp_path / "models" / "ltx" / "text_encoder" / LTX_HERETIC_Q3_CONVERTED_FOLDER
    model = root / "model.safetensors"
    model.parent.mkdir(parents=True)
    for filename in (
        "model.safetensors",
        "vision_projector.safetensors",
        "tokenizer.model",
        "preprocessor_config.json",
    ):
        (root / filename).write_bytes(b"fake")

    record = classify_pipeline_asset("ltx", "ltx", model, source="models")

    assert record.status == "metadata-only"
    assert record.route == "ltx-one-stage-hf-gemma"
    assert record.required_text_encoder == str(root)
    assert "parent text_encoder folder" in record.suggested_action


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
    assert "--allow-blocked" in record.smoke_command


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


def test_known_bad_flux_runtime_assets_are_broken_runtime(tmp_path: Path):
    cases = [
        (
            tmp_path / "models" / "flux" / "UNet" / "fluxedUpFluxNSFW_110FP8.safetensors",
            "checkpoint keys do not match",
        ),
        (
            tmp_path / "models" / "flux" / "GGUF" / "fluxFusionV24StepsGGUFNF4_V2GGUFQ4KM.gguf",
            "GGUF/NF4 mismatch",
        ),
    ]
    for path, expected_reason in cases:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

        record = classify_pipeline_asset("runtime_asset", "flux", path)

        assert record.status == "broken-runtime"
        assert record.route == "flux"
        assert expected_reason in record.reason


def test_incomplete_qwen_diffusers_snapshot_is_blocked_cleanly(tmp_path: Path):
    root = tmp_path / "models" / "qwen-image" / "Diffusers" / "Qwen-Image"
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}), encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "transformer_blocks.0.attn.to_q.weight": "diffusion_pytorch_model-00001-of-00009.safetensors"
                },
            }
        ),
        encoding="utf-8",
    )

    record = classify_pipeline_asset("runtime_asset", "qwen_image", root)

    assert record.status == "blocked-cleanly"
    assert record.route == "qwen-image"
    assert "required local shard files are missing" in record.reason


def test_supported_image_gguf_runtime_assets_are_not_unsupported_no_route(tmp_path: Path):
    cases = [
        ("flux1-dev-Q4_K_M.gguf", "flux", "flux"),
        ("fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf", "flux2_klein", "flux2-klein"),
        ("fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf", "z_image", "z-image"),
    ]
    import os

    for filename, architecture, route in cases:
        path = tmp_path / "models" / "flux" / "GGUF" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")

        record = classify_pipeline_asset("runtime_asset", architecture, path)

        assert record.route == route
        if architecture == "z_image" and os.name == "nt":
            # Z-Image GGUF is blocked on Windows (fused GGUF kernels are
            # Linux-only; the fallback dequant path pages out 16 GB GPUs).
            assert record.status == "blocked-cleanly"
        else:
            assert record.status == "metadata-only"
            assert "runtime path" in record.reason


def test_krea2_and_anima_split_assets_are_blocked_until_native_loader(tmp_path: Path):
    krea = tmp_path / "models" / "krea2" / "UNet" / "krea2_turbo_fp8_scaled.safetensors"
    anima = tmp_path / "models" / "anima" / "UNet" / "anima-base-v1.0.safetensors"
    krea.parent.mkdir(parents=True)
    anima.parent.mkdir(parents=True)
    krea.write_bytes(b"fake")
    anima.write_bytes(b"fake")

    krea_record = classify_pipeline_asset("runtime_asset", "krea2", krea)
    anima_record = classify_pipeline_asset("runtime_asset", "anima", anima)

    assert krea_record.status == "blocked-cleanly"
    assert krea_record.route == "krea2"
    assert "split-file Krea2 loader" in krea_record.reason
    assert anima_record.status == "unsupported-no-route"
    assert anima_record.route == "anima"
    assert "native Anima loader" in anima_record.reason


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
