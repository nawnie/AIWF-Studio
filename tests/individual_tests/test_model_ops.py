from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import Checkpoint, LoraInfo
from aiwf.services.model_ops import (
    ModelOpsService,
    inspect_model_asset,
    sanitize_output_name,
    write_model_op_receipt,
)


def _flags(tmp_path: Path) -> RuntimeFlags:
    return RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")


def _ckpt(tmp_path: Path, name: str, architecture: str = "sdxl") -> Checkpoint:
    path = tmp_path / f"{name}.safetensors"
    path.write_bytes(b"not real weights")
    return Checkpoint(
        id=name,
        title=name,
        filename=path.name,
        path=str(path),
        architecture=architecture,
    )


def _lora(tmp_path: Path, name: str) -> LoraInfo:
    path = tmp_path / f"{name}.safetensors"
    path.write_bytes(b"not real lora")
    return LoraInfo(id=name, title=name, filename=path.name, path=str(path))


def test_sanitize_output_name_removes_path_characters():
    assert sanitize_output_name("../bad name!") == "bad_name"
    assert sanitize_output_name("  ") == "aiwf_model_output"


def test_inspect_diffusers_folder(tmp_path: Path):
    model_dir = tmp_path / "sdxl_model"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    asset = inspect_model_asset(model_dir)

    assert asset.storage == "diffusers"
    assert asset.family == "image"
    assert asset.architecture == "sdxl"


def test_inspect_onnx_folder(tmp_path: Path):
    model_dir = tmp_path / "onnx_model"
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "unet" / "model.onnx").write_bytes(b"onnx")

    asset = inspect_model_asset(model_dir)

    assert asset.storage == "onnx-folder"
    assert asset.family == "image"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf", "flux2_klein"),
        ("fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf", "z_image"),
    ],
)
def test_inspect_transformer_assets_keeps_flux2_and_z_image_separate(tmp_path: Path, filename: str, expected: str):
    model = tmp_path / filename
    model.write_bytes(b"GGUF")

    asset = inspect_model_asset(model)

    assert asset.storage == "gguf"
    assert asset.family == "video-or-transformer"
    assert asset.architecture == expected


def test_checkpoint_blend_blocks_architecture_mismatch(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    left = _ckpt(tmp_path, "left", "sd15")
    right = _ckpt(tmp_path, "right", "sdxl")

    result = svc.preflight_checkpoint_blend(left, right, ratio=0.5, output_name="out")

    assert not result.ok
    assert result.command is None
    assert "Architecture mismatch" in result.markdown()


def test_checkpoint_blend_builds_worker_command(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    left = _ckpt(tmp_path, "left", "sdxl")
    right = _ckpt(tmp_path, "right", "sdxl")

    result = svc.preflight_checkpoint_blend(left, right, ratio=0.25, output_name="out")

    assert result.ok
    assert result.command is not None
    assert result.command.name == "model-ops-checkpoint-blend"
    assert "aiwf.workers.model_ops" in result.command.args
    assert "--ratio" in result.command.args


def test_lora_fuse_builds_worker_command(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    base = _ckpt(tmp_path, "base", "sdxl")
    lora = _lora(tmp_path, "style")

    result = svc.preflight_lora_fuse(base, [lora], weights="0.7", output_name="fused")

    assert result.ok
    assert result.command is not None
    assert result.command.name == "model-ops-lora-fuse"
    assert "--lora" in result.command.args
    assert "--weight" in result.command.args


def test_lora_fuse_blocks_bad_weight_count(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    base = _ckpt(tmp_path, "base", "sdxl")
    loras = [_lora(tmp_path, "a"), _lora(tmp_path, "b")]

    result = svc.preflight_lora_fuse(base, loras, weights="0.5,0.6,0.7", output_name="fused")

    assert not result.ok
    assert "Weights must be numbers" in result.markdown()


def test_conversion_preflight_allows_single_to_diffusers(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"fake")

    result = svc.preflight_conversion(
        source_path=str(source),
        operation="single-to-diffusers",
        output_name="converted",
        architecture="sdxl",
    )

    assert result.ok
    assert result.command is not None
    assert "single-to-diffusers" in result.command.args


def test_fp16_quantization_builds_real_export_command(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"fake")

    result = svc.preflight_quantization(
        source_path=str(source),
        target="model",
        quant="fp16",
        output_name="small",
        architecture="sdxl",
    )

    assert result.ok
    assert result.command is not None
    assert result.command.name == "model-ops-quantize"
    assert "converted safetensors copy" in result.markdown()


def test_diffusers_to_single_is_blocked_first_pass(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    source = tmp_path / "folder"
    source.mkdir()
    (source / "model_index.json").write_text("{}", encoding="utf-8")

    result = svc.preflight_conversion(
        source_path=str(source),
        operation="diffusers-to-single",
        output_name="single",
        architecture="sdxl",
    )

    assert not result.ok
    assert "preflight-only" in result.markdown()


def test_onnx_export_is_blocked_until_exporter_is_wired(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    source = tmp_path / "folder"
    source.mkdir()
    (source / "model_index.json").write_text("{}", encoding="utf-8")

    result = svc.preflight_conversion(
        source_path=str(source),
        operation="onnx-export",
        output_name="onnx_model",
        architecture="sdxl",
    )

    assert not result.ok
    assert result.command is None
    assert "Optimum exporter path" in result.markdown()


def test_nvfp4_quant_warns_storage_only(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"fake")

    result = svc.preflight_quantization(
        source_path=str(source),
        target="model",
        quant="nvfp4",
        output_name="small",
        architecture="sdxl",
    )

    assert result.ok
    assert "storage" in result.markdown().lower()
    assert "RTX 4070 Ti SUPER" in result.markdown()
    assert "receipt only" in result.markdown().lower()


def test_aiwf_fp8_ready_quantization_builds_real_export_command(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    source = tmp_path / "wan_high.safetensors"
    source.write_bytes(b"fake")

    result = svc.preflight_quantization(
        source_path=str(source),
        target="model",
        quant="aiwf_fp8_ready",
        output_name="wan_high_aiwf_fp8",
        architecture="wan",
    )

    text = result.markdown()
    assert result.ok
    assert result.command is not None
    assert "--quant" in result.command.args
    assert "aiwf_fp8_ready" in result.command.args
    assert "receipt only" not in text.lower()
    assert "AIWF FP8-ready safetensors package" in text
    assert ".weight_scale" in text
    assert "strict no-fallback" in text


def test_vae_quantization_is_preflight_only(tmp_path: Path):
    svc = ModelOpsService(_flags(tmp_path))
    source = tmp_path / "vae.safetensors"
    source.write_bytes(b"fake")

    result = svc.preflight_quantization(
        source_path=str(source),
        target="vae",
        quant="nvfp4",
        output_name="vae_small",
        architecture="sdxl",
    )

    assert not result.ok
    assert "VAE quantization is preflight-only" in result.markdown()


def test_worker_quantize_fp16_writes_safetensors_and_receipt(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from aiwf.workers import model_ops as worker_model_ops

    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    receipt = tmp_path / "output.safetensors.receipt.json"
    safetensors.save_file(
        {
            "float_weight": torch.ones((2, 2), dtype=torch.float32),
            "integer_weight": torch.ones((2,), dtype=torch.int64),
        },
        str(source),
        metadata={"modelspec.architecture": "sdxl"},
    )

    rc = worker_model_ops.quantize(
        SimpleNamespace(
            source=str(source),
            output=str(output),
            target="model",
            quant="fp16",
            architecture="sdxl",
            receipt=str(receipt),
        )
    )

    state = safetensors.load_file(str(output), device="cpu")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert rc == 0
    assert state["float_weight"].dtype == torch.float16
    assert state["integer_weight"].dtype == torch.int64
    assert data["operation"] == "quantize_dtype_export"
    assert data["output"] == str(output)


def test_worker_quantize_aiwf_fp8_ready_writes_scaled_fp8_package(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is not available")
    from aiwf.workers import model_ops as worker_model_ops

    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    receipt = tmp_path / "output.safetensors.receipt.json"
    safetensors.save_file(
        {
            "blocks.0.attn.to_q.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "blocks.0.attn.to_q.weight_scale": torch.tensor(999.0, dtype=torch.float32),
            "text_embedding.weight": torch.ones((4, 4), dtype=torch.float32),
            "patch_embedding.weight": torch.ones((2, 2, 3, 3), dtype=torch.float32),
            "integer_weight": torch.ones((2,), dtype=torch.int64),
        },
        str(source),
        metadata={"modelspec.architecture": "wan"},
    )

    rc = worker_model_ops.quantize(
        SimpleNamespace(
            source=str(source),
            output=str(output),
            target="model",
            quant="aiwf_fp8_ready",
            architecture="wan",
            receipt=str(receipt),
        )
    )

    state = safetensors.load_file(str(output), device="cpu")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert rc == 0
    assert state["blocks.0.attn.to_q.weight"].dtype == torch.float8_e4m3fn
    assert state["blocks.0.attn.to_q.weight_scale"].dtype == torch.float32
    assert state["blocks.0.attn.to_q.weight_scale"].reshape(()).item() != pytest.approx(999.0)
    assert state["text_embedding.weight"].dtype == torch.float32
    assert state["patch_embedding.weight"].dtype == torch.float32
    assert state["integer_weight"].dtype == torch.int64
    assert data["operation"] == "quantize_aiwf_fp8_ready"
    assert data["output"] == str(output)
    assert data["fp8_tensor_count"] == 1
    assert data["scale_tensor_count"] == 1
    assert data["skipped_existing_scale_count"] == 1


def test_write_model_op_receipt(tmp_path: Path):
    receipt = write_model_op_receipt(tmp_path / "receipt.json", {"operation": "test", "warnings": []})

    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["app"] == "AIWF Studio"
    assert data["operation"] == "test"
    assert "created_at" in data
