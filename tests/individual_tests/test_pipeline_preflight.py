from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.ltx import (
    LTX_DIFFUSERS_2B_CHECKPOINT,
    LTX_FULL_CHECKPOINT,
    LTX_FULL_CHECKPOINT_FP8,
    LTX_GEMMA_BACKEND_GGUF,
    LTX_HERETIC_Q3_CONVERTED_FOLDER,
    LTX_HERETIC_Q3_GGUF,
    LTX_GEMMA_REPO,
    LTX_PIPELINE_DIFFUSERS_2B,
    LTX_PIPELINE_ONE_STAGE,
    LTX_T5XXL_FP16,
    LtxVideoRequest,
)
from aiwf.core.domain.sana_video import SanaVideoRequest
from aiwf.services.pipeline_preflight import (
    preflight_anima_pipeline,
    preflight_diffusers_pipeline,
    preflight_krea2_pipeline,
    preflight_ltx_pipeline,
    preflight_onnx_pipeline,
    preflight_qwen_nunchaku_pipeline,
    preflight_sana_video_pipeline,
    preflight_wan_pipeline,
)
from aiwf.services.worker_tenant import python_exe_for_venv


def _onnx_dir(root: Path, *, tokenizer: bool = True) -> Path:
    model = root / "sdxl_onnx"
    for sub in ("text_encoder", "unet", "vae_decoder"):
        path = model / sub
        path.mkdir(parents=True)
        (path / "model.onnx").write_bytes(b"fake")
    if tokenizer:
        tok = model / "tokenizer"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def _write_ltx_t5_tokenizer(root: Path) -> None:
    tokenizer = root / "models" / "ltx" / "tokenizer" / "t5-v1_1-xxl"
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "config.json").write_text(json.dumps({"model_type": "t5", "vocab_size": 32128}), encoding="utf-8")
    (tokenizer / "special_tokens_map.json").write_text(json.dumps({"pad_token": "<pad>"}), encoding="utf-8")
    (tokenizer / "spiece.model").write_bytes(b"sentencepiece")
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({"extra_ids": 100}), encoding="utf-8")


def test_onnx_preflight_passes_complete_folder_with_cuda_provider(tmp_path: Path):
    model = _onnx_dir(tmp_path)

    result = preflight_onnx_pipeline(
        model,
        provider_preference="cuda",
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert result.ok
    assert "CUDAExecutionProvider is available" in result.markdown()


def test_onnx_preflight_blocks_missing_tokenizer(tmp_path: Path):
    model = _onnx_dir(tmp_path, tokenizer=False)

    result = preflight_onnx_pipeline(
        model,
        provider_preference="auto",
        available_providers=["CPUExecutionProvider"],
    )

    assert not result.ok
    assert "tokenizer" in result.markdown()
    assert "Expected local tokenizer assets" in result.markdown()


def test_onnx_preflight_blocks_zero_byte_model_files(tmp_path: Path):
    model = _onnx_dir(tmp_path)
    (model / "unet" / "model.onnx").write_bytes(b"")

    result = preflight_onnx_pipeline(
        model,
        provider_preference="cpu",
        available_providers=["CPUExecutionProvider"],
    )

    assert not result.ok
    assert "non-empty unet/model.onnx" in result.markdown()


def test_onnx_preflight_blocks_provider_mismatch(tmp_path: Path):
    model = _onnx_dir(tmp_path)

    result = preflight_onnx_pipeline(
        model,
        provider_preference="cuda",
        available_providers=["CPUExecutionProvider"],
    )

    assert not result.ok
    assert "CUDAExecutionProvider is not available" in result.markdown()


def test_onnx_preflight_auto_selects_available_provider(tmp_path: Path):
    model = _onnx_dir(tmp_path)

    result = preflight_onnx_pipeline(
        model,
        provider_preference="auto",
        available_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )

    assert result.ok
    assert "auto will use DmlExecutionProvider" in result.markdown()


def test_onnx_backend_exposes_preflight(tmp_path: Path):
    model = _onnx_dir(tmp_path)

    with patch(
        "aiwf.services.pipeline_preflight._load_available_onnx_providers",
        return_value=["CPUExecutionProvider"],
    ):
        from aiwf.infrastructure.onnx.backend import ONNXBackend

        backend = ONNXBackend(tmp_path, provider="cpu")
        result = backend.preflight_checkpoint(model.name)

    assert result.ok
    assert result.metadata["model_dir"] == str(model.resolve())


def test_diffusers_preflight_reports_transformers_5_as_blocked(monkeypatch):
    fake_transformers = types.SimpleNamespace(__version__="5.0.0")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "safetensors", types.SimpleNamespace())

    result = preflight_diffusers_pipeline()

    assert not result.ok
    assert "unsupported" in result.markdown()


def test_qwen_nunchaku_preflight_rejects_empty_asset_placeholders(tmp_path: Path):
    engine = tmp_path / "engines" / "qwen_nunchaku"
    python = engine / ".venv" / "Scripts" / "python.exe"
    runner = engine / "run_qwen_lightning.py"
    base_dir = tmp_path / "models" / "qwen-image" / "Diffusers" / "Qwen-Image"
    transformer = tmp_path / "models" / "qwen-image" / "Nunchaku" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("print('ok')", encoding="utf-8")
    base_dir.mkdir(parents=True)
    (base_dir / "model_index.json").write_text("{}", encoding="utf-8")
    transformer.parent.mkdir(parents=True, exist_ok=True)
    transformer.write_bytes(b"")

    result = preflight_qwen_nunchaku_pipeline(tmp_path)

    assert not result.ok
    assert any(not item.ok for item in result.items if item.name in {"base components", "transformer"})
    assert result.metadata["transformer_path"].endswith(transformer.name)
    assert result.metadata["storage_mode"] == "single_transformer_safetensors_plus_base_components"


def test_qwen_nunchaku_preflight_uses_configured_shared_roots(tmp_path: Path, monkeypatch):
    from aiwf.core.config.settings import RuntimeFlags

    monkeypatch.setattr(
        "aiwf.services.qwen_nunchaku.diffusers_dir_has_required_local_files",
        lambda path: True,
    )
    monkeypatch.setattr("aiwf.services.qwen_nunchaku._runtime_dependency_issue", lambda _python: "")
    engine = tmp_path / "engines" / "qwen_nunchaku"
    python = engine / ".venv" / "Scripts" / "python.exe"
    runner = engine / "run_qwen_lightning.py"
    shared = tmp_path / "shared-models"
    base_dir = shared / "qwen-image" / "Diffusers" / "Qwen-Image"
    transformer = shared / "qwen-image" / "Nunchaku" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"python")
    runner.write_text("print('ok')", encoding="utf-8")
    base_dir.mkdir(parents=True)
    (base_dir / "model_index.json").write_text("{}", encoding="utf-8")
    transformer.parent.mkdir(parents=True, exist_ok=True)
    transformer.write_bytes(b"transformer")
    flags = RuntimeFlags(
        data_dir=tmp_path,
        models_dir=tmp_path / "primary-models",
        extra_model_dirs=[shared],
    )

    result = preflight_qwen_nunchaku_pipeline(flags)

    assert not result.ok
    assert result.metadata["runtime_ready"] is True
    assert result.metadata["generation_verified"] is False
    assert next(item for item in result.items if item.name == "generation smoke").ok is False
    assert Path(result.metadata["base_dir"]) == base_dir.resolve()
    assert Path(result.metadata["transformer_path"]) == transformer.resolve()


def test_qwen_nunchaku_preflight_blocks_incomplete_base_snapshot(tmp_path: Path):
    engine = tmp_path / "engines" / "qwen_nunchaku"
    python = engine / ".venv" / "Scripts" / "python.exe"
    runner = engine / "run_qwen_lightning.py"
    base_dir = tmp_path / "models" / "qwen-image" / "Diffusers" / "Qwen-Image"
    transformer_dir = base_dir / "transformer"
    transformer_path = (
        tmp_path
        / "models"
        / "qwen-image"
        / "Nunchaku"
        / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
    )
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")
    runner.write_text("print('ok')", encoding="utf-8")
    transformer_dir.mkdir(parents=True)
    (base_dir / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}), encoding="utf-8")
    (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
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
    transformer_path.parent.mkdir(parents=True, exist_ok=True)
    transformer_path.write_bytes(b"")

    result = preflight_qwen_nunchaku_pipeline(tmp_path)

    assert not result.ok
    assert "base components incomplete" in result.markdown()


def test_qwen_nunchaku_preflight_blocks_missing_runtime(tmp_path: Path):
    result = preflight_qwen_nunchaku_pipeline(tmp_path)

    assert not result.ok
    assert "engine runtime missing" in result.markdown().lower() or "missing engine runtime" in result.markdown().lower()


def test_krea2_preflight_reports_missing_runtime_class_and_assets(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace())

    result = preflight_krea2_pipeline(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    assert not result.ok
    text = result.markdown()
    assert "Krea2Pipeline" in text
    assert "Krea 2 Diffusers folder" in text
    assert "Krea 2 split transformer" in text
    assert "diffusers>=0.39.0" in " ".join(result.warnings)
    assert result.metadata["recommended_bundle"].startswith("krea2")


def test_krea2_preflight_detects_split_sidecars(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace())

    models = tmp_path / "models"
    (models / "krea2" / "UNet").mkdir(parents=True)
    (models / "krea2" / "Textencoder").mkdir(parents=True)
    (models / "krea2" / "VAE").mkdir(parents=True)
    (models / "krea2" / "UNet" / "krea2_turbo_fp8_scaled.safetensors").write_bytes(b"fake")
    (models / "krea2" / "Textencoder" / "qwen3vl_4b_fp8_scaled.safetensors").write_bytes(b"fake")
    (models / "krea2" / "VAE" / "qwen_image_vae.safetensors").write_bytes(b"fake")

    result = preflight_krea2_pipeline(RuntimeFlags(data_dir=tmp_path, models_dir=models))

    assert not result.ok
    by_name = {item.name: item for item in result.items}
    assert by_name["Krea 2 split transformer"].ok
    assert by_name["Qwen3-VL text encoder"].ok
    assert by_name["Qwen Image VAE"].ok
    assert not by_name["Krea2Pipeline"].ok
    assert not by_name["Krea 2 Diffusers folder"].ok
    assert "split-file loader" in " ".join(result.warnings)


def test_krea2_preflight_discovers_assets_in_shared_root(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace())
    primary = tmp_path / "primary-models"
    shared = tmp_path / "shared-models"
    for relative, payload in (
        (Path("krea2/UNet/krea2_turbo_fp8_scaled.safetensors"), b"transformer"),
        (Path("krea2/Textencoder/qwen3vl_4b_fp8_scaled.safetensors"), b"encoder"),
        (Path("krea2/VAE/qwen_image_vae.safetensors"), b"vae"),
    ):
        path = shared / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    result = preflight_krea2_pipeline(
        RuntimeFlags(data_dir=tmp_path, models_dir=primary, extra_model_dirs=[shared])
    )

    by_name = {item.name: item for item in result.items}
    assert by_name["Krea 2 split transformer"].path == shared / "krea2" / "UNet" / "krea2_turbo_fp8_scaled.safetensors"
    assert by_name["Qwen3-VL text encoder"].path == shared / "krea2" / "Textencoder" / "qwen3vl_4b_fp8_scaled.safetensors"
    assert by_name["Qwen Image VAE"].path == shared / "krea2" / "VAE" / "qwen_image_vae.safetensors"


def test_krea2_preflight_accepts_complete_diffusers_folder_without_split_sidecars(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    root = models / "krea2" / "Diffusers" / "Krea-2-Turbo"
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "Krea2Pipeline", "transformer": ["diffusers", "Krea2Transformer2DModel"]}),
        encoding="utf-8",
    )
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "single_transformer_blocks.0.attn.to_q.weight": "diffusion_pytorch_model-00001-of-00001.safetensors"
                },
            }
        ),
        encoding="utf-8",
    )
    (transformer / "diffusion_pytorch_model-00001-of-00001.safetensors").write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(Krea2Pipeline=object))

    result = preflight_krea2_pipeline(RuntimeFlags(data_dir=tmp_path, models_dir=models))

    assert result.ok
    by_name = {item.name: item for item in result.items}
    assert by_name["Krea2Pipeline"].ok
    assert by_name["Krea 2 Diffusers folder"].ok
    assert not by_name["Krea 2 split transformer"].ok
    assert result.metadata["diffusers_folder"] == str(root)


def test_anima_preflight_detects_split_sidecars_but_blocks_loader(tmp_path: Path):
    models = tmp_path / "models"
    (models / "anima" / "UNet").mkdir(parents=True)
    (models / "anima" / "Textencoder").mkdir(parents=True)
    (models / "anima" / "VAE").mkdir(parents=True)
    (models / "anima" / "UNet" / "anima-base-v1.0.safetensors").write_bytes(b"fake")
    (models / "anima" / "Textencoder" / "qwen_3_06b_base.safetensors").write_bytes(b"fake")
    (models / "anima" / "VAE" / "qwen_image_vae.safetensors").write_bytes(b"fake")

    result = preflight_anima_pipeline(RuntimeFlags(data_dir=tmp_path, models_dir=models))

    assert not result.ok
    by_name = {item.name: item for item in result.items}
    assert not by_name["native Anima loader"].ok
    assert by_name["Anima transformer"].ok
    assert by_name["Qwen 0.6B text encoder"].ok
    assert by_name["Qwen Image VAE"].ok


def test_anima_preflight_discovers_blocked_assets_in_shared_root(tmp_path: Path):
    shared = tmp_path / "shared-models"
    assets = (
        ("anima/UNet/anima-base-v1.0.safetensors", b"transformer"),
        ("anima/Textencoder/qwen_3_06b_base.safetensors", b"encoder"),
        ("anima/VAE/qwen_image_vae.safetensors", b"vae"),
    )
    for relative, payload in assets:
        path = shared / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    result = preflight_anima_pipeline(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "primary", extra_model_dirs=[shared])
    )

    by_name = {item.name: item for item in result.items}
    assert not by_name["native Anima loader"].ok
    assert by_name["Anima transformer"].path == shared / assets[0][0]
    assert by_name["Qwen 0.6B text encoder"].path == shared / assets[1][0]
    assert by_name["Qwen Image VAE"].path == shared / assets[2][0]


def _write_fake_safetensors(path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.save_file({"blocks.0.weight": torch.ones(1)}, path)


def _write_wan_fast_5b_assets(root: Path) -> RuntimeFlags:
    flags = RuntimeFlags(data_dir=root, models_dir=root / "models", output_dir=root / "outputs")
    component_base = flags.resolved_models_dir() / "wan" / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    (component_base / "text_encoder").mkdir(parents=True)
    (component_base / "tokenizer").mkdir()
    (component_base / "scheduler").mkdir()
    (component_base / "model_index.json").write_text('{"_class_name": "WanPipeline"}', encoding="utf-8")
    (component_base / "text_encoder" / "config.json").write_text('{"hidden_size": 8}', encoding="utf-8")
    (component_base / "text_encoder" / "model.safetensors").write_bytes(b"fake")
    (component_base / "tokenizer" / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (component_base / "scheduler" / "scheduler_config.json").write_text('{"_class_name": "FlowMatchEulerDiscreteScheduler"}', encoding="utf-8")
    _write_fake_safetensors(
        flags.resolved_models_dir() / "wan" / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors"
    )
    _write_fake_safetensors(flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors")
    return flags


def _write_ready_ltx_worker(root: Path) -> None:
    worker = root / "engines" / "ltx" / "worker.py"
    repo = root / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(root / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (root / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")


def _write_ready_ltx_gemma(root: Path, *, indexed: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"gemma3"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text('{"tokenizer_class":"GemmaTokenizer"}', encoding="utf-8")
    (root / "preprocessor_config.json").write_text('{"image_processor_type":"Gemma3ImageProcessor"}', encoding="utf-8")
    if indexed:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"}}),
            encoding="utf-8",
        )
        (root / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    else:
        (root / "model.safetensors").write_bytes(b"weights")
    return root


def test_wan_fast_5b_preflight_reports_ready_local_assets(tmp_path: Path, monkeypatch):
    flags = _write_wan_fast_5b_assets(tmp_path)
    monkeypatch.setattr("aiwf.services.wan.WanService.available", lambda self: True)

    result = preflight_wan_pipeline(flags)

    assert result.ok
    assert result.metadata["runtime_mode"] == "fast_5b"
    assert result.metadata["sampler"] == "unipc"
    assert result.metadata["offload"] == "balanced"


def test_wan_fast_5b_full_diffusers_snapshot_uses_its_embedded_vae(tmp_path: Path, monkeypatch):
    flags = _write_wan_fast_5b_assets(tmp_path)
    monkeypatch.setattr("aiwf.services.wan.WanService.available", lambda self: True)
    model = flags.resolved_models_dir() / "wan" / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    transformer = model / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    vae = model / "vae"
    vae.mkdir()
    (vae / "config.json").write_text("{}", encoding="utf-8")
    (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (model / "model_index.json").write_text('{"_class_name":"WanImageToVideoPipeline"}', encoding="utf-8")
    (flags.resolved_models_dir() / "wan" / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors").unlink()
    (flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors").unlink()
    (flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors").write_bytes(b"wrong-generation-sidecar")

    result = preflight_wan_pipeline(flags)

    assert result.ok, result.errors
    assert Path(result.metadata["vae"]).resolve() == vae.resolve()


def test_ltx_preflight_uses_installed_one_stage_when_distilled_missing(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    _write_ready_ltx_gemma(gemma)

    result = preflight_ltx_pipeline(flags)

    assert result.ok
    assert result.metadata["selected_pipeline"] == LTX_PIPELINE_ONE_STAGE
    assert any("falls back" in warning for warning in result.warnings)


def test_ltx_preflight_prefers_fp8_one_stage_no_offload(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT_FP8
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    _write_ready_ltx_gemma(gemma)

    result = preflight_ltx_pipeline(flags, request=LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE))

    assert result.ok
    assert result.metadata["checkpoint_path"] == str(checkpoint.resolve())
    assert result.metadata["offload"] == "none"
    assert result.metadata["quantization"] == "fp8-cast"
    assert any("FP8 checkpoint uses offload=none" in warning for warning in result.warnings)


def test_ltx_preflight_uses_converted_heretic_gemma_root_when_complete(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT_FP8
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    converted = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_HERETIC_Q3_CONVERTED_FOLDER
    converted.mkdir(parents=True)
    for filename in (
        "model.safetensors",
        "vision_projector.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.model",
    ):
        (converted / filename).write_bytes(b"fake")

    result = preflight_ltx_pipeline(flags, request=LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE))

    assert result.ok
    assert result.metadata["gemma_root"] == str(converted.resolve())


def test_ltx_preflight_accepts_complete_indexed_hf_gemma_assets(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    _write_ready_ltx_gemma(gemma, indexed=True)

    result = preflight_ltx_pipeline(flags, request=LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE))

    assert result.ok
    assert "OK **Gemma tokenizer/processor:**" in result.markdown()

    (gemma / "model-00001-of-00001.safetensors").unlink()
    incomplete = preflight_ltx_pipeline(flags, request=LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE))

    assert not incomplete.ok
    assert "model.safetensors.index.json" in incomplete.markdown()


def test_ltx_preflight_rejects_empty_hf_gemma_directory(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    gemma.mkdir(parents=True)

    result = preflight_ltx_pipeline(flags, request=LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE))

    assert not result.ok
    assert "incomplete Gemma assets" in result.markdown()


def test_ltx_gguf_preflight_reports_generation_as_blocked_even_when_assets_exist(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    _write_ready_ltx_gemma(gemma)
    gguf = flags.resolved_models_dir() / "LLM" / "GGUF" / LTX_HERETIC_Q3_GGUF
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    request = LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE, gemma_backend=LTX_GEMMA_BACKEND_GGUF, gemma_gguf_path=str(gguf))

    result = preflight_ltx_pipeline(flags, request=request)

    assert not result.ok
    assert "native Gemma GGUF generation backend" in result.markdown()
    assert "every Gemma hidden-state layer" in result.markdown()


def test_ltx_preflight_blocks_unloadable_native_checkpoint(tmp_path: Path, monkeypatch):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    gemma.mkdir(parents=True)
    monkeypatch.setattr("aiwf.services.ltx.ltx_checkpoint_openability_error", lambda _path: "pagefile too small")

    result = preflight_ltx_pipeline(flags, request=LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE))

    assert not result.ok
    assert "checkpoint openability" in result.markdown()
    assert "pagefile too small" in result.markdown()


def test_ltx_preflight_blocks_native_worker_runtime_crash(tmp_path: Path, monkeypatch):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    gemma.mkdir(parents=True)
    monkeypatch.setattr("aiwf.services.ltx.ltx_checkpoint_openability_error", lambda _path: "")
    monkeypatch.setattr("aiwf.services.ltx.ltx_native_checkpoint_runtime_blocker", lambda _path: "access violation 3221225477")

    result = preflight_ltx_pipeline(flags, request=LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE))

    assert not result.ok
    assert "native worker compatibility" in result.markdown()
    assert "access violation 3221225477" in result.markdown()


def test_ltx_preflight_uses_local_diffusers_2b_without_worker(tmp_path: Path, monkeypatch):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_DIFFUSERS_2B_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    t5 = flags.resolved_models_dir() / "flux" / "Textencoder" / LTX_T5XXL_FP16
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake")
    _write_ltx_t5_tokenizer(tmp_path)
    monkeypatch.setattr(
        "aiwf.services.ltx_diffusers.ltx2b_diffusers_runtime_error",
        lambda: None,
    )

    result = preflight_ltx_pipeline(flags)

    assert result.ok
    assert result.pipeline == "LTX 2B"
    assert result.metadata["selected_pipeline"] == LTX_PIPELINE_DIFFUSERS_2B
    assert result.metadata["t5_encoder_path"] == str(t5.resolve())


def test_ltx_diffusers_2b_preflight_blocks_missing_in_process_runtime(tmp_path: Path, monkeypatch):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_DIFFUSERS_2B_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    t5 = flags.resolved_models_dir() / "flux" / "Textencoder" / LTX_T5XXL_FP16
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake")
    _write_ltx_t5_tokenizer(tmp_path)
    monkeypatch.setattr(
        "aiwf.services.ltx_diffusers.ltx2b_diffusers_runtime_error",
        lambda: "The installed diffusers runtime does not expose LTXPipeline.from_single_file.",
    )

    result = preflight_ltx_pipeline(
        flags,
        request=LtxVideoRequest(pipeline=LTX_PIPELINE_DIFFUSERS_2B),
    )

    assert not result.ok
    runtime_item = next(item for item in result.items if item.name == "Diffusers LTX 2B runtime")
    assert not runtime_item.ok
    assert "LTXPipeline.from_single_file" in result.markdown()


def test_ltx_diffusers_2b_preflight_rejects_source_image(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_DIFFUSERS_2B_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    t5 = flags.resolved_models_dir() / "flux" / "Textencoder" / LTX_T5XXL_FP16
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake")
    _write_ltx_t5_tokenizer(tmp_path)
    source = tmp_path / "source.png"
    source.write_bytes(b"image")

    result = preflight_ltx_pipeline(
        flags,
        request=LtxVideoRequest(pipeline=LTX_PIPELINE_DIFFUSERS_2B, source_image_path=str(source)),
    )

    assert not result.ok
    item = next(item for item in result.items if item.name == "source image compatibility")
    assert not item.ok
    assert "text-to-video only" in item.message


@pytest.mark.parametrize("empty_asset", ["checkpoint", "t5", "tokenizer"])
def test_ltx_diffusers_2b_preflight_rejects_empty_local_assets(tmp_path: Path, empty_asset: str):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_DIFFUSERS_2B_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"" if empty_asset == "checkpoint" else b"checkpoint")
    t5 = flags.resolved_models_dir() / "flux" / "Textencoder" / LTX_T5XXL_FP16
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"" if empty_asset == "t5" else b"t5")
    if empty_asset != "tokenizer":
        _write_ltx_t5_tokenizer(tmp_path)

    result = preflight_ltx_pipeline(
        flags,
        request=LtxVideoRequest(pipeline=LTX_PIPELINE_DIFFUSERS_2B),
    )

    assert not result.ok
    item_name = "checkpoint" if empty_asset == "checkpoint" else "T5 tokenizer" if empty_asset == "tokenizer" else "T5XXL text encoder"
    item = next(item for item in result.items if item.name == item_name)
    assert not item.ok
    assert "empty" in item.message or "incomplete" in item.message


def test_ltx_preflight_blocks_missing_launch_checkpoint(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    gemma.mkdir(parents=True)

    result = preflight_ltx_pipeline(flags)

    assert not result.ok
    assert "missing LTX checkpoint" in result.markdown()


def test_sana_video_preflight_reports_runtime_and_default_model_path(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")

    result = preflight_sana_video_pipeline(flags)

    assert not result.ok
    assert result.metadata["default_repo"] == "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    assert result.metadata["model_path"].endswith("SANA-Video_2B_480p_diffusers")
    assert result.metadata["model_installed"] == "false"
    assert "sage_attention" in result.metadata
    assert "bitsandbytes" in result.metadata
    assert result.metadata["default_quantization"] == "auto"
    assert result.metadata["vae_tiling"] == "auto"
    assert any("silent MP4" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("pipeline", "required_class"),
    [
        ("text_to_video", "SanaVideoPipeline"),
        ("image_to_video", "SanaImageToVideoPipeline"),
    ],
)
def test_sana_video_preflight_checks_only_the_selected_pipeline_class(
    tmp_path: Path,
    monkeypatch,
    pipeline: str,
    required_class: str,
):
    from aiwf.services import pipeline_preflight

    checked_classes: list[str] = []

    def record_class_check(name: str, message: str):
        checked_classes.append(name)
        return pipeline_preflight.PipelineCheckItem(name, True, message)

    monkeypatch.setattr(pipeline_preflight, "_diffusers_attr_check", record_class_check)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")

    pipeline_preflight.preflight_sana_video_pipeline(
        flags,
        request=SanaVideoRequest(pipeline=pipeline),
    )

    assert checked_classes == [required_class]


def test_sana_video_preflight_uses_selected_variant_path(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    selected = flags.resolved_models_dir() / "sana-video" / "Diffusers" / "SANA-Video_2B_720p_diffusers"
    selected.mkdir(parents=True)
    (selected / "model_index.json").write_text("{}", encoding="utf-8")

    result = preflight_sana_video_pipeline(flags, request=SanaVideoRequest(model_variant="720p"))

    assert result.metadata["model_variant"] == "720p"
    assert result.metadata["default_repo"].endswith("720p_diffusers")
    assert result.metadata["model_path"] == str(selected.resolve())


def test_sana_video_preflight_rejects_snapshot_with_missing_encoder_shard(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    model = flags.resolved_models_dir() / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    (model / "text_encoder").mkdir(parents=True)
    (model / "model_index.json").write_text(
        json.dumps({"_class_name": "SanaVideoPipeline", "text_encoder": ["transformers", "T5EncoderModel"]}),
        encoding="utf-8",
    )
    (model / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (model / "text_encoder" / "model-00002-of-00002.safetensors").write_bytes(b"present")

    result = preflight_sana_video_pipeline(flags)

    assert not result.ok
    assert result.metadata["model_installed"] == "false"
    assert "model-00001-of-00002.safetensors" in result.metadata["model_status"]


def test_sana_video_preflight_rejects_model_index_without_pipeline_components(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    model = flags.resolved_models_dir() / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text(
        json.dumps({"_class_name": "SanaVideoPipeline"}),
        encoding="utf-8",
    )

    result = preflight_sana_video_pipeline(flags)

    assert not result.ok
    assert result.metadata["model_installed"] == "false"
    assert "scheduler_config.json" in result.metadata["model_status"]
    assert "transformer" in result.metadata["model_status"]
    assert "text_encoder" in result.metadata["model_status"]
