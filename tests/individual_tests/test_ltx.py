from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.ltx import (
    LTX_DIFFUSERS_2B_CHECKPOINT,
    LTX_FULL_CHECKPOINT,
    LTX_FULL_CHECKPOINT_FP8,
    LTX_GEMMA_BACKEND_GGUF,
    LTX_GEMMA_REPO,
    LTX_PIPELINE_DIFFUSERS_2B,
    LTX_PIPELINE_DISTILLED,
    LTX_PIPELINE_ONE_STAGE,
    LTX_HERETIC_Q3_GGUF,
    LTX_T5XXL_FP16,
    LtxVideoRequest,
    snap_ltx_num_frames,
)
from aiwf.services.ltx import LtxService, LtxUnavailable, _last_error, ltx_checkpoint_openability_error
from aiwf.services.worker_tenant import WorkerTenantRegistry, python_exe_for_venv
from aiwf.web.tabs.wan_i2v import _ltx_quantization_default
from scripts.probe_ltx_runtime import (
    _extract_recent_output,
    _is_clean_ltx_gguf_blocker,
    _is_gemma_gguf_candidate,
    _mark_smallest_heretic,
    _quant_from_filename,
)
from scripts.convert_gemma_gguf_to_hf import _output_shape


@pytest.fixture(autouse=True)
def _make_device_memory_probe_deterministic(monkeypatch):
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi test isolation")),
    )


def _write_ltx_t5_tokenizer(service: LtxService) -> Path:
    tokenizer = service.default_t5_tokenizer_path()
    tokenizer.mkdir(parents=True, exist_ok=True)
    (tokenizer / "config.json").write_text(json.dumps({"model_type": "t5", "vocab_size": 32128}), encoding="utf-8")
    (tokenizer / "special_tokens_map.json").write_text(json.dumps({"pad_token": "<pad>"}), encoding="utf-8")
    (tokenizer / "spiece.model").write_bytes(b"sentencepiece")
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({"extra_ids": 100}), encoding="utf-8")
    return tokenizer


def _write_ltx_gemma_assets(root: Path, *, with_hf_weights: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text('{"tokenizer_class":"GemmaTokenizer"}', encoding="utf-8")
    (root / "preprocessor_config.json").write_text('{"image_processor_type":"Gemma3ImageProcessor"}', encoding="utf-8")
    if with_hf_weights:
        (root / "config.json").write_text('{"model_type":"gemma3"}', encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"weights")
    return root


def test_ltx2b_cache_identity_changes_when_checkpoint_is_replaced_in_place(tmp_path: Path, monkeypatch):
    from aiwf.services import ltx_diffusers

    checkpoint = tmp_path / "ltx-2b.safetensors"
    t5_weights = tmp_path / "t5.safetensors"
    checkpoint.write_bytes(b"old checkpoint")
    t5_weights.write_bytes(b"encoder weights")
    monkeypatch.setattr(ltx_diffusers, "_ltx_dtype", lambda: "bf16")
    monkeypatch.setattr(ltx_diffusers, "_ltx_should_offload", lambda _path: True)
    before = ltx_diffusers._ltx2b_cache_key(
        checkpoint=checkpoint, t5_weights=t5_weights, tokenizer_id="t5-v1_1-xxl"
    )
    original_stat = checkpoint.stat()
    checkpoint.write_bytes(b"replacement checkpoint")
    import os
    os.utime(checkpoint, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))

    after = ltx_diffusers._ltx2b_cache_key(
        checkpoint=checkpoint, t5_weights=t5_weights, tokenizer_id="t5-v1_1-xxl"
    )

    assert before != after


def test_ltx_frame_count_validation():
    assert snap_ltx_num_frames(80) == 81
    assert LtxVideoRequest(prompt="dance", num_frames=81).num_frames == 81


def test_ltx_ui_quantization_default_matches_selected_pipeline():
    assert _ltx_quantization_default(LTX_PIPELINE_DIFFUSERS_2B) == ""
    assert _ltx_quantization_default(LTX_PIPELINE_ONE_STAGE) == "fp8-cast"
    assert _ltx_quantization_default(LTX_PIPELINE_DISTILLED) == "fp8-cast"


def test_ltx_gradio_initial_status_does_not_claim_route_ready():
    ui_source = (Path(__file__).resolve().parents[2] / "aiwf" / "web" / "tabs" / "wan_i2v.py").read_text(encoding="utf-8")
    assert '"**LTX route not checked** - Readiness is checked for the selected pipeline when you generate."' in ui_source
    assert '"**LTX ready**' not in ui_source

    with pytest.raises(ValueError, match="8\\*k\\+1"):
        LtxVideoRequest(prompt="dance", num_frames=82)


def test_ltx_request_requires_32_multiple_resolution():
    with pytest.raises(ValueError, match="divisible by 32"):
        LtxVideoRequest(prompt="dance", width=500, height=512)


def test_ltx_usable_default_request_targets_one_stage_smoke():
    request = LtxVideoRequest(prompt="dance")

    assert request.pipeline == LTX_PIPELINE_ONE_STAGE
    assert request.num_frames == 9
    assert request.fps == 8
    assert request.steps == 1
    assert request.width == 128
    assert request.height == 128
    assert request.offload == "disk"


def test_ltx_smoketest_batch_is_automation_safe():
    text = Path("scripts/run_ltx_smoketest.bat").read_text(encoding="utf-8")

    assert 'if not "%AIWF_NO_PAUSE%"=="1" pause' in text
    assert "set \"EXIT_CODE=%ERRORLEVEL%\"" in text
    assert "exit /b %EXIT_CODE%" in text


def test_ltx_service_blocks_when_engine_missing(tmp_path: Path):
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )

    with pytest.raises(LtxUnavailable, match="LTX 2.3 engine is not ready"):
        service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_DISTILLED))


def test_ltx_service_blocks_when_models_missing_after_engine_ready(tmp_path: Path):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"ltx": {"enabled": True}}),
        encoding="utf-8",
    )
    registry = WorkerTenantRegistry(tmp_path)
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
        registry=registry,
    )

    with pytest.raises(LtxUnavailable, match="checkpoint missing"):
        service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_DISTILLED))


def test_ltx_default_launch_uses_installed_one_stage_when_distilled_missing(tmp_path: Path):
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")

    assert service.default_launch_pipeline() == LTX_PIPELINE_ONE_STAGE

    payload = service._resolve_request(LtxVideoRequest(prompt="simple dance"))

    assert payload["pipeline"] == LTX_PIPELINE_ONE_STAGE
    assert payload["checkpoint_path"] == str(checkpoint.resolve())


def test_ltx_default_launch_prefers_fp8_one_stage_runtime_profile(tmp_path: Path):
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    checkpoint = service.models_root() / "checkpoints" / LTX_FULL_CHECKPOINT_FP8
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")

    assert service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE) == checkpoint
    assert service.default_launch_pipeline() == LTX_PIPELINE_ONE_STAGE

    payload = service._resolve_request(LtxVideoRequest(prompt="simple dance"))

    assert payload["checkpoint_path"] == str(checkpoint.resolve())
    assert payload["offload"] == "none"
    assert payload["quantization"] == "fp8-cast"


def test_ltx_runtime_defaults_discover_assets_in_configured_shared_root(tmp_path: Path):
    shared_root = tmp_path / "shared-models"
    service = LtxService(
        RuntimeFlags(
            data_dir=tmp_path / "app",
            models_dir=tmp_path / "primary-models",
            extra_model_dirs=[shared_root],
            output_dir=tmp_path / "outputs",
        ),
        UserSettings(),
    )
    checkpoint = shared_root / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT_FP8
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    t5 = shared_root / "flux" / "Textencoder" / LTX_T5XXL_FP16
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"encoder")

    assert service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE) == checkpoint.resolve()
    assert service.default_t5_encoder_path() == t5.resolve()


def test_ltx_default_gemma_root_prefers_complete_converted_heretic_q3(tmp_path: Path):
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    official = service.default_official_gemma_root()
    official.mkdir(parents=True)

    assert service.default_gemma_root() == official

    converted = service.default_heretic_converted_gemma_root()
    converted.mkdir(parents=True)
    for filename in (
        "model.safetensors",
        "vision_projector.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.model",
    ):
        (converted / filename).write_bytes(b"fake")

    assert service.default_gemma_root() == converted

    payload = service._resolve_request(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE))

    assert payload["gemma_root"] == str(converted.resolve())


def test_ltx_default_launch_prefers_local_diffusers_2b_when_assets_exist(tmp_path: Path):
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    t5 = service.default_t5_encoder_path()
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake")
    _write_ltx_t5_tokenizer(service)

    assert service.default_launch_pipeline() == LTX_PIPELINE_DIFFUSERS_2B

    payload = service._resolve_request(service.default_launch_request())

    assert payload["pipeline"] == LTX_PIPELINE_DIFFUSERS_2B
    assert payload["checkpoint_path"] == str(checkpoint.resolve())
    assert payload["t5_encoder_path"] == str(t5.resolve())


def test_ltx_default_launch_does_not_treat_empty_diffusers_checkpoint_as_installed(tmp_path: Path):
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    t5 = service.default_t5_encoder_path()
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"encoder")

    assert service.default_launch_pipeline() == LTX_PIPELINE_DISTILLED


def test_ltx_checkpoint_openability_check_skips_tiny_placeholders(tmp_path: Path):
    checkpoint = tmp_path / "tiny.safetensors"
    checkpoint.write_bytes(b"fake")

    assert ltx_checkpoint_openability_error(checkpoint) == ""


def test_ltx_checkpoint_openability_reports_invalid_large_safetensors(tmp_path: Path):
    checkpoint = tmp_path / "large.safetensors"
    checkpoint.write_bytes(b"not-a-safetensors-file" * 65536)

    message = ltx_checkpoint_openability_error(checkpoint)

    assert "LTX checkpoint" in message
    assert str(checkpoint) in message


def test_ltx_diffusers_2b_route_does_not_require_isolated_engine(tmp_path: Path, monkeypatch):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings())
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    t5 = service.default_t5_encoder_path()
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake")
    _write_ltx_t5_tokenizer(service)

    def fake_run_ltx2b_diffusers(**kwargs):
        output = Path(kwargs["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"video")
        return SimpleNamespace(
            output_path=output,
            frame_count=kwargs["frames"],
            fps=kwargs["fps"],
            width=kwargs["width"],
            height=kwargs["height"],
            bytes=output.stat().st_size,
            cache_hit=False,
        )

    monkeypatch.setattr("aiwf.services.ltx_diffusers.run_ltx2b_diffusers", fake_run_ltx2b_diffusers)
    monkeypatch.setattr("aiwf.services.ltx_diffusers.is_ltx2b_pipeline_cached", lambda **_kwargs: True)
    monkeypatch.setattr(service, "_ltx2b_headroom_issue", lambda _path: pytest.fail("cache hit must skip VRAM gate"))

    result = service.generate(
        LtxVideoRequest(
            prompt="simple dance",
            pipeline=LTX_PIPELINE_DIFFUSERS_2B,
            checkpoint_path=str(checkpoint),
            t5_encoder_path=str(t5),
        )
    )

    assert result.output_path.endswith(".mp4")
    assert result.audio_mode == "none"
    assert "LTX 2B Diffusers" in result.message


def test_ltx_diffusers_2b_prepare_loads_and_confirms_residency_without_generation(tmp_path: Path, monkeypatch):
    from aiwf.services import ltx_diffusers

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings())
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake checkpoint")
    t5 = service.default_t5_encoder_path()
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake t5")
    _write_ltx_t5_tokenizer(service)
    calls = []
    monkeypatch.setattr(ltx_diffusers, "load_ltx2b_pipeline", lambda **kwargs: (calls.append(kwargs) or object(), False))
    monkeypatch.setattr(ltx_diffusers, "is_ltx2b_pipeline_cached", lambda **kwargs: True)
    monkeypatch.setattr(service, "_ltx2b_headroom_issue", lambda _path: pytest.fail("cache hit must skip VRAM gate"))

    prepared = service.prepare(LtxVideoRequest(
        prompt="selection warmup",
        pipeline=LTX_PIPELINE_DIFFUSERS_2B,
        checkpoint_path=str(checkpoint),
        t5_encoder_path=str(t5),
    ))

    assert len(calls) == 1
    assert calls[0]["checkpoint"] == checkpoint.resolve()
    assert calls[0]["t5_weights"] == t5.resolve()
    assert prepared["loaded"] is True
    assert prepared["resident"] is True
    assert "No generation was run" in prepared["detail"]


@pytest.mark.parametrize("empty_asset", ["checkpoint", "t5", "tokenizer"])
def test_ltx_diffusers_2b_prepare_rejects_empty_local_assets_before_loading(
    tmp_path: Path,
    monkeypatch,
    empty_asset: str,
):
    from aiwf.services import ltx_diffusers

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings())
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"" if empty_asset == "checkpoint" else b"checkpoint")
    t5 = service.default_t5_encoder_path()
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"" if empty_asset == "t5" else b"t5")
    if empty_asset != "tokenizer":
        _write_ltx_t5_tokenizer(service)
    load_calls = []
    monkeypatch.setattr(ltx_diffusers, "load_ltx2b_pipeline", lambda **kwargs: load_calls.append(kwargs))

    with pytest.raises(LtxUnavailable, match="missing or empty|missing or incomplete"):
        service.prepare(LtxVideoRequest(
            prompt="selection warmup",
            pipeline=LTX_PIPELINE_DIFFUSERS_2B,
            checkpoint_path=str(checkpoint),
            t5_encoder_path=str(t5),
        ))

    assert load_calls == []


def test_ltx_diffusers_2b_prepare_defers_before_uncached_load_with_low_vram(tmp_path: Path, monkeypatch):
    import torch

    from aiwf.services import ltx_diffusers

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings())
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake checkpoint")
    t5 = service.default_t5_encoder_path()
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake t5")
    _write_ltx_t5_tokenizer(service)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args, **_kwargs: (1 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(ltx_diffusers, "is_ltx2b_pipeline_cached", lambda **_kwargs: False)
    load_calls = []
    monkeypatch.setattr(ltx_diffusers, "load_ltx2b_pipeline", lambda **kwargs: load_calls.append(kwargs))

    with pytest.raises(LtxUnavailable, match="1.0 GB VRAM is free"):
        service.prepare(LtxVideoRequest(
            prompt="selection warmup",
            pipeline=LTX_PIPELINE_DIFFUSERS_2B,
            checkpoint_path=str(checkpoint),
            t5_encoder_path=str(t5),
        ))

    assert load_calls == []


def test_ltx_external_worker_defers_before_request_or_launch_when_gpu_headroom_is_low(tmp_path: Path, monkeypatch):
    service = LtxService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"), UserSettings())
    service.registry = SimpleNamespace(
        status=lambda _engine: SimpleNamespace(ready=True, messages=[], repo_dir=tmp_path),
        build_command=lambda *_args, **_kwargs: pytest.fail("LTX worker command constructed"),
    )
    monkeypatch.setattr(service, "_resolve_request", lambda _request: {
        "pipeline": LTX_PIPELINE_ONE_STAGE,
        "checkpoint_path": str(tmp_path / "ltx.safetensors"),
        "output_path": str(tmp_path / "output.mp4"),
    })
    monkeypatch.setattr(service, "_validate_request_paths", lambda _payload: None)
    monkeypatch.setattr(
        service,
        "_ltx_worker_headroom_issue",
        lambda _payload: "LTX worker launch deferred: 2.4 GB VRAM is free; the selected disk route needs at least 6.0 GB headroom.",
    )
    monkeypatch.setattr(service, "_write_worker_request", lambda *_args: pytest.fail("LTX request file created"))
    monkeypatch.setattr(service, "_run_worker", lambda *_args: pytest.fail("LTX worker started"))

    with pytest.raises(LtxUnavailable, match="2.4 GB VRAM is free"):
        service.generate(LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE))


def test_ltx_worker_headroom_uses_nvidia_smi_when_studio_torch_cuda_is_unavailable(tmp_path: Path, monkeypatch):
    import sys

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.nvidia_smi_free_bytes",
        lambda: int(2.4 * 1024**3),
    )
    service = LtxService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"), UserSettings())

    issue = service._ltx_worker_headroom_issue({"offload": "disk"})

    assert issue is not None
    assert "2.4 GB VRAM is free" in issue


def test_ltx_diffusers_2b_generate_defers_before_uncached_load_with_low_vram(tmp_path: Path, monkeypatch):
    import torch

    from aiwf.services import ltx_diffusers

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings())
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake checkpoint")
    t5 = service.default_t5_encoder_path()
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake t5")
    _write_ltx_t5_tokenizer(service)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args, **_kwargs: (1 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(ltx_diffusers, "is_ltx2b_pipeline_cached", lambda **_kwargs: False)
    run_calls = []
    monkeypatch.setattr(ltx_diffusers, "run_ltx2b_diffusers", lambda **kwargs: run_calls.append(kwargs))

    with pytest.raises(LtxUnavailable, match="1.0 GB VRAM is free"):
        service.generate(LtxVideoRequest(
            prompt="generate",
            pipeline=LTX_PIPELINE_DIFFUSERS_2B,
            checkpoint_path=str(checkpoint),
            t5_encoder_path=str(t5),
        ))

    assert run_calls == []


def test_ltx2b_diffusers_pipeline_cache_reuses_matching_assets(tmp_path: Path, monkeypatch):
    import sys
    import types

    from aiwf.services import ltx_diffusers

    ltx_diffusers.unload_ltx2b_diffusers_cache()
    checkpoint = tmp_path / "ltx.safetensors"
    checkpoint.write_bytes(b"fake")
    t5 = tmp_path / "t5.safetensors"
    t5.write_bytes(b"fake")
    load_count = {"count": 0}

    class FakePipe:
        @classmethod
        def from_single_file(cls, *_args, **_kwargs):
            load_count["count"] += 1
            return cls()

        def enable_model_cpu_offload(self):
            return None

        def to(self, *_args, **_kwargs):
            return self

    fake_diffusers = types.SimpleNamespace(LTXPipeline=FakePipe)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object())
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(ltx_diffusers, "load_t5_encoder", lambda _path, **_kwargs: object())

    pipe1, hit1 = ltx_diffusers.load_ltx2b_pipeline(
        checkpoint=checkpoint,
        t5_weights=t5,
        tokenizer_id="local-tokenizer",
    )
    pipe2, hit2 = ltx_diffusers.load_ltx2b_pipeline(
        checkpoint=checkpoint,
        t5_weights=t5,
        tokenizer_id="local-tokenizer",
    )

    assert hit1 is False
    assert hit2 is True
    assert pipe1 is pipe2
    assert load_count["count"] == 1
    assert ltx_diffusers.unload_ltx2b_diffusers_cache() is True
    assert ltx_diffusers.unload_ltx2b_diffusers_cache() is False


def test_ltx_resolves_default_heretic_gguf_path(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings())
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = service.default_gemma_root()
    _write_ltx_gemma_assets(gemma, with_hf_weights=False)
    gguf = flags.resolved_models_dir() / "LLM" / "GGUF" / LTX_HERETIC_Q3_GGUF
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")

    payload = service._resolve_request(
        LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE, gemma_backend=LTX_GEMMA_BACKEND_GGUF)
    )

    assert payload["gemma_backend"] == LTX_GEMMA_BACKEND_GGUF
    assert payload["gemma_gguf_path"] == str(gguf.resolve())
    assert payload["gemma_root"] == str(gemma.resolve())


def test_ltx_resolves_gguf_from_its_catalog_destination(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings())
    gguf = flags.resolved_models_dir() / "ltx" / "GGUF" / LTX_HERETIC_Q3_GGUF
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")

    assert service.default_gemma_gguf_path() == gguf.resolve()


def test_ltx_hf_backend_does_not_switch_when_gguf_textbox_has_value(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings())
    payload = service._resolve_request(
        LtxVideoRequest(
            prompt="simple dance",
            pipeline=LTX_PIPELINE_ONE_STAGE,
            gemma_gguf_path=str(service.default_gemma_gguf_path()),
        )
    )

    assert payload["gemma_backend"] != LTX_GEMMA_BACKEND_GGUF
    assert payload["gemma_gguf_path"] == ""


def test_ltx_generate_blocks_native_gguf_before_worker(tmp_path: Path):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings(), registry=WorkerTenantRegistry(tmp_path))
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    _write_ltx_gemma_assets(service.default_gemma_root(), with_hf_weights=False)
    gguf = service.default_gemma_gguf_path()
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")

    request = LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE, gemma_backend=LTX_GEMMA_BACKEND_GGUF)

    with pytest.raises(LtxUnavailable, match="embedding output plus every layer"):
        service.generate(request)


def test_ltx_probe_runtime_classifies_clean_gguf_blocker():
    error = (
        "Worker exited with code 1.\n\nRecent LTX output:\n"
        "GGUF metadata: architecture=gemma3, hidden_size=3840, layers=48\n"
        "Native Gemma GGUF is wired for selection/probing, but generation is blocked: "
        "LTX needs raw Gemma hidden states and an attention mask."
    )

    assert _is_clean_ltx_gguf_blocker(error)
    assert _extract_recent_output(error).startswith("GGUF metadata: architecture=gemma3")


def test_ltx_gemma_gguf_inventory_helpers_classify_candidates():
    assert _quant_from_filename("gemma-3-12b-it-heretic-Q3_K_M.gguf") == "Q3_K_M"
    assert _quant_from_filename("gemma-3-12b-it-Q4_0.gguf") == "Q4_0"
    assert _quant_from_filename("model.safetensors") == ""
    assert _is_gemma_gguf_candidate(Path("models/LLM/GGUF/gemma-3-12b-it-heretic-Q3_K_M.gguf"))
    assert not _is_gemma_gguf_candidate(Path("models/wan/GGUF/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"))


def test_ltx_gemma_gguf_inventory_marks_smallest_heretic():
    assets = [
        {"filename": "gemma-3-12b-it-heretic-Q4_K_M.gguf", "is_heretic": True, "size_gib": 6.8},
        {"filename": "gemma-3-12b-it-heretic-Q3_K_M.gguf", "is_heretic": True, "size_gib": 5.6},
        {"filename": "gemma-3-12b-it-Q4_0.gguf", "is_heretic": False, "size_gib": 0.25},
    ]

    _mark_smallest_heretic(assets)

    assert assets[0]["smallest_heretic"] is False
    assert assets[1]["smallest_heretic"] is True
    assert assets[2]["smallest_heretic"] is False


def test_ltx_gemma_gguf_converter_transposes_2d_tensors():
    assert _output_shape(SimpleNamespace(shape=[3840, 262208])) == [262208, 3840]
    assert _output_shape(SimpleNamespace(shape=[3840])) == [3840]


def test_ltx_generate_blocks_unloadable_native_checkpoint_before_worker(tmp_path: Path, monkeypatch):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings(), registry=WorkerTenantRegistry(tmp_path))
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    _write_ltx_gemma_assets(service.default_gemma_root())
    monkeypatch.setattr("aiwf.services.ltx.ltx_checkpoint_openability_error", lambda _path: "pagefile too small")
    monkeypatch.setattr(service, "_run_worker", lambda *_args, **_kwargs: pytest.fail("worker should not launch"))

    with pytest.raises(LtxUnavailable, match="pagefile too small"):
        service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE))


def test_ltx_generate_blocks_native_runtime_crash_before_worker(tmp_path: Path, monkeypatch):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings(), registry=WorkerTenantRegistry(tmp_path))
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    _write_ltx_gemma_assets(service.default_gemma_root())
    monkeypatch.setattr("aiwf.services.ltx.ltx_checkpoint_openability_error", lambda _path: "")
    monkeypatch.setattr("aiwf.services.ltx.ltx_native_checkpoint_runtime_blocker", lambda _path: "access violation 3221225477")
    monkeypatch.setattr(service, "_run_worker", lambda *_args, **_kwargs: pytest.fail("worker should not launch"))

    with pytest.raises(LtxUnavailable, match="access violation 3221225477"):
        service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE))


def test_ltx_generate_rejects_empty_hf_gemma_folder_before_worker(tmp_path: Path, monkeypatch):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    service = LtxService(flags, UserSettings(), registry=WorkerTenantRegistry(tmp_path))
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    service.default_gemma_root().mkdir(parents=True)
    monkeypatch.setattr("aiwf.services.ltx.ltx_checkpoint_openability_error", lambda _path: "")
    monkeypatch.setattr("aiwf.services.ltx.ltx_native_checkpoint_runtime_blocker", lambda _path: "")
    monkeypatch.setattr(service, "_run_worker", lambda *_args, **_kwargs: pytest.fail("worker should not launch"))

    with pytest.raises(LtxUnavailable, match="LTX Gemma assets are incomplete"):
        service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE))


def test_ltx_last_error_keeps_detail_and_status_tail():
    events = [
        {"kind": "status", "message": "RuntimeError: Attempted to access the data pointer on an invalid python storage."},
        {"kind": "error", "message": "LTX pipeline exited with code 1", "detail": "Traceback detail"},
    ]

    message = _last_error(events)

    assert "Traceback detail" in message
    assert "invalid python storage" in message


def test_ltx_last_error_ignores_status_without_error_event():
    events = [
        {
            "kind": "status",
            "message": "UserWarning: expandable_segments not supported on this platform",
        }
    ]

    assert _last_error(events) == ""


def test_ltx_service_reports_native_audio_when_output_has_audio(tmp_path: Path, monkeypatch):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    _write_ltx_gemma_assets(gemma)
    service = LtxService(flags, UserSettings(), registry=WorkerTenantRegistry(tmp_path))
    monkeypatch.setattr(service, "_ltx_worker_headroom_issue", lambda _payload: None)

    def fake_run_worker(_job_id, _command, _events, **_kwargs):
        request_files = sorted((flags.resolved_output_dir() / "ltx-videos" / "requests").glob("*.json"))
        payload = json.loads(request_files[-1].read_text(encoding="utf-8"))
        Path(payload["output_path"]).write_bytes(b"video")

    monkeypatch.setattr(service, "_run_worker", fake_run_worker)
    monkeypatch.setattr("aiwf.services.ltx.VideoProcessor.probe", lambda self, path: SimpleNamespace(has_audio=True))

    result = service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE))

    assert result.has_audio is True
    assert result.audio_mode == "native"
    assert "native audio" in result.message


def test_ltx_service_accepts_warning_status_when_output_exists(tmp_path: Path, monkeypatch):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT_FP8
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    _write_ltx_gemma_assets(gemma)
    service = LtxService(flags, UserSettings(), registry=WorkerTenantRegistry(tmp_path))
    monkeypatch.setattr(service, "_ltx_worker_headroom_issue", lambda _payload: None)

    def fake_run_worker(_job_id, _command, events, **_kwargs):
        request_files = sorted((flags.resolved_output_dir() / "ltx-videos" / "requests").glob("*.json"))
        payload = json.loads(request_files[-1].read_text(encoding="utf-8"))
        events.append({"kind": "status", "message": "UserWarning: expandable_segments not supported on this platform"})
        Path(payload["output_path"]).write_bytes(b"video")

    monkeypatch.setattr(service, "_run_worker", fake_run_worker)
    monkeypatch.setattr("aiwf.services.ltx.VideoProcessor.probe", lambda self, path: SimpleNamespace(has_audio=False))

    result = service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE))

    assert result.output_path.endswith(".mp4")
    assert result.audio_mode == "native"


def test_ltx_worker_progress_callback_can_cancel_active_worker(tmp_path: Path):
    import threading

    from aiwf.core.domain.errors import GenerationCancelledError

    stopped = threading.Event()

    class FakeProcessSupervisor:
        def start(self, _worker_id, _command, *, check):
            assert check is True
            yield json.dumps({"kind": "status", "message": "LTX worker started"})
            yield json.dumps({"kind": "status", "message": "next event"})

        def is_running(self, _worker_id):
            return True

        def stop(self, _worker_id):
            stopped.set()
            return "stopped"

    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"),
        UserSettings(),
        process_supervisor=FakeProcessSupervisor(),
    )
    events = []

    def on_progress(event):
        events.append(event)
        service.cancel_active_generation()

    with pytest.raises(GenerationCancelledError, match="cancelled"):
        service._run_worker("ltx_test", ["worker"], [], on_progress=on_progress)

    assert events[0]["message"] == "LTX worker started"
    assert stopped.wait(timeout=1)


def test_ltx2b_diffusers_step_callback_reports_progress_and_cancels(tmp_path: Path, monkeypatch):
    import contextlib
    import sys
    import types

    from aiwf.core.domain.errors import GenerationCancelledError
    from aiwf.services import ltx_diffusers

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    t5 = tmp_path / "t5.safetensors"
    t5.write_bytes(b"t5")
    cancel = {"requested": False}
    reported = []

    class FakePipe:
        def __call__(self, **kwargs):
            callback = kwargs["callback_on_step_end"]
            callback(self, 0, None, {})
            callback(self, 1, None, {})
            return types.SimpleNamespace(frames=[["frame"]])

    monkeypatch.setattr(ltx_diffusers, "load_ltx2b_pipeline", lambda **_kwargs: (FakePipe(), True))
    fake_torch = types.SimpleNamespace(
        Generator=lambda **_kwargs: types.SimpleNamespace(manual_seed=lambda _seed: object()),
        inference_mode=contextlib.nullcontext,
    )
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.__path__ = []
    fake_utils = types.ModuleType("diffusers.utils")
    fake_utils.export_to_video = lambda _frames, path, fps: Path(path).write_bytes(b"video")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", fake_utils)

    def on_progress(event):
        reported.append(event)
        cancel["requested"] = True

    with pytest.raises(GenerationCancelledError, match="cancelled"):
        ltx_diffusers.run_ltx2b_diffusers(
            checkpoint=checkpoint,
            t5_weights=t5,
            tokenizer_id="local-tokenizer",
            output=tmp_path / "out.mp4",
            prompt="motion",
            negative_prompt="",
            width=128,
            height=128,
            frames=9,
            fps=8,
            steps=2,
            seed=1,
            on_progress=on_progress,
            should_cancel=lambda: cancel["requested"],
        )

    assert len(reported) == 1
    assert reported[0]["progress"] == 0.5
