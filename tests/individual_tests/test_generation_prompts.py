from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from PIL import Image

from aiwf import __version__
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult
from aiwf.core.domain.models import Checkpoint, LoraInfo, VaeInfo
from aiwf.core.events.bus import EventBus
from aiwf.services.engine_supervisor import EngineSupervisor
from aiwf.services.queue import JobQueue
from aiwf.services.failure_archive import FailureArchiveService
from aiwf.services.generation import GenerationService
from aiwf.services.gpu_tenant_lock import GpuTenantLock
from aiwf.services.metadata import MetadataService
from aiwf.services.prompt_processor import PromptProcessorService


def test_generation_service_resolves_prompts(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path)
    wildcards = tmp_path / "wildcards"
    wildcards.mkdir()
    (wildcards / "color.txt").write_text("blue\n", encoding="utf-8")

    models = MagicMock()
    models.expand_prompt_keywords.side_effect = lambda text: text
    prompts = PromptProcessorService(flags, UserSettings(), models)

    service = GenerationService(
        backend=MagicMock(),
        store=MagicMock(),
        metadata=MagicMock(),
        queue=MagicMock(),
        events=MagicMock(),
        settings=UserSettings(),
        prompts=prompts,
    )

    request = GenerationRequest(
        mode=GenerationMode.TXT2IMG,
        prompt="sky __color__",
        seed=7,
    )
    resolved = service._resolve_prompts(request)
    assert resolved.prompt == "sky blue"


def test_generation_service_enriches_saved_infotext(tmp_path: Path):
    vae_path = tmp_path / "clear.vae.safetensors"
    lora_path = tmp_path / "detail.safetensors"
    vae_path.write_bytes(b"vae")
    lora_path.write_bytes(b"lora")

    backend = MagicMock()
    backend.list_vaes.return_value = [
        VaeInfo(id="clear", title="Clear VAE", filename=vae_path.name, path=str(vae_path))
    ]
    backend.list_loras.return_value = [
        LoraInfo(id="detail", title="Detail", filename=lora_path.name, path=str(lora_path))
    ]

    service = GenerationService(
        backend=backend,
        store=MagicMock(),
        metadata=MetadataService(),
        queue=MagicMock(),
        events=MagicMock(),
        settings=UserSettings(),
    )
    request = GenerationRequest(prompt="<lora:detail:0.7> portrait", vae_id="clear")
    checkpoint = Checkpoint(
        id="model",
        title="Model",
        filename="model.safetensors",
        path=str(tmp_path / "model.safetensors"),
        hash="abc123",
    )

    enriched = service._enrich_saved_infotext("portrait\nSteps: 20, Model: Model", request, checkpoint)

    assert "Model hash: abc123" in enriched
    assert "VAE: Clear VAE" in enriched
    assert "VAE hash:" in enriched
    assert "Lora hashes: detail:" in enriched
    assert "AIWF Studio:" in enriched


def test_streaming_generation_reports_model_loading_before_backend_steps():
    checkpoint = Checkpoint(
        id="tiny",
        title="Tiny Model",
        filename="tiny.safetensors",
        path="/models/tiny.safetensors",
        hash="abc123",
    )

    backend = MagicMock()
    backend.resolve_checkpoint.return_value = checkpoint

    def generate(request, *, on_progress=None, **_kwargs):
        if on_progress:
            on_progress(1, request.steps, "Step 1/1", None)
        return GenerationResult(
            job_id=uuid4(),
            images=[Image.new("RGB", (8, 8))],
            seeds=[1],
            infotexts=[""],
            mode=request.mode,
        )

    backend.generate.side_effect = generate
    events = EventBus()
    service = GenerationService(
        backend=backend,
        store=MagicMock(),
        metadata=MagicMock(),
        queue=JobQueue(events),
        events=events,
        settings=UserSettings(save_images=False),
    )

    request = GenerationRequest(prompt="cat", steps=1)
    output = list(service.submit_streaming(request))
    progress = [item for item in output if item[0] == "progress"]

    assert progress[0][1:4] == (0, 1, "Loading image model: Tiny Model")
    assert progress[1][3] == "Step 1/1"


def test_generation_postprocess_reuses_image_tenant_for_nested_enhance(monkeypatch):
    checkpoint = Checkpoint(
        id="tiny",
        title="Tiny Model",
        filename="tiny.safetensors",
        path="/models/tiny.safetensors",
        hash="abc123",
    )
    image = Image.new("RGB", (8, 8))

    backend = MagicMock()
    backend.resolve_checkpoint.return_value = checkpoint
    backend.generate.return_value = GenerationResult(
        job_id=uuid4(),
        images=[image],
        seeds=[1],
        infotexts=[""],
        mode=GenerationMode.TXT2IMG,
    )
    events = EventBus()
    supervisor = EngineSupervisor(gpu_lock=GpuTenantLock())
    monkeypatch.setattr(supervisor, "_flush_cuda", lambda: None)
    service = GenerationService(
        backend=backend,
        store=MagicMock(),
        metadata=MagicMock(),
        queue=JobQueue(events),
        events=events,
        settings=UserSettings(save_images=False),
        supervisor=supervisor,
    )

    def postprocess(img):
        with supervisor.tenant_session(EngineTenant.ENHANCE, reason="postprocess") as owner:
            assert owner
            assert supervisor.active_tenant == EngineTenant.IMAGE
        return img

    job = service.submit(GenerationRequest(prompt="cat", steps=1), image_postprocess=postprocess)

    assert job.result is not None
    assert job.result.images == [image]
    assert supervisor.active_tenant == EngineTenant.IDLE


def test_image_generation_keeps_backend_loaded_after_success():
    checkpoint = Checkpoint(
        id="tiny",
        title="Tiny Model",
        filename="tiny.safetensors",
        path="/models/tiny.safetensors",
        hash="abc123",
    )

    backend = MagicMock()
    backend.resolve_checkpoint.return_value = checkpoint
    backend.generate.return_value = GenerationResult(
        job_id=uuid4(),
        images=[Image.new("RGB", (8, 8))],
        seeds=[1],
        infotexts=[""],
        mode=GenerationMode.TXT2IMG,
    )
    events = EventBus()
    service = GenerationService(
        backend=backend,
        store=MagicMock(),
        metadata=MagicMock(),
        queue=JobQueue(events),
        events=events,
        settings=UserSettings(save_images=False),
    )

    job = service.submit(GenerationRequest(prompt="cat", steps=1))

    assert job.result is not None
    backend.unload.assert_not_called()


def test_image_generation_archives_backend_failure(tmp_path: Path):
    checkpoint = Checkpoint(
        id="tiny",
        title="Tiny Model",
        filename="tiny.safetensors",
        path="/models/tiny.safetensors",
        hash="abc123",
    )
    preview = Image.new("RGB", (8, 8), "purple")

    backend = MagicMock()
    backend.resolve_checkpoint.return_value = checkpoint

    def fail_generate(request, *, on_progress=None, **_kwargs):
        if on_progress:
            on_progress(1, request.steps, "Step 1/1", preview)
        raise RuntimeError("bad latent soup")

    backend.generate.side_effect = fail_generate
    events = EventBus()
    archive = FailureArchiveService(tmp_path / "outputs")
    service = GenerationService(
        backend=backend,
        store=MagicMock(),
        metadata=MagicMock(),
        queue=JobQueue(events),
        events=events,
        settings=UserSettings(save_images=False),
        failure_archive=archive,
    )

    with pytest.raises(RuntimeError, match="bad latent soup"):
        service.submit(GenerationRequest(prompt="dance", steps=1))

    index_path = tmp_path / "outputs" / "failures" / "index.jsonl"
    entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["kind"] == "image"
    assert entries[0]["stage"] == "image-generation"
    assert entries[0]["request"]["prompt"] == "dance"
    assert entries[0]["error"]["message"] == "bad latent soup"
    assert entries[0]["extra"]["checkpoint_id"] == "tiny"
    assert list((tmp_path / "outputs" / "failures").rglob("preview.png"))


def test_generation_service_records_model_throughput(monkeypatch):
    checkpoint = Checkpoint(
        id="tiny",
        title="Tiny Model",
        filename="tiny.safetensors",
        path="/models/tiny.safetensors",
        hash="abc123",
    )

    backend = MagicMock()
    backend.resolve_checkpoint.return_value = checkpoint

    def generate(request, *, on_progress=None, **_kwargs):
        if on_progress:
            on_progress(1, request.steps, "Step 1/1", None)
        return GenerationResult(
            job_id=uuid4(),
            images=[Image.new("RGB", (8, 8))],
            seeds=[1],
            infotexts=[""],
            mode=request.mode,
        )

    backend.generate.side_effect = generate
    events = EventBus()
    service = GenerationService(
        backend=backend,
        store=MagicMock(),
        metadata=MagicMock(),
        queue=JobQueue(events),
        events=events,
        settings=UserSettings(save_images=False),
    )

    captured = {}
    monkeypatch.setattr("aiwf.services.generation.trace_model_throughput", lambda **kwargs: captured.update(kwargs))

    request = GenerationRequest(prompt="cat", steps=4)
    list(service.submit_streaming(request))

    assert captured["kind"] == "txt2img"
    assert captured["model_id"] == "tiny"
    assert captured["units_label"] == "steps"
    assert captured["units"] == 4
    assert captured["app_version"] == __version__
