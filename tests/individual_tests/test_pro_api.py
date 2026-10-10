from __future__ import annotations

import base64
import io
import logging
import json
import os
import subprocess
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

import aiwf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AIWF_ROOT = str(_REPO_ROOT / "aiwf")
if _AIWF_ROOT not in aiwf.__path__:
    aiwf.__path__.insert(0, _AIWF_ROOT)

from aiwf.app_pro import create_app
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.errors import ModelNotFoundError
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobRecord, JobState, SavedArtifact
from aiwf.core.domain.ltx import LtxVideoResult
from aiwf.core.domain.models import Checkpoint, SamplerInfo
from aiwf.core.domain.sana_video import SanaVideoProgressEvent, SanaVideoRequest, SanaVideoResult
from aiwf.services import audio_licenses
from aiwf.services.model_download import ModelDownloadService
from aiwf.services.pipeline_readiness import PipelineReadinessRecord
from aiwf.web import pro_api


def test_runtime_loaded_model_prefers_poll_friendly_backend_status_check(monkeypatch):
    calls: list[str] = []

    class Backend:
        _active = object()
        _txt2img = None
        _inpaint = None

        def is_checkpoint_loaded_for_status(self, checkpoint_id=None):
            calls.append(f"status:{checkpoint_id}")
            return True

        def is_checkpoint_loaded(self, checkpoint_id=None):
            raise AssertionError("runtime polling should use the poll-friendly status check")

    monkeypatch.setattr(pro_api, "_dump_model", lambda _active: {"id": "model-a", "title": "Model A", "path": "model.safetensors"})
    ctx = SimpleNamespace(generation=SimpleNamespace(backend=Backend()), flags=None)

    status = pro_api._runtime_loaded_model(ctx)

    assert status["loaded"] is True
    assert calls == ["status:model-a"]


class _Devices:
    def describe(self):
        return "CPU (test)"


class _Generation:
    backend = SimpleNamespace(devices=_Devices())

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.submitted = []
        self.submitted_kwargs = []
        self._recent = []
        self._active = None

    def list_checkpoints(self):
        model_path = self.output_dir.parent / "models" / "model-a.safetensors"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"fake checkpoint fixture")
        return [
            Checkpoint(
                id="model-a",
                title="Model A",
                filename="model-a.safetensors",
                path=str(self.output_dir.parent / "models" / "model-a.safetensors"),
                architecture="sdxl",
            )
        ]

    def list_samplers(self):
        return [SamplerInfo(id="euler_a", label="Euler a")]

    def get_model_preset(self, checkpoint_id=None):
        return {
            "steps": 28,
            "cfg_scale": 6.0,
            "sampler": "dpmpp_2m",
            "scheduler": "automatic",
            "width": 1024,
            "height": 1024,
        }

    def list_loras(self):
        return [SimpleNamespace(id="style-a", title="Style A")]

    def active_job(self):
        return self._active

    def pending_count(self):
        return 0

    def recent_jobs(self, _limit=20):
        return list(self._recent)

    def submit(self, request: GenerationRequest, **_kwargs):
        self.submitted.append(request)
        self.submitted_kwargs.append(_kwargs)
        image = Image.new("RGB", (16, 16), "blue")
        artifact_path = self.output_dir / "txt2img-images" / "generated.png"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(artifact_path)
        job = JobRecord(
            request=request,
            state=JobState.COMPLETED,
            result=GenerationResult(
                job_id=uuid4(),
                images=[image],
                seeds=[1234],
                infotexts=["ok"],
                artifacts=[SavedArtifact(path=str(artifact_path), infotext="ok")],
                mode=GenerationMode.TXT2IMG,
            ),
        )
        self._recent.insert(0, job)
        return job


class _SanaVideo:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.submitted = []
        self.prepared = []
        self.unloaded = False
        self._prepared_pipeline = None
        self._prepared_pipeline_key = None
        self.confirm_unload = True

    def default_model_path(self, variant="480p"):
        return self.output_dir.parent / "models" / "sana-video" / "Diffusers" / f"SANA-Video_2B_{variant}_diffusers"

    def runtime_available(self, *, image_to_video=False):
        return True

    def prepare(self, request):
        self.prepared.append(request)
        self._prepared_pipeline = object()
        self._prepared_pipeline_key = (request.model_variant, request.model_path)
        return {
            "loaded": True,
            "modelPath": str(request.model_path or self.default_model_path(request.model_variant)),
            "quantization": "bf16",
            "attentionBackend": "native",
        }

    def unload(self):
        self.unloaded = True
        if self.confirm_unload:
            self._prepared_pipeline = None
            self._prepared_pipeline_key = None
        return self.confirm_unload

    def release_cached_model_for_modality_switch(self):
        return self.unload()

    def generate(self, request, *, on_progress=None):
        self.submitted.append(request)
        if request.generate_audio:
            self.unload()
        if on_progress is not None:
            on_progress("load", 0.1, "Loading Sana Video pipeline", 0, 0, 0.01)
            on_progress("done", 1.0, "Sana video saved", 0, 0, 0.2)
        path = self.output_dir / "sana-videos" / "generated.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        receipt = self.output_dir.parent / "_local" / "logs" / "sana_video_latest.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text("{}", encoding="utf-8")
        return SanaVideoResult(
            output_path=str(path),
            message="Sana video saved",
            frames=request.frames,
            fps=request.fps,
            width=request.width,
            height=request.height,
            timings={"load": 0.01, "inference": 0.1, "decode": 0.09},
            progress=[
                SanaVideoProgressEvent(stage="load", progress=0.1, message="Loading Sana Video pipeline").model_dump(),
                SanaVideoProgressEvent(stage="done", progress=1.0, message="Sana video saved", seconds=0.2).model_dump(),
            ],
            attention_backend="native",
            quantization=request.quantization,
            vae_tiling=request.vae_tiling,
            receipt_path=str(receipt),
        )


class _FailingSanaVideo(_SanaVideo):
    def generate(self, request, *, on_progress=None):
        self.submitted.append(request)
        if on_progress is not None:
            on_progress("decode", 0.9, "Decoding latents", 0, 0, 0.2)
        receipt = self.output_dir.parent / "_local" / "logs" / "sana_video_latest.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps(
                {
                    "created_at": "2026-06-30T10:00:00+00:00",
                    "status": "error",
                    "error": {"type": "RuntimeError", "message": "decode failed"},
                    "progress": [{"stage": "decode", "message": "Decoding latents"}],
                }
            ),
            encoding="utf-8",
        )
        raise RuntimeError("decode failed")


class _LtxVideo:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.submitted = []
        self.progress_events = []

    def generate(self, request, *, on_progress=None):
        self.submitted.append(request)
        if on_progress is not None:
            event = {"kind": "progress", "step": 1, "total": 2, "progress": 0.5, "message": "LTX denoising step 1 of 2."}
            self.progress_events.append(event)
            on_progress(event)
        path = self.output_dir / "ltx-videos" / "generated.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ltx video")
        return LtxVideoResult(
            output_path=str(path),
            message="LTX video saved",
            events=[{"kind": "complete", "message": "LTX video complete"}],
            has_audio=True,
            audio_mode="native",
        )


def _seed_sana_video_snapshot(model_path: Path) -> None:
    (model_path / "model_index.json").parent.mkdir(parents=True, exist_ok=True)
    (model_path / "model_index.json").write_text(
        json.dumps({
            "_class_name": "SanaVideoPipeline",
            "scheduler": ["diffusers", "DPMSolverMultistepScheduler"],
            "text_encoder": ["transformers", "Gemma2Model"],
            "tokenizer": ["transformers", "GemmaTokenizerFast"],
            "transformer": ["diffusers", "SanaVideoTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        }),
        encoding="utf-8",
    )
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        (model_path / component).mkdir(parents=True, exist_ok=True)
    (model_path / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (model_path / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_path / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"transformer")
    (model_path / "vae" / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    (model_path / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"fake.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    (model_path / "text_encoder" / "model-00001-of-00001.safetensors").write_bytes(b"encoder")


class _Enhance:
    def __init__(self, root: Path):
        self.calls = []
        self.model_paths = {
            "upscale-a": root / "models" / "upscale-a.pth",
            "restore-a": root / "models" / "restore-a.pth",
        }
        self.catalog_invalidations = 0
        for path in self.model_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"installed fixture")

    def list_upscalers(self):
        return [SimpleNamespace(id="upscale-a", path=str(self.model_paths["upscale-a"]))]

    def list_restorers(self):
        return [SimpleNamespace(id="restore-a", path=str(self.model_paths["restore-a"]))]

    def list_model_status(self):
        return [
            {"id": "upscale-a", "title": "Upscale A", "filename": "upscale-a.pth", "kind": "upscaler", "architecture": "fixture", "scale": 2, "installed": self.model_paths["upscale-a"].is_file(), "installAvailable": True},
            {"id": "restore-a", "title": "Restore A", "filename": "restore-a.pth", "kind": "restorer", "architecture": "fixture", "scale": 1, "installed": self.model_paths["restore-a"].is_file(), "installAvailable": True},
        ]

    def invalidate_model_catalog(self):
        self.catalog_invalidations += 1

    def prepare_model(self, model_id):
        path = self.model_paths[model_id]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"installed fixture")
        return path

    def run_pipeline(self, image, *, restore=None, upscale=None, restore_first=True):
        self.calls.append({"restore": restore, "upscale": upscale, "restore_first": restore_first})
        scale = int(getattr(upscale, "scale", 1) or 1) if upscale is not None else 1
        result = image.resize((image.width * scale, image.height * scale))
        steps = []
        if restore is not None:
            steps.append(f"Restore: {restore.model_id}")
        if upscale is not None:
            steps.append(f"Upscale: {upscale.model_id} ({upscale.scale:g}x)")
        return result, " | ".join(steps)


class _Vsr:
    def __init__(self):
        self.calls = []

    def install_info(self):
        return SimpleNamespace(
            available=True,
            upscale_available=True,
            denoise_available=False,
            aigs_available=False,
            relight_available=False,
            sdk_root=Path("C:/VideoFX"),
            model_count=1,
            features=["SuperRes"],
            feature_names=["SuperRes"],
        )

    def folder_help(self):
        return ""

    def upscale_image(self, image, options):
        self.calls.append(options)
        scale = int(getattr(options, "scale", 1) or 1)
        return image.resize((image.width * scale, image.height * scale))


def _ctx(tmp_path: Path):
    output_dir = tmp_path / "outputs"
    flags = RuntimeFlags(data_dir=tmp_path, output_dir=output_dir)
    # research mode keeps the existing MusicGen/MMAudio route tests meaningful; those models are
    # CC-BY-NC 4.0 and hidden by default (the default is tested in test_audio_licenses.py)
    settings = UserSettings(default_sampler="euler_a", default_width=640, default_height=768,
                            allow_noncommercial_audio_models=True)
    controlnet = SimpleNamespace(
        list_models=lambda: [SimpleNamespace(id="control-a")],
        list_modules=lambda: ["none", "canny"],
    )
    segment = SimpleNamespace(list_models=lambda: [SimpleNamespace(id="sam-b")])
    enhance = _Enhance(tmp_path)
    faceswap = SimpleNamespace(
        list_models=lambda: [SimpleNamespace(id="inswapper")],
        list_face_models=lambda: [SimpleNamespace(id="saved-face")],
    )
    wan = SimpleNamespace(
        list_local_models=lambda: ["wan-a.safetensors"],
        list_local_models_labeled=lambda: [("Wan 2.2 Fast 5B", "wan-a.safetensors")],
        list_local_loras=lambda: ["wan-lora.safetensors"],
        available=lambda: True,
    )
    return SimpleNamespace(
        flags=flags,
        settings=settings,
        generation=_Generation(output_dir),
        model_download=ModelDownloadService(flags),
        controlnet=controlnet,
        segment=segment,
        enhance=enhance,
        faceswap=faceswap,
        wan=wan,
        sana_video=_SanaVideo(output_dir),
        ltx=_LtxVideo(output_dir),
        vsr=_Vsr(),
        runtime_port=9876,
    )


def _client(ctx, frontend_dist: Path | None = None):
    return TestClient(
        create_app(ctx, frontend_dist=frontend_dist or Path("__missing_frontend_dist__")),
        client=("127.0.0.1", 51000),
    )


def test_runtime_endpoint_reports_warm_latency_under_budget(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    client.get("/api/pro/runtime")

    started = time.perf_counter()
    response = client.get("/api/pro/runtime")
    roundtrip_ms = (time.perf_counter() - started) * 1000

    assert response.status_code == 200
    server_ms = float(response.headers["X-AIWF-Elapsed-Ms"])
    assert response.headers["Server-Timing"].startswith("aiwf;dur=")
    assert server_ms < 75
    assert roundtrip_ms < 75


def test_runtime_endpoint_reports_gerror_flag(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.flags = ctx.flags.model_copy(update={"gerror": True})
    client = _client(ctx)

    response = client.get("/api/pro/runtime")

    assert response.status_code == 200
    assert response.json()["gerror"] is True


def test_startup_endpoint_tracks_window_ready_callback(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    initial = client.get("/api/pro/startup")

    assert initial.status_code == 200
    assert initial.json()["serverReady"] is True
    assert initial.json()["windowReady"] is False
    assert initial.json()["minSplashMs"] >= 1000

    ready = client.post("/api/pro/startup/window-ready")

    assert ready.status_code == 200
    data = ready.json()
    assert data["status"] == "window-ready"
    assert data["windowReady"] is True
    assert data["windowReadyAt"]


def test_app_startup_dispatches_saved_model_load_in_background(tmp_path, monkeypatch):
    from aiwf.services import model_startup

    ctx = _ctx(tmp_path)
    ctx.model_download.recover_interrupted_snapshot_replacements = lambda: None
    preload_started = threading.Event()
    monkeypatch.setattr(model_startup, "pro_startup_model_loading_enabled", lambda: True)

    def preload(_ctx):
        preload_started.set()
        return {"status": "loaded", "modelId": "saved-image", "detail": "backend confirmed resident"}

    monkeypatch.setattr(model_startup, "preload_pro_image_model", preload)
    client = _client(ctx)

    with client:
        assert client.get("/api/pro/startup").status_code == 200
        assert preload_started.wait(timeout=2)
        startup = client.get("/api/pro/startup").json()

    assert startup["modelLoad"] == {
        "status": "loaded", "modelId": "saved-image", "detail": "backend confirmed resident",
    }


def test_bootstrap_returns_catalog_defaults_runtime_and_recent_images(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.generation.submit(GenerationRequest(prompt="seed recent"))
    client = _client(ctx)

    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert data["runtime"]["device"] == "CPU (test)"
    assert data["runtime"]["gerror"] is False
    assert data["settings"]["width"] == 640
    assert data["checkpoints"][0]["id"] == "model-a"
    assert data["checkpoints"][0]["engineId"] == "sdxl"
    assert data["checkpoints"][0]["engineLabel"] == "Stable Diffusion XL"
    assert data["checkpoints"][0]["checkpointPathStatus"] == "present"
    assert data["checkpoints"][0]["routeStatus"] == "request-eligible"
    assert data["checkpoints"][0]["generationPreset"]["sampler"] == "dpmpp_2m"
    assert data["checkpoints"][0]["generationPreset"]["width"] == 1024
    assert {"sdxl", "wan"}.issubset({item["id"] for item in data["engines"]})
    assert "sana_video" not in {item["id"] for item in data["engines"]}, "blocked-only routes must not create a filter with no selectable rows"
    sana_blocked = next(item for item in data["blockedCheckpoints"] if item["engineId"] == "sana_video")
    assert sana_blocked["status"] == "missing-assets"
    assert sana_blocked["setupBundleKey"] == "sana-video"
    assert any(item["engineId"] == "wan" and item["kind"] == "video" for item in data["checkpoints"])
    assert "path" not in data["checkpoints"][0]
    assert data["samplers"][0]["supportsKarras"] is False
    assert data["recentImages"][0]["dataUrl"].startswith("data:image/png;base64,")


def test_settings_defaults_choose_ready_image_before_video_inventory_entries(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    video = {
        "id": "sana-video-720p",
        "engineId": "sana_video",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
    }
    image = {
        "id": "image-model",
        "engineId": "sdxl",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
    }
    monkeypatch.setattr(pro_api, "_selectable_checkpoint_payloads", lambda _ctx: ([video, image], []))

    defaults = pro_api._settings_defaults(ctx)

    assert defaults["checkpointId"] == "image-model"


def test_settings_defaults_replace_missing_saved_checkpoint_with_ready_local_image(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.settings.last_checkpoint_id = "removed-model-id"
    fallback = {
        "id": "installed-flux2-klein",
        "engineId": "flux2",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
    }
    monkeypatch.setattr(pro_api, "_selectable_checkpoint_payloads", lambda _ctx: ([fallback], []))
    monkeypatch.setattr(pro_api, "_is_checkpoint_id_selectable", lambda _ctx, _model_id: False)
    ctx.generation.list_checkpoints = lambda: [SimpleNamespace(id="another-installed-model")]

    defaults = pro_api._settings_defaults(ctx)

    assert defaults["checkpointId"] == "installed-flux2-klein"
    assert ctx.settings.last_checkpoint_id == "removed-model-id"


def test_settings_defaults_skip_video_and_unready_image_routes(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    video = {
        "id": "wan-model",
        "engineId": "wan",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
    }
    unready_image = {
        "id": "image-missing-support",
        "engineId": "flux",
        "routeStatus": "unknown",
        "checkpointPathStatus": "present",
    }
    unknown_path_image = {
        "id": "image-unknown-path",
        "engineId": "sd15",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "unknown",
    }
    unknown_family = {
        "id": "unknown-family",
        "engineId": "unknown",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
    }
    ready_image = {
        "id": "ready-image-model",
        "engineId": "sdxl",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
    }
    monkeypatch.setattr(
        pro_api,
        "_selectable_checkpoint_payloads",
        lambda _ctx: ([video, unready_image, unknown_path_image, unknown_family, ready_image], []),
    )

    assert pro_api._settings_defaults(ctx)["checkpointId"] == "ready-image-model"

    monkeypatch.setattr(pro_api, "_selectable_checkpoint_payloads", lambda _ctx: ([video], []))
    assert pro_api._settings_defaults(ctx)["checkpointId"] is None


def test_settings_defaults_preserve_an_explicitly_saved_video_selection(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.settings.last_checkpoint_id = "saved-video"
    video = {
        "id": "saved-video",
        "engineId": "sana_video",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
    }
    image = {
        "id": "image-model",
        "engineId": "sdxl",
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
    }
    monkeypatch.setattr(pro_api, "_selectable_checkpoint_payloads", lambda _ctx: ([video, image], []))
    monkeypatch.setattr(pro_api, "_is_checkpoint_id_selectable", lambda _ctx, _id: True)

    assert pro_api._settings_defaults(ctx)["checkpointId"] == "saved-video"


def test_bootstrap_only_marks_wan_ready_after_route_preflight(tmp_path):
    ctx = _ctx(tmp_path)
    calls = []
    result = SimpleNamespace(ok=False, errors=("Missing Wan VAE",))

    def preflight(request, *, image_present=True):
        calls.append((request, image_present))
        return result

    ctx.wan.preflight = preflight
    first = _client(ctx).get("/api/pro/bootstrap").json()
    wan_model = next(item for item in first["blockedCheckpoints"] if item["engineId"] == "wan")
    assert wan_model["status"] == "missing-assets"
    assert wan_model["routeStatus"] == "blocked"
    assert "Missing Wan VAE" in wan_model["readinessReason"]
    assert calls[0][0].model_id == "wan-a.safetensors"

    result.ok = True
    result.errors = ()
    second = _client(ctx).get("/api/pro/bootstrap").json()
    wan_model = next(item for item in second["checkpoints"] if item["engineId"] == "wan")
    assert wan_model["status"] == "Ready"
    assert wan_model["routeStatus"] == "request-eligible"


def test_bootstrap_does_not_claim_wan_ready_without_preflight_capability(tmp_path):
    ctx = _ctx(tmp_path)

    data = _client(ctx).get("/api/pro/bootstrap").json()

    wan_model = next(item for item in data["checkpoints"] if item["engineId"] == "wan")
    assert wan_model["status"] == "Installed"
    assert wan_model["routeStatus"] == "unknown"


def test_wan_payloads_expose_fp8_setup_only_for_a_configured_complete_pair(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    high = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    low = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    ctx.settings.last_wan_runtime_mode = "native_high_low_fp8_experimental"
    ctx.settings.last_wan_high = high
    ctx.settings.last_wan_low = low

    class FakeWanService:
        def list_local_models_labeled(self):
            return [("High noise", high), ("Low noise", low)]

    service = FakeWanService()
    monkeypatch.setattr(pro_api, "_wan_service", lambda _ctx: service)
    monkeypatch.setattr(pro_api, "_wan_model_readiness", lambda *_args: {"status": "Ready", "routeStatus": "request-eligible"})

    payloads = pro_api._wan_model_payloads(ctx)
    assert {item["id"] for item in payloads if item.get("setupBundleKey") == "wan-14b-components"} == {high, low}
    assert all(item["setupRoute"]["routeKey"] == "pro.video.wan.fp8-pair" for item in payloads)

    monkeypatch.setattr(service, "list_local_models_labeled", lambda: [("High noise", high)])
    partial = pro_api._wan_model_payloads(ctx)
    assert all("setupRoute" not in item for item in partial)
    assert all("setupBundleKey" not in item for item in partial)


def test_wan_payloads_expose_gguf_setup_only_for_configured_complete_pair(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    high = "wan2.2_i2v_high_noise_14B_Q5_K_M.gguf"
    low = "wan2.2_i2v_low_noise_14B_Q5_K_M.gguf"
    ctx.settings.last_wan_runtime_mode = "native_high_low"
    ctx.settings.last_wan_high = high
    ctx.settings.last_wan_low = low

    class FakeWanService:
        def list_local_models_labeled(self):
            return [("High noise", high), ("Low noise", low)]

    service = FakeWanService()
    monkeypatch.setattr(pro_api, "_wan_service", lambda _ctx: service)
    monkeypatch.setattr(pro_api, "_wan_model_readiness", lambda *_args: {"status": "Ready", "routeStatus": "request-eligible"})

    payloads = pro_api._wan_model_payloads(ctx)
    assert {item["id"] for item in payloads if item.get("setupBundleKey") == "wan-14b-components"} == {high, low}
    assert all(item["setupRoute"]["routeKey"] == "pro.video.wan.gguf-pair" for item in payloads)

    monkeypatch.setattr(service, "list_local_models_labeled", lambda: [("High noise", high)])
    partial = pro_api._wan_model_payloads(ctx)
    assert all("setupRoute" not in item for item in partial)


def test_wan_payloads_expose_ti2v_setup_only_for_a_complete_diffusers_folder(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    root = tmp_path / "wan" / "Diffusers"
    model = root / "wan-ti2v-5b"
    (model / "transformer").mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    identifier = "wan-ti2v-5b"

    class FakeWanService:
        def list_local_models_labeled(self):
            return [("Wan TI2V 5B Diffusers", identifier)]

        def _wan_diffusers_roots(self):
            return [root]

        def _is_full_fast_5b_diffusers_model(self, path):
            return path == model

    service = FakeWanService()
    monkeypatch.setattr(pro_api, "_wan_service", lambda _ctx: service)
    monkeypatch.setattr(pro_api, "_wan_model_readiness", lambda *_args: {"status": "Ready", "routeStatus": "request-eligible"})

    payload = pro_api._wan_model_payloads(ctx)[0]
    assert payload["setupBundleKey"] == "wan-ti2v-diffusers"
    assert payload["setupRoute"]["routeKey"] == "pro.video.wan.ti2v-diffusers"


@pytest.mark.parametrize("architecture", ["flux", "flux_fill"])
def test_bootstrap_blocks_flux_when_local_conditioning_assets_are_missing(tmp_path, architecture):
    from aiwf.core.domain.errors import ModelNotFoundError

    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "flux-dev.safetensors"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"Flux transformer fixture")
    checkpoint = Checkpoint(
        id="flux-dev",
        title="Flux Dev",
        filename=model_path.name,
        path=str(model_path),
        architecture=architecture,
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]

    def missing_conditioning():
        raise ModelNotFoundError(
            "Flux prompt conditioning needs a complete local CLIP-L tokenizer snapshot. "
            "Searched: test model root"
        )

    ctx.generation.backend = SimpleNamespace(
        _validate_flux_prompt_assets=missing_conditioning,
        _flux_prompt_conditioning=SimpleNamespace(mode="teacher"),
    )
    response = _client(ctx).get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert all(item["id"] != checkpoint.id for item in data["checkpoints"])
    blocked = next(item for item in data["blockedCheckpoints"] if item["id"] == checkpoint.id)
    assert blocked["status"] == "missing-assets"
    assert blocked["setupBundleKey"] == "flux-components"
    assert "CLIP-L" in blocked["reason"]
    assert "Model Setup" in blocked["suggestedAction"]

    generate = _client(ctx).post(
        "/api/pro/generate",
        json={"prompt": "test", "mode": "image", "model_id": checkpoint.id},
    )
    assert generate.status_code == 422
    assert generate.json()["detail"]["status"] == "missing-assets"
    assert "CLIP-L tokenizer" in generate.json()["detail"]["reason"]
    assert ctx.generation.submitted == []


@pytest.mark.parametrize(
    ("architecture", "family", "pipeline_class"),
    [
        ("sd15", "SD 1.5", "StableDiffusionPipeline"),
        ("sdxl", "SDXL", "StableDiffusionXLPipeline"),
        ("sdxl_inpaint", "SDXL", "StableDiffusionXLInpaintPipeline"),
        ("sd35", "SD 3.5", "StableDiffusion3Pipeline"),
    ],
)
def test_incomplete_standard_diffusers_folders_are_blocked_from_picker(
    tmp_path, architecture, family, pipeline_class
):
    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / architecture
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text(
        json.dumps({"_class_name": pipeline_class}), encoding="utf-8"
    )
    checkpoint = Checkpoint(
        id=f"{architecture}-folder",
        title=f"{family} folder",
        filename=model_path.name,
        path=str(model_path),
        architecture=architecture,
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]

    selectable, blocked = pro_api._selectable_checkpoint_payloads(ctx)

    assert selectable == []
    assert len(blocked) == 1
    assert blocked[0]["status"] == "missing-assets"
    assert "required component configs or weights" in blocked[0]["reason"]


@pytest.mark.parametrize(
    ("architecture", "pipeline_class", "components"),
    [
        ("sd15", "StableDiffusionPipeline", ("unet", "vae", "text_encoder", "scheduler", "tokenizer")),
        ("sdxl", "StableDiffusionXLPipeline", ("unet", "vae", "text_encoder", "text_encoder_2", "scheduler", "tokenizer", "tokenizer_2")),
        ("sdxl_inpaint", "StableDiffusionXLInpaintPipeline", ("unet", "vae", "text_encoder", "text_encoder_2", "scheduler", "tokenizer", "tokenizer_2")),
        ("sd35", "StableDiffusion3Pipeline", ("transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3", "scheduler", "tokenizer", "tokenizer_2", "tokenizer_3")),
    ],
)
def test_complete_standard_diffusers_folders_remain_selectable(
    tmp_path, monkeypatch, architecture, pipeline_class, components
):
    ctx = _ctx(tmp_path)
    model_folder = f"{architecture}-large" if architecture == "sd35" else architecture
    model_path = tmp_path / "models" / model_folder
    model_path.mkdir(parents=True)
    model_index = {"_class_name": pipeline_class}
    for component in components:
        model_index[component] = ["diffusers", "DummyComponent"]
        component_path = model_path / component
        component_path.mkdir()
        if component == "scheduler":
            (component_path / "scheduler_config.json").write_text("{}", encoding="utf-8")
        elif component.startswith("tokenizer"):
            (component_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        else:
            (component_path / "config.json").write_text("{}", encoding="utf-8")
            (component_path / "model.safetensors").write_bytes(b"fixture")
    (model_path / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    checkpoint = Checkpoint(
        id=f"{model_folder}-folder",
        title=architecture.upper(),
        filename=model_path.name,
        path=str(model_path),
        architecture=architecture,
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]
    if architecture == "sd35":
        monkeypatch.setattr(pro_api, "_sd35_large_access_available", lambda: False)

    selectable, blocked = pro_api._selectable_checkpoint_payloads(ctx)

    assert blocked == []
    assert [item["id"] for item in selectable] == [checkpoint.id]
    assert selectable[0]["routeStatus"] == "request-eligible"


def test_bootstrap_blocks_flux2_model_when_matching_components_are_missing(tmp_path):
    from aiwf.core.domain.errors import ModelNotFoundError

    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "flux2-klein-4b.safetensors"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"Flux.2 Klein transformer fixture")
    checkpoint = Checkpoint(
        id="flux2-klein-4b",
        title="Flux.2 Klein 4B",
        filename=model_path.name,
        path=str(model_path),
        architecture="flux2_klein",
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]

    def missing_components(_architecture, _checkpoint):
        raise ModelNotFoundError("Flux.2 Klein needs the matching Diffusers component folder.")

    ctx.generation.backend = SimpleNamespace(_resolve_component_dir=missing_components)
    response = _client(ctx).get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert all(item["id"] != checkpoint.id for item in data["checkpoints"])
    blocked = next(item for item in data["blockedCheckpoints"] if item["id"] == checkpoint.id)
    assert blocked["status"] == "missing-assets"
    assert "component folder" in blocked["reason"]


def test_full_flux2_diffusers_readiness_requires_transformer_weights(tmp_path):
    model_path = tmp_path / "models" / "flux2" / "Diffusers" / "FLUX.2-klein-4B"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"}), encoding="utf-8"
    )
    checkpoint = Checkpoint(
        id="flux2-full-incomplete",
        title="Flux.2 Klein 4B pipeline",
        filename=model_path.name,
        path=str(model_path),
        architecture="flux2_klein",
    )
    ctx = _ctx(tmp_path)
    # The generic component probe must not be allowed to claim a full pipeline
    # is ready when its transformer files are absent.
    ctx.generation.backend = SimpleNamespace(_looks_like_diffusers_component_dir=lambda _path: True)

    block = pro_api._runtime_checkpoint_block(ctx, checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "pipeline is incomplete" in block["reason"]


def test_bootstrap_blocks_flux_kontext_gguf_without_verified_transformer_header(tmp_path):
    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "flux-kontext.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"GGUF")
    ctx.generation.backend = SimpleNamespace(_is_flux_kontext_gguf_transformer=lambda _path: False)
    checkpoint = Checkpoint(
        id="flux-kontext-file",
        title="Flux Kontext",
        filename=model_path.name,
        path=str(model_path),
        architecture="flux_kontext",
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]

    response = _client(ctx).get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    blocked = next(item for item in data["blockedCheckpoints"] if item["id"] == checkpoint.id)
    assert blocked["status"] == "blocked-cleanly"
    assert "header-verified" in blocked["reason"]


def test_sana_video_picker_uses_selected_snapshot_preflight_and_refreshes_readiness(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight
    from aiwf.services.model_download_catalog import QUICK_START_BUNDLES

    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "sana-video" / "selected-pipeline"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text("{}", encoding="utf-8")
    checkpoint = Checkpoint(
        id="sana-video-selected",
        title="Sana Video selected pipeline",
        filename=model_path.name,
        path=str(model_path),
        architecture="sana_video",
    )
    model_path_720p = tmp_path / "models" / "sana-video" / "SANA-Video_2B_720p_diffusers"
    model_path_720p.mkdir(parents=True)
    (model_path_720p / "model_index.json").write_text("{}", encoding="utf-8")
    checkpoint_720p = Checkpoint(
        id="sana-video-720p",
        title="Sana Video 720p",
        filename="SANA-Video_2B_720p_diffusers",
        path=str(model_path_720p),
        architecture="sana_video",
    )
    assert pro_api._checkpoint_payload(ctx, checkpoint)["setupBundleKey"] == "sana-video"
    assert pro_api._checkpoint_payload(ctx, checkpoint_720p)["setupBundleKey"] == "sana-video-720p"
    assert pro_api._checkpoint_payload(ctx, checkpoint)["setupRoute"]["routeKey"] == "pro.video.sana-video.480p"
    assert pro_api._checkpoint_payload(ctx, checkpoint_720p)["setupRoute"]["routeKey"] == "pro.video.sana-video.720p"
    assert "sana-video-720p" in QUICK_START_BUNDLES
    ctx.generation.list_checkpoints = lambda: [checkpoint, checkpoint_720p]
    observed_paths = []
    readiness = {"complete": False}

    def preflight(_flags, _settings, *, request):
        selected_path = request.model_path in {str(model_path), str(model_path_720p)}
        if selected_path:
            observed_paths.append(request.model_path)
        installed = selected_path and readiness["complete"]
        return SimpleNamespace(
            ok=installed,
            metadata={"model_installed": str(installed).lower()},
            warnings=() if installed else ("selected Sana Video snapshot is incomplete",),
            message=lambda: "Sana Video runtime is unavailable.",
        )

    monkeypatch.setattr(pipeline_preflight, "preflight_sana_video_pipeline", preflight)

    response = _client(ctx).get("/api/pro/bootstrap")

    assert response.status_code == 200
    payload = response.json()
    assert all(item["id"] != checkpoint.id for item in payload["checkpoints"])
    blocked = next(item for item in payload["blockedCheckpoints"] if item["id"] == checkpoint.id)
    assert blocked["status"] == "missing-assets"
    assert blocked["setupBundleKey"] == "sana-video"
    assert "snapshot is incomplete" in blocked["reason"]
    blocked_generation = _client(ctx).post(
        "/api/pro/generate",
        json={"prompt": "slow camera move", "mode": "video", "model_id": checkpoint.id},
    )
    assert blocked_generation.status_code == 422
    assert ctx.sana_video.submitted == []

    # Once the route preflight sees the complete selected snapshot, refresh moves
    # it from blocked to selectable without changing its model identity.
    readiness["complete"] = True
    refreshed = _client(ctx).get("/api/pro/bootstrap").json()
    selectable = next(item for item in refreshed["checkpoints"] if item["id"] == checkpoint.id)
    assert selectable["routeStatus"] == "request-eligible"
    assert checkpoint.id not in {item["id"] for item in refreshed["blockedCheckpoints"]}
    generated = _client(ctx).post(
        "/api/pro/generate",
        json={"prompt": "slow camera move", "mode": "video", "model_id": checkpoint.id},
    )
    assert generated.status_code == 200
    assert len(ctx.sana_video.submitted) == 1
    assert ctx.sana_video.submitted[0].model_path == str(model_path)
    assert ctx.sana_video.submitted[0].model_variant == "480p"
    assert len(observed_paths) >= 2
    assert str(model_path) in observed_paths

    generated_720p = _client(ctx).post(
        "/api/pro/generate",
        json={"prompt": "slow camera move", "mode": "video", "model_id": checkpoint_720p.id},
    )
    assert generated_720p.status_code == 200
    assert len(ctx.sana_video.submitted) == 2
    assert ctx.sana_video.submitted[1].model_path == str(model_path_720p)
    assert ctx.sana_video.submitted[1].model_variant == "720p"
    assert str(model_path_720p) in observed_paths

    selected_720_request = pro_api._sana_video_request_from_payload(
        ctx,
        pro_api.ProGeneratePayload(
            prompt="720p selection",
            mode="video",
            checkpoint_id=checkpoint_720p.id,
            checkpoint_title=checkpoint_720p.title,
        ),
    )
    assert selected_720_request.model_variant == "720p"
    assert selected_720_request.model_path == str(model_path_720p)


def test_sana_video_generation_uses_saved_video_audio_model(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.settings.last_video_audio_model_id = "mmaudio:large_44k_v2"

    request = pro_api._sana_video_request_from_payload(
        ctx,
        pro_api.ProGeneratePayload(
            prompt="Sana with selected soundtrack",
            mode="video",
            generate_audio=True,
        ),
    )

    assert request.generate_audio is True
    assert request.audio_model_id == "mmaudio:large_44k_v2"


def test_bootstrap_uses_specific_flux2_and_krea2_labels_while_blocking_split_krea(tmp_path):
    ctx = _ctx(tmp_path)
    models = tmp_path / "models"
    flux2_path = models / "flux2-klein.safetensors"
    krea_path = models / "krea2" / "UNet" / "krea2_turbo_fp8_scaled.safetensors"
    flux2_path.parent.mkdir(parents=True, exist_ok=True)
    krea_path.parent.mkdir(parents=True)
    flux2_path.write_bytes(b"test Flux.2 checkpoint")
    krea_path.write_bytes(b"test Krea 2 split transformer")
    flux2 = Checkpoint(
        id="flux2-klein",
        title="Flux.2 Klein",
        filename=flux2_path.name,
        path=str(flux2_path),
        architecture="flux2_klein",
    )
    krea = Checkpoint(
        id="krea2-split",
        title="Krea 2 Turbo FP8",
        filename=krea_path.name,
        path=str(krea_path),
        architecture="krea2",
    )
    ctx.generation.list_checkpoints = lambda: [flux2, krea]

    response = _client(ctx).get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    flux2_payload = next(item for item in data["checkpoints"] if item["id"] == flux2.id)
    krea_payload = next(item for item in data["blockedCheckpoints"] if item["id"] == krea.id)
    assert flux2_payload["engineId"] == "flux2"
    assert flux2_payload["engineLabel"] == "Flux.2 Klein"
    assert flux2_payload["setupBundleKey"] is None
    assert krea_payload["engineId"] == "krea2"
    assert krea_payload["engineLabel"] == "Krea 2"
    assert krea_payload["status"] == "blocked-cleanly"
    assert "complete Diffusers pipeline folder" in krea_payload["reason"]
    assert next(item for item in data["engines"] if item["id"] == "flux2")["label"] == "Flux.2 Klein"


def test_generic_flux2_checkpoint_does_not_claim_klein_label(tmp_path):
    ctx = _ctx(tmp_path)
    checkpoint = Checkpoint(
        id="flux2-generic",
        title="Flux.2 model",
        filename="flux2-model.safetensors",
        path=str(tmp_path / "flux2-model.safetensors"),
        # A generic model name must remain generic even if inventory contains
        # a stale family classification from the broad Flux.2 detector.
        architecture="flux2_klein",
    )

    payload = pro_api._checkpoint_payload(ctx, checkpoint)
    assert payload["engineLabel"] == "Flux.2"
    assert payload["architecture"] == "flux2"
    assert payload["engineId"] == "unknown"
    assert next(item for item in pro_api._engine_summaries([payload]) if item["id"] == "flux2_generic")["label"] == "Flux.2 (generic)"


def test_generic_flux2_architecture_keeps_specific_label_while_unsupported(tmp_path):
    ctx = _ctx(tmp_path)
    checkpoint = Checkpoint(
        id="flux2-generic-real-architecture",
        title="Flux.2 base model",
        filename="flux2-base.safetensors",
        path=str(tmp_path / "flux2-base.safetensors"),
        architecture="flux2",
    )

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["engineLabel"] == "Flux.2"
    assert payload["engineId"] == "unknown"
    assert payload["setupBundleKey"] is None


def test_recognized_architecture_wins_over_incidental_flux_filename_tokens(tmp_path):
    ctx = _ctx(tmp_path)
    checkpoint = Checkpoint(
        id="community-flux2-derived-sdxl",
        title="Community model",
        filename="community-flux2-kontext-derived-sdxl.safetensors",
        path=str(tmp_path / "community-flux2-kontext-derived-sdxl.safetensors"),
        architecture="sdxl",
    )

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["architecture"] == "sdxl"
    assert payload["engineId"] == "sdxl"
    assert payload["engineLabel"] == "Stable Diffusion XL"


def test_sd_single_file_checkpoint_requires_local_config_and_exposes_install_bundle(tmp_path):
    ctx = _ctx(tmp_path)
    path = tmp_path / "models" / "model.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    checkpoint = Checkpoint(
        id="sd15-local",
        title="SD 1.5 local checkpoint",
        filename=path.name,
        path=str(path),
        architecture="sd15",
    )
    ctx.generation.backend = SimpleNamespace(can_preload_checkpoint_locally=lambda _checkpoint_id=None: False)

    block = pro_api._runtime_checkpoint_block(ctx, checkpoint)
    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "local Diffusers config/tokenizer" in block["reason"]
    assert payload["setupBundleKey"] == "sd-components"
    assert "Install the SD 1.5 support config" in block["suggestedAction"]

    ctx.generation.backend.can_preload_checkpoint_locally = lambda _checkpoint_id=None: True
    assert pro_api._runtime_checkpoint_block(ctx, checkpoint) is None


def test_flux_kontext_filename_recovers_family_from_broad_cached_flux_architecture(tmp_path):
    ctx = _ctx(tmp_path)
    model_path = tmp_path / "flux1-kontext-dev-Q5_K_M.gguf"
    model_path.write_bytes(b"fixture")
    checkpoint = Checkpoint(
        id="flux1-kontext-dev-Q5_K_M",
        title="flux1-kontext-dev-Q5_K_M [Flux]",
        filename=model_path.name,
        path=str(model_path),
        architecture="flux",
    )
    ctx.generation.backend = SimpleNamespace(_is_flux_kontext_gguf_transformer=lambda _path: False)

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["architecture"] == "flux_kontext"
    assert payload["engineId"] == "flux"
    assert payload["engineLabel"] == "Flux Kontext"
    assert payload["setupRoute"]["routeKey"] == "pro.image.flux-kontext"
    block = pro_api._runtime_checkpoint_block(ctx, checkpoint)
    assert block["status"] == "blocked-cleanly"
    assert "header-verified" in block["reason"]


def test_flux_kontext_gguf_readiness_requires_local_companion_snapshot(tmp_path):
    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "flux1-kontext-dev-Q5_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"nonempty fixture")
    checkpoint = Checkpoint(
        id="flux1-kontext-dev-Q5_K_M",
        title="Flux Kontext GGUF",
        filename=model_path.name,
        path=str(model_path),
        architecture="flux_kontext",
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]
    component_dir = tmp_path / "models" / "flux" / "Components" / "flux-kontext-4bit-fp4"
    calls = []
    ctx.generation.backend = SimpleNamespace(
        _is_flux_kontext_gguf_transformer=lambda path: path == model_path,
        _resolve_component_dir=lambda arch, item: calls.append((arch, item)) or component_dir,
    )

    assert pro_api._runtime_checkpoint_block(ctx, checkpoint) is None
    calls.clear()
    selectable, blocked = pro_api._selectable_checkpoint_payloads(ctx)
    assert not blocked
    assert [item["id"] for item in selectable] == [checkpoint.id]
    assert selectable[0]["routeStatus"] == "request-eligible"
    assert calls == [("flux_kontext", checkpoint)]

    def incomplete(_arch, _item):
        raise ModelNotFoundError("Flux Kontext companion snapshot is incomplete")

    ctx.generation.backend._resolve_component_dir = incomplete
    block = pro_api._runtime_checkpoint_block(ctx, checkpoint)
    assert block is not None
    assert block["status"] == "missing-assets"
    assert "companion snapshot is incomplete" in block["reason"]
    selectable, blocked = pro_api._selectable_checkpoint_payloads(ctx)
    assert selectable == []
    assert len(blocked) == 1
    assert blocked[0]["status"] == "missing-assets"


def test_flux2_kontext_filename_recovers_specific_family_before_flux2(tmp_path):
    ctx = _ctx(tmp_path)
    model_path = tmp_path / "flux2-kontext-dev-Q5_K_M.gguf"
    model_path.write_bytes(b"fixture")
    checkpoint = Checkpoint(
        id="flux2-kontext-dev-Q5_K_M",
        title="flux2-kontext-dev-Q5_K_M [Flux.2]",
        filename=model_path.name,
        path=str(model_path),
        architecture="flux2",
    )

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["architecture"] == "flux_kontext"
    assert payload["engineLabel"] == "Flux Kontext"
    assert payload["setupRoute"]["routeKey"] == "pro.image.flux-kontext"


@pytest.mark.parametrize("folder", ["Lora", "Loras"])
def test_lora_folder_model_is_blocked_from_generate_picker(tmp_path, folder):
    ctx = _ctx(tmp_path)
    lora_path = tmp_path / "models" / folder / "Flux2" / "Realism_Engine_Klein_V2.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fixture")
    lora = Checkpoint(
        id="Realism_Engine_Klein_V2",
        title="Realism Engine Klein V2",
        filename=lora_path.name,
        path=str(lora_path),
        architecture="flux",
    )
    ctx.generation.list_checkpoints = lambda: [lora]

    selectable, blocked = pro_api._selectable_checkpoint_payloads(ctx)

    assert selectable == []
    assert len(blocked) == 1
    assert blocked[0]["routeStatus"] == "blocked"
    assert "Auxiliary model asset" in blocked[0]["reason"]


def test_flux_kontext_readiness_rejects_empty_component_directories(tmp_path):
    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "FLUX.1-Kontext-dev"
    for component in ("scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "transformer", "vae"):
        (model_path / component).mkdir(parents=True)
    (model_path / "model_index.json").write_text('{"_class_name":"FluxKontextPipeline"}', encoding="utf-8")
    checkpoint = Checkpoint(
        id="flux-kontext-incomplete",
        title="Flux Kontext",
        filename=model_path.name,
        path=str(model_path),
        architecture="flux_kontext",
    )

    block = pro_api._runtime_checkpoint_block(ctx, checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "missing local files" in block["reason"]


def test_generic_flux_diffusers_folder_is_blocked_with_supported_route_guidance(tmp_path):
    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "flux-pipeline"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text('{"_class_name":"FluxPipeline"}', encoding="utf-8")
    checkpoint = Checkpoint(
        id="flux-pipeline",
        title="FluxPipeline",
        filename=model_path.name,
        path=str(model_path),
        architecture="flux",
    )

    block = pro_api._runtime_checkpoint_block(ctx, checkpoint)

    assert block is not None
    assert block["status"] == "blocked-cleanly"
    assert "generic Flux Diffusers folder" in block["reason"]
    assert "Flux .gguf or .safetensors" in block["suggestedAction"]


def test_flux2_setup_bundle_follows_selected_9b_variant(tmp_path):
    ctx = _ctx(tmp_path)
    path = tmp_path / "models" / "Flux2Klein9B-Q4_K_M.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Flux.2 Klein 9B fixture")
    checkpoint = Checkpoint(
        id="flux2-9b",
        title="Flux.2 Klein 9B",
        filename=path.name,
        path=str(path),
        architecture="flux2_klein",
    )

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["setupBundleKey"] == "flux2-9b-components"


@pytest.mark.parametrize(
    ("mode", "route_key", "bundle_key"),
    [
        ("teacher", "pro.image.flux-teacher", "flux-components"),
        ("distillt5-control", "pro.image.flux-distillt5-control", "flux-distillt5-control-components"),
    ],
)
def test_flux_setup_route_tracks_active_conditioning_mode(tmp_path, mode, route_key, bundle_key):
    ctx = _ctx(tmp_path)
    ctx.generation.backend = SimpleNamespace(
        devices=_Devices(),
        _flux_prompt_conditioning=SimpleNamespace(mode=mode),
    )
    path = tmp_path / "models" / "flux-dev.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    checkpoint = Checkpoint(
        id="flux-dev",
        title="Flux dev",
        filename=path.name,
        path=str(path),
        architecture="flux",
    )

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["setupRoute"]["routeKey"] == route_key
    assert payload["setupBundleKey"] == bundle_key


def test_flux_setup_route_does_not_guess_when_active_conditioning_mode_is_unavailable(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.generation.backend = SimpleNamespace(devices=_Devices())
    path = tmp_path / "models" / "flux-dev.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    checkpoint = Checkpoint(
        id="flux-dev",
        title="Flux dev",
        filename=path.name,
        path=str(path),
        architecture="flux",
    )

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["setupRoute"] is None
    assert payload["setupBundleKey"] is None


@pytest.mark.parametrize(
    ("available", "expected_state"),
    [(True, "supported-when-folder-installed"), (False, "blocked-runtime")],
)
def test_qwen_21_setup_route_reports_installed_runtime_capability(tmp_path, monkeypatch, available, expected_state):
    ctx = _ctx(tmp_path)
    path = tmp_path / "models" / "Qwen-Image-2.1"
    path.mkdir(parents=True)
    (path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImage21Pipeline"}), encoding="utf-8"
    )
    checkpoint = Checkpoint(
        id="qwen-image-21",
        title="Qwen Image 2.1",
        filename=path.name,
        path=str(path),
        architecture="qwen_image",
    )
    monkeypatch.setattr(pro_api, "_qwen_image_21_pipeline_available", lambda: available)

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["setupRoute"]["supportState"] == expected_state


def test_ltx_route_does_not_recommend_the_ltx23_bundle_for_unwired_pro_picker(tmp_path):
    ctx = _ctx(tmp_path)
    path = tmp_path / "models" / "ltx-2b-diffusers"
    path.mkdir(parents=True)
    checkpoint = Checkpoint(
        id="ltx-2b",
        title="LTX 2B Diffusers",
        filename=path.name,
        path=str(path),
        architecture="ltx",
    )

    payload = pro_api._checkpoint_payload(ctx, checkpoint)

    assert payload["setupBundleKey"] is None


def _ready_ltx_preflight(_flags, _settings=None, *, request):
    pipeline = request.pipeline
    return SimpleNamespace(
        ok=True,
        items=(),
        warnings=(),
        metadata={"checkpoint_path": f"C:/models/{pipeline}.safetensors"},
        message=lambda: "ready",
    )


def test_ltx_distilled_selection_keeps_its_own_missing_checkpoint_path(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight
    from aiwf.services.ltx import LtxService

    ctx = _ctx(tmp_path)
    ctx.ltx = LtxService(ctx.flags, ctx.settings)
    one_stage = ctx.ltx.default_checkpoint_path("one_stage")
    one_stage.parent.mkdir(parents=True, exist_ok=True)
    one_stage.write_bytes(b"installed one-stage placeholder")

    def preflight(_flags, _settings, *, request):
        assert request.pipeline == "distilled"
        assert Path(request.checkpoint_path).name == "ltx-2.3-22b-distilled-1.1.safetensors"
        return SimpleNamespace(
            ok=False,
            items=(SimpleNamespace(name="checkpoint", ok=False, message="missing LTX checkpoint"),),
            warnings=(),
            metadata={"checkpoint_path": request.checkpoint_path},
            message=lambda: "missing LTX checkpoint",
        )

    monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", preflight)
    model = pro_api._ltx_model_payload(ctx, "ltx:distilled")

    assert model["pipeline"] == "distilled"
    assert model["status"] == "missing-assets"
    assert Path(model["path"]).name == "ltx-2.3-22b-distilled-1.1.safetensors"


def test_ltx_one_stage_fp8_selection_recommends_matching_setup_bundle(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight
    from aiwf.services.ltx import LTX_FULL_CHECKPOINT_FP8, LtxService

    ctx = _ctx(tmp_path)
    ctx.ltx = LtxService(ctx.flags, ctx.settings)
    fp8_checkpoint = ctx.ltx.models_root() / "checkpoints" / LTX_FULL_CHECKPOINT_FP8
    fp8_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fp8_checkpoint.write_bytes(b"FP8 LTX fixture")

    def missing_support(_flags, _settings, *, request):
        return SimpleNamespace(
            ok=False,
            items=(SimpleNamespace(name="spatial upscaler", ok=False, message="missing LTX spatial upscaler"),),
            warnings=(),
            metadata={"checkpoint_path": request.checkpoint_path},
            message=lambda: "missing LTX spatial upscaler",
        )

    monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", missing_support)
    payload = pro_api._ltx_model_payload(ctx, "ltx:one_stage")

    assert payload["filename"] == LTX_FULL_CHECKPOINT_FP8
    assert payload["setupBundleKey"] == "ltx23-one-stage-fp8"
    assert payload["setupRoute"]["setupBundleKey"] == "ltx23-one-stage-fp8"


def test_ltx_synthetic_bootstrap_options_report_preflight_and_setup_bundle(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    def preflight(_flags, _settings, *, request):
        result = _ready_ltx_preflight(_flags, _settings, request=request)
        if request.pipeline == "distilled":
            result.ok = False
            result.items = (SimpleNamespace(name="spatial upscaler", ok=False, message="missing LTX spatial upscaler"),)
        return result

    monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", preflight)
    response = _client(_ctx(tmp_path)).get("/api/pro/bootstrap")

    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()["checkpoints"] + response.json()["blockedCheckpoints"]}
    assert models["ltx:diffusers_2b"]["status"] == "Ready"
    assert models["ltx:diffusers_2b"]["setupBundleKey"] is None
    assert models["ltx:diffusers_2b"]["setupRoute"]["routeKey"] == "pro.video.ltx.diffusers-2b"
    assert models["ltx:distilled"]["status"] == "missing-assets", models["ltx:distilled"].get("reason")
    assert models["ltx:distilled"]["setupBundleKey"] == "ltx23"
    assert models["ltx:distilled"]["setupRoute"]["routeKey"] == "pro.video.ltx.distilled"
    assert models["ltx:one_stage"]["status"] == "blocked-runtime"
    assert models["ltx:one_stage"]["routeStatus"] == "blocked"
    assert models["ltx:one_stage"]["setupBundleKey"] is None
    assert "Installing the BF16 one-stage bundle will not remove" in models["ltx:one_stage"]["suggestedAction"]
    assert models["ltx:one_stage"]["setupRoute"]["routeKey"] == "pro.video.ltx.one-stage"


def test_ltx_2b_model_payload_blocks_when_runtime_preflight_fails(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    def missing_runtime(_flags, _settings, *, request):
        assert request.pipeline == "diffusers_2b"
        return SimpleNamespace(
            ok=False,
            items=(SimpleNamespace(
                name="Diffusers LTX 2B runtime",
                ok=False,
                message="The installed diffusers runtime does not expose LTXPipeline.from_single_file.",
            ),),
            warnings=(),
            metadata={"checkpoint_path": request.checkpoint_path},
            message=lambda: "runtime capability missing",
        )

    monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", missing_runtime)
    payload = pro_api._ltx_model_payload(_ctx(tmp_path), "ltx:diffusers_2b")

    assert payload["status"] == "blocked-runtime"
    assert payload["routeStatus"] == "blocked"
    assert "LTXPipeline.from_single_file" in payload["readinessReason"]


def test_generate_maps_ltx_request_and_serves_output(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight
    from aiwf.services.ltx import LTX_FULL_CHECKPOINT_FP8

    monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", _ready_ltx_preflight)
    ctx = _ctx(tmp_path)
    ctx.ltx.default_checkpoint_path = lambda _pipeline: tmp_path / LTX_FULL_CHECKPOINT_FP8
    released_image_models = []
    pro_api.select_route(ctx, route="image.txt2img", model_id="model-a", setup_ready=True, resident=True)
    monkeypatch.setattr(pro_api, "_runtime_loaded_model", lambda _ctx: {"loaded": True, "id": "model-a"})
    monkeypatch.setattr(pro_api, "_unload_generation_model", lambda _ctx: released_image_models.append("model-a"))
    client = _client(ctx)
    source = Image.new("RGB", (16, 16), "red")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "slow camera move",
            "mode": "video",
            "model_id": "ltx:one_stage",
            "steps": 2,
            "width": 640,
            "height": 512,
            "frames": 17,
            "fps": 12,
            "seed": 42,
            "source_image_data_url": data_url,
            "ltx_image_strength": 0.6,
            "ltx_offload": "cpu",
            "ltx_quantization": "fp8-scaled-mm",
            "ltx_enhance_prompt": True,
        },
    )

    assert response.status_code == 200, response.text
    assert released_image_models == ["model-a"]
    image_state = next(
        item for item in client.get("/api/pro/runtime").json()["routeLifecycle"]
        if item["route"] == "image.txt2img"
    )
    assert image_state["resident"] is False
    request = ctx.ltx.submitted[0]
    assert request.pipeline == "one_stage"
    assert request.prompt == "slow camera move"
    assert request.width == 640 and request.height == 512
    assert request.num_frames == 17 and request.fps == 12
    assert request.seed == 42 and request.steps == 2
    assert request.image_strength == 0.6
    assert request.offload == "cpu" and request.quantization == "fp8-scaled-mm"
    assert request.enhance_prompt is True
    assert request.gemma_backend == "hf_safetensors"
    assert request.source_image_path and Path(request.source_image_path).is_file()

    data = response.json()
    assert data["output"]["mode"] == "video"
    assert data["output"]["source"] == "ltx"
    assert data["output"]["url"].startswith("/api/pro/outputs/ltx-videos/generated.mp4")
    assert data["hasAudio"] is True
    assert ctx.ltx.progress_events[0]["progress"] == 0.5
    assert client.get(data["output"]["url"]).content == b"ltx video"


def test_generate_ltx_diffusers_2b_text_to_video(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", _ready_ltx_preflight)
    ctx = _ctx(tmp_path)
    response = _client(ctx).post(
        "/api/pro/generate",
        json={"prompt": "simple motion", "mode": "video", "model_id": "ltx:diffusers_2b"},
    )

    assert response.status_code == 200, response.text
    request = ctx.ltx.submitted[0]
    assert request.pipeline == "diffusers_2b"
    assert request.source_image_path is None
    assert response.json()["output"]["source"] == "ltx"


def test_ltx_generation_fails_closed_when_support_preflight_fails(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from aiwf.services import pipeline_preflight
    from aiwf.services.pipeline_preflight import PipelineCheckItem, PipelinePreflightResult

    ctx = _ctx(tmp_path)
    missing_support = tmp_path / "models" / "ltx" / "gemma" / "model.safetensors"
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_ltx_pipeline",
        lambda *_args, **_kwargs: PipelinePreflightResult(
            pipeline="LTX Diffusers 2B",
            ok=False,
            items=(PipelineCheckItem("Gemma support", False, "Missing Gemma weights", missing_support),),
        ),
    )

    with pytest.raises(HTTPException) as error:
        pro_api._generate_ltx_video_response(
            ctx,
            pro_api.ProGeneratePayload(prompt="test", mode="video", checkpoint_id="ltx:diffusers_2b"),
        )

    assert error.value.status_code == 422
    assert "Missing Gemma weights" in str(error.value.detail)
    assert ctx.ltx.submitted == []
    lifecycle = _client(ctx).get("/api/pro/runtime").json()["routeLifecycle"]
    route = next(item for item in lifecycle if item["route"] == "video.ltx.diffusers_2b")
    assert route["status"] == "needs-setup"
    assert route["modelId"] == "ltx:diffusers_2b"
    assert len(route["supportRevision"]) == 16


def test_ltx_route_rejects_image_mode_and_video_mode_rejects_image_model(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", _ready_ltx_preflight)
    client = _client(_ctx(tmp_path))
    image_mode = client.post(
        "/api/pro/generate",
        json={"prompt": "bad mode", "mode": "image", "model_id": "ltx:one_stage"},
    )
    assert image_mode.status_code == 422
    assert "LTX models" in image_mode.json()["detail"]["message"]

    video_image_model = client.post(
        "/api/pro/generate",
        json={"prompt": "wrong family", "mode": "video", "model_id": "model-a"},
    )
    assert video_image_model.status_code == 422
    assert "image model" in video_image_model.json()["detail"]["message"]


def test_qwen_image_21_is_blocked_when_diffusers_runtime_lacks_its_pipeline(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "qwen-image" / "Qwen-Image-2.1"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text('{"_class_name":"QwenImage21Pipeline"}', encoding="utf-8")
    checkpoint = Checkpoint(
        id="qwen-image-21",
        title="Qwen Image 2.1",
        filename=model_path.name,
        path=str(model_path),
        architecture="qwen_image",
    )
    monkeypatch.setattr(pro_api, "_qwen_image_21_pipeline_available", lambda: False)

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "blocked-runtime"
    assert "QwenImage21Pipeline" in block["reason"]


def test_z_image_single_file_is_blocked_when_component_assets_are_missing(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "models" / "z-image" / "z-image-turbo.safetensors"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fixture")
    checkpoint = Checkpoint(
        id="z-image-turbo",
        title="Z-Image Turbo",
        filename=checkpoint_path.name,
        path=str(checkpoint_path),
        architecture="z_image",
    )
    backend = SimpleNamespace(
        devices=_Devices(),
        _resolve_component_dir=lambda *_args: (_ for _ in ()).throw(
            pro_api.ModelNotFoundError("Z-Image needs text_encoder, tokenizer, scheduler and VAE components")
        ),
    )
    ctx = _ctx(tmp_path)
    ctx.generation.backend = backend
    monkeypatch.setattr(pro_api, "_z_image_runtime_block", lambda: None)

    block = pro_api._runtime_checkpoint_block(ctx, checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "Z-Image needs" in block["reason"]
    assert "Z-Image Turbo components" in block["suggestedAction"]


def test_z_image_diffusers_folder_is_checked_as_the_selected_snapshot(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "z-image" / "Z-Image-Turbo"
    _write_complete_z_image_folder(model_path)
    checkpoint = Checkpoint(
        id="z-image-folder",
        title="Z-Image Turbo",
        filename=model_path.name,
        path=str(model_path),
        architecture="z_image",
    )
    monkeypatch.setattr(pro_api, "_z_image_runtime_block", lambda: None)
    monkeypatch.setattr(
        "aiwf.infrastructure.diffusers.checkpoints.missing_diffusers_local_files",
        lambda _path, limit=8: [model_path / "transformer" / "missing.safetensors"],
    )

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "missing.safetensors" in block["reason"]


def test_z_image_folder_with_only_model_index_is_missing_required_components(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "z-image" / "Z-Image-Turbo"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text('{"_class_name":"ZImagePipeline"}', encoding="utf-8")
    checkpoint = Checkpoint(
        id="z-image-folder",
        title="Z-Image Turbo",
        filename=model_path.name,
        path=str(model_path),
        architecture="z_image",
    )
    monkeypatch.setattr(pro_api, "_z_image_runtime_block", lambda: None)

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "transformer" in block["reason"]
    assert "text_encoder" in block["reason"] and "model.safetensors" in block["reason"]


def test_complete_z_image_diffusers_folder_passes_local_component_check(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "z-image" / "Z-Image-Turbo"
    _write_complete_z_image_folder(model_path)
    checkpoint = Checkpoint(
        id="z-image-folder",
        title="Z-Image Turbo",
        filename=model_path.name,
        path=str(model_path),
        architecture="z_image",
    )
    monkeypatch.setattr(pro_api, "_z_image_runtime_block", lambda: None)

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is None


def test_z_image_folder_rejects_unrecognized_component_weight_names(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "z-image" / "Z-Image-Turbo"
    _write_complete_z_image_folder(model_path)
    (model_path / "transformer" / "diffusion_pytorch_model.safetensors").unlink()
    (model_path / "transformer" / "junk.safetensors").write_bytes(b"fixture")
    checkpoint = Checkpoint(
        id="z-image-folder",
        title="Z-Image Turbo",
        filename=model_path.name,
        path=str(model_path),
        architecture="z_image",
    )
    monkeypatch.setattr(pro_api, "_z_image_runtime_block", lambda: None)

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "diffusion_pytorch_model.safetensors" in block["reason"]


def test_z_image_folder_rejects_malformed_safe_tensor_weights(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "z-image" / "Z-Image-Turbo"
    _write_complete_z_image_folder(model_path)
    (model_path / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    checkpoint = Checkpoint(
        id="z-image-folder",
        title="Z-Image Turbo",
        filename=model_path.name,
        path=str(model_path),
        architecture="z_image",
    )
    monkeypatch.setattr(pro_api, "_z_image_runtime_block", lambda: None)

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "diffusion_pytorch_model.safetensors" in block["reason"]


def _write_complete_z_image_folder(model_path: Path) -> None:
    model_index = {
        "_class_name": "ZImagePipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3Model"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    component_configs = {
        "scheduler": {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000},
        "text_encoder": {"model_type": "qwen3", "hidden_size": 2, "num_hidden_layers": 1, "vocab_size": 4},
        "tokenizer": {"tokenizer_class": "Qwen2Tokenizer"},
        "transformer": {"_class_name": "ZImageTransformer2DModel", "in_channels": 16, "dim": 2, "n_layers": 1},
        "vae": {"_class_name": "AutoencoderKL", "in_channels": 3, "out_channels": 3, "latent_channels": 16},
    }
    for component in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae"):
        folder = model_path / component
        folder.mkdir()
        config_name = "scheduler_config.json" if component == "scheduler" else "tokenizer_config.json" if component == "tokenizer" else "config.json"
        (folder / config_name).write_text(json.dumps(component_configs[component]), encoding="utf-8")
    tokenizer_payload = {"version": "1.0", "model": {"type": "BPE", "vocab": {}, "merges": []}, "added_tokens": []}
    (model_path / "tokenizer" / "tokenizer.json").write_text(json.dumps(tokenizer_payload), encoding="utf-8")
    # Minimal valid safe-tensors framing. This checks metadata and byte layout
    # only; it does not assert these fixture weights load a Z-Image model.
    tensor_header = b'{"fixture":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    tiny_tensor = struct.pack("<Q", len(tensor_header)) + tensor_header + bytes(4)
    (model_path / "text_encoder" / "model.safetensors").write_bytes(tiny_tensor)
    for component in ("transformer", "vae"):
        (model_path / component / "diffusion_pytorch_model.safetensors").write_bytes(tiny_tensor)


def test_z_image_folder_is_blocked_when_runtime_classes_are_missing(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "z-image" / "Z-Image-Turbo"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text('{"_class_name":"ZImagePipeline"}', encoding="utf-8")
    checkpoint = Checkpoint(
        id="z-image-folder",
        title="Z-Image Turbo",
        filename=model_path.name,
        path=str(model_path),
        architecture="z_image",
    )
    monkeypatch.setattr(
        pro_api,
        "_z_image_runtime_block",
        lambda: {
            "status": "blocked-runtime",
            "reason": "Z-Image runtime is missing required classes: diffusers.ZImagePipeline",
            "suggestedAction": "Install a compatible Diffusers runtime.",
        },
    )

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "blocked-runtime"
    assert "diffusers.ZImagePipeline" in block["reason"]


def test_qwen_image_21_requires_complete_local_snapshot_when_pipeline_exists(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "qwen-image" / "Qwen-Image-2.1"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text('{"_class_name":"QwenImage21Pipeline"}', encoding="utf-8")
    checkpoint = Checkpoint(
        id="qwen-image-21",
        title="Qwen Image 2.1",
        filename=model_path.name,
        path=str(model_path),
        architecture="qwen_image",
    )
    monkeypatch.setattr(pro_api, "_qwen_image_21_pipeline_available", lambda: True)
    monkeypatch.setattr(
        "aiwf.infrastructure.diffusers.checkpoints.missing_diffusers_local_files",
        lambda _path, limit=8: [model_path / "transformer" / "missing.safetensors"],
    )

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "missing.safetensors" in block["reason"]


def test_qwen_image_1x_requires_selected_folder_to_be_complete(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "qwen-image" / "Qwen-Image"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text('{"_class_name":"QwenImagePipeline"}', encoding="utf-8")
    checkpoint = Checkpoint(
        id="qwen-image-1x",
        title="Qwen Image",
        filename=model_path.name,
        path=str(model_path),
        architecture="qwen_image",
    )
    monkeypatch.setattr(
        "aiwf.infrastructure.diffusers.checkpoints.missing_diffusers_local_files",
        lambda _path, limit=8: [model_path / "transformer" / "missing.safetensors"],
    )

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "missing.safetensors" in block["reason"]


def test_sana_requires_selected_folder_to_be_a_complete_supported_snapshot(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "sana" / "Sana-Sprint"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text('{"_class_name":"SanaSprintPipeline"}', encoding="utf-8")
    checkpoint = Checkpoint(
        id="sana-sprint",
        title="Sana Sprint",
        filename=model_path.name,
        path=str(model_path),
        architecture="sana",
    )
    monkeypatch.setattr(
        "aiwf.infrastructure.diffusers.checkpoints.missing_diffusers_local_files",
        lambda _path, limit=8: [model_path / "transformer" / "missing.safetensors"],
    )

    block = pro_api._runtime_checkpoint_block(_ctx(tmp_path), checkpoint)

    assert block is not None
    assert block["status"] == "missing-assets"
    assert "missing.safetensors" in block["reason"]


def test_bootstrap_offers_krea2_only_when_selected_diffusers_folder_preflights(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "krea2" / "Diffusers" / "Krea2-Turbo"
    model_path.mkdir(parents=True)
    checkpoint = Checkpoint(
        id="krea2-turbo",
        title="Krea 2 Turbo",
        filename=model_path.name,
        path=str(model_path),
        architecture="krea2",
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_krea2_pipeline",
        lambda _flags: SimpleNamespace(ok=True, warnings=(), metadata={"diffusers_folder": str(model_path)}),
    )

    response = _client(ctx).get("/api/pro/bootstrap")

    assert response.status_code == 200
    model = next(item for item in response.json()["checkpoints"] if item["id"] == checkpoint.id)
    assert model["engineId"] == "krea2"
    assert model["engineLabel"] == "Krea 2"
    assert model["routeStatus"] == "request-eligible"


def test_bootstrap_recent_images_validates_disk_artifact_format_and_bounds(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    valid_path = output_dir / "valid.png"
    Image.new("RGB", (8, 6), (20, 40, 60)).save(valid_path, format="PNG")

    wrong_format_path = output_dir / "wrong-format.png"
    Image.new("RGB", (8, 6), (60, 40, 20)).save(wrong_format_path, format="JPEG")
    corrupt_path = output_dir / "corrupt.png"
    corrupt_path.write_bytes(b"not an image")

    def chunk(kind: bytes, content: bytes) -> bytes:
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 9000, 9000, 8, 2, 0, 0, 0)
    huge_path = output_dir / "oversized.png"
    huge_path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", b"") + chunk(b"IEND", b"")
    )

    client = _client(_ctx(tmp_path))
    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    images = response.json()["recentImages"]
    served_paths = {item["path"] for item in images}
    assert str(valid_path) in served_paths
    assert str(wrong_format_path) not in served_paths
    assert str(corrupt_path) not in served_paths
    assert str(huge_path) not in served_paths


def test_recent_image_preview_size_limit_precedes_artifact_decode(tmp_path, monkeypatch):
    oversized = tmp_path / "oversized.png"
    oversized.write_bytes(b"x" * (pro_api._RECENT_MAX_BYTES + 1))
    calls = []

    def validate(_path):
        calls.append("validate")
        return (1, 1)

    monkeypatch.setattr(pro_api, "image_artifact_dimensions", validate)
    assert pro_api._image_path_to_data_url(oversized) is None
    assert calls == []


def test_bootstrap_and_generate_report_missing_checkpoint_assets(tmp_path):
    ctx = _ctx(tmp_path)
    checkpoint = ctx.generation.list_checkpoints()[0]
    ctx.generation.list_checkpoints = lambda: [checkpoint]
    Path(checkpoint.path).unlink()
    client = _client(ctx)

    bootstrap = client.get("/api/pro/bootstrap")
    assert bootstrap.status_code == 200
    blocked = next(item for item in bootstrap.json()["blockedCheckpoints"] if item["id"] == "model-a")
    assert blocked["checkpointPathStatus"] == "missing"
    assert blocked["routeStatus"] == "blocked"
    assert blocked["status"] == "missing-assets"
    assert "no longer present" in blocked["reason"]

    response = client.post("/api/pro/generate", json={"prompt": "test", "model_id": "model-a"})
    assert response.status_code == 422


def test_ltx_worker_error_marks_job_failed_without_sana_audio_fields(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight
    from aiwf.services.ltx import LTX_FULL_CHECKPOINT_FP8

    monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", _ready_ltx_preflight)
    ctx = _ctx(tmp_path)
    ctx.ltx.default_checkpoint_path = lambda _pipeline: tmp_path / LTX_FULL_CHECKPOINT_FP8
    ctx.ltx.generate = lambda _request, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker failed"))

    response = _client(ctx).post(
        "/api/pro/generate",
        json={"prompt": "motion", "mode": "video", "model_id": "ltx:one_stage"},
    )

    assert response.status_code == 500
    assert response.json()["detail"]["message"] == "worker failed"
    assert pro_api._pro_video_job_status(ctx)["state"] == "failed"


def test_bootstrap_does_not_claim_route_ready_without_absolute_checkpoint_path(tmp_path):
    ctx = _ctx(tmp_path)
    checkpoint = Checkpoint(
        id="relative-model",
        title="Relative model",
        filename="relative-model.safetensors",
        path="models/relative-model.safetensors",
        architecture="sdxl",
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]
    client = _client(ctx)

    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    model = next(item for item in response.json()["checkpoints"] if item["id"] == "relative-model")
    assert model["checkpointPathStatus"] == "unknown"
    assert model["routeStatus"] == "unknown"


def test_bootstrap_path_presence_does_not_claim_directory_contents_verified(tmp_path):
    ctx = _ctx(tmp_path)
    model_dir = tmp_path / "models" / "incomplete-diffusers"
    model_dir.mkdir(parents=True)
    checkpoint = Checkpoint(
        id="incomplete-diffusers",
        title="Incomplete Diffusers directory",
        filename="incomplete-diffusers",
        path=str(model_dir),
        architecture="sdxl",
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]
    client = _client(ctx)

    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    model = next(item for item in response.json()["blockedCheckpoints"] if item["id"] == checkpoint.id)
    assert model["checkpointPathStatus"] == "present"
    assert model["routeStatus"] == "blocked"
    assert "requires a complete Diffusers folder with model_index.json" in model["readinessReason"]
    assert "verificationStatus" not in model


def test_generate_rejects_unknown_explicit_model_id(tmp_path):
    ctx = _ctx(tmp_path)
    # GenerationService resolvers can return the configured default when an
    # explicit ID is unknown; the API guard must reject that fallback.
    ctx.generation.resolve_checkpoint = lambda checkpoint_id=None: ctx.generation.list_checkpoints()[0]
    client = _client(ctx)

    response = client.post("/api/pro/generate", json={"prompt": "test", "model_id": "removed-model"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["status"] == "unavailable-model"
    assert detail["checkpointId"] == "removed-model"
    assert ctx.generation.submitted == []

    default_response = client.post("/api/pro/generate", json={"prompt": "test default"})
    assert default_response.status_code == 200
    assert len(ctx.generation.submitted) == 1


def test_bootstrap_hides_blocked_selectable_checkpoints(tmp_path):
    ctx = _ctx(tmp_path)
    blocked = Checkpoint(
        id="blocked-upscaler",
        title="4xBHI dat2 multiblur",
        filename="4xBHI_dat2_multiblurjpg.safetensors",
        path=str(tmp_path / "models" / "upscale_models" / "4xBHI_dat2_multiblurjpg.safetensors"),
        architecture="sd15",
    )
    ctx.generation.list_checkpoints = lambda: [blocked, *_Generation(ctx.generation.output_dir).list_checkpoints()]
    client = _client(ctx)

    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert "blocked-upscaler" not in {item["id"] for item in data["checkpoints"]}
    assert data["counts"]["blockedCheckpoints"] == 6
    blocked_ids = {item["id"] for item in data["blockedCheckpoints"]}
    assert "blocked-upscaler" in blocked_ids
    assert sum(1 for item in data["blockedCheckpoints"] if item["engineId"] == "sana_video") == 2
    assert data["blockedCheckpoints"][0]["status"] == "broken-runtime"
    assert "missing the expected CLIP text model" in data["blockedCheckpoints"][0]["reason"]


def test_bootstrap_exposes_qwen_nunchaku_as_blocked_setup_candidate(tmp_path):
    ctx = _ctx(tmp_path)
    qwen = Checkpoint(
        id="qwen-nunchaku",
        title="Qwen Nunchaku Lightning",
        filename="svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors",
        path=str(tmp_path / "models" / "qwen-image" / "Nunchaku" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"),
        architecture="qwen_image_nunchaku",
    )
    ctx.settings.last_checkpoint_id = "qwen-nunchaku"
    ctx.generation.list_checkpoints = lambda: [qwen, *_Generation(ctx.generation.output_dir).list_checkpoints()]
    client = _client(ctx)

    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert "qwen-nunchaku" not in {item["id"] for item in data["checkpoints"]}
    blocked = {item["id"]: item for item in data["blockedCheckpoints"]}
    assert "qwen-nunchaku" in blocked
    assert blocked["qwen-nunchaku"]["setupBundleKey"] == "qwen-nunchaku"
    assert blocked["qwen-nunchaku"]["setupRoute"]["supportState"] == "setup-available"
    assert "generation remains blocked" in blocked["qwen-nunchaku"]["reason"].lower()
    assert data["settings"]["checkpointId"] == "qwen-nunchaku"


def test_bootstrap_hides_sd35_large_without_gated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_api, "_sd35_large_access_available", lambda: False)
    ctx = _ctx(tmp_path)
    sd35 = Checkpoint(
        id="sd35-large",
        title="SD3.5 Large FP8",
        filename="sd3.5_large_fp8_scaled.safetensors",
        path=str(tmp_path / "models" / "sd3.5_large_fp8_scaled.safetensors"),
        architecture="sd35",
    )
    Path(sd35.path).parent.mkdir(parents=True, exist_ok=True)
    Path(sd35.path).write_bytes(b"fake SD3.5 checkpoint fixture")
    ctx.settings.last_checkpoint_id = "sd35-large"
    ctx.generation.list_checkpoints = lambda: [sd35, *_Generation(ctx.generation.output_dir).list_checkpoints()]
    client = _client(ctx)

    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert "sd35-large" not in {item["id"] for item in data["checkpoints"]}
    assert data["settings"]["checkpointId"] == "model-a"
    blocked = next(item for item in data["blockedCheckpoints"] if item["id"] == "sd35-large")
    assert blocked["status"] == "blocked-cleanly"
    assert "gated Stability AI config files" in blocked["reason"]


def test_model_upload_sorts_gguf_and_refreshes_inventory(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/pro/models/upload",
        files={"file": ("flux1-dev-Q5_K_M.gguf", b"GGUF", "application/octet-stream")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["moved"] == 1
    assert data["counts"]["inventoryCount"] >= 1
    action = data["actions"][0]
    assert action["filename"] == "flux1-dev-Q5_K_M.gguf"
    assert action["destSubdir"] == "flux/GGUF"
    assert (tmp_path / "models" / "flux" / "GGUF" / "flux1-dev-Q5_K_M.gguf").is_file()


def test_model_inventory_refresh_invalidates_runtime_model_catalogs(monkeypatch, tmp_path):
    calls = []
    backend = SimpleNamespace(
        invalidate_checkpoints=lambda: calls.append("checkpoints"),
        invalidate_loras=lambda: calls.append("loras"),
        invalidate_embeddings=lambda: calls.append("embeddings"),
        invalidate_vaes=lambda: calls.append("vaes"),
    )
    ctx = SimpleNamespace(
        flags=RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"),
        generation=SimpleNamespace(backend=backend),
        _pro_capability_cache={"stale": True},
    )
    monkeypatch.setattr(pro_api, "scan_and_write_model_inventory", lambda _flags: [object()])

    result = pro_api._refresh_model_inventory_after_sort(ctx)

    assert result == {"inventoryCount": 1}
    assert calls == ["checkpoints", "loras", "embeddings", "vaes"]
    assert ctx._pro_capability_cache is None


def test_model_roots_scan_endpoint_returns_inventory_scan_report(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path)
    calls = []
    ctx.generation.backend = SimpleNamespace(
        invalidate_checkpoints=lambda: calls.append("checkpoints"),
        invalidate_loras=lambda: calls.append("loras"),
        invalidate_embeddings=lambda: calls.append("embeddings"),
        invalidate_vaes=lambda: calls.append("vaes"),
    )
    ctx._pro_capability_cache = {"stale": True}
    report = {
        "inventoryCount": 1,
        "roots": [{"label": "Shared", "path": str(tmp_path / "shared"), "status": "scanned", "assetCount": 1, "familyCounts": {"runtime_asset": 1}}],
        "assets": [{"path": str(tmp_path / "shared" / "asset.safetensors"), "filename": "asset.safetensors", "family": "runtime_asset", "architecture": "unknown", "currentSubdir": "", "recommendedSubdir": "", "placement": "in-place", "signals": {}}],
        "assetsTruncated": 0,
    }
    monkeypatch.setattr(pro_api, "scan_model_inventory_report", lambda flags, **_kwargs: report)

    response = _client(ctx).post("/api/pro/models/scan")

    assert response.status_code == 200
    result = response.json()
    assert result["inventoryCount"] == 1
    assert result["assets"] == report["assets"]
    assert result["matchedCount"] == 1
    assert result["assetsTruncated"] == 0
    assert result["offset"] == 0
    assert result["hasMore"] is False
    assert result["scanId"]
    assert calls == ["checkpoints", "loras", "embeddings", "vaes"]
    assert ctx._pro_capability_cache is None


def test_model_roots_scan_paginates_past_250_and_searches_later_proposals(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path)
    from aiwf.infrastructure import model_inventory
    from aiwf.infrastructure.model_inventory import ModelInventoryRecord

    ctx.flags = ctx.flags.model_copy(update={"models_dir": tmp_path / "inventory-models"})
    model_root = ctx.flags.resolved_models_dir()
    model_root.mkdir(parents=True, exist_ok=True)
    for index in range(300):
        (model_root / f"asset-{index:03}.safetensors").write_bytes(b"fixture")

    def classify(path, _roots):
        return ModelInventoryRecord(
            path=str(path.resolve()), filename=path.name, family="checkpoint", architecture="sdxl",
            current_subdir="", recommended_subdir="Stable-diffusion", should_move=True,
        )

    monkeypatch.setattr(model_inventory, "classify_model_file", classify)
    client = _client(ctx)

    first = client.post("/api/pro/models/scan?limit=50")
    assert first.status_code == 200
    first_page = first.json()
    assert len(first_page["assets"]) == 50
    assert first_page["inventoryCount"] == 300
    assert first_page["matchedCount"] == 300
    assert first_page["assetsTruncated"] == 250
    assert first_page["nextOffset"] == 50

    later = client.get(f"/api/pro/models/scan/{first_page['scanId']}?offset=250&limit=50")
    assert later.status_code == 200
    assert later.json()["assets"][0]["filename"] == "asset-250.safetensors"
    assert later.json()["assetsTruncated"] == 0

    searched = client.get(f"/api/pro/models/scan/{first_page['scanId']}?query=asset-275.safetensors")
    assert searched.status_code == 200
    assert searched.json()["matchedCount"] == 1
    assert searched.json()["assets"][0]["filename"] == "asset-275.safetensors"
    assert searched.json()["query"] == "asset-275.safetensors"


def _shared_model_placement_preview(client, source: Path):
    scan = client.post("/api/pro/models/scan")
    assert scan.status_code == 200
    scan_data = scan.json()
    preview = client.post(
        f"/api/pro/models/scan/{scan_data['scanId']}/placements/preview",
        json={"scanId": scan_data["scanId"], "path": str(source.resolve())},
    )
    return scan_data, preview


def _shared_model_placement_context(tmp_path: Path):
    shared = tmp_path / "shared-models"
    source = shared / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"GGUF test fixture")
    ctx = _ctx(tmp_path)
    ctx.flags = RuntimeFlags(data_dir=tmp_path, extra_model_dirs=[shared])
    return ctx, source


def test_shared_root_candidate_preview_and_apply_copy_preserves_source(tmp_path):
    ctx, source = _shared_model_placement_context(tmp_path)
    client = _client(ctx)
    scan, preview = _shared_model_placement_preview(client, source)

    assert preview.status_code == 200, preview.text
    details = preview.json()
    assert details["scanId"] == scan["scanId"]
    assert details["source"] == str(source.resolve())
    assert details["destination"] == str((tmp_path / "models" / "flux" / "GGUF" / source.name).resolve())
    assert details["sizeBytes"] == source.stat().st_size
    assert details["requiredFreeBytes"] > details["sizeBytes"]
    assert details["availableFreeBytes"] >= 0
    assert details["collision"] is False
    assert details["canApply"] is True

    applied = client.post("/api/pro/models/scan/placements/apply", json={"planId": details["planId"]})
    assert applied.status_code == 200, applied.text
    destination = Path(applied.json()["destination"])
    assert destination.read_bytes() == b"GGUF test fixture"
    assert source.read_bytes() == b"GGUF test fixture"
    assert applied.json()["sourcePreserved"] is True


def test_shared_checkpoint_root_candidate_can_be_placed(tmp_path):
    ctx, source = _shared_model_placement_context(tmp_path)
    ctx.flags = RuntimeFlags(data_dir=tmp_path, extra_ckpt_dirs=[source.parents[1]])
    client = _client(ctx)
    _scan, preview = _shared_model_placement_preview(client, source)

    assert preview.status_code == 200, preview.text
    assert preview.json()["canApply"] is True
    applied = client.post("/api/pro/models/scan/placements/apply", json={"planId": preview.json()["planId"]})
    assert applied.status_code == 200, applied.text
    assert Path(applied.json()["destination"]).read_bytes() == b"GGUF test fixture"
    assert source.read_bytes() == b"GGUF test fixture"


def test_shared_root_candidate_placement_rejects_unscanned_traversal_path(tmp_path):
    ctx, _source = _shared_model_placement_context(tmp_path)
    client = _client(ctx)
    scan = client.post("/api/pro/models/scan").json()

    response = client.post(
        f"/api/pro/models/scan/{scan['scanId']}/placements/preview",
        json={"scanId": scan["scanId"], "path": str((tmp_path / "shared-models" / ".." / "outside.gguf").absolute())},
    )

    assert response.status_code == 404
    assert "exact path was not included" in response.json()["detail"]


def test_shared_root_candidate_placement_rejects_primary_root_source(tmp_path):
    ctx, _shared_source = _shared_model_placement_context(tmp_path)
    source = tmp_path / "models" / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"GGUF primary fixture")
    client = _client(ctx)
    scan = client.post("/api/pro/models/scan").json()

    response = client.post(
        f"/api/pro/models/scan/{scan['scanId']}/placements/preview",
        json={"scanId": scan["scanId"], "path": str(source.resolve())},
    )

    assert response.status_code == 422
    assert "inside the primary models root" in response.json()["detail"]


def test_shared_root_candidate_placement_rejects_overlapping_configured_roots(tmp_path):
    shared = tmp_path / "shared-parent"
    source = shared / "another-library" / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"GGUF overlap fixture")
    ctx = _ctx(tmp_path)
    ctx.flags = RuntimeFlags(data_dir=tmp_path, models_dir=shared / "models", extra_model_dirs=[shared])
    client = _client(ctx)
    scan = client.post("/api/pro/models/scan").json()

    response = client.post(
        f"/api/pro/models/scan/{scan['scanId']}/placements/preview",
        json={"scanId": scan["scanId"], "path": str(source.resolve())},
    )

    assert response.status_code == 409
    assert "roots overlap" in response.json()["detail"]


def test_shared_root_candidate_placement_rejects_destination_symlink_parent(tmp_path):
    ctx, source = _shared_model_placement_context(tmp_path)
    client = _client(ctx)
    _scan, preview = _shared_model_placement_preview(client, source)
    assert preview.status_code == 200, preview.text
    destination_parent = tmp_path / "models" / "flux"
    destination_parent.parent.mkdir(parents=True, exist_ok=True)
    redirected = tmp_path / "redirected-model-destination"
    redirected.mkdir()
    try:
        destination_parent.symlink_to(redirected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink is unavailable in this Windows test environment: {exc}")

    response = client.post("/api/pro/models/scan/placements/apply", json={"planId": preview.json()["planId"]})

    assert response.status_code == 422
    assert "symlink, reparse point" in response.json()["detail"]
    assert source.is_file()
    assert not (redirected / "GGUF" / source.name).exists()


def test_shared_root_candidate_placement_rejects_destination_collision(tmp_path):
    ctx, source = _shared_model_placement_context(tmp_path)
    destination = tmp_path / "models" / "flux" / "GGUF" / source.name
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"user-owned destination")
    client = _client(ctx)
    _scan, preview = _shared_model_placement_preview(client, source)

    assert preview.status_code == 200, preview.text
    details = preview.json()
    assert details["collision"] is True
    assert details["canApply"] is False
    response = client.post("/api/pro/models/scan/placements/apply", json={"planId": details["planId"]})
    assert response.status_code == 409
    assert destination.read_bytes() == b"user-owned destination"
    assert source.read_bytes() == b"GGUF test fixture"


def test_shared_root_candidate_placement_reports_insufficient_space(tmp_path, monkeypatch):
    ctx, source = _shared_model_placement_context(tmp_path)
    monkeypatch.setattr(pro_api.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    client = _client(ctx)
    _scan, preview = _shared_model_placement_preview(client, source)

    assert preview.status_code == 200, preview.text
    details = preview.json()
    assert details["status"] == "insufficient_space"
    assert details["canApply"] is False
    response = client.post("/api/pro/models/scan/placements/apply", json={"planId": details["planId"]})
    assert response.status_code == 409
    assert source.is_file()
    assert not Path(details["destination"]).exists()


def test_shared_root_candidate_placement_rejects_source_changed_after_preview(tmp_path):
    ctx, source = _shared_model_placement_context(tmp_path)
    client = _client(ctx)
    _scan, preview = _shared_model_placement_preview(client, source)
    assert preview.status_code == 200, preview.text
    plan_id = preview.json()["planId"]
    source.write_bytes(b"changed after preview")

    response = client.post("/api/pro/models/scan/placements/apply", json={"planId": plan_id})

    assert response.status_code == 409
    assert "changed after preview" in response.json()["detail"]
    assert not Path(preview.json()["destination"]).exists()
    assert source.read_bytes() == b"changed after preview"


def test_shared_root_candidate_placement_rejects_same_size_rewrite_with_restored_mtime(tmp_path):
    ctx, source = _shared_model_placement_context(tmp_path)
    client = _client(ctx)
    _scan, preview = _shared_model_placement_preview(client, source)
    assert preview.status_code == 200, preview.text
    original_stat = source.stat()
    original = source.read_bytes()
    replacement = bytes((byte ^ 1) for byte in original)
    assert len(replacement) == len(original)
    source.write_bytes(replacement)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    response = client.post("/api/pro/models/scan/placements/apply", json={"planId": preview.json()["planId"]})

    assert response.status_code == 409
    assert "changed after preview" in response.json()["detail"]
    assert not Path(preview.json()["destination"]).exists()
    assert source.read_bytes() == replacement


def test_shared_root_candidate_placement_rejects_active_generation(tmp_path):
    ctx, source = _shared_model_placement_context(tmp_path)
    client = _client(ctx)
    _scan, preview = _shared_model_placement_preview(client, source)
    assert preview.status_code == 200, preview.text
    pro_api._pro_video_job_start(ctx, SimpleNamespace(steps=1), message="running placement exclusion test")

    response = client.post("/api/pro/models/scan/placements/apply", json={"planId": preview.json()["planId"]})

    assert response.status_code == 409
    assert "GPU generation job is active" in response.json()["detail"]
    assert source.is_file()
    assert not Path(preview.json()["destination"]).exists()


def test_model_roots_scan_is_rejected_while_video_generation_is_active(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path)
    calls = []
    monkeypatch.setattr(pro_api, "scan_model_inventory_report", lambda _flags: calls.append("scan") or {})
    pro_api._pro_video_job_start(ctx, SimpleNamespace(steps=1), message="running test video")

    response = _client(ctx).post("/api/pro/models/scan")

    assert response.status_code == 409
    assert "GPU generation job is active" in response.json()["detail"]
    assert calls == []


def test_model_reorganize_rereads_headers_and_moves_confident_matches(tmp_path):
    ctx = _ctx(tmp_path)
    models = tmp_path / "models"
    misplaced = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(b"GGUF")
    client = _client(ctx)

    preview = client.get("/api/pro/models/reorganize/plan")
    assert preview.status_code == 200
    plan_id = preview.json()["planId"]
    response = client.post("/api/pro/models/reorganize", json={"planId": plan_id})

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["moved"] == 1
    flux_action = next(item for item in data["actions"] if item["filename"] == misplaced.name)
    assert flux_action["destSubdir"] == "flux/GGUF"
    assert not misplaced.exists()
    assert (models / "flux" / "GGUF" / "flux1-dev-Q5_K_M.gguf").is_file()


def test_model_reorganize_rejects_stale_review_without_moving_any_model(tmp_path):
    ctx = _ctx(tmp_path)
    models = tmp_path / "models"
    misplaced = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(b"GGUF")
    client = _client(ctx)

    preview = client.get("/api/pro/models/reorganize/plan")
    assert preview.status_code == 200
    plan_id = preview.json()["planId"]
    added_after_review = models / "Stable-diffusion" / "flux1-schnell-Q5_K_M.gguf"
    added_after_review.write_bytes(b"GGUF")

    response = client.post("/api/pro/models/reorganize", json={"planId": plan_id})

    assert response.status_code == 409
    assert "changed after the preview" in response.json()["detail"]
    assert misplaced.is_file()
    assert added_after_review.is_file()
    assert not (models / "flux" / "GGUF" / misplaced.name).exists()
    assert not (models / "flux" / "GGUF" / added_after_review.name).exists()


def test_model_reorganize_rejects_active_model_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    ctx = _ctx(tmp_path)
    misplaced = tmp_path / "models" / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(b"GGUF")
    client = _client(ctx)
    preview = client.get("/api/pro/models/reorganize/plan")
    assert preview.status_code == 200
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(ctx)
    assert lock.acquire(blocking=False)
    try:
        response = client.post("/api/pro/models/reorganize", json={"planId": preview.json()["planId"]})
    finally:
        lock.release()

    assert response.status_code == 409
    assert "model operation is in progress" in response.json()["detail"]
    assert misplaced.is_file()


def test_model_reorganize_plan_previews_without_moving_files(tmp_path):
    ctx = _ctx(tmp_path)
    models = tmp_path / "models"
    misplaced = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(b"GGUF")

    response = _client(ctx).get("/api/pro/models/reorganize/plan")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "planned"
    assert data["counts"]["planned"] == 1
    assert data["counts"]["moved"] == 0
    flux_action = next(item for item in data["actions"] if item["filename"] == misplaced.name)
    assert flux_action["status"] == "would-move"
    assert flux_action["destSubdir"] == "flux/GGUF"
    assert misplaced.is_file()
    assert not (models / "flux" / "GGUF" / misplaced.name).exists()


def test_model_reorganize_plan_does_not_invalidate_live_model_inventory(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(
        pro_api,
        "_refresh_model_inventory_after_sort",
        lambda _ctx: pytest.fail("read-only plan must not rewrite or invalidate inventory"),
    )

    response = _client(ctx).get("/api/pro/models/reorganize/plan")

    assert response.status_code == 200
    assert response.json()["status"] == "planned"


def test_model_upload_stays_staged_when_generation_blocks_automatic_placement(tmp_path):
    ctx = _ctx(tmp_path)
    pro_api._pro_video_job_start(ctx, SimpleNamespace(steps=1), message="running test video")

    response = _client(ctx).post(
        "/api/pro/models/upload",
        files={"file": ("uploaded-model.safetensors", b"uploaded model bytes", "application/octet-stream")},
    )

    staged = pro_api._model_upload_root(ctx) / "uploaded-model.safetensors"
    assert response.status_code == 409
    assert "safely staged" in response.json()["detail"]
    assert staged.is_file()
    assert not (ctx.flags.resolved_models_dir() / "Stable-diffusion" / staged.name).exists()


def test_model_unload_endpoint_calls_backend_unload(tmp_path):
    ctx = _ctx(tmp_path)
    calls = []
    tenant_calls = []
    from aiwf.core.domain.engine import EngineSwitchResult, EngineTenant

    def request_switch(request):
        tenant_calls.append(request.target)
        return EngineSwitchResult(True, request.target, "ok")

    ctx.supervisor = SimpleNamespace(request_switch=request_switch)
    active = Checkpoint(
        id="model-a",
        title="Model A",
        filename="model-a.safetensors",
        path=str(tmp_path / "models" / "model-a.safetensors"),
        architecture="sdxl",
    )
    ctx.generation.backend = SimpleNamespace(
        devices=_Devices(),
        _active=active,
        _txt2img=object(),
        is_checkpoint_loaded=lambda _checkpoint_id=None: True,
        unload=lambda: calls.append("unload"),
    )
    client = _client(ctx)

    response = client.post("/api/pro/models/unload")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unloaded"
    assert data["unloadedModel"]["name"] == "Model A"
    assert calls == ["unload"]
    assert tenant_calls == [EngineTenant.IMAGE, EngineTenant.IDLE]


def test_model_unload_endpoint_rejects_when_another_gpu_tenant_owns_device(tmp_path):
    ctx = _ctx(tmp_path)
    calls = []
    from aiwf.core.domain.engine import EngineSwitchResult, EngineTenant

    def request_switch(request):
        if request.target == EngineTenant.IMAGE:
            return EngineSwitchResult(False, EngineTenant.AUDIO, "Audio generation owns the GPU.")
        return EngineSwitchResult(True, EngineTenant.IDLE, "ok")

    ctx.supervisor = SimpleNamespace(request_switch=request_switch)
    ctx.generation.backend = SimpleNamespace(unload=lambda: calls.append("unload"))

    response = _client(ctx).post("/api/pro/models/unload")

    assert response.status_code == 409
    assert "Audio generation owns the GPU" in response.json()["detail"]
    assert calls == []


def test_support_terminal_endpoint_is_disabled(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    launched = []
    monkeypatch.setattr(
        pro_api.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)) or SimpleNamespace(),
    )
    client = _client(ctx)

    response = client.post("/api/pro/support/terminal")

    assert response.status_code == 410
    assert "Visible support terminals are disabled" in response.json()["detail"]
    assert launched == []


def test_bootstrap_allows_sd35_large_when_hf_auth_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_api, "_sd35_large_access_available", lambda: True)
    ctx = _ctx(tmp_path)
    sd35 = Checkpoint(
        id="sd35-large",
        title="SD3.5 Large FP8",
        filename="sd3.5_large_fp8_scaled.safetensors",
        path=str(tmp_path / "models" / "sd3.5_large_fp8_scaled.safetensors"),
        architecture="sd35",
    )
    Path(sd35.path).parent.mkdir(parents=True, exist_ok=True)
    Path(sd35.path).write_bytes(b"fake SD3.5 checkpoint fixture")
    ctx.generation.list_checkpoints = lambda: [sd35, *_Generation(ctx.generation.output_dir).list_checkpoints()]
    client = _client(ctx)

    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert "sd35-large" in {item["id"] for item in data["checkpoints"]}


def test_runtime_does_not_submit_generation(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.get("/api/pro/runtime")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert "GPU utilization" in {item["label"] for item in data["resources"]}
    assert ctx.generation.submitted == []


def test_runtime_reports_pending_generation_queue(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.generation.pending_count = lambda: 2
    client = _client(ctx)

    response = client.get("/api/pro/runtime")

    assert response.status_code == 200
    assert response.json()["queueCount"] == 2


def test_runtime_ignores_stale_completed_active_image_job(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.generation._active = JobRecord(request=GenerationRequest(prompt="cat"), state=JobState.COMPLETED)
    client = _client(ctx)

    response = client.get("/api/pro/runtime")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["queueCount"] == 0
    assert data["job"]["state"] == "idle"


def test_runtime_reports_active_sana_video_job(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    job_id = pro_api._pro_video_job_start(ctx, SanaVideoRequest(prompt="slow camera move", steps=4))
    pro_api._pro_video_job_update(ctx, job_id, progress=0.5, message="Denoising step 2/4", step=2, total=4)

    response = client.get("/api/pro/runtime")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["queueCount"] == 1
    assert data["job"]["id"] == job_id
    assert data["job"]["progress"] == 50
    assert data["job"]["message"] == "Denoising step 2/4"

    pro_api._pro_video_job_finish(ctx, job_id, "completed", message="done")


def test_video_job_start_uses_route_specific_initial_label(tmp_path):
    ctx = _ctx(tmp_path)

    ltx_job = pro_api._pro_video_job_start(
        ctx, SanaVideoRequest(prompt="test", steps=1), message="Starting LTX video generation."
    )
    assert pro_api._pro_video_job_status(ctx)["message"] == "Starting LTX video generation."
    pro_api._pro_video_job_finish(ctx, ltx_job, "completed", message="done")

    sana_job = pro_api._pro_video_job_start(
        ctx, SanaVideoRequest(prompt="test", steps=1), message="Starting Sana video generation."
    )
    assert pro_api._pro_video_job_status(ctx)["message"] == "Starting Sana video generation."
    pro_api._pro_video_job_finish(ctx, sana_job, "completed", message="done")

    wan_job = pro_api._pro_video_job_start(
        ctx, SanaVideoRequest(prompt="test", steps=1), message="Starting Wan video generation."
    )
    assert pro_api._pro_video_job_status(ctx)["message"] == "Starting Wan video generation."
    pro_api._pro_video_job_finish(ctx, wan_job, "completed", message="done")


def test_interrupt_routes_ltx_cancel_to_active_service(tmp_path):
    ctx = _ctx(tmp_path)
    cancelled = []
    ctx.ltx.cancel_active_generation = lambda: cancelled.append(True)
    job_id = pro_api._pro_video_job_start(
        ctx, SanaVideoRequest(prompt="test", steps=1), cancel_target="ltx"
    )

    assert pro_api._request_pro_video_cancel(ctx) == job_id
    assert cancelled == [True]
    cancel_message = pro_api._pro_video_job_status(ctx)["message"]
    assert "active generation" in cancel_message
    assert "Sana" not in cancel_message
    assert "Wan" not in cancel_message
    assert "LTX" not in cancel_message
    pro_api._pro_video_job_finish(ctx, job_id, "cancelled", message="cancelled")


def test_runtime_exposes_route_lifecycle_without_claiming_residency(tmp_path):
    ctx = _ctx(tmp_path)
    pro_api.select_route(
        ctx,
        route="video.ltx.distilled",
        model_id="ltx:distilled",
        setup_ready=True,
        support_ids=["ltx-distilled"],
    )

    response = _client(ctx).get("/api/pro/runtime")

    assert response.status_code == 200
    route = next(item for item in response.json()["routeLifecycle"] if item["route"] == "video.ltx.distilled")
    assert route["status"] == "setup-ready"
    assert route["resident"] is None


def test_pro_controlnet_models_exposes_local_ids_for_the_model_picker(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.controlnet.list_models = lambda: [
        SimpleNamespace(
            id="control_v11p_sd15_canny_fp16",
            title="control_v11p_sd15_canny_fp16",
            path="F:/AIWF_Studio/models/ControlNet/control_v11p_sd15_canny_fp16.safetensors",
        ),
    ]

    response = _client(ctx).get("/api/pro/controlnet/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == [
        {
            "id": "control_v11p_sd15_canny_fp16",
            "title": "control_v11p_sd15_canny_fp16",
            "path": "F:/AIWF_Studio/models/ControlNet/control_v11p_sd15_canny_fp16.safetensors",
        },
    ]
    options = {option["key"]: option for option in payload["setupOptions"]}
    assert options["cn15-canny"] == {
        "key": "cn15-canny",
        "family": "sd15",
        "label": "ControlNet v1.1 Canny (SD1.5)",
        "modelId": "control_v11p_sd15_canny_fp16",
        "sizeMb": 689,
    }
    assert options["cnxl-canny"]["family"] == "sdxl"
    assert options["cnxl-canny"]["modelId"] == "controlnet-canny-sdxl-1.0"


def test_concurrent_video_job_starts_keep_one_tracked_job_and_cancel_handle(tmp_path):
    ctx = _ctx(tmp_path)
    start_gate = threading.Barrier(2)

    def start_video_job():
        start_gate.wait(timeout=5)
        try:
            return ("started", pro_api._pro_video_job_start(ctx, SanaVideoRequest(prompt="slow camera move", steps=4)))
        except Exception as exc:
            return ("rejected", getattr(exc, "status_code", None))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start_video_job(), range(2)))

    started = [value for outcome, value in results if outcome == "started"]
    rejected = [value for outcome, value in results if outcome == "rejected"]
    assert len(started) == 1
    assert rejected == [409]
    job_id = started[0]
    assert pro_api._pro_video_job_status(ctx)["id"] == job_id
    assert pro_api._request_pro_video_cancel(ctx) == job_id
    assert pro_api._pro_video_cancel_requested(ctx, job_id) is True
    pro_api._pro_video_job_finish(ctx, job_id, "cancelled", message="cancelled")


def test_runtime_reports_terminal_failed_sana_video_job(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    job_id = pro_api._pro_video_job_start(ctx, SanaVideoRequest(prompt="slow camera move", steps=4))
    pro_api._pro_video_job_finish(ctx, job_id, "failed", message="decode failed", error="decode failed")

    response = client.get("/api/pro/runtime")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["queueCount"] == 0
    assert data["job"]["id"] == job_id
    assert data["job"]["state"] == "failed"
    assert data["job"]["error"] == "decode failed"


def test_restart_endpoint_requests_process_exit(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    recorded: list[str] = []

    monkeypatch.setattr(pro_api, "_schedule_process_restart", lambda delay_seconds=0.25: recorded.append("restart"))

    response = client.post("/api/pro/restart")

    assert response.status_code == 200
    assert response.json()["status"] == "restart_requested"
    assert recorded == ["restart"]


def test_interrupt_marks_active_sana_video_job_for_cancel(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    job_id = pro_api._pro_video_job_start(ctx, SanaVideoRequest(prompt="slow camera move", steps=4))

    response = client.post("/api/pro/interrupt")

    assert response.status_code == 200
    assert response.json()["videoJobId"] == job_id
    assert pro_api._pro_video_cancel_requested(ctx, job_id) is True
    pro_api._pro_video_job_finish(ctx, job_id, "cancelled", message="cancelled")


def test_active_sana_video_rejects_image_generate(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    job_id = pro_api._pro_video_job_start(ctx, SanaVideoRequest(prompt="slow camera move", steps=4))

    response = client.post("/api/pro/generate", json={"prompt": "cat", "mode": "image"})

    assert response.status_code == 409
    assert ctx.generation.submitted == []
    pro_api._pro_video_job_finish(ctx, job_id, "completed", message="done")


def test_active_sana_video_rejects_second_video_generate(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    job_id = pro_api._pro_video_job_start(ctx, SanaVideoRequest(prompt="slow camera move", steps=4))

    response = client.post("/api/pro/generate", json={"prompt": "another move", "mode": "video"})

    assert response.status_code == 409
    assert ctx.sana_video.submitted == []
    pro_api._pro_video_job_finish(ctx, job_id, "completed", message="done")


def test_sana_video_no_checkpoint_readiness_uses_requested_variant(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    inspected = []
    monkeypatch.setattr(pro_api, "_pro_sana_video_backend_enabled", lambda: True)

    def model_payload(_ctx, variant="480p"):
        inspected.append(variant)
        return {"id": f"sana-{variant}", "status": "Ready"}

    monkeypatch.setattr(pro_api, "_sana_video_model_payload", model_payload)

    pro_api._assert_video_route_checkpoint(ctx, None, sana_model_variant="720p")

    assert inspected == ["720p"]


def test_active_image_job_rejects_sana_video_generate(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    ctx.generation._active = JobRecord(request=GenerationRequest(prompt="cat"), state=JobState.RUNNING)

    response = client.post("/api/pro/generate", json={"prompt": "slow camera move", "mode": "video"})

    assert response.status_code == 409
    assert ctx.sana_video.submitted == []


def test_cancelled_active_image_slot_rejects_sana_video_generate(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    ctx.generation._active = JobRecord(request=GenerationRequest(prompt="cat"), state=JobState.CANCELLED)

    response = client.post("/api/pro/generate", json={"prompt": "slow camera move", "mode": "video"})

    assert response.status_code == 409
    assert ctx.sana_video.submitted == []


def test_cancelled_active_image_slot_reports_busy_runtime(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    ctx.generation._active = JobRecord(request=GenerationRequest(prompt="cat"), state=JobState.CANCELLED)

    response = client.get("/api/pro/runtime")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["queueCount"] == 1
    assert data["job"]["state"] == "cancelled"


def test_active_image_job_rejects_second_image_generate(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    ctx.generation._active = JobRecord(request=GenerationRequest(prompt="cat"), state=JobState.RUNNING)

    response = client.post("/api/pro/generate", json={"prompt": "second cat", "mode": "image"})

    assert response.status_code == 409
    assert ctx.generation.submitted == []


def test_generate_maps_payload_and_returns_first_image(tmp_path, caplog):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    caplog.set_level(logging.INFO)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "private prompt phrase that should not enter application logs",
            "mode": "image",
            "negativePrompt": "blurry",
            "model_id": "model-a",
            "sampler": "Euler a",
            "scheduler": "karras",
            "steps": 12,
            "cfgScale": 6.5,
            "width": 512,
            "height": 768,
            "seed": 99,
            "batchSize": 2,
            "batchCount": 1,
            "enableHires": True,
            "hiresScale": 1.5,
            "hiresSteps": 8,
            "hiresDenoise": 0.25,
            "hiresUpscaler": "bicubic",
        },
    )

    assert "private prompt phrase" not in caplog.text
    assert response.status_code == 200
    request = ctx.generation.submitted[0]
    assert request.mode == GenerationMode.TXT2IMG
    assert request.prompt == "private prompt phrase that should not enter application logs"
    assert request.negative_prompt == "blurry"
    assert request.checkpoint_id == "model-a"
    assert request.sampler == "euler_a"
    assert request.scheduler == "karras"
    assert request.enable_hr is True
    assert request.hr_scale == 1.5
    assert request.hr_steps == 8
    assert request.hr_denoising_strength == 0.25
    assert request.hr_upscaler == "bicubic"
    assert request.controlnet_units == []
    data = response.json()
    assert data["image"].startswith("data:image/png;base64,")
    assert data["recentOutputs"][0]["url"].startswith("data:image/png;base64,")
    assert data["recentOutputs"][0]["path"].endswith("generated.png")
    assert data["recentOutputs"][0]["modelName"] == "model-a"
    assert data["recentOutputs"][0]["generationSettings"]["prompt"] == "private prompt phrase that should not enter application logs"
    assert data["recentOutputs"][0]["generationSettings"]["negativePrompt"] == "blurry"
    assert data["recentOutputs"][0]["generationSettings"]["modelId"] == "model-a"
    assert data["recentOutputs"][0]["generationSettings"]["width"] == 512
    assert data["recentOutputs"][0]["generationSettings"]["height"] == 768
    assert data["recentOutputs"][0]["generationSettings"]["enableHires"] is True
    assert data["status"] == "completed"
    assert data["verificationStatus"] == "verified"
    assert data["job"]["state"] == "completed"
    assert data["seeds"] == [1234]
    assert data["artifacts"][0]["path"].endswith("generated.png")


def test_generate_defers_before_loading_unresident_model_when_gpu_headroom_is_low(tmp_path, monkeypatch):
    from aiwf.services import model_startup

    ctx = _ctx(tmp_path)
    monkeypatch.setattr(model_startup, "_gpu_headroom_status", lambda _ctx, _model: "Only 2.4 GB VRAM is free.")

    response = _client(ctx).post("/api/pro/generate", json={"prompt": "test", "model_id": "model-a"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Only 2.4 GB VRAM is free."
    assert ctx.generation.submitted == []


def test_generate_maps_pipeline_backend_to_request_when_dual_runtime(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.flags = ctx.flags.model_copy(update={"inference_backend": "dual"})
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
            "mode": "image",
            "model_id": "model-a",
            "pipelineBackend": "sdcpp",
        },
    )

    assert response.status_code == 200
    assert ctx.generation.submitted[0].pipeline_backend == "sdcpp"


def test_generate_maps_dual_pipeline_backend_to_request_when_dual_runtime(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.flags = ctx.flags.model_copy(update={"inference_backend": "dual"})
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
            "mode": "image",
            "model_id": "model-a",
            "pipelineBackend": "dual",
        },
    )

    assert response.status_code == 200
    assert ctx.generation.submitted[0].pipeline_backend == "sdcpp"


def test_generate_rejects_sdcpp_pipeline_backend_without_cpp_runtime(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
            "mode": "image",
            "model_id": "model-a",
            "pipelineBackend": "sdcpp",
        },
    )

    assert response.status_code == 409
    assert "dual (or sdcpp)" in response.json()["detail"]
    assert ctx.generation.submitted == []


def test_generate_rejects_dual_pipeline_backend_without_cpp_runtime(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
            "mode": "image",
            "model_id": "model-a",
            "pipelineBackend": "dual",
        },
    )

    assert response.status_code == 409
    assert "dual (or sdcpp)" in response.json()["detail"]
    assert ctx.generation.submitted == []


def test_generate_maps_controlnet_unit(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    source = Image.new("RGB", (8, 8), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    control_image = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
            "mode": "image",
            "model_id": "model-a",
            "controlnet_units": [
                {
                    "enabled": True,
                    "model": "control-a",
                    "module": "canny",
                    "image": control_image,
                    "weight": 0.75,
                    "guidance_start": 0.1,
                    "guidance_end": 0.8,
                    "processor_res": 768,
                }
            ],
        },
    )

    assert response.status_code == 200
    request = ctx.generation.submitted[0]
    assert len(request.controlnet_units) == 1
    unit = request.controlnet_units[0]
    assert unit.enabled is True
    assert unit.model == "control-a"
    assert unit.module == "canny"
    assert unit.image == control_image
    assert unit.weight == 0.75
    assert unit.guidance_start == 0.1
    assert unit.guidance_end == 0.8
    assert unit.processor_res == 768


def test_generate_inpaint_sends_init_and_mask_images(tmp_path):
    ctx = _ctx(tmp_path)
    inpaint = Checkpoint(
        id="sd15-inpaint",
        title="SD 1.5 Inpaint",
        filename="realisticVisionV60-inpainting15.safetensors",
        path=str(tmp_path / "models" / "Stable-diffusion" / "realisticVisionV60-inpainting15.safetensors"),
        architecture="inpaint",
    )
    Path(inpaint.path).parent.mkdir(parents=True, exist_ok=True)
    Path(inpaint.path).write_bytes(b"fake inpaint checkpoint fixture")
    ctx.generation.list_checkpoints = lambda: [inpaint]
    client = _client(ctx)
    source = Image.new("RGB", (8, 8), "red")
    mask = Image.new("L", (8, 8), 255)
    source_buffer = io.BytesIO()
    mask_buffer = io.BytesIO()
    source.save(source_buffer, format="PNG")
    mask.save(mask_buffer, format="PNG")

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "repair the jacket",
            "mode": "inpaint",
            "model_id": "sd15-inpaint",
            "init_image_data_url": f"data:image/png;base64,{base64.b64encode(source_buffer.getvalue()).decode('ascii')}",
            "mask_image_data_url": f"data:image/png;base64,{base64.b64encode(mask_buffer.getvalue()).decode('ascii')}",
            "denoising_strength": 0.55,
            "mask_blur": 6,
        },
    )

    assert response.status_code == 200
    request = ctx.generation.submitted[0]
    assert request.mode == GenerationMode.INPAINT
    assert request.checkpoint_id == "sd15-inpaint"
    assert request.denoising_strength == 0.55
    assert request.mask_blur == 6
    kwargs = ctx.generation.submitted_kwargs[0]
    assert len(kwargs["init_images"]) == 1
    assert len(kwargs["mask_images"]) == 1
    assert kwargs["init_images"][0].mode == "RGB"
    assert kwargs["mask_images"][0].mode == "L"


def test_generate_rejects_blocked_selectable_checkpoint(tmp_path):
    ctx = _ctx(tmp_path)
    blocked = Checkpoint(
        id="blocked-flux",
        title="Blocked Flux",
        filename="fluxedUpFluxNSFW_110FP8.safetensors",
        path=str(tmp_path / "models" / "flux" / "UNet" / "fluxedUpFluxNSFW_110FP8.safetensors"),
        architecture="flux",
    )
    ctx.generation.list_checkpoints = lambda: [blocked, *_Generation(ctx.generation.output_dir).list_checkpoints()]
    client = _client(ctx)

    response = client.post("/api/pro/generate", json={"prompt": "cat", "mode": "image", "model_id": "blocked-flux"})

    assert response.status_code == 422
    assert ctx.generation.submitted == []
    detail = response.json()["detail"]
    assert detail["status"] == "broken-runtime"
    assert "checkpoint keys do not match" in detail["reason"]


def test_generate_rejects_runtime_blocked_qwen_nunchaku_checkpoint(tmp_path):
    ctx = _ctx(tmp_path)
    blocked = Checkpoint(
        id="qwen-nunchaku",
        title="Qwen Nunchaku Lightning",
        filename="svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors",
        path=str(tmp_path / "models" / "qwen-image" / "Nunchaku" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"),
        architecture="qwen_image_nunchaku",
    )
    ctx.generation.list_checkpoints = lambda: [blocked, *_Generation(ctx.generation.output_dir).list_checkpoints()]
    client = _client(ctx)

    response = client.post("/api/pro/generate", json={"prompt": "cat", "mode": "image", "model_id": "qwen-nunchaku"})

    assert response.status_code == 422
    assert ctx.generation.submitted == []
    detail = response.json()["detail"]
    assert detail["status"] == "blocked-runtime"
    assert "generation remains blocked" in detail["reason"].lower()


def test_generate_rejects_unbounded_batch(tmp_path):
    client = _client(_ctx(tmp_path))

    response = client.post(
        "/api/pro/generate",
        json={"prompt": "cat", "batchSize": 3, "batchCount": 2},
    )

    assert response.status_code == 422


def test_generate_image_failure_returns_failure_metadata(tmp_path):
    ctx = _ctx(tmp_path)
    failure_index = tmp_path / "outputs" / "failures" / "index.jsonl"
    failure_index.parent.mkdir(parents=True)
    failure_index.write_text(json.dumps({"status": "failed", "error": {"message": "image failed"}}) + "\n", encoding="utf-8")

    def fail_submit(request: GenerationRequest, **_kwargs):
        job = JobRecord(request=request, state=JobState.FAILED, error="image failed")
        ctx.generation._recent.insert(0, job)
        raise RuntimeError("image failed")

    ctx.generation.submit = fail_submit
    client = _client(ctx)

    response = client.post("/api/pro/generate", json={"prompt": "cat", "mode": "image"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "image failed"
    assert Path(detail["failureLogPath"]).name == "index.jsonl"
    assert detail["job"]["state"] == "failed"
    assert detail["job"]["error"] == "image failed"


def test_generate_reconciles_image_route_residency_after_checkpoint_switch(tmp_path, monkeypatch):
    from aiwf.services.route_lifecycle import select_route

    ctx = _ctx(tmp_path)
    model_root = tmp_path / "models"
    checkpoints = []
    for model_id in ("model-a", "model-b"):
        path = model_root / f"{model_id}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(model_id.encode("ascii"))
        checkpoints.append(Checkpoint(
            id=model_id, title=model_id.upper(), filename=path.name,
            path=str(path), architecture="sdxl",
        ))
    ctx.generation.list_checkpoints = lambda: checkpoints
    resident_id = {"value": "model-a"}
    ctx.generation.backend.is_checkpoint_loaded = lambda model_id: model_id == resident_id["value"]
    select_route(
        ctx, route="image.txt2img", model_id="model-a", setup_ready=True,
        resident=True, detail="Seeded prior resident model.",
    )
    original_submit = ctx.generation.submit

    def submit_switch(request, **kwargs):
        resident_id["value"] = request.checkpoint_id
        return original_submit(request, **kwargs)

    ctx.generation.submit = submit_switch
    monkeypatch.setattr("aiwf.services.model_startup._gpu_headroom_status", lambda _ctx, _model: None)
    client = _client(ctx)

    response = client.post("/api/pro/generate", json={"prompt": "cat", "mode": "image", "model_id": "model-b"})

    assert response.status_code == 200
    runtime = client.get("/api/pro/runtime").json()
    image_route = next(item for item in runtime["routeLifecycle"] if item["route"] == "image.txt2img")
    assert image_route["modelId"] == "model-b"
    assert image_route["resident"] is True
    assert runtime["modelLoad"]["status"] == "loaded"
    assert runtime["modelLoad"]["modelId"] == "model-b"


def test_generate_does_not_claim_image_residency_when_backend_cannot_confirm_switch(tmp_path, monkeypatch):
    from aiwf.services.route_lifecycle import select_route

    ctx = _ctx(tmp_path)
    model_root = tmp_path / "models"
    checkpoints = []
    for model_id in ("model-a", "model-b"):
        path = model_root / f"{model_id}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(model_id.encode("ascii"))
        checkpoints.append(Checkpoint(
            id=model_id, title=model_id.upper(), filename=path.name,
            path=str(path), architecture="sdxl",
        ))
    ctx.generation.list_checkpoints = lambda: checkpoints
    ctx.generation.backend.is_checkpoint_loaded = lambda _model_id: False
    select_route(
        ctx, route="image.txt2img", model_id="model-a", setup_ready=True,
        resident=True, detail="Seeded prior resident model.",
    )
    monkeypatch.setattr("aiwf.services.model_startup._gpu_headroom_status", lambda _ctx, _model: None)
    client = _client(ctx)

    response = client.post("/api/pro/generate", json={"prompt": "cat", "mode": "image", "model_id": "model-b"})

    assert response.status_code == 200
    runtime = client.get("/api/pro/runtime").json()
    image_route = next(item for item in runtime["routeLifecycle"] if item["route"] == "image.txt2img")
    assert image_route["modelId"] == "model-b"
    assert image_route["resident"] is False
    assert runtime["modelLoad"]["status"] == "not-ready"
    assert runtime["modelLoad"]["modelId"] == "model-b"


def test_generate_maps_sana_video_payload_and_serves_output(tmp_path):
    ctx = _ctx(tmp_path)
    _seed_sana_video_snapshot(ctx.sana_video.default_model_path())
    assert str(ctx.sana_video.default_model_path().resolve()) in pro_api._video_model_ids(ctx)
    client = _client(ctx)
    source = Image.new("RGB", (8, 8), "red")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    source_data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "slow camera move",
            "mode": "video",
            "model_id": str(ctx.sana_video.default_model_path()),
            "steps": 2,
            "cfgScale": 6.0,
            "width": 832,
            "height": 480,
            "frames": 9,
            "fps": 8,
            "seed": 99,
            "sana_quantization": "fp8_layerwise",
            "sana_vae_tiling": "auto",
            "offload_text_encoder_after_encode": True,
            "use_sage_attention": False,
            "source_image_data_url": source_data_url,
            "source_image_name": "first-frame.png",
        },
    )

    assert response.status_code == 200, response.text
    request = ctx.sana_video.submitted[0]
    assert request.prompt == "slow camera move"
    assert request.width == 832
    assert request.height == 480
    assert request.frames == 9
    assert request.fps == 8
    assert request.steps == 2
    assert request.quantization == "fp8_layerwise"
    assert request.vae_tiling == "auto"
    assert request.use_sage_attention is False
    assert request.source_image_path is not None
    assert request.wants_image_to_video is True
    assert Path(request.source_image_path).is_file()

    data = response.json()
    assert data["output"]["mode"] == "video"
    assert data["output"]["url"].startswith("/api/pro/outputs/sana-videos/generated.mp4")
    assert data["progress"][-1]["stage"] == "done"
    assert data["timings"]["inference"] == 0.1
    assert data["receiptPath"].endswith("sana_video_latest.json")

    asset_response = client.get(data["output"]["url"])
    assert asset_response.status_code == 200
    assert asset_response.content == b"video"


def test_generate_sana_video_allows_text_to_video_without_source_image(tmp_path):
    ctx = _ctx(tmp_path)
    _seed_sana_video_snapshot(ctx.sana_video.default_model_path())
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "slow camera move",
            "mode": "video",
            "model_id": str(ctx.sana_video.default_model_path()),
            "width": 832,
            "height": 480,
            "frames": 9,
        },
    )

    assert response.status_code == 200
    request = ctx.sana_video.submitted[0]
    assert request.source_image_path is None
    assert request.wants_image_to_video is False
    assert response.json()["output"]["mode"] == "video"


@pytest.mark.parametrize(("confirm_unload", "expected_resident"), [(True, False), (False, None)])
def test_sana_audio_generation_reconciles_prepared_pipeline_residency(
    tmp_path, confirm_unload: bool, expected_resident: bool | None,
):
    ctx = _ctx(tmp_path)
    ctx.audio = _AudioStub(ctx.flags.output_dir)
    ctx.audio.installed = True
    ctx.audio.installed_variants.add("large_44k_v2")
    ctx.settings.last_video_audio_model_id = "mmaudio:large_44k_v2"
    _seed_sana_video_snapshot(ctx.sana_video.default_model_path())
    ctx.sana_video.confirm_unload = confirm_unload
    client = _client(ctx)
    model_path = str(ctx.sana_video.default_model_path())

    prepared = client.post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "prepare Sana", "checkpointId": model_path},
    )
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["loaded"] is True
    assert prepared.json()["routeLifecycle"]["resident"] is True

    generated = client.post(
        "/api/pro/generate",
        json={
            "prompt": "ambient scene",
            "mode": "video",
            "checkpointId": model_path,
            "frames": 9,
            "generateAudio": True,
        },
    )
    assert generated.status_code == 200, generated.text
    assert ctx.sana_video.submitted[-1].generate_audio is True
    assert ctx.sana_video.submitted[-1].audio_model_id == "mmaudio:large_44k_v2"

    lifecycle = client.get("/api/pro/runtime").json()["routeLifecycle"]
    route = next(item for item in lifecycle if item["route"] == "video.sana.480p")
    assert route["status"] == "completed"
    assert route["resident"] is expected_resident


@pytest.mark.parametrize("failure_stage", ["postprocess", "output-validation"])
def test_sana_audio_failure_reconciles_unloaded_pipeline_residency(tmp_path, monkeypatch, failure_stage):
    from fastapi import HTTPException

    ctx = _ctx(tmp_path)
    ctx.audio = _AudioStub(ctx.flags.output_dir)
    ctx.audio.installed = True
    _seed_sana_video_snapshot(ctx.sana_video.default_model_path())
    client = _client(ctx)
    model_path = str(ctx.sana_video.default_model_path())
    prepared = client.post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "prepare Sana", "checkpointId": model_path},
    )
    assert prepared.status_code == 200
    if failure_stage == "postprocess":
        def fail_after_unload(request, *, on_progress=None):
            ctx.sana_video.unload()
            raise RuntimeError("audio post-processing failed")

        ctx.sana_video.generate = fail_after_unload
    else:
        monkeypatch.setattr(
            pro_api, "_sana_video_output_payload",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(HTTPException(status_code=500, detail="bad output")),
        )

    response = client.post(
        "/api/pro/generate",
        json={"prompt": "ambient scene", "mode": "video", "checkpointId": model_path, "frames": 9, "generateAudio": True},
    )
    assert response.status_code == 500
    lifecycle = client.get("/api/pro/runtime").json()["routeLifecycle"]
    route = next(item for item in lifecycle if item["route"] == "video.sana.480p")
    assert route["status"] == "failed"
    assert route["resident"] is False


def test_sana_generation_fails_closed_when_selected_snapshot_is_missing(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from aiwf.services import pipeline_preflight
    from aiwf.services.pipeline_preflight import PipelineCheckItem, PipelinePreflightResult

    ctx = _ctx(tmp_path)
    model_path = ctx.sana_video.default_model_path("480p")
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_sana_video_pipeline",
        lambda *_args, **_kwargs: PipelinePreflightResult(
            pipeline="Sana Video 480p",
            ok=False,
            items=(PipelineCheckItem("local model snapshot", False, "Selected snapshot is incomplete", model_path),),
        ),
    )

    with pytest.raises(HTTPException) as error:
        pro_api._generate_sana_video_response(
            ctx,
            pro_api.ProGeneratePayload(
                prompt="test", mode="video", checkpoint_id=str(model_path), sana_model_variant="480p",
            ),
        )

    assert error.value.status_code == 422
    assert "Selected snapshot is incomplete" in str(error.value.detail)
    assert ctx.sana_video.submitted == []
    lifecycle = _client(ctx).get("/api/pro/runtime").json()["routeLifecycle"]
    route = next(item for item in lifecycle if item["route"] == "video.sana.480p")
    assert route["status"] == "needs-setup"
    assert route["modelId"] == str(model_path)
    assert len(route["supportRevision"]) == 16


def test_pro_ltx_engine_installer_route_starts_fixed_setup_and_exposes_status(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    install = {"status": "started", "pid": 987, "logPath": str(tmp_path / "ltx.log"), "message": "started"}
    status = {"status": "running", "running": True, "pid": 987, "logPath": install["logPath"]}
    monkeypatch.setattr(pro_api, "start_ltx_engine_install", lambda: install)
    monkeypatch.setattr(pro_api, "ltx_engine_install_status", lambda: status)

    client = _client(ctx)
    started = client.post("/api/pro/engines/ltx/install")
    current = client.get("/api/pro/engines/ltx/install-status")

    assert started.status_code == 200
    assert started.json() == install
    from aiwf.services.model_startup import pro_model_load_lock

    operation_lock = pro_model_load_lock(ctx)
    assert operation_lock.acquire(blocking=False), "LTX setup must release the shared operation lock after starting"
    operation_lock.release()
    assert current.status_code == 200
    assert current.json()["running"] is True
    assert current.json()["logPath"] == install["logPath"]


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("/api/pro/models/prepare", {"mode": "video", "prompt": "selection prepare", "checkpointId": "ltx:distilled"}),
        ("/api/pro/generate", {"mode": "video", "prompt": "test", "checkpointId": "ltx:distilled"}),
    ],
)
def test_ltx_prepare_and_generate_reject_while_engine_install_is_running(tmp_path, monkeypatch, endpoint, payload):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(
        pro_api,
        "ltx_engine_install_status",
        lambda: {"status": "running", "running": True},
    )
    monkeypatch.setattr(
        pro_api,
        "_prepare_pro_video_route",
        lambda *_args: pytest.fail("LTX route preparation must not run during worker setup"),
    )
    monkeypatch.setattr(
        pro_api,
        "_generate_ltx_video_response",
        lambda *_args: pytest.fail("LTX generation must not run during worker setup"),
    )

    response = _client(ctx).post(endpoint, json=payload)

    assert response.status_code == 409
    assert "LTX engine setup is in progress" in response.json()["detail"]


@pytest.mark.parametrize("active_operation", ["prepare", "generate"])
def test_ltx_engine_installer_rejects_while_ltx_operation_holds_model_lock(tmp_path, monkeypatch, active_operation):
    ctx = _ctx(tmp_path)
    entered_operation = threading.Event()
    finish_operation = threading.Event()
    installer_calls = []

    def blocked_operation(*_args):
        entered_operation.set()
        assert finish_operation.wait(timeout=5)
        return {"ready": True, "loaded": False, "routeLifecycle": {"route": "video.ltx.distilled"}}

    monkeypatch.setattr(pro_api, "start_ltx_engine_install", lambda: installer_calls.append(True) or {"status": "started"})
    monkeypatch.setattr(pro_api, "_prepare_pro_video_route", blocked_operation)
    monkeypatch.setattr(pro_api, "_generate_ltx_video_response", blocked_operation)
    monkeypatch.setattr(pro_api, "_assert_video_route_checkpoint", lambda *_args, **_kwargs: None)
    client = _client(ctx)

    with ThreadPoolExecutor(max_workers=1) as executor:
        operation = executor.submit(
            client.post,
            "/api/pro/models/prepare" if active_operation == "prepare" else "/api/pro/generate",
            json=(
                {"mode": "video", "prompt": "selection prepare", "checkpointId": "ltx:distilled"}
                if active_operation == "prepare"
                else {"mode": "video", "prompt": "test", "checkpointId": "ltx:distilled"}
            ),
        )
        assert entered_operation.wait(timeout=5)
        install = client.post("/api/pro/engines/ltx/install")
        finish_operation.set()
        completed = operation.result(timeout=5)

    assert install.status_code == 409
    assert installer_calls == []
    assert completed.status_code == 200


def test_pro_qwen_nunchaku_installer_route_exposes_setup_and_runtime_status(tmp_path, monkeypatch):
    from aiwf.services.qwen_nunchaku import QwenNunchakuService

    ctx = _ctx(tmp_path)
    install = {"status": "started", "pid": 654, "logPath": str(tmp_path / "qwen.log"), "message": "started"}
    status = {"status": "finished", "running": False, "exitCode": 0, "logPath": install["logPath"]}
    install_roots = []
    status_roots = []
    monkeypatch.setattr(pro_api, "start_qwen_nunchaku_engine_install", lambda **kwargs: install_roots.append(kwargs) or install)
    monkeypatch.setattr(pro_api, "qwen_nunchaku_engine_install_status", lambda root: status_roots.append(root) or status)
    monkeypatch.setattr(
        QwenNunchakuService,
        "status",
        lambda self, *_args: SimpleNamespace(ready=False, messages=("Qwen base model missing",)),
    )

    client = _client(ctx)
    started = client.post("/api/pro/engines/qwen_nunchaku/install")
    current = client.get("/api/pro/engines/qwen_nunchaku/install-status")

    assert started.status_code == 200
    assert install_roots == [{"data_root": ctx.flags.data_dir}]
    assert started.json() == install
    assert current.status_code == 200
    assert current.json()["runtimeReady"] is False
    assert current.json()["runtimeMessages"] == ["Qwen base model missing"]
    assert status_roots == [ctx.flags.data_dir, ctx.flags.data_dir]


def test_pro_qwen_nunchaku_installer_rejects_during_active_model_operation(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    installer_calls = []
    monkeypatch.setattr(pro_api, "start_qwen_nunchaku_engine_install", lambda **kwargs: installer_calls.append(kwargs))
    from aiwf.services.model_startup import pro_model_load_lock

    operation_lock = pro_model_load_lock(ctx)
    assert operation_lock.acquire(blocking=False)
    try:
        response = _client(ctx).post("/api/pro/engines/qwen_nunchaku/install")
    finally:
        operation_lock.release()

    assert response.status_code == 409
    assert installer_calls == []


@pytest.mark.parametrize("active_state", ["image-load", "route-prepare", "image-generation", "video-generation", "workflow", "installer"])
def test_pro_qwen_nunchaku_installer_rejects_all_active_runtime_states(tmp_path, monkeypatch, active_state):
    ctx = _ctx(tmp_path)
    installer_calls = []
    monkeypatch.setattr(pro_api, "start_qwen_nunchaku_engine_install", lambda **kwargs: installer_calls.append(kwargs))
    if active_state == "image-load":
        ctx._pro_model_load_state = {"status": "loading"}
    elif active_state == "route-prepare":
        ctx._pro_route_prepare_state = {"status": "preparing"}
    elif active_state == "image-generation":
        monkeypatch.setattr(pro_api, "_image_generation_running", lambda *_args: True)
    elif active_state == "video-generation":
        monkeypatch.setattr(pro_api, "_pro_video_job_running", lambda *_args: True)
    elif active_state == "workflow":
        monkeypatch.setattr(pro_api, "_pro_workflow_runs_active", lambda *_args: True)
    else:
        monkeypatch.setattr(
            pro_api,
            "qwen_nunchaku_engine_install_status",
            lambda _root: {"status": "running", "running": True},
        )

    response = _client(ctx).post("/api/pro/engines/qwen_nunchaku/install")

    assert response.status_code == 409
    assert installer_calls == []


@pytest.mark.parametrize("endpoint", ["/api/pro/models/load", "/api/pro/generate"])
def test_qwen_nunchaku_routes_reject_while_engine_setup_is_running(tmp_path, monkeypatch, endpoint):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(
        pro_api,
        "qwen_nunchaku_engine_install_status",
        lambda _root: {"status": "running", "running": True},
    )
    monkeypatch.setattr(pro_api, "_checkpoint_engine_id", lambda *_args: "qwen_nunchaku")
    monkeypatch.setattr(pro_api, "_is_checkpoint_id_selectable", lambda *_args: True)
    monkeypatch.setattr(pro_api, "_checkpoint_id_from_payload", lambda *_args: "qwen-nunchaku")
    monkeypatch.setattr(pro_api, "_assert_image_route_checkpoint", lambda *_args: None)
    monkeypatch.setattr(pro_api, "_assert_checkpoint_selectable", lambda *_args: None)
    ctx.generation.backend.can_preload_checkpoint_locally = lambda *_args: True

    payload = {"modelId": "qwen-nunchaku"} if endpoint.endswith("/load") else {"prompt": "test", "checkpointId": "qwen-nunchaku"}
    response = _client(ctx).post(endpoint, json=payload)

    assert response.status_code == 409
    assert "Qwen Nunchaku setup is in progress" in response.json()["detail"]


def test_pro_ltx_engine_installer_refreshes_cached_worker_registry_after_success(tmp_path, monkeypatch):
    from aiwf.services.worker_tenant import WorkerTenantRegistry

    ctx = _ctx(tmp_path)
    (tmp_path / "engines.json").write_text(
        json.dumps({"ltx": {"enabled": False, "venv_dir": "engines/ltx/.venv"}}),
        encoding="utf-8",
    )
    ctx.ltx.registry = WorkerTenantRegistry(tmp_path)
    assert ctx.ltx.registry.status("ltx").enabled is False
    (tmp_path / "engines.json").write_text(
        json.dumps({"ltx": {"enabled": True, "venv_dir": "engines/ltx/.venv"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pro_api,
        "ltx_engine_install_status",
        lambda: {"status": "finished", "running": False, "exitCode": 0, "logPath": "installer.log"},
    )

    response = _client(ctx).get("/api/pro/engines/ltx/install-status")

    assert response.status_code == 200
    assert isinstance(ctx.ltx.registry, WorkerTenantRegistry)
    assert ctx.ltx.registry.repo_root == tmp_path.resolve()
    assert ctx.ltx.registry.status("ltx").enabled is True


def test_video_route_prepare_checks_worker_ltx_readiness_without_claiming_residency(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    support_file = tmp_path / "models" / "ltx" / "checkpoint.safetensors"
    support_file.parent.mkdir(parents=True)
    support_file.write_bytes(b"local test weights")
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_ltx_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            items=[SimpleNamespace(path=support_file)],
            markdown=lambda: "LTX route support passed preflight.",
        ),
    )
    prepare_calls = []
    ctx.ltx.prepare = lambda request: (
        prepare_calls.append(request)
        or {"loaded": False, "resident": None, "detail": "LTX worker loads weights when generation starts."}
    )

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "readiness", "checkpointId": "ltx:distilled"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ready"] is True
    assert payload["loaded"] is False
    assert prepare_calls and prepare_calls[0].pipeline == "distilled"
    route = payload["routeLifecycle"]
    assert route["route"] == "video.ltx.distilled"
    assert route["modelId"] == "ltx:distilled"
    assert route["status"] == "setup-ready"
    assert route["resident"] is None
    assert str(support_file) not in route["supportRevision"]
    assert ctx.ltx.submitted == []


def test_video_route_prepare_loads_diffusers_ltx_2b_and_records_residency(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    support_file = tmp_path / "models" / "ltx" / "ltx2b.safetensors"
    support_file.parent.mkdir(parents=True)
    support_file.write_bytes(b"local test weights")
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_ltx_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            items=[SimpleNamespace(path=support_file)],
            markdown=lambda: "LTX 2B route support passed preflight.",
        ),
    )
    prepared_requests = []
    ctx.ltx.prepare = lambda request: (
        prepared_requests.append(request)
        or {"loaded": True, "resident": True, "detail": "LTX 2B pipeline resident. No generation was run."}
    )
    ctx.ltx.unload = lambda: False

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "selection", "checkpointId": "ltx:diffusers_2b"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ready"] is True
    assert payload["loaded"] is True
    assert prepared_requests and prepared_requests[0].pipeline == "diffusers_2b"
    assert payload["routeLifecycle"]["resident"] is True
    assert payload["routeLifecycle"]["status"] == "loaded"
    assert ctx.ltx.submitted == []


def test_ltx_eviction_without_service_method_fails_closed_for_tracked_residency(tmp_path, monkeypatch):
    from aiwf.services import ltx_diffusers

    ctx = _ctx(tmp_path)
    pro_api.select_route(
        ctx,
        route="video.ltx.diffusers_2b",
        model_id="ltx:diffusers_2b",
        setup_ready=True,
        resident=True,
    )
    ctx.ltx = SimpleNamespace()
    monkeypatch.setattr(ltx_diffusers, "unload_ltx2b_diffusers_cache", lambda: False)

    with pytest.raises(HTTPException) as error:
        pro_api._unload_cached_ltx_model(ctx)

    assert error.value.status_code == 503
    assert "did not confirm model release" in str(error.value.detail)


def test_video_route_prepare_reports_only_backend_confirmed_wan_load_without_generating(tmp_path):
    ctx = _ctx(tmp_path)
    calls = []
    prepare_calls = []
    released_audio = []
    released_ltx = []
    ctx.audio = SimpleNamespace(
        release_cached_model_for_modality_switch=lambda: released_audio.append(True) or True,
    )
    ctx.ltx.unload = lambda: released_ltx.append(True) or True

    def preflight(request, *, image_present=True):
        calls.append((request, image_present))
        return SimpleNamespace(
            ok=True,
            errors=(),
            warnings=(),
            message=lambda: "Wan route support passed preflight.",
        )

    ctx.wan.preflight = preflight
    ctx.wan.prepare = lambda request, *, image_present, preflight: (
        prepare_calls.append((request, image_present, preflight))
        or {
            "loaded": True,
            "resident": True,
            "modelId": request.model_id,
            "detail": "Wan pipeline cached. No generation was run.",
        }
    )

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "wan", "prompt": "selection readiness", "checkpointId": "wan-a.safetensors"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ready"] is True
    assert payload["loaded"] is True
    assert calls
    assert calls[-1][0].model_id == "wan-a.safetensors"
    assert calls[-1][1] is False
    assert prepare_calls and prepare_calls[-1][1] is False
    assert prepare_calls[-1][2].ok is True
    assert released_audio == [True]
    assert released_ltx == [True]
    route = payload["routeLifecycle"]
    assert route["route"] == "video.wan.fast_5b"
    assert route["modelId"] == "wan-a.safetensors"
    assert route["status"] == "loaded"
    assert route["resident"] is True


def test_preflight_support_paths_keep_auto_resolved_wan_assets_in_lifecycle_revision(tmp_path):
    asset = tmp_path / "wan" / "vae.safetensors"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"vae")

    paths = pro_api._preflight_support_paths(SimpleNamespace(
        items=[SimpleNamespace(path=asset), SimpleNamespace(path=None)],
    ))

    assert paths == [str(asset)]


def test_preflight_support_paths_include_resolved_wan_dataclass_assets(tmp_path):
    assets = [tmp_path / name for name in (
        "wan-5b.safetensors", "components", "high.safetensors", "low.safetensors",
        "vae.safetensors", "umt5.safetensors", "high-lora.safetensors", "low-lora.safetensors",
    )]
    paths = pro_api._preflight_support_paths(SimpleNamespace(**dict(zip((
        "model_id", "components_base", "high_noise_model", "low_noise_model",
        "vae", "text_encoder", "high_noise_lora", "low_noise_lora",
    ), map(str, assets)))))

    assert paths == list(map(str, assets))


@pytest.mark.parametrize("operation", ["prepare", "generate"])
def test_wan_route_lifecycle_tracks_resolved_assets_for_basename_selection_and_replacement(
    tmp_path, monkeypatch, operation,
):
    from aiwf.services import route_lifecycle

    assets_root = tmp_path / "resolved-wan-assets"
    components = assets_root / "components"
    components.mkdir(parents=True)
    (components / "model_index.json").write_text("{}", encoding="utf-8")
    tokenizer_dir = components / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_path.write_text("original tokenizer", encoding="utf-8")
    asset_paths = {
        "model_id": assets_root / "wan-high.safetensors",
        "components_base": components,
        "high_noise_model": assets_root / "wan-high.safetensors",
        "low_noise_model": assets_root / "wan-low.safetensors",
        "vae": assets_root / "wan-vae.safetensors",
        "text_encoder": assets_root / "umt5.safetensors",
        "high_noise_lora": assets_root / "wan-high-lora.safetensors",
        "low_noise_lora": assets_root / "wan-low-lora.safetensors",
    }
    for field, path in asset_paths.items():
        if field != "components_base":
            path.write_bytes(field.encode("ascii"))
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    source_image = output_dir / "source.png"
    Image.new("RGB", (2, 2), color="white").save(source_image)
    video_path = output_dir / "wan-output.mp4"
    video_path.write_bytes(b"test video")

    preflight = SimpleNamespace(
        ok=True,
        errors=(),
        warnings=(),
        message=lambda: "Wan route support passed preflight.",
        **{field: str(path) for field, path in asset_paths.items()},
    )
    ctx = _ctx(tmp_path)
    ctx.wan.preflight = lambda *_args, **_kwargs: preflight
    ctx.wan.prepare = lambda *_args, **_kwargs: {
        "loaded": True,
        "resident": True,
        "detail": "Wan route prepared.",
    }

    def generate_and_replace_support(*_args, **_kwargs):
        if operation == "generate":
            tokenizer_path.write_text("replacement during generation", encoding="utf-8")
        return SimpleNamespace(output_path=str(video_path))

    ctx.wan.generate = generate_and_replace_support
    payload = pro_api.ProGeneratePayload(
        prompt="test",
        mode="wan",
        checkpoint_id="wan-high.safetensors",
        wan_runtime_mode="native_high_low",
        high_noise_model_id="wan-high.safetensors",
        low_noise_model_id="wan-low.safetensors",
        vae_id="wan-vae.safetensors",
        text_encoder_path="umt5.safetensors",
        high_noise_lora_id="wan-high-lora.safetensors",
        low_noise_lora_id="wan-low-lora.safetensors",
        source_image_path=str(source_image),
    )

    if operation == "prepare":
        result = pro_api._prepare_pro_video_route(ctx, payload)
        assert result["ready"] is True
        assert result["routeLifecycle"]["modelId"] == "wan-high.safetensors"
    else:
        result = pro_api._generate_wan_video_response(ctx, payload)
        assert result["status"] == "completed"

    route_id = "video.wan.native_high_low"
    tracked_ids = set(route_lifecycle._support_ids_store(ctx)[route_id])
    assert {str(path) for path in asset_paths.values()}.issubset(tracked_ids)
    assert "wan-high.safetensors" in tracked_ids
    if operation == "generate":
        route = next(item for item in route_lifecycle.lifecycle_snapshot(ctx) if item["route"] == route_id)
        assert route["status"] == "needs-setup"
        assert "Support assets changed while this operation was running" in route["detail"]
    else:
        before = next(item for item in route_lifecycle.lifecycle_snapshot(ctx) if item["route"] == route_id)
        monkeypatch.setattr(route_lifecycle, "_SNAPSHOT_SUPPORT_CHECK_SECONDS", 0)
        tokenizer_path.write_text("replacement tokenizer", encoding="utf-8")
        after = next(item for item in route_lifecycle.lifecycle_snapshot(ctx) if item["route"] == route_id)

        assert after["supportRevision"] != before["supportRevision"]
        assert after["status"] == "needs-setup"
        assert after["resident"] is False


def test_civitai_catalog_entries_are_installable_and_link_to_the_exact_version():
    item = SimpleNamespace(
        source="civitai",
        civitai_model_id=193072,
        civitai_version_id=257123,
        repo_id="",
        url="",
        coming_soon=False,
        notes="",
    )

    assert pro_api._catalog_entry_can_download(item) is True
    assert pro_api._catalog_entry_page_url(item) == "https://civitai.com/models/193072?modelVersionId=257123"
    assert pro_api._catalog_entry_hf_url(item) == ""


def test_civitai_catalog_without_resolvable_ids_stays_link_only():
    item = SimpleNamespace(source="civitai", civitai_model_id=None, civitai_version_id=None, coming_soon=False)

    assert pro_api._catalog_entry_can_download(item) is False


def test_sana_prepare_blocks_audio_toggle_when_mmaudio_bundle_is_missing(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    ctx.settings.last_checkpoint_id = "previous-default"
    model_path = ctx.sana_video.default_model_path("720p")
    model_path.mkdir(parents=True)
    ctx.audio = SimpleNamespace(
        _mmaudio_variant_ready=lambda _variant: False,
        _mmaudio_runtime_import_error=lambda: "",
        _mmaudio_root=lambda: tmp_path / "engines" / "mmaudio",
        _mmaudio_clip_hub_cache=lambda: None,
    )
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_sana_video_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            items=[SimpleNamespace(path=model_path)],
            markdown=lambda: "Sana Video 720p route support passed preflight.",
        ),
    )

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "selection warmup", "checkpointId": str(model_path), "generateAudio": True},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ready"] is False
    assert payload["loaded"] is False
    assert "complete MMAudio small_16k model bundle" in payload["routeLifecycle"]["detail"]
    assert ctx.sana_video.prepared == []
    assert ctx.settings.last_checkpoint_id == "previous-default"


def test_sana_prepare_blocks_video_audio_when_mux_tools_are_missing(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    model_path = ctx.sana_video.default_model_path("720p")
    _seed_sana_video_snapshot(model_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio.installed_variants.add("small_16k")
    audio.mux_ready = False
    ctx.audio = audio
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_sana_video_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            items=[SimpleNamespace(path=model_path)],
            markdown=lambda: "Sana Video 720p route support passed preflight.",
        ),
    )

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "selection warmup", "checkpointId": str(model_path), "generateAudio": True},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ready"] is False
    assert "FFmpeg and ffprobe" in payload["routeLifecycle"]["detail"]
    assert ctx.sana_video.prepared == []


def test_video_route_prepare_loads_selected_sana_variant_without_generating(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    saved_defaults = []
    ctx.save_settings = lambda: saved_defaults.append(ctx.settings.last_checkpoint_id)
    model_path = ctx.sana_video.default_model_path("720p")
    model_path.mkdir(parents=True)
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_sana_video_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            items=[SimpleNamespace(path=model_path)],
            markdown=lambda: "Sana Video 720p route support passed preflight.",
        ),
    )

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "selection warmup", "checkpointId": str(model_path)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ready"] is True
    assert payload["loaded"] is True
    assert ctx.sana_video.prepared[0].model_variant == "720p"
    assert ctx.sana_video.submitted == []
    route = payload["routeLifecycle"]
    assert route["route"] == "video.sana.720p"
    assert route["status"] == "loaded"
    assert route["resident"] is True
    assert ctx.settings.last_checkpoint_id == str(model_path)
    assert saved_defaults == [str(model_path)]
    assert payload["startupDefaultSaved"] is True


def test_checkpoint_default_is_not_mutated_when_settings_cannot_be_saved(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.settings.last_checkpoint_id = "previous-model"

    assert pro_api._persist_last_checkpoint_id(ctx, "new-model") is False
    assert ctx.settings.last_checkpoint_id == "previous-model"


def test_checkpoint_default_rolls_back_when_settings_save_fails(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.settings.last_checkpoint_id = "previous-model"
    ctx.save_settings = lambda: (_ for _ in ()).throw(OSError("settings write failed"))

    assert pro_api._persist_last_checkpoint_id(ctx, "new-model") is False
    assert ctx.settings.last_checkpoint_id == "previous-model"


def test_video_route_prepare_keeps_setup_ready_when_sana_warm_load_is_deferred(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight
    from aiwf.services.sana_video import SanaVideoUnavailable

    ctx = _ctx(tmp_path)
    model_path = ctx.sana_video.default_model_path("480p")
    model_path.mkdir(parents=True)
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_sana_video_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            items=[SimpleNamespace(path=model_path)],
            markdown=lambda: "Sana Video setup passed preflight.",
        ),
    )
    ctx.sana_video.prepare = lambda _request: (_ for _ in ()).throw(
        SanaVideoUnavailable("Sana Video preparation deferred: only 1.7 GB VRAM is free.")
    )

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "selection warmup", "checkpointId": str(model_path)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["loaded"] is False
    assert payload["routeLifecycle"]["status"] == "setup-ready"
    assert payload["routeLifecycle"]["resident"] is None
    assert "preparation deferred" in payload["routeLifecycle"]["detail"]


@pytest.mark.parametrize("family", ["wan", "ltx", "sana"])
def test_video_route_prepare_marks_unclassified_load_errors_failed_and_does_not_save_default(
    tmp_path, monkeypatch, family,
):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    ctx.settings.last_checkpoint_id = "previous-default"
    saved = []
    ctx.save_settings = lambda: saved.append(ctx.settings.last_checkpoint_id)
    support = tmp_path / "models" / "video" / "support.safetensors"
    support.parent.mkdir(parents=True)
    support.write_bytes(b"fixture support")
    preflight = SimpleNamespace(
        ok=True,
        items=[SimpleNamespace(path=support)],
        markdown=lambda: "Video setup passed preflight.",
        message=lambda: "Video setup passed preflight.",
        errors=(),
        warnings=(),
    )

    if family == "wan":
        pro_api.select_route(
            ctx, route="video.wan.high_low", model_id="wan-a.safetensors",
            setup_ready=True, resident=True,
        )
        ctx.wan.preflight = lambda *_args, **_kwargs: preflight
        # Model that the failed replacement evicted the prior cached
        # pipeline so the API can safely clear its old residency receipt.
        ctx.wan._backend = SimpleNamespace(has_cached_pipeline=lambda: False)
        ctx.wan.prepare = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected backend failure"))
        payload = {"mode": "wan", "prompt": "readiness", "checkpointId": "wan-a.safetensors"}
    elif family == "ltx":
        monkeypatch.setattr(pipeline_preflight, "preflight_ltx_pipeline", lambda *_args, **_kwargs: preflight)
        ctx.ltx.prepare = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected backend failure"))
        payload = {"mode": "video", "prompt": "readiness", "checkpointId": "ltx:distilled"}
    else:
        pro_api.select_route(
            ctx, route="video.sana.720p", model_id="prior-sana-720p",
            setup_ready=True, resident=True,
        )
        monkeypatch.setattr(pipeline_preflight, "preflight_sana_video_pipeline", lambda *_args, **_kwargs: preflight)
        ctx.sana_video.prepare = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected backend failure"))
        payload = {"mode": "video", "prompt": "readiness", "checkpointId": str(ctx.sana_video.default_model_path("480p"))}

    response = _client(ctx).post("/api/pro/models/prepare", json=payload)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ready"] is False
    assert result["loaded"] is False
    assert result["routeLifecycle"]["status"] == "failed"
    assert result["routeLifecycle"]["resident"] is None
    assert "preparation failed after setup checks" in result["routeLifecycle"]["detail"].lower()
    assert ctx.settings.last_checkpoint_id == "previous-default"
    assert saved == []
    if family in {"wan", "sana"}:
        runtime_routes = {item["route"]: item for item in _client(ctx).get("/api/pro/runtime").json()["routeLifecycle"]}
        previous_route = "video.wan.high_low" if family == "wan" else "video.sana.720p"
        assert runtime_routes[previous_route]["resident"] is False


def test_video_route_prepare_records_missing_ltx_support_without_generation(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    missing_path = tmp_path / "models" / "ltx" / "missing.safetensors"
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_ltx_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=False,
            items=[SimpleNamespace(path=missing_path)],
            markdown=lambda: "LTX checkpoint is missing.",
        ),
    )

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "readiness", "checkpointId": "ltx:distilled"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["loaded"] is False
    assert payload["routeLifecycle"]["status"] == "needs-setup"
    assert payload["routeLifecycle"]["resident"] is None
    assert "LTX checkpoint is missing" in payload["routeLifecycle"]["detail"]
    assert ctx.ltx.submitted == []


def test_switching_from_sana_to_ltx_releases_cached_sana_pipeline(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    released_image_models = []
    pro_api.select_route(ctx, route="image.txt2img", model_id="model-a", setup_ready=True, resident=True)
    monkeypatch.setattr(pro_api, "_runtime_loaded_model", lambda _ctx: {"loaded": True, "id": "model-a"})
    monkeypatch.setattr(pro_api, "_unload_generation_model", lambda _ctx: released_image_models.append("model-a"))
    sana_id = "SANA-Video_2B_480p"
    pro_api.select_route(
        ctx,
        route="video.sana.480p",
        model_id=sana_id,
        setup_ready=True,
        resident=True,
    )
    support_file = tmp_path / "models" / "ltx" / "checkpoint.safetensors"
    support_file.parent.mkdir(parents=True)
    support_file.write_bytes(b"local test weights")
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_ltx_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            items=[SimpleNamespace(path=support_file)],
            markdown=lambda: "LTX route support passed preflight.",
        ),
    )

    response = _client(ctx).post(
        "/api/pro/models/prepare",
        json={"mode": "video", "prompt": "switch route", "checkpointId": "ltx:distilled"},
    )

    assert response.status_code == 200, response.text
    assert released_image_models == ["model-a"]
    assert ctx.sana_video.unloaded is True
    route_states = _client(ctx).get("/api/pro/runtime").json()["routeLifecycle"]
    sana_state = next(item for item in route_states if item["route"] == "video.sana.480p")
    image_state = next(item for item in route_states if item["route"] == "image.txt2img")
    assert sana_state["status"] == "setup-ready"
    assert sana_state["resident"] is False
    assert image_state["resident"] is False


def test_audio_video_unload_callbacks_reconcile_image_route_residency(tmp_path):
    factories = (
        ("sana_video", pro_api._sana_video_service, "_unload_image_models"),
        ("wan", pro_api._wan_service, "_unload_image_models"),
        ("audio", pro_api._audio_service, "unload_image_models"),
    )
    for attribute, factory, callback_name in factories:
        ctx = _ctx(tmp_path / attribute)
        if hasattr(ctx, attribute):
            delattr(ctx, attribute)
        calls = []
        ctx.generation.backend.unload = lambda: calls.append("unload")
        pro_api.select_route(ctx, route="image.txt2img", model_id="model-a", setup_ready=True, resident=True)

        service = factory(ctx)
        callback = getattr(service, callback_name)
        callback()

        assert calls == ["unload"]
        image_state = next(item for item in pro_api.lifecycle_snapshot(ctx) if item["route"] == "image.txt2img")
        assert image_state["resident"] is False


def test_generation_is_rejected_while_video_route_preparation_is_active(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    entered_prepare = threading.Event()
    finish_prepare = threading.Event()

    def blocked_prepare(_ctx, _payload):
        entered_prepare.set()
        assert finish_prepare.wait(timeout=5)
        return {"ready": True, "loaded": False, "routeLifecycle": {"route": "video.wan.fast_5b"}}

    monkeypatch.setattr(pro_api, "_prepare_pro_video_route", blocked_prepare)
    client = _client(ctx)
    with ThreadPoolExecutor(max_workers=1) as executor:
        preparation = executor.submit(
            client.post,
            "/api/pro/models/prepare",
            json={"mode": "video", "prompt": "selection prepare", "checkpointId": "ltx:distilled"},
        )
        assert entered_prepare.wait(timeout=5)
        generation = client.post("/api/pro/generate", json={"prompt": "test", "mode": "txt2img"})
        finish_prepare.set()
        prepared = preparation.result(timeout=5)

    assert generation.status_code == 409
    assert "model operation is already in progress" in generation.json()["detail"]
    assert prepared.status_code == 200
    assert getattr(ctx, "_pro_route_prepare_state", None) is None


def test_route_preparation_is_rejected_after_generation_claims_model_lock(tmp_path, monkeypatch):
    from fastapi import HTTPException

    ctx = _ctx(tmp_path)
    entered_generation = threading.Event()
    finish_generation = threading.Event()

    def blocked_generate(_ctx, _payload):
        entered_generation.set()
        assert finish_generation.wait(timeout=5)
        raise HTTPException(status_code=422, detail="test stopped after the lock assertion")

    monkeypatch.setattr(pro_api, "_assert_requested_pipeline_backend", blocked_generate)
    client = _client(ctx)
    with ThreadPoolExecutor(max_workers=1) as executor:
        generation_future = executor.submit(
            client.post,
            "/api/pro/generate",
            json={"prompt": "test", "mode": "txt2img"},
        )
        assert entered_generation.wait(timeout=5)
        preparation = client.post(
            "/api/pro/models/prepare",
            json={"mode": "video", "prompt": "selection prepare", "checkpointId": "ltx:distilled"},
        )
        finish_generation.set()
        generated = generation_future.result(timeout=5)

    assert preparation.status_code == 409
    assert generated.status_code == 422


def test_enhance_image_runs_restore_and_upscale(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    source = Image.new("RGB", (8, 8), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    response = client.post(
        "/api/pro/enhance/image",
        json={
            "imageDataUrl": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}",
            "restoreEnabled": True,
            "restoreModel": "restore-a",
            "restoreVisibility": 0.8,
            "codeformerWeight": 0.4,
            "upscaleEnabled": True,
            "upscaleModel": "upscale-a",
            "upscaleScale": 2,
            "tileSize": 128,
            "tileOverlap": 16,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["width"] == 16
    assert data["height"] == 16
    assert data["image"].startswith("data:image/png;base64,")
    assert data["url"].startswith("/api/pro/outputs/")
    assert Path(data["outputPath"]).is_file()
    call = ctx.enhance.calls[0]
    assert call["restore"].model_id == "restore-a"
    assert call["restore"].visibility == 0.8
    assert call["restore"].codeformer_weight == 0.4
    assert call["upscale"].model_id == "upscale-a"
    assert call["upscale"].scale == 2
    assert call["upscale"].tile_size == 128
    assert call["upscale"].tile_overlap == 16


def test_enhance_inventory_reports_install_state_and_explicit_install(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    missing_path = ctx.enhance.model_paths["restore-a"]
    missing_path.unlink()

    inventory = client.get("/api/pro/enhance/models")
    assert inventory.status_code == 200
    assert ctx.enhance.catalog_invalidations == 1
    restore = next(item for item in inventory.json()["models"] if item["id"] == "restore-a")
    assert restore["installed"] is False
    assert restore["installAvailable"] is True
    before_install = client.get("/api/pro/capabilities")
    enhance_before_install = next(item for item in before_install.json()["tools"] if item["id"] == "enhance")
    assert enhance_before_install["status"] == "needs-assets"

    installed = client.post("/api/pro/enhance/models/restore-a/install")
    assert installed.status_code == 200
    assert installed.json()["model"]["installed"] is True
    assert missing_path.is_file()
    after_install = client.get("/api/pro/capabilities")
    enhance_after_install = next(item for item in after_install.json()["tools"] if item["id"] == "enhance")
    assert enhance_after_install["status"] == "ready"


def test_enhance_generation_requires_explicit_setup_for_missing_weights(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    ctx.enhance.model_paths["restore-a"].unlink()
    source = Image.new("RGB", (8, 8), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    response = client.post(
        "/api/pro/enhance/image",
        json={
            "imageDataUrl": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}",
            "restoreEnabled": True,
            "restoreModel": "restore-a",
        },
    )

    assert response.status_code == 409
    assert "Install it from Enhance" in response.json()["detail"]
    assert ctx.enhance.calls == []


def test_enhance_image_rejects_while_model_operation_lock_is_held(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    source = Image.new("RGB", (8, 8), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(ctx)
    assert lock.acquire(blocking=False)
    try:
        response = client.post(
            "/api/pro/enhance/image",
            json={
                "imageDataUrl": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}",
                "restoreEnabled": False,
                "upscaleEnabled": True,
                "upscaleModel": "upscale-a",
            },
        )
    finally:
        lock.release()

    assert response.status_code == 409
    assert ctx.enhance.calls == []


@pytest.mark.parametrize("endpoint", ["/api/pro/faceswap", "/api/pro/vsr/image"])
def test_auxiliary_gpu_image_routes_reject_while_model_operation_lock_is_held(tmp_path, endpoint):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    source = Image.new("RGB", (8, 8), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    payload = (
        {"targetImageDataUrl": data_url, "sourceImageDataUrl": data_url}
        if endpoint.endswith("faceswap")
        else {"imageDataUrl": data_url}
    )
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(ctx)
    assert lock.acquire(blocking=False)
    try:
        response = client.post(endpoint, json=payload)
    finally:
        lock.release()

    assert response.status_code == 409


def test_video_lab_route_rejects_while_model_operation_lock_is_held(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    src = tmp_path / "input.mp4"
    src.write_bytes(b"video")
    called = []
    monkeypatch.setattr(pro_api, "_video_lab_resolve_source", lambda *_args: src)
    monkeypatch.setattr(pro_api, "_video_lab_run_vsr", lambda *_args: called.append(True))
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(ctx)
    assert lock.acquire(blocking=False)
    try:
        response = client.post("/api/pro/video-lab/run", json={"op": "vsr"})
    finally:
        lock.release()

    assert response.status_code == 409
    assert called == []


@pytest.mark.parametrize("op,runner_name", [("vsr", "_video_lab_run_vsr"), ("rife", "_video_lab_run_rife")])
def test_video_lab_gpu_routes_release_audio_before_operation(tmp_path, monkeypatch, op, runner_name):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    src = tmp_path / "input.mp4"
    src.write_bytes(b"video")
    events = []
    ctx.audio = SimpleNamespace(release_cached_model_for_modality_switch=lambda: events.append("release") or True)
    monkeypatch.setattr(pro_api, "_video_lab_resolve_source", lambda *_args: src)
    monkeypatch.setattr(pro_api, runner_name, lambda *_args: events.append("operation") or {"status": "ok"})

    response = client.post("/api/pro/video-lab/run", json={"op": op})

    assert response.status_code == 200
    assert events == ["release", "operation"]


@pytest.mark.parametrize("op,runner_name", [("vsr", "_video_lab_run_vsr"), ("rife", "_video_lab_run_rife")])
def test_video_lab_gpu_routes_stop_when_audio_release_is_blocked(tmp_path, monkeypatch, op, runner_name):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    src = tmp_path / "input.mp4"
    src.write_bytes(b"video")
    called = []
    ctx.audio = SimpleNamespace(release_cached_model_for_modality_switch=lambda: False)
    monkeypatch.setattr(pro_api, "_video_lab_resolve_source", lambda *_args: src)
    monkeypatch.setattr(pro_api, runner_name, lambda *_args: called.append(True))

    response = client.post("/api/pro/video-lab/run", json={"op": op})

    assert response.status_code == 409
    assert "Audio still owns or is using its model" in response.json()["detail"]
    assert called == []


@pytest.mark.parametrize("op,runner_name", [("audio", "_video_lab_run_audio"), ("extend", "_video_lab_run_extend")])
def test_video_lab_non_gpu_routes_do_not_evict_musicgen(tmp_path, monkeypatch, op, runner_name):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    src = tmp_path / "input.mp4"
    src.write_bytes(b"video")
    ctx.audio = SimpleNamespace(release_cached_model_for_modality_switch=lambda: pytest.fail("non-GPU route should not evict MusicGen"))
    monkeypatch.setattr(pro_api, "_video_lab_resolve_source", lambda *_args: src)
    monkeypatch.setattr(pro_api, runner_name, lambda *_args: {"status": "ok"})

    response = client.post("/api/pro/video-lab/run", json={"op": op})

    assert response.status_code == 200


def test_generate_response_maps_vram_headroom_deferral_to_conflict():
    job = SimpleNamespace(result=None, error="Generation deferred: 2.4 GB VRAM is free; this model needs an estimated 10.5 GB headroom.")

    with pytest.raises(HTTPException) as error:
        pro_api._generate_response(job)

    assert error.value.status_code == 409
    assert "2.4 GB VRAM is free" in error.value.detail


def test_vsr_image_endpoint_returns_processed_image(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    source = Image.new("RGB", (8, 8), "white")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    response = client.post(
        "/api/pro/vsr/image",
        json={
            "imageDataUrl": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}",
            "scale": 2,
            "mode": 0,
            "effect": "SuperRes",
            "strength": 0.5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["width"] == 16
    assert data["height"] == 16
    assert data["image"].startswith("data:image/png;base64,")
    assert data["url"].startswith("/api/pro/outputs/")
    assert Path(data["outputPath"]).is_file()
    assert ctx.vsr.calls[0].scale == 2
    assert ctx.vsr.calls[0].mode == 0
    assert ctx.vsr.calls[0].effect == "SuperRes"


def test_generate_image_rejects_sana_video_model(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
            "mode": "image",
            "model_id": str(ctx.sana_video.default_model_path()),
        },
    )

    assert response.status_code == 422
    assert ctx.generation.submitted == []
    detail = response.json()["detail"]
    assert detail["message"] == "Video models are only available from the Video tab."


def test_generate_image_rejects_wan_video_model(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
            "mode": "image",
            "model_id": "wan-a.safetensors",
        },
    )

    assert response.status_code == 422
    assert ctx.generation.submitted == []
    assert response.json()["detail"]["message"] == "Video models are only available from the Video tab."


def test_generate_video_rejects_image_checkpoint(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "slow camera move",
            "mode": "video",
            "model_id": "model-a",
            "width": 832,
            "height": 480,
            "frames": 9,
        },
    )

    assert response.status_code == 422
    assert ctx.sana_video.submitted == []
    detail = response.json()["detail"]
    assert "Choose a Wan or Sana Video model" in detail["message"]


def test_generate_sana_video_rejects_source_path_outside_outputs(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 8), "red").save(outside)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "slow camera move",
            "mode": "video",
            "source_image_path": str(outside),
        },
    )

    assert response.status_code == 422
    assert ctx.sana_video.submitted == []


def test_generate_sana_video_failure_returns_receipt_path_and_runtime_error(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.sana_video = _FailingSanaVideo(tmp_path / "outputs")
    _seed_sana_video_snapshot(ctx.sana_video.default_model_path())
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={"prompt": "slow camera move", "mode": "video", "width": 832, "height": 480, "frames": 9},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "decode failed"
    assert detail["receiptPath"].endswith("sana_video_latest.json")
    assert detail["job"]["state"] == "failed"

    runtime = client.get("/api/pro/runtime").json()
    assert runtime["job"]["state"] == "failed"
    assert runtime["job"]["error"] == "decode failed"


def test_sana_and_wan_output_payloads_reject_missing_video_files(tmp_path):
    ctx = _ctx(tmp_path)
    payload = pro_api.ProGeneratePayload(prompt="test output validation")

    for builder, family in (
        (pro_api._sana_video_output_payload, "Sana Video"),
        (pro_api._wan_video_output_payload, "Wan"),
    ):
        try:
            builder(ctx, SimpleNamespace(output_path=""), payload)
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 500
            assert family in str(getattr(exc, "detail", exc))
        else:
            raise AssertionError(f"{family} output validation accepted a missing output file")


def test_wan_generation_fails_closed_when_preflight_fails(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    calls = []
    ctx.wan = SimpleNamespace(
        preflight=lambda *_args, **_kwargs: SimpleNamespace(
            ok=False, errors=("Missing Wan VAE",), warnings=(),
            message=lambda: "Wan model check failed:\n- Missing Wan VAE",
        ),
        generate=lambda *_args, **_kwargs: calls.append("generate"),
    )
    monkeypatch.setattr(
        pro_api, "_wan_video_request_from_payload",
        lambda _ctx, _payload: SimpleNamespace(
            runtime_mode="fast_5b", model_id="wan-a", steps=2,
            vae_id=None, text_encoder_path=None, high_noise_model_id=None, low_noise_model_id=None,
        ),
    )
    monkeypatch.setattr(pro_api, "_assert_video_route_checkpoint", lambda *_args, **_kwargs: None)

    response = _client(ctx).post(
        "/api/pro/generate",
        json={"prompt": "test", "mode": "wan", "model_id": "wan-a"},
    )

    assert response.status_code == 422
    assert "Missing Wan VAE" in response.json()["detail"]
    assert calls == []
    lifecycle = _client(ctx).get("/api/pro/runtime").json()["routeLifecycle"]
    assert next(item for item in lifecycle if item["route"] == "video.wan.fast_5b")["status"] == "needs-setup"
def test_sana_video_backend_can_be_disabled_without_loading_service(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    monkeypatch.setattr(pro_api, "_PRO_SANA_VIDEO_BACKEND_ENABLED", 0)

    bootstrap = client.get("/api/pro/bootstrap")
    assert bootstrap.status_code == 200
    assert not any(item.get("engineId") == "sana_video" for item in bootstrap.json()["checkpoints"])

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "slow camera move",
            "mode": "video",
            "width": 832,
            "height": 480,
            "frames": 9,
        },
    )

    assert response.status_code == 503
    assert ctx.sana_video.submitted == []


def test_create_app_serves_frontend_dist_when_present(tmp_path):
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<main>AIWF Pro</main>", encoding="utf-8")
    (dist / "asset.txt").write_text("asset", encoding="utf-8")
    client = _client(_ctx(tmp_path), frontend_dist=dist)

    assert client.get("/api/pro/runtime").status_code == 200
    assert "AIWF Pro" in client.get("/").text
    assert client.get("/asset.txt").text == "asset"
    assert "AIWF Pro" in client.get("/unknown/route").text
    assert client.get("/api/pro/removed-route").status_code == 404


def test_create_app_serves_pro_icons_without_frontend_dist(tmp_path):
    client = _client(_ctx(tmp_path))

    assert client.get("/favicon.ico").status_code == 200
    assert client.get("/app-icon.png").status_code == 200
    manifest = client.get("/manifest.webmanifest")
    assert manifest.status_code == 200
    assert manifest.json()["short_name"] == "AIWF Pro"


def test_data_endpoint_returns_output_receipts_and_counts(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.generation.submit(GenerationRequest(prompt="seed recent"))
    client = _client(ctx)

    response = client.get("/api/pro/data")

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["checkpoints"] == 1
    assert data["counts"]["recentOutputs"] >= 1
    assert data["outputRoot"].endswith("outputs")
    assert data["recentOutputs"][0]["url"].startswith("data:image/png;base64,")


def test_data_endpoint_reads_png_settings_for_output_dock(tmp_path):
    ctx = _ctx(tmp_path)
    output_path = tmp_path / "outputs" / "txt2img-images" / "with-settings.png"
    output_path.parent.mkdir(parents=True)
    infotext = (
        "a beautiful woman\n"
        "Negative prompt: back towards camera\n"
        "Steps: 7, Sampler: Euler a, CFG scale: 4.5, Seed: 42, Size: 512x768, "
        "Model: test-model, Schedule type: Karras"
    )
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", infotext)
    Image.new("RGB", (512, 768), "blue").save(output_path, pnginfo=pnginfo)
    client = _client(ctx)

    response = client.get("/api/pro/data")

    assert response.status_code == 200
    output = response.json()["recentOutputs"][0]
    assert output["prompt"] == "a beautiful woman"
    assert output["negativePrompt"] == "back towards camera"
    assert output["steps"] == 7
    assert output["cfgScale"] == 4.5
    assert output["seed"] == 42
    assert output["sampler"] == "Euler a"
    assert output["scheduler"] == "Karras"
    assert output["modelName"] == "test-model"
    assert output["width"] == 512
    assert output["height"] == 768


def test_metadata_import_reads_aiwf_generation_settings(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    generation_payload = {
        "metadata_schema": "aiwf.generation.v1",
        "pro_settings": {
            "mode": "image",
            "prompt": "neon city",
            "negativePrompt": "blur",
            "modelId": "model-a",
            "width": 768,
            "height": 512,
            "steps": 12,
            "cfgScale": 4.25,
            "sampler": "euler_a",
            "scheduler": "automatic",
            "seed": 123,
        },
        "model": {"id": "model-a", "title": "Model A", "filename": "model-a.safetensors"},
        "receipt": {"elapsed_seconds": 3.5, "steps_per_second": 3.428571},
    }
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", "neon city\nSteps: 12, CFG scale: 4.25, Seed: 123")
    pnginfo.add_text("aiwf_generation", json.dumps(generation_payload))
    pnginfo.add_text("aiwf_generation_settings", json.dumps(generation_payload["pro_settings"]))
    pnginfo.add_text("aiwf_generation_receipt", json.dumps(generation_payload["receipt"]))
    buffer = io.BytesIO()
    Image.new("RGB", (768, 512), "purple").save(buffer, format="PNG", pnginfo=pnginfo)

    response = client.post(
        "/api/pro/metadata/import",
        json={
            "filename": "import.png",
            "imageDataUrl": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["settings"]["prompt"] == "neon city"
    assert data["settings"]["modelId"] == "model-a"
    assert data["settings"]["seed"] == 123
    assert data["receipt"]["elapsed_seconds"] == 3.5
    assert data["metadata"]["model"]["title"] == "Model A"


def test_data_endpoint_reads_sidecar_settings_for_output_dock(tmp_path):
    ctx = _ctx(tmp_path)
    output_path = tmp_path / "outputs" / "txt2img-images" / "sidecar.jpg"
    output_path.parent.mkdir(parents=True)
    Image.new("RGB", (320, 320), "green").save(output_path, format="JPEG")
    output_path.with_suffix(".txt").write_text(
        "sidecar prompt\nSteps: 3, Sampler: UniPC, CFG scale: 2.25, Seed: 99, Size: 320x320, Model: sidecar-model",
        encoding="utf-8",
    )
    client = _client(ctx)

    response = client.get("/api/pro/data")

    assert response.status_code == 200
    output = response.json()["recentOutputs"][0]
    assert output["prompt"] == "sidecar prompt"
    assert output["steps"] == 3
    assert output["cfgScale"] == 2.25
    assert output["seed"] == 99
    assert output["modelName"] == "sidecar-model"


def test_logs_endpoint_returns_runtime_files_and_events(tmp_path):
    ctx = _ctx(tmp_path)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "client-events.jsonl").write_text(
        json.dumps({"action": "qa-open", "detail": "opened logs"}) + "\n",
        encoding="utf-8",
    )
    client = _client(ctx)

    response = client.get("/api/pro/logs")

    assert response.status_code == 200
    data = response.json()
    assert data["runtime"]["status"] == "idle"
    assert any(item["name"] == "client-events.jsonl" for item in data["files"])
    assert any(item["title"] == "qa-open" for item in data["events"])


def test_logs_endpoint_returns_sana_video_receipts(tmp_path):
    ctx = _ctx(tmp_path)
    sana_log_dir = tmp_path / "_local" / "logs"
    sana_log_dir.mkdir(parents=True)
    (sana_log_dir / "sana_video_latest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-06-30T10:00:00+00:00",
                "status": "error",
                "error": {"type": "RuntimeError", "message": "decode failed"},
            }
        ),
        encoding="utf-8",
    )
    client = _client(ctx)

    response = client.get("/api/pro/logs")

    assert response.status_code == 200
    data = response.json()
    assert any(item["name"] == "sana_video_latest.json" for item in data["files"])
    assert any(item["title"] == "Sana video error" and "decode failed" in item["detail"] for item in data["events"])


def test_pro_app_mounts_client_log_ingestion(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/v1/client-events",
        json={"action": "pro-open", "detail": "opened from React shell"},
    )

    assert response.status_code == 200
    assert (tmp_path / "outputs" / "client-events.jsonl").is_file()
    logs = client.get("/api/pro/logs").json()
    assert any(item["title"] == "pro-open" for item in logs["events"])


def test_settings_endpoint_returns_paths_defaults_and_runtime_flags(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.settings_path = tmp_path / "config.json"
    ctx.launch_settings_path = tmp_path / "launch.json"
    client = _client(ctx)

    response = client.get("/api/pro/settings")

    assert response.status_code == 200
    data = response.json()
    assert data["paths"]["settings"].endswith("config.json")
    assert data["paths"]["outputs"].endswith("outputs")
    assert data["generationDefaults"]["width"] == 640
    assert data["ui"]["galleryColumns"] == 2
    assert data["output"]["imageFormat"] == "png"
    assert data["video"]["wanOffload"] == "balanced"
    assert data["runtime"]["port"] == 7860
    assert data["runtime"]["backend"] == "diffusers"


def test_settings_endpoint_does_not_enumerate_wan_assets(tmp_path):
    ctx = _ctx(tmp_path)

    def unexpected_inventory_call():
        raise AssertionError("Settings must not recursively enumerate model roots")

    ctx.wan.list_local_models_labeled = unexpected_inventory_call
    ctx.wan.list_local_vaes_labeled = unexpected_inventory_call
    ctx.wan.list_local_text_encoders_labeled = unexpected_inventory_call

    response = _client(ctx).get("/api/pro/settings")

    assert response.status_code == 200


def test_settings_endpoint_updates_generation_and_ui_defaults(tmp_path):
    ctx = _ctx(tmp_path)
    saved = []
    ctx.save_settings = lambda: saved.append(True)
    client = _client(ctx)

    response = client.post(
        "/api/pro/settings",
        json={
            "generationDefaults": {
                "modelId": "model-a",
                "negativePrompt": "low quality",
                "sampler": "euler_a",
                "scheduler": "automatic",
                "steps": 28,
                "cfgScale": 6.5,
                "width": 768,
                "height": 1024,
            },
            "ui": {
                "galleryColumns": 4,
                "galleryHeight": 360,
                "livePreview": False,
                "showProgressEveryNSteps": 3,
                "livePreviewDecoder": "vae",
                "livePreviewTitleProgress": False,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["generationDefaults"]["width"] == 768
    assert data["generationDefaults"]["height"] == 1024
    assert data["generationDefaults"]["steps"] == 28
    assert data["ui"]["galleryColumns"] == 4
    assert data["ui"]["livePreview"] is False
    assert data["ui"]["showProgressEveryNSteps"] == 3
    assert data["ui"]["livePreviewTitleProgress"] is False
    assert ctx.settings.default_width == 768
    assert ctx.settings.default_height == 1024
    assert ctx.settings.enable_live_preview is False
    assert ctx.settings.live_preview_title_progress is False
    assert saved == [True]


def test_settings_endpoint_updates_output_video_and_launch_profile(tmp_path):
    ctx = _ctx(tmp_path)
    saved_settings = []
    saved_launch = []
    ctx.save_settings = lambda: saved_settings.append(True)
    ctx.save_launch_settings = lambda launch: saved_launch.append(launch)
    client = _client(ctx)

    response = client.post(
        "/api/pro/settings",
        json={
            "output": {
                "imageFormat": "webp",
                "imageQuality": 82,
                "embedMetadata": False,
                "saveSidecarTxt": True,
                "saveGrid": True,
                "filenamePattern": "[model_name]-[seed]-[seq]",
                "saveBeforeHires": True,
                "saveInterrupted": True,
                "metadataIncludeModelHash": False,
                "metadataIncludeVaeHash": False,
                "metadataIncludeLoraHashes": False,
                "metadataIncludeAppVersion": False,
                "metadataIncludeOptimizationProfile": False,
                "optimizationProfileId": "manual_break_glass",
            },
            "video": {
                "wanHigh": "high.safetensors",
                "wanLow": "low.safetensors",
                "wanVae": "vae.safetensors",
                "wanTextEncoder": "umt5.safetensors",
                "wanOffload": "sequential",
                "wanSampler": "heun",
                "wanFlowShift": 9.5,
                "wanRuntimeMode": "high_low",
            },
            "runtime": {
                "port": 7899,
                "listen": True,
                "api": True,
                "genlog": True,
                "backend": "onnx",
                "onnxProvider": "cuda",
                "onnxModelDir": str(tmp_path / "onnx-pipeline"),
                "attention": "sdpa",
                "vramProfile": "high",
                "highvram": True,
                "asyncOffload": False,
                "pinnedMemory": False,
                "cudaMalloc": True,
                "torchCompile": True,
                "channelsLast": True,
                "apiRateLimitPerMinute": 120,
                "modelsDir": str(tmp_path / "models-custom"),
                "checkpointDir": str(tmp_path / "checkpoints-custom"),
                "outputDir": str(tmp_path / "outputs-custom"),
                "extraModelDirs": str(tmp_path / "extra-models"),
                "extraCheckpointDirs": str(tmp_path / "extra-checkpoints"),
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["output"]["imageFormat"] == "webp"
    assert data["output"]["saveSidecarTxt"] is True
    assert data["output"]["metadataIncludeModelHash"] is False
    assert data["video"]["wanOffload"] == "sequential"
    assert data["video"]["wanRuntimeMode"] == "high_low"
    assert data["runtime"]["port"] == 7899
    assert data["runtime"]["backend"] == "onnx"
    assert data["runtime"]["onnxProvider"] == "cuda"
    assert data["runtime"]["onnxModelDir"] == str(tmp_path / "onnx-pipeline")
    assert ctx.settings.onnx_model_dir == str(tmp_path / "onnx-pipeline")
    assert data["runtime"]["attention"] == "sdpa"
    assert data["runtime"]["vramProfile"] == "high"
    assert data["runtime"]["highvram"] is True
    assert data["runtime"]["medvram"] is False
    assert data["runtime"]["lowvram"] is False
    assert data["runtime"]["asyncOffload"] is False
    assert ctx.settings.image_format == "webp"
    assert ctx.settings.last_wan_high == "high.safetensors"
    assert ctx.flags.port == 7899
    assert ctx.flags.inference_backend == "onnx"
    assert ctx.flags.attention_backend == "sdpa"
    assert saved_launch and saved_launch[0].port == 7899
    assert saved_launch[0].models_dir.endswith("models-custom")
    assert ctx.enhance.catalog_invalidations == 1
    assert saved_settings == [True]


def test_settings_defaults_restore_saved_wan_pair_and_derive_support_assets(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.settings.last_wan_runtime_mode = "high_low"
    ctx.settings.last_wan_high = "wan-high.safetensors"
    ctx.settings.last_wan_low = "wan-low.safetensors"
    ctx.settings.last_wan_vae = ""
    ctx.settings.last_wan_text_encoder = ""
    ctx.wan.preferred_vae = lambda mode: f"vae-for-{mode}.safetensors"
    ctx.wan.default_text_encoder = lambda: "umt5-fp8.safetensors"
    monkeypatch.setattr(pro_api, "_is_checkpoint_id_selectable", lambda _ctx, _id: True)

    defaults = pro_api._settings_defaults(ctx)

    assert defaults["wanRuntimeMode"] == pro_api._canonical_wan_runtime_mode("high_low")
    assert defaults["highNoiseModelId"] == "wan-high.safetensors"
    assert defaults["lowNoiseModelId"] == "wan-low.safetensors"
    assert defaults["vaeId"] == f"vae-for-{defaults['wanRuntimeMode']}.safetensors"
    assert defaults["textEncoderPath"] == "umt5-fp8.safetensors"


def test_wan_request_uses_saved_vae_when_request_omits_vae_id(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.settings.last_wan_vae = "saved-wan-vae.safetensors"
    ctx.wan.preferred_vae = lambda _mode: "preferred-wan-vae.safetensors"

    request = pro_api._wan_video_request_from_payload(
        ctx,
        pro_api.ProGeneratePayload(prompt="slow camera move", checkpoint_id="wan-a.safetensors"),
    )

    assert request.vae_id == "saved-wan-vae.safetensors"


@pytest.mark.parametrize(
    ("runtime_mode", "saved_vae", "preferred_vae"),
    [
        ("fast_5b", "wan2.1_vae.safetensors", "wan2.2_vae.safetensors"),
        ("high_low", "wan2.2_vae.safetensors", "wan2.1_vae.safetensors"),
    ],
)
def test_wan_request_does_not_reuse_saved_vae_from_another_runtime_family(
    tmp_path, runtime_mode, saved_vae, preferred_vae
):
    ctx = _ctx(tmp_path)
    ctx.settings.last_wan_runtime_mode = runtime_mode
    ctx.settings.last_wan_vae = saved_vae
    ctx.wan.preferred_vae = lambda mode: preferred_vae

    request = pro_api._wan_video_request_from_payload(
        ctx,
        pro_api.ProGeneratePayload(
            prompt="slow camera move",
            checkpoint_id="wan-a.safetensors",
            wan_runtime_mode=runtime_mode,
        ),
    )

    assert request.vae_id == preferred_vae


def test_wan_request_preserves_an_explicit_vae_even_if_it_conflicts(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.settings.last_wan_runtime_mode = "fast_5b"
    ctx.settings.last_wan_vae = "wan2.1_vae.safetensors"
    ctx.wan.preferred_vae = lambda _mode: "wan2.2_vae.safetensors"

    request = pro_api._wan_video_request_from_payload(
        ctx,
        pro_api.ProGeneratePayload(
            prompt="slow camera move",
            checkpoint_id="wan-a.safetensors",
            wan_runtime_mode="fast_5b",
            vae_id="wan2.1_vae.safetensors",
        ),
    )

    assert request.vae_id == "wan2.1_vae.safetensors"


def test_downloads_endpoint_reports_catalog_install_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_api, "_huggingface_token", lambda: "")
    ctx = _ctx(tmp_path)
    # The developer machine may have catalog snapshots under a shared model
    # root; this test is about catalog presentation, not host inventory.
    monkeypatch.setattr(ctx.model_download, "is_catalog_installed", lambda _item, **_kwargs: False)
    client = _client(ctx)

    response = client.get("/api/pro/downloads")

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["catalog"] > 0
    assert data["counts"]["installed"] == 0
    assert any(item["key"] == "hf-sdxl-base" for item in data["catalog"])
    public_entry = next(item for item in data["catalog"] if item["key"] == "hf-sd15-pruned")
    assert public_entry["canDownload"] is True
    assert public_entry["requiresAuth"] is False
    gated_entry = next(item for item in data["catalog"] if item["key"] == "hf-sd35-medium")
    assert gated_entry["canDownload"] is False
    assert gated_entry["requiresAuth"] is True
    assert "anima-base-v1" not in {item["key"] for item in data["catalog"]}
    assert "qwen-nunchaku-image-lightning-int4-r32" in {item["key"] for item in data["catalog"]}
    assert data["categories"][0]["destination"]


def test_downloads_endpoint_does_not_offer_shared_copy_for_primary_installed_snapshot(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    entry = ctx.model_download.find_catalog("hf-sd35-medium")
    assert entry is not None and entry.snapshot
    monkeypatch.setattr(ctx.model_download, "is_catalog_installed", lambda item, **_kwargs: item.key == entry.key)
    monkeypatch.setattr(ctx.model_download, "_find_shared_catalog_snapshot", lambda _item: None)

    response = _client(ctx).get("/api/pro/downloads")

    assert response.status_code == 200
    item = next(row for row in response.json()["catalog"] if row["key"] == entry.key)
    assert item["installed"] is True
    assert item["sharedSnapshotAvailable"] is False


def test_downloads_endpoint_hides_shared_copy_when_primary_snapshot_is_ready(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    entry = ctx.model_download.find_catalog("hf-sd35-medium")
    assert entry is not None and entry.snapshot
    monkeypatch.setattr(ctx.model_download, "is_catalog_installed", lambda item, **_kwargs: item.key == entry.key)
    monkeypatch.setattr(ctx.model_download, "_catalog_snapshot_ready", lambda _entry, _target: True)
    monkeypatch.setattr(ctx.model_download, "_find_shared_catalog_snapshot", lambda _item: tmp_path / "shared-snapshot")

    response = _client(ctx).get("/api/pro/downloads")

    item = next(row for row in response.json()["catalog"] if row["key"] == entry.key)
    assert item["installed"] is True
    assert item["sharedSnapshotAvailable"] is False


def test_gated_catalog_entry_is_installable_when_huggingface_token_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_api, "_huggingface_token", lambda: "configured-token")
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.get("/api/pro/downloads")

    assert response.status_code == 200
    gated_entry = next(item for item in response.json()["catalog"] if item["key"] == "hf-sd35-medium")
    assert gated_entry["requiresAuth"] is True
    assert gated_entry["canDownload"] is True

    def fake_download(key: str):
        assert key == "hf-sd35-medium"
        return tmp_path / "sd35-medium.safetensors"

    ctx.model_download.download_catalog = fake_download
    install = client.post("/api/pro/downloads/catalog/hf-sd35-medium")
    assert install.status_code == 200
    assert install.json()["downloaded"]["key"] == "hf-sd35-medium"


def test_downloads_catalog_endpoint_downloads_public_entry_and_refreshes(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    def fake_download(key: str):
        assert key == "hf-sd15-pruned"
        dest = tmp_path / "models" / "Stable-diffusion" / "v1-5-pruned-emaonly-fp16.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")
        return dest

    ctx.model_download.download_catalog = fake_download

    response = client.post("/api/pro/downloads/catalog/hf-sd15-pruned")

    assert response.status_code == 200
    data = response.json()
    assert data["downloaded"]["key"] == "hf-sd15-pruned"
    assert data["downloaded"]["path"].endswith("v1-5-pruned-emaonly-fp16.safetensors")


def test_downloads_catalog_endpoint_reports_misplaced_asset_placement(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    monkeypatch.setattr(ctx.model_download, "place_misplaced_catalog_asset", lambda entry: entry.key == "flux-clip-l")

    response = client.post("/api/pro/downloads/catalog/flux-clip-l")

    assert response.status_code == 200
    assert response.json()["catalogAction"] == {"key": "flux-clip-l", "status": "placed"}
    assert "downloaded" not in response.json()


def test_downloads_catalog_endpoint_imports_verified_shared_root_asset(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    copied = {
        "source": str(tmp_path / "shared" / "misc" / "ae.safetensors"),
        "target": str(tmp_path / "models" / "flux" / "VAE" / "ae.safetensors"),
    }
    monkeypatch.setattr(
        ctx.model_download,
        "copy_shared_catalog_asset_to_primary",
        lambda entry: copied if entry.key == "flux-ae-vae" else None,
    )

    response = client.post("/api/pro/downloads/catalog/flux-ae-vae")

    assert response.status_code == 200
    assert response.json()["catalogAction"] == {
        "key": "flux-ae-vae",
        "status": "copied_from_shared_root",
        "source": copied["source"],
        "path": copied["target"],
    }
    assert "downloaded" not in response.json()


def test_downloads_catalog_endpoint_reports_insufficient_space_without_network_fallback(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    downloads = []

    def insufficient_space(_entry):
        raise ValueError(
            "Insufficient disk space to copy verified shared asset `flux-ae-vae`: "
            "400000000 bytes required, 100000000 bytes available."
        )

    monkeypatch.setattr(ctx.model_download, "copy_shared_catalog_asset_to_primary", insufficient_space)
    ctx.model_download.download_catalog = lambda key: downloads.append(key)

    response = client.post("/api/pro/downloads/catalog/flux-ae-vae")

    assert response.status_code == 422
    assert "Insufficient disk space" in response.json()["detail"]
    assert "400000000 bytes required" in response.json()["detail"]
    assert downloads == []


def test_downloads_catalog_endpoint_downloads_when_no_valid_shared_asset_exists(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    downloads = []
    monkeypatch.setattr(ctx.model_download, "copy_shared_catalog_asset_to_primary", lambda _entry: None)
    monkeypatch.setattr(ctx.model_download, "is_catalog_installed", lambda _entry: False)
    monkeypatch.setattr(ctx.model_download, "place_misplaced_catalog_asset", lambda _entry: False)

    def download(key):
        downloads.append(key)
        return tmp_path / "downloaded.safetensors"

    ctx.model_download.download_catalog = download

    response = client.post("/api/pro/downloads/catalog/flux-ae-vae")

    assert response.status_code == 200
    assert response.json()["downloaded"]["key"] == "flux-ae-vae"
    assert downloads == ["flux-ae-vae"]


def test_catalog_shared_asset_copy_waits_until_generation_is_idle(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    calls = []
    monkeypatch.setattr(
        ctx.model_download,
        "copy_shared_catalog_asset_to_primary",
        lambda entry: calls.append(entry.key) or {"source": "shared", "target": "primary"},
    )
    pro_api._pro_video_job_start(ctx, SimpleNamespace(steps=1), message="running test video")

    response = _client(ctx).post("/api/pro/downloads/catalog/flux-ae-vae")

    assert response.status_code == 409
    assert "GPU generation job is active" in response.json()["detail"]
    assert calls == []


def test_catalog_download_waits_until_generation_is_idle(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    calls = []
    ctx.model_download.download_catalog = lambda key: calls.append(key) or tmp_path / "installed.safetensors"
    monkeypatch.setattr(pro_api, "_catalog_entry_can_download", lambda _entry: True)
    pro_api._pro_video_job_start(ctx, SimpleNamespace(steps=1), message="running test video")

    response = _client(ctx).post("/api/pro/downloads/catalog/flux-dev-q4km")

    assert response.status_code == 409
    assert "GPU generation job is active" in response.json()["detail"]
    assert calls == []


def test_shared_snapshot_import_endpoint_requires_explicit_confirmation(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    called = []
    monkeypatch.setattr(
        ctx.model_download,
        "copy_shared_catalog_snapshot_to_primary",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    response = client.post("/api/pro/downloads/catalog/hf-sd35-medium/import-shared")

    assert response.status_code == 400
    assert called == []


def test_shared_snapshot_preview_endpoint_returns_copy_estimate(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    preview = {
        "source": str(tmp_path / "shared" / "Diffusers" / "snapshot"),
        "target": str(tmp_path / "models" / "family" / "Diffusers" / "snapshot"),
        "sizeBytes": 1024,
        "requiredBytes": 64 * 1024 * 1024,
        "freeBytes": 100 * 1024 * 1024,
        "enoughSpace": True,
    }
    monkeypatch.setattr(ctx.model_download, "preview_shared_catalog_snapshot_import", lambda _entry: preview)

    response = client.get("/api/pro/downloads/catalog/hf-sd35-medium/shared-import-preview")

    assert response.status_code == 200
    assert response.json()["preview"] == preview


def test_shared_snapshot_import_endpoint_uses_confirmed_preview_identity(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    source = str(tmp_path / "shared" / "snapshot")
    target = str(tmp_path / "models" / "snapshot")
    calls = []
    monkeypatch.setattr(
        ctx.model_download,
        "copy_shared_catalog_snapshot_to_primary",
        lambda entry, **kwargs: calls.append((entry.key, kwargs)) or {"source": source, "target": target},
    )
    monkeypatch.setattr(pro_api, "_refresh_model_inventory_after_sort", lambda _ctx: {"inventoryCount": 0})

    response = client.post(
        "/api/pro/downloads/catalog/hf-sd35-medium/import-shared",
        params={"confirm": "true", "source": source, "size_bytes": 99},
    )

    assert response.status_code == 200
    assert calls == [("hf-sd35-medium", {"expected_source": source, "expected_size_bytes": 99})]
    assert response.json()["catalogAction"]["status"] == "copied_snapshot_from_shared_root"


def test_downloads_catalog_endpoint_blocks_gated_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_api, "_huggingface_token", lambda: "")
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post("/api/pro/downloads/catalog/hf-sd35-medium")

    assert response.status_code == 422
    assert "requires upstream access" in response.json()["detail"]


def test_downloads_bundle_endpoint_skips_installed_and_reports_each_item(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    monkeypatch.setattr(pro_api, "_catalog_entry_can_download", lambda _entry: True)
    ctx.model_download.is_catalog_installed = lambda entry, **_kwargs: entry.key == "flux-clip-l"
    ctx.model_download.place_misplaced_catalog_asset = lambda entry: entry.key == "flux-ae-vae"
    downloaded = []

    def fake_download(key: str):
        downloaded.append(key)
        return tmp_path / f"{key}.safetensors"

    ctx.model_download.download_catalog = fake_download

    response = client.post("/api/pro/downloads/bundles/flux")

    assert response.status_code == 200
    items = {item["key"]: item["status"] for item in response.json()["bundleInstall"]["items"]}
    assert response.json()["bundleInstall"]["installationStatus"] == "all-items-installed"
    assert response.json()["bundleInstall"]["readiness"] == "refresh-required"
    assert items["flux-clip-l"] == "already_installed"
    assert items["flux-ae-vae"] == "placed"
    assert items["flux-dev-q4km"] == "downloaded"
    assert "flux-clip-l" not in downloaded


@pytest.mark.parametrize(
    ("enough_space", "expected_status", "expected_installation_status"),
    [
        (True, "shared_snapshot_confirmation_required", "confirmation-required"),
        (False, "shared_snapshot_insufficient_space", "insufficient-space"),
    ],
)
def test_bundle_snapshot_shared_root_returns_preview_without_download(
    tmp_path, monkeypatch, enough_space, expected_status, expected_installation_status,
):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    entry = ctx.model_download.find_catalog("sana-video-2b-480p-diffusers")
    assert entry is not None and entry.snapshot
    preview = {
        "source": str(tmp_path / "shared" / "SANA-Video_2B_480p_diffusers"),
        "target": str(tmp_path / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"),
        "sizeBytes": 5_000_000,
        "requiredBytes": 5_500_000,
        "freeBytes": 8_000_000 if enough_space else 1_000_000,
        "enoughSpace": enough_space,
    }
    monkeypatch.setattr(pro_api, "_catalog_entry_can_download", lambda _entry: True)
    ctx.model_download.is_catalog_installed = lambda _entry, **_kwargs: False
    ctx.model_download.preview_shared_catalog_snapshot_import = lambda candidate: preview if candidate.key == entry.key else None
    ctx.model_download.place_misplaced_catalog_asset = lambda _entry: False
    downloaded = []
    ctx.model_download.download_catalog = lambda key: downloaded.append(key)

    response = client.post("/api/pro/downloads/bundles/sana-video")

    assert response.status_code == 200
    bundle = response.json()["bundleInstall"]
    assert bundle["installationStatus"] == expected_installation_status
    assert bundle["readiness"] == "refresh-required"
    assert bundle["items"] == [{
        "key": entry.key,
        "status": expected_status,
        "sharedSnapshotPreview": preview,
    }]
    assert downloaded == []


def test_bundle_install_requires_confirmation_for_shared_snapshot_and_skips_download(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    entry = ctx.model_download.find_catalog("sana-video-2b-480p-diffusers")
    assert entry is not None and entry.snapshot
    preview = {
        "source": str(tmp_path / "shared" / "SANA-Video_2B_480p_diffusers"),
        "target": str(tmp_path / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"),
        "sizeBytes": 5_000_000,
        "requiredBytes": 5_500_000,
        "freeBytes": 8_000_000,
        "enoughSpace": True,
    }
    monkeypatch.setattr(pro_api, "_catalog_entry_can_download", lambda _entry: True)
    ctx.model_download.is_catalog_installed = lambda _entry, **_kwargs: False
    ctx.model_download.preview_shared_catalog_snapshot_import = lambda candidate: preview if candidate.key == entry.key else None
    ctx.model_download.place_misplaced_catalog_asset = lambda _entry: False
    downloaded = []
    ctx.model_download.download_catalog = lambda key: downloaded.append(key)

    response = client.post("/api/pro/downloads/bundles/sana-video")

    assert response.status_code == 200
    bundle = response.json()["bundleInstall"]
    assert bundle["installationStatus"] == "confirmation-required"
    assert bundle["readiness"] == "refresh-required"
    assert bundle["items"] == [{
        "key": entry.key,
        "status": "shared_snapshot_confirmation_required",
        "sharedSnapshotPreview": preview,
    }]
    assert downloaded == []


def test_bundle_install_reports_insufficient_shared_snapshot_space_without_downloading(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    entry = ctx.model_download.find_catalog("sana-video-2b-480p-diffusers")
    assert entry is not None and entry.snapshot
    preview = {
        "source": str(tmp_path / "shared" / "SANA-Video_2B_480p_diffusers"),
        "target": str(tmp_path / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"),
        "sizeBytes": 5_000_000,
        "requiredBytes": 5_500_000,
        "freeBytes": 1_000_000,
        "enoughSpace": False,
    }
    monkeypatch.setattr(pro_api, "_catalog_entry_can_download", lambda _entry: True)
    ctx.model_download.is_catalog_installed = lambda _entry, **_kwargs: False
    ctx.model_download.preview_shared_catalog_snapshot_import = lambda candidate: preview if candidate.key == entry.key else None
    ctx.model_download.place_misplaced_catalog_asset = lambda _entry: False
    downloaded = []
    ctx.model_download.download_catalog = lambda key: downloaded.append(key)

    response = client.post("/api/pro/downloads/bundles/sana-video")

    assert response.status_code == 200
    bundle = response.json()["bundleInstall"]
    assert bundle["installationStatus"] == "insufficient-space"
    assert bundle["items"] == [{
        "key": entry.key,
        "status": "shared_snapshot_insufficient_space",
        "sharedSnapshotPreview": preview,
    }]
    assert downloaded == []


def test_downloads_bundle_defers_downloads_while_generation_is_active(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    monkeypatch.setattr(pro_api, "_catalog_entry_can_download", lambda _entry: True)
    ctx.model_download.is_catalog_installed = lambda _entry, **_kwargs: False
    ctx.model_download.place_misplaced_catalog_asset = lambda _entry: False
    downloaded = []
    ctx.model_download.download_catalog = lambda key: downloaded.append(key) or tmp_path / f"{key}.safetensors"
    pro_api._pro_video_job_start(ctx, SimpleNamespace(steps=1), message="running test video")

    response = client.post("/api/pro/downloads/bundles/flux")

    assert response.status_code == 409
    assert "GPU generation job is active" in response.json()["detail"]
    assert downloaded == []


def test_downloads_bundle_reports_partial_when_items_need_manual_access(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    monkeypatch.setattr(pro_api, "_catalog_entry_can_download", lambda _entry: False)
    ctx.model_download.is_catalog_installed = lambda _entry, **_kwargs: False
    ctx.model_download.place_misplaced_catalog_asset = lambda _entry: False

    response = client.post("/api/pro/downloads/bundles/flux-components")

    assert response.status_code == 200
    bundle = response.json()["bundleInstall"]
    assert bundle["installationStatus"] == "blocked"
    assert bundle["readiness"] == "refresh-required"
    assert bundle["items"]
    assert all(item["status"] == "manual_access_required" for item in bundle["items"])


def test_sdxl_route_bundle_installs_support_assets_without_another_base_model(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    monkeypatch.setattr(pro_api, "_catalog_entry_can_download", lambda _entry: True)
    ctx.model_download.is_catalog_installed = lambda _entry, **_kwargs: False
    ctx.model_download.place_misplaced_catalog_asset = lambda _entry: False
    downloaded = []
    ctx.model_download.download_catalog = lambda key: downloaded.append(key) or tmp_path / f"{key}.safetensors"

    response = client.post("/api/pro/downloads/bundles/sdxl-components")

    assert response.status_code == 200
    bundle = response.json()["bundleInstall"]
    assert bundle["installationStatus"] == "all-items-installed"
    assert set(downloaded) == {
        "hf-vae-sdxl",
        "hf-sdxl-singlefile-config",
        "hf-sdxl-inpaint-singlefile-config",
        "hf-sdxl-refiner-singlefile-config",
    }
    assert "hf-sdxl-base" not in downloaded
    assert "hf-sdxl-refiner" not in downloaded


def test_capabilities_endpoint_reports_gradio_tool_readiness(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.get("/api/pro/capabilities")

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["gradioTabs"] >= 14
    assert data["counts"]["loras"] == 1
    assert data["counts"]["controlnet"] == 1
    assert data["counts"]["sam"] == 1
    assert data["counts"]["reactor"] == 1
    assert data["counts"]["enhance"] == 2
    assert data["counts"]["wan"] == 1
    labels = {item["label"] for item in data["tools"]}
    assert {"ControlNet", "Segment / SAM", "ReActor", "Sana / Wan / LTX video"}.issubset(labels)


def test_capabilities_do_not_call_enhance_ready_when_catalog_weights_are_missing(tmp_path):
    ctx = _ctx(tmp_path)
    for path in ctx.enhance.model_paths.values():
        path.unlink()

    response = _client(ctx).get("/api/pro/capabilities")

    assert response.status_code == 200
    enhance = next(item for item in response.json()["tools"] if item["id"] == "enhance")
    assert enhance["status"] == "needs-assets"
    assert enhance["count"] == 0
    assert enhance["details"][:2] == ["0 of 1 upscalers installed", "0 of 1 restorers installed"]


def test_sort_inventory_refresh_invalidates_enhance_catalog(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(pro_api, "scan_and_write_model_inventory", lambda _flags: [])

    assert pro_api._refresh_model_inventory_after_sort(ctx) == {"inventoryCount": 0}
    assert ctx.enhance.catalog_invalidations == 1


def test_capabilities_endpoint_reports_pipeline_readiness(tmp_path, monkeypatch):
    def fake_collect(flags, settings=None, *, include_downloads=True, download_roots=None, force_rescan=False):
        assert flags is not None
        assert settings is not None
        assert include_downloads is False
        assert download_roots is None
        assert force_rescan is False
        return [
            PipelineReadinessRecord(
                id="sdxl-ready",
                family="image",
                asset_type="checkpoint",
                path=str(tmp_path / "sdxl-ready.safetensors"),
                status="working",
                route="react-pro",
                reason="Smoke receipt exists.",
                smoke_command="python scripts/probe_image_runtime.py",
                receipt_path="docs/qa/sdxl-ready.json",
            ),
            PipelineReadinessRecord(
                id="ltx-candidate",
                family="video",
                asset_type="pipeline",
                path=str(tmp_path / "ltx"),
                status="metadata-only",
                route="gradio",
                reason="Pipeline metadata exists but runtime smoke is pending.",
                suggested_action="Run the LTX smoke test.",
            ),
            PipelineReadinessRecord(
                id="qwen-vl",
                family="llm-vl",
                asset_type="gguf",
                path=str(tmp_path / "qwen.gguf"),
                status="unsupported-no-route",
                route="planned",
                reason="No promoted Pro worker yet.",
            ),
        ]

    monkeypatch.setattr(pro_api, "collect_pipeline_readiness", fake_collect)
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.get("/api/pro/capabilities")

    assert response.status_code == 200
    data = response.json()
    readiness = data["readiness"]
    assert readiness["counts"]["working"] == 1
    assert readiness["counts"]["metadata-only"] == 1
    assert readiness["counts"]["unsupported-no-route"] == 1
    assert readiness["metadataOnlyCount"] == 1
    assert readiness["total"] == 3
    assert readiness["error"] == ""
    assert readiness["working"][0]["label"] == "sdxl-ready.safetensors"
    assert {item["id"] for item in readiness["needsWork"]} == {"ltx-candidate", "qwen-vl"}
    assert any(item["family"] == "llm-vl" and item["total"] == 1 for item in readiness["families"])
    llm_tool = next(item for item in data["tools"] if item["id"] == "llm-vl")
    assert llm_tool["status"] == "not-wired"
    assert llm_tool["count"] == 1


def test_capabilities_endpoint_uses_cached_readiness_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "_local" / "logs" / "pipeline_readiness_current_inventory.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(
        json.dumps(
            {
                "summary": {"working": 1, "metadata-only": 0, "blocked-cleanly": 0, "broken-runtime": 1, "unsupported-no-route": 0},
                "records": [
                    {
                        "id": "flux2-klein-ready",
                        "family": "image",
                        "asset_type": "checkpoint",
                        "path": str(tmp_path / "flux2.safetensors"),
                        "status": "working",
                        "route": "flux2-klein",
                        "reason": "Warm smoke exists.",
                    },
                    {
                        "id": "qwen-needs-base",
                        "family": "image",
                        "asset_type": "checkpoint",
                        "path": str(tmp_path / "qwen.safetensors"),
                        "status": "broken-runtime",
                        "route": "qwen-image",
                        "reason": "Base shards are missing.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def broken_collect(*args, **kwargs):
        raise AssertionError("live readiness collection should not block capabilities when a snapshot exists")

    monkeypatch.setattr(pro_api, "collect_pipeline_readiness", broken_collect)
    ctx = _ctx(tmp_path)
    ctx._pro_capability_refresh_at = 10**12
    client = _client(ctx)

    response = client.get("/api/pro/capabilities")

    assert response.status_code == 200
    data = response.json()
    readiness = data["readiness"]
    assert readiness["counts"]["working"] == 1
    assert readiness["counts"]["broken-runtime"] == 1
    assert readiness["total"] == 2
    assert readiness["error"] == ""
    assert "pipeline_readiness_current_inventory.json" in readiness["source"]
    assert "cached readiness ledger" in readiness["sourceMessage"]
    assert any("cached readiness ledger" in note for note in data["notes"])
    assert readiness["working"][0]["id"] == "flux2-klein-ready"
    assert readiness["needsWork"][0]["id"] == "qwen-needs-base"


def test_capabilities_endpoint_keeps_working_when_readiness_fails(tmp_path, monkeypatch):
    def broken_collect(*args, **kwargs):
        raise RuntimeError("readiness scan failed")

    monkeypatch.setattr(pro_api, "collect_pipeline_readiness", broken_collect)
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.get("/api/pro/capabilities")

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["loras"] == 1
    assert data["readiness"]["total"] == 0
    assert data["readiness"]["counts"]["working"] == 0
    assert "readiness scan failed" in data["readiness"]["error"]


def test_ping_reports_name_version_and_auth_state(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.get("/api/pro/ping")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AIWF Studio Pro"
    assert data["version"]
    assert data["authRequired"] is True
    assert data["mobileAccessEnabled"] is False


def test_mobile_pairing_get_requires_loopback(tmp_path):
    ctx = _ctx(tmp_path)
    app = create_app(ctx, frontend_dist=Path("__missing_frontend_dist__"))

    loopback_client = TestClient(app, client=("127.0.0.1", 51000))
    remote_client = TestClient(app, client=("192.168.1.50", 51000))

    ok = loopback_client.get("/api/pro/mobile-pairing")
    blocked = remote_client.get("/api/pro/mobile-pairing")

    assert ok.status_code == 200
    assert ok.json()["enabled"] is False
    # Loopback pairing can provision a token before the user enables access,
    # while remote API enforcement remains closed until access is enabled.
    assert ok.json()["token"]
    assert blocked.status_code == 403
    assert remote_client.get("/api/pro/runtime").status_code == 403


def test_fresh_remote_client_is_blocked_before_pairing_provisions_a_token(tmp_path):
    ctx = _ctx(tmp_path)
    app = create_app(ctx, frontend_dist=Path("__missing_frontend_dist__"))
    remote_client = TestClient(app, client=("192.168.1.50", 51000))

    assert not (tmp_path / "_local" / "mobile_auth.json").exists()
    response = remote_client.get("/api/pro/runtime")

    assert response.status_code == 403
    assert not (tmp_path / "_local" / "mobile_auth.json").exists()


def test_mobile_pairing_enable_then_enforces_token_for_remote_clients(tmp_path):
    ctx = _ctx(tmp_path)
    app = create_app(ctx, frontend_dist=Path("__missing_frontend_dist__"))

    loopback_client = TestClient(app, client=("127.0.0.1", 51000))
    remote_client = TestClient(app, client=("192.168.1.50", 51000))

    enable_response = loopback_client.post("/api/pro/mobile-pairing", json={"enabled": True})
    assert enable_response.status_code == 200
    payload = enable_response.json()
    assert payload["enabled"] is True
    token = payload["token"]
    assert token
    assert payload["pairingUri"].startswith("aiwf://pair?")

    # Remote client without the token is rejected on any /api/pro route...
    denied = remote_client.get("/api/pro/runtime")
    assert denied.status_code == 401

    # ...but succeeds once it presents the correct token.
    allowed = remote_client.get("/api/pro/runtime", headers={"X-AIWF-Token": token})
    assert allowed.status_code == 200

    # /ping never requires the token, so an unpaired phone can still discover the server.
    ping = remote_client.get("/api/pro/ping")
    assert ping.status_code == 200
    assert ping.json()["authRequired"] is True
    assert ping.json()["mobileAccessEnabled"] is True

    # The loopback desktop app itself is never challenged.
    loopback_runtime = loopback_client.get("/api/pro/runtime")
    assert loopback_runtime.status_code == 200

    # Rotating invalidates the old token immediately.
    rotated = loopback_client.post("/api/pro/mobile-pairing", json={"enabled": True, "rotate": True})
    new_token = rotated.json()["token"]
    assert new_token != token
    assert remote_client.get("/api/pro/runtime", headers={"X-AIWF-Token": token}).status_code == 401
    assert remote_client.get("/api/pro/runtime", headers={"X-AIWF-Token": new_token}).status_code == 200

    # Disabling access closes the remote API without needing a restart.
    loopback_client.post("/api/pro/mobile-pairing", json={"enabled": False})
    assert remote_client.get("/api/pro/runtime").status_code == 403
    disabled_ping = remote_client.get("/api/pro/ping")
    assert disabled_ping.status_code == 200
    assert disabled_ping.json()["authRequired"] is True
    assert disabled_ping.json()["mobileAccessEnabled"] is False


def test_video_lab_uploads_with_same_name_get_unique_paths(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    def fake_probe(path: Path):
        return {"path": str(path), "filename": path.name}

    monkeypatch.setattr(pro_api, "_video_lab_probe", fake_probe)

    first = client.post(
        "/api/pro/video-lab/upload",
        files={"file": ("clip.mp4", b"first-video", "video/mp4")},
    )
    second = client.post(
        "/api/pro/video-lab/upload",
        files={"file": ("clip.mp4", b"second-video", "video/mp4")},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_path = Path(first.json()["path"])
    second_path = Path(second.json()["path"])
    assert first_path != second_path
    assert first_path.read_bytes() == b"first-video"
    assert second_path.read_bytes() == b"second-video"


def test_video_lab_upload_collision_preserves_existing_file(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    upload_root = pro_api._video_lab_upload_root(ctx)
    existing_path = upload_root / "clip_existing.mp4"
    new_path = upload_root / "clip_unique.mp4"
    existing_path.write_bytes(b"existing-video")
    destinations = iter((existing_path, new_path))

    monkeypatch.setattr(pro_api, "_video_lab_upload_destination", lambda *_args: next(destinations))
    monkeypatch.setattr(
        pro_api,
        "_video_lab_probe",
        lambda path: {"path": str(path), "filename": path.name},
    )

    response = client.post(
        "/api/pro/video-lab/upload",
        files={"file": ("clip.mp4", b"new-video", "video/mp4")},
    )

    assert response.status_code == 200
    assert Path(response.json()["path"]) == new_path
    assert existing_path.read_bytes() == b"existing-video"
    assert new_path.read_bytes() == b"new-video"


def test_outputs_thumb_query_param_returns_resized_jpeg(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    image_path = ctx.flags.output_dir / "txt2img-images" / "thumb-source.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "green").save(image_path)

    full = client.get("/api/pro/outputs/txt2img-images/thumb-source.png")
    thumbed = client.get("/api/pro/outputs/txt2img-images/thumb-source.png?thumb=128")

    assert full.status_code == 200
    assert thumbed.status_code == 200
    assert thumbed.headers["content-type"] == "image/jpeg"
    thumb_image = Image.open(io.BytesIO(thumbed.content))
    assert max(thumb_image.size) <= 128
    assert len(thumbed.content) < len(full.content)


def test_outputs_index_paginates_by_recency(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    out_dir = ctx.flags.output_dir / "txt2img-images"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(5):
        path = out_dir / f"page-{index}.png"
        Image.new("RGB", (8, 8), "red").save(path)
        paths.append(path)
        os.utime(path, (1_700_000_000 + index, 1_700_000_000 + index))

    page1 = client.get("/api/pro/outputs-index?offset=0&limit=2")
    page2 = client.get("/api/pro/outputs-index?offset=2&limit=2")

    assert page1.status_code == 200
    data1 = page1.json()
    data2 = page2.json()
    assert data1["total"] == 5
    assert len(data1["items"]) == 2
    assert len(data2["items"]) == 2
    # Most recently modified files come first, and pages don't overlap.
    assert data1["items"][0]["path"].endswith("page-4.png")
    assert data1["items"][1]["path"].endswith("page-3.png")
    assert data2["items"][0]["path"].endswith("page-2.png")
    page1_paths = {item["path"] for item in data1["items"]}
    page2_paths = {item["path"] for item in data2["items"]}
    assert page1_paths.isdisjoint(page2_paths)


class _AudioStub:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.generated = []
        self.installed = False
        self.installed_variants = set()
        self.variant_install_calls = []
        self.musicgen_install_calls = []
        self.installed_musicgen_variants = set()
        self.prepared = []
        self.parked_musicgen_variants = set()
        self.mux_ready = True

    def setup_status(self, *, deep=False):
        ready = self.installed
        return {
            "minimumReady": ready,
            "installing": False,
            "musicReady": ready,
            "musicDependenciesReady": True,
            "sfxReady": ready,
            "videoAudioReady": ready and self.mux_ready,
            "labReady": ready,
            "muxReady": self.mux_ready,
            "message": "ready" if ready else "setup needed",
            "estimatedDownload": "test",
            "licenseNotice": "test license",
            "defaults": {
                "music": "facebook/musicgen-small",
                "sfx": "mmaudio:small_16k",
                "videoAudio": "mmaudio:small_16k",
            },
            "components": [],
            "deep": deep,
        }

    def install_minimum(self):
        self.installed = True
        self.installed_variants.add("small_16k")
        return self.setup_status(deep=True)

    def _mmaudio_variant_ready(self, variant):
        return variant in self.installed_variants or (variant == "small_16k" and self.installed)

    def _mmaudio_runtime_import_error(self):
        return ""

    def _musicgen_variant_ready(self, variant):
        return variant in self.installed_musicgen_variants or (variant == "small" and self.installed)

    def model_support_paths(self, model_id):
        marker = self.output_root / "audio-support" / (model_id.replace("/", "_").replace(":", "_") + ".weights")
        return [str(marker)]

    def prepare(self, *, kind, model_id):
        self.prepared.append((kind, model_id))
        if model_id.startswith("facebook/musicgen-"):
            return {"kind": kind, "modelId": model_id, "ready": True, "resident": True}
        return {
            "kind": kind, "modelId": model_id, "ready": True, "resident": None,
            "detail": "MMAudio is installed and ready; its isolated runtime loads weights per render, so residency is unreported.",
        }

    def install_musicgen_variant(self, variant):
        from aiwf.services.audio import AudioUnavailable
        if variant not in {"small", "medium", "melody", "stereo-small"}:
            raise AudioUnavailable(f"Unsupported MusicGen variant: {variant}")
        self.musicgen_install_calls.append(variant)
        self.installed_musicgen_variants.add(variant)
        return {"variant": variant, "installed": True}

    def install_mmaudio_variant(self, variant):
        self.variant_install_calls.append(variant)
        if variant not in {"small_16k", "large_44k_v2", "large_44k", "medium_44k", "small_44k"}:
            from aiwf.services.audio import AudioUnavailable

            raise AudioUnavailable(f"Unsupported MMAudio variant: {variant}")
        self.installed_variants.add(variant)
        return {"variant": variant, "installed": True}

    def music_model_choices(self):
        return [("MusicGen small", "facebook/musicgen-small"), ("MusicGen medium", "facebook/musicgen-medium")]

    def sfx_model_choices(self):
        return [
            ("MMAudio small 16k", "mmaudio:small_16k"),
            ("MMAudio large 44k v2", "mmaudio:large_44k_v2"),
        ]

    def video_audio_model_choices(self):
        return [("MMAudio small 16k", "mmaudio:small_16k")]

    def available_video_audio_model_choices(self):
        return self.video_audio_model_choices() if self.installed else []

    def generate(self, options):
        self.generated.append(options)
        if options.model_id.startswith("facebook/musicgen-"):
            self.parked_musicgen_variants.add(options.model_id)
        path = self.output_root / "audio" / "test.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFaudio")
        return SimpleNamespace(
            output_path=str(path),
            prompt=options.prompt,
            kind=options.kind,
            model_id=options.model_id,
            duration_seconds=options.duration_seconds,
            sample_rate=32000,
            message="Audio complete.",
            infotext="Audio music: test",
            license=audio_licenses.license_for(options.model_id),
        )

    def musicgen_model_is_parked_on_cpu(self, model_id):
        return model_id in self.parked_musicgen_variants


def test_audio_status_and_minimum_setup_routes_share_one_contract(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.audio = _AudioStub(ctx.flags.output_dir)
    client = _client(ctx)

    before = client.get("/api/pro/audio/status")
    installed = client.post("/api/pro/audio/setup/minimum")
    after = client.get("/api/pro/audio/status?deep=true")

    assert before.status_code == 200
    assert before.headers["cache-control"] == "no-store"
    assert before.json()["minimumReady"] is False
    assert before.json()["models"]["music"][0]["id"] == "facebook/musicgen-small"
    mmaudio = next(item for item in before.json()["models"]["sfx"] if item["id"] == "mmaudio:large_44k_v2")
    assert mmaudio["available"] is True
    assert mmaudio["installed"] is False
    assert mmaudio["installable"] is True
    assert mmaudio["setupRoute"]["routeKey"] == "pro.audio.mmaudio.large-44k-v2"
    assert installed.status_code == 200
    assert installed.json()["minimumReady"] is True
    assert after.json()["minimumReady"] is True


def test_video_audio_status_attaches_install_route_to_events_moss_sfx(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.video_audio_model_choices = lambda: [
        ("Video soundtrack: describe scene + MOSS-SoundEffect", "events:moss-sfx"),
        ("MMAudio small 16k", "mmaudio:small_16k"),
        ("Unmapped fixture", "custom:fixture"),
    ]
    ctx.audio = audio

    response = _client(ctx).get("/api/pro/audio/status")

    assert response.status_code == 200
    choices = {item["id"]: item for item in response.json()["models"]["videoAudio"]}
    events = choices["events:moss-sfx"]
    assert events["installed"] is False
    assert events["installable"] is True
    assert events["setupRoute"]["routeKey"] == "pro.audio.moss-sfx.v2"
    assert events["setupRoute"]["setupAction"] == "POST /api/pro/audio/setup/engine/moss-sfx"

    # Existing video-audio setup stays mapped, while unrelated choices stay unmapped.
    assert choices["mmaudio:small_16k"]["setupRoute"]["routeKey"] == "pro.video.audio.mmaudio"
    assert choices["custom:fixture"]["setupRoute"] is None


def test_audio_setup_rejects_while_model_operation_lock_is_held(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    ctx.audio = audio
    client = _client(ctx)
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(ctx)
    assert lock.acquire(blocking=False)
    try:
        response = client.post("/api/pro/audio/setup/minimum")
    finally:
        lock.release()

    assert response.status_code == 409
    assert audio.installed is False


def test_audio_route_lifecycle_tracks_the_exact_variant_support_revision(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed_musicgen_variants.add("medium")
    ctx.audio = audio
    client = _client(ctx)

    first = client.get("/api/pro/audio/status").json()
    medium_route = next(
        item for item in first["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-medium"
    )
    small_route = next(
        item for item in first["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-small"
    )
    assert medium_route["status"] == "setup-ready"
    assert small_route["status"] == "needs-setup"
    assert "audio-support" not in medium_route["detail"]

    support_file = Path(audio.model_support_paths("facebook/musicgen-medium")[0])
    support_file.parent.mkdir(parents=True, exist_ok=True)
    support_file.write_bytes(b"variant-one")
    second = client.get("/api/pro/audio/status").json()
    second_medium = next(
        item for item in second["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-medium"
    )
    assert second_medium["supportRevision"] != medium_route["supportRevision"]


def test_audio_generation_keeps_the_status_support_revision_for_the_selected_variant(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed_musicgen_variants.add("medium")
    ctx.audio = audio
    client = _client(ctx)

    before = client.get("/api/pro/audio/status").json()
    before_route = next(
        item for item in before["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-medium"
    )
    generated = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "soft piano", "kind": "music", "modelId": "facebook/musicgen-medium"},
    )
    after = client.get("/api/pro/audio/status").json()
    after_route = next(
        item for item in after["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-medium"
    )

    assert generated.status_code == 200
    assert after_route["supportRevision"] == before_route["supportRevision"]


def test_default_image_checkpoint_picker_skips_ready_ltx_video_routes():
    selectable = [
        {"id": "ltx-ready", "engineId": "ltx", "routeStatus": "request-eligible", "checkpointPathStatus": "present"},
        {"id": "sdxl-ready", "engineId": "sdxl", "routeStatus": "request-eligible", "checkpointPathStatus": "present"},
    ]

    assert pro_api._first_ready_image_checkpoint_id(selectable) == "sdxl-ready"


def test_video_lab_status_exposes_installable_soundtrack_choices(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.audio = _AudioStub(ctx.flags.output_dir)

    response = _client(ctx).get("/api/pro/video-lab/status")

    assert response.status_code == 200
    choices = {item["id"]: item for item in response.json()["audio"]["modelChoices"]}
    assert choices["mmaudio:small_16k"]["installable"] is True
    assert choices["mmaudio:small_16k"]["installed"] is False
    assert choices["mmaudio:small_16k"]["conditioningMode"] == "video-conditioned"
    assert choices["facebook/musicgen-small"]["installable"] is True
    assert choices["facebook/musicgen-small"]["installed"] is False
    assert choices["facebook/musicgen-small"]["conditioningMode"] == "prompt-only"


def test_video_lab_status_blocks_soundtrack_choices_when_mux_tools_are_missing(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio.mux_ready = False
    ctx.audio = audio

    response = _client(ctx).get("/api/pro/video-lab/status")

    assert response.status_code == 200
    status = response.json()["audio"]
    choices = {item["id"]: item for item in status["modelChoices"]}
    assert status["ready"] is False
    assert status["musicGenReady"] is False
    assert choices["mmaudio:small_16k"]["installed"] is True
    assert choices["mmaudio:small_16k"]["available"] is False
    assert choices["mmaudio:small_16k"]["ready"] is False
    assert "FFmpeg and ffprobe" in choices["mmaudio:small_16k"]["unavailableReason"]
    assert choices["facebook/musicgen-small"]["ready"] is False


def test_video_lab_audio_run_blocks_before_generation_when_mux_tools_are_missing(tmp_path):
    from fastapi import HTTPException

    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio.mux_ready = False
    audio.generate_and_mux = lambda *_args, **_kwargs: pytest.fail("must not generate without mux tools")
    ctx.audio = audio
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")

    with pytest.raises(HTTPException) as error:
        pro_api._video_lab_run_audio(
            ctx,
            source,
            pro_api.ProVideoLabRunPayload(op="audio", audio_prompt="soft soundtrack"),
        )

    assert error.value.status_code == 409
    assert "FFmpeg and ffprobe" in error.value.detail
    assert audio.generated == []


def test_video_lab_reports_mmaudio_runtime_recovery_and_blocks_prepare(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio._mmaudio_runtime_import_error = lambda: "missing torchcodec runtime"
    ctx.audio = audio
    client = _client(ctx)

    status = client.get("/api/pro/video-lab/status")
    mmaudio = next(
        item for item in status.json()["audio"]["modelChoices"]
        if item["id"] == "mmaudio:small_16k"
    )
    prepared = client.post(
        "/api/pro/video-lab/prepare-audio",
        json={"kind": "sfx", "modelId": "mmaudio:small_16k"},
    )

    assert "minimum Audio setup" in mmaudio["unavailableReason"]
    assert mmaudio["ready"] is False
    assert prepared.status_code == 409
    assert audio.prepared == []


@pytest.mark.parametrize("mux_ready", [True, False])
def test_video_lab_status_exposes_uninstalled_events_moss_setup_route(tmp_path, mux_ready):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.mux_ready = mux_ready
    audio.commercial_engine_ready = lambda _model_id: False
    audio.video_audio_model_choices = lambda: [
        ("Video soundtrack: describe scene + MOSS-SoundEffect", "events:moss-sfx"),
    ]
    audio.music_model_choices = lambda: [("ACE-Step 1.5 turbo", "acestep:1.5-turbo")]
    ctx.audio = audio

    response = _client(ctx).get("/api/pro/video-lab/status")

    assert response.status_code == 200
    choices = {item["id"]: item for item in response.json()["audio"]["modelChoices"]}
    events = choices["events:moss-sfx"]
    assert events["installed"] is False
    assert events["installable"] is mux_ready
    assert events["available"] is mux_ready
    assert events["ready"] is False
    control_route = choices["acestep:1.5-turbo"].get("setupRoute") or {}
    assert control_route.get("routeKey") != "pro.audio.moss-sfx.v2"
    assert events.get("setupRoute") == pro_api._setup_route_descriptor(
        {"routeKey": "pro.audio.moss-sfx.v2"}
    )
    assert events["setupRoute"]["setupAction"] == "POST /api/pro/audio/setup/engine/moss-sfx"
    assert audio.generated == []
    assert audio.prepared == []


def test_pro_audio_status_blocks_mmaudio_when_runtime_import_check_fails(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio._mmaudio_runtime_import_error = lambda: "missing torchcodec runtime"
    ctx.audio = audio
    client = _client(ctx)

    status = client.get("/api/pro/audio/status")
    mmaudio = next(item for item in status.json()["models"]["sfx"] if item["id"] == "mmaudio:small_16k")
    route = next(item for item in status.json()["routeLifecycle"] if item["route"] == "audio.sfx.mmaudio:small_16k")
    generated = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "rain", "kind": "sfx", "modelId": "mmaudio:small_16k"},
    )

    assert mmaudio["available"] is False
    assert mmaudio["installed"] is True
    assert "missing torchcodec runtime" in mmaudio["unavailableReason"]
    assert route["status"] == "needs-setup"
    assert generated.status_code == 409


def test_pro_video_audio_choices_require_mux_tools(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio.mux_ready = False
    ctx.audio = audio

    status = _client(ctx).get("/api/pro/audio/status")

    assert status.status_code == 200
    payload = status.json()
    assert payload["videoAudioReady"] is False
    video_choice = next(item for item in payload["models"]["videoAudio"] if item["id"] == "mmaudio:small_16k")
    assert video_choice["available"] is False
    assert video_choice["routeStatus"] == "needs-setup"
    assert "FFmpeg and ffprobe" in video_choice["unavailableReason"]
    assert audio.generated == []


def test_video_lab_audio_prepare_uses_video_audio_lifecycle_route(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    ctx.audio = audio
    saved = []
    ctx.save_settings = lambda: saved.append(True)

    response = _client(ctx).post(
        "/api/pro/video-lab/prepare-audio",
        json={"kind": "sfx", "modelId": "mmaudio:small_16k"},
    )

    assert response.status_code == 200
    assert response.json()["startupDefaultSaved"] is True
    assert ctx.settings.last_video_audio_model_id == "mmaudio:small_16k"
    assert ctx.settings.last_sana_audio_model_id == "mmaudio:small_16k"
    assert _client(ctx).get("/api/pro/audio/status").json()["defaults"]["videoAudio"] == "mmaudio:small_16k"
    assert saved == [True]
    assert audio.prepared == [("sfx", "mmaudio:small_16k")]
    route = next(
        item for item in _client(ctx).get("/api/pro/audio/status").json()["routeLifecycle"]
        if item["route"] == "audio.video.audio.mmaudio:small_16k"
    )
    assert route["status"] == "prepared"


def test_video_lab_event_soundtrack_prepare_uses_video_audio_choices(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio.video_audio_model_choices = lambda: [
        ("Video soundtrack: MOSS-SoundEffect", "events:moss-sfx"),
        *audio.sfx_model_choices(),
    ]
    audio.commercial_engine_ready = lambda model_id: model_id == "events:moss-sfx"
    ctx.audio = audio
    ctx.save_settings = lambda: None
    client = _client(ctx)

    prepared = client.post(
        "/api/pro/video-lab/prepare-audio",
        json={"kind": "music", "modelId": "events:moss-sfx"},
    )
    ordinary_audio = client.post(
        "/api/pro/audio/prepare",
        json={"kind": "sfx", "modelId": "events:moss-sfx"},
    )

    assert prepared.status_code == 200
    assert audio.prepared == [("sfx", "events:moss-sfx")]
    route = next(
        item for item in client.get("/api/pro/audio/status").json()["routeLifecycle"]
        if item["route"] == "audio.video.audio.events:moss-sfx"
    )
    assert route["status"] == "prepared"
    assert ordinary_audio.status_code == 422


def test_video_lab_musicgen_default_does_not_break_sana_audio_conditioning(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    ctx.audio = audio
    ctx.save_settings = lambda: None
    client = _client(ctx)

    response = client.post(
        "/api/pro/video-lab/prepare-audio",
        json={"kind": "music", "modelId": "facebook/musicgen-small"},
    )

    assert response.status_code == 200
    assert ctx.settings.last_video_audio_model_id == "facebook/musicgen-small"
    assert ctx.settings.last_sana_audio_model_id == ""
    sana_request = pro_api._sana_video_request_from_payload(
        ctx,
        pro_api.ProGeneratePayload(mode="video", prompt="gentle movement", generate_audio=True),
    )
    assert sana_request.audio_model_id == "mmaudio:small_16k"


def test_video_lab_audio_generation_honors_explicit_musicgen_choice_when_mmaudio_is_available(
    tmp_path, monkeypatch,
):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio.installed_musicgen_variants.add("medium")
    captured = []

    def generate_and_mux(_src, options):
        captured.append(options)
        return (
            SimpleNamespace(
                output_path=str(ctx.flags.output_dir / "audio.wav"),
                infotext=options.model_id,
                license=audio_licenses.license_for(options.model_id),
            ),
            SimpleNamespace(output_path=str(ctx.flags.output_dir / "muxed.mp4")),
        )

    audio.generate_and_mux = generate_and_mux
    ctx.audio = audio
    src = tmp_path / "input.mp4"
    src.write_bytes(b"video")
    monkeypatch.setattr(
        pro_api, "_video_lab_output",
        lambda *_args, **_kwargs: {"status": "complete", **(_args[3] if len(_args) > 3 else {})},
    )

    result = pro_api._video_lab_run_audio(
        ctx,
        src,
        pro_api.ProVideoLabRunPayload(
            op="audio", audio_prompt="warm soundtrack", audio_model="facebook/musicgen-medium",
        ),
    )

    assert result["status"] == "complete"
    assert len(captured) == 1
    assert captured[0].model_id == "facebook/musicgen-medium"
    assert captured[0].kind == "music"
    assert result["license"] == audio_licenses.license_for("facebook/musicgen-medium")


def test_video_lab_generic_output_does_not_invent_audio_license(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(pro_api, "_video_lab_probe", lambda _path: {})

    result = pro_api._video_lab_output(ctx, tmp_path / "upscaled.mp4", "Upscale complete.")

    assert result["status"] == "completed"
    assert "license" not in result


def test_video_audio_generation_uses_the_status_lifecycle_route_and_support_revision(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio.generate_and_mux = lambda _src, options: (
        SimpleNamespace(output_path=str(ctx.flags.output_dir / "audio.wav"), infotext=options.model_id),
        SimpleNamespace(output_path=str(ctx.flags.output_dir / "video-audio.mp4")),
    )
    ctx.audio = audio
    src = tmp_path / "input.mp4"
    src.write_bytes(b"video")
    monkeypatch.setattr(pro_api, "_video_lab_output", lambda *_args, **_kwargs: {"status": "complete"})

    status_before = _client(ctx).get("/api/pro/audio/status").json()
    before_route = next(
        item for item in status_before["routeLifecycle"]
        if item["route"] == "audio.video.audio.mmaudio:small_16k"
    )
    result = pro_api._video_lab_run_audio(
        ctx, src, pro_api.ProVideoLabRunPayload(op="audio", audio_prompt="soft ambient soundtrack")
    )
    status_after = _client(ctx).get("/api/pro/audio/status").json()
    after_route = next(
        item for item in status_after["routeLifecycle"]
        if item["route"] == "audio.video.audio.mmaudio:small_16k"
    )

    assert result["status"] == "complete"
    assert after_route["status"] == "completed"
    assert after_route["supportRevision"] == before_route["supportRevision"]


@pytest.mark.parametrize(("parked", "expected_resident"), [(True, False), (False, None)])
def test_video_lab_musicgen_fallback_clears_residency_only_after_exact_cpu_parking_confirmation(
    tmp_path, monkeypatch, parked, expected_resident,
):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    # Force the Video Lab fallback to MusicGen; the post-render parking check is
    # the only authority for whether its prepared residency still holds.
    audio.available_video_audio_model_choices = lambda: []
    checked_ids = []

    def generate_and_mux(_src, options):
        if parked:
            audio.parked_musicgen_variants.add(options.model_id)
        return (
            SimpleNamespace(output_path=str(ctx.flags.output_dir / "audio.wav"), infotext=options.model_id),
            SimpleNamespace(output_path=str(ctx.flags.output_dir / "video-audio.mp4")),
        )

    audio.generate_and_mux = generate_and_mux
    original_parked_check = audio.musicgen_model_is_parked_on_cpu

    def parked_check(model_id):
        checked_ids.append(model_id)
        return original_parked_check(model_id)

    audio.musicgen_model_is_parked_on_cpu = parked_check
    ctx.audio = audio
    src = tmp_path / "input.mp4"
    src.write_bytes(b"video")
    monkeypatch.setattr(pro_api, "_video_lab_output", lambda *_args, **_kwargs: {"status": "complete"})
    client = _client(ctx)

    prepared = client.post(
        "/api/pro/audio/prepare",
        json={"kind": "music", "modelId": "facebook/musicgen-small"},
    )
    result = pro_api._video_lab_run_audio(
        ctx,
        src,
        pro_api.ProVideoLabRunPayload(
            op="audio", audio_prompt="soft ambient soundtrack", audio_model="facebook/musicgen-small",
        ),
    )
    route = next(
        item for item in client.get("/api/pro/audio/status").json()["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-small"
    )

    assert prepared.status_code == 200
    assert prepared.json()["resident"] is True
    assert result["status"] == "complete"
    assert checked_ids == ["facebook/musicgen-small"]
    assert route["status"] == "completed"
    assert route["resident"] is expected_resident


def test_musicgen_generation_rejects_installed_weights_when_runtime_dependencies_are_missing(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed_musicgen_variants.add("medium")
    audio.setup_status = lambda *, deep=False: {
        **_AudioStub.setup_status(audio, deep=deep),
        "musicDependenciesReady": False,
        "musicReady": False,
    }
    ctx.audio = audio

    response = _client(ctx).post(
        "/api/pro/audio/generate",
        json={"prompt": "soft piano", "kind": "music", "modelId": "facebook/musicgen-medium"},
    )
    route = next(
        item for item in _client(ctx).get("/api/pro/audio/status").json()["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-medium"
    )

    assert response.status_code == 409
    assert "not installed yet" in response.json()["detail"]
    assert audio.generated == []
    assert route["status"] == "needs-setup"


def test_audio_variant_install_route_installs_only_the_selected_allowlisted_variant(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    ctx.audio = audio
    client = _client(ctx)

    response = client.post("/api/pro/audio/setup/mmaudio/large_44k_v2")

    assert response.status_code == 200
    assert audio.variant_install_calls == ["large_44k_v2"]
    installed = next(item for item in response.json()["models"]["sfx"] if item["id"] == "mmaudio:large_44k_v2")
    assert installed["installed"] is True


def test_audio_variant_install_route_rejects_unlisted_variant(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.audio = _AudioStub(ctx.flags.output_dir)
    client = _client(ctx)

    response = client.post("/api/pro/audio/setup/mmaudio/unknown")

    assert response.status_code == 503
    assert "Unsupported MMAudio variant" in response.json()["detail"]
    assert ctx.audio.variant_install_calls == ["unknown"]


def test_musicgen_variant_install_route_installs_only_the_selected_allowlisted_variant(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    ctx.audio = audio
    client = _client(ctx)

    response = client.post("/api/pro/audio/setup/musicgen/medium")

    assert response.status_code == 200
    assert audio.musicgen_install_calls == ["medium"]
    choice = next(item for item in response.json()["models"]["music"] if item["id"] == "facebook/musicgen-medium")
    assert choice["installed"] is True
    assert choice["installable"] is True


def test_musicgen_variant_install_route_rejects_unlisted_variant(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    ctx.audio = audio
    client = _client(ctx)

    response = client.post("/api/pro/audio/setup/musicgen/unknown")

    assert response.status_code == 503
    assert "Unsupported MusicGen variant" in response.json()["detail"]
    assert audio.musicgen_install_calls == []


def test_audio_prepare_loads_musicgen_variant_and_swap_without_generating(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed_musicgen_variants.update({"small", "medium"})
    ctx.audio = audio
    saved = []
    ctx.save_settings = lambda: saved.append(ctx.settings.last_audio_music_model_id)
    from aiwf.services.route_lifecycle import select_route

    wan_releases = []
    ctx.wan.release_cached_model_for_modality_switch = lambda: wan_releases.append(True) or True
    select_route(ctx, route="video.wan.fast_5b", model_id="wan-5b", setup_ready=True, resident=True)
    select_route(ctx, route="video.sana.480p", model_id="sana-480p", setup_ready=True, resident=True)
    client = _client(ctx)

    first = client.post("/api/pro/audio/prepare", json={"kind": "music", "modelId": "facebook/musicgen-small"})
    second = client.post("/api/pro/audio/prepare", json={"kind": "music", "modelId": "facebook/musicgen-medium"})

    assert first.status_code == second.status_code == 200
    assert first.json()["startupDefaultSaved"] is True
    assert second.json()["startupDefaultSaved"] is True
    assert ctx.settings.last_audio_music_model_id == "facebook/musicgen-medium"
    assert saved == ["facebook/musicgen-small", "facebook/musicgen-medium"]
    assert first.json()["resident"] is True
    assert second.json()["routeStatus"] == "prepared"
    routes = {item["route"]: item for item in client.get("/api/pro/audio/status").json()["routeLifecycle"]}
    assert routes["audio.music.facebook/musicgen-small"]["resident"] is False
    assert routes["audio.music.facebook/musicgen-medium"]["resident"] is True
    assert routes["video.wan.fast_5b"]["resident"] is False
    assert routes["video.sana.480p"]["resident"] is False
    assert wan_releases == [True, True]
    assert audio.prepared == [("music", "facebook/musicgen-small"), ("music", "facebook/musicgen-medium")]
    assert audio.generated == []
    assert audio.musicgen_install_calls == []


def test_failed_musicgen_variant_switch_clears_previous_residency_receipt(tmp_path):
    from aiwf.services.audio import AudioUnavailable
    from aiwf.services.route_lifecycle import lifecycle_snapshot, select_route

    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed_musicgen_variants.update({"small", "medium"})
    ctx.audio = audio
    select_route(
        ctx,
        route="audio.music.facebook/musicgen-small",
        model_id="facebook/musicgen-small",
        setup_ready=True,
        resident=True,
    )

    def fail_medium(*, kind, model_id):
        audio.prepared.append((kind, model_id))
        raise AudioUnavailable("MusicGen medium failed to load")

    audio.prepare = fail_medium
    client = _client(ctx)
    response = client.post(
        "/api/pro/audio/prepare",
        json={"kind": "music", "modelId": "facebook/musicgen-medium"},
    )

    routes = {item["route"]: item for item in lifecycle_snapshot(ctx)}
    assert response.status_code == 503
    assert routes["audio.music.facebook/musicgen-small"]["resident"] is False
    assert routes["audio.music.facebook/musicgen-medium"]["status"] == "failed"
    assert audio.prepared == [("music", "facebook/musicgen-medium")]


def test_audio_status_reconciles_musicgen_residency_with_backend_probe(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed_musicgen_variants.add("small")
    audio._musicgen_model_is_resident = lambda _model_id: False
    ctx.audio = audio
    pro_api.select_route(
        ctx, route="audio.music.facebook/musicgen-small",
        model_id="facebook/musicgen-small", setup_ready=True, resident=True,
    )

    status = _client(ctx).get("/api/pro/audio/status").json()
    route = next(item for item in status["routeLifecycle"] if item["route"] == "audio.music.facebook/musicgen-small")
    choice = next(item for item in status["models"]["music"] if item["id"] == "facebook/musicgen-small")

    assert route["resident"] is False
    assert choice["resident"] is False


def test_video_family_switch_clears_old_residency_before_replacement_attempt(tmp_path):
    from aiwf.services.route_lifecycle import lifecycle_snapshot, select_route

    ctx = _ctx(tmp_path)
    select_route(ctx, route="video.wan.fast_5b", model_id="same-model", setup_ready=True, resident=True)
    select_route(ctx, route="video.sana.480p", model_id="sana", setup_ready=True, resident=True)

    pro_api._clear_prior_video_family_residency(ctx, "video.wan.high_low")

    routes = {item["route"]: item for item in lifecycle_snapshot(ctx)}
    assert routes["video.wan.fast_5b"]["resident"] is False
    assert routes["video.sana.480p"]["resident"] is True


def test_musicgen_prepare_then_generate_updates_route_residency_after_cpu_parking(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    ctx.audio = audio
    client = _client(ctx)

    prepared = client.post(
        "/api/pro/audio/prepare",
        json={"kind": "music", "modelId": "facebook/musicgen-small"},
    )
    before = client.get("/api/pro/audio/status").json()
    before_route = next(
        item for item in before["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-small"
    )
    generated = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "soft piano", "kind": "music", "modelId": "facebook/musicgen-small"},
    )
    after = client.get("/api/pro/audio/status").json()
    after_route = next(
        item for item in after["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-small"
    )

    assert prepared.status_code == 200
    assert prepared.json()["resident"] is True
    assert before_route["resident"] is True
    assert generated.status_code == 200
    assert after_route["status"] == "completed"
    assert after_route["resident"] is False


@pytest.mark.parametrize("parking_probe", [False, None], ids=["not-parked", "probe-missing"])
def test_musicgen_generation_clears_stale_residency_when_cpu_parking_is_unconfirmed(tmp_path, parking_probe):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    audio.musicgen_model_is_parked_on_cpu = parking_probe
    ctx.audio = audio
    client = _client(ctx)

    prepared = client.post(
        "/api/pro/audio/prepare",
        json={"kind": "music", "modelId": "facebook/musicgen-small"},
    )
    generated = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "soft piano", "kind": "music", "modelId": "facebook/musicgen-small"},
    )
    route = next(
        item for item in client.get("/api/pro/audio/status").json()["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-small"
    )

    assert prepared.status_code == 200
    assert prepared.json()["resident"] is True
    assert generated.status_code == 200
    assert route["status"] == "completed"
    assert route["resident"] is None


def test_audio_prepare_reports_mmaudio_setup_ready_without_residency(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed_variants.add("large_44k_v2")
    ctx.audio = audio

    response = _client(ctx).post(
        "/api/pro/audio/prepare",
        json={"kind": "sfx", "modelId": "mmaudio:large_44k_v2"},
    )

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["resident"] is None
    assert response.json()["routeStatus"] == "prepared"
    route = next(
        item for item in _client(ctx).get("/api/pro/audio/status").json()["routeLifecycle"]
        if item["route"] == "audio.sfx.mmaudio:large_44k_v2"
    )
    assert route["resident"] is None
    assert route["status"] == "prepared"
    assert audio.generated == []
    assert audio.variant_install_calls == []


def test_audio_prepare_rejects_uninstalled_variant_without_implicit_install(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    ctx.audio = audio

    response = _client(ctx).post(
        "/api/pro/audio/prepare",
        json={"kind": "music", "modelId": "facebook/musicgen-medium"},
    )

    assert response.status_code == 409
    assert "Install it explicitly" in response.json()["detail"]
    assert audio.prepared == []
    assert audio.musicgen_install_calls == []
    assert audio.generated == []


@pytest.mark.parametrize(("kind", "model_id"), [("music", "mmaudio:small_16k"), ("sfx", "facebook/musicgen-small")])
def test_audio_prepare_rejects_cross_kind_model(tmp_path, kind, model_id):
    ctx = _ctx(tmp_path)
    ctx.audio = _AudioStub(ctx.flags.output_dir)

    response = _client(ctx).post("/api/pro/audio/prepare", json={"kind": kind, "modelId": model_id})

    assert response.status_code == 422
    assert ctx.audio.prepared == []


def test_audio_prepare_is_blocked_while_workflow_run_is_active(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    ctx.audio = audio
    ctx._pro_workflow_run_service = SimpleNamespace(has_active_runs=lambda: True)

    response = _client(ctx).post(
        "/api/pro/audio/prepare",
        json={"kind": "music", "modelId": "facebook/musicgen-small"},
    )

    assert response.status_code == 409
    assert "workflow work" in response.json()["detail"]
    assert audio.prepared == []


def test_audio_generate_route_returns_playable_output_url(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    ctx.audio = audio
    client = _client(ctx)

    response = client.post(
        "/api/pro/audio/generate",
        json={
            "prompt": "quiet modular synth pulse",
            "kind": "music",
            "modelId": "facebook/musicgen-small",
            "durationSeconds": 8,
            "cfgCoef": 3,
            "steps": 25,
            "seed": 11,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "complete"
    assert data["url"].startswith("/api/pro/outputs/")
    assert client.get(data["url"]).content == b"RIFFaudio"
    assert audio.generated[0].model_id == "facebook/musicgen-small"
    assert audio.generated[0].seed == 11


def test_audio_generate_requires_the_selected_model_variant_to_be_installed(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.sfx_model_choices = lambda: [("MMAudio large 44k", "mmaudio:large_44k_v2")]
    ctx.audio = audio
    client = _client(ctx)

    response = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "rain", "kind": "sfx", "modelId": "mmaudio:large_44k_v2"},
    )

    assert response.status_code == 409
    assert "not installed" in response.json()["detail"]
    assert audio.generated == []


def test_audio_generate_route_rejects_video_conditioned_kind(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.audio = _AudioStub(ctx.flags.output_dir)
    client = _client(ctx)

    response = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "footsteps", "kind": "video_audio", "modelId": "mmaudio:small_16k"},
    )

    assert response.status_code == 422


def test_audio_generate_route_rejects_model_from_other_audio_kind(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    ctx.audio = audio
    client = _client(ctx)

    response = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "footsteps", "kind": "sfx", "modelId": "facebook/musicgen-small"},
    )

    assert response.status_code == 422
    assert "sound effects model" in response.json()["detail"]
    assert audio.generated == []


def test_audio_status_marks_audiogen_unavailable_and_generate_rejects_it(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.sfx_model_choices = lambda: [
        ("MMAudio small 16k", "mmaudio:small_16k"),
        ("AudioGen medium (AudioCraft required)", "facebook/audiogen-medium"),
    ]
    ctx.audio = audio
    client = _client(ctx)

    status = client.get("/api/pro/audio/status")
    audiogen = next(item for item in status.json()["models"]["sfx"] if item["id"] == "facebook/audiogen-medium")
    response = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "rain", "kind": "sfx", "modelId": "facebook/audiogen-medium"},
    )

    assert audiogen["available"] is False
    assert "AudioCraft" in audiogen["unavailableReason"]
    assert response.status_code == 422
    assert "AudioCraft is not installed" in response.json()["detail"]
    assert audio.generated == []


def test_audio_generate_route_rejects_unknown_kind(tmp_path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    ctx.audio = audio
    client = _client(ctx)

    response = client.post(
        "/api/pro/audio/generate",
        json={"prompt": "footsteps", "kind": "bogus", "modelId": "mmaudio:small_16k"},
    )

    assert response.status_code == 422
    assert audio.generated == []


def test_explicit_model_load_route_loads_selected_model_without_generating(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    ctx = _ctx(tmp_path)
    saved_defaults = []
    ctx.save_settings = lambda: saved_defaults.append(ctx.settings.last_checkpoint_id)
    from aiwf.services.route_lifecycle import select_route

    ctx.audio = SimpleNamespace(release_cached_model_for_modality_switch=lambda: True)
    select_route(
        ctx, route="audio.music.facebook/musicgen-small", model_id="facebook/musicgen-small",
        setup_ready=True, resident=True,
    )
    loaded = []
    monkeypatch.setattr(ctx.generation.backend, "can_preload_checkpoint_locally", lambda model_id: model_id == "model-a", raising=False)
    ctx.generation.load_checkpoint = lambda model_id: loaded.append(model_id)
    monkeypatch.setattr(ctx.generation.backend, "is_checkpoint_loaded", lambda model_id: model_id == "model-a", raising=False)
    client = _client(ctx)

    response = client.post("/api/pro/models/load", json={"modelId": "model-a"})

    assert response.status_code == 200
    assert response.json()["modelLoad"]["status"] == "loaded"
    assert response.json()["modelLoad"]["modelId"] == "model-a"
    assert response.json()["startupDefaultSaved"] is True
    assert ctx.settings.last_checkpoint_id == "model-a"
    assert saved_defaults == ["model-a"]
    assert loaded == ["model-a"]
    audio_state = next(
        item for item in response.json()["runtime"]["routeLifecycle"]
        if item["route"] == "audio.music.facebook/musicgen-small"
    )
    assert audio_state["resident"] is False
    assert ctx.generation.submitted == []


def test_explicit_model_load_defers_when_audio_cache_cannot_be_released(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    ctx = _ctx(tmp_path)
    release_blocked = {"value": True}
    ctx.audio = SimpleNamespace(
        release_cached_model_for_modality_switch=lambda: not release_blocked["value"]
    )
    loaded = []
    monkeypatch.setattr(ctx.generation.backend, "can_preload_checkpoint_locally", lambda _model_id: True, raising=False)
    monkeypatch.setattr(ctx.generation.backend, "is_checkpoint_loaded", lambda _model_id: True, raising=False)
    monkeypatch.setattr("aiwf.services.model_startup._gpu_headroom_status", lambda _ctx, _model: None)
    ctx.generation.load_checkpoint = lambda model_id: loaded.append(model_id)

    response = _client(ctx).post("/api/pro/models/load", json={"modelId": "model-a"})

    assert response.status_code == 409
    assert "Audio still owns or is using its model" in response.json()["detail"]
    assert ctx._pro_model_load_state["status"] == "deferred"
    assert loaded == []

    release_blocked["value"] = False
    retry = _client(ctx).post("/api/pro/models/load", json={"modelId": "model-a"})

    assert retry.status_code == 200
    assert retry.json()["modelLoad"]["status"] == "loaded"
    assert ctx._pro_model_load_state["status"] == "loaded"
    assert loaded == ["model-a"]


def test_explicit_model_load_defers_when_gpu_headroom_is_insufficient(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    from aiwf.services import model_startup

    ctx = _ctx(tmp_path)
    loaded = []
    monkeypatch.setattr(ctx.generation.backend, "can_preload_checkpoint_locally", lambda _model_id: True, raising=False)
    monkeypatch.setattr(model_startup, "_gpu_headroom_status", lambda _ctx, _model: "Only 1.0 GB VRAM is free.")
    ctx.generation.load_checkpoint = lambda model_id: loaded.append(model_id)

    response = _client(ctx).post("/api/pro/models/load", json={"modelId": "model-a"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Only 1.0 GB VRAM is free."
    assert ctx._pro_model_load_state["status"] == "deferred"
    assert loaded == []


def test_explicit_model_load_does_not_send_ltx_video_model_to_image_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    ctx = _ctx(tmp_path)
    loaded = []
    ctx.generation.load_checkpoint = lambda model_id: loaded.append(model_id)
    monkeypatch.setattr(pro_api, "_is_checkpoint_id_selectable", lambda _ctx, _model_id: True)
    monkeypatch.setattr(pro_api, "_checkpoint_engine_id", lambda _ctx, _model_id: "ltx")

    response = _client(ctx).post("/api/pro/models/load", json={"modelId": "ltx-video"})

    assert response.status_code == 422
    assert "video route" in response.json()["detail"]
    assert loaded == []


def test_pro_generate_is_rejected_while_model_and_support_assets_are_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    ctx = _ctx(tmp_path)
    ctx._pro_model_load_state = {
        "status": "loading",
        "modelId": "model-a",
        "detail": "Loading model and support assets",
    }
    client = _client(ctx)

    response = client.post("/api/pro/generate", json={"prompt": "test", "model_id": "model-a"})

    assert response.status_code == 409
    assert "still loading" in response.json()["detail"]
    assert ctx.generation.submitted == []


def test_explicit_model_load_rejects_when_another_load_owns_operation_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    ctx = _ctx(tmp_path)
    loaded = []
    monkeypatch.setattr(ctx.generation.backend, "can_preload_checkpoint_locally", lambda _model_id: True, raising=False)
    ctx.generation.load_checkpoint = lambda model_id: loaded.append(model_id)
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(ctx)
    assert lock.acquire(blocking=False)
    try:
        response = _client(ctx).post("/api/pro/models/load", json={"modelId": "model-a"})
    finally:
        lock.release()

    assert response.status_code == 409
    assert loaded == []


def test_explicit_model_load_waits_for_active_pro_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    ctx = _ctx(tmp_path)
    loaded = []
    monkeypatch.setattr(ctx.generation.backend, "can_preload_checkpoint_locally", lambda _model_id: True, raising=False)
    ctx.generation.load_checkpoint = lambda model_id: loaded.append(model_id)
    ctx._pro_workflow_run_service = SimpleNamespace(has_active_runs=lambda: True)

    response = _client(ctx).post("/api/pro/models/load", json={"modelId": "model-a"})

    assert response.status_code == 409
    assert "generation job is active" in response.json()["detail"]
    assert ctx._pro_model_load_state["status"] == "deferred"
    assert loaded == []


def test_workflow_submit_rejects_while_model_operation_lock_is_held(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_PRO_STARTUP_MODEL_LOAD", "0")
    ctx = _ctx(tmp_path)
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(ctx)
    assert lock.acquire(blocking=False)
    try:
        response = _client(ctx).post(
            "/api/pro/workflows/runs",
            json={"workflow": {"name": "locked", "steps": [{"id": "generate", "type": "txt2img", "params": {"prompt": "test", "checkpoint_id": "model-a"}}]}},
        )
    finally:
        lock.release()

    assert response.status_code == 409
    assert "model operation is in progress" in response.json()["detail"]


def test_direct_generation_is_blocked_while_workflow_run_is_active(tmp_path):
    ctx = _ctx(tmp_path)
    ctx._pro_workflow_run_service = SimpleNamespace(has_active_runs=lambda: True)

    response = _client(ctx).post("/api/pro/generate", json={"prompt": "do not overlap", "model_id": "model-a"})

    assert response.status_code == 409
    assert "workflow generation job is active" in response.json()["detail"]
    assert ctx.generation.submitted == []

    audio_response = _client(ctx).post(
        "/api/pro/audio/generate",
        json={"prompt": "do not overlap", "kind": "music", "modelId": "facebook/musicgen-small"},
    )
    assert audio_response.status_code == 409
    assert ctx.generation.submitted == []


def test_direct_image_generation_releases_cached_cross_modality_models_first(tmp_path):
    ctx = _ctx(tmp_path)
    events = []
    ctx.audio = SimpleNamespace(release_cached_model_for_modality_switch=lambda: events.append("audio") or True)
    ctx.wan.release_cached_model_for_modality_switch = lambda: events.append("wan") or True
    ctx.sana_video.unload = lambda: events.append("sana") or True
    original_submit = ctx.generation.submit

    def submit_after_release(request, **kwargs):
        events.append("image-submit")
        return original_submit(request, **kwargs)

    ctx.generation.submit = submit_after_release

    response = _client(ctx).post("/api/pro/generate", json={"prompt": "switch modalities", "model_id": "model-a"})

    assert response.status_code == 200, response.text
    assert events == ["audio", "wan", "sana", "image-submit"]


def test_pro_audio_and_image_switches_reject_when_sana_cannot_release_video_tenant(tmp_path):
    ctx = _ctx(tmp_path / "image")
    ctx.sana_video.release_cached_model_for_modality_switch = lambda: False
    image_response = _client(ctx).post(
        "/api/pro/generate",
        json={"prompt": "do not overlap", "modelId": "model-a"},
    )
    assert image_response.status_code == 409
    assert "another GPU tenant owns" in image_response.json()["detail"]
    assert ctx.generation.submitted == []

    audio_ctx = _ctx(tmp_path / "audio")
    audio = _AudioStub(audio_ctx.flags.output_dir)
    audio.installed = True
    audio_ctx.audio = audio
    audio_ctx.sana_video.release_cached_model_for_modality_switch = lambda: False
    audio_response = _client(audio_ctx).post(
        "/api/pro/audio/generate",
        json={
            "prompt": "do not overlap",
            "kind": "music",
            "modelId": "facebook/musicgen-small",
        },
    )
    assert audio_response.status_code == 409
    assert "another GPU tenant owns" in audio_response.json()["detail"]
    assert audio.generated == []


def test_sana_runtime_block_does_not_offer_model_bundle_when_snapshot_is_complete(tmp_path, monkeypatch):
    from aiwf.services import pipeline_preflight

    ctx = _ctx(tmp_path)
    model_path = tmp_path / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    _seed_sana_video_snapshot(model_path)
    checkpoint = Checkpoint(
        id="sana-video-ready-assets",
        title="Sana Video 480p",
        filename=model_path.name,
        path=str(model_path),
        architecture="sana_video",
    )
    ctx.generation.list_checkpoints = lambda: [checkpoint]
    monkeypatch.setattr(
        pipeline_preflight,
        "preflight_sana_video_pipeline",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=False,
            metadata={"model_installed": "true"},
            warnings=("Sana Video pipeline classes are unavailable.",),
            message=lambda: "Sana Video runtime is unavailable.",
        ),
    )

    selectable, blocked = pro_api._selectable_checkpoint_payloads(ctx)

    assert selectable == []
    assert len(blocked) == 1
    assert blocked[0]["status"] == "blocked-runtime"
    assert blocked[0]["setupBundleKey"] is None
    assert blocked[0]["setupRoute"]["setupBundleKey"] is None
    assert "model files are already present" in blocked[0]["suggestedAction"]
    assert "reinstalling the model files is not needed" in blocked[0]["suggestedAction"]


def test_sana_video_inventory_distinguishes_t2v_from_i2v_runtime(tmp_path, monkeypatch):
    from aiwf.infrastructure.diffusers import checkpoints

    model_path = tmp_path / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    model_path.mkdir(parents=True)
    service = SimpleNamespace(
        default_model_path=lambda _variant="480p": model_path,
        runtime_available=lambda image_to_video=False: not image_to_video,
    )
    monkeypatch.setattr(pro_api, "_pro_sana_video_backend_enabled", lambda: True)
    monkeypatch.setattr(pro_api, "_sana_video_service", lambda _ctx: service)
    monkeypatch.setattr(checkpoints, "sana_video_missing_local_files", lambda _path: [])

    payload = pro_api._sana_video_model_payload(_ctx(tmp_path), "480p")

    assert payload["status"] == "Ready"
    assert payload["routeStatus"] == "request-eligible"
    assert payload["generationModes"] == {"textToVideo": True, "imageToVideo": False}


def test_sana_video_inventory_keeps_i2v_only_runtime_selectable(tmp_path, monkeypatch):
    from aiwf.infrastructure.diffusers import checkpoints

    model_path = tmp_path / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    model_path.mkdir(parents=True)
    service = SimpleNamespace(
        default_model_path=lambda _variant="480p": model_path,
        runtime_available=lambda image_to_video=False: image_to_video,
    )
    monkeypatch.setattr(pro_api, "_pro_sana_video_backend_enabled", lambda: True)
    monkeypatch.setattr(pro_api, "_sana_video_service", lambda _ctx: service)
    monkeypatch.setattr(checkpoints, "sana_video_missing_local_files", lambda _path: [])

    payload = pro_api._sana_video_model_payload(_ctx(tmp_path), "480p")

    assert payload["status"] == "Ready"
    assert payload["routeStatus"] == "request-eligible"
    assert payload["generationModes"] == {"textToVideo": False, "imageToVideo": True}
