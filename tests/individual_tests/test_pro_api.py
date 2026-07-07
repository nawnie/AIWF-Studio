from __future__ import annotations

import base64
import io
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from aiwf.app_pro import create_app
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobRecord, JobState, SavedArtifact
from aiwf.core.domain.models import Checkpoint, SamplerInfo
from aiwf.core.domain.sana_video import SanaVideoProgressEvent, SanaVideoRequest, SanaVideoResult
from aiwf.services.model_download import ModelDownloadService
from aiwf.services.pipeline_readiness import PipelineReadinessRecord
from aiwf.web import pro_api


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
        return [
            Checkpoint(
                id="model-a",
                title="Model A",
                filename="model-a.safetensors",
                path="F:/models/model-a.safetensors",
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

    def default_model_path(self):
        return self.output_dir.parent / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"

    def generate(self, request, *, on_progress=None):
        self.submitted.append(request)
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


class _Enhance:
    def __init__(self):
        self.calls = []

    def list_upscalers(self):
        return [SimpleNamespace(id="upscale-a")]

    def list_restorers(self):
        return [SimpleNamespace(id="restore-a")]

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
    settings = UserSettings(default_sampler="euler_a", default_width=640, default_height=768)
    controlnet = SimpleNamespace(
        list_models=lambda: [SimpleNamespace(id="control-a")],
        list_modules=lambda: ["none", "canny"],
    )
    segment = SimpleNamespace(list_models=lambda: [SimpleNamespace(id="sam-b")])
    enhance = _Enhance()
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
        vsr=_Vsr(),
        runtime_port=9876,
    )


def _client(ctx, frontend_dist: Path | None = None):
    return TestClient(create_app(ctx, frontend_dist=frontend_dist or Path("__missing_frontend_dist__")))


def test_runtime_endpoint_reports_warm_latency_under_budget(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)
    client.get("/api/pro/runtime")

    started = time.perf_counter()
    response = client.get("/api/pro/runtime")
    roundtrip_ms = (time.perf_counter() - started) * 1000

    assert response.status_code == 200
    server_ms = float(response.headers["X-AIWF-Elapsed-Ms"])
    assert server_ms < 75
    assert roundtrip_ms < 75


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


def test_bootstrap_returns_catalog_defaults_runtime_and_recent_images(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.generation.submit(GenerationRequest(prompt="seed recent"))
    client = _client(ctx)

    response = client.get("/api/pro/bootstrap")

    assert response.status_code == 200
    data = response.json()
    assert data["runtime"]["device"] == "CPU (test)"
    assert data["settings"]["width"] == 640
    assert data["checkpoints"][0]["id"] == "model-a"
    assert data["checkpoints"][0]["engineId"] == "sdxl"
    assert data["checkpoints"][0]["engineLabel"] == "Stable Diffusion XL"
    assert data["checkpoints"][0]["generationPreset"]["sampler"] == "dpmpp_2m"
    assert data["checkpoints"][0]["generationPreset"]["width"] == 1024
    assert {"sdxl", "sana_video", "wan"}.issubset({item["id"] for item in data["engines"]})
    assert any(item["engineId"] == "sana_video" for item in data["checkpoints"])
    assert any(item["engineId"] == "wan" and item["kind"] == "video" for item in data["checkpoints"])
    assert "path" not in data["checkpoints"][0]
    assert data["samplers"][0]["supportsKarras"] is False
    assert data["recentImages"][0]["dataUrl"].startswith("data:image/png;base64,")


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
    assert data["counts"]["blockedCheckpoints"] == 1
    assert data["blockedCheckpoints"][0]["status"] == "broken-runtime"
    assert "missing the expected CLIP text model" in data["blockedCheckpoints"][0]["reason"]


def test_bootstrap_hides_qwen_nunchaku_from_v1_app(tmp_path):
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
    assert "qwen-nunchaku" not in {item["id"] for item in data["blockedCheckpoints"]}
    assert data["settings"]["checkpointId"] == "model-a"


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


def test_model_reorganize_rereads_headers_and_moves_confident_matches(tmp_path):
    ctx = _ctx(tmp_path)
    models = tmp_path / "models"
    misplaced = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(b"GGUF")
    client = _client(ctx)

    response = client.post("/api/pro/models/reorganize")

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["moved"] == 1
    assert data["actions"][0]["destSubdir"] == "flux/GGUF"
    assert not misplaced.exists()
    assert (models / "flux" / "GGUF" / "flux1-dev-Q5_K_M.gguf").is_file()


def test_model_unload_endpoint_calls_backend_unload(tmp_path):
    ctx = _ctx(tmp_path)
    calls = []
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


def test_generate_maps_payload_and_returns_first_image(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
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

    assert response.status_code == 200
    request = ctx.generation.submitted[0]
    assert request.mode == GenerationMode.TXT2IMG
    assert request.prompt == "cat"
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
    assert data["status"] == "completed"
    assert data["job"]["state"] == "completed"
    assert data["seeds"] == [1234]
    assert data["artifacts"][0]["path"].endswith("generated.png")


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
    assert detail["status"] == "coming-soon"
    assert "blocked for the v1 app" in detail["reason"]


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


def test_generate_maps_sana_video_payload_and_serves_output(tmp_path):
    ctx = _ctx(tmp_path)
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

    assert response.status_code == 200
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
    assert ctx.settings.default_width == 768
    assert ctx.settings.default_height == 1024
    assert ctx.settings.enable_live_preview is False
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
    assert saved_settings == [True]


def test_downloads_endpoint_reports_catalog_install_state(tmp_path):
    ctx = _ctx(tmp_path)
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
    assert "qwen-nunchaku-image-lightning-int4-r32" not in {item["key"] for item in data["catalog"]}
    assert data["categories"][0]["destination"]


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


def test_downloads_catalog_endpoint_blocks_gated_entries(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post("/api/pro/downloads/catalog/hf-sd35-medium")

    assert response.status_code == 422
    assert "requires upstream access" in response.json()["detail"]


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
