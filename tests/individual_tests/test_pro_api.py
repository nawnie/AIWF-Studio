from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

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

    def list_loras(self):
        return [SimpleNamespace(id="style-a", title="Style A")]

    def active_job(self):
        return self._active

    def pending_count(self):
        return 0

    def recent_jobs(self, _limit=20):
        return list(self._recent)

    def submit(self, request: GenerationRequest):
        self.submitted.append(request)
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


def _ctx(tmp_path: Path):
    output_dir = tmp_path / "outputs"
    flags = RuntimeFlags(data_dir=tmp_path, output_dir=output_dir)
    settings = UserSettings(default_sampler="euler_a", default_width=640, default_height=768)
    controlnet = SimpleNamespace(
        list_models=lambda: [SimpleNamespace(id="control-a")],
        list_modules=lambda: ["none", "canny"],
    )
    segment = SimpleNamespace(list_models=lambda: [SimpleNamespace(id="sam-b")])
    enhance = SimpleNamespace(
        list_upscalers=lambda: [SimpleNamespace(id="upscale-a")],
        list_restorers=lambda: [SimpleNamespace(id="restore-a")],
    )
    faceswap = SimpleNamespace(
        list_models=lambda: [SimpleNamespace(id="inswapper")],
        list_face_models=lambda: [SimpleNamespace(id="saved-face")],
    )
    wan = SimpleNamespace(
        list_local_models=lambda: ["wan-a.safetensors"],
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
        runtime_port=9876,
    )


def _client(ctx, frontend_dist: Path | None = None):
    return TestClient(create_app(ctx, frontend_dist=frontend_dist or Path("__missing_frontend_dist__")))


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
    assert {"sdxl", "sana_video"}.issubset({item["id"] for item in data["engines"]})
    assert any(item["engineId"] == "sana_video" for item in data["checkpoints"])
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
    assert request.enable_hr is False
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

    def fail_submit(request: GenerationRequest):
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
    assert data["runtime"]["backend"] == "diffusers"


def test_downloads_endpoint_reports_catalog_install_state(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.get("/api/pro/downloads")

    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["catalog"] > 0
    assert data["counts"]["installed"] == 0
    assert any(item["key"] == "hf-sdxl-base" for item in data["catalog"])
    assert data["categories"][0]["destination"]


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
