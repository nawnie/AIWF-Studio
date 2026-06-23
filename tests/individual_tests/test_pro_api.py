from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

from aiwf.app_pro import create_app
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobRecord, JobState, SavedArtifact
from aiwf.core.domain.models import Checkpoint, SamplerInfo


class _Devices:
    def describe(self):
        return "CPU (test)"


class _Generation:
    backend = SimpleNamespace(devices=_Devices())

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.submitted = []
        self._recent = []

    def list_checkpoints(self):
        return [
            Checkpoint(
                id="model-a",
                title="Model A",
                filename="model-a.safetensors",
                path="F:/models/model-a.safetensors",
            )
        ]

    def list_samplers(self):
        return [SamplerInfo(id="euler_a", label="Euler a")]

    def list_loras(self):
        return []

    def active_job(self):
        return None

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


def _ctx(tmp_path: Path):
    output_dir = tmp_path / "outputs"
    settings = UserSettings(default_sampler="euler_a", default_width=640, default_height=768)
    return SimpleNamespace(
        flags=RuntimeFlags(data_dir=tmp_path, output_dir=output_dir),
        settings=settings,
        generation=_Generation(output_dir),
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
    assert "path" not in data["checkpoints"][0]
    assert data["samplers"][0]["supportsKarras"] is False
    assert data["recentImages"][0]["dataUrl"].startswith("data:image/png;base64,")


def test_runtime_does_not_submit_generation(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.get("/api/pro/runtime")

    assert response.status_code == 200
    assert response.json()["status"] == "idle"
    assert ctx.generation.submitted == []


def test_generate_maps_payload_and_returns_first_image(tmp_path):
    ctx = _ctx(tmp_path)
    client = _client(ctx)

    response = client.post(
        "/api/pro/generate",
        json={
            "prompt": "cat",
            "negativePrompt": "blurry",
            "checkpointTitle": "Model A",
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
    assert data["seeds"] == [1234]
    assert data["artifacts"][0]["path"].endswith("generated.png")


def test_generate_rejects_unbounded_batch(tmp_path):
    client = _client(_ctx(tmp_path))

    response = client.post(
        "/api/pro/generate",
        json={"prompt": "cat", "batchSize": 3, "batchCount": 2},
    )

    assert response.status_code == 422


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
