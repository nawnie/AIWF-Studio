from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from aiwf.api.v1.routes import build_router
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobProgress, JobRecord, JobState
from aiwf.core.domain.models import Checkpoint, SamplerInfo
from aiwf.core.interfaces.plugins import PluginInfo
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.services.optimization import CapabilityDetector, OptimizationPlanner
from aiwf.services.optimization_diagnostics import OptimizationDiagnosticsService


def _b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeGeneration:
    def __init__(self):
        self.submitted = []
        self.progress_job = None

    def submit(self, request, init_images=None, mask_images=None):
        self.submitted.append((request, init_images, mask_images))
        image = Image.new("RGB", (16, 16), "blue")
        return JobRecord(
            request=request,
            state=JobState.COMPLETED,
            result=GenerationResult(
                job_id=uuid4(),
                images=[image],
                seeds=[request.seed],
                infotexts=["ok"],
                mode=request.mode,
            ),
        )

    def list_checkpoints(self):
        return [Checkpoint(id="model-a", title="Model A", filename="a.safetensors", path="models/a.safetensors")]

    def list_samplers(self):
        return [SamplerInfo(id="euler_a", label="Euler a")]

    def list_loras(self):
        return []

    def list_vaes(self):
        return []

    def resolve_checkpoint(self, _checkpoint_id=None):
        return self.list_checkpoints()[0]

    def active_job(self):
        return self.progress_job

    def recent_jobs(self, _limit=20):
        return [self.progress_job] if self.progress_job else []

    def get_job(self, job_id):
        return self.progress_job if self.progress_job and self.progress_job.id == job_id else None

    def interrupt(self):
        return None


class FakeControlNet:
    def list_models(self):
        return []

    def model_ids(self):
        return ["control-a"]

    def list_modules(self):
        return ["none", "canny"]


class FakePlots:
    def run(self, _request):
        image = Image.new("RGB", (16, 16), "red")
        return SimpleNamespace(labels=["seed=1"], images=[image], grid=image, infotexts=["plot"])


def make_client(fake_generation=None, optimization_diagnostics=None):
    generation = fake_generation or FakeGeneration()
    ctx = SimpleNamespace(
        generation=generation,
        controlnet=FakeControlNet(),
        plots=FakePlots(),
        settings=SimpleNamespace(save_images=True, optimization_profile_id="balanced_sdpa_fp16"),
        plugins=SimpleNamespace(list_plugins=lambda: [PluginInfo(id="hello", name="hello", version="1")]),
    )
    if optimization_diagnostics is not None:
        ctx.optimization_diagnostics = optimization_diagnostics
    app = FastAPI()
    app.include_router(build_router(ctx))
    return TestClient(app), generation


def test_sdapi_txt2img_maps_a1111_payload_to_generation_request():
    client, generation = make_client()

    response = client.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": "cat",
            "sampler_name": "Euler a",
            "n_iter": 2,
            "enable_hr": True,
            "denoising_strength": 0.42,
            "override_settings": {"sd_model_checkpoint": "model-a"},
        },
    )

    assert response.status_code == 200
    request = generation.submitted[0][0]
    assert request.mode == GenerationMode.TXT2IMG
    assert request.prompt == "cat"
    assert request.batch_count == 2
    assert request.sampler == "euler_a"
    assert request.hr_denoising_strength == 0.42
    assert request.checkpoint_id == "model-a"
    assert response.json()["images"]


def test_sdapi_txt2img_maps_a1111_override_and_controlnet_aliases():
    client, generation = make_client()
    control = Image.new("RGB", (16, 16), "white")

    response = client.post(
        "/sdapi/v1/txt2img",
        json={
            "prompt": "cat",
            "sampler_name": "DPM++ 2M Karras",
            "override_settings": {"CLIP_stop_at_last_layers": 3},
            "alwayson_scripts": {
                "ControlNet": {
                    "args": [
                        {
                            "enabled": True,
                            "model": "control-a",
                            "module": "canny",
                            "input_image": _b64(control),
                            "detect_resolution": 768,
                            "pixel_perfect": True,
                        }
                    ]
                }
            },
        },
    )

    assert response.status_code == 200
    request = generation.submitted[0][0]
    assert request.sampler == "dpmpp_2m_karras"
    assert request.clip_skip == 3
    assert len(request.controlnet_units) == 1
    unit = request.controlnet_units[0]
    assert unit.model == "control-a"
    assert unit.image
    assert unit.processor_res == 768


def test_sdapi_img2img_maps_a1111_inpaint_fields():
    client, generation = make_client()
    source = Image.new("RGB", (16, 16), "white")
    mask = Image.new("L", (16, 16), 0)
    mask.paste(255, (4, 4, 12, 12))

    response = client.post(
        "/sdapi/v1/img2img",
        json={
            "prompt": "painted repair",
            "init_images": [_b64(source)],
            "mask": _b64(mask),
            "mask_blur_x": 6,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 48,
            "inpainting_fill": 2,
        },
    )

    assert response.status_code == 200
    request, init_images, mask_images = generation.submitted[0]
    assert request.mode == GenerationMode.INPAINT
    assert request.mask_blur == 6
    assert request.inpaint_only_masked is True
    assert request.inpaint_masked_padding == 48
    assert request.inpaint_mask_content == "latent noise"
    assert init_images and init_images[0].size == (16, 16)
    assert mask_images and mask_images[0].getbbox() is not None


def test_sdapi_progress_includes_current_image():
    generation = FakeGeneration()
    job = JobRecord(request=GenerationRequest(prompt="cat"), state=JobState.RUNNING)
    job.progress = JobProgress(
        job_id=job.id,
        state=JobState.RUNNING,
        step=5,
        total_steps=10,
        message="half",
        current_image=Image.new("RGB", (8, 8), "green"),
    )
    generation.progress_job = job
    client, _ = make_client(generation)

    response = client.get("/sdapi/v1/progress")

    assert response.status_code == 200
    data = response.json()
    assert data["progress"] == 0.5
    assert data["current_image"]
    assert data["state"]["message"] == "half"


def test_native_controlnet_and_plugin_endpoints_are_available():
    client, _ = make_client()

    assert client.get("/api/v1/controlnet/modules").json() == ["none", "canny"]
    assert client.get("/sdapi/v1/controlnet/model_list").json() == {"model_list": ["control-a"]}
    assert client.get("/api/v1/plugins").json()[0]["name"] == "hello"
    assert client.get("/sdapi/v1/extensions").json()[0]["enabled"] is True


def test_native_image_maturity_endpoint_lists_core_routes():
    client, _ = make_client()

    response = client.get("/api/v1/image/maturity")

    assert response.status_code == 200
    data = response.json()
    routes = {route["route"]: route for route in data["routes"]}
    assert data["target_score"] == 8.0
    assert routes["txt2img"]["benchmark_kind"] == "txt2img"
    assert routes["controlnet"]["benchmark_kind"] == "controlnet"
    assert routes["flux-txt2img"]["status"] == "maturing"


def test_native_xyz_plot_endpoint_returns_grid():
    client, _ = make_client()

    response = client.post(
        "/api/v1/xyz-plot",
        json={"base": {"prompt": "cat"}, "axes": [{"field": "seed", "values": [1]}]},
    )

    assert response.status_code == 200
    assert response.json()["labels"] == ["seed=1"]
    assert response.json()["grid"]


def test_native_optimization_status_endpoint_fallback():
    client, _ = make_client()

    response = client.get("/api/v1/optimization/status")

    assert response.status_code == 200
    data = response.json()
    assert data["profile_id"] == "balanced_sdpa_fp16"
    assert data["promotion_gates"]["status"] == "unavailable"


def test_native_optimization_status_endpoint_with_service(tmp_path):
    diagnostics = OptimizationDiagnosticsService(
        flags=RuntimeFlags(data_dir=tmp_path),
        settings=UserSettings(),
        detector=CapabilityDetector(core_packages=(), optional_packages={}),
        planner=OptimizationPlanner(),
        output_dir=tmp_path,
    )
    client, _ = make_client(optimization_diagnostics=diagnostics)

    response = client.get("/api/v1/optimization/status")

    assert response.status_code == 200
    data = response.json()
    assert data["profile_id"] == "balanced_sdpa_fp16"
    assert "capability_report" in data
    assert "known_failures" in data
