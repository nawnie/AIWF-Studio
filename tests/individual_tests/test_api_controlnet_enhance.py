from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from aiwf.api.v1.routes import build_router
from aiwf.core.config.settings import RuntimeFlags
from aiwf.services.controlnet import ControlNetService


def _b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeEnhance:
    def list_upscalers(self):
        return []

    def list_restorers(self):
        return []

    def run_pipeline(self, image, upscale=None, restore=None):
        return image.resize((image.width * 2, image.height * 2)), "upscaled x2"


def _client(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    controlnet = ControlNetService(flags)
    controlnet.ensure_dir()
    (controlnet.models_dir() / "control_canny.safetensors").write_bytes(b"x")
    ctx = SimpleNamespace(controlnet=controlnet, enhance=FakeEnhance())
    app = FastAPI()
    app.include_router(build_router(ctx))
    return TestClient(app)


def test_controlnet_detect_returns_control_map(tmp_path: Path):
    client = _client(tmp_path)
    rng = np.random.default_rng(0)
    src = Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8"))
    resp = client.post(
        "/api/v1/controlnet/detect",
        json={"image": _b64(src), "module": "canny", "processor_res": 64},
    )
    assert resp.status_code == 200
    images = resp.json()["images"]
    assert images and isinstance(images[0], str)


def test_controlnet_downloadable_lists_catalog(tmp_path: Path):
    client = _client(tmp_path)
    resp = client.get("/api/v1/controlnet/downloadable")
    assert resp.status_code == 200
    payload = resp.json()
    keys = {item["key"] for item in payload}
    assert {"cn15-canny", "cn15-depth", "cn15-openpose"} <= keys
    assert {"cnxl-canny", "cnxl-depth", "cnxl-openpose"} <= keys
    sd15_keys = {"cn15-canny", "cn15-depth", "cn15-openpose"}
    for item in [item for item in payload if item["key"] in sd15_keys]:
        assert item["size_mb"] <= 700
        assert item["filename"].endswith(".safetensors")


def test_sdapi_controlnet_detect_alias(tmp_path: Path):
    client = _client(tmp_path)
    src = Image.new("RGB", (24, 24), "white")
    resp = client.post(
        "/sdapi/v1/controlnet/detect",
        json={"controlnet_input_images": [_b64(src)], "controlnet_module": "invert"},
    )
    assert resp.status_code == 200
    assert resp.json()["images"]


def test_enhance_endpoint(tmp_path: Path):
    client = _client(tmp_path)
    src = Image.new("RGB", (16, 16), "blue")
    resp = client.post(
        "/api/v1/enhance",
        json={"image": _b64(src), "upscaler_id": "realesrgan-x2"},
    )
    assert resp.status_code == 200
    assert resp.json()["image"]
    assert "x2" in resp.json()["infotext"]


def test_enhance_requires_a_model(tmp_path: Path):
    client = _client(tmp_path)
    src = Image.new("RGB", (16, 16), "blue")
    resp = client.post("/api/v1/enhance", json={"image": _b64(src)})
    assert resp.status_code == 422
