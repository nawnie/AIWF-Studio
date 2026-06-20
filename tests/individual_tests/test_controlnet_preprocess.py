from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.controlnet import ControlNetUnit
from aiwf.infrastructure.controlnet.preprocess import (
    PREPROCESS_MODULES,
    PreprocessParams,
    preprocess_control_image,
)
from aiwf.services.controlnet import ControlNetService


def _sample_image(w=48, h=32) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8"))


def test_all_modules_return_rgb_at_processor_res():
    img = _sample_image()
    params = PreprocessParams(processor_res=64, threshold_a=80, threshold_b=160)
    for module in PREPROCESS_MODULES:
        out = preprocess_control_image(img, module, params)
        assert out.mode == "RGB"
        assert max(out.size) == 64


def test_canny_is_binary_edges():
    out = preprocess_control_image(_sample_image(), "canny", PreprocessParams(64, 80, 160))
    values = set(np.unique(np.asarray(out)))
    assert values.issubset({0, 255})


def test_unknown_module_falls_back_to_passthrough():
    img = _sample_image()
    out = preprocess_control_image(img, "does-not-exist", PreprocessParams(processor_res=64))
    assert out.mode == "RGB"
    assert max(out.size) == 64


def test_service_preprocess_and_modules(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    svc = ControlNetService(flags)
    assert svc.list_modules()[0] == "none"
    out = svc.preprocess(_sample_image(), "canny", processor_res=64, threshold_a=80, threshold_b=160)
    assert out.mode == "RGB"


def test_decode_control_image_base64_and_path(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    svc = ControlNetService(flags)

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode()
    assert svc.decode_control_image(raw_b64).size == (8, 8)
    assert svc.decode_control_image("data:image/png;base64," + raw_b64).size == (8, 8)

    path = tmp_path / "cn.png"
    Image.new("RGB", (5, 5), "blue").save(path)
    assert svc.decode_control_image(str(path)).size == (5, 5)
    assert svc.decode_control_image(None) is None
    assert svc.decode_control_image("not-base64-not-a-path") is None


def test_active_units_filters_disabled_and_unresolved(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    svc = ControlNetService(flags)
    svc.ensure_dir()
    (svc.models_dir() / "control_canny.safetensors").write_bytes(b"x")

    enabled = ControlNetUnit(enabled=True, model="control_canny")
    disabled = ControlNetUnit(enabled=False, model="control_canny")
    missing = ControlNetUnit(enabled=True, model="ghost")

    resolved = svc.active_units([enabled, disabled, missing])
    assert [u.model for u in resolved] == ["control_canny"]


def test_download_catalog_present_and_resolvable(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    svc = ControlNetService(flags)
    items = svc.list_downloadable()
    keys = {item.key for item in items}
    assert {"cn15-canny", "cn15-depth", "cn15-openpose"} <= keys
    assert {"cnxl-canny", "cnxl-depth", "cnxl-openpose"} <= keys
    canny = svc.find_downloadable("cn15-canny")
    assert canny is not None
    assert canny.url.startswith("https://huggingface.co/")
    assert not svc.is_installed(canny)
