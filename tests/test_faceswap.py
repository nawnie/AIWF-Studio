from __future__ import annotations

from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.faceswap import FaceSwapModelInfo, FaceSwapOptions
from aiwf.infrastructure.faceswap import FaceSwapUnavailable
from aiwf.services.faceswap import DOWNLOADABLE_FACESWAP, FaceSwapService


def _svc(tmp_path: Path) -> FaceSwapService:
    svc = FaceSwapService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    svc.ensure_dir()
    return svc


def test_options_defaults():
    opts = FaceSwapOptions()
    assert opts.source_face_index == 0
    assert opts.target_face_index == -1
    assert opts.model_id == "inswapper_128"


def test_model_info_from_path(tmp_path: Path):
    info = FaceSwapModelInfo.from_path(tmp_path / "inswapper_128.onnx")
    assert info.id == "inswapper_128"


def test_catalog_and_install_state(tmp_path: Path):
    svc = _svc(tmp_path)
    assert svc.models_dir().name == "insightface"
    assert [d.key for d in svc.list_downloadable()] == ["inswapper_128"]
    item = DOWNLOADABLE_FACESWAP[0]
    assert not svc.is_installed(item)
    assert svc.find_downloadable("inswapper_128").url.startswith("https://huggingface.co/")
    assert svc.available() is False

    (svc.models_dir() / "inswapper_128.onnx").write_bytes(b"x")
    assert svc.is_installed(item)
    assert svc.available() is True
    assert [m.id for m in svc.list_models()] == ["inswapper_128"]


def test_swap_without_model_raises(tmp_path: Path):
    svc = _svc(tmp_path)
    from PIL import Image

    with pytest.raises(FaceSwapUnavailable):
        svc.swap(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)))
