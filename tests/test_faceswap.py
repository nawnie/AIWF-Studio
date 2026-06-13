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
    assert [d.key for d in svc.list_downloadable()] == ["inswapper_128_fp16", "inswapper_128"]
    item = DOWNLOADABLE_FACESWAP[0]
    assert item.key == "inswapper_128_fp16"
    assert not svc.is_installed(item)
    assert svc.find_downloadable("inswapper_128_fp16").url.startswith("https://huggingface.co/")
    assert svc.available() is False

    (svc.models_dir() / "inswapper_128_fp16.onnx").write_bytes(b"x")
    assert svc.is_installed(item)
    assert svc.available() is True
    assert [m.id for m in svc.list_models()] == ["inswapper_128_fp16"]


def test_swap_without_model_raises(tmp_path: Path):
    svc = _svc(tmp_path)
    from PIL import Image

    with pytest.raises(FaceSwapUnavailable):
        svc.swap(Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8)))


def test_resolve_model_path_falls_back_to_installed(tmp_path: Path):
    """Selecting 'inswapper_128' must still resolve when only the fp16 file
    is installed — the bug that made swaps silently fail."""
    svc = _svc(tmp_path)
    (svc.models_dir() / "inswapper_128_fp16.onnx").write_bytes(b"x")
    path = svc.resolve_model_path("inswapper_128")  # not the installed stem
    assert path is not None and path.name == "inswapper_128_fp16.onnx"


def test_resolve_model_path_none_when_empty(tmp_path: Path):
    svc = _svc(tmp_path)
    assert svc.resolve_model_path("inswapper_128") is None


def test_parse_face_indices():
    from aiwf.core.domain.faceswap import parse_face_indices

    assert parse_face_indices("0, 1, 2", [0]) == [0, 1, 2]
    assert parse_face_indices(3, [0]) == [3]
    assert parse_face_indices("", [0]) == [0]
    assert parse_face_indices(None, [9]) == [9]
    assert parse_face_indices("a,b", [0]) == [0]


def test_gender_filter_and_sex():
    from types import SimpleNamespace

    from aiwf.infrastructure.faceswap.swapper import _face_sex, _gender_filter

    f = SimpleNamespace(sex="F")
    m = SimpleNamespace(sex="M")
    g0 = SimpleNamespace(gender=0)  # female
    g1 = SimpleNamespace(gender=1)  # male
    assert _face_sex(f) == "F" and _face_sex(m) == "M"
    assert _face_sex(g0) == "F" and _face_sex(g1) == "M"
    faces = [f, m, g0, g1]
    assert _gender_filter(faces, 0) == faces           # no filter
    assert _gender_filter(faces, 1) == [f, g0]         # female only
    assert _gender_filter(faces, 2) == [m, g1]         # male only


def test_new_options_defaults():
    opts = FaceSwapOptions()
    assert opts.gender_source == 0 and opts.gender_target == 0
    assert opts.mask_face is False
    assert opts.restore_first is True
    assert opts.source_faces_index == [0]
    assert opts.target_faces_index == []
