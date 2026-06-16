from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_vfi_root_finds_documents_comfy_pack():
    from aiwf.infrastructure.rife.backend import resolve_vfi_root

    root = resolve_vfi_root()
    if Path(r"C:\Users\Shawn\Documents\ComfyUI\custom_nodes\comfyui-frame-interpolation").is_dir():
        assert root is not None
        assert (root / "vfi_utils.py").is_file()
    else:
        pytest.skip("ComfyUI-Frame-Interpolation not installed on this machine")


def test_list_rife_checkpoints_includes_rife47():
    from aiwf.infrastructure.rife.backend import list_rife_checkpoints

    names = list_rife_checkpoints()
    assert "rife47.pth" in names


def test_rife_options_defaults():
    from aiwf.core.domain.rife import RifeOptions

    opts = RifeOptions()
    assert opts.multiplier == 2
    assert opts.ckpt_name == "rife47.pth"