from __future__ import annotations

def test_resolve_vfi_root_uses_explicit_extra_root(tmp_path):
    from aiwf.infrastructure.rife.backend import resolve_vfi_root

    vfi = tmp_path / "comfyui-frame-interpolation"
    vfi.mkdir()
    (vfi / "vfi_utils.py").write_text("", encoding="utf-8")
    (vfi / "config.yaml").write_text("{}", encoding="utf-8")

    assert resolve_vfi_root(extra_roots=[vfi]) == vfi.resolve()


def test_list_rife_checkpoints_includes_rife47():
    from aiwf.infrastructure.rife.backend import list_rife_checkpoints

    names = list_rife_checkpoints()
    assert "rife47.pth" in names


def test_rife_service_detects_local_engine_folder(tmp_path):
    from aiwf.core.config.settings import RuntimeFlags, UserSettings
    from aiwf.services.rife import RifeService

    vfi = tmp_path / "engines" / "ComfyUI-Frame-Interpolation"
    vfi.mkdir(parents=True)
    (vfi / "vfi_utils.py").write_text("", encoding="utf-8")
    (vfi / "config.yaml").write_text("{}", encoding="utf-8")

    service = RifeService(RuntimeFlags(data_dir=tmp_path), UserSettings(), devices=None)

    assert service._vfi_root() == vfi.resolve()
    assert "Detected VFI pack" in service.folder_help()


def test_rife_options_defaults():
    from aiwf.core.domain.rife import RifeOptions

    opts = RifeOptions()
    assert opts.multiplier == 2
    assert opts.ckpt_name == "rife47.pth"
