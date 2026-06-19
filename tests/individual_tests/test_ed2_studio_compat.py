from __future__ import annotations

from aiwf.services.training.ed2_studio_compat import (
    check_ed2_studio_compat,
    ed2_studio_compat_markdown,
)


def _complete_versions() -> dict[str, str]:
    return {
        "torch": "2.6.0+cu124",
        "torchvision": "0.21.0+cu124",
        "diffusers": "0.38.0",
        "transformers": "4.57.6",
        "accelerate": "1.14.0",
        "peft": "0.19.1",
        "safetensors": "0.8.0",
        "numpy": "2.2.6",
        "compel": "2.3.1",
        "ftfy": "6.3.1",
        "torchsde": "0.2.6",
        "colorama": "0.4.6",
        "tensorboard": "2.20.0",
        "omegaconf": "2.3.1",
        "pyre-extensions": "0.0.32",
        "lion-pytorch": "0.2.4",
        "tiktoken": "0.13.0",
        "aiohttp": "3.14.1",
        "wandb": "0.27.2",
        "pynvml": "11.4.1",
        "bitsandbytes": "0.49.2",
        "dowg": "0.3.1",
    }


def test_ed2_studio_compat_passes_with_modern_overlay_versions():
    result = check_ed2_studio_compat(_complete_versions())

    assert result.ok
    assert result.missing_required == ()
    assert any("Do not install EveryDream2trainer/requirements.txt" in warning for warning in result.warnings)


def test_ed2_studio_compat_reports_missing_required_overlay_packages():
    versions = _complete_versions()
    versions["tensorboard"] = None
    versions["bitsandbytes"] = None

    result = check_ed2_studio_compat(versions)

    assert not result.ok
    assert "tensorboard" in result.missing_required
    assert "bitsandbytes" in result.missing_required


def test_ed2_studio_compat_warns_on_transformers_five():
    versions = _complete_versions()
    versions["transformers"] = "5.0.0"

    result = check_ed2_studio_compat(versions)

    assert any("transformers 5.x" in warning for warning in result.warnings)


def test_ed2_studio_compat_rejects_pynvml_thirteen_package_shape():
    versions = _complete_versions()
    versions["pynvml"] = "13.0.1"

    result = check_ed2_studio_compat(versions)

    assert not result.ok
    assert "pynvml" in result.missing_required
    assert any("pynvml 13.x is incompatible" in warning for warning in result.warnings)


def test_ed2_studio_compat_markdown_is_human_readable():
    versions = _complete_versions()
    versions["dowg"] = None

    text = ed2_studio_compat_markdown(check_ed2_studio_compat(versions))

    assert "**ED2 on Studio runtime:** Missing" in text
    assert "`dowg`" in text
