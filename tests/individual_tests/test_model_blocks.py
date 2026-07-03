from pathlib import Path

from aiwf.infrastructure.diffusers.model_blocks import (
    is_non_selectable_image_asset_path,
    known_broken_selectable_image_asset,
)


def test_auxiliary_inpaint_asset_dir_is_not_selectable_checkpoint() -> None:
    path = Path("models") / "inpaint" / "big-lama.pt"

    assert is_non_selectable_image_asset_path(path)


def test_ultralytics_detector_asset_is_not_selectable_checkpoint() -> None:
    path = Path("models") / "ultralytics" / "bbox" / "Eyes.pt"

    assert is_non_selectable_image_asset_path(path)


def test_inpaint_named_checkpoint_in_stable_diffusion_dir_stays_selectable() -> None:
    path = Path("models") / "Stable-diffusion" / "realisticVisionV60-inpainting15.safetensors"

    assert not is_non_selectable_image_asset_path(path)


def test_known_flux_schema_mismatch_is_blocked() -> None:
    path = Path("models") / "flux" / "snofsSexNudesAndOtherFunStuff_distilledV12Fp8.safetensors"

    block = known_broken_selectable_image_asset(path)

    assert block is not None
    assert block.status == "broken-runtime"
    assert "loader-schema mismatch" in block.reason
