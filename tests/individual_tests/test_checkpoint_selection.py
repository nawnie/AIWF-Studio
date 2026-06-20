import json
from pathlib import Path
from unittest.mock import MagicMock

from aiwf.core.config.settings import UserSettings
from aiwf.core.domain.models import Checkpoint
from aiwf.infrastructure.diffusers import backend as diffusers_backend
from aiwf.services.generation import GenerationService
from aiwf.web.components.checkpoints import (
    default_checkpoint_title,
    resolve_default_checkpoint,
)


def _checkpoint(
    checkpoint_id: str,
    title: str | None = None,
    *,
    kind: str = "checkpoint",
    architecture: str = "sd15",
) -> Checkpoint:
    return Checkpoint(
        id=checkpoint_id,
        title=title or checkpoint_id,
        filename=f"{checkpoint_id}.safetensors",
        path=f"/models/{checkpoint_id}.safetensors",
        hash="abc123",
        kind=kind,
        architecture=architecture,
    )


def test_resolve_default_checkpoint_uses_last_saved_id():
    checkpoints = [_checkpoint("juggernaut"), _checkpoint("realistic_vision")]

    selected = resolve_default_checkpoint(checkpoints, "realistic_vision")

    assert selected is not None
    assert selected.id == "realistic_vision"


def test_resolve_default_checkpoint_falls_back_when_saved_missing():
    checkpoints = [_checkpoint("alpha"), _checkpoint("beta")]

    selected = resolve_default_checkpoint(checkpoints, "deleted_model")

    assert selected is not None
    assert selected.id == "alpha"


def test_default_checkpoint_title_returns_display_title():
    checkpoints = [
        _checkpoint("realistic_vision", "realisticVisionV60B1 [SD1.5] [f8b8450ebc]"),
        _checkpoint("juggernaut", "Juggernaut_X [SDXL] [0936a4dc33]"),
    ]

    title = default_checkpoint_title(checkpoints, "realistic_vision")

    assert title == "realisticVisionV60B1 [SD1.5] [f8b8450ebc]"


def test_resolve_default_checkpoint_skips_saved_inpaint_checkpoint():
    checkpoints = [
        _checkpoint("inpaint", "Juggernaut inpaint", kind="inpaint", architecture="sdxl_inpaint"),
        _checkpoint("base", "Juggernaut base", architecture="sdxl"),
    ]

    selected = resolve_default_checkpoint(checkpoints, "inpaint")

    assert selected is not None
    assert selected.id == "base"


def test_load_checkpoint_persists_last_model_to_config(tmp_path: Path):
    settings_path = tmp_path / "config.json"
    settings_path.write_text("{}", encoding="utf-8")
    settings = UserSettings()
    backend = MagicMock()
    backend.load_checkpoint.return_value = _checkpoint("realistic_vision")

    service = GenerationService(
        backend=backend,
        store=MagicMock(),
        metadata=MagicMock(),
        queue=MagicMock(),
        events=MagicMock(),
        settings=settings,
        settings_path=settings_path,
    )

    service.load_checkpoint("realistic_vision")

    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["last_checkpoint_id"] == "realistic_vision"


def test_remember_checkpoint_selection_persists_without_loading(tmp_path: Path):
    settings_path = tmp_path / "config.json"
    settings_path.write_text("{}", encoding="utf-8")
    settings = UserSettings()
    backend = MagicMock()
    backend.resolve_checkpoint.return_value = _checkpoint("realistic_vision")

    service = GenerationService(
        backend=backend,
        store=MagicMock(),
        metadata=MagicMock(),
        queue=MagicMock(),
        events=MagicMock(),
        settings=settings,
        settings_path=settings_path,
    )

    selected = service.remember_checkpoint_selection("realistic_vision")

    assert selected.id == "realistic_vision"
    backend.resolve_checkpoint.assert_called_once_with("realistic_vision")
    backend.load_checkpoint.assert_not_called()
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["last_checkpoint_id"] == "realistic_vision"


def test_cached_single_file_config_is_added_to_load_kwargs(tmp_path: Path, monkeypatch):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    model_index = snapshot / "model_index.json"
    model_index.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        diffusers_backend,
        "_try_to_load_from_cache",
        lambda repo_id, filename: str(model_index)
        if repo_id == "stabilityai/stable-diffusion-xl-base-1.0" and filename == "model_index.json"
        else None,
    )

    kwargs = {}
    diffusers_backend._add_cached_single_file_config(kwargs, diffusers_backend.StableDiffusionXLPipeline)

    assert kwargs["config"] == str(snapshot)
