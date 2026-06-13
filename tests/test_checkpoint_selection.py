import json
from pathlib import Path
from unittest.mock import MagicMock

from aiwf.core.config.settings import UserSettings
from aiwf.core.domain.models import Checkpoint
from aiwf.services.generation import GenerationService
from aiwf.web.components.checkpoints import (
    default_checkpoint_title,
    resolve_default_checkpoint,
)


def _checkpoint(checkpoint_id: str, title: str | None = None) -> Checkpoint:
    return Checkpoint(
        id=checkpoint_id,
        title=title or checkpoint_id,
        filename=f"{checkpoint_id}.safetensors",
        path=f"/models/{checkpoint_id}.safetensors",
        hash="abc123",
        kind="checkpoint",
        architecture="sd15",
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
        _checkpoint("realistic_vision", "realisticVisionV60B1 [inpaint] [f8b8450ebc]"),
        _checkpoint("juggernaut", "Juggernaut_X [SDXL] [0936a4dc33]"),
    ]

    title = default_checkpoint_title(checkpoints, "realistic_vision")

    assert title == "realisticVisionV60B1 [inpaint] [f8b8450ebc]"


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