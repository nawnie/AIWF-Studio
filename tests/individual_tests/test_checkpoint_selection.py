import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.models import Checkpoint
from aiwf.infrastructure.diffusers import backend as diffusers_backend
from aiwf.infrastructure.model_inventory import ModelInventoryRecord
from aiwf.services.generation import GenerationService
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.web.components.checkpoints import (
    _filter_checkpoints,
    default_checkpoint_title,
    format_model_status,
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


def test_engine_filter_keeps_only_compatible_models():
    checkpoints = [
        _checkpoint("flux", architecture="flux"),
        _checkpoint("flux2", architecture="flux2_klein"),
        _checkpoint("sdxl", architecture="sdxl"),
        _checkpoint("z", architecture="z_image"),
    ]

    assert [c.id for c in _filter_checkpoints(checkpoints, "Flux")] == ["flux"]
    assert [c.id for c in _filter_checkpoints(checkpoints, "Flux 2")] == ["flux2"]
    assert [c.id for c in _filter_checkpoints(checkpoints, "Stable Diffusion XL")] == ["sdxl"]
    assert [c.id for c in _filter_checkpoints(checkpoints, "Z-Image")] == ["z"]


def test_engine_status_uses_model_language(tmp_path: Path):
    checkpoints = [
        _checkpoint("flux", architecture="flux"),
        _checkpoint("sdxl", architecture="sdxl"),
    ]
    ctx = SimpleNamespace(
        generation=SimpleNamespace(list_checkpoints=lambda: checkpoints),
        flags=SimpleNamespace(
            resolved_ckpt_dir=lambda: tmp_path / "base_models",
            resolved_models_dir=lambda: tmp_path / "models",
        ),
    )

    status = format_model_status(ctx, "Flux")

    assert "**1** models for Flux" in status
    assert "checkpoints" not in status
    assert "flux.safetensors" in status


def test_common_prompt_prewarm_populates_flux2_cache(monkeypatch):
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend._txt2img = object()
    backend._active = _checkpoint("flux2", architecture="flux2_klein")
    backend._common_prompt_cache_warmed_for = set()
    backend._flux2_prompt_cache = {}

    monkeypatch.setattr(backend, "_execution_device", lambda _pipe: "cuda")

    def fake_encode(_pipe, prompt, _device):
        backend._flux2_prompt_cache[prompt] = f"encoded:{prompt}"
        return backend._flux2_prompt_cache[prompt]

    monkeypatch.setattr(backend, "_encode_flux2_prompt", fake_encode)

    warmed = backend.prewarm_common_prompt_embeddings(limit=16, budget_seconds=999)

    assert warmed == 16
    assert {
        "woman",
        "portrait",
        "person",
        "body",
        "full body",
        "close up portrait",
        "portrait of a woman",
        "beautiful woman",
    } <= set(backend._flux2_prompt_cache)
    assert backend.prewarm_common_prompt_embeddings(limit=16, budget_seconds=999) == 0


def test_flux_text_encoder_choices_use_cached_inventory(tmp_path: Path, monkeypatch):
    t5 = tmp_path / "models" / "flux" / "Textencoder" / "t5xxl_fp16.safetensors"
    clip = tmp_path / "models" / "flux" / "Textencoder" / "clip_l.safetensors"
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"t5")
    clip.write_bytes(b"clip")
    records = [
        ModelInventoryRecord(
            path=str(t5),
            filename=t5.name,
            family="text_encoder",
            architecture="flux",
            current_subdir="flux/Textencoder",
            recommended_subdir="flux/Textencoder",
            should_move=False,
            header_identifiers={"header_arch": "t5xxl"},
        ),
        ModelInventoryRecord(
            path=str(clip),
            filename=clip.name,
            family="text_encoder",
            architecture="flux",
            current_subdir="flux/Textencoder",
            recommended_subdir="flux/Textencoder",
            should_move=False,
            header_identifiers={"header_arch": "clip"},
        ),
    ]
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    monkeypatch.setattr(diffusers_backend, "get_model_inventory", lambda _flags: records)
    monkeypatch.setattr(
        DiffusersBackend,
        "_scan_flux_text_encoders_from_dirs",
        lambda _self: (_ for _ in ()).throw(AssertionError("directory scan should not run")),
    )

    choices = backend.list_flux_text_encoders()

    assert choices == [(f"{t5.stem}  [2 bytes]", str(t5.resolve()))]


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
