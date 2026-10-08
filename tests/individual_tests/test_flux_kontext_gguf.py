from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aiwf.core.domain.errors import ModelNotFoundError
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.models import Checkpoint
from aiwf.infrastructure.diffusers import backend as backend_module
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.infrastructure.diffusers.model_arch import ARCH_FLUX_KONTEXT


_KONTEXT_COMPONENTS = {
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "FluxTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def _write_complete_kontext_snapshot(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxKontextPipeline", **_KONTEXT_COMPONENTS}),
        encoding="utf-8",
    )
    for component, config_name in (
        ("scheduler", "scheduler_config.json"),
        ("text_encoder", "config.json"),
        ("text_encoder_2", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("tokenizer_2", "tokenizer_config.json"),
        ("transformer", "config.json"),
        ("vae", "config.json"),
    ):
        target = path / component / config_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    for component, weight_name in (
        ("text_encoder", "pytorch_model.bin"),
        ("text_encoder_2", "pytorch_model.bin"),
        ("transformer", "diffusion_pytorch_model.bin"),
        ("vae", "diffusion_pytorch_model.bin"),
    ):
        (path / component / weight_name).write_bytes(b"fixture")
    (path / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_2" / "spiece.model").write_bytes(b"fixture")
    return path


def _checkpoint(path: Path) -> Checkpoint:
    return Checkpoint(
        id=path.stem,
        title=path.name,
        filename=path.name,
        path=str(path),
        architecture=ARCH_FLUX_KONTEXT,
    )


def test_flux_kontext_component_resolver_accepts_components_without_transformer_weights(tmp_path: Path):
    from aiwf.infrastructure.diffusers.backend import DiffusersBackend

    models_root = tmp_path / "models"
    snapshot = _write_complete_kontext_snapshot(
        models_root / "flux" / "Components" / "flux-kontext-4bit-fp4"
    )
    backend = object.__new__(DiffusersBackend)
    backend._diffusers_component_search_roots = lambda: [models_root]
    selected = _checkpoint(tmp_path / "flux1-kontext-dev-Q5_K_M.gguf")

    assert backend._resolve_component_dir(ARCH_FLUX_KONTEXT, selected) == snapshot.resolve()
    assert DiffusersBackend._looks_like_diffusers_component_dir(
        snapshot, architecture=ARCH_FLUX_KONTEXT
    )

    # The selected GGUF supplies the transformer weights; its companion only
    # needs the transformer config for Diffusers pipeline construction.
    (snapshot / "transformer" / "diffusion_pytorch_model.bin").unlink()
    assert backend._resolve_component_dir(ARCH_FLUX_KONTEXT, selected) == snapshot.resolve()

    (snapshot / "text_encoder_2" / "pytorch_model.bin").unlink()
    with pytest.raises(ModelNotFoundError, match="complete local Kontext"):
        backend._resolve_component_dir(ARCH_FLUX_KONTEXT, selected)


def test_flux_kontext_component_resolver_rejects_complete_snapshot_linked_outside_configured_root(tmp_path: Path):
    models_root = tmp_path / "models"
    external_snapshot = _write_complete_kontext_snapshot(tmp_path / "outside" / "kontext-components")
    candidate = models_root / "flux" / "Components" / "flux-kontext-4bit-fp4"
    candidate.parent.mkdir(parents=True)
    try:
        candidate.symlink_to(external_snapshot, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable in this environment: {exc}")

    backend = object.__new__(DiffusersBackend)
    backend._diffusers_component_search_roots = lambda: [models_root]
    selected = _checkpoint(tmp_path / "flux1-kontext-dev-Q5_K_M.gguf")

    with pytest.raises(ModelNotFoundError, match="complete local Kontext"):
        backend._resolve_component_dir(ARCH_FLUX_KONTEXT, selected)


def test_flux_kontext_gguf_header_check_requires_nonempty_known_transformer(tmp_path: Path, monkeypatch):
    from aiwf.infrastructure import model_header

    selected = tmp_path / "selected.gguf"
    selected.write_bytes(b"header fixture")
    monkeypatch.setattr(
        model_header,
        "read_model_info",
        lambda _path: SimpleNamespace(arch=model_header.ARCH_FLUX_KONTEXT_TRANSFORMER, tensor_count=12),
    )
    assert DiffusersBackend._is_flux_kontext_gguf_transformer(selected)

    monkeypatch.setattr(
        model_header,
        "read_model_info",
        lambda _path: SimpleNamespace(arch="flux-transformer", tensor_count=12),
    )
    assert not DiffusersBackend._is_flux_kontext_gguf_transformer(selected)
    selected.write_bytes(b"")
    assert not DiffusersBackend._is_flux_kontext_gguf_transformer(selected)


def test_flux_kontext_preload_requires_selected_header_and_complete_companion(tmp_path: Path):
    selected = tmp_path / "selected.gguf"
    selected.write_bytes(b"header fixture")
    companion = _write_complete_kontext_snapshot(tmp_path / "companion")
    checkpoint = _checkpoint(selected)
    backend = object.__new__(DiffusersBackend)
    backend._resolve_checkpoint = lambda _model_id: checkpoint
    backend._is_flux_kontext_gguf_transformer = lambda path: path == selected
    calls = []
    backend._resolve_component_dir = lambda arch, item: calls.append((arch, item)) or companion

    assert backend.can_preload_checkpoint_locally(checkpoint.id)
    assert calls == [(ARCH_FLUX_KONTEXT, checkpoint)]

    backend._is_flux_kontext_gguf_transformer = lambda _path: False
    assert not backend.can_preload_checkpoint_locally(checkpoint.id)


def test_flux_kontext_loader_overrides_snapshot_transformer_with_selected_gguf(tmp_path: Path, monkeypatch):
    selected = tmp_path / "flux1-kontext-dev-Q5_K_M.gguf"
    selected.write_bytes(b"selected gguf fixture")
    companion = _write_complete_kontext_snapshot(
        tmp_path / "flux" / "Components" / "flux-kontext-4bit-fp4"
    )
    checkpoint = _checkpoint(selected)
    marker_transformer = object()
    calls = {}
    pipe = SimpleNamespace(
        set_progress_bar_config=lambda **kwargs: calls.setdefault("progress", kwargs),
        safety_checker=object(),
    )
    backend = object.__new__(DiffusersBackend)
    backend._txt2img = None
    backend._img2img = None
    backend._inpaint_active = None
    backend._active = None
    backend.devices = SimpleNamespace(empty_cache=lambda: None)
    backend.flags = SimpleNamespace()
    backend._is_flux_kontext_gguf_transformer = lambda path: path == selected
    backend._resolve_component_dir = lambda arch, item: companion
    backend._dtype_for_architecture = lambda _arch: torch.float16
    backend._load_dit_transformer_single_file = lambda model_cls, path, **kwargs: (
        calls.update(transformer_path=path, model_cls=model_cls, transformer_kwargs=kwargs)
        or marker_transformer
    )
    backend._remember_base_scheduler_config = lambda _pipe: None
    backend._compile_allowed_for_architecture = lambda _arch: False
    backend._place_pipeline = lambda candidate, **_kwargs: candidate
    backend._wants_offload = lambda _arch: False
    backend._tune_vae_memory = lambda *_args: None
    monkeypatch.setattr(backend_module, "apply_attention_optimizations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        backend_module.FluxKontextPipeline,
        "from_pretrained",
        classmethod(lambda _cls, path, **kwargs: calls.update(pipeline_path=path, pipeline_kwargs=kwargs) or pipe),
    )

    result = backend._load_flux_kontext_checkpoint(checkpoint)

    assert result.path == str(selected)
    assert backend._active.path == str(selected)
    assert calls["transformer_path"] == selected
    assert calls["transformer_kwargs"]["config_dir"] == str(companion / "transformer")
    assert calls["transformer_kwargs"]["family"] == "Flux Kontext"
    assert calls["pipeline_path"] == str(companion)
    assert calls["pipeline_kwargs"]["transformer"] is marker_transformer
    assert calls["pipeline_kwargs"]["local_files_only"] is True
    assert calls["pipeline_kwargs"]["torch_dtype"] == torch.float16
    assert pipe.safety_checker is None


@pytest.mark.parametrize("mode", [GenerationMode.IMG2IMG, GenerationMode.INPAINT])
def test_flux_kontext_generation_rejects_unwired_image_editing(tmp_path: Path, mode):
    checkpoint = _checkpoint(tmp_path / "selected.gguf")
    backend = object.__new__(DiffusersBackend)
    backend._resolve_checkpoint = lambda _model_id: checkpoint
    request = GenerationRequest(prompt="edit this", checkpoint_id=checkpoint.id, mode=mode)

    with pytest.raises(ValueError, match="image-conditioned editing and inpaint are not wired"):
        backend.generate(request)
