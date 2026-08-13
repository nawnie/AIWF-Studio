from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.errors import ModelNotFoundError
from aiwf.infrastructure.diffusers import backend as backend_module
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.infrastructure.diffusers.flux_prompt_conditioning import (
    FLUX_CONDITIONING_DISTILLT5_CONTROL,
    FLUX_CONDITIONING_TEACHER,
    FLUX_CONDITIONING_UNIVERSAL,
    FluxPromptConditioningConfig,
    FluxT5ProjectedEncoder,
    load_universal_conditioner,
    normalize_flux_conditioning_mode,
    validate_flux_conditioning,
)


def test_flux_conditioning_modes_are_explicit() -> None:
    assert normalize_flux_conditioning_mode(None) == FLUX_CONDITIONING_TEACHER
    assert (
        normalize_flux_conditioning_mode(" DistillT5-Control ")
        == FLUX_CONDITIONING_DISTILLT5_CONTROL
    )
    assert normalize_flux_conditioning_mode("universal") == FLUX_CONDITIONING_UNIVERSAL
    with pytest.raises(ValueError, match="Unsupported FLUX"):
        normalize_flux_conditioning_mode("automatic-magic")


def test_universal_cache_scope_tracks_adapter_revision(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter.safetensors"
    config_path.write_text("{}", encoding="utf-8")
    weights_path.write_bytes(b"first")
    config = FluxPromptConditioningConfig(
        mode=FLUX_CONDITIONING_UNIVERSAL,
        universal_adapter_path=adapter_dir,
        universal_core_model=str(tmp_path / "umt5-base"),
        universal_core_revision="local-test",
    )

    first_scope = config.cache_scope()
    weights_path.write_bytes(b"second-revision")
    second_scope = config.cache_scope()

    assert first_scope != second_scope


def test_universal_loader_rejects_remote_core_before_import(tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter.safetensors").write_bytes(b"placeholder")
    config = FluxPromptConditioningConfig(
        mode=FLUX_CONDITIONING_UNIVERSAL,
        universal_adapter_path=adapter_dir,
        universal_core_model="google/umt5-base",
    )

    with pytest.raises(ModelNotFoundError, match="local-only"):
        load_universal_conditioner(
            config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_universal_preflight_does_not_require_teacher_encoders(
    tmp_path,
    monkeypatch,
) -> None:
    models_dir = tmp_path / "models"
    vae_path = models_dir / "flux" / "VAE" / "ae.safetensors"
    adapter_dir = tmp_path / "adapter"
    core_dir = tmp_path / "umt5-base"
    vae_path.parent.mkdir(parents=True)
    adapter_dir.mkdir()
    core_dir.mkdir()
    vae_path.write_bytes(b"vae")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter.safetensors").write_bytes(b"adapter")
    backend = DiffusersBackend(
        RuntimeFlags(data_dir=tmp_path),
        SimpleNamespace(),
    )
    backend._flux_prompt_conditioning = FluxPromptConditioningConfig(
        mode=FLUX_CONDITIONING_UNIVERSAL,
        universal_adapter_path=adapter_dir,
        universal_core_model=str(core_dir),
    )
    monkeypatch.setattr(
        backend_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "ute" else None,
    )

    components = backend._resolve_flux_component_paths()

    assert components == {"vae": vae_path.resolve()}


def test_distillt5_control_discovers_local_control_assets(tmp_path) -> None:
    models_dir = tmp_path / "models"
    vae_path = models_dir / "flux" / "VAE" / "ae.safetensors"
    text_root = models_dir / "flux" / "Textencoder"
    clip_path = text_root / "clip_l.safetensors"
    distillt5_path = text_root / "DistillT5"
    tokenizer_path = text_root / "t5-v1_1-base"
    vae_path.parent.mkdir(parents=True)
    distillt5_path.mkdir(parents=True)
    tokenizer_path.mkdir()
    vae_path.write_bytes(b"vae")
    clip_path.write_bytes(b"clip")
    backend = DiffusersBackend(
        RuntimeFlags(data_dir=tmp_path),
        SimpleNamespace(),
    )
    backend._flux_prompt_conditioning = FluxPromptConditioningConfig(
        mode=FLUX_CONDITIONING_DISTILLT5_CONTROL
    )

    components = backend._resolve_flux_component_paths()
    resolved = backend._resolved_flux_prompt_conditioning()

    assert components == {
        "clip_l": clip_path.resolve(),
        "vae": vae_path.resolve(),
    }
    assert resolved.distillt5_path == distillt5_path.resolve()
    assert resolved.distillt5_tokenizer_path == tokenizer_path.resolve()


def test_validate_flux_conditioning_checks_shape_and_finiteness() -> None:
    valid = SimpleNamespace(
        sequence=torch.zeros(1, 256, 4096),
        pooled=torch.zeros(1, 768),
    )
    sequence, pooled = validate_flux_conditioning(
        valid,
        expected_batch=1,
        expected_sequence_length=256,
    )
    assert sequence.shape == (1, 256, 4096)
    assert pooled.shape == (1, 768)

    invalid = SimpleNamespace(
        sequence=torch.zeros(1, 255, 4096),
        pooled=torch.zeros(1, 768),
    )
    with pytest.raises(RuntimeError, match="wrong shape"):
        validate_flux_conditioning(
            invalid,
            expected_batch=1,
            expected_sequence_length=256,
        )

    nonfinite = SimpleNamespace(
        sequence=torch.full((1, 256, 4096), float("nan")),
        pooled=torch.zeros(1, 768),
    )
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        validate_flux_conditioning(
            nonfinite,
            expected_batch=1,
            expected_sequence_length=256,
        )


def test_distillt5_projection_emits_flux_width() -> None:
    from transformers import T5Config

    config = T5Config(
        vocab_size=32,
        d_model=8,
        d_ff=16,
        d_kv=4,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        is_encoder_decoder=True,
        use_cache=False,
        project_in_dim=8,
        project_out_dim=32,
    )
    model = FluxT5ProjectedEncoder(config).eval()

    output = model(
        input_ids=torch.ones(1, 4, dtype=torch.long),
        return_dict=False,
    )

    assert output[0].shape == (1, 4, 32)


def test_universal_backend_cache_repeats_batches_consistently() -> None:
    class FakeConditioner:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.calls = 0

        def encode(self, prompt: str):
            self.calls += 1
            assert prompt == "a red robot"
            return SimpleNamespace(
                sequence=torch.ones(1, 256, 4096),
                pooled=torch.ones(1, 768),
            )

    conditioner = FakeConditioner()
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend._flux_prompt_conditioning = FluxPromptConditioningConfig(
        mode=FLUX_CONDITIONING_UNIVERSAL
    )
    backend._flux_conditioning_scope = "test-universal-scope"
    backend._flux_component_paths = {}
    backend._flux_prompt_cache = OrderedDict()
    backend._flux_universal_conditioner = conditioner

    first = backend._encode_flux_prompt(
        "a red robot",
        device=torch.device("cpu"),
        batch_size=2,
    )
    second = backend._encode_flux_prompt(
        "a red robot",
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert first[0].shape == (2, 256, 4096)
    assert first[1].shape == (2, 768)
    assert second[0].shape == first[0].shape
    assert second[1].shape == first[1].shape
    assert conditioner.calls == 1
