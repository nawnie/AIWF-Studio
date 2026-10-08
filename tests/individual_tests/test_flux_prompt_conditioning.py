from __future__ import annotations

import builtins
from collections import OrderedDict
from pathlib import Path
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


def test_missing_flux_teacher_assets_show_auto_sort_compatible_destinations(tmp_path):
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._flux_prompt_conditioning = FluxPromptConditioningConfig(mode=FLUX_CONDITIONING_TEACHER)

    with pytest.raises(ModelNotFoundError) as error:
        backend._resolve_flux_component_paths()

    detail = str(error.value)
    assert "CLIP-L under Textencoder" in detail
    assert "T5-XXL under flux/Textencoder" in detail
    assert "ae.safetensors under flux/VAE" in detail
    assert "Auto-sort can place these files" in detail


def test_flux_prompt_asset_validation_rejects_missing_teacher_tokenizers(monkeypatch, tmp_path):
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._flux_prompt_conditioning = FluxPromptConditioningConfig(mode=FLUX_CONDITIONING_TEACHER)
    monkeypatch.setattr(backend, "_resolve_flux_component_paths", lambda: {"vae": tmp_path / "vae"})
    monkeypatch.setattr(backend, "_resolve_flux_clip_tokenizer_path", lambda: (_ for _ in ()).throw(
        ModelNotFoundError("missing CLIP tokenizer")
    ))

    with pytest.raises(ModelNotFoundError, match="missing CLIP tokenizer"):
        backend._validate_flux_prompt_assets()


def test_fp8_flux_transformer_uses_compute_dtype_layerwise() -> None:
    class FakeTransformer:
        def __init__(self):
            self.weight = torch.nn.Parameter(
                torch.ones((1,), dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.casting = None
            self._skip_layerwise_casting_patterns = ["norm"]

        def parameters(self):
            return iter((self.weight,))

        def enable_layerwise_casting(self, **kwargs):
            self.casting = kwargs

    backend = object.__new__(DiffusersBackend)
    transformer = FakeTransformer()

    backend._prepare_flux_transformer_compute(transformer, torch.bfloat16)

    assert transformer.casting == {
        "storage_dtype": torch.float8_e4m3fn,
        "compute_dtype": torch.bfloat16,
        "skip_modules_pattern": (),
    }
    assert transformer._skip_layerwise_casting_patterns is None


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


def test_universal_loader_does_not_recommend_an_unverified_package_install(tmp_path, monkeypatch) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter.safetensors").write_bytes(b"fixture")
    core_dir = tmp_path / "umt5-base"
    core_dir.mkdir()
    config = FluxPromptConditioningConfig(
        mode=FLUX_CONDITIONING_UNIVERSAL,
        universal_adapter_path=adapter_dir,
        universal_core_model=str(core_dir),
    )
    original_import = builtins.__import__

    def fail_ute_import(name, *args, **kwargs):
        if name == "ute.integrations":
            raise ImportError("fixture: package unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_ute_import)

    with pytest.raises(ModelNotFoundError, match="does not currently provide a verified installation source"):
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


def test_flux_teacher_discovers_assets_in_adjacent_comfyui_library(tmp_path) -> None:
    models_dir = tmp_path / "AIWF"
    comfy_models = tmp_path / "ComfyUI Models"
    clip_path = comfy_models / "text_encoders" / "CLIP-L" / "123M fp16" / "clip_l.safetensors"
    t5_path = comfy_models / "text_encoders" / "T5" / "4.9B fp8" / "t5xxl_fp8_e4m3fn.safetensors"
    vae_path = comfy_models / "vae" / "Flux1" / "ae.safetensors"
    for path in (clip_path, t5_path, vae_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture asset")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models_dir)
    backend = DiffusersBackend(flags, SimpleNamespace())

    assert comfy_models.resolve() in flags.resolved_extra_model_dirs()
    components = backend._resolve_flux_component_paths()

    assert components == {
        "vae": vae_path.resolve(),
        "clip_l": clip_path.resolve(),
        "t5xxl": t5_path.resolve(),
    }


def test_flux_teacher_discovers_assets_in_configured_aiwf_family_folders(tmp_path) -> None:
    models_dir = tmp_path / "AIWF"
    clip_path = models_dir / "flux" / "Textencoder" / "clip_l.safetensors"
    t5_path = models_dir / "flux" / "Textencoder" / "t5xxl_fp8_e4m3fn.safetensors"
    vae_path = models_dir / "flux" / "VAE" / "ae.safetensors"
    for path in (clip_path, t5_path, vae_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture asset")
    backend = DiffusersBackend(
        RuntimeFlags(data_dir=tmp_path, models_dir=models_dir),
        SimpleNamespace(),
    )

    components = backend._resolve_flux_component_paths()

    assert components == {
        "vae": vae_path.resolve(),
        "clip_l": clip_path.resolve(),
        "t5xxl": t5_path.resolve(),
    }


def test_flux_teacher_tokenizers_resolve_from_configured_local_hub_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "AIWF"
    shared_models = tmp_path / "ComfyUI Models"
    clip_snapshot = (
        models_dir / "hub" / "models--openai--clip-vit-large-patch14" / "snapshots" / "clip-revision"
    )
    t5_snapshot = (
        shared_models
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--google--t5-v1_1-xxl"
        / "snapshots"
        / "t5-revision"
    )
    for snapshot, filenames in (
        (clip_snapshot, ("vocab.json", "merges.txt", "tokenizer_config.json")),
        (t5_snapshot, ("spiece.model", "tokenizer_config.json")),
    ):
        snapshot.mkdir(parents=True)
        for filename in filenames:
            (snapshot / filename).write_bytes(b"local tokenizer fixture")
    (clip_snapshot.parents[1] / "refs").mkdir()
    (clip_snapshot.parents[1] / "refs" / "main").write_text("clip-revision", encoding="utf-8")
    (t5_snapshot.parents[1] / "refs").mkdir()
    (t5_snapshot.parents[1] / "refs" / "main").write_text("t5-revision", encoding="utf-8")

    flags = RuntimeFlags(
        data_dir=tmp_path,
        models_dir=models_dir,
        extra_model_dirs=[shared_models],
    )
    backend = DiffusersBackend(flags, SimpleNamespace(device=lambda: torch.device("cpu")))
    loaded_tokenizers: dict[str, tuple[str, dict[str, object]]] = {}

    class FakeConfig:
        def __init__(self, **_kwargs) -> None:
            pass

    class FakeClipTextModel:
        def __init__(self, _config) -> None:
            pass

        def eval(self):
            return self

        def to(self, **_kwargs):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            loaded_tokenizers[cls.__name__] = (path, kwargs)
            return cls()

    class FakeClipTokenizer(FakeTokenizer):
        pass

    class FakeT5Tokenizer(FakeTokenizer):
        pass

    import transformers

    monkeypatch.setattr(transformers, "CLIPTextConfig", FakeConfig)
    monkeypatch.setattr(transformers, "CLIPTextModel", FakeClipTextModel)
    monkeypatch.setattr(transformers, "T5Config", FakeConfig)
    monkeypatch.setattr(transformers, "CLIPTokenizer", FakeClipTokenizer)
    monkeypatch.setattr(transformers, "T5TokenizerFast", FakeT5Tokenizer)
    monkeypatch.setattr(backend_module, "load_file", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(backend, "_load_encoder_state_dict", lambda *_args: None)
    monkeypatch.setattr(
        backend,
        "_load_flux_t5_encoder",
        lambda *_args: (object(), torch.device("cpu")),
    )

    backend._load_flux_prompt_models(
        {
            "clip_l": models_dir / "clip_l.safetensors",
            "t5xxl": models_dir / "t5xxl.safetensors",
        }
    )

    assert loaded_tokenizers == {
        "FakeClipTokenizer": (str(clip_snapshot), {"local_files_only": True}),
        "FakeT5Tokenizer": (
            str(t5_snapshot),
            {"legacy": True, "local_files_only": True},
        ),
    }


def test_flux_teacher_resolves_tokenizers_installed_by_aiwf_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models_dir = tmp_path / "AIWF"
    clip_snapshot = models_dir / "flux" / "tokenizer" / "clip-vit-large-patch14"
    t5_snapshot = models_dir / "flux" / "tokenizer" / "t5-v1_1-xxl"
    for snapshot, filenames in (
        (clip_snapshot, ("vocab.json", "merges.txt", "tokenizer_config.json")),
        (t5_snapshot, ("spiece.model", "tokenizer_config.json")),
    ):
        snapshot.mkdir(parents=True)
        for filename in filenames:
            (snapshot / filename).write_bytes(b"fixture tokenizer")
    backend = DiffusersBackend(
        RuntimeFlags(data_dir=tmp_path, models_dir=models_dir),
        SimpleNamespace(),
    )
    monkeypatch.setattr(backend, "_find_flux_tokenizer_snapshot", lambda *_args: None)

    assert backend._resolve_flux_clip_tokenizer_path() == clip_snapshot.resolve()
    assert backend._resolve_flux_t5_tokenizer_path() == t5_snapshot.resolve()


def test_flux_component_lookup_skips_empty_local_placeholders(tmp_path) -> None:
    models_dir = tmp_path / "AIWF"
    empty_local_vae = models_dir / "flux" / "VAE" / "ae.safetensors"
    shared_vae = tmp_path / "ComfyUI Models" / "vae" / "Flux1" / "ae.safetensors"
    empty_local_vae.parent.mkdir(parents=True)
    shared_vae.parent.mkdir(parents=True)
    empty_local_vae.touch()
    shared_vae.write_bytes(b"fixture asset")

    backend = DiffusersBackend(
        RuntimeFlags(data_dir=tmp_path, models_dir=models_dir),
        SimpleNamespace(),
    )

    assert backend._find_flux_component(
        ("ae.safetensors",),
        ("flux/VAE", "VAE", "vae"),
    ) == shared_vae.resolve()


def test_flux_component_lookup_rejects_assets_linked_outside_configured_root(tmp_path) -> None:
    models_dir = tmp_path / "AIWF"
    outside = tmp_path / "outside" / "ae.safetensors"
    linked_asset = models_dir / "flux" / "VAE" / "ae.safetensors"
    outside.parent.mkdir(parents=True)
    linked_asset.parent.mkdir(parents=True)
    outside.write_bytes(b"outside asset")
    try:
        linked_asset.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Creating a file symlink is not permitted in this environment: {exc}")

    backend = DiffusersBackend(
        RuntimeFlags(data_dir=tmp_path, models_dir=models_dir),
        SimpleNamespace(),
    )

    assert backend._find_flux_component(
        ("ae.safetensors",),
        ("flux/VAE", "VAE", "vae"),
    ) is None


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


def test_flux_fp8_preparation_upcasts_norm_parameters_only() -> None:
    class FakeFluxTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.norm_q = torch.nn.RMSNorm(8)
            self.attn.norm_q.weight = torch.nn.Parameter(
                torch.ones(8, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.proj = torch.nn.Linear(8, 8)
            self.proj.weight = torch.nn.Parameter(
                self.proj.weight.detach().to(torch.float8_e4m3fn), requires_grad=False
            )
            self.casting = None

        def enable_layerwise_casting(self, **kwargs) -> None:
            self.casting = kwargs

    transformer = FakeFluxTransformer()
    backend = DiffusersBackend.__new__(DiffusersBackend)

    backend._prepare_flux_transformer_compute(transformer, torch.bfloat16)

    assert transformer.attn.norm_q.weight.dtype == torch.bfloat16
    assert transformer.proj.weight.dtype == torch.float8_e4m3fn
    assert transformer.casting == {
        "storage_dtype": torch.float8_e4m3fn,
        "compute_dtype": torch.bfloat16,
        "skip_modules_pattern": (),
    }


def test_fp8_vae_is_cast_to_runtime_compute_dtype() -> None:
    vae = torch.nn.Module()
    vae.register_parameter(
        "weight",
        torch.nn.Parameter(torch.ones(4, dtype=torch.float8_e4m3fn), requires_grad=False),
    )

    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend._gguf_compute_dtype = lambda requested_dtype: torch.bfloat16

    prepared = backend._prepare_vae_compute(vae, torch.float8_e4m3fn)

    assert prepared is vae
    assert vae.weight.dtype == torch.bfloat16
