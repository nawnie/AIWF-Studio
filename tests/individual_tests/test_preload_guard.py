"""
Regression tests for the startup checkpoint preload guard.

SCP-3: _add_cached_single_file_config must add both config= and
local_files_only=True when a local HF cache entry is found, and neither
when the cache is absent.  Without local_files_only=True, Diffusers calls
HF hub download routines that crash via tqdm.contrib.concurrent.ensure_lock
(AttributeError: _lock) in background threads.

sys.modules stubs are injected at module level so the real torch/diffusers
packages need not be installed.
"""

from __future__ import annotations

import sys
import types
import importlib
from unittest.mock import patch

import pytest

_MISSING = object()
_ORIGINAL_MODULES: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Lightweight stubs -- injected before backend is imported.
# Only LEAF modules are stubbed; parent packages are loaded from disk so
# that sub-module discovery still works (aiwf.infrastructure.diffusers.__path__
# must point to the real directory so Python can find backend.py).
# ---------------------------------------------------------------------------

def _make_mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _set_stub(name: str, module: types.ModuleType) -> None:
    if name not in _ORIGINAL_MODULES:
        _ORIGINAL_MODULES[name] = sys.modules.get(name, _MISSING)
    sys.modules[name] = module


def _inject_stubs():
    # torch
    _set_stub(
        "torch",
        _make_mod("torch", float16="float16", bfloat16="bfloat16", float32="float32"),
    )

    # diffusers -- stub as a plain module (not a package) so Python uses OUR
    # version instead of looking for the real package on disk.
    _cls_names = [
        "StableDiffusionPipeline", "StableDiffusionXLPipeline",
        "StableDiffusionInpaintPipeline", "StableDiffusionXLInpaintPipeline",
        "StableDiffusionImg2ImgPipeline", "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline",
        "StableDiffusion3InpaintPipeline",
        "AutoencoderKL", "FluxInpaintPipeline", "FluxKontextPipeline",
        "FluxPipeline", "FluxTransformer2DModel",
        "GGUFQuantizationConfig",
        "DDIMScheduler", "DEISMultistepScheduler",
        "DPMSolverMultistepScheduler", "DPMSolverSDEScheduler",
        "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
        "FlowMatchEulerDiscreteScheduler",
        "HeunDiscreteScheduler", "KDPM2AncestralDiscreteScheduler",
        "KDPM2DiscreteScheduler", "LCMScheduler", "LMSDiscreteScheduler",
        "SASolverScheduler", "TCDScheduler", "UniPCMultistepScheduler",
    ]
    d = _make_mod("diffusers")
    for n in _cls_names:
        setattr(d, n, type(n, (), {}))
    utils = _make_mod("diffusers.utils")
    utils_log = _make_mod("diffusers.utils.logging",
                           disable_progress_bar=lambda: None)
    d.utils = utils
    utils.logging = utils_log
    _set_stub("diffusers", d)
    _set_stub("diffusers.utils", utils)
    _set_stub("diffusers.utils.logging", utils_log)

    _set_stub("PIL", _make_mod("PIL"))
    _set_stub("PIL.Image", _make_mod("PIL.Image", Image=object))
    _set_stub("safetensors", _make_mod("safetensors"))
    _set_stub("safetensors.torch", _make_mod("safetensors.torch", load_file=lambda *a, **k: {}))

    # aiwf leaf modules -- only the ones backend.py imports from.
    # Parent packages (aiwf, aiwf.infrastructure, aiwf.infrastructure.diffusers)
    # are NOT stubbed so Python loads the real __init__.py files from disk.
    _leaf = {
        "aiwf.core.config.settings": dict(
            RuntimeFlags=type("RuntimeFlags", (), {})),
        "aiwf.core.domain.errors": dict(
            GenerationCancelledError=Exception,
            ModelNotFoundError=Exception),
        "aiwf.core.domain.extra_networks": dict(
            parse_extra_networks=lambda *a, **k: []),
        "aiwf.core.domain.controlnet": dict(
            ControlNetUnit=type("ControlNetUnit", (), {})),
        "aiwf.core.domain.generation": dict(
            GenerationMode=type("GenerationMode", (), {}),
            GenerationRequest=type("GenerationRequest", (), {}),
            GenerationResult=type("GenerationResult", (), {}),
        ),
        "aiwf.core.domain.models": dict(
            LoraInfo=type("LoraInfo", (), {}), SAMPLERS=[],
            Checkpoint=type("Checkpoint", (), {}),
            SamplerInfo=type("SamplerInfo", (), {}),
            VaeInfo=type("VaeInfo", (), {}),
        ),
        "aiwf.core.infotext": dict(format_infotext=lambda *a, **k: ""),
        "aiwf.infrastructure.diffusers.checkpoints": dict(
            scan_from_flags=lambda *a: []),
        "aiwf.infrastructure.diffusers.embeddings": dict(
            find_referenced_embeddings=lambda *a: [],
            scan_embeddings=lambda *a: []),
        "aiwf.infrastructure.diffusers.extra_networks": dict(
            apply_loras=lambda *a, **k: None,
            clear_loras=lambda *a, **k: None),
        "aiwf.infrastructure.diffusers.loras": dict(
            scan_loras=lambda *a: []),
        "aiwf.infrastructure.diffusers.mask": dict(
            align_to_multiple_of_8=lambda x, *a: x,
            align_to_multiple_of_16=lambda x, *a: x,
            apply_masked_content=lambda *a, **k: None,
            blur_mask=lambda *a, **k: None,
            composite_inpaint_result=lambda *a, **k: None,
            crop_to_masked=lambda *a, **k: None,
            prepare_inpaint_mask=lambda *a, **k: None,
            resize_for_inpaint=lambda *a, **k: None,
        ),
        "aiwf.infrastructure.diffusers.controlnet_pipe": dict(
            ControlNetModelCache=type("ControlNetModelCache", (), {}),
            assert_controlnet_checkpoint_compatible=lambda *a, **k: None,
            build_controlnet_pipeline=lambda *a, **k: None,
        ),
        "aiwf.infrastructure.controlnet.images": dict(
            decode_control_image=lambda *a, **k: None),
        "aiwf.infrastructure.controlnet.catalog": dict(
            iter_controlnet_model_paths=lambda *a, **k: [],
            resolve_controlnet_roots=lambda *a, **k: [],
        ),
        "aiwf.infrastructure.controlnet.preprocess": dict(
            CV2_MODULES=set(),
            PREPROCESS_MODULES={},
            PreprocessParams=type("PreprocessParams", (), {}),
            preprocess_control_image=lambda *a, **k: None,
        ),
        "aiwf.infrastructure.diffusers.model_arch": dict(
            ARCH_FLUX="flux",
            ARCH_FLUX_KONTEXT="flux_kontext",
            ARCH_FLUX2_KLEIN="flux2_klein",
            ARCH_INPAINT="inpaint",
            ARCH_QWEN_IMAGE="qwen_image",
            ARCH_QWEN_IMAGE_NUNCHAKU="qwen_image_nunchaku",
            ARCH_SANA="sana",
            ARCH_SANA_VIDEO="sana_video",
            ARCH_SD15="sd15",
            ARCH_SDXL="sdxl", ARCH_SDXL_INPAINT="sdxl_inpaint",
            ARCH_SD35="sd35",
            ARCH_Z_IMAGE="z_image",
            UNET_INPUT_KEY="model.diffusion_model.input_blocks.0.0.weight",
            detect_checkpoint_architecture=lambda *a, **k: "sd15",
            infer_architecture_from_shapes=lambda *a, **k: "sd15",
            is_flux2_klein_architecture=lambda *a: False,
            is_flux_architecture=lambda *a: False,
            is_flux_fill_architecture=lambda *a: False,
            is_flux_kontext_architecture=lambda *a: False,
            is_qwen_image_architecture=lambda *a: False,
            is_qwen_nunchaku_architecture=lambda *a: False,
            is_sana_architecture=lambda *a: False,
            is_sdxl_architecture=lambda *a: False,
            is_sd3_architecture=lambda *a: False,
            is_inpaint_architecture=lambda *a: False,
            is_transformer_image_architecture=lambda *a: False,
            is_z_image_architecture=lambda *a: False,
            looks_like_lora_weights=lambda *a, **k: False,
            _safetensors_tensor_shapes=lambda *a, **k: {},
        ),
        "aiwf.infrastructure.diffusers.prompt_encode": dict(
            build_prompt_kwargs=lambda *a, **k: {}),
        "aiwf.infrastructure.diffusers.flux_bnb_loader": dict(
            load_flux_original_bnb_transformer=lambda *a, **k: None),
        "aiwf.infrastructure.quant.bnb_nf4_format": dict(
            build_bnb_4bit_quantization_config=lambda *a, **k: None,
            inspect_bnb_4bit_safetensors=lambda *a, **k: None,
            normalize_bnb_4bit_compute_dtype=lambda *a, **k: None,
            resolve_transformer_load_format=lambda *a, **k: "standard",
        ),
        "aiwf.infrastructure.diffusers.vae": dict(
            resolve_vae=lambda *a, **k: None,
            scan_vaes=lambda *a: []),
        "aiwf.services.qwen_nunchaku": dict(
            QwenNunchakuService=type("QwenNunchakuService", (), {}),
            QwenNunchakuUnavailable=Exception,
        ),
        "aiwf.infrastructure.torch.attention": dict(
            apply_attention_optimizations=lambda *a, **k: None,
            apply_image_pipeline_optimizations=lambda *a, **k: None),
        "aiwf.infrastructure.torch.devices": dict(
            DeviceManager=type("DeviceManager", (), {})),
    }
    for mod_name, attrs in _leaf.items():
        _set_stub(mod_name, _make_mod(mod_name, **attrs))


def _restore_modules() -> None:
    for name, original in reversed(_ORIGINAL_MODULES.items()):
        if original is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original
    _ORIGINAL_MODULES.clear()


@pytest.fixture()
def preload_backend():
    _ORIGINAL_MODULES.clear()
    _ORIGINAL_MODULES["aiwf.infrastructure.diffusers.backend"] = sys.modules.get(
        "aiwf.infrastructure.diffusers.backend",
        _MISSING,
    )
    sys.modules.pop("aiwf.infrastructure.diffusers.backend", None)
    _inject_stubs()
    backend = importlib.import_module("aiwf.infrastructure.diffusers.backend")
    sd15_cls = sys.modules["diffusers"].StableDiffusionPipeline
    sdxl_cls = sys.modules["diffusers"].StableDiffusionXLPipeline
    try:
        yield backend, sd15_cls, sdxl_cls
    finally:
        _restore_modules()


# ---------------------------------------------------------------------------
# _cached_single_file_config_dir
# ---------------------------------------------------------------------------

class TestCachedSingleFileConfigDir:
    def test_returns_none_when_pipeline_not_in_registry(self, preload_backend):
        backend, _, _ = preload_backend
        assert backend._cached_single_file_config_dir(type("Unknown", (), {})) is None

    def test_returns_none_when_try_to_load_returns_none(self, preload_backend):
        backend, sd15_cls, _ = preload_backend
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value=None):
            assert backend._cached_single_file_config_dir(sd15_cls) is None

    def test_returns_none_when_try_to_load_returns_non_string(self, preload_backend):
        backend, sd15_cls, _ = preload_backend
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value=object()):
            assert backend._cached_single_file_config_dir(sd15_cls) is None

    def test_returns_parent_dir_when_model_index_cached(self, tmp_path, preload_backend):
        backend, sd15_cls, _ = preload_backend
        f = tmp_path / "model_index.json"
        f.write_text("{}")
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value=str(f)):
            assert backend._cached_single_file_config_dir(sd15_cls) == str(tmp_path)

    def test_returns_none_when_cached_path_does_not_exist(self, preload_backend):
        backend, sd15_cls, _ = preload_backend
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value="/no/such/model_index.json"):
            assert backend._cached_single_file_config_dir(sd15_cls) is None

    def test_returns_none_when_try_to_load_raises(self, preload_backend):
        backend, sd15_cls, _ = preload_backend
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   side_effect=Exception("err")):
            assert backend._cached_single_file_config_dir(sd15_cls) is None

    def test_sdxl_uses_sdxl_repo(self, tmp_path, preload_backend):
        backend, _, sdxl_cls = preload_backend
        f = tmp_path / "model_index.json"
        f.write_text("{}")
        expected = backend._SINGLE_FILE_CONFIG_REPOS[sdxl_cls]
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value=str(f)) as m:
            backend._cached_single_file_config_dir(sdxl_cls)
            m.assert_called_once_with(expected, "model_index.json")

    def test_all_registered_repos_are_valid(self, preload_backend):
        backend, _, _ = preload_backend
        for cls, repo in backend._SINGLE_FILE_CONFIG_REPOS.items():
            assert isinstance(repo, str) and "/" in repo


# ---------------------------------------------------------------------------
# _add_cached_single_file_config
# ---------------------------------------------------------------------------

class TestAddCachedSingleFileConfig:
    def test_adds_config_and_local_files_only_when_cached(self, tmp_path, preload_backend):
        backend, sd15_cls, _ = preload_backend
        # Core regression: a local config MUST set local_files_only=True.
        # Without it Diffusers calls tqdm.contrib.concurrent.ensure_lock
        # which crashes with AttributeError: _lock in background threads.
        f = tmp_path / "model_index.json"
        f.write_text("{}")
        kw = {}
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value=str(f)):
            backend._add_cached_single_file_config(kw, sd15_cls)
        assert kw.get("config") == str(tmp_path)
        assert kw.get("local_files_only") is True

    def test_adds_nothing_when_cache_absent(self, preload_backend):
        backend, sd15_cls, _ = preload_backend
        kw = {"torch_dtype": "float16"}
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value=None):
            backend._add_cached_single_file_config(kw, sd15_cls)
        assert "config" not in kw
        assert "local_files_only" not in kw

    def test_preserves_existing_kwargs(self, tmp_path, preload_backend):
        backend, sd15_cls, _ = preload_backend
        f = tmp_path / "model_index.json"
        f.write_text("{}")
        kw = {"torch_dtype": "bfloat16", "use_safetensors": True}
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value=str(f)):
            backend._add_cached_single_file_config(kw, sd15_cls)
        assert kw["torch_dtype"] == "bfloat16"
        assert kw["use_safetensors"] is True

    def test_sdxl_also_gets_local_files_only(self, tmp_path, preload_backend):
        backend, _, sdxl_cls = preload_backend
        f = tmp_path / "model_index.json"
        f.write_text("{}")
        kw = {}
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   return_value=str(f)):
            backend._add_cached_single_file_config(kw, sdxl_cls)
        assert kw.get("local_files_only") is True

    def test_no_local_files_only_when_cache_raises(self, preload_backend):
        backend, sd15_cls, _ = preload_backend
        kw = {}
        with patch("aiwf.infrastructure.diffusers.backend._try_to_load_from_cache",
                   side_effect=Exception("hub error")):
            backend._add_cached_single_file_config(kw, sd15_cls)
        assert "local_files_only" not in kw


def test_load_inpaint_checkpoint_uses_inpaint_cache(preload_backend, monkeypatch, tmp_path):
    backend, _, _ = preload_backend
    inpaint_cls = sys.modules["diffusers"].StableDiffusionInpaintPipeline
    pipe = types.SimpleNamespace()
    monkeypatch.setattr(
        inpaint_cls,
        "from_single_file",
        classmethod(lambda cls, path, **kwargs: pipe),
        raising=False,
    )

    service = object.__new__(backend.DiffusersBackend)
    service.flags = types.SimpleNamespace(no_half=False, lowvram=False, medvram=False)
    service.devices = types.SimpleNamespace(
        dtype=lambda no_half: "float16",
        device=lambda: types.SimpleNamespace(type="cpu"),
        empty_cache=lambda: None,
        total_vram_gb=lambda: 0.0,
    )
    service._txt2img = "existing txt2img"
    service._img2img = None
    service._inpaint = None
    service._active = None
    service._inpaint_active = None
    service._active_vae_id = None
    monkeypatch.setattr(service, "_remember_base_scheduler_config", lambda pipe: None)
    monkeypatch.setattr(service, "_apply_fp8_storage", lambda pipe: None)
    monkeypatch.setattr(service, "_place_pipeline", lambda pipe, **kwargs: pipe)
    monkeypatch.setattr(service, "_tune_vae_memory", lambda pipe, architecture: None)

    checkpoint = types.SimpleNamespace(
        title="Inpaint",
        architecture="inpaint",
        path=str(tmp_path / "inpaint.safetensors"),
    )

    loaded = service._load_inpaint_checkpoint(checkpoint)

    assert loaded is pipe
    assert service._inpaint is pipe
    assert service._txt2img == "existing txt2img"
    assert service._inpaint_active is checkpoint


def _make_component_dir(path):
    (path / "scheduler").mkdir(parents=True)
    (path / "text_encoder").mkdir()
    (path / "tokenizer").mkdir()
    (path / "vae").mkdir()
    (path / "model_index.json").write_text("{}", encoding="utf-8")
    (path / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "text_encoder" / "model.safetensors").write_bytes(b"x")
    (path / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")


@pytest.mark.parametrize(
    "architecture,filename,component_rel,kind",
    [
        (
            "flux2_klein",
            "fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf",
            ("flux2", "Components", "FLUX.2-klein-9B"),
            "flux2",
        ),
        (
            "z_image",
            "fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf",
            ("z-image", "Components", "Z-Image-Turbo"),
            "z-image",
        ),
    ],
)
def test_transformer_image_preload_requires_component_folder(
    preload_backend,
    monkeypatch,
    tmp_path,
    architecture,
    filename,
    component_rel,
    kind,
):
    backend, _, _ = preload_backend
    root = tmp_path / "models"
    model_path = root / kind / "GGUF" / filename
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"GGUF")

    monkeypatch.setattr(backend, "is_flux2_klein_architecture", lambda arch: arch == "flux2_klein")
    monkeypatch.setattr(backend, "is_z_image_architecture", lambda arch: arch == "z_image")

    service = object.__new__(backend.DiffusersBackend)
    service._flux_search_roots = lambda: [root]
    checkpoint = types.SimpleNamespace(
        id=model_path.stem,
        title=model_path.stem,
        filename=filename,
        path=str(model_path),
        architecture=architecture,
    )
    service._resolve_checkpoint = lambda checkpoint_id=None: checkpoint

    assert service.can_preload_checkpoint_locally() is False
    _make_component_dir(root.joinpath(*component_rel))
    assert service.can_preload_checkpoint_locally() is True


def test_flux2_klein_9b_preload_rejects_public_4b_components(
    preload_backend,
    monkeypatch,
    tmp_path,
):
    backend, _, _ = preload_backend
    root = tmp_path / "models"
    model_path = root / "flux2" / "GGUF" / "fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"GGUF")
    _make_component_dir(root / "flux2" / "Components" / "FLUX.2-klein-4B")

    monkeypatch.setattr(backend, "is_flux2_klein_architecture", lambda arch: arch == "flux2_klein")
    monkeypatch.setattr(backend, "is_z_image_architecture", lambda arch: arch == "z_image")

    service = object.__new__(backend.DiffusersBackend)
    service._flux_search_roots = lambda: [root]
    checkpoint = types.SimpleNamespace(
        id=model_path.stem,
        title=model_path.stem,
        filename=model_path.name,
        path=str(model_path),
        architecture="flux2_klein",
    )
    service._resolve_checkpoint = lambda checkpoint_id=None: checkpoint

    assert service._flux2_component_repo_name(checkpoint) == "FLUX.2-klein-9B"
    assert service.can_preload_checkpoint_locally() is False
