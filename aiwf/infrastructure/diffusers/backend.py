from __future__ import annotations

import copy
import gc
import json
import logging
import random
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable
from uuid import uuid4

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxInpaintPipeline,
    FluxKontextPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    SASolverScheduler,
    TCDScheduler,
    UniPCMultistepScheduler,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import logging as diffusers_logging
from PIL import Image
from safetensors.torch import load_file

# Patch diffusers conversion logic to support custom Flux FP8 weights that use .weight instead of .scale for query/key norms
try:
    import diffusers.loaders.single_file_utils
    _orig_convert_flux = diffusers.loaders.single_file_utils.convert_flux_transformer_checkpoint_to_diffusers

    def _patched_convert_flux(checkpoint, *args, **kwargs):
        for key in list(checkpoint.keys()):
            if "query_norm.weight" in key:
                scale_key = key.replace("query_norm.weight", "query_norm.scale")
                if scale_key not in checkpoint:
                    checkpoint[scale_key] = checkpoint[key]
            elif "key_norm.weight" in key:
                scale_key = key.replace("key_norm.weight", "key_norm.scale")
                if scale_key not in checkpoint:
                    checkpoint[scale_key] = checkpoint[key]
        return _orig_convert_flux(checkpoint, *args, **kwargs)

    diffusers.loaders.single_file_utils.convert_flux_transformer_checkpoint_to_diffusers = _patched_convert_flux
except Exception:
    pass

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.errors import GenerationCancelledError, ModelNotFoundError
from aiwf.core.domain.extra_networks import parse_extra_networks
from aiwf.core.domain.controlnet import ControlNetUnit
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult
from aiwf.core.domain.models import LoraInfo, SAMPLERS, Checkpoint, SamplerInfo, VaeInfo
from aiwf.core.infotext import format_infotext
from aiwf.infrastructure.diffusers.checkpoints import scan_from_flags
from aiwf.infrastructure.diffusers.embeddings import find_referenced_embeddings, scan_embeddings
from aiwf.infrastructure.diffusers.extra_networks import apply_loras, clear_loras
from aiwf.infrastructure.diffusers.loras import scan_loras
from aiwf.infrastructure.diffusers.mask import (
    align_to_multiple_of_8,
    align_to_multiple_of_16,
    apply_masked_content,
    blur_mask,
    composite_inpaint_result,
    crop_to_masked,
    prepare_inpaint_mask,
    resize_for_inpaint,
)
from aiwf.infrastructure.diffusers.controlnet_pipe import (
    ControlNetModelCache,
    assert_controlnet_checkpoint_compatible,
    build_controlnet_pipeline,
)
from aiwf.infrastructure.controlnet.catalog import iter_controlnet_model_paths, resolve_controlnet_roots
from aiwf.infrastructure.controlnet.images import decode_control_image
from aiwf.infrastructure.controlnet.preprocess import PreprocessParams, preprocess_control_image
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_FLUX,
    ARCH_FLUX_KONTEXT,
    ARCH_FLUX2_KLEIN,
    ARCH_QWEN_IMAGE,
    ARCH_QWEN_IMAGE_NUNCHAKU,
    ARCH_SANA,
    ARCH_SD35,
    ARCH_SDXL,
    ARCH_SDXL_INPAINT,
    ARCH_Z_IMAGE,
    is_inpaint_architecture,
    is_flux2_klein_architecture,
    is_flux_architecture,
    is_flux_kontext_architecture,
    is_qwen_image_architecture,
    is_qwen_nunchaku_architecture,
    is_sana_architecture,
    is_sd3_architecture,
    is_sdxl_architecture,
    is_transformer_image_architecture,
    is_z_image_architecture,
)
from aiwf.infrastructure.diffusers.prompt_encode import build_prompt_kwargs
from aiwf.infrastructure.diffusers.vae import resolve_vae, scan_vaes
try:
    from aiwf.infrastructure.torch.attention import (
        apply_attention_optimizations,
        apply_image_pipeline_optimizations,
        attention_call_context,
    )
except ImportError:
    from aiwf.infrastructure.torch.attention import apply_attention_optimizations, apply_image_pipeline_optimizations

    @contextmanager
    def attention_call_context(_flags):
        yield "none"
from aiwf.infrastructure.torch.devices import DeviceManager
from aiwf.infrastructure.quant.bnb_nf4_format import (
    build_bnb_4bit_quantization_config,
    inspect_bnb_4bit_safetensors,
    normalize_bnb_4bit_compute_dtype,
    resolve_transformer_load_format,
)
from aiwf.infrastructure.diffusers.flux_bnb_loader import load_flux_original_bnb_transformer
from aiwf.services.qwen_nunchaku import QwenNunchakuService, QwenNunchakuUnavailable

logger = logging.getLogger(__name__)


def _supports_runtime_lora_adapters(architecture: str | None) -> bool:
    """Return whether AIWF can apply prompt LoRA tags at generation time."""
    architecture = (architecture or "").lower()
    if architecture in {ARCH_FLUX2_KLEIN, ARCH_Z_IMAGE, ARCH_QWEN_IMAGE, ARCH_QWEN_IMAGE_NUNCHAKU, ARCH_SANA}:
        return False
    return True


class _DiffusersSafetyWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "You have disabled the safety checker" not in record.getMessage()


diffusers_logging.disable_progress_bar()
logging.getLogger("diffusers").addFilter(_DiffusersSafetyWarningFilter())
try:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:
    pass
try:
    from huggingface_hub import try_to_load_from_cache as _try_to_load_from_cache
except Exception:  # pragma: no cover - huggingface_hub is a runtime dependency
    _try_to_load_from_cache = None
warnings.filterwarnings(
    "ignore",
    message="You have disabled the safety checker.*",
    category=UserWarning,
    module="diffusers.*",
)
warnings.filterwarnings(
    "ignore",
    message="`upcast_vae` is deprecated.*",
    category=FutureWarning,
    module="diffusers.*",
)

SAMPLER_CLASSES = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "ddim": DDIMScheduler,
    "unipc": UniPCMultistepScheduler,
    "dpm2": KDPM2DiscreteScheduler,
    "dpm2_a": KDPM2AncestralDiscreteScheduler,
    "deis": DEISMultistepScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "dpmpp_2m_sde": DPMSolverMultistepScheduler,
    "dpmpp_3m_sde": DPMSolverMultistepScheduler,
    "dpmpp_sde": DPMSolverSDEScheduler,
    "dpmpp_2m_karras": DPMSolverMultistepScheduler,
    "sa_solver": SASolverScheduler,
    "lcm": LCMScheduler,
    "tcd": TCDScheduler,
}


_SINGLE_FILE_CONFIG_REPOS = {
    StableDiffusionPipeline: "stable-diffusion-v1-5/stable-diffusion-v1-5",
    StableDiffusionXLPipeline: "stabilityai/stable-diffusion-xl-base-1.0",
    StableDiffusion3Pipeline: "stabilityai/stable-diffusion-3.5-medium",
    StableDiffusionInpaintPipeline: "stable-diffusion-v1-5/stable-diffusion-inpainting",
    StableDiffusionXLInpaintPipeline: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    StableDiffusion3InpaintPipeline: "stabilityai/stable-diffusion-3.5-medium",
}

_FLUX_TRANSFORMER_CONFIG_BASE = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.38.0",
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": None,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "axes_dims_rope": [16, 56, 56],
}

_FLUX_VAE_CONFIG = {
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.38.0",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "force_upcast": True,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
}

_FLUX2_KLEIN_9B_TRANSFORMER_CONFIG = {
    "_class_name": "Flux2Transformer2DModel",
    "_diffusers_version": "0.38.0",
    "patch_size": 1,
    "in_channels": 128,
    "out_channels": None,
    "num_layers": 8,
    "num_single_layers": 48,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "joint_attention_dim": 15360,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": [32, 32, 32, 32],
    "rope_theta": 2000,
    "eps": 1e-6,
    # Distilled Klein variants ignore CFG in the official Diffusers pipeline.
    # Fluxtrait's model page still recommends CFG 1, which is safe here.
    "guidance_embeds": False,
}

_FLUX2_KLEIN_4B_TRANSFORMER_CONFIG = {
    **_FLUX2_KLEIN_9B_TRANSFORMER_CONFIG,
    "num_layers": 5,
    "num_single_layers": 20,
    "num_attention_heads": 24,
    "joint_attention_dim": 7680,
}

_Z_IMAGE_TRANSFORMER_CONFIG = {
    "_class_name": "ZImageTransformer2DModel",
    "_diffusers_version": "0.38.0",
    "all_patch_size": [2],
    "all_f_patch_size": [1],
    "in_channels": 16,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "n_heads": 30,
    "n_kv_heads": 30,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 2560,
    "siglip_feat_dim": None,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
}


def _cached_single_file_config_dir(pipeline_cls) -> str | None:
    """Return a locally cached Diffusers config directory for single-file loads.

    Diffusers downloads a default config repo when ``config`` is omitted from
    ``from_single_file``. Startup preload must stay local-only, so use the HF
    cache index directly and never call ``snapshot_download`` here.
    """
    repo_id = _SINGLE_FILE_CONFIG_REPOS.get(pipeline_cls)
    if not repo_id:
        return None
    if _try_to_load_from_cache is None:
        return None
    try:
        cached = _try_to_load_from_cache(repo_id, "model_index.json")
    except Exception:
        return None
    if not isinstance(cached, str):
        return None
    model_index = Path(cached)
    if not model_index.is_file():
        return None
    return str(model_index.parent)


def _add_cached_single_file_config(load_kwargs: dict, pipeline_cls) -> None:
    config_dir = _cached_single_file_config_dir(pipeline_cls)
    if config_dir:
        load_kwargs["config"] = config_dir
        # Prevent Diffusers from attempting HF hub downloads for sub-components
        # (tokenizer vocab, feature extractor, etc.) when a local config is already
        # supplied.  Those download calls internally use tqdm.contrib.concurrent
        # which crashes with AttributeError: _lock on some tqdm versions.
        # With local_files_only=True any missing remote asset raises a clean
        # EnvironmentError that the caller's except-block catches gracefully.
        load_kwargs["local_files_only"] = True


class DiffusersBackend:
    _QWEN_NUNCHAKU_SENTINEL = object()

    def __init__(self, flags: RuntimeFlags, devices: DeviceManager) -> None:
        self.flags = flags
        self.devices = devices
        self.ckpt_dir = flags.resolved_ckpt_dir()
        self._txt2img: StableDiffusionPipeline | None = None
        self._img2img: StableDiffusionImg2ImgPipeline | None = None
        self._inpaint: StableDiffusionInpaintPipeline | None = None
        self._refiner: StableDiffusionXLImg2ImgPipeline | None = None
        self._active: Checkpoint | None = None
        self._inpaint_active: Checkpoint | None = None
        self._refiner_active: Checkpoint | None = None
        self._active_vae_id: str | None = None
        self._lora_catalog: list[LoraInfo] | None = None
        self._vae_catalog: list[VaeInfo] | None = None
        self._checkpoint_catalog: list[Checkpoint] | None = None
        self._embedding_catalog: list | None = None
        self._controlnet_cache = ControlNetModelCache()
        self._offload_active = False
        self._flux_text_encoder = None
        self._flux_text_encoder_2 = None
        self._flux_tokenizer = None
        self._flux_tokenizer_2 = None
        self._flux_component_paths: dict[str, str] = {}
        # User-selected Flux T5 text encoder path (None = auto-pick best available).
        self._flux_text_encoder_override: str | None = None
        self._flux_clip_device: torch.device | None = None
        self._flux_t5_device: torch.device | None = None
        self._flux_prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._flux2_prompt_cache: dict[str, torch.Tensor] = {}
        # Rolling average encode duration per family, used to show a live ETA
        # while the prompt is being encoded (no per-step granularity exists
        # inside a single forward pass, so we estimate from past runs instead).
        self._encode_duration_estimates: dict[str, float] = {}
        self._z_image_prompt_cache: dict[tuple[str, str], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        self._common_prompt_cache_warmed_for: set[str] = set()
        self._qwen_nunchaku = QwenNunchakuService(flags)

    _COMMON_PROMPT_STARTERS: tuple[str, ...] = (
        "woman",
        "portrait",
        "person",
        "body",
        "man",
        "face",
        "eyes",
        "hair",
        "hands",
        "full body",
        "upper body",
        "close up portrait",
        "portrait of a woman",
        "full body portrait of a woman",
        "beautiful woman",
        "beautiful woman portrait",
        "portrait of a person",
        "full body person",
        "standing person",
        "sitting person",
        "female model",
        "male model",
        "young woman",
        "adult woman",
        "person standing",
        "person walking",
        "woman standing",
        "woman sitting",
        "woman looking at camera",
        "portrait photograph",
        "professional portrait",
        "cinematic portrait",
        "studio portrait",
        "fashion portrait",
        "headshot",
        "profile portrait",
        "natural skin texture",
        "detailed face",
        "detailed eyes",
        "realistic body",
        "realistic person",
        "human anatomy",
        "full body photograph",
        "waist up portrait",
        "soft natural lighting",
        "cinematic lighting",
        "studio lighting",
        "high quality photo",
    )

    def _flux_config_cache_root(self) -> Path:
        root = self.flags.data_dir / "cache" / "flux_configs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _cached_transformer_config_dir(self, family: str, checkpoint_path: Path, config: dict) -> str:
        """Persist derived GGUF transformer configs so checkpoint switches skip re-derivation."""
        cache_root = self._flux_config_cache_root() / family / checkpoint_path.stem
        return self._write_temp_config(cache_root, "transformer", config)

    def is_checkpoint_warm(self, checkpoint_id: str | None = None) -> bool:
        return self.is_checkpoint_loaded(checkpoint_id)

    def _flux_encoder_device(self, t5_path: Path | None = None) -> torch.device:
        if self.devices.device().type != "cuda":
            return torch.device("cpu")
        try:
            free_bytes, _total = torch.cuda.mem_get_info(self.devices.device())
            reserve_bytes = 1024**3
            if t5_path is not None:
                t5_name = t5_path.name.lower()
                t5_bytes = int(5.5 * 1024**3) if "fp8" in t5_name else int(10.5 * 1024**3)
                if free_bytes < t5_bytes + reserve_bytes:
                    return torch.device("cpu")
            elif free_bytes < 6 * 1024**3:
                return torch.device("cpu")
            return self.devices.device()
        except Exception:
            logger.debug("Could not query CUDA free memory for Flux encoders", exc_info=True)
        return torch.device("cpu")

    @staticmethod
    def _load_encoder_state_dict(model, state_dict, *, assign: bool = False) -> None:
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.is_meta:
            model.load_state_dict(state_dict, strict=True, assign=True)
            return
        model.load_state_dict(state_dict, strict=True, assign=assign)

    @staticmethod
    def _flux_t5_weights_are_fp8(state_dict) -> bool:
        for tensor in state_dict.values():
            if getattr(tensor, "dtype", None) == torch.float8_e4m3fn:
                return True
        return False

    def _load_flux_t5_encoder(self, t5_path: Path, t5_config) -> tuple["T5EncoderModel", torch.device]:
        from transformers import T5EncoderModel

        state = load_file(str(t5_path), device="cpu")
        is_fp8_source = "fp8" in t5_path.name.lower() or self._flux_t5_weights_are_fp8(state)
        if is_fp8_source:
            with torch.device("meta"):
                t5 = T5EncoderModel(t5_config)
            self._load_encoder_state_dict(t5, state)
            del state
            # Transformers T5 cannot execute float8 weights; keep a warm fp16 copy in RAM
            # so the GPU stays reserved for the Flux transformer.
            t5_device = torch.device("cpu")
            t5 = t5.to(dtype=torch.float16, device=t5_device)
            t5.eval()
            logger.info(
                "Flux T5-XXL: %s -> fp16 CPU resident (cached in RAM; GPU left for transformer)",
                t5_path.name,
            )
            return t5, t5_device

        t5 = T5EncoderModel(t5_config)
        self._load_encoder_state_dict(t5, state)
        del state
        t5_device = self._flux_encoder_device(t5_path)
        t5 = t5.to(dtype=torch.float16, device=t5_device)
        t5.eval()
        logger.info("Flux T5-XXL: %s on %s", t5_path.name, t5_device)
        return t5, t5_device

    def is_checkpoint_loaded(self, checkpoint_id: str | None = None) -> bool:
        try:
            checkpoint = self._resolve_checkpoint(checkpoint_id)
        except ModelNotFoundError:
            return False
        if self._txt2img is None or self._active is None:
            return False
        if self._active.path != checkpoint.path:
            return False
        if is_flux_architecture(checkpoint.architecture):
            return (
                self._flux_text_encoder is not None
                and self._flux_text_encoder_2 is not None
            )
        return True

    @staticmethod
    def _snapshot_scheduler_config(scheduler) -> dict:
        config = getattr(scheduler, "config", None)
        if config is None:
            return {}
        if hasattr(config, "to_dict"):
            return copy.deepcopy(config.to_dict())
        return copy.deepcopy(dict(config))

    def _remember_base_scheduler_config(self, pipe) -> None:
        scheduler = getattr(pipe, "scheduler", None)
        if scheduler is None:
            return
        pipe._aiwf_base_scheduler_config = self._snapshot_scheduler_config(scheduler)

    def _base_scheduler_config_for_pipe(self, pipe) -> dict:
        stored = getattr(pipe, "_aiwf_base_scheduler_config", None)
        if stored:
            return copy.deepcopy(stored)
        config = self._snapshot_scheduler_config(getattr(pipe, "scheduler", None))
        if config:
            pipe._aiwf_base_scheduler_config = copy.deepcopy(config)
        return config

    def invalidate_checkpoints(self) -> None:
        self._checkpoint_catalog = None

    def invalidate_embeddings(self) -> None:
        self._embedding_catalog = None

    def list_embeddings(self):
        if self._embedding_catalog is None:
            self._embedding_catalog = scan_embeddings(self.flags)
        return self._embedding_catalog

    def invalidate_vaes(self) -> None:
        self._vae_catalog = None

    def invalidate_loras(self) -> None:
        self._lora_catalog = None

    def list_checkpoints(self) -> list[Checkpoint]:
        if self._checkpoint_catalog is None:
            self._checkpoint_catalog = scan_from_flags(self.flags)
        return self._checkpoint_catalog

    def list_samplers(self) -> list[SamplerInfo]:
        return list(SAMPLERS)

    def list_loras(self) -> list[LoraInfo]:
        if self._lora_catalog is None:
            self._lora_catalog = scan_loras(self.flags)
        return self._lora_catalog

    def list_vaes(self) -> list[VaeInfo]:
        if self._vae_catalog is None:
            self._vae_catalog = scan_vaes(self.flags)
        return self._vae_catalog

    def _resolve_checkpoint(self, checkpoint_id: str | None) -> Checkpoint:
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise ModelNotFoundError(f"No checkpoints in {self.ckpt_dir}")

        if checkpoint_id:
            for checkpoint in checkpoints:
                if checkpoint.id == checkpoint_id or checkpoint.title == checkpoint_id:
                    return checkpoint

        if self.flags.default_checkpoint:
            for checkpoint in checkpoints:
                if Path(checkpoint.path) == self.flags.default_checkpoint.resolve():
                    return checkpoint

        return checkpoints[0]

    _AUTO_OFFLOAD_VRAM_GB = 10.0

    def _place_pipeline(self, pipe, *, prefer_offload: bool = False, architecture: str | None = None):
        if self.devices.device().type not in ("cuda",) and (self.flags.lowvram or self.flags.medvram or prefer_offload):
            logger.warning("CPU offload modes need a CUDA device; loading fully on %s.", self.devices.device())
            pipe = pipe.to(self.devices.device())
            self._offload_active = False
            return pipe
        if self.flags.lowvram:
            pipe.enable_sequential_cpu_offload()
            self._offload_active = True
        elif self.flags.medvram:
            pipe.enable_model_cpu_offload()
            self._offload_active = True
        elif prefer_offload:
            arch_label = {
                ARCH_FLUX2_KLEIN: "Flux.2 Klein",
                ARCH_Z_IMAGE: "Z-Image",
                ARCH_QWEN_IMAGE: "Qwen Image",
                ARCH_QWEN_IMAGE_NUNCHAKU: "Qwen Image Nunchaku",
                ARCH_SANA: "Sana",
                ARCH_SD35: "SD3.5",
                ARCH_SDXL: "SDXL",
                ARCH_SDXL_INPAINT: "SDXL",
            }.get(architecture or "", "Large model")
            # Use the *actual* threshold this architecture was evaluated against
            # (see _offload_threshold_gb) rather than the SDXL-only constant, so
            # this message can't claim "<10 GB" when the real cutoff was 20 GB.
            threshold = self._offload_threshold_gb(architecture or "")
            logger.info(
                "%s on a <%.0f GB GPU — enabling model CPU offload automatically "
                "(use Low VRAM mode in Settings if you still hit out-of-memory).",
                arch_label,
                threshold,
            )
            pipe.enable_model_cpu_offload()
            self._offload_active = True
        else:
            pipe = pipe.to(self.devices.device())
            self._offload_active = False
        return pipe

    def _place_transformer_pipeline_keep_text_cpu(self, pipe, *, architecture: str):
        """Place image denoising modules on the active device, but keep large
        text encoders in system RAM.

        Flux.2 Klein, Z-Image, Qwen Image, and Sana pipelines include full
        language-model text encoders. Moving those encoders with `pipe.to(cuda)`
        makes cold start feel like a second model load and can consume most of a 16 GB GPU.
        Prompt encoding can run from CPU once, then only the compact embeddings
        need to move to the denoise device.
        """
        device = self.devices.device()
        if is_qwen_image_architecture(architecture) or is_sana_architecture(architecture):
            return self._place_pipeline(
                pipe,
                prefer_offload=self._wants_offload(architecture),
                architecture=architecture,
            )
        if device.type != "cuda" or self.flags.lowvram or self.flags.medvram:
            return self._place_pipeline(
                pipe,
                prefer_offload=self._wants_offload(architecture),
                architecture=architecture,
            )

        for module_name in ("transformer", "vae"):
            module = getattr(pipe, module_name, None)
            if module is not None and hasattr(module, "to"):
                module.to(device)
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None and hasattr(text_encoder, "to"):
            text_encoder.to("cpu")
        try:
            pipe._execution_device = device
        except Exception:
            logger.debug("Could not set pipeline execution device", exc_info=True)
        self._offload_active = False
        logger.info("%s text encoder kept on CPU; denoiser/VAE loaded on %s.", architecture, device)
        return pipe

    def _ensure_embeddings_for_prompt(
        self, pipe, prompt: str | None, negative: str | None, architecture: str
    ) -> None:
        """On-demand load of only the textual-inversion embeddings referenced in the prompt.

        Embeddings are discovered from the embeddings/ folder but are *not* loaded
        into the text encoders at checkpoint load time. Instead we inspect the
        (already style/wildcard processed) prompt and negative for bare token names
        that match known embedding ids (e.g. "AS-Young"), and load just those.

        This means unused embedding files the user never put in a prompt are never
        "selected" or loaded — fixing the previous eager behavior.
        """
        if is_sd3_architecture(architecture):
            return
        items = self.list_embeddings()
        if not items or not hasattr(pipe, "load_textual_inversion"):
            return

        refs = find_referenced_embeddings(prompt, negative, items)
        if not refs:
            return

        # Track which embeddings have been loaded onto *this* pipe instance.
        loaded: set[str] = getattr(pipe, "_aiwf_ti_loaded", None) or set()
        pipe._aiwf_ti_loaded = loaded

        sdxl = is_sdxl_architecture(architecture)
        newly: list[str] = []
        for item in refs:
            if item.id in loaded:
                continue
            try:
                self._load_single_embedding(pipe, item, sdxl=sdxl)
                loaded.add(item.id)
                newly.append(item.id)
            except Exception:
                # Incompatible for this architecture (e.g. SD1.5 .pt on SDXL pipe).
                logger.info("Embedding %s not compatible with %s (skipped)", item.id, architecture)
                loaded.add(item.id)  # avoid retrying on every subsequent generate with same prompt
        if newly:
            logger.info("Loaded %d embedding(s) for prompt: %s", len(newly), ", ".join(newly))

    @staticmethod
    def _load_single_embedding(pipe, item, *, sdxl: bool) -> None:
        if sdxl:
            # SDXL embeddings carry separate vectors for both text encoders.
            from safetensors.torch import load_file

            if not item.path.endswith(".safetensors"):
                raise ValueError("SDXL needs dual-encoder safetensors embeddings")
            state = load_file(item.path)
            if "clip_l" not in state or "clip_g" not in state:
                raise ValueError("not an SDXL (clip_l/clip_g) embedding")
            pipe.load_textual_inversion(
                state["clip_l"], token=item.id, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer
            )
            pipe.load_textual_inversion(
                state["clip_g"], token=item.id, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2
            )
            return
        pipe.load_textual_inversion(item.path, token=item.id)

    def _dtype_for_architecture(self, architecture: str) -> torch.dtype:
        if self.flags.no_half:
            return torch.float32
        if (
            getattr(self.flags, "fluxfp8", False)
            and is_transformer_image_architecture(architecture)
            and not is_qwen_image_architecture(architecture)
            and not is_sana_architecture(architecture)
        ):
            return torch.float8_e4m3fn
        if (is_sd3_architecture(architecture) or is_transformer_image_architecture(architecture)) and self.devices.device().type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return self.devices.dtype(self.flags.no_half)

    def _gguf_compute_dtype(self, requested_dtype: torch.dtype) -> torch.dtype:
        if requested_dtype != torch.float8_e4m3fn:
            return requested_dtype
        if self.devices.device().type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16

    def _load_dit_transformer_single_file(
        self,
        model_cls,
        path: Path,
        *,
        config_dir: str,
        dtype: torch.dtype,
        family: str,
    ):
        load_kwargs = {
            "config": config_dir,
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        load_format = resolve_transformer_load_format(path)
        if path.suffix.lower() == ".gguf":
            dtype = self._gguf_compute_dtype(dtype)
            load_kwargs["torch_dtype"] = dtype
            load_kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dtype)
            self._patch_gguf_linear_input_dtype()
        elif path.suffix.lower() == ".safetensors":
            bnb_report = inspect_bnb_4bit_safetensors(path)
            if bnb_report.is_bnb_4bit:
                if self.devices.device().type != "cuda":
                    raise ModelNotFoundError(
                        f"{path.name} is a bitsandbytes {bnb_report.quant_type.upper()} checkpoint; "
                        "CUDA is required for NF4/FP4 inference."
                    )
                if bnb_report.needs_custom_flux_bnb_loader:
                    if family.lower() != "flux":
                        raise ModelNotFoundError(
                            f"{path.name} is a packed Flux bitsandbytes {bnb_report.quant_type.upper()} "
                            "single-file checkpoint, but this loader was called for a non-Flux family."
                        )
                    transformer = load_flux_original_bnb_transformer(
                        path,
                        config_dir=config_dir,
                        dtype=normalize_bnb_4bit_compute_dtype(dtype),
                        device=self.devices.device(),
                        quant_type=bnb_report.quant_type or "nf4",
                    )
                    load_format = f"{bnb_report.load_format_label}-flux-original"
                    logger.info(
                        "Loading %s transformer (%s, %d source quantized layers) from %s",
                        family,
                        load_format,
                        bnb_report.quantized_linear_layers,
                        path.name,
                    )
                    return transformer
                bnb_compute_dtype = normalize_bnb_4bit_compute_dtype(dtype)
                load_kwargs["quantization_config"] = build_bnb_4bit_quantization_config(
                    bnb_report,
                    compute_dtype=bnb_compute_dtype,
                )
                load_kwargs["torch_dtype"] = bnb_compute_dtype
                load_format = bnb_report.load_format_label
                logger.info(
                    "Loading %s transformer (%s, %d quantized layers) from %s",
                    family,
                    load_format,
                    bnb_report.quantized_linear_layers,
                    path.name,
                )

        transformer_t0 = time.perf_counter()
        try:
            transformer = model_cls.from_single_file(str(path), **load_kwargs)
        except Exception as exc:
            # KeyError (missing expected state-dict key, e.g. a QK-norm key the
            # standard BFL->diffusers converter expects) and RuntimeError from
            # split_with_sizes (a tensor shaped differently than the converter
            # assumes) both mean: this file's key layout doesn't match what the
            # standard converter for this family expects - usually a non-standard
            # FP8/scaled export rather than a corrupt file. Surface clearly instead
            # of an unhandled thread crash.
            logger.error(
                "%s transformer failed to load from %s (format=%s): %s: %s",
                family,
                path.name,
                load_format,
                type(exc).__name__,
                exc,
            )
            raise ModelNotFoundError(
                f"'{path.name}' failed to load as a {family} transformer "
                f"({type(exc).__name__}: {exc}). The file's internal layout doesn't "
                "match the standard converter for this architecture - likely a "
                "non-standard quantization/export. Try a different checkpoint."
            ) from exc
        if load_format in {"nf4", "fp4"}:
            transformer = transformer.to(self.devices.device())
        logger.info(
            "%s transformer loaded in %.1fs (format=%s) from %s",
            family,
            time.perf_counter() - transformer_t0,
            load_format,
            path.name,
        )
        return transformer

    @staticmethod
    def _flux_transformer_has_guidance(path: Path) -> bool:
        suffix = path.suffix.lower()
        try:
            if suffix == ".gguf":
                import gguf

                reader = gguf.GGUFReader(str(path))
                return any("guidance_in" in tensor.name for tensor in reader.tensors)
            if suffix == ".safetensors":
                from safetensors import safe_open

                with safe_open(path, framework="pt", device="cpu") as handle:
                    return any("guidance_in" in key for key in handle.keys())
        except Exception:
            logger.debug("Could not inspect Flux guidance tensors for %s", path, exc_info=True)
        name = path.name.lower()
        return not any(token in name for token in ("schnell", "fusion", "turbo", "lightning", "4step", "4-step"))

    def _flux_search_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[str] = set()
        for root in [self.flags.resolved_models_dir(), *self.flags.resolved_extra_model_dirs()]:
            try:
                resolved = root.resolve()
            except Exception:
                continue
            key = str(resolved).lower()
            if resolved.exists() and key not in seen:
                seen.add(key)
                roots.append(resolved)
        return roots

    def _find_flux_component(self, filenames: tuple[str, ...], subdirs: tuple[str, ...]) -> Path | None:
        # Prefer the first filename (e.g. fp8 T5) across all search roots before any fallback.
        for filename in filenames:
            for root in self._flux_search_roots():
                for subdir in subdirs:
                    base = root / subdir if subdir else root
                    candidate = base / filename
                    if candidate.is_file():
                        return candidate.resolve()
        return None

    def _diffusers_component_search_roots(self) -> list[Path]:
        return self._flux_search_roots()

    @staticmethod
    def _looks_like_diffusers_component_dir(path: Path) -> bool:
        try:
            if not (path / "model_index.json").is_file():
                return False
            required_dirs = ("scheduler", "text_encoder", "tokenizer", "vae")
            if not all((path / name).is_dir() for name in required_dirs):
                return False
            if not (path / "scheduler" / "scheduler_config.json").is_file():
                return False
            if not any((path / "tokenizer" / name).is_file() for name in ("tokenizer.json", "vocab.json")):
                return False
            if not any((path / "text_encoder").glob("*.safetensors")) and not any((path / "text_encoder").glob("*.bin")):
                return False
            if not any((path / "vae").glob("*.safetensors")) and not any((path / "vae").glob("*.bin")):
                return False
            return True
        except OSError:
            return False

    @staticmethod
    def _flux2_component_repo_name(checkpoint: Checkpoint) -> str:
        blob = f"{checkpoint.id} {checkpoint.title} {checkpoint.filename} {checkpoint.path}".lower()
        if "4b" in blob or "klein-4" in blob:
            return "FLUX.2-klein-4B"
        return "FLUX.2-klein-9B"

    def _cached_hf_component_dir(self, repo_id: str) -> Path | None:
        if _try_to_load_from_cache is None:
            return None
        try:
            cached = _try_to_load_from_cache(repo_id, "model_index.json")
        except Exception:
            return None
        if not isinstance(cached, str):
            return None
        path = Path(cached).parent
        return path if path.is_dir() else None

    def _component_dir_candidates(self, architecture: str, checkpoint: Checkpoint) -> list[Path]:
        roots = self._diffusers_component_search_roots()
        candidates: list[Path] = []
        if is_flux2_klein_architecture(architecture):
            repo_name = self._flux2_component_repo_name(checkpoint)
            candidates.extend(
                root / subdir
                for root in roots
                for subdir in (
                    f"flux2/Components/{repo_name}",
                    f"flux2/{repo_name}",
                    f"Flux2/{repo_name}",
                    repo_name,
                )
            )
            cached = self._cached_hf_component_dir(f"black-forest-labs/{repo_name}")
            if cached:
                candidates.append(cached)
        elif is_z_image_architecture(architecture):
            repo_name = "Z-Image-Turbo"
            candidates.extend(
                root / subdir
                for root in roots
                for subdir in (
                    f"z-image/Components/{repo_name}",
                    f"z-image/{repo_name}",
                    f"Z-Image/{repo_name}",
                    repo_name,
                )
            )
            cached = self._cached_hf_component_dir("Tongyi-MAI/Z-Image-Turbo")
            if cached:
                candidates.append(cached)
        seen: set[str] = set()
        unique: list[Path] = []
        for candidate in candidates:
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    def _resolve_component_dir(self, architecture: str, checkpoint: Checkpoint) -> Path:
        for candidate in self._component_dir_candidates(architecture, checkpoint):
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if self._looks_like_diffusers_component_dir(resolved):
                return resolved

        searched = ", ".join(str(path) for path in self._component_dir_candidates(architecture, checkpoint))
        if is_flux2_klein_architecture(architecture):
            repo_name = self._flux2_component_repo_name(checkpoint)
            key = "flux2-klein-4b-components" if repo_name.endswith("4B") else "flux2-klein-9b-components"
            raise ModelNotFoundError(
                f"Flux.2 Klein needs the matching Diffusers component folder for {repo_name}: "
                "text_encoder, tokenizer, scheduler, and VAE. "
                f"In the Models tab, use the 'Text-to-image Flux.2 Klein' quick-start button "
                f"(downloads the GGUF + `{key}` components), or manually download `{key}` from the Curated catalog. "
                f"9B components are HF-gated: accept the license and set your Hugging Face token in Settings first. "
                f"Alternative: place full `{repo_name}` snapshot under `models/flux2/Components/{repo_name}`. "
                f"Searched: {searched}"
            )
        raise ModelNotFoundError(
            "Z-Image needs the Z-Image-Turbo Diffusers component folder: text_encoder, tokenizer, "
            "scheduler, and VAE. Download `z-image-turbo-components` from Models, or place the full "
            f"snapshot under `models/z-image/Components/Z-Image-Turbo`. Searched: {searched}"
        )

    @staticmethod
    def _local_config_dir(component_dir: Path, subfolder: str) -> str | None:
        path = component_dir / subfolder
        return str(path) if (path / "config.json").is_file() else None

    @staticmethod
    def _component_dir_matches_repo(component_dir: Path, repo_name: str) -> bool:
        normalized_path = component_dir.as_posix().lower()
        normalized_repo = repo_name.lower()
        return normalized_repo in normalized_path or normalized_repo.replace(".", "-") in normalized_path

    @staticmethod
    @contextmanager
    def _gguf_singleton_shape_compat():
        """Allow GGUF files that omit leading singleton dims on non-linear params.

        Some Comfy-style Z-Image GGUF files store pad tokens as `[dim]`, while
        Diffusers declares those parameters as `[1, dim]`. Linear weights must
        stay quantized, so this only reshapes/dequantizes params whose inferred
        unquantized shape has the same element count but not the exact target
        shape.
        """
        try:
            import torch
            from diffusers.quantizers.gguf.gguf_quantizer import GGUFQuantizer
            from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
            from diffusers.quantizers.gguf.utils import _quant_shape_from_byte_shape
            from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES
            from diffusers.utils import get_module_from_name
        except Exception:
            yield
            return

        original_check = GGUFQuantizer.check_quantized_param_shape
        original_create = GGUFQuantizer.create_quantized_param

        def _numel(shape) -> int:
            total = 1
            for dim in shape:
                total *= int(dim)
            return total

        def _inferred_shape(param):
            block_size, type_size = GGML_QUANT_SIZES[param.quant_type]
            return tuple(_quant_shape_from_byte_shape(tuple(param.shape), type_size, block_size))

        def check_quantized_param_shape(self, param_name, current_param, loaded_param):
            try:
                return original_check(self, param_name, current_param, loaded_param)
            except ValueError:
                if isinstance(loaded_param, GGUFParameter):
                    inferred = _inferred_shape(loaded_param)
                    current = tuple(current_param.shape)
                    if inferred != current and _numel(inferred) == _numel(current):
                        logger.info(
                            "Allowing GGUF singleton reshape for %s: %s -> %s",
                            param_name,
                            inferred,
                            current,
                        )
                        return True
                raise

        def create_quantized_param(
            self,
            model,
            param_value,
            param_name,
            target_device,
            state_dict=None,
            unexpected_keys=None,
            **kwargs,
        ):
            if isinstance(param_value, GGUFParameter):
                module, tensor_name = get_module_from_name(model, param_name)
                target = None
                is_parameter = tensor_name in module._parameters
                if is_parameter:
                    target = module._parameters[tensor_name]
                elif tensor_name in module._buffers:
                    target = module._buffers[tensor_name]
                if target is not None:
                    inferred = _inferred_shape(param_value)
                    target_shape = tuple(target.shape)
                    if inferred != target_shape and _numel(inferred) == _numel(target_shape):
                        value = dequantize_gguf_tensor(param_value).reshape(target_shape).to(target_device)
                        if is_parameter:
                            module._parameters[tensor_name] = torch.nn.Parameter(
                                value,
                                requires_grad=getattr(target, "requires_grad", False),
                            )
                        else:
                            module._buffers[tensor_name] = value
                        return
            return original_create(
                self,
                model,
                param_value,
                param_name,
                target_device,
                state_dict=state_dict,
                unexpected_keys=unexpected_keys,
                **kwargs,
            )

        GGUFQuantizer.check_quantized_param_shape = check_quantized_param_shape
        GGUFQuantizer.create_quantized_param = create_quantized_param
        try:
            yield
        finally:
            GGUFQuantizer.check_quantized_param_shape = original_check
            GGUFQuantizer.create_quantized_param = original_create

    @staticmethod
    def _flux2_config_from_gguf(path: Path, fallback: dict) -> dict:
        if path.suffix.lower() != ".gguf":
            return fallback
        try:
            from gguf import GGUFReader
        except Exception:
            return fallback
        try:
            reader = GGUFReader(str(path))
            shapes = {tensor.name: [int(dim) for dim in tensor.shape] for tensor in reader.tensors}
        except Exception:
            logger.debug("Could not derive Flux.2 config from GGUF header for %s", path, exc_info=True)
            return fallback

        image_in = shapes.get("img_in.weight")
        text_in = shapes.get("txt_in.weight")
        time_in = shapes.get("time_in.in_layer.weight")
        norm = next(
            (
                shape
                for name, shape in shapes.items()
                if name.endswith("img_attn.norm.query_norm.weight") and shape
            ),
            None,
        )
        if not image_in or len(image_in) < 2:
            return fallback
        hidden = int(image_in[1])
        attention_head_dim = int(norm[0]) if norm else int(fallback.get("attention_head_dim", 128))
        num_attention_heads = max(1, hidden // max(1, attention_head_dim))
        double_layers = {
            int(name.split(".", 2)[1])
            for name in shapes
            if name.startswith("double_blocks.") and name.split(".", 2)[1].isdigit()
        }
        single_layers = {
            int(name.split(".", 2)[1])
            for name in shapes
            if name.startswith("single_blocks.") and name.split(".", 2)[1].isdigit()
        }
        derived = dict(fallback)
        derived.update(
            {
                "in_channels": int(image_in[0]),
                "num_layers": len(double_layers) or int(fallback.get("num_layers", 5)),
                "num_single_layers": len(single_layers) or int(fallback.get("num_single_layers", 20)),
                "attention_head_dim": attention_head_dim,
                "num_attention_heads": num_attention_heads,
                "joint_attention_dim": int(text_in[0]) if text_in and len(text_in) >= 2 else fallback.get("joint_attention_dim", 7680),
                "timestep_guidance_channels": int(time_in[0]) if time_in and len(time_in) >= 2 else fallback.get("timestep_guidance_channels", 256),
                "guidance_embeds": False,
            }
        )
        logger.info(
            "Derived Flux.2 GGUF config for %s: hidden=%s, heads=%s, double=%s, single=%s, joint=%s",
            path.name,
            hidden,
            num_attention_heads,
            derived["num_layers"],
            derived["num_single_layers"],
            derived["joint_attention_dim"],
        )
        return derived

    @staticmethod
    def _patch_gguf_linear_input_dtype() -> None:
        try:
            import torch
            from diffusers.quantizers.gguf import utils as gguf_utils
            from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
        except Exception:
            return
        cls = getattr(gguf_utils, "GGUFLinear", None)
        if cls is None or getattr(cls, "_aiwf_input_dtype_patch", False):
            return
        original = cls.forward_native

        def forward_native(self, inputs: torch.Tensor):
            compute_dtype = getattr(self, "compute_dtype", None)
            if compute_dtype is not None and inputs.dtype != compute_dtype:
                inputs = inputs.to(compute_dtype)
            try:
                weight = dequantize_gguf_tensor(self.weight)
                target_dtype = compute_dtype or weight.dtype
                weight = weight.to(device=inputs.device, dtype=target_dtype)
                bias = (
                    self.bias.to(device=inputs.device, dtype=target_dtype)
                    if self.bias is not None
                    else None
                )
                return torch.nn.functional.linear(inputs, weight, bias)
            except Exception:
                return original(self, inputs)

        cls.forward_native = forward_native
        cls._aiwf_input_dtype_patch = True

    def _resolve_flux_component_paths(self) -> dict[str, Path]:
        clip = self._find_flux_component(
            ("clip_l.safetensors",),
            ("flux/Textencoder", "Textencoder", "textencoder", "text_encoders", "Clip", "clip"),
        )
        override = self._flux_text_encoder_override
        if override and Path(override).is_file():
            t5: Path | None = Path(override).resolve()
        else:
            if override:
                logger.warning(
                    "Selected Flux text encoder %s no longer exists; falling back to auto-pick.",
                    override,
                )
            t5 = self._find_flux_component(
                (
                    "t5xxl_fp8_e4m3fn.safetensors",
                    "t5xxl_fp8_e4m3fn_scaled.safetensors",
                    "t5xxl_fp16.safetensors",
                ),
                ("flux/Textencoder", "Textencoder", "textencoder", "text_encoders"),
            )
        vae = self._find_flux_component(
            ("ae.safetensors",),
            ("flux/VAE", "VAE", "vae"),
        )
        missing = [
            name
            for name, value in (("CLIP-L", clip), ("T5-XXL", t5), ("Flux VAE", vae))
            if value is None
        ]
        if missing:
            roots = ", ".join(str(root) for root in self._flux_search_roots())
            raise ModelNotFoundError(
                "Flux generation needs local CLIP-L, T5-XXL, and ae.safetensors assets. "
                f"Missing: {', '.join(missing)}. Put them under models/flux/Textencoder and "
                f"models/flux/VAE, or add a shared model root in Settings. Searched: {roots}"
            )
        return {"clip_l": clip, "t5xxl": t5, "vae": vae}  # type: ignore[dict-item]

    def list_flux_text_encoders(self) -> list[tuple[str, str]]:
        """Return (label, path) for Flux-compatible T5-XXL text encoders found locally.

        UMT5 encoders (Wan video only) are excluded — they are not compatible
        with Flux and would produce broken output if selected.
        """
        from aiwf.infrastructure.model_header import read_model_info

        subdirs = ("flux/Textencoder", "Textencoder", "textencoder", "text_encoders")
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for root in self._flux_search_roots():
            for sub in subdirs:
                directory = root / sub
                if not directory.is_dir():
                    continue
                for path in sorted(directory.glob("*"), key=lambda p: p.name.lower()):
                    if path.suffix.lower() not in (".safetensors", ".gguf"):
                        continue
                    key = str(path.resolve()).lower()
                    if key in seen:
                        continue
                    try:
                        info = read_model_info(path)
                    except Exception:
                        continue
                    if not info.is_t5xxl():
                        continue
                    seen.add(key)
                    out.append((f"{path.stem}  [{info.size_label()}]", str(path.resolve())))
        return out

    def set_flux_text_encoder(self, path: str | None) -> None:
        """Choose which T5-XXL text encoder Flux uses (None = automatic best).

        Switching drops the resident encoders so the next generation reloads the
        chosen T5, and clears warm pipes' cached component map so they re-resolve
        the new path without needing a checkpoint switch.
        """
        new = path or None
        if new == self._flux_text_encoder_override:
            return
        self._flux_text_encoder_override = new
        self._flux_text_encoder = None
        self._flux_text_encoder_2 = None
        self._flux_tokenizer = None
        self._flux_tokenizer_2 = None
        self._flux_component_paths = {}
        self._flux_clip_device = None
        self._flux_t5_device = None
        self._flux_prompt_cache.clear()
        for pipe in (self._txt2img, self._img2img):
            if pipe is not None and hasattr(pipe, "_aiwf_flux_components"):
                try:
                    delattr(pipe, "_aiwf_flux_components")
                except Exception:
                    logger.debug("Could not clear cached Flux components on pipe", exc_info=True)
        gc.collect()
        self.devices.empty_cache()
        logger.info("Flux text encoder set to: %s", new or "automatic")

    @staticmethod
    def _write_temp_config(root: Path, name: str, config: dict) -> str:
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return str(target)

    def _load_flux_prompt_models(self, component_paths: dict[str, Path]) -> None:
        if (
            self._flux_text_encoder is not None
            and self._flux_text_encoder_2 is not None
            and self._flux_tokenizer is not None
            and self._flux_tokenizer_2 is not None
            and self._flux_component_paths == {key: str(value) for key, value in component_paths.items()}
        ):
            return

        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5TokenizerFast

        clip_config = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=77,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            bos_token_id=49406,
            eos_token_id=49407,
            pad_token_id=1,
            projection_dim=768,
        )
        t5_path = component_paths["t5xxl"]
        clip_device = self.devices.device() if self.devices.device().type == "cuda" else torch.device("cpu")

        clip = CLIPTextModel(clip_config)
        self._load_encoder_state_dict(clip, load_file(str(component_paths["clip_l"]), device="cpu"))
        clip.eval()
        clip = clip.to(dtype=torch.float16, device=clip_device)

        t5_config = T5Config(
            vocab_size=32128,
            d_model=4096,
            d_ff=10240,
            d_kv=64,
            num_layers=24,
            num_decoder_layers=24,
            num_heads=64,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.0,
            layer_norm_epsilon=1e-6,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=False,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=1,
        )
        t5, t5_device = self._load_flux_t5_encoder(t5_path, t5_config)

        self._flux_text_encoder = clip
        self._flux_text_encoder_2 = t5
        self._flux_clip_device = clip_device
        self._flux_t5_device = t5_device
        self._flux_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self._flux_tokenizer_2 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl", legacy=True)
        self._flux_component_paths = {key: str(value) for key, value in component_paths.items()}
        self._flux_prompt_cache.clear()
        logger.info("Flux text encoders warm: CLIP-L on %s, T5 on %s", clip_device, t5_device)

    def _encode_flux_prompt(
        self,
        prompt: str,
        *,
        device: torch.device,
        batch_size: int,
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._flux_text_encoder is not None
        assert self._flux_text_encoder_2 is not None
        assert self._flux_tokenizer is not None
        assert self._flux_tokenizer_2 is not None

        cache_key = f"{prompt}|{batch_size}|{max_sequence_length}"
        cached = self._flux_prompt_cache.get(cache_key)
        if cached is not None:
            prompt_embeds, pooled = cached
            return (
                prompt_embeds.to(dtype=torch.bfloat16, device=device),
                pooled.to(dtype=torch.bfloat16, device=device),
            )

        clip_device = self._flux_clip_device or device
        t5_device = self._flux_t5_device or device
        clip_inputs = self._flux_tokenizer(
            [prompt],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        t5_inputs = self._flux_tokenizer_2(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        clip_inputs = clip_inputs.to(clip_device)
        t5_inputs = t5_inputs.to(t5_device)
        encode_t0 = time.perf_counter()
        with torch.no_grad():
            pooled = self._flux_text_encoder(clip_inputs.input_ids, output_hidden_states=False).pooler_output
            prompt_embeds = self._flux_text_encoder_2(
                t5_inputs.input_ids,
                output_hidden_states=False,
            )[0]
        logger.info(
            "Flux prompt encoded in %.2fs (CLIP %s, T5 %s)",
            time.perf_counter() - encode_t0,
            clip_device,
            t5_device,
        )
        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16, device=device)
        pooled = pooled.to(dtype=torch.bfloat16, device=device)
        self._flux_prompt_cache[cache_key] = (
            prompt_embeds.detach().cpu(),
            pooled.detach().cpu(),
        )
        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            pooled = pooled.repeat(batch_size, 1)
        return prompt_embeds, pooled

    def _encode_flux_prompt_fast(
        self,
        pipe,
        prompt: str,
        *,
        device: torch.device,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the Flux prompt on the GPU when the T5 encoder is pinned to CPU.

        Loading the transformer first leaves no room for the ~9.5 GB T5-XXL, so
        it falls back to CPU and a single prompt takes ~2 minutes. Here we briefly
        evict the transformer to system RAM, run CLIP + T5 on the GPU (where T5
        fp16 has native fast kernels), then restore the transformer for denoising.

        Anything unexpected falls back to the original CPU encode, so this is
        never slower-than-broken: the worst case is today's behaviour.
        """
        cuda = self.devices.device()
        t5_on_cpu = self._flux_t5_device is not None and self._flux_t5_device.type == "cpu"

        # Cached prompts don't benefit from the swap.
        cache_key = f"{prompt}|{batch_size}|256"
        already_cached = self._flux_prompt_cache.get(cache_key) is not None
        if (
            cuda.type != "cuda"
            or not t5_on_cpu
            or already_cached
            or self._flux_text_encoder_2 is None
        ):
            return self._encode_flux_prompt(prompt, device=device, batch_size=batch_size)

        if self._offload_active:
            # Under model offload, the transformer is already managed by accelerate.
            # We only need to swap T5 (and CLIP) to GPU for encoding, then park them back on CPU.
            try:
                self._flux_text_encoder_2.to(cuda)
                self._flux_t5_device = cuda
                if self._flux_text_encoder is not None:
                    self._flux_text_encoder.to(cuda)
                    self._flux_clip_device = cuda
                logger.info("Flux offloaded encode: T5/CLIP moved to GPU for encoding")
                return self._encode_flux_prompt(prompt, device=cuda, batch_size=batch_size)
            except Exception:
                logger.warning(
                    "Flux offloaded GPU prompt-encode swap failed; falling back to CPU encode.", exc_info=True
                )
                return self._encode_flux_prompt(prompt, device=device, batch_size=batch_size)
            finally:
                try:
                    self._flux_text_encoder_2.to("cpu")
                    self._flux_t5_device = torch.device("cpu")
                    if self._flux_text_encoder is not None:
                        self._flux_text_encoder.to("cpu")
                        self._flux_clip_device = torch.device("cpu")
                    self.devices.empty_cache()
                except Exception:
                    logger.debug("Could not park T5/CLIP back on CPU under offload", exc_info=True)

        transformer = getattr(pipe, "transformer", None)
        swap_t0 = time.perf_counter()
        moved_transformer = False
        try:
            if transformer is not None:
                transformer.to("cpu")
                moved_transformer = True
                self.devices.empty_cache()
            self._flux_text_encoder_2.to(cuda)
            self._flux_t5_device = cuda
            if self._flux_text_encoder is not None:
                self._flux_text_encoder.to(cuda)
                self._flux_clip_device = cuda
            logger.info("Flux fast encode: T5 moved to %s for prompt encoding", cuda)
            return self._encode_flux_prompt(prompt, device=cuda, batch_size=batch_size)
        except Exception:
            logger.warning(
                "Flux GPU prompt-encode swap failed; falling back to CPU encode.", exc_info=True
            )
            try:
                self._flux_text_encoder_2.to("cpu")
                self._flux_t5_device = torch.device("cpu")
            except Exception:
                logger.debug("Could not park T5 back on CPU after failed swap", exc_info=True)
            return self._encode_flux_prompt(prompt, device=device, batch_size=batch_size)
        finally:
            # Always return T5 to CPU and the transformer to the GPU so the
            # denoise loop and the warm-pipeline cache keep working.
            try:
                self._flux_text_encoder_2.to("cpu")
                self._flux_t5_device = torch.device("cpu")
                self.devices.empty_cache()
            except Exception:
                logger.debug("Could not park T5 back on CPU", exc_info=True)
            if moved_transformer and transformer is not None:
                try:
                    transformer.to(cuda)
                except Exception:
                    logger.warning("Failed to restore Flux transformer to GPU after encode", exc_info=True)
            logger.info("Flux fast encode: transformer restored in %.2fs", time.perf_counter() - swap_t0)

    def _encode_flux2_prompt(self, pipe, prompt: str, device: torch.device) -> torch.Tensor:
        """Cache Qwen3 prompt embeddings — Flux.2 Klein re-encodes every call otherwise."""
        cache_key = prompt
        cached = self._flux2_prompt_cache.get(cache_key)
        if cached is not None:
            return cached.to(device=device)
        try:
            from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
        except ImportError as exc:
            raise ModelNotFoundError("Flux.2 Klein prompt encoding is unavailable.") from exc
        text_encoder_device = getattr(getattr(pipe, "text_encoder", None), "device", None)
        target_device = text_encoder_device or device
        encode_t0 = time.perf_counter()
        offload_note = f" on {target_device}"
        logger.info("Flux.2 Klein: starting prompt encode%s", offload_note)
        prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            prompt=prompt,
            device=target_device,
        )
        logger.info("Flux.2 Klein prompt encoded in %.2fs", time.perf_counter() - encode_t0)
        self._flux2_prompt_cache[cache_key] = prompt_embeds.detach().cpu()
        return prompt_embeds.to(device=device)

    def _encode_z_image_prompts(
        self,
        pipe,
        prompt: str,
        negative_prompt: str | None,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        cache_key = (prompt, negative_prompt or "")
        cached = self._z_image_prompt_cache.get(cache_key)
        if cached is not None:
            prompt_embeds, negative_embeds = cached
            return (
                [tensor.to(device=device) for tensor in prompt_embeds],
                [tensor.to(device=device) for tensor in negative_embeds],
            )
        text_encoder_device = getattr(getattr(pipe, "text_encoder", None), "device", None)
        target_device = text_encoder_device or device
        encode_t0 = time.perf_counter()
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=target_device,
            do_classifier_free_guidance=getattr(pipe, "do_classifier_free_guidance", True),
        )
        logger.info("Z-Image prompt encoded in %.2fs", time.perf_counter() - encode_t0)
        self._z_image_prompt_cache[cache_key] = (
            [tensor.detach().cpu() for tensor in prompt_embeds],
            [tensor.detach().cpu() for tensor in negative_embeds],
        )
        return (
            [tensor.to(device=device) for tensor in prompt_embeds],
            [tensor.to(device=device) for tensor in negative_embeds],
        )

    def prewarm_common_prompt_embeddings(self, *, limit: int = 24, budget_seconds: float = 300.0) -> int:
        """Prime runtime prompt-embedding caches with short common starters.

        This is intentionally in-memory and bounded. It is meant for the
        background warmup thread so common portrait/person/body tests do not pay
        the full text-encoder cost on first use after the model is already warm.
        """
        pipe = self._txt2img
        checkpoint = self._active
        if pipe is None or checkpoint is None:
            return 0
        if checkpoint.id in self._common_prompt_cache_warmed_for:
            return 0
        prompts = [p for p in self._COMMON_PROMPT_STARTERS[: max(0, int(limit))] if p]
        if not prompts:
            return 0

        device = self._execution_device(pipe)
        started = time.perf_counter()
        warmed = 0
        try:
            if is_flux2_klein_architecture(checkpoint.architecture):
                for prompt in prompts:
                    if warmed and time.perf_counter() - started >= budget_seconds:
                        break
                    if prompt in self._flux2_prompt_cache:
                        continue
                    self._encode_flux2_prompt(pipe, prompt, device)
                    warmed += 1
            elif is_z_image_architecture(checkpoint.architecture):
                for prompt in prompts:
                    if warmed and time.perf_counter() - started >= budget_seconds:
                        break
                    cache_key = (prompt, "")
                    if cache_key in self._z_image_prompt_cache:
                        continue
                    self._encode_z_image_prompts(pipe, prompt, None, device)
                    warmed += 1
            elif is_flux_architecture(checkpoint.architecture):
                component_paths = getattr(pipe, "_aiwf_flux_components", None) or self._resolve_flux_component_paths()
                self._load_flux_prompt_models({key: Path(value) for key, value in component_paths.items()})
                for prompt in prompts:
                    if warmed and time.perf_counter() - started >= budget_seconds:
                        break
                    cache_key = f"{prompt}|1|256"
                    if cache_key in self._flux_prompt_cache:
                        continue
                    self._encode_flux_prompt(prompt, device=device, batch_size=1)
                    warmed += 1
        finally:
            self._common_prompt_cache_warmed_for.add(checkpoint.id)

        if warmed:
            logger.info("Prewarmed %d common prompt embeddings for %s.", warmed, checkpoint.title)
        return warmed

    @staticmethod
    def _txt2img_pipeline_cls_for_architecture(architecture: str):
        if is_flux_kontext_architecture(architecture):
            return FluxKontextPipeline
        if is_flux_architecture(architecture):
            return FluxPipeline
        if is_sd3_architecture(architecture):
            return StableDiffusion3Pipeline
        if is_sdxl_architecture(architecture):
            return StableDiffusionXLPipeline
        return StableDiffusionPipeline

    @staticmethod
    def _is_sd3_pipe(pipe) -> bool:
        return hasattr(pipe, "transformer") and hasattr(pipe, "text_encoder_3")

    @staticmethod
    def _is_flux_pipe(pipe) -> bool:
        return isinstance(pipe, FluxPipeline)

    @staticmethod
    def _is_flux2_pipe(pipe) -> bool:
        return pipe.__class__.__name__ == "Flux2KleinPipeline"

    @staticmethod
    def _is_z_image_pipe(pipe) -> bool:
        return pipe.__class__.__name__ == "ZImagePipeline"

    @staticmethod
    def _is_qwen_image_pipe(pipe) -> bool:
        return pipe.__class__.__name__ == "QwenImagePipeline"

    @staticmethod
    def _is_sana_pipe(pipe) -> bool:
        return pipe.__class__.__name__ in {"SanaPipeline", "SanaSprintPipeline", "SanaPAGPipeline"}

    @staticmethod
    def _is_flux_kontext_pipe(pipe) -> bool:
        return pipe.__class__.__name__ == "FluxKontextPipeline"

    def _apply_fp8_storage(self, pipe) -> None:
        """Store classic UNet weights in FP8 (SD/SDXL VRAM saver — not for GGUF/DiT transformers)."""
        if not self.flags.fp8:
            return
        if self.flags.lowvram:
            logger.warning("FP8 weight storage is skipped in Low VRAM mode (conflicting offload hooks).")
            return
        denoiser = getattr(pipe, "unet", None)
        if denoiser is None:
            # Flux / Flux2 / Z-Image / SD3 use transformers or GGUF quants — FP8 storage is not applicable.
            return
        if not hasattr(denoiser, "enable_layerwise_casting"):
            logger.warning("FP8 weight storage not supported by this diffusers version; continuing at fp16.")
            return
        try:
            denoiser.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=getattr(denoiser, "dtype", None) or self.devices.dtype(self.flags.no_half),
            )
            logger.info("Denoiser weights stored in FP8 (compute fp16); roughly half the denoiser VRAM.")
        except Exception:
            logger.exception("Could not enable FP8 weight storage; continuing at fp16.")

    def _execution_device(self, pipe) -> torch.device:
        """Device computation actually runs on — never the 'meta' placeholder.

        Offloaded pipelines (lowvram/medvram/SDXL auto-offload) report their
        device as 'meta'; torch.Generator and .to() need the real one.
        """
        if self._is_flux2_pipe(pipe) or self._is_z_image_pipe(pipe) or self._is_qwen_image_pipe(pipe) or self._is_sana_pipe(pipe):
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None:
                try:
                    transformer_device = next(transformer.parameters()).device
                    if getattr(transformer_device, "type", None) not in (None, "cpu", "meta"):
                        return transformer_device
                except StopIteration:
                    pass
                except Exception:
                    logger.debug("Could not inspect transformer execution device", exc_info=True)
        try:
            dev = pipe._execution_device
        except Exception:
            dev = getattr(pipe, "device", None)
        if dev is None or getattr(dev, "type", "meta") == "meta":
            return self.devices.device()
        return dev

    def _offload_threshold_gb(self, architecture: str) -> float:
        """The VRAM threshold (GB) actually used to decide CPU offload for `architecture`.

        Kept as a single source of truth so the auto-offload log message
        (`_place_pipeline`) can never drift from the real decision logic here
        — it used to hardcode `_AUTO_OFFLOAD_VRAM_GB` (10.0) for every
        architecture, which is wrong for Flux/Flux.2 Klein/Z-Image (20.0, or
        11.0 under fluxfp8) and produced misleading log lines like
        "Flux.2 Klein on a <10 GB GPU" on a 16GB card that was actually
        evaluated against a 20GB threshold.
        """
        if (
            is_flux_architecture(architecture)
            or is_flux2_klein_architecture(architecture)
            or is_z_image_architecture(architecture)
            or is_qwen_image_architecture(architecture)
            or is_sana_architecture(architecture)
        ):
            return 11.0 if getattr(self.flags, "fluxfp8", False) else 20.0
        if is_sd3_architecture(architecture):
            return 24.0
        if is_sdxl_architecture(architecture):
            return 7.0 if self.flags.fp8 else self._AUTO_OFFLOAD_VRAM_GB
        return self._AUTO_OFFLOAD_VRAM_GB

    def _wants_offload(self, architecture: str) -> bool:
        # DiT image families (Flux / Flux.2 / Z-Image / Qwen / Sana) stay GPU-resident for warm reuse.
        # CPU offload shuffles weights every generation and feels like a full reload.
        vram = self.devices.total_vram_gb()
        if (
            is_flux_architecture(architecture)
            or is_flux2_klein_architecture(architecture)
            or is_z_image_architecture(architecture)
            or is_qwen_image_architecture(architecture)
            or is_sana_architecture(architecture)
        ):
            # On cards with less than 20 GB (e.g. 16 GB, 12 GB, 8 GB), we must offload large DiT models
            # to avoid VRAM exhaustion / driver-level system memory paging.
            # If the user runs Flux in FP8, it fits in VRAM on 12/16GB cards, so we don't offload (threshold is 11.0 GB).
            return vram < self._offload_threshold_gb(architecture)
        if is_sd3_architecture(architecture):
            return 0.0 < vram < 24.0
        if not is_sdxl_architecture(architecture) or vram <= 0.0:
            return False
        # FP8 storage halves the active denoiser, so ~8GB cards can keep the whole
        # pipeline resident — much faster than offloading.
        return vram < self._offload_threshold_gb(architecture)

    def _compile_allowed_for_architecture(self, architecture: str) -> bool:
        if is_transformer_image_architecture(architecture):
            return False
        return not (self.flags.lowvram or self.flags.medvram or self._wants_offload(architecture))

    def _tune_vae_memory(self, pipe, architecture: str) -> None:
        """SDXL's 1024px VAE decode is the peak-VRAM step — slice and tile it."""
        if not (
            is_sdxl_architecture(architecture)
            or is_sd3_architecture(architecture)
            or is_transformer_image_architecture(architecture)
        ):
            return
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return
        try:
            vae.enable_slicing()
            vae.enable_tiling()
            pipe._aiwf_sdxl = True
            logger.info("VAE slicing + tiling enabled (cuts decode VRAM spike)")
        except Exception:
            logger.debug("Could not enable VAE slicing/tiling", exc_info=True)

    def _sync_img2img_from_txt2img(self) -> None:
        assert self._txt2img is not None
        pipe = self._txt2img
        if self._active is not None and is_sd3_architecture(self._active.architecture):
            self._img2img = StableDiffusion3Img2ImgPipeline.from_pipe(pipe)
        elif hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
            self._img2img = StableDiffusionXLImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
            )
        else:
            self._img2img = StableDiffusionImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        if not self._offload_active:
            self._img2img = self._img2img.to(self.devices.device())
        base_config = self._base_scheduler_config_for_pipe(pipe)
        if self._img2img is not None and base_config:
            self._img2img._aiwf_base_scheduler_config = copy.deepcopy(base_config)

    def _apply_vae(self, pipe, vae_id: str | None) -> None:
        # Track the applied VAE per pipeline: txt2img and inpaint are separate
        # pipes, so a single global id would skip applying to the second pipe.
        if (vae_id or None) == getattr(pipe, "_aiwf_vae_id", None):
            return

        if not vae_id:
            self._active_vae_id = None
            return

        vae_info = resolve_vae(self.list_vaes(), vae_id)
        if vae_info is None:
            logger.warning("VAE %s not found", vae_id)
            return

        logger.info("Loading VAE %s", vae_info.title)
        dtype = self.devices.dtype(self.flags.no_half)
        vae = AutoencoderKL.from_single_file(vae_info.path, torch_dtype=dtype)
        device = self._execution_device(pipe)
        pipe.vae = vae.to(device)
        if getattr(pipe, "_aiwf_sdxl", False):
            try:
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
            except Exception:
                logger.debug("Could not re-enable VAE slicing/tiling", exc_info=True)
        apply_image_pipeline_optimizations(
            pipe,
            self.flags,
            compile_allowed=not self._offload_active,
            include_unet=False,
            include_vae=True,
        )
        pipe._aiwf_vae_id = vae_info.id
        self._active_vae_id = vae_info.id
        if pipe is self._txt2img:
            self._sync_img2img_from_txt2img()
            if self._img2img is not None:
                self._img2img._aiwf_vae_id = vae_info.id

    def resolve_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        return self._resolve_checkpoint(checkpoint_id)

    def can_preload_checkpoint_locally(self, checkpoint_id: str | None = None) -> bool:
        checkpoint = self._resolve_checkpoint(checkpoint_id)
        if is_flux_kontext_architecture(checkpoint.architecture):
            path = Path(checkpoint.path)
            return path.is_dir() and (path / "model_index.json").is_file()
        if is_flux_architecture(checkpoint.architecture):
            try:
                self._resolve_flux_component_paths()
            except ModelNotFoundError:
                return False
            return Path(checkpoint.path).is_file()
        if is_flux2_klein_architecture(checkpoint.architecture) or is_z_image_architecture(checkpoint.architecture):
            path = Path(checkpoint.path)
            if path.is_dir():
                return (path / "model_index.json").is_file()
            try:
                self._resolve_component_dir(checkpoint.architecture, checkpoint)
            except ModelNotFoundError:
                return False
            return path.is_file()
        if is_qwen_nunchaku_architecture(checkpoint.architecture):
            return self._qwen_nunchaku.status(checkpoint.path).ready
        if is_qwen_image_architecture(checkpoint.architecture) or is_sana_architecture(checkpoint.architecture):
            path = Path(checkpoint.path)
            return path.is_dir() and (path / "model_index.json").is_file()
        pipeline_cls = self._txt2img_pipeline_cls_for_architecture(checkpoint.architecture)
        if Path(checkpoint.path).is_dir():
            return (Path(checkpoint.path) / "model_index.json").is_file()
        return _cached_single_file_config_dir(pipeline_cls) is not None

    def _load_flux_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        if self._txt2img is not None and self._active and self._active.path == checkpoint.path:
            logger.debug("Flux checkpoint already warm: %s", checkpoint.title)
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            # Switching to another Flux transformer: keep the (identical, ~9.8 GB)
            # CLIP-L + T5 text encoders resident so we skip the disk reload.
            self.unload(keep_flux_encoders=True)
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        path = Path(checkpoint.path)
        if path.suffix.lower() not in {".gguf", ".safetensors"}:
            raise ModelNotFoundError("Flux generation currently expects a .gguf or .safetensors transformer file.")
        component_paths = self._resolve_flux_component_paths()
        guidance_embeds = self._flux_transformer_has_guidance(path)
        dtype = self._dtype_for_architecture(ARCH_FLUX)
        if path.suffix.lower() == ".gguf":
            dtype = self._gguf_compute_dtype(dtype)
        elif path.suffix.lower() == ".safetensors":
            bnb_report = inspect_bnb_4bit_safetensors(path)
            if bnb_report.is_bnb_4bit:
                dtype = normalize_bnb_4bit_compute_dtype(dtype)

        transformer_config = dict(_FLUX_TRANSFORMER_CONFIG_BASE)
        transformer_config["guidance_embeds"] = guidance_embeds

        config_root = self._flux_config_cache_root()
        transformer_config_dir = self._write_temp_config(config_root, "transformer", transformer_config)
        vae_config_dir = self._write_temp_config(config_root, "vae", _FLUX_VAE_CONFIG)
        transformer = self._load_dit_transformer_single_file(
            FluxTransformer2DModel,
            path,
            config_dir=transformer_config_dir,
            dtype=dtype,
            family="Flux",
        )
        vae = AutoencoderKL.from_single_file(
            str(component_paths["vae"]),
            config=vae_config_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )

        pipe = FluxPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            transformer=transformer,
        )
        pipe._aiwf_flux_guidance_embeds = guidance_embeds
        pipe._aiwf_flux_components = {key: str(value) for key, value in component_paths.items()}
        if getattr(transformer, "_aiwf_bnb_original_layout", False):
            pipe._aiwf_cast_vae_decode_dtype = True
        pipe = pipe.to(self.devices.device())
        pipe.set_progress_bar_config(disable=True)
        self._remember_base_scheduler_config(pipe)
        apply_attention_optimizations(
            pipe,
            self.flags,
            compile_allowed=self._compile_allowed_for_architecture(ARCH_FLUX),
        )
        self._tune_vae_memory(pipe, ARCH_FLUX)

        self._load_flux_prompt_models(component_paths)

        self._active = checkpoint
        self._txt2img = pipe
        self._img2img = None
        return checkpoint

    def _load_flux2_klein_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        if self._txt2img is not None and self._active and self._active.path == checkpoint.path:
            logger.debug("Flux.2 Klein checkpoint already warm: %s", checkpoint.title)
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            self.unload()
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        try:
            from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
            from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
        except ImportError as exc:
            raise ModelNotFoundError(
                "Flux.2 Klein support needs a newer Diffusers/Transformers stack. "
                "Install the optional Flux.2 Klein engine dependencies first."
            ) from exc

        dtype = self._dtype_for_architecture(ARCH_FLUX2_KLEIN)
        path = Path(checkpoint.path)
        if path.suffix.lower() == ".gguf":
            dtype = self._gguf_compute_dtype(dtype)
        if path.is_dir():
            if dtype is torch.float8_e4m3fn:
                dtype = self._gguf_compute_dtype(dtype)
            pipe = Flux2KleinPipeline.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
        else:
            if path.suffix.lower() not in {".gguf", ".safetensors"}:
                raise ModelNotFoundError("Flux.2 Klein expects a .gguf or .safetensors transformer file.")
            component_dir = self._resolve_component_dir(ARCH_FLUX2_KLEIN, checkpoint)
            repo_name = self._flux2_component_repo_name(checkpoint)
            fallback_config = (
                _FLUX2_KLEIN_4B_TRANSFORMER_CONFIG
                if repo_name.endswith("4B")
                else _FLUX2_KLEIN_9B_TRANSFORMER_CONFIG
            )
            fallback_config = self._flux2_config_from_gguf(path, fallback_config)
            transformer_config_dir = (
                (
                    self._local_config_dir(component_dir, "transformer")
                    if path.suffix.lower() != ".gguf"
                    and self._component_dir_matches_repo(component_dir, repo_name)
                    else None
                )
                or self._cached_transformer_config_dir("flux2", path, fallback_config)
            )
            transformer = self._load_dit_transformer_single_file(
                Flux2Transformer2DModel,
                path,
                config_dir=transformer_config_dir,
                dtype=dtype,
                family="Flux.2 Klein",
            )

            vae = AutoencoderKLFlux2.from_pretrained(
                str(component_dir / "vae"),
                torch_dtype=dtype,
                local_files_only=True,
            )
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                str(component_dir / "text_encoder"),
                dtype=dtype,
                local_files_only=True,
            )
            tokenizer = Qwen2TokenizerFast.from_pretrained(str(component_dir / "tokenizer"), local_files_only=True)
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(component_dir / "scheduler"),
                local_files_only=True,
            )
            pipe = Flux2KleinPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
                is_distilled=True,
            )

        self._remember_base_scheduler_config(pipe)
        apply_attention_optimizations(
            pipe,
            self.flags,
            compile_allowed=self._compile_allowed_for_architecture(ARCH_FLUX2_KLEIN),
        )
        pipe = self._place_transformer_pipeline_keep_text_cpu(pipe, architecture=ARCH_FLUX2_KLEIN)
        pipe.set_progress_bar_config(disable=True)
        self._tune_vae_memory(pipe, ARCH_FLUX2_KLEIN)

        self._flux2_prompt_cache.clear()
        self._z_image_prompt_cache.clear()
        self._active = checkpoint
        self._txt2img = pipe
        self._img2img = None
        return checkpoint

    def _load_z_image_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        if self._txt2img is not None and self._active and self._active.path == checkpoint.path:
            logger.debug("Z-Image checkpoint already warm: %s", checkpoint.title)
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            self.unload()
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        try:
            from diffusers import ZImagePipeline, ZImageTransformer2DModel
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ModelNotFoundError(
                "Z-Image support needs a newer Diffusers/Transformers stack. "
                "Install the optional Z-Image engine dependencies first."
            ) from exc

        dtype = self._dtype_for_architecture(ARCH_Z_IMAGE)
        path = Path(checkpoint.path)
        if path.suffix.lower() == ".gguf":
            dtype = self._gguf_compute_dtype(dtype)
        if path.is_dir():
            if dtype is torch.float8_e4m3fn:
                dtype = self._gguf_compute_dtype(dtype)
            pipe = ZImagePipeline.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
        else:
            if path.suffix.lower() not in {".gguf", ".safetensors"}:
                raise ModelNotFoundError("Z-Image expects a .gguf or .safetensors transformer file.")
            component_dir = self._resolve_component_dir(ARCH_Z_IMAGE, checkpoint)
            transformer_config_dir = (
                self._local_config_dir(component_dir, "transformer")
                or self._cached_transformer_config_dir("z_image", path, _Z_IMAGE_TRANSFORMER_CONFIG)
            )
            load_kwargs = {
                "config": transformer_config_dir,
                "torch_dtype": dtype,
                "local_files_only": True,
            }
            if path.suffix.lower() == ".gguf":
                load_kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dtype)
                self._patch_gguf_linear_input_dtype()
                with self._gguf_singleton_shape_compat():
                    transformer_t0 = time.perf_counter()
                    transformer = ZImageTransformer2DModel.from_single_file(str(path), **load_kwargs)
            else:
                transformer_t0 = time.perf_counter()
                transformer = ZImageTransformer2DModel.from_single_file(str(path), **load_kwargs)
            logger.info(
                "Z-Image transformer loaded in %.1fs from %s",
                time.perf_counter() - transformer_t0,
                path.name,
            )

            vae = AutoencoderKL.from_pretrained(
                str(component_dir / "vae"),
                torch_dtype=dtype,
                local_files_only=True,
            )
            text_encoder = AutoModel.from_pretrained(
                str(component_dir / "text_encoder"),
                dtype=dtype,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(component_dir / "tokenizer"), local_files_only=True)
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(component_dir / "scheduler"),
                local_files_only=True,
            )
            pipe = ZImagePipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
            )

        self._remember_base_scheduler_config(pipe)
        apply_attention_optimizations(
            pipe,
            self.flags,
            compile_allowed=self._compile_allowed_for_architecture(ARCH_Z_IMAGE),
        )
        pipe = self._place_transformer_pipeline_keep_text_cpu(pipe, architecture=ARCH_Z_IMAGE)
        pipe.set_progress_bar_config(disable=True)
        self._tune_vae_memory(pipe, ARCH_Z_IMAGE)

        self._z_image_prompt_cache.clear()
        self._active = checkpoint
        self._txt2img = pipe
        self._img2img = None
        return checkpoint

    @staticmethod
    def _pipeline_class_name(path: Path) -> str:
        try:
            payload = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
            return str(payload.get("_class_name") or "")
        except Exception:
            return ""

    def _load_qwen_nunchaku_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        if self._txt2img is self._QWEN_NUNCHAKU_SENTINEL and self._active and self._active.path == checkpoint.path:
            logger.debug("Qwen Nunchaku checkpoint already warm: %s", checkpoint.title)
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            self.unload()
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        try:
            status = self._qwen_nunchaku.status(checkpoint.path)
        except Exception as exc:
            raise ModelNotFoundError(f"Qwen Nunchaku runtime probe failed: {exc}") from exc
        if not status.ready:
            details = "; ".join(status.messages) if status.messages else "runtime not ready"
            raise ModelNotFoundError(
                "Qwen Nunchaku needs its isolated engine runtime and local base assets. "
                f"Details: {details}"
            )

        self._active = checkpoint
        self._txt2img = self._QWEN_NUNCHAKU_SENTINEL
        self._img2img = None
        return checkpoint

    def _load_qwen_image_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        if self._txt2img is not None and self._active and self._active.path == checkpoint.path:
            logger.debug("Qwen Image checkpoint already warm: %s", checkpoint.title)
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            self.unload()
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        path = Path(checkpoint.path)
        if not path.is_dir() or not (path / "model_index.json").is_file():
            raise ModelNotFoundError(
                f"'{checkpoint.title}' must be a full QwenImagePipeline folder with model_index.json. "
                "Single-file Qwen Image checkpoints are not wired in AIWF yet."
            )

        try:
            from diffusers import QwenImagePipeline
        except ImportError as exc:
            raise ModelNotFoundError(
                "Qwen Image support needs a newer Diffusers stack. Install the Qwen Image engine dependencies first."
            ) from exc

        dtype = self._dtype_for_architecture(ARCH_QWEN_IMAGE)
        logger.info("Loading Qwen Image checkpoint %s from %s", checkpoint.title, path)
        try:
            pipe = QwenImagePipeline.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
        except TypeError:
            pipe = QwenImagePipeline.from_pretrained(str(path), dtype=dtype, local_files_only=True)
        except Exception as exc:
            raise ModelNotFoundError(
                f"'{checkpoint.title}' failed to load as Qwen Image ({type(exc).__name__}: {exc}). "
                "Check that the full Diffusers snapshot downloaded completely."
            ) from exc

        self._remember_base_scheduler_config(pipe)
        apply_attention_optimizations(
            pipe,
            self.flags,
            compile_allowed=self._compile_allowed_for_architecture(ARCH_QWEN_IMAGE),
        )
        pipe = self._place_transformer_pipeline_keep_text_cpu(pipe, architecture=ARCH_QWEN_IMAGE)
        pipe.set_progress_bar_config(disable=True)
        self._tune_vae_memory(pipe, ARCH_QWEN_IMAGE)

        self._active = checkpoint
        self._txt2img = pipe
        self._img2img = None
        return checkpoint

    def _load_sana_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        if self._txt2img is not None and self._active and self._active.path == checkpoint.path:
            logger.debug("Sana checkpoint already warm: %s", checkpoint.title)
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            self.unload()
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        path = Path(checkpoint.path)
        if not path.is_dir() or not (path / "model_index.json").is_file():
            raise ModelNotFoundError(
                f"'{checkpoint.title}' must be a full SanaPipeline/SanaSprintPipeline folder with model_index.json. "
                "Single-file Sana checkpoints are not wired in AIWF yet."
            )

        class_name = self._pipeline_class_name(path)
        try:
            from diffusers import SanaPipeline, SanaSprintPipeline
        except ImportError as exc:
            raise ModelNotFoundError(
                "Sana support needs a newer Diffusers stack. Install the Sana engine dependencies first."
            ) from exc
        pipeline_cls = SanaSprintPipeline if class_name == "SanaSprintPipeline" else SanaPipeline

        dtype = self._dtype_for_architecture(ARCH_SANA)
        logger.info("Loading %s checkpoint %s from %s", pipeline_cls.__name__, checkpoint.title, path)
        try:
            pipe = pipeline_cls.from_pretrained(str(path), torch_dtype=dtype, local_files_only=True)
        except TypeError:
            pipe = pipeline_cls.from_pretrained(str(path), dtype=dtype, local_files_only=True)
        except Exception as exc:
            raise ModelNotFoundError(
                f"'{checkpoint.title}' failed to load as Sana ({type(exc).__name__}: {exc}). "
                "Check that the full Diffusers snapshot downloaded completely."
            ) from exc

        self._remember_base_scheduler_config(pipe)
        # Sana uses a custom linear attention layout; the generic AttnProcessor2_0
        # swap used for SD/SDXL corrupts its q/k/v tensor shapes.
        pipe = self._place_transformer_pipeline_keep_text_cpu(pipe, architecture=ARCH_SANA)
        pipe.set_progress_bar_config(disable=True)
        self._tune_vae_memory(pipe, ARCH_SANA)

        self._active = checkpoint
        self._txt2img = pipe
        self._img2img = None
        return checkpoint

    def _run_flux_kontext_txt2img_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        *,
        width: int,
        height: int,
        steps: int,
        should_cancel: Callable[[], bool] | None = None,
    ):
        """Text-only generation through a FluxKontextPipeline (no init image).

        Unlike base Flux, this pipeline loads its own text_encoder/text_encoder_2/
        tokenizer/tokenizer_2 directly via from_pretrained (not the AIWF single-file
        CLIP/T5 cache-encode path), so prompts are passed straight through and
        diffusers handles encoding internally - no _aiwf_flux_components lookup.
        This is the smoke-test path only; real Kontext image-conditioned editing
        (passing `image=`) is not wired into generate() yet.
        """
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        return self._call_pipe(
            pipe,
            prompt=parsed_prompt,
            num_inference_steps=steps,
            guidance_scale=float(request.cfg_scale),
            num_images_per_prompt=1,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            width=width,
            height=height,
            max_sequence_length=512,
            output_type="pil",
        )

    def _load_flux_kontext_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        """Load a FluxKontextPipeline checkpoint (image-to-image Kontext variant of
        Flux.1-Kontext-dev, e.g. eramth/flux-kontext-4bit-fp4).

        Unlike base Flux, AIWF doesn't support this as a single-file .safetensors/.gguf
        transformer - Kontext checkpoints ship as a full HF-style multi-component
        directory (transformer/text_encoder/text_encoder_2/vae/tokenizers), already
        quantized in their own config (e.g. bnb 4-bit fp4), so a plain
        `from_pretrained()` on the directory is the correct and only load path. This is
        a probe/smoke-test loader only - no img2img dispatch wiring yet (generate()
        still raises a clean ValueError for IMG2IMG on transformer-image architectures).
        """
        if self._txt2img is not None and self._active and self._active.path == checkpoint.path:
            logger.debug("Flux Kontext checkpoint already warm: %s", checkpoint.title)
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            self.unload()
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        path = Path(checkpoint.path)
        if not path.is_dir() or not (path / "model_index.json").is_file():
            raise ModelNotFoundError(
                f"'{checkpoint.title}' doesn't look like a FluxKontextPipeline export - "
                "expected a directory with model_index.json (transformer/text_encoder/"
                "text_encoder_2/vae/tokenizer components), not a single-file checkpoint."
            )

        dtype = self._dtype_for_architecture(ARCH_FLUX_KONTEXT)
        if dtype is torch.float8_e4m3fn:
            # Kontext checkpoints ship pre-quantized via bitsandbytes (bnb 4-bit
            # fp4); the global --fluxfp8 cast is meant for raw/unquantized Flux
            # weights and doesn't apply here. Forcing torch_dtype=float8_e4m3fn
            # makes transformers call _set_default_dtype(float8_e4m3fn), which
            # crashes with "couldn't find storage object Float8_e4m3fnStorage"
            # regardless of the checkpoint's actual tensor dtypes (see
            # huggingface/transformers#39409). Fall back to the bf16/fp16
            # compute dtype bnb already expects.
            dtype = self._gguf_compute_dtype(dtype)
        logger.info("Loading Flux Kontext checkpoint %s from %s", checkpoint.title, path)
        try:
            pipe = FluxKontextPipeline.from_pretrained(
                str(path),
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception as exc:
            logger.error(
                "Failed to load Flux Kontext checkpoint '%s': %s: %s",
                checkpoint.title,
                type(exc).__name__,
                exc,
            )
            raise ModelNotFoundError(
                f"'{checkpoint.title}' failed to load ({type(exc).__name__}: {exc}). "
                "This is a FluxKontextPipeline export - check that all components "
                "(transformer/text_encoder/text_encoder_2/vae) downloaded completely."
            ) from exc

        self._remember_base_scheduler_config(pipe)
        apply_attention_optimizations(
            pipe,
            self.flags,
            compile_allowed=self._compile_allowed_for_architecture(ARCH_FLUX_KONTEXT),
        )
        pipe = self._place_pipeline(
            pipe,
            prefer_offload=self._wants_offload(ARCH_FLUX_KONTEXT),
            architecture=ARCH_FLUX_KONTEXT,
        )
        pipe.set_progress_bar_config(disable=True)
        self._tune_vae_memory(pipe, ARCH_FLUX_KONTEXT)
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        self._active = checkpoint
        self._txt2img = pipe
        self._img2img = None
        return checkpoint

    def load_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        checkpoint = self._resolve_checkpoint(checkpoint_id)
        if is_flux_kontext_architecture(checkpoint.architecture):
            return self._load_flux_kontext_checkpoint(checkpoint)
        if is_flux_architecture(checkpoint.architecture):
            return self._load_flux_checkpoint(checkpoint)
        if is_flux2_klein_architecture(checkpoint.architecture):
            return self._load_flux2_klein_checkpoint(checkpoint)
        if is_z_image_architecture(checkpoint.architecture):
            return self._load_z_image_checkpoint(checkpoint)
        if is_qwen_nunchaku_architecture(checkpoint.architecture):
            return self._load_qwen_nunchaku_checkpoint(checkpoint)
        if is_qwen_image_architecture(checkpoint.architecture):
            return self._load_qwen_image_checkpoint(checkpoint)
        if is_sana_architecture(checkpoint.architecture):
            return self._load_sana_checkpoint(checkpoint)
        if self._txt2img is not None and self._active and self._active.path == checkpoint.path:
            logger.debug("Checkpoint already warm: %s", checkpoint.title)
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            self.unload()
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        if self._txt2img is not None:
            self._active = checkpoint
            logger.debug("Reusing warm pipeline for checkpoint alias: %s", checkpoint.title)
            return checkpoint

        logger.info("Loading checkpoint %s (%s)", checkpoint.title, checkpoint.architecture)

        dtype = self._dtype_for_architecture(checkpoint.architecture)
        path = Path(checkpoint.path)
        # An inpaint checkpoint has a 9-channel UNet conv_in; loading it through the
        # 4-channel txt2img pipeline raises a cryptic
        # "conv_in.weight expected [320,4,3,3], got [320,9,3,3]" ValueError. Detection
        # already happened at scan time — branch on it and give an actionable message
        # instead. (Inpaint checkpoints load via _load_inpaint_checkpoint in Inpaint mode.)
        if is_inpaint_architecture(checkpoint.architecture):
            raise ModelNotFoundError(
                f"'{checkpoint.title}' is an inpaint checkpoint (9-channel UNet) and can't be "
                "loaded for txt2img/img2img. Switch to Inpaint mode, or pick a standard checkpoint."
            )
        pipeline_cls = self._txt2img_pipeline_cls_for_architecture(checkpoint.architecture)
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": path.suffix.lower() == ".safetensors",
        }
        if path.is_dir():
            load_kwargs.pop("use_safetensors", None)
        else:
            _add_cached_single_file_config(load_kwargs, pipeline_cls)
        if pipeline_cls is StableDiffusionPipeline:
            load_kwargs["requires_safety_checker"] = False
        # AIWF loads local single-file checkpoints with app-level controls around
        # requests and file selection. Diffusers' optional safety checker needs
        # extra model assets and is disabled here so backend selection stays
        # deterministic/offline.
        if path.is_dir() and not (path / "model_index.json").exists():
            # A directory checkpoint without model_index.json isn't a full pipeline -
            # it's almost always a bare component folder (a ControlNet/VAE/text-encoder
            # export that ended up under the checkpoints scan path). Loading it through
            # a full pipeline class raises a cryptic
            # "OSError: ... does not appear to have a file named model_index.json" deep
            # inside from_pretrained. Fail clearly instead.
            raise ModelNotFoundError(
                f"'{checkpoint.title}' doesn't look like a full model pipeline (no "
                "model_index.json) - it's likely a ControlNet, VAE, or other component "
                "export. If it's a ControlNet model, move it under models/ControlNet "
                "and pick it from the ControlNet panel instead."
            )
        try:
            if path.is_dir():
                pipe = pipeline_cls.from_pretrained(checkpoint.path, **load_kwargs)
            else:
                pipe = pipeline_cls.from_single_file(checkpoint.path, **load_kwargs)
        except ModelNotFoundError:
            raise
        except Exception as exc:
            # Surface a clear, actionable error instead of letting a raw KeyError/
            # RuntimeError/NotImplementedError from deep inside diffusers' converter
            # propagate as an unhandled thread crash. Most commonly this means the
            # checkpoint uses a quantization/key layout (e.g. a non-bnb FP8 export,
            # or a state dict with renamed/missing keys) that the standard single-file
            # converter for this architecture doesn't recognize.
            logger.error(
                "Failed to load '%s' (%s): %s: %s",
                checkpoint.title,
                checkpoint.architecture,
                type(exc).__name__,
                exc,
            )
            raise ModelNotFoundError(
                f"'{checkpoint.title}' failed to load ({type(exc).__name__}: {exc}). "
                "This usually means the file uses a quantization or key layout that "
                "isn't supported for this architecture yet (e.g. a non-standard FP8 "
                "export). Try a different checkpoint, or report this one with its name."
            ) from exc
        self._remember_base_scheduler_config(pipe)
        self._apply_fp8_storage(pipe)

        compile_allowed = self._compile_allowed_for_architecture(checkpoint.architecture)
        apply_attention_optimizations(pipe, self.flags, compile_allowed=compile_allowed)
        pipe = self._place_pipeline(
            pipe,
            prefer_offload=self._wants_offload(checkpoint.architecture),
            architecture=checkpoint.architecture,
        )
        self._tune_vae_memory(pipe, checkpoint.architecture)
        # Embeddings are now loaded on-demand in generate() only for those referenced
        # by the actual prompt (see _ensure_embeddings_for_prompt). Unused ones are
        # never loaded even if present in the embeddings/ folder.
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        self._active = checkpoint
        self._txt2img = pipe
        self._sync_img2img_from_txt2img()
        return checkpoint

    def _load_inpaint_checkpoint(
        self, checkpoint: Checkpoint
    ) -> StableDiffusionInpaintPipeline | StableDiffusionXLInpaintPipeline | StableDiffusion3InpaintPipeline:
        if self._inpaint and self._inpaint_active and self._inpaint_active.path == checkpoint.path:
            return self._inpaint

        if self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None

        if is_flux_architecture(checkpoint.architecture):
            # Flux inpaint reuses the loaded txt2img pipeline's components via
            # from_pipe (no dedicated inpaint UNet exists for base Flux), so it
            # must not fall through to the generic SD/SDXL/SD3.5 loading below.
            return self._load_flux_inpaint_pipeline(checkpoint)

        if self._active and self._active.path != checkpoint.path:
            self._txt2img = None
            self._img2img = None
            self._active = None
            self._active_vae_id = None
            self.devices.empty_cache()

        logger.info("Loading inpaint pipeline for %s (%s)", checkpoint.title, checkpoint.architecture)
        dtype = self._dtype_for_architecture(checkpoint.architecture)
        path = Path(checkpoint.path)
        if is_sd3_architecture(checkpoint.architecture):
            pipeline_cls = StableDiffusion3InpaintPipeline
        elif checkpoint.architecture == ARCH_SDXL_INPAINT:
            pipeline_cls = StableDiffusionXLInpaintPipeline
        elif checkpoint.architecture == ARCH_SDXL:
            logger.warning(
                "Checkpoint %s is SDXL base, not an inpaint model — using XL inpaint pipeline anyway.",
                checkpoint.title,
            )
            pipeline_cls = StableDiffusionXLInpaintPipeline
        else:
            pipeline_cls = StableDiffusionInpaintPipeline

        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": path.suffix.lower() == ".safetensors",
        }
        if path.is_dir():
            load_kwargs.pop("use_safetensors", None)
        else:
            _add_cached_single_file_config(load_kwargs, pipeline_cls)
        if pipeline_cls is StableDiffusionInpaintPipeline:
            load_kwargs["requires_safety_checker"] = False
        if (
            pipeline_cls is StableDiffusion3InpaintPipeline
            and self._txt2img is not None
            and self._active
            and self._active.path == checkpoint.path
        ):
            pipe = StableDiffusion3InpaintPipeline.from_pipe(self._txt2img)
        elif path.is_dir():
            pipe = pipeline_cls.from_pretrained(checkpoint.path, **load_kwargs)
        else:
            pipe = pipeline_cls.from_single_file(checkpoint.path, **load_kwargs)
        self._remember_base_scheduler_config(pipe)
        self._apply_fp8_storage(pipe)

        compile_allowed = self._compile_allowed_for_architecture(checkpoint.architecture)
        apply_attention_optimizations(pipe, self.flags, compile_allowed=compile_allowed)
        pipe = self._place_pipeline(
            pipe,
            prefer_offload=self._wants_offload(checkpoint.architecture),
            architecture=checkpoint.architecture,
        )
        self._tune_vae_memory(pipe, checkpoint.architecture)
        # Embeddings are now loaded on-demand in generate() only for those referenced
        # by the actual prompt (see _ensure_embeddings_for_prompt). Unused ones are
        # never loaded even if present in the embeddings/ folder.
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        self._inpaint = pipe
        self._inpaint_active = checkpoint
        return pipe

    def _load_flux_inpaint_pipeline(self, checkpoint: Checkpoint) -> FluxInpaintPipeline:
        """Wrap the loaded Flux txt2img pipeline's components into a
        FluxInpaintPipeline via from_pipe — no extra weights are loaded since
        base Flux has no dedicated inpaint transformer (that's Flux.1-Fill,
        a different checkpoint). Quality/behavior is strength-based blending,
        like SD img2img-style inpaint, not a purpose-built inpaint model."""
        self.load_checkpoint(checkpoint.id)
        assert self._txt2img is not None
        pipe = FluxInpaintPipeline.from_pipe(self._txt2img)
        # from_pipe only copies registered modules/config, not the custom
        # attributes _run_flux_txt2img_pass (mirrored by the inpaint pass)
        # relies on for fast prompt encoding and guidance-embed detection.
        pipe._aiwf_flux_components = getattr(self._txt2img, "_aiwf_flux_components", None)
        pipe._aiwf_flux_guidance_embeds = getattr(self._txt2img, "_aiwf_flux_guidance_embeds", False)
        pipe._aiwf_cast_vae_decode_dtype = getattr(self._txt2img, "_aiwf_cast_vae_decode_dtype", False)
        self._inpaint = pipe
        self._inpaint_active = checkpoint
        return pipe

    def _load_refiner_checkpoint(self, checkpoint_id: str | None) -> StableDiffusionXLImg2ImgPipeline:
        if not checkpoint_id:
            raise ValueError("SDXL refiner is enabled but no refiner checkpoint is selected.")
        checkpoint = self._resolve_checkpoint(checkpoint_id)
        if not is_sdxl_architecture(checkpoint.architecture):
            raise ValueError("SDXL refiner checkpoint must be an SDXL checkpoint.")
        if self._refiner is not None and self._refiner_active and self._refiner_active.path == checkpoint.path:
            return self._refiner

        logger.info("Loading SDXL refiner pipeline for %s", checkpoint.title)
        dtype = self._dtype_for_architecture(checkpoint.architecture)
        path = Path(checkpoint.path)
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": path.suffix.lower() == ".safetensors",
        }
        _add_cached_single_file_config(load_kwargs, StableDiffusionXLImg2ImgPipeline)
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(checkpoint.path, **load_kwargs)
        self._remember_base_scheduler_config(pipe)
        self._apply_fp8_storage(pipe)
        compile_allowed = self._compile_allowed_for_architecture(checkpoint.architecture)
        apply_attention_optimizations(pipe, self.flags, compile_allowed=compile_allowed)
        pipe = self._place_pipeline(
            pipe,
            prefer_offload=self._wants_offload(checkpoint.architecture),
            architecture=checkpoint.architecture,
        )
        self._tune_vae_memory(pipe, checkpoint.architecture)
        self._refiner = pipe
        self._refiner_active = checkpoint
        return pipe

    @contextmanager
    def _vae_decode_dtype_context(self, pipe):
        vae = getattr(pipe, "vae", None)
        if vae is None or not getattr(pipe, "_aiwf_cast_vae_decode_dtype", False):
            yield
            return
        original_decode = vae.decode

        def decode_with_matching_dtype(latents, *args, **decode_kwargs):
            target_dtype = getattr(vae, "dtype", None)
            target_device = getattr(vae, "device", None)
            if isinstance(latents, torch.Tensor) and target_dtype is not None and latents.dtype != target_dtype:
                latents = latents.to(device=target_device or latents.device, dtype=target_dtype)
            return original_decode(latents, *args, **decode_kwargs)

        vae.decode = decode_with_matching_dtype
        try:
            yield
        finally:
            vae.decode = original_decode

    def _call_pipe(self, pipe, **kwargs):
        with attention_call_context(getattr(self, "flags", None)):
            with self._vae_decode_dtype_context(pipe):
                return pipe(**kwargs)

    @staticmethod
    def _hr_resample_filter(upscaler: str) -> int:
        normalized = (upscaler or "lanczos").strip().lower().replace(" ", "_")
        if normalized == "bicubic":
            return Image.Resampling.BICUBIC
        if normalized == "nearest":
            return Image.Resampling.NEAREST
        return Image.Resampling.LANCZOS

    def unload(self, *, keep_flux_encoders: bool = False) -> None:
        self._txt2img = None
        self._img2img = None
        self._inpaint = None
        self._refiner = None
        self._active = None
        self._inpaint_active = None
        self._refiner_active = None
        self._active_vae_id = None
        self._controlnet_cache.clear()
        # The Flux CLIP-L + T5-XXL text encoders are identical across every Flux
        # checkpoint, and T5 is ~9.8 GB to read from disk. When switching between
        # two Flux transformers we keep them resident (T5 lives on CPU, CLIP is
        # tiny) so the next generation skips a multi-second disk reload.
        if not keep_flux_encoders:
            self._flux_text_encoder = None
            self._flux_text_encoder_2 = None
            self._flux_tokenizer = None
            self._flux_tokenizer_2 = None
            self._flux_component_paths = {}
            self._flux_clip_device = None
            self._flux_t5_device = None
            self._flux_prompt_cache.clear()
        self._flux2_prompt_cache.clear()
        self._z_image_prompt_cache.clear()
        gc.collect()
        self.devices.empty_cache()
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            logger.debug("CUDA cache cleanup after image unload failed", exc_info=True)

    _SAMPLER_EXTRA_KWARGS = {
        "dpmpp_2m_sde": {"algorithm_type": "sde-dpmsolver++"},
        "dpmpp_3m_sde": {"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
        "dpmpp_2m_karras": {"use_karras_sigmas": True},
    }

    def _apply_sampler(self, pipe, sampler_id: str, schedule_type: str = "automatic") -> None:
        scheduler_signature = f"{sampler_id}|{schedule_type}"
        if getattr(pipe, "_aiwf_scheduler_signature", None) == scheduler_signature:
            return
        if self._is_flux_pipe(pipe):
            logger.info("Flux uses its Diffusers FlowMatch scheduler; Studio sampler selection is ignored.")
            pipe._aiwf_scheduler_signature = scheduler_signature
            return
        if self._is_flux_kontext_pipe(pipe):
            logger.info("Flux Kontext uses its Diffusers FlowMatch scheduler; Studio sampler selection is ignored.")
            pipe._aiwf_scheduler_signature = scheduler_signature
            return
        if self._is_flux2_pipe(pipe):
            logger.info("Flux.2 Klein uses its Diffusers FlowMatch scheduler; Studio sampler selection is ignored.")
            pipe._aiwf_scheduler_signature = scheduler_signature
            return
        if self._is_z_image_pipe(pipe):
            logger.info("Z-Image uses its Diffusers FlowMatch scheduler; Studio sampler selection is ignored.")
            pipe._aiwf_scheduler_signature = scheduler_signature
            return
        if self._is_qwen_image_pipe(pipe):
            logger.info("Qwen Image uses its Diffusers scheduler; Studio sampler selection is ignored.")
            pipe._aiwf_scheduler_signature = scheduler_signature
            return
        if self._is_sana_pipe(pipe):
            logger.info("Sana uses its Diffusers scheduler; Studio sampler selection is ignored.")
            pipe._aiwf_scheduler_signature = scheduler_signature
            return
        if self._is_sd3_pipe(pipe):
            logger.info("SD3.5 uses its Diffusers FlowMatch scheduler; Studio sampler selection is ignored.")
            pipe._aiwf_scheduler_signature = scheduler_signature
            return
        cls = SAMPLER_CLASSES.get(sampler_id, EulerAncestralDiscreteScheduler)
        base_config = self._base_scheduler_config_for_pipe(pipe)
        kwargs = dict(self._SAMPLER_EXTRA_KWARGS.get(sampler_id, {}))
        # Sigma schedule must be passed at construction — the scheduler's
        # config is frozen, so mutating it afterwards has no effect.
        if schedule_type == "karras":
            kwargs.update(use_karras_sigmas=True, use_exponential_sigmas=False, use_beta_sigmas=False)
        elif schedule_type == "exponential":
            kwargs.update(use_karras_sigmas=False, use_exponential_sigmas=True, use_beta_sigmas=False)
        elif schedule_type == "beta":
            kwargs.update(use_karras_sigmas=False, use_exponential_sigmas=False, use_beta_sigmas=True)
        elif schedule_type == "sgm_uniform":
            kwargs.update(
                use_karras_sigmas=False,
                use_exponential_sigmas=False,
                use_beta_sigmas=False,
                timestep_spacing="trailing",
            )
        elif schedule_type == "uniform":
            kwargs.update(use_karras_sigmas=False, use_exponential_sigmas=False, use_beta_sigmas=False)
        # "automatic" keeps the sampler's own default schedule.
        try:
            scheduler = cls.from_config(base_config, **kwargs)
        except TypeError:
            # Scheduler doesn't accept sigma-schedule kwargs (e.g. DDIM).
            scheduler = cls.from_config(base_config)
        except ImportError as exc:
            # e.g. DPM++ SDE needs the torchsde package.
            logger.warning("Sampler %s unavailable (%s); keeping current scheduler.", sampler_id, exc)
            return
        except Exception as exc:
            logger.warning(
                "Sampler/schedule combo %s/%s unavailable (%s); falling back to sampler defaults.",
                sampler_id,
                schedule_type,
                exc,
            )
            try:
                scheduler = cls.from_config(base_config)
            except ImportError as import_exc:
                logger.warning("Sampler %s unavailable (%s); keeping current scheduler.", sampler_id, import_exc)
                return
            except Exception as fallback_exc:
                logger.warning(
                    "Sampler %s could not be restored with default settings (%s); keeping current scheduler.",
                    sampler_id,
                    fallback_exc,
                )
                return
        pipe.scheduler = scheduler
        pipe._aiwf_scheduler_signature = scheduler_signature

    def _decode_latent_preview(self, pipe, latents) -> Image.Image:
        latents = latents.to(dtype=pipe.vae.dtype, device=pipe.vae.device)
        latents = latents / pipe.vae.config.scaling_factor
        with torch.no_grad():
            decoded = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    def _make_callback(
        self,
        pipe,
        request: GenerationRequest,
        on_progress: Callable[[int, int, str, Image.Image | None], None] | None,
        should_cancel: Callable[[], bool] | None,
        *,
        step_offset: int = 0,
        total_steps: int | None = None,
        preview_every_n_steps: int = 0,
    ):
        total = total_steps or request.steps

        def callback(pipe_obj, step_index, _timestep, callback_kwargs):
            if should_cancel and should_cancel():
                setattr(pipe_obj, "_interrupt", True)

            preview: Image.Image | None = None
            if (
                on_progress
                and preview_every_n_steps > 0
                and (step_index + 1) % preview_every_n_steps == 0
                and "latents" in callback_kwargs
            ):
                try:
                    preview = self._decode_latent_preview(pipe, callback_kwargs["latents"])
                except Exception:
                    logger.debug("Live preview decode failed at step %s", step_index, exc_info=True)

            if on_progress:
                current = step_offset + step_index + 1
                on_progress(current, total, f"Step {current}/{total}", preview)

            if should_cancel and should_cancel():
                raise GenerationCancelledError()
            return callback_kwargs

        return callback

    def _run_txt2img_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        *,
        width: int,
        height: int,
        steps: int,
    ):
        prompt_kwargs = build_prompt_kwargs(
            pipe,
            parsed_prompt,
            request.negative_prompt,
            request.clip_skip,
        )
        return self._call_pipe(
            pipe,
            **prompt_kwargs,
            num_inference_steps=steps,
            guidance_scale=request.cfg_scale,
            num_images_per_prompt=request.batch_size,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            width=width,
            height=height,
        )

    def _encode_with_progress(
        self,
        family: str,
        encode_fn: Callable[[], object],
        *,
        steps: int,
        on_progress: Callable[[int, int, str, Image.Image | None], None] | None,
        tick_seconds: float = 0.4,
    ):
        """Run a blocking prompt-encode call while emitting a live elapsed/ETA
        progress message, so "Encoding prompt" doesn't sit frozen at 0% for
        models with a slow text encoder (e.g. Flux's T5). There's no real
        sub-step granularity inside a single encode forward pass, so the ETA
        is estimated from this family's rolling average of past encode runs
        on this machine - it gets more accurate after the first generation."""
        total = max(1, int(steps))
        if on_progress is None:
            return encode_fn()

        known_avg = self._encode_duration_estimates.get(family)
        start = time.perf_counter()
        stop_event = threading.Event()

        def _tick() -> None:
            while not stop_event.wait(tick_seconds):
                elapsed = time.perf_counter() - start
                if known_avg:
                    remaining = max(0.0, known_avg - elapsed)
                    message = f"Encoding prompt… {elapsed:.1f}s (~{remaining:.1f}s left)"
                else:
                    message = f"Encoding prompt… {elapsed:.1f}s"
                try:
                    on_progress(0, total, message, None)
                except Exception:
                    logger.debug("Encode progress callback failed", exc_info=True)

        on_progress(0, total, "Encoding prompt…", None)
        ticker = threading.Thread(target=_tick, name=f"encode-eta-{family}", daemon=True)
        ticker.start()
        try:
            result = encode_fn()
        finally:
            stop_event.set()
            ticker.join(timeout=1.0)

        elapsed_total = time.perf_counter() - start
        if known_avg is None:
            self._encode_duration_estimates[family] = elapsed_total
        else:
            self._encode_duration_estimates[family] = (known_avg * 0.7) + (elapsed_total * 0.3)
        logger.info("Prompt encode (%s) took %.2fs", family, elapsed_total)
        return result

    def _run_flux_txt2img_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        *,
        width: int,
        height: int,
        steps: int,
        on_progress: Callable[[int, int, str, Image.Image | None], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ):
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        component_paths = getattr(pipe, "_aiwf_flux_components", None) or self._resolve_flux_component_paths()
        self._load_flux_prompt_models({key: Path(value) for key, value in component_paths.items()})
        device = self._execution_device(pipe)
        prompt_embeds, pooled_prompt_embeds = self._encode_with_progress(
            "flux",
            lambda: self._encode_flux_prompt_fast(
                pipe,
                parsed_prompt,
                device=device,
                batch_size=request.batch_size,
            ),
            steps=steps,
            on_progress=on_progress,
        )
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        guidance_embeds = bool(getattr(pipe, "_aiwf_flux_guidance_embeds", False))
        guidance_scale = float(request.cfg_scale) if guidance_embeds else 0.0
        if not guidance_embeds and request.cfg_scale != 0:
            logger.info("Flux distilled transformer has no guidance block; CFG/guidance is forced to 0.0.")
        return self._call_pipe(
            pipe,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            width=width,
            height=height,
            max_sequence_length=256,
            output_type="pil",
        )

    def _run_flux_inpaint_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        *,
        image: Image.Image,
        mask_image: Image.Image,
        width: int,
        height: int,
        steps: int,
        strength: float,
        on_progress: Callable[[int, int, str, Image.Image | None], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ):
        """Flux inpaint via FluxInpaintPipeline: strength-based blending against
        the masked region, same prompt encoding path as Flux txt2img. There's
        no dedicated inpaint transformer for base Flux (that's Flux.1-Fill, a
        different checkpoint) so quality is closer to SD img2img-style inpaint
        than a purpose-built inpaint model — still genuinely useful, just don't
        expect Fill-model-grade seams on tricky masks."""
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        component_paths = getattr(pipe, "_aiwf_flux_components", None) or self._resolve_flux_component_paths()
        self._load_flux_prompt_models({key: Path(value) for key, value in component_paths.items()})
        device = self._execution_device(pipe)
        prompt_embeds, pooled_prompt_embeds = self._encode_with_progress(
            "flux",
            lambda: self._encode_flux_prompt_fast(
                pipe,
                parsed_prompt,
                device=device,
                batch_size=request.batch_size,
            ),
            steps=steps,
            on_progress=on_progress,
        )
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        guidance_embeds = bool(getattr(pipe, "_aiwf_flux_guidance_embeds", False))
        guidance_scale = float(request.cfg_scale) if guidance_embeds else 0.0
        if not guidance_embeds and request.cfg_scale != 0:
            logger.info("Flux distilled transformer has no guidance block; CFG/guidance is forced to 0.0.")
        return self._call_pipe(
            pipe,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            width=width,
            height=height,
            max_sequence_length=256,
            output_type="pil",
        )

    def _run_flux2_klein_txt2img_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        *,
        width: int,
        height: int,
        steps: int,
        on_progress: Callable[[int, int, str, Image.Image | None], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ):
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        device = self._execution_device(pipe)
        prompt_embeds = self._encode_with_progress(
            "flux2_klein",
            lambda: self._encode_flux2_prompt(pipe, parsed_prompt, device),
            steps=steps,
            on_progress=on_progress,
        )
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        text_encoder = getattr(pipe, "text_encoder", None)
        try:
            pipe.text_encoder = None
            return self._call_pipe(
                pipe,
                prompt_embeds=prompt_embeds,
                num_inference_steps=steps,
                guidance_scale=float(request.cfg_scale),
                num_images_per_prompt=request.batch_size,
                generator=generator,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
                width=width,
                height=height,
                output_type="pil",
            )
        finally:
            pipe.text_encoder = text_encoder

    def _run_z_image_txt2img_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        *,
        width: int,
        height: int,
        steps: int,
        on_progress: Callable[[int, int, str, Image.Image | None], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ):
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        device = self._execution_device(pipe)
        prompt_embeds, negative_prompt_embeds = self._encode_with_progress(
            "z_image",
            lambda: self._encode_z_image_prompts(
                pipe,
                parsed_prompt,
                request.negative_prompt or None,
                device,
            ),
            steps=steps,
            on_progress=on_progress,
        )
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        text_encoder = getattr(pipe, "text_encoder", None)
        try:
            pipe.text_encoder = None
            return self._call_pipe(
                pipe,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=steps,
                guidance_scale=float(request.cfg_scale),
                num_images_per_prompt=request.batch_size,
                generator=generator,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
                width=width,
                height=height,
                output_type="pil",
            )
        finally:
            pipe.text_encoder = text_encoder

    def _generate_qwen_nunchaku_result(
        self,
        checkpoint: Checkpoint,
        request: GenerationRequest,
        parsed_prompt: str,
        on_progress,
        should_cancel,
    ) -> GenerationResult:
        job_id = uuid4()
        images: list[Image.Image] = []
        seeds: list[int] = []
        infotexts: list[str] = []

        width, height = align_to_multiple_of_16(request.width, request.height)
        if (width, height) != (request.width, request.height):
            logger.info(
                "Rounding %dx%d to %dx%d for Qwen Nunchaku (multiple of 16 required)",
                request.width,
                request.height,
                width,
                height,
            )

        total_batches = max(1, int(request.batch_count))
        total_images = max(1, int(request.batch_size))
        for batch_index in range(total_batches):
            batch_images: list[Image.Image] = []
            batch_seeds: list[int] = []
            for image_index in range(total_images):
                if should_cancel and should_cancel():
                    raise GenerationCancelledError()
                seed = self._qwen_nunchaku.suggested_seed(request, batch_index=batch_index, image_index=image_index)
                image, _output_path = self._qwen_nunchaku.generate(
                    checkpoint,
                    request,
                    prompt=parsed_prompt,
                    width=width,
                    height=height,
                    steps=int(request.steps),
                    seed=seed,
                    should_cancel=should_cancel,
                )
                images.append(image)
                seeds.append(seed)
                batch_images.append(image)
                batch_seeds.append(seed)
                infotexts.append(
                    format_infotext(
                        request,
                        seed=seed,
                        checkpoint=checkpoint,
                        output_width=width,
                        output_height=height,
                    )
                )
            if on_progress and batch_images:
                on_progress(
                    batch_index + 1,
                    total_batches,
                    f"Batch {batch_index + 1}/{request.batch_count} complete",
                    batch_images[-1],
                    batch_images,
                    batch_seeds,
                )

        return GenerationResult(
            job_id=job_id,
            images=images,
            seeds=seeds,
            infotexts=infotexts,
            before_hires_images=[],
            mode=request.mode,
        )

    def _run_qwen_image_txt2img_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        *,
        width: int,
        height: int,
        steps: int,
        should_cancel: Callable[[], bool] | None = None,
    ):
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        return self._call_pipe(
            pipe,
            prompt=parsed_prompt,
            negative_prompt=request.negative_prompt or None,
            true_cfg_scale=float(request.cfg_scale),
            num_inference_steps=steps,
            num_images_per_prompt=request.batch_size,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            width=width,
            height=height,
            max_sequence_length=512,
            output_type="pil",
        )

    def _run_sana_txt2img_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        *,
        width: int,
        height: int,
        steps: int,
        should_cancel: Callable[[], bool] | None = None,
    ):
        if should_cancel and should_cancel():
            raise GenerationCancelledError()
        common = dict(
            prompt=parsed_prompt,
            num_inference_steps=steps,
            guidance_scale=float(request.cfg_scale),
            num_images_per_prompt=request.batch_size,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            width=width,
            height=height,
            output_type="pil",
            clean_caption=False,
            use_resolution_binning=True,
        )
        if pipe.__class__.__name__ == "SanaSprintPipeline":
            if steps != 2:
                common["intermediate_timesteps"] = None
        else:
            common["negative_prompt"] = request.negative_prompt or ""
        return self._call_pipe(pipe, **common)

    def _run_img2img_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        image: Image.Image,
        *,
        steps: int,
        strength: float,
    ):
        prompt_kwargs = build_prompt_kwargs(
            pipe,
            parsed_prompt,
            request.negative_prompt,
            request.clip_skip,
        )
        return self._call_pipe(
            pipe,
            **prompt_kwargs,
            num_inference_steps=steps,
            guidance_scale=request.cfg_scale,
            num_images_per_prompt=request.batch_size,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            image=image.convert("RGB"),
            strength=strength,
        )

    def _run_controlnet_pass(
        self,
        pipe,
        request: GenerationRequest,
        parsed_prompt: str,
        generator,
        callback,
        units: list[ControlNetUnit],
        control_images: list[Image.Image],
        init_images: list[Image.Image] | None,
        mask_images: list[Image.Image] | None = None,
    ):
        if not units or not control_images:
            raise ValueError("ControlNet requires at least one active unit and control image.")
        multi = len(units) > 1
        scales = [float(unit.weight) for unit in units]
        starts = [float(unit.guidance_start) for unit in units]
        ends = [float(unit.guidance_end) for unit in units]
        prompt_kwargs = build_prompt_kwargs(
            pipe,
            parsed_prompt,
            request.negative_prompt,
            request.clip_skip,
        )
        common = dict(
            **prompt_kwargs,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            num_images_per_prompt=request.batch_size,
            generator=generator,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            controlnet_conditioning_scale=scales if multi else scales[0],
            control_guidance_start=starts if multi else starts[0],
            control_guidance_end=ends if multi else ends[0],
        )
        if request.mode == GenerationMode.INPAINT:
            assert init_images is not None and mask_images is not None
            orig = init_images[0].convert("RGB")
            raw_mask = mask_images[0]
            only_masked = bool(getattr(request, "inpaint_only_masked", False))
            pad = int(getattr(request, "inpaint_masked_padding", 32))
            content = getattr(request, "inpaint_mask_content", "original") or "original"

            if only_masked:
                src, msk, crop_box = crop_to_masked(orig, raw_mask, padding=pad)
                src = apply_masked_content(src, msk, content)
                if request.mask_blur > 0:
                    msk = blur_mask(msk, request.mask_blur)
                pipeline_src, pipeline_msk, width, height = resize_for_inpaint(src, msk)
                controls = [
                    image.crop(crop_box).resize((width, height), Image.Resampling.LANCZOS)
                    for image in control_images
                ]
                output = self._call_pipe(
                    pipe,
                    **common,
                    image=pipeline_src,
                    mask_image=pipeline_msk,
                    control_image=controls if multi else controls[0],
                    strength=request.denoising_strength,
                    width=width,
                    height=height,
                )
                full_mask = prepare_inpaint_mask(raw_mask, size=orig.size)
                pw = crop_box[2] - crop_box[0]
                ph = crop_box[3] - crop_box[1]
                seam_erode = int(getattr(request, "seam_erode", 0) or 0)
                composites = []
                for gen in output.images:
                    full_gen = orig.copy()
                    gen_r = gen.resize((pw, ph), Image.Resampling.LANCZOS) if gen.size != (pw, ph) else gen
                    full_gen.paste(gen_r, (crop_box[0], crop_box[1]))
                    composites.append(
                        composite_inpaint_result(
                            full_gen,
                            orig,
                            full_mask,
                            mask_blur=request.mask_blur,
                            seam_erode=seam_erode,
                        )
                    )
                output.images = composites
                return output, orig.width, orig.height

            source, mask, width, height = resize_for_inpaint(orig, raw_mask)
            source = apply_masked_content(source, mask, content)
            if request.mask_blur > 0:
                mask = blur_mask(mask, request.mask_blur)
            controls = [
                image.resize((width, height), Image.Resampling.LANCZOS)
                for image in control_images
            ]
            output = self._call_pipe(
                pipe,
                **common,
                image=source,
                mask_image=mask,
                control_image=controls if multi else controls[0],
                strength=request.denoising_strength,
                width=width,
                height=height,
            )
            seam_erode = int(getattr(request, "seam_erode", 0) or 0)
            paste_mask = prepare_inpaint_mask(raw_mask, size=orig.size)
            output.images = [
                composite_inpaint_result(
                    img if img.size == orig.size else img.resize(orig.size, Image.Resampling.LANCZOS),
                    orig,
                    paste_mask,
                    mask_blur=request.mask_blur,
                    seam_erode=seam_erode,
                )
                for img in output.images
            ]
            return output, orig.width, orig.height

        if request.mode == GenerationMode.IMG2IMG:
            assert init_images is not None
            base = init_images[0].convert("RGB")
            width, height = align_to_multiple_of_8(base.width, base.height)
            if base.size != (width, height):
                base = base.resize((width, height), Image.Resampling.LANCZOS)
            controls = [
                image.resize((width, height), Image.Resampling.LANCZOS)
                for image in control_images
            ]
            output = self._call_pipe(
                pipe,
                **common,
                image=base,
                control_image=controls if multi else controls[0],
                strength=request.denoising_strength,
            )
            return output, width, height

        width, height = align_to_multiple_of_8(request.width, request.height)
        controls = [
            image.resize((width, height), Image.Resampling.LANCZOS)
            for image in control_images
        ]
        output = self._call_pipe(
            pipe,
            **common,
            image=controls if multi else controls[0],
            width=width,
            height=height,
        )
        return output, width, height

    def _controlnet_dir(self) -> Path:
        return self.flags.resolved_models_dir() / "ControlNet"

    def _resolve_controlnet_path(self, model_id: str | None) -> Path | None:
        if not model_id:
            return None
        for path in iter_controlnet_model_paths(self.flags):
            if path.stem == model_id or path.name == model_id:
                return path
        return None

    def _prepare_controlnets(
        self,
        request: GenerationRequest,
        control_images: list[Image.Image] | None,
    ) -> list[tuple[ControlNetUnit, Image.Image, Path]]:
        """Resolve usable ControlNet units into (unit, control_image, path) tuples.

        Returns an empty list when ControlNet is not active or cannot be satisfied.
        Missing optional units are skipped here; generate() turns an explicitly
        enabled-but-empty ControlNet request into a user-facing error.
        """
        if request.mode not in (GenerationMode.TXT2IMG, GenerationMode.IMG2IMG, GenerationMode.INPAINT):
            return []
        supplied = list(control_images or [])
        supplied_index = 0
        prepared: list[tuple[ControlNetUnit, Image.Image, Path]] = []
        for unit in request.controlnet_units or []:
            if not unit.enabled or not unit.model:
                continue
            supplied_control = supplied[supplied_index] if supplied_index < len(supplied) else None
            supplied_index += 1
            path = self._resolve_controlnet_path(unit.model)
            if path is None:
                roots = ", ".join(str(root) for root in resolve_controlnet_roots(self.flags)) or str(self._controlnet_dir())
                logger.warning("ControlNet model %s not found in %s", unit.model, roots)
                continue
            control = supplied_control or decode_control_image(unit.image)
            if control is None:
                logger.warning("ControlNet unit %s has no control image; skipping.", unit.model)
                continue
            if unit.module and unit.module not in ("none", "passthrough"):
                control = preprocess_control_image(
                    control,
                    unit.module,
                    PreprocessParams(
                        processor_res=unit.processor_res,
                        threshold_a=unit.threshold_a,
                        threshold_b=unit.threshold_b,
                        annotator_dir=str(self._controlnet_dir() / "Annotators"),
                    ),
                )
            prepared.append((unit, control.convert("RGB"), path))
        return prepared

    def generate(
        self,
        request: GenerationRequest,
        init_images: list[Image.Image] | None = None,
        mask_images: list[Image.Image] | None = None,
        control_images: list[Image.Image] | None = None,
        on_progress: Callable[[int, int, str, Image.Image | None], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        preview_every_n_steps: int = 0,
    ) -> GenerationResult:
        checkpoint = self._resolve_checkpoint(request.checkpoint_id)
        job_id = uuid4()
        images: list[Image.Image] = []
        seeds: list[int] = []
        infotexts: list[str] = []
        before_hires_images: list[Image.Image] = []

        parsed = parse_extra_networks(request.prompt)
        is_flux_checkpoint = is_flux_architecture(checkpoint.architecture)
        is_flux_kontext_checkpoint = is_flux_kontext_architecture(checkpoint.architecture)
        is_flux2_checkpoint = is_flux2_klein_architecture(checkpoint.architecture)
        is_z_image_checkpoint = is_z_image_architecture(checkpoint.architecture)
        is_qwen_nunchaku_checkpoint = is_qwen_nunchaku_architecture(checkpoint.architecture)
        is_qwen_image_checkpoint = is_qwen_image_architecture(checkpoint.architecture) and not is_qwen_nunchaku_checkpoint
        is_sana_checkpoint = is_sana_architecture(checkpoint.architecture)
        is_transformer_image_checkpoint = is_transformer_image_architecture(checkpoint.architecture)
        if is_sd3_architecture(checkpoint.architecture):
            if request.vae_id:
                raise ValueError("External SD1.5/SDXL VAE selection is not supported with SD3.5 checkpoints.")
            if request.sdxl_refiner_enabled:
                raise ValueError("SDXL refiner only works with SDXL checkpoints, not SD3.5.")
            if any(unit.enabled for unit in (request.controlnet_units or [])):
                raise ValueError(
                "The current ControlNet stack is for SD1.5/SDXL. Disable ControlNet for SD3.5."
                )
        if is_transformer_image_checkpoint:
            family_label = (
                "Flux.2 Klein"
                if is_flux2_checkpoint
                else (
                    "Z-Image"
                    if is_z_image_checkpoint
                    else (
                        "Qwen Image Nunchaku"
                        if is_qwen_nunchaku_checkpoint
                        else (
                            "Qwen Image"
                            if is_qwen_image_checkpoint
                            else ("Sana" if is_sana_checkpoint else ("Flux Kontext" if is_flux_kontext_checkpoint else "Flux"))
                        )
                    )
                )
            )
            # Plain Flux (not Flux.2 Klein / Z-Image) also supports inpaint via
            # FluxInpaintPipeline, reusing the already-loaded transformer/text
            # encoders. Everything else on this family stays txt2img-only.
            flux_inpaint_allowed = is_flux_checkpoint and request.mode == GenerationMode.INPAINT
            if request.mode != GenerationMode.TXT2IMG and not flux_inpaint_allowed:
                if is_flux_checkpoint:
                    raise ValueError(
                        "Flux supports txt2img and inpaint only. Use SD/SDXL/SD3.5 for img2img/ControlNet."
                    )
                raise ValueError(
                    f"{family_label} is currently wired for txt2img only. "
                    "Use SD/SDXL/SD3.5 for img2img or inpaint."
                )
            if request.vae_id:
                raise ValueError(f"External SD VAE selection is not supported with {family_label}.")
            if request.enable_hr:
                raise ValueError(
                    f"Hires fix is not wired for {family_label} yet. Generate at the target resolution directly."
                )
            if request.sdxl_refiner_enabled:
                raise ValueError(f"SDXL refiner only works with SDXL checkpoints, not {family_label}.")
            if any(unit.enabled for unit in (request.controlnet_units or [])):
                raise ValueError(
                    f"ControlNet is not wired for {family_label} yet. Disable ControlNet or use an SD/SDXL model."
                )
            if parsed.loras and not _supports_runtime_lora_adapters(checkpoint.architecture):
                raise ValueError(f"{family_label} LoRA application is not wired yet. Remove LoRAs for this pass.")

        if request.mode == GenerationMode.INPAINT:
            if not init_images:
                raise ValueError("inpaint requires init_images")
            if not mask_images:
                raise ValueError("inpaint requires mask_images")
            if (
                not is_inpaint_architecture(checkpoint.architecture)
                and not is_sd3_architecture(checkpoint.architecture)
                and not is_flux_architecture(checkpoint.architecture)
            ):
                logger.warning(
                    "Checkpoint %s is not an inpaint model (%s); results may be poor or fail.",
                    checkpoint.title,
                    checkpoint.architecture,
                )
            pipe = self._load_inpaint_checkpoint(checkpoint)
        elif request.mode == GenerationMode.IMG2IMG:
            if not init_images:
                raise ValueError("img2img requires init_images")
            self.load_checkpoint(request.checkpoint_id)
            pipe = self._img2img
            assert pipe is not None
        elif is_qwen_nunchaku_checkpoint:
            self.load_checkpoint(request.checkpoint_id)
            return self._generate_qwen_nunchaku_result(
                checkpoint,
                request,
                parsed.prompt,
                on_progress,
                should_cancel,
            )
        else:
            self.load_checkpoint(request.checkpoint_id)
            pipe = self._txt2img
            assert pipe is not None

        if not is_transformer_image_checkpoint:
            self._apply_vae(pipe, request.vae_id)
            if request.mode != GenerationMode.INPAINT and self._img2img is not None:
                self._apply_vae(self._img2img, request.vae_id)

        self._apply_sampler(pipe, request.sampler, request.scheduler)
        if self._img2img is not None and request.mode == GenerationMode.TXT2IMG and request.enable_hr:
            self._apply_sampler(self._img2img, request.sampler, request.scheduler)

        adapter_names = []
        if _supports_runtime_lora_adapters(checkpoint.architecture):
            adapter_names = apply_loras(
                pipe,
                parsed.loras,
                self.list_loras(),
                base_architecture=checkpoint.architecture,
            )
            if self._img2img is not None and adapter_names:
                apply_loras(
                    self._img2img,
                    parsed.loras,
                    self.list_loras(),
                    base_architecture=checkpoint.architecture,
                )

        controlnets = self._prepare_controlnets(request, control_images)
        if any(unit.enabled for unit in (request.controlnet_units or [])) and not controlnets:
            raise ValueError(
                "ControlNet was enabled but could not run — check that the model file exists "
                "under models/ControlNet and a control image was provided."
            )
        cn_pipe = None
        cn_units: list[ControlNetUnit] = []
        cn_images: list[Image.Image] = []
        if controlnets:
            cn_units = [unit for unit, _image, _path in controlnets]
            cn_images = [image for _unit, image, _path in controlnets]
            cn_paths = [path for _unit, _image, path in controlnets]
            # Validate every branch before building the combined pipeline so a
            # mixed SD1.5/SDXL stack fails cleanly before any heavy reload work.
            for cn_path in cn_paths:
                assert_controlnet_checkpoint_compatible(cn_path, checkpoint.architecture)
            if request.enable_hr:
                logger.warning("Hires fix is ignored while ControlNet is active.")
            dtype = self.devices.dtype(self.flags.no_half)
            cn_models = [self._controlnet_cache.load(str(path), dtype=dtype) for path in cn_paths]
            cn_model_arg = cn_models if len(cn_models) > 1 else cn_models[0]
            cn_pipe = build_controlnet_pipeline(
                pipe,
                cn_model_arg,
                mode=request.mode.value,
            )
            cn_pipe = self._place_pipeline(
                cn_pipe,
                prefer_offload=self._wants_offload(checkpoint.architecture),
                architecture=checkpoint.architecture,
            )

        # Load textual inversions referenced by the (processed) prompt/negative.
        # Only embeddings the user actually uses in the prompt text are loaded here;
        # the rest stay on disk and are never injected into the text encoders.
        if not is_transformer_image_checkpoint:
            self._ensure_embeddings_for_prompt(
                pipe, request.prompt, request.negative_prompt, checkpoint.architecture
            )
            if self._img2img is not None:
                self._ensure_embeddings_for_prompt(
                    self._img2img, request.prompt, request.negative_prompt, checkpoint.architecture
                )
            if cn_pipe is not None:
                self._ensure_embeddings_for_prompt(
                    cn_pipe, request.prompt, request.negative_prompt, checkpoint.architecture
                )

        try:
            run_pipe = cn_pipe if cn_pipe is not None else pipe
            if is_sana_checkpoint:
                generator = torch.Generator()
            else:
                try:
                    generator = torch.Generator(device=self._execution_device(run_pipe))
                except Exception:
                    # DirectML (and other non-CUDA backends) reject device generators.
                    generator = torch.Generator()
            base_step_count = request.steps + (request.hr_steps if request.enable_hr and cn_pipe is None else 0)
            total_steps = base_step_count
            if request.sdxl_refiner_enabled and request.mode in (GenerationMode.TXT2IMG, GenerationMode.IMG2IMG):
                total_steps += request.sdxl_refiner_steps

            for batch_index in range(request.batch_count):
                seed = request.seed if batch_index == 0 and request.seed >= 0 else random.randint(0, 2**32 - 1)
                generator.manual_seed(seed)
                width, height = request.width, request.height
                if is_transformer_image_checkpoint:
                    # Flux / Flux.2 Klein / Z-Image patchify in 16x16 blocks and
                    # raise "Height/Width must be divisible by 16" otherwise.
                    aligned_width, aligned_height = align_to_multiple_of_16(width, height)
                    if (aligned_width, aligned_height) != (width, height):
                        logger.info(
                            "Rounding %dx%d to %dx%d for transformer architecture (multiple of 16 required)",
                            width,
                            height,
                            aligned_width,
                            aligned_height,
                        )
                        width, height = aligned_width, aligned_height

                if is_flux_checkpoint and request.mode == GenerationMode.TXT2IMG:
                    callback = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output = self._run_flux_txt2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        width=width,
                        height=height,
                        steps=request.steps,
                        on_progress=on_progress,
                        should_cancel=should_cancel,
                    )
                    batch_images = output.images

                elif is_flux_kontext_checkpoint:
                    callback = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output = self._run_flux_kontext_txt2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        width=width,
                        height=height,
                        steps=request.steps,
                        should_cancel=should_cancel,
                    )
                    batch_images = output.images

                elif is_flux2_checkpoint:
                    callback = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output = self._run_flux2_klein_txt2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        width=width,
                        height=height,
                        steps=request.steps,
                        on_progress=on_progress,
                        should_cancel=should_cancel,
                    )
                    batch_images = output.images

                elif is_z_image_checkpoint:
                    callback = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output = self._run_z_image_txt2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        width=width,
                        height=height,
                        steps=request.steps,
                        on_progress=on_progress,
                        should_cancel=should_cancel,
                    )
                    batch_images = output.images

                elif is_qwen_image_checkpoint:
                    callback = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output = self._run_qwen_image_txt2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        width=width,
                        height=height,
                        steps=request.steps,
                        should_cancel=should_cancel,
                    )
                    batch_images = output.images

                elif is_sana_checkpoint:
                    callback = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output = self._run_sana_txt2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        width=width,
                        height=height,
                        steps=request.steps,
                        should_cancel=should_cancel,
                    )
                    batch_images = output.images

                elif is_flux_checkpoint and request.mode == GenerationMode.INPAINT:
                    assert mask_images is not None and init_images is not None
                    orig = init_images[0].convert("RGB")
                    raw_mask = mask_images[0]
                    only_masked = bool(getattr(request, "inpaint_only_masked", False))
                    pad = int(getattr(request, "inpaint_masked_padding", 32))
                    content = getattr(request, "inpaint_mask_content", "original") or "original"

                    if only_masked:
                        src, msk, crop_box = crop_to_masked(orig, raw_mask, padding=pad)
                        src = apply_masked_content(src, msk, content)
                        if request.mask_blur > 0:
                            msk = blur_mask(msk, request.mask_blur)
                        pipeline_src, pipeline_msk, pipe_w, pipe_h = resize_for_inpaint(src, msk)
                        pipe_w, pipe_h = align_to_multiple_of_16(pipe_w, pipe_h)
                        callback = self._make_callback(
                            pipe,
                            request,
                            on_progress,
                            should_cancel,
                            total_steps=request.steps,
                            preview_every_n_steps=preview_every_n_steps,
                        )
                        output = self._run_flux_inpaint_pass(
                            pipe,
                            request,
                            parsed.prompt,
                            generator,
                            callback,
                            image=pipeline_src,
                            mask_image=pipeline_msk,
                            width=pipe_w,
                            height=pipe_h,
                            steps=request.steps,
                            strength=request.denoising_strength,
                            on_progress=on_progress,
                            should_cancel=should_cancel,
                        )
                        full_mask = prepare_inpaint_mask(raw_mask, size=orig.size)
                        pw = crop_box[2] - crop_box[0]
                        ph = crop_box[3] - crop_box[1]
                        seam_erode = int(getattr(request, "seam_erode", 0) or 0)
                        batch_images = []
                        for gen in output.images:
                            full_gen = orig.copy()
                            gen_r = (
                                gen.resize((pw, ph), Image.Resampling.LANCZOS)
                                if gen.size != (pw, ph)
                                else gen
                            )
                            full_gen.paste(gen_r, (crop_box[0], crop_box[1]))
                            comp = composite_inpaint_result(
                                full_gen,
                                orig,
                                full_mask,
                                mask_blur=request.mask_blur,
                                seam_erode=seam_erode,
                            )
                            batch_images.append(comp)
                        width, height = orig.size
                    else:
                        source, mask, width, height = resize_for_inpaint(orig, raw_mask)
                        width, height = align_to_multiple_of_16(width, height)
                        source = apply_masked_content(source, mask, content)
                        if request.mask_blur > 0:
                            mask = blur_mask(mask, request.mask_blur)
                        callback = self._make_callback(
                            pipe,
                            request,
                            on_progress,
                            should_cancel,
                            total_steps=request.steps,
                            preview_every_n_steps=preview_every_n_steps,
                        )
                        output = self._run_flux_inpaint_pass(
                            pipe,
                            request,
                            parsed.prompt,
                            generator,
                            callback,
                            image=source,
                            mask_image=mask,
                            width=width,
                            height=height,
                            steps=request.steps,
                            strength=request.denoising_strength,
                            on_progress=on_progress,
                            should_cancel=should_cancel,
                        )
                        seam_erode = int(getattr(request, "seam_erode", 0) or 0)
                        paste_mask = prepare_inpaint_mask(raw_mask, size=orig.size)
                        batch_images = [
                            composite_inpaint_result(
                                img if img.size == orig.size else img.resize(orig.size, Image.Resampling.LANCZOS),
                                orig,
                                paste_mask,
                                mask_blur=request.mask_blur,
                                seam_erode=seam_erode,
                            )
                            for img in output.images
                        ]
                        width, height = orig.size

                elif cn_pipe is not None:
                    callback = self._make_callback(
                        run_pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output, width, height = self._run_controlnet_pass(
                        run_pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        cn_units,
                        cn_images,
                        init_images,
                        mask_images,
                    )
                    batch_images = output.images

                elif request.mode == GenerationMode.INPAINT:
                    assert mask_images is not None and init_images is not None
                    orig = init_images[0].convert("RGB")
                    raw_mask = mask_images[0]
                    only_masked = bool(getattr(request, "inpaint_only_masked", False))
                    pad = int(getattr(request, "inpaint_masked_padding", 32))
                    content = getattr(request, "inpaint_mask_content", "original") or "original"

                    if only_masked:
                        src, msk, crop_box = crop_to_masked(orig, raw_mask, padding=pad)
                        src = apply_masked_content(src, msk, content)
                        if request.mask_blur > 0:
                            msk = blur_mask(msk, request.mask_blur)
                        pipeline_src, pipeline_msk, pipe_w, pipe_h = resize_for_inpaint(src, msk)
                        callback = self._make_callback(
                            pipe,
                            request,
                            on_progress,
                            should_cancel,
                            total_steps=request.steps,
                            preview_every_n_steps=preview_every_n_steps,
                        )
                        prompt_kwargs = build_prompt_kwargs(
                            pipe,
                            parsed.prompt,
                            request.negative_prompt,
                            request.clip_skip,
                        )
                        output = self._call_pipe(
                            pipe,
                            **prompt_kwargs,
                            num_inference_steps=request.steps,
                            guidance_scale=request.cfg_scale,
                            num_images_per_prompt=request.batch_size,
                            generator=generator,
                            callback_on_step_end=callback,
                            callback_on_step_end_tensor_inputs=["latents"],
                            image=pipeline_src,
                            mask_image=pipeline_msk,
                            strength=request.denoising_strength,
                            width=pipe_w,
                            height=pipe_h,
                        )
                        # Paste generated crop back, then composite with seam control.
                        full_mask = prepare_inpaint_mask(raw_mask, size=orig.size)
                        pw = crop_box[2] - crop_box[0]
                        ph = crop_box[3] - crop_box[1]
                        seam_erode = int(getattr(request, "seam_erode", 0) or 0)
                        batch_images = []
                        for gen in output.images:
                            full_gen = orig.copy()
                            gen_r = (
                                gen.resize((pw, ph), Image.Resampling.LANCZOS)
                                if gen.size != (pw, ph)
                                else gen
                            )
                            full_gen.paste(gen_r, (crop_box[0], crop_box[1]))
                            comp = composite_inpaint_result(
                                full_gen,
                                orig,
                                full_mask,
                                mask_blur=request.mask_blur,
                                seam_erode=seam_erode,
                            )
                            batch_images.append(comp)
                        width, height = orig.size
                    else:
                        # Whole picture (classic): prefill masked region, run at (aligned) full size,
                        # then composite to keep unmasked pixels pixel-exact.
                        source, mask, width, height = resize_for_inpaint(orig, raw_mask)
                        source = apply_masked_content(source, mask, content)
                        if request.mask_blur > 0:
                            mask = blur_mask(mask, request.mask_blur)
                        callback = self._make_callback(
                            pipe,
                            request,
                            on_progress,
                            should_cancel,
                            total_steps=request.steps,
                            preview_every_n_steps=preview_every_n_steps,
                        )
                        prompt_kwargs = build_prompt_kwargs(
                            pipe,
                            parsed.prompt,
                            request.negative_prompt,
                            request.clip_skip,
                        )
                        output = self._call_pipe(
                            pipe,
                            **prompt_kwargs,
                            num_inference_steps=request.steps,
                            guidance_scale=request.cfg_scale,
                            num_images_per_prompt=request.batch_size,
                            generator=generator,
                            callback_on_step_end=callback,
                            callback_on_step_end_tensor_inputs=["latents"],
                            image=source,
                            mask_image=mask,
                            strength=request.denoising_strength,
                            width=width,
                            height=height,
                        )
                        seam_erode = int(getattr(request, "seam_erode", 0) or 0)
                        paste_mask = prepare_inpaint_mask(raw_mask, size=orig.size)
                        batch_images = [
                            composite_inpaint_result(
                                img if img.size == orig.size else img.resize(orig.size, Image.Resampling.LANCZOS),
                                orig,
                                paste_mask,
                                mask_blur=request.mask_blur,
                                seam_erode=seam_erode,
                            )
                            for img in output.images
                        ]
                        width, height = orig.size

                elif request.mode == GenerationMode.IMG2IMG:
                    assert init_images is not None
                    callback = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output = self._run_img2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        init_images[0],
                        steps=request.steps,
                        strength=request.denoising_strength,
                    )
                    batch_images = output.images
                    width, height = init_images[0].size

                elif request.enable_hr:
                    callback_pass1 = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=total_steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    first = self._run_txt2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback_pass1,
                        width=request.width,
                        height=request.height,
                        steps=request.steps,
                    )
                    hr_width, hr_height = align_to_multiple_of_8(
                        int(request.width * request.hr_scale),
                        int(request.height * request.hr_scale),
                    )
                    if request.save_before_hires:
                        before_hires_images.extend(first.images)
                    assert self._img2img is not None
                    callback_pass2 = self._make_callback(
                        self._img2img,
                        request,
                        on_progress,
                        should_cancel,
                        step_offset=request.steps,
                        total_steps=total_steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    resample = self._hr_resample_filter(request.hr_upscaler)
                    upscaled_images = []
                    for image in first.images:
                        upscaled_images.append(image.resize((hr_width, hr_height), resample))
                    batch_images = []
                    hr_request = request.model_copy(update={"batch_size": 1})
                    for image in upscaled_images:
                        hr_output = self._run_img2img_pass(
                            self._img2img,
                            hr_request,
                            parsed.prompt,
                            generator,
                            callback_pass2,
                            image,
                            steps=request.hr_steps,
                            strength=request.hr_denoising_strength,
                        )
                        batch_images.extend(hr_output.images)
                    width, height = hr_width, hr_height

                else:
                    callback = self._make_callback(
                        pipe,
                        request,
                        on_progress,
                        should_cancel,
                        total_steps=request.steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    output = self._run_txt2img_pass(
                        pipe,
                        request,
                        parsed.prompt,
                        generator,
                        callback,
                        width=request.width,
                        height=request.height,
                        steps=request.steps,
                    )
                    batch_images = output.images

                if request.sdxl_refiner_enabled and request.mode != GenerationMode.INPAINT:
                    if not is_sdxl_architecture(checkpoint.architecture):
                        raise ValueError("SDXL refiner can only run with an SDXL base checkpoint.")
                    refiner_pipe = self._load_refiner_checkpoint(request.sdxl_refiner_checkpoint_id)
                    self._apply_vae(refiner_pipe, request.vae_id)
                    self._apply_sampler(refiner_pipe, request.sampler, request.scheduler)
                    self._ensure_embeddings_for_prompt(
                        refiner_pipe,
                        request.prompt,
                        request.negative_prompt,
                        checkpoint.architecture,
                    )
                    callback_refiner = self._make_callback(
                        refiner_pipe,
                        request,
                        on_progress,
                        should_cancel,
                        step_offset=base_step_count,
                        total_steps=total_steps,
                        preview_every_n_steps=preview_every_n_steps,
                    )
                    refiner_request = request.model_copy(update={"batch_size": 1})
                    refined_images: list[Image.Image] = []
                    for image in batch_images:
                        refined = self._run_img2img_pass(
                            refiner_pipe,
                            refiner_request,
                            parsed.prompt,
                            generator,
                            callback_refiner,
                            image,
                            steps=request.sdxl_refiner_steps,
                            strength=request.sdxl_refiner_strength,
                        )
                        refined_images.extend(refined.images)
                    batch_images = refined_images

                images.extend(batch_images)
                batch_seeds = [seed] * len(batch_images)
                seeds.extend(batch_seeds)
                for img in batch_images:
                    infotexts.append(
                        format_infotext(
                            request,
                            seed=seed,
                            checkpoint=checkpoint,
                            output_width=width,
                            output_height=height,
                        )
                    )
                if on_progress and batch_images:
                    on_progress(
                        batch_index + 1,
                        max(1, int(request.batch_count)),
                        f"Batch {batch_index + 1}/{request.batch_count} complete",
                        batch_images[-1],
                        batch_images,
                        batch_seeds,
                    )

        except (KeyboardInterrupt, StopIteration):
            raise
        except GenerationCancelledError:
            # User clicked Stop (or the stall watchdog force-failed this
            # job) — expected, not a bug. Log it quietly instead of an
            # ERROR-level traceback that looks like a crash.
            logger.info("Generation cancelled (user stop or watchdog force-fail)")
            raise
        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            raise

        return GenerationResult(
            job_id=job_id,
            images=images,
            seeds=seeds,
            infotexts=infotexts,
            before_hires_images=before_hires_images,
            mode=request.mode,
        )
