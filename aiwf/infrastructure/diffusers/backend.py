from __future__ import annotations

import copy
import logging
import random
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
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    SASolverScheduler,
    TCDScheduler,
    UniPCMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import logging as diffusers_logging
from PIL import Image

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
    ARCH_SDXL,
    ARCH_SDXL_INPAINT,
    is_inpaint_architecture,
    is_sdxl_architecture,
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

logger = logging.getLogger(__name__)


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
    StableDiffusionInpaintPipeline: "stable-diffusion-v1-5/stable-diffusion-inpainting",
    StableDiffusionXLInpaintPipeline: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
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

    def _place_pipeline(self, pipe, *, prefer_offload: bool = False):
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
            logger.info(
                "SDXL on a <%.0f GB GPU — enabling model CPU offload automatically "
                "(use Low VRAM mode in Settings if you still hit out-of-memory).",
                self._AUTO_OFFLOAD_VRAM_GB,
            )
            pipe.enable_model_cpu_offload()
            self._offload_active = True
        else:
            pipe = pipe.to(self.devices.device())
            self._offload_active = False
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

    def _apply_fp8_storage(self, pipe) -> None:
        """Store UNet weights in FP8, compute in fp16 (diffusers layerwise casting)."""
        if not self.flags.fp8:
            return
        if self.flags.lowvram:
            logger.warning("FP8 weight storage is skipped in Low VRAM mode (conflicting offload hooks).")
            return
        unet = getattr(pipe, "unet", None)
        if unet is None or not hasattr(unet, "enable_layerwise_casting"):
            logger.warning("FP8 weight storage not supported by this diffusers version; continuing at fp16.")
            return
        try:
            unet.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=self.devices.dtype(self.flags.no_half),
            )
            logger.info("UNet weights stored in FP8 (compute fp16) — roughly half the UNet VRAM.")
        except Exception:
            logger.exception("Could not enable FP8 weight storage; continuing at fp16.")

    def _execution_device(self, pipe) -> torch.device:
        """Device computation actually runs on — never the 'meta' placeholder.

        Offloaded pipelines (lowvram/medvram/SDXL auto-offload) report their
        device as 'meta'; torch.Generator and .to() need the real one.
        """
        try:
            dev = pipe._execution_device
        except Exception:
            dev = getattr(pipe, "device", None)
        if dev is None or getattr(dev, "type", "meta") == "meta":
            return self.devices.device()
        return dev

    def _wants_offload(self, architecture: str) -> bool:
        vram = self.devices.total_vram_gb()
        if not is_sdxl_architecture(architecture) or vram <= 0.0:
            return False
        # FP8 storage halves the UNet, so ~8GB cards can keep the whole
        # pipeline resident — much faster than offloading.
        threshold = 7.0 if self.flags.fp8 else self._AUTO_OFFLOAD_VRAM_GB
        return vram < threshold

    def _compile_allowed_for_architecture(self, architecture: str) -> bool:
        return not (self.flags.lowvram or self.flags.medvram or self._wants_offload(architecture))

    def _tune_vae_memory(self, pipe, architecture: str) -> None:
        """SDXL's 1024px VAE decode is the peak-VRAM step — slice and tile it."""
        if not is_sdxl_architecture(architecture):
            return
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return
        try:
            vae.enable_slicing()
            vae.enable_tiling()
            pipe._aiwf_sdxl = True
            logger.info("SDXL VAE slicing + tiling enabled (cuts decode VRAM spike)")
        except Exception:
            logger.debug("Could not enable VAE slicing/tiling", exc_info=True)

    def _sync_img2img_from_txt2img(self) -> None:
        assert self._txt2img is not None
        pipe = self._txt2img
        if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
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
        pipeline_cls = (
            StableDiffusionXLPipeline
            if is_sdxl_architecture(checkpoint.architecture)
            else StableDiffusionPipeline
        )
        return _cached_single_file_config_dir(pipeline_cls) is not None

    def load_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        checkpoint = self._resolve_checkpoint(checkpoint_id)
        if self._txt2img is not None and self._active and self._active.path == checkpoint.path:
            return checkpoint

        if self._active and self._active.path != checkpoint.path:
            self.unload()
        elif self._txt2img is None and self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None
            self.devices.empty_cache()

        if self._txt2img is not None:
            self._active = checkpoint
            return checkpoint

        logger.info("Loading checkpoint %s (%s)", checkpoint.title, checkpoint.architecture)

        dtype = self.devices.dtype(self.flags.no_half)
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
        pipeline_cls = (
            StableDiffusionXLPipeline
            if is_sdxl_architecture(checkpoint.architecture)
            else StableDiffusionPipeline
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": path.suffix.lower() == ".safetensors",
        }
        _add_cached_single_file_config(load_kwargs, pipeline_cls)
        if pipeline_cls is StableDiffusionPipeline:
            load_kwargs["requires_safety_checker"] = False
        # AIWF loads local single-file checkpoints with app-level controls around
        # requests and file selection. Diffusers' optional safety checker needs
        # extra model assets and is disabled here so backend selection stays
        # deterministic/offline.
        pipe = pipeline_cls.from_single_file(checkpoint.path, **load_kwargs)
        self._remember_base_scheduler_config(pipe)
        self._apply_fp8_storage(pipe)

        compile_allowed = self._compile_allowed_for_architecture(checkpoint.architecture)
        apply_attention_optimizations(pipe, self.flags, compile_allowed=compile_allowed)
        pipe = self._place_pipeline(pipe, prefer_offload=self._wants_offload(checkpoint.architecture))
        self._tune_vae_memory(pipe, checkpoint.architecture)
        # Embeddings are now loaded on-demand in generate() only for those referenced
        # by the actual prompt (see _ensure_embeddings_for_prompt). Unused ones are
        # never loaded even if present in the embeddings/ folder.
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        self._txt2img = pipe
        self._sync_img2img_from_txt2img()
        self._active = checkpoint
        return checkpoint

    def _load_inpaint_checkpoint(
        self, checkpoint: Checkpoint
    ) -> StableDiffusionInpaintPipeline | StableDiffusionXLInpaintPipeline:
        if self._inpaint and self._inpaint_active and self._inpaint_active.path == checkpoint.path:
            return self._inpaint

        if self._inpaint_active and self._inpaint_active.path != checkpoint.path:
            self._inpaint = None
            self._inpaint_active = None

        if self._active and self._active.path != checkpoint.path:
            self._txt2img = None
            self._img2img = None
            self._active = None
            self._active_vae_id = None
            self.devices.empty_cache()

        logger.info("Loading inpaint pipeline for %s (%s)", checkpoint.title, checkpoint.architecture)
        dtype = self.devices.dtype(self.flags.no_half)
        path = Path(checkpoint.path)
        if checkpoint.architecture == ARCH_SDXL_INPAINT:
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
        _add_cached_single_file_config(load_kwargs, pipeline_cls)
        if pipeline_cls is StableDiffusionInpaintPipeline:
            load_kwargs["requires_safety_checker"] = False
        pipe = pipeline_cls.from_single_file(checkpoint.path, **load_kwargs)
        self._remember_base_scheduler_config(pipe)
        self._apply_fp8_storage(pipe)

        compile_allowed = self._compile_allowed_for_architecture(checkpoint.architecture)
        apply_attention_optimizations(pipe, self.flags, compile_allowed=compile_allowed)
        pipe = self._place_pipeline(pipe, prefer_offload=self._wants_offload(checkpoint.architecture))
        self._tune_vae_memory(pipe, checkpoint.architecture)
        # Embeddings are now loaded on-demand in generate() only for those referenced
        # by the actual prompt (see _ensure_embeddings_for_prompt). Unused ones are
        # never loaded even if present in the embeddings/ folder.
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

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
        dtype = self.devices.dtype(self.flags.no_half)
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
        pipe = self._place_pipeline(pipe, prefer_offload=self._wants_offload(checkpoint.architecture))
        self._tune_vae_memory(pipe, checkpoint.architecture)
        self._refiner = pipe
        self._refiner_active = checkpoint
        return pipe

    def _call_pipe(self, pipe, **kwargs):
        with attention_call_context(getattr(self, "flags", None)):
            return pipe(**kwargs)

    @staticmethod
    def _hr_resample_filter(upscaler: str) -> int:
        normalized = (upscaler or "lanczos").strip().lower().replace(" ", "_")
        if normalized == "bicubic":
            return Image.Resampling.BICUBIC
        if normalized == "nearest":
            return Image.Resampling.NEAREST
        return Image.Resampling.LANCZOS

    def unload(self) -> None:
        self._txt2img = None
        self._img2img = None
        self._inpaint = None
        self._refiner = None
        self._active = None
        self._inpaint_active = None
        self._refiner_active = None
        self._active_vae_id = None
        self._controlnet_cache.clear()
        self.devices.empty_cache()

    _SAMPLER_EXTRA_KWARGS = {
        "dpmpp_2m_sde": {"algorithm_type": "sde-dpmsolver++"},
        "dpmpp_3m_sde": {"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
        "dpmpp_2m_karras": {"use_karras_sigmas": True},
    }

    def _apply_sampler(self, pipe, sampler_id: str, schedule_type: str = "automatic") -> None:
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
        prepared: list[tuple[ControlNetUnit, Image.Image, Path]] = []
        for index, unit in enumerate(request.controlnet_units or []):
            if not unit.enabled or not unit.model:
                continue
            path = self._resolve_controlnet_path(unit.model)
            if path is None:
                roots = ", ".join(str(root) for root in resolve_controlnet_roots(self.flags)) or str(self._controlnet_dir())
                logger.warning("ControlNet model %s not found in %s", unit.model, roots)
                continue
            control = supplied[index] if index < len(supplied) else decode_control_image(unit.image)
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

        if request.mode == GenerationMode.INPAINT:
            if not init_images:
                raise ValueError("inpaint requires init_images")
            if not mask_images:
                raise ValueError("inpaint requires mask_images")
            if not is_inpaint_architecture(checkpoint.architecture):
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
        else:
            self.load_checkpoint(request.checkpoint_id)
            pipe = self._txt2img
            assert pipe is not None

        self._apply_vae(pipe, request.vae_id)
        if request.mode != GenerationMode.INPAINT and self._img2img is not None:
            self._apply_vae(self._img2img, request.vae_id)

        self._apply_sampler(pipe, request.sampler, request.scheduler)
        if self._img2img is not None and request.mode == GenerationMode.TXT2IMG and request.enable_hr:
            self._apply_sampler(self._img2img, request.sampler, request.scheduler)

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
            cn_pipe = self._place_pipeline(cn_pipe, prefer_offload=self._wants_offload(checkpoint.architecture))

        # Load textual inversions referenced by the (processed) prompt/negative.
        # Only embeddings the user actually uses in the prompt text are loaded here;
        # the rest stay on disk and are never injected into the text encoders.
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

                if cn_pipe is not None:
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
                seeds.extend([seed] * len(batch_images))
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

        except (KeyboardInterrupt, StopIteration):
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
