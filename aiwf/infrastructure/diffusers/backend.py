from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Callable
from uuid import uuid4

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
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
from aiwf.infrastructure.diffusers.extra_networks import apply_loras, clear_loras
from aiwf.infrastructure.diffusers.loras import scan_loras
from aiwf.infrastructure.diffusers.mask import align_to_multiple_of_8, blur_mask, resize_for_inpaint
from aiwf.infrastructure.diffusers.controlnet_pipe import (
    ControlNetModelCache,
    build_controlnet_pipeline,
)
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
from aiwf.infrastructure.torch.attention import apply_attention_optimizations
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
warnings.filterwarnings(
    "ignore",
    message="You have disabled the safety checker.*",
    category=UserWarning,
    module="diffusers.*",
)

SAMPLER_CLASSES = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
    "dpmpp_2m_karras": DPMSolverMultistepScheduler,
}


class DiffusersBackend:
    def __init__(self, flags: RuntimeFlags, devices: DeviceManager) -> None:
        self.flags = flags
        self.devices = devices
        self.ckpt_dir = flags.resolved_ckpt_dir()
        self._txt2img: StableDiffusionPipeline | None = None
        self._img2img: StableDiffusionImg2ImgPipeline | None = None
        self._inpaint: StableDiffusionInpaintPipeline | None = None
        self._active: Checkpoint | None = None
        self._inpaint_active: Checkpoint | None = None
        self._active_vae_id: str | None = None
        self._lora_catalog: list[LoraInfo] | None = None
        self._vae_catalog: list[VaeInfo] | None = None
        self._checkpoint_catalog: list[Checkpoint] | None = None
        self._controlnet_cache = ControlNetModelCache()

    def invalidate_checkpoints(self) -> None:
        self._checkpoint_catalog = None

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

    def _place_pipeline(self, pipe):
        if self.flags.lowvram:
            pipe.enable_sequential_cpu_offload()
        elif self.flags.medvram:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.devices.device())
        return pipe

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
        if not (self.flags.lowvram or self.flags.medvram):
            self._img2img = self._img2img.to(self.devices.device())

    def _apply_vae(self, pipe, vae_id: str | None) -> None:
        if vae_id == self._active_vae_id:
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
        device = pipe.device if hasattr(pipe, "device") else self.devices.device()
        pipe.vae = vae.to(device)
        self._active_vae_id = vae_info.id
        self._sync_img2img_from_txt2img()

    def resolve_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        return self._resolve_checkpoint(checkpoint_id)

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
        pipeline_cls = (
            StableDiffusionXLPipeline
            if is_sdxl_architecture(checkpoint.architecture)
            else StableDiffusionPipeline
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": path.suffix.lower() == ".safetensors",
        }
        if pipeline_cls is StableDiffusionPipeline:
            load_kwargs["requires_safety_checker"] = False
        pipe = pipeline_cls.from_single_file(checkpoint.path, **load_kwargs)

        apply_attention_optimizations(pipe, self.flags)
        pipe = self._place_pipeline(pipe)
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
        if pipeline_cls is StableDiffusionInpaintPipeline:
            load_kwargs["requires_safety_checker"] = False
        pipe = pipeline_cls.from_single_file(checkpoint.path, **load_kwargs)

        apply_attention_optimizations(pipe, self.flags)
        pipe = self._place_pipeline(pipe)
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        self._inpaint = pipe
        self._inpaint_active = checkpoint
        return pipe

    def unload(self) -> None:
        self._txt2img = None
        self._img2img = None
        self._inpaint = None
        self._active = None
        self._inpaint_active = None
        self._active_vae_id = None
        self._controlnet_cache.clear()
        self.devices.empty_cache()

    def _apply_sampler(self, pipe, sampler_id: str) -> None:
        cls = SAMPLER_CLASSES.get(sampler_id, EulerAncestralDiscreteScheduler)
        scheduler = cls.from_config(pipe.scheduler.config)
        if sampler_id == "dpmpp_2m_karras":
            scheduler.config.use_karras_sigmas = True
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
        return pipe(
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
        return pipe(
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
        unit: ControlNetUnit,
        control_image: Image.Image,
        init_images: list[Image.Image] | None,
    ):
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
            controlnet_conditioning_scale=float(unit.weight),
            control_guidance_start=float(unit.guidance_start),
            control_guidance_end=float(unit.guidance_end),
        )
        if request.mode == GenerationMode.IMG2IMG:
            assert init_images is not None
            base = init_images[0].convert("RGB")
            width, height = base.size
            control = control_image.resize((width, height), Image.Resampling.LANCZOS)
            output = pipe(
                **common,
                image=base,
                control_image=control,
                strength=request.denoising_strength,
            )
            return output, width, height

        width, height = align_to_multiple_of_8(request.width, request.height)
        control = control_image.resize((width, height), Image.Resampling.LANCZOS)
        output = pipe(
            **common,
            image=control,
            width=width,
            height=height,
        )
        return output, width, height

    _CONTROLNET_EXTS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}

    def _controlnet_dir(self) -> Path:
        return self.flags.resolved_models_dir() / "ControlNet"

    def _resolve_controlnet_path(self, model_id: str | None) -> Path | None:
        if not model_id:
            return None
        root = self._controlnet_dir()
        if not root.exists():
            return None
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self._CONTROLNET_EXTS:
                continue
            if path.stem == model_id or path.name == model_id:
                return path
        return None

    def _prepare_controlnet(
        self,
        request: GenerationRequest,
        control_images: list[Image.Image] | None,
    ) -> tuple[ControlNetUnit, Image.Image, Path] | None:
        """Resolve the first usable ControlNet unit into (unit, control_image, path).

        Returns None when ControlNet is not active or cannot be satisfied.
        """
        if request.mode not in (GenerationMode.TXT2IMG, GenerationMode.IMG2IMG):
            return None
        supplied = list(control_images or [])
        for index, unit in enumerate(request.controlnet_units or []):
            if not unit.enabled or not unit.model:
                continue
            path = self._resolve_controlnet_path(unit.model)
            if path is None:
                logger.warning("ControlNet model %s not found in %s", unit.model, self._controlnet_dir())
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
                    ),
                )
            return unit, control.convert("RGB"), path
        return None

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

        self._apply_sampler(pipe, request.sampler)
        if self._img2img is not None and request.mode == GenerationMode.TXT2IMG and request.enable_hr:
            self._apply_sampler(self._img2img, request.sampler)

        adapter_names = apply_loras(pipe, parsed.loras, self.list_loras())
        if self._img2img is not None and adapter_names:
            apply_loras(self._img2img, parsed.loras, self.list_loras())

        controlnet = self._prepare_controlnet(request, control_images)
        cn_pipe = None
        if controlnet is not None:
            cn_unit, cn_image, cn_path = controlnet
            if request.enable_hr:
                logger.warning("Hires fix is ignored while ControlNet is active.")
            dtype = self.devices.dtype(self.flags.no_half)
            cn_model = self._controlnet_cache.load(str(cn_path), dtype=dtype)
            cn_pipe = build_controlnet_pipeline(
                pipe, cn_model, img2img=request.mode == GenerationMode.IMG2IMG
            )
            cn_pipe = self._place_pipeline(cn_pipe)

        try:
            run_pipe = cn_pipe if cn_pipe is not None else pipe
            generator = torch.Generator(device=run_pipe.device)
            total_steps = request.steps + (request.hr_steps if request.enable_hr else 0)

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
                        cn_unit,
                        cn_image,
                        init_images,
                    )
                    batch_images = output.images

                elif request.mode == GenerationMode.INPAINT:
                    assert mask_images is not None and init_images is not None
                    source, mask, width, height = resize_for_inpaint(init_images[0], mask_images[0])
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
                    output = pipe(
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
                    batch_images = output.images

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
                    batch_images = []
                    for image in first.images:
                        upscaled = image.resize((hr_width, hr_height), Image.Resampling.LANCZOS)
                        second = self._run_img2img_pass(
                            self._img2img,
                            request,
                            parsed.prompt,
                            generator,
                            callback_pass2,
                            upscaled,
                            steps=request.hr_steps,
                            strength=request.hr_denoising_strength,
                        )
                        batch_images.extend(second.images)
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

                for image in batch_images:
                    images.append(image)
                    seeds.append(seed)
                    infotexts.append(
                        format_infotext(
                            request,
                            seed,
                            checkpoint,
                            output_width=width,
                            output_height=height,
                        )
                    )
        finally:
            clear_loras(pipe)
            if self._img2img is not None:
                clear_loras(self._img2img)

        return GenerationResult(
            job_id=job_id,
            images=images,
            seeds=seeds,
            infotexts=infotexts,
            mode=request.mode,
        )
