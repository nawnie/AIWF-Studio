from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from aiwf import __version__
from aiwf.core.domain.extra_networks import parse_extra_networks
from aiwf.core.domain.generation import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    JobRecord,
    JobState,
)
from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.core.domain.optimization import (
    ModelFamily,
    OptimizationPlan,
    OptimizationRequest,
    PipelineKind,
)
from aiwf.core.events.bus import EventBus
from aiwf.core.events.types import AfterGenerate, BeforeGenerate, JobProgressed
from aiwf.core.interfaces.backend import InferenceBackend
from aiwf.core.interfaces.storage import ImageStore
from aiwf.core.config.settings import UserSettings
from aiwf.services.metadata import MetadataService
from aiwf.services.failure_archive import FailureArchiveService
from aiwf.services.genlog import GenerationLogService
from aiwf.dev.diagnostics import trace_exception_safe, trace_model_throughput, trace_safe
from aiwf.core.model_profile import detect_model_profile
from aiwf.services.queue import JobQueue

if TYPE_CHECKING:
    from aiwf.services.prompt_processor import PromptProcessorService
    from aiwf.services.engine_supervisor import EngineSupervisor
    from aiwf.services.optimization import OptimizationPlanner


DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, normal quality, lowres, blurry, jpeg artifacts, "
    "watermark, signature, text, error, cropped, out of frame, duplicate"
)

# How long an image job can go with zero progress updates (no "Encoding prompt",
# no model-load message, no denoise step) before we treat it as stalled. This is
# generous on purpose — CPU-offloaded text encoders on large DiT models (Flux,
# Flux.2 Klein, Z-Image) can legitimately take a couple of minutes on first use.
# Past this point there is no plausible legitimate explanation, only a hang.
STALL_TIMEOUT_SECONDS = 240.0
_STALL_POLL_INTERVAL_SECONDS = 5.0
logger = logging.getLogger(__name__)


def _print_generation_progress(job_id: Any, step: int, total: int, message: str) -> None:
    total = max(1, int(total or 1))
    step = max(0, int(step or 0))
    label = str(job_id)[:8]
    print(f"[AIWF] Generation {label}: {step}/{total} - {message}", flush=True)


class GenerationService:
    """Application boundary for image generation orchestration.

    UI/API callers stay here while the backend owns torch, diffusers, ONNX, and
    sampler details. This layer coordinates prompts, queue state, metadata,
    tenant ownership, and receipts without making backend-specific assumptions.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        store: ImageStore,
        metadata: MetadataService,
        queue: JobQueue,
        events: EventBus,
        settings: UserSettings,
        prompts: PromptProcessorService | None = None,
        settings_path: Path | None = None,
        supervisor: EngineSupervisor | None = None,
        optimization_planner: OptimizationPlanner | None = None,
        failure_archive: FailureArchiveService | None = None,
        genlog: GenerationLogService | None = None,
    ) -> None:
        self.backend = backend
        self.store = store
        self.metadata = metadata
        self.queue = queue
        self.events = events
        self.settings = settings
        self.prompts = prompts
        self._settings_path = settings_path
        self.supervisor = supervisor
        self.optimization_planner = optimization_planner
        self.failure_archive = failure_archive
        self.genlog = genlog

    def _apply_default_negative(self, request):
        """When the user leaves the negative prompt blank, fall back to a generic
        quality negative (no style/subject words). Off if the user disabled it or
        set their own default. Toggle: settings.use_default_negative."""
        if (request.negative_prompt or "").strip():
            return request
        if not getattr(self.settings, "use_default_negative", True):
            return request
        default = (getattr(self.settings, "default_negative_prompt", "") or "").strip()
        return request.model_copy(update={"negative_prompt": default or DEFAULT_NEGATIVE_PROMPT})

    def _guard_distilled_cfg(self, request, checkpoint):
        """Distilled models (Lightning/Hyper-SD/Turbo/LCM/TCD) overexpose badly at
        normal CFG. When the requested guidance is too high for such a model, clamp
        it to a safe value and record it in the trace log so it is visible."""
        try:
            profile = detect_model_profile(
                getattr(checkpoint, "title", None),
                getattr(checkpoint, "filename", None),
                getattr(checkpoint, "id", None),
            )
        except Exception:
            return request
        if (
            profile.is_distilled
            and getattr(self.settings, "auto_cfg_for_distilled", True)
            and float(request.cfg_scale) > profile.cfg_max
        ):
            trace_safe(
                "generation.cfg_clamp",
                "Clamped CFG for distilled model to avoid overexposure",
                family=profile.family,
                requested_cfg=float(request.cfg_scale),
                applied_cfg=profile.recommended_cfg,
                checkpoint_id=getattr(checkpoint, "id", None),
            )
            return request.model_copy(update={"cfg_scale": profile.recommended_cfg})
        return request

    def _route_inpaint_checkpoint_for_txt2img(
        self,
        request: GenerationRequest,
        checkpoint,
        init_images: list[Image.Image] | None,
        mask_images: list[Image.Image] | None,
    ) -> tuple[GenerationRequest, list[Image.Image] | None, list[Image.Image] | None]:
        """Let an inpaint (9-channel UNet) checkpoint run from the regular txt2img tab.

        These checkpoints can't load through the normal txt2img pipeline (conv_in
        expects 9 channels, not 4) so the backend used to hard-fail with
        ModelNotFoundError and tell the user to switch to Inpaint mode. That's an
        unnecessary mode switch for what should just work: feed the inpaint UNet a
        blank canvas with a fully-white (fully masked) mask, which makes it denoise
        the whole image from scratch - functionally equivalent to txt2img.
        """
        from aiwf.infrastructure.diffusers.model_arch import is_inpaint_architecture

        if request.mode != GenerationMode.TXT2IMG:
            return request, init_images, mask_images
        if not is_inpaint_architecture(getattr(checkpoint, "architecture", "")):
            return request, init_images, mask_images
        if init_images:
            # User already supplied an init image (e.g. via an extra-networks flow);
            # don't override their intent.
            return request, init_images, mask_images

        width = max(64, int(request.width))
        height = max(64, int(request.height))
        blank = Image.new("RGB", (width, height), (127, 127, 127))
        full_mask = Image.new("L", (width, height), 255)
        trace_safe(
            "generation.inpaint_checkpoint_in_txt2img",
            "Routing inpaint checkpoint through inpaint pipeline with a full-canvas mask",
            checkpoint_id=getattr(checkpoint, "id", None),
            width=width,
            height=height,
        )
        routed_request = request.model_copy(update={"mode": GenerationMode.INPAINT})
        return routed_request, [blank], [full_mask]

    def _persist_last_checkpoint(self, checkpoint_id: str) -> None:
        """Remember the last model used so the next launch restores it."""
        if not checkpoint_id or self.settings.last_checkpoint_id == checkpoint_id:
            return
        self.settings.last_checkpoint_id = checkpoint_id
        if self._settings_path is None:
            return
        self._settings_path.write_text(
            self.settings.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def get_model_preset(self, checkpoint_id: str | None = None) -> dict:
        """Resolve the generation settings to show for a checkpoint: last-used
        for that exact checkpoint if we have it, else a sane default for its
        architecture. Returns {} if the checkpoint can't be resolved."""
        from aiwf.infrastructure.diffusers.model_presets import resolve_model_preset

        try:
            checkpoint = self.backend.resolve_checkpoint(checkpoint_id)
        except Exception:
            return {}
        return resolve_model_preset(
            self.settings.model_settings,
            checkpoint.id,
            getattr(checkpoint, "architecture", None),
        )

    def _persist_model_settings(self, checkpoint_id: str, request: GenerationRequest) -> None:
        """Remember this request's generation knobs against the checkpoint that
        produced it, so the next time this model is selected the UI defaults to
        whatever last worked rather than a generic global default."""
        from aiwf.infrastructure.diffusers.model_presets import extract_preset_fields

        if not checkpoint_id:
            return
        fields = extract_preset_fields(request)
        if not fields:
            return
        if self.settings.model_settings.get(checkpoint_id) == fields:
            return
        self.settings.model_settings[checkpoint_id] = fields
        if self._settings_path is None:
            return
        self._settings_path.write_text(
            self.settings.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _resolve_prompts(self, request: GenerationRequest) -> GenerationRequest:
        if self.prompts is None:
            return request
        seed = request.prompt_seed
        if seed is None and request.seed >= 0:
            seed = request.seed
        style_override = None
        if (request.style_prompt_template or "").strip() or (request.style_negative_template or "").strip():
            from aiwf.core.domain.prompt_style import PromptStyle

            style_override = PromptStyle(
                name=request.style_name or "",
                prompt=request.style_prompt_template or "",
                negative_prompt=request.style_negative_template or "",
            )
        prompt, negative = self.prompts.prepare_prompt(
            request.prompt,
            negative_text=request.negative_prompt,
            prompt_file=request.prompt_file,
            use_prompt_file=request.use_prompt_file,
            style_name=request.style_name,
            style_override=style_override,
            seed=seed,
        )
        return request.model_copy(update={"prompt": prompt, "negative_prompt": negative})

    @staticmethod
    def _catalog_match(items, item_id: str | None):
        if not item_id:
            return None
        lowered = str(item_id).lower()
        for item in items:
            candidates = (
                getattr(item, "id", ""),
                getattr(item, "title", ""),
                getattr(item, "filename", ""),
            )
            if any(str(candidate).lower() == lowered for candidate in candidates):
                return item
        return None

    @staticmethod
    def _pipeline_kind_for_request(request: GenerationRequest) -> PipelineKind:
        if request.mode == GenerationMode.INPAINT:
            return PipelineKind.INPAINT
        if any(unit.enabled for unit in request.controlnet_units):
            return PipelineKind.CONTROLNET
        if request.enable_hr:
            return PipelineKind.HIRES
        if request.mode == GenerationMode.IMG2IMG:
            return PipelineKind.IMG2IMG
        return PipelineKind.TXT2IMG

    @staticmethod
    def _model_family_for_checkpoint(checkpoint) -> ModelFamily:
        try:
            prof = detect_model_profile(
                getattr(checkpoint, "title", None),
                getattr(checkpoint, "filename", None),
                getattr(checkpoint, "id", None),
            )
            if prof.family == "turbo":
                return ModelFamily.SDXL_TURBO
        except Exception:
            pass
        architecture = str(getattr(checkpoint, "architecture", "") or "").lower()
        if architecture in {"sd35", "sd3", "stable-diffusion-3", "stable-diffusion-3.5"}:
            return ModelFamily.SD3
        blob = " ".join(
            str(value or "")
            for value in (
                getattr(checkpoint, "title", None),
                getattr(checkpoint, "filename", None),
                getattr(checkpoint, "id", None),
            )
        ).lower()
        if "sdxl" in blob or "_xl" in blob or "-xl" in blob or " xl" in blob:
            return ModelFamily.SDXL
        if "sd1.5" in blob or "sd15" in blob or "stable-diffusion-v1" in blob:
            return ModelFamily.SD15
        return ModelFamily.UNKNOWN

    @staticmethod
    def _genlog_generate_type(checkpoint) -> str:
        architecture = str(getattr(checkpoint, "architecture", "") or "").lower()
        if architecture.startswith("sdxl"):
            return "sdxl"
        if architecture in {"sd15", "sd1", "sd1.5", "inpaint"} or architecture.startswith("sd1"):
            return "sd"
        if architecture.startswith("sd3"):
            return "sd3"
        return architecture or "unknown"

    def _genlog_step_count(self, request: GenerationRequest) -> int:
        per_batch = max(1, int(request.steps))
        controlnet_active = any(unit.enabled for unit in request.controlnet_units)
        if request.enable_hr and request.mode == GenerationMode.TXT2IMG and not controlnet_active:
            per_batch += max(1, int(request.hr_steps))
        if request.sdxl_refiner_enabled and request.mode in (GenerationMode.TXT2IMG, GenerationMode.IMG2IMG):
            per_batch += max(1, int(request.sdxl_refiner_steps))
        return max(1, per_batch * max(1, int(request.batch_count)))

    def _write_generation_log(
        self,
        *,
        job: JobRecord,
        result: GenerationResult,
        checkpoint,
        optimization_plan: OptimizationPlan | None,
    ) -> None:
        if self.genlog is None or not self.genlog.enabled:
            return
        try:
            request = job.request
            parsed = parse_extra_networks(request.prompt)
            generate_type = self._genlog_generate_type(checkpoint)
            pipeline_kind = self._pipeline_kind_for_request(request).value
            flags = getattr(self.backend, "flags", None)
            backend_name = getattr(flags, "inference_backend", None) or self.backend.__class__.__name__.replace(
                "Backend",
                "",
            ).lower()
            step_count = self._genlog_step_count(request)
            elapsed = float(result.elapsed_seconds or 0.0)
            controlnet_units = [
                {
                    "model_id": unit.model,
                    "module": unit.module,
                    "weight": unit.weight,
                    "guidance_start": unit.guidance_start,
                    "guidance_end": unit.guidance_end,
                }
                for unit in request.controlnet_units
                if unit.enabled
            ]
            settings: dict[str, Any] = {
                "mode": request.mode.value,
                "width": int(request.width),
                "height": int(request.height),
                "steps": int(request.steps),
                "cfg_scale": float(request.cfg_scale),
                "sampler": request.sampler,
                "scheduler": request.scheduler,
                "batch_size": int(request.batch_size),
                "batch_count": int(request.batch_count),
                "seed": int(request.seed),
                "clip_skip": int(request.clip_skip),
                "vae_id": request.vae_id,
                "enable_hr": bool(request.enable_hr),
                "sdxl_refiner_enabled": bool(request.sdxl_refiner_enabled),
                "controlnet_units": controlnet_units,
            }
            if request.mode in (GenerationMode.IMG2IMG, GenerationMode.INPAINT):
                settings["denoising_strength"] = float(request.denoising_strength)
            if request.mode == GenerationMode.INPAINT:
                settings.update(
                    {
                        "mask_blur": int(request.mask_blur),
                        "inpaint_only_masked": bool(request.inpaint_only_masked),
                        "inpaint_masked_padding": int(request.inpaint_masked_padding),
                        "inpaint_mask_content": request.inpaint_mask_content,
                    }
                )
            if request.enable_hr:
                settings.update(
                    {
                        "hr_scale": float(request.hr_scale),
                        "hr_steps": int(request.hr_steps),
                        "hr_denoising_strength": float(request.hr_denoising_strength),
                        "hr_upscaler": request.hr_upscaler,
                    }
                )
            if request.sdxl_refiner_enabled:
                settings.update(
                    {
                        "sdxl_refiner_checkpoint_id": request.sdxl_refiner_checkpoint_id,
                        "sdxl_refiner_steps": int(request.sdxl_refiner_steps),
                        "sdxl_refiner_strength": float(request.sdxl_refiner_strength),
                    }
                )
            runtime_flags = {}
            if flags is not None:
                runtime_flags = {
                    "attention_backend": getattr(flags, "attention_backend", ""),
                    "vram_profile": flags.effective_vram_profile() if hasattr(flags, "effective_vram_profile") else "",
                    "fp8": bool(getattr(flags, "fp8", False)),
                    "medvram": bool(getattr(flags, "medvram", False)),
                    "lowvram": bool(getattr(flags, "lowvram", False)),
                    "highvram": bool(getattr(flags, "highvram", False)),
                    "no_half": bool(getattr(flags, "no_half", False)),
                    "torch_compile": bool(getattr(flags, "torch_compile", False)),
                    "channels_last": bool(getattr(flags, "channels_last", False)),
                }
            self.genlog.append(
                {
                    "event": "generation_completed",
                    "kind": "image",
                    "generate_type": generate_type,
                    "backend": backend_name,
                    "pipeline": f"{backend_name}.{generate_type}.{pipeline_kind}",
                    "pipeline_kind": pipeline_kind,
                    "job_id": str(job.id),
                    "model": {
                        "id": getattr(checkpoint, "id", None),
                        "title": getattr(checkpoint, "title", None),
                        "filename": getattr(checkpoint, "filename", None),
                        "architecture": getattr(checkpoint, "architecture", None),
                    },
                    "loras": [{"name": ref.name, "weight": ref.weight} for ref in parsed.loras],
                    "settings": settings,
                    "runtime_flags": runtime_flags,
                    "optimization_profile_id": (
                        optimization_plan.profile_id if optimization_plan is not None else None
                    ),
                    "timing": {
                        "elapsed_seconds": round(elapsed, 6),
                        "step_count": step_count,
                        "steps_per_second": round(step_count / elapsed, 6) if elapsed > 0 else None,
                        "image_count": len(result.images),
                        "images_per_second": round(len(result.images) / elapsed, 6) if elapsed > 0 else None,
                    },
                    "outputs": {
                        "paths": [artifact.path for artifact in result.artifacts],
                        "image_count": len(result.images),
                    },
                }
            )
        except Exception as exc:
            trace_exception_safe("generation.genlog", exc, job_id=str(job.id))

    def _resolve_optimization_plan(self, request: GenerationRequest, checkpoint) -> OptimizationPlan | None:
        """Resolve optional tuning flags without making them boot-critical."""
        if self.optimization_planner is None:
            return None
        try:
            parsed = parse_extra_networks(request.prompt)
            opt_request = OptimizationRequest(
                profile_id=getattr(self.settings, "optimization_profile_id", "balanced_sdpa_fp16"),
                pipeline_kind=self._pipeline_kind_for_request(request),
                model_family=self._model_family_for_checkpoint(checkpoint),
                width=int(request.width),
                height=int(request.height),
                batch_size=int(request.batch_size),
                lora_count=len(parsed.loras),
                controlnet_count=sum(1 for unit in request.controlnet_units if unit.enabled),
            )
            plan = self.optimization_planner.resolve(opt_request)
            trace_safe(
                "generation.optimization_plan",
                "Resolved generation optimization profile",
                profile_id=plan.profile_id,
                requested_profile_id=plan.requested_profile_id,
                blocked=plan.blocked,
                decisions=[d.model_dump(mode="json") for d in plan.decisions],
            )
            trace_safe(
                "optimization.profile_resolved",
                "Optimization profile resolved",
                profile_id=plan.profile_id,
                requested_profile_id=plan.requested_profile_id,
                pipeline_kind=opt_request.pipeline_kind.value,
                model_family=opt_request.model_family.value,
            )
            for decision in plan.decisions:
                if decision.decision in {"blocked", "disabled"}:
                    trace_safe(
                        "optimization.flag_blocked",
                        decision.reason,
                        key=decision.key,
                        decision=decision.decision,
                        severity=decision.severity,
                    )
            return plan
        except Exception as exc:
            trace_exception_safe("generation.optimization_plan", exc)
            return None

    def _enrich_saved_infotext(
        self,
        infotext: str,
        request: GenerationRequest,
        checkpoint,
        optimization_plan: OptimizationPlan | None = None,
    ) -> str:
        vae_name = None
        vae_hash = None
        if self.settings.metadata_include_vae_hash and request.vae_id:
            vae = self._catalog_match(self.backend.list_vaes(), request.vae_id)
            if vae is not None:
                vae_name = getattr(vae, "title", None) or getattr(vae, "id", None)
                vae_hash = self.metadata.file_fingerprint(getattr(vae, "path", ""))

        lora_hashes: dict[str, str] = {}
        if self.settings.metadata_include_lora_hashes:
            parsed = parse_extra_networks(request.prompt)
            loras = self.backend.list_loras()
            for ref in parsed.loras:
                lora = self._catalog_match(loras, ref.name)
                if lora is None:
                    continue
                fingerprint = self.metadata.file_fingerprint(getattr(lora, "path", ""))
                if fingerprint:
                    name = getattr(lora, "id", None) or getattr(lora, "title", ref.name)
                    lora_hashes[str(name)] = fingerprint

        return self.metadata.enrich_infotext(
            infotext,
            model_hash=(
                getattr(checkpoint, "hash", None)
                if self.settings.metadata_include_model_hash
                else None
            ),
            vae_name=vae_name,
            vae_hash=vae_hash,
            lora_hashes=lora_hashes,
            app_version=__version__ if self.settings.metadata_include_app_version else None,
            optimization_profile_id=(
                optimization_plan.profile_id
                if optimization_plan is not None
                and getattr(self.settings, "metadata_include_optimization_profile", True)
                else None
            ),
        )

    def _apply_generation_settings(self, request: GenerationRequest) -> GenerationRequest:
        updates = {}
        if getattr(self.settings, "save_before_hires", False):
            updates["save_before_hires"] = True
        if getattr(self.settings, "save_interrupted", False):
            updates["save_interrupted"] = True
        default_upscaler = getattr(self.settings, "default_hr_upscaler", "")
        if default_upscaler and request.hr_upscaler in {"", "lanczos"}:
            updates["hr_upscaler"] = default_upscaler
        # NOTE: the SDXL refiner toggle in Settings must never silently force
        # the refiner pass onto a request that didn't explicitly ask for it.
        # Doing so loads a second full SDXL pipeline (its own UNet/VAE/2 text
        # encoders) into VRAM on top of the already-cached base pipe -- which
        # is what was doubling VRAM use even at low resolutions. Only fill in
        # a missing refiner checkpoint id when the request itself already
        # opted in.
        if request.sdxl_refiner_enabled and not request.sdxl_refiner_checkpoint_id:
            refiner_id = getattr(self.settings, "sdxl_refiner_checkpoint_id", None)
            if refiner_id:
                updates["sdxl_refiner_checkpoint_id"] = refiner_id
        return request.model_copy(update=updates) if updates else request

    def _output_subdir(self, mode: GenerationMode) -> str:
        return {
            GenerationMode.TXT2IMG: self.settings.txt2img_output_subdir,
            GenerationMode.IMG2IMG: self.settings.img2img_output_subdir,
            GenerationMode.INPAINT: self.settings.inpaint_output_subdir,
        }[mode]

    @staticmethod
    def _metadata_model_name(checkpoint) -> str:
        filename = getattr(checkpoint, "filename", "") or ""
        fallback = Path(filename).stem if filename else ""
        return (
            getattr(checkpoint, "title", None)
            or getattr(checkpoint, "name", None)
            or fallback
            or getattr(checkpoint, "id", None)
            or "model"
        )

    @staticmethod
    def _prompt_head(prompt: str) -> str:
        return (" ".join((prompt or "").split())[:24] or "prompt").strip()

    def _training_filename_stem(self, request: GenerationRequest, checkpoint, counter: int) -> str:
        model_name = str(self._metadata_model_name(checkpoint)).strip()[:80] or "model"
        return f"{self._prompt_head(request.prompt)}-{model_name}-{counter}"

    def _training_caption(
        self,
        request: GenerationRequest,
        checkpoint,
        *,
        seed: int | None,
        index: int,
        image: Image.Image,
    ) -> str:
        prompt = " ".join((request.prompt or "").split())
        negative = " ".join((request.negative_prompt or "").split())
        model_name = self._metadata_model_name(checkpoint)
        parts = [prompt or "Generated image"]
        if request.tags:
            parts.append("Tags: " + ", ".join(request.tags) + ".")
        if negative:
            parts.append("Negative prompt: " + negative + ".")
        details = [
            f"mode {request.mode.value}",
            f"model {model_name}",
            f"{getattr(image, 'width', request.width)}x{getattr(image, 'height', request.height)}",
            f"seed {seed if seed is not None else request.seed}",
            f"image {index + 1}",
            f"{request.steps} steps",
            f"CFG {request.cfg_scale:g}",
            f"sampler {request.sampler}",
        ]
        scheduler = getattr(request, "scheduler", "automatic")
        if scheduler and scheduler != "automatic":
            details.append(f"schedule {scheduler}")
        if request.enable_hr:
            details.append(f"hires {request.hr_scale:g}x")
        if request.controlnet_units:
            active_units = sum(1 for unit in request.controlnet_units if getattr(unit, "enabled", False))
            if active_units:
                details.append(f"{active_units} ControlNet unit(s)")
        parts.append("Generation details: " + ", ".join(details) + ".")
        return " ".join(parts)

    def _training_metadata_payload(
        self,
        request: GenerationRequest,
        checkpoint,
        *,
        caption: str,
        seed: int | None,
        index: int,
        image: Image.Image,
        optimization_plan: OptimizationPlan | None,
    ) -> dict[str, Any]:
        return {
            "for_ai_training": True,
            "caption": caption,
            "full_prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "image_index": index,
            "counter": index + 1,
            "seed": seed if seed is not None else request.seed,
            "output_width": getattr(image, "width", request.width),
            "output_height": getattr(image, "height", request.height),
            "request": request.model_dump(mode="json"),
            "model": {
                "id": getattr(checkpoint, "id", None),
                "title": getattr(checkpoint, "title", None),
                "filename": getattr(checkpoint, "filename", None),
                "architecture": getattr(checkpoint, "architecture", None),
                "hash": getattr(checkpoint, "hash", None),
            },
            "optimization_profile_id": (
                optimization_plan.profile_id if optimization_plan is not None else None
            ),
            "metadata_schema": "aiwf.training.v1",
        }

    def _runtime_flags_payload(self) -> dict[str, Any]:
        flags = getattr(self.backend, "flags", None)
        if flags is None:
            return {}
        def flag_text(name: str) -> str:
            value = getattr(flags, name, "")
            return value if isinstance(value, str) else ""

        def flag_bool(name: str) -> bool:
            value = getattr(flags, name, False)
            return value if isinstance(value, bool) else False

        vram_profile = flags.effective_vram_profile() if hasattr(flags, "effective_vram_profile") else ""
        if not isinstance(vram_profile, str):
            vram_profile = ""
        return {
            "inference_backend": flag_text("inference_backend"),
            "attention_backend": flag_text("attention_backend"),
            "vram_profile": vram_profile,
            "fp8": flag_bool("fp8"),
            "medvram": flag_bool("medvram"),
            "lowvram": flag_bool("lowvram"),
            "highvram": flag_bool("highvram"),
            "no_half": flag_bool("no_half"),
            "torch_compile": flag_bool("torch_compile"),
            "channels_last": flag_bool("channels_last"),
        }

    def _pro_settings_payload(
        self,
        request: GenerationRequest,
        *,
        seed: int | None,
        image: Image.Image,
    ) -> dict[str, Any]:
        mode = "inpaint" if request.mode == GenerationMode.INPAINT else "image"
        active_control_unit = next((unit for unit in request.controlnet_units if getattr(unit, "enabled", False)), None)
        payload: dict[str, Any] = {
            "mode": mode,
            "prompt": request.prompt,
            "negativePrompt": request.negative_prompt,
            "modelId": request.checkpoint_id or "",
            "width": int(getattr(image, "width", request.width)),
            "height": int(getattr(image, "height", request.height)),
            "steps": int(request.steps),
            "cfgScale": float(request.cfg_scale),
            "sampler": request.sampler,
            "scheduler": request.scheduler,
            "seed": seed if seed is not None else request.seed,
            "clipSkip": int(request.clip_skip),
            "batchSize": int(request.batch_size),
            "batchCount": int(request.batch_count),
            "enableHires": bool(request.enable_hr),
            "hiresScale": float(request.hr_scale),
            "hiresSteps": int(request.hr_steps),
            "hiresDenoise": float(request.hr_denoising_strength),
            "hiresUpscaler": request.hr_upscaler,
            "denoisingStrength": float(request.denoising_strength),
            "maskBlur": int(request.mask_blur),
            "inpaintOnlyMasked": bool(request.inpaint_only_masked),
            "inpaintMaskedPadding": int(request.inpaint_masked_padding),
            "inpaintMaskContent": request.inpaint_mask_content,
            "controlNetEnabled": bool(active_control_unit),
            "controlNetModel": getattr(active_control_unit, "model", "") if active_control_unit else "",
            "controlNetModule": getattr(active_control_unit, "module", "none") if active_control_unit else "none",
            "controlNetWeight": float(getattr(active_control_unit, "weight", 1.0)) if active_control_unit else 1.0,
            "controlNetGuidanceStart": (
                float(getattr(active_control_unit, "guidance_start", 0.0)) if active_control_unit else 0.0
            ),
            "controlNetGuidanceEnd": (
                float(getattr(active_control_unit, "guidance_end", 1.0)) if active_control_unit else 1.0
            ),
            "controlNetProcessorRes": (
                int(getattr(active_control_unit, "processor_res", 512)) if active_control_unit else 512
            ),
            "saveImages": bool(request.save_images),
        }
        if request.vae_id:
            payload["vaeId"] = request.vae_id
        return payload

    def _generation_metadata_payload(
        self,
        result: GenerationResult,
        request: GenerationRequest,
        checkpoint,
        *,
        seed: int | None,
        index: int,
        image: Image.Image,
        optimization_plan: OptimizationPlan | None,
    ) -> dict[str, Any]:
        elapsed = float(result.elapsed_seconds or 0.0)
        step_count = self._genlog_step_count(request)
        parsed = parse_extra_networks(request.prompt)
        return {
            "metadata_schema": "aiwf.generation.v1",
            "generator": "aiwf-studio",
            "app_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "image",
            "image_index": index,
            "seed": seed if seed is not None else request.seed,
            "output": {
                "width": int(getattr(image, "width", request.width)),
                "height": int(getattr(image, "height", request.height)),
            },
            "model": {
                "id": getattr(checkpoint, "id", None),
                "title": getattr(checkpoint, "title", None),
                "filename": getattr(checkpoint, "filename", None),
                "architecture": getattr(checkpoint, "architecture", None),
                "hash": getattr(checkpoint, "hash", None),
                "kind": getattr(checkpoint, "kind", None),
                "asset_summary": getattr(checkpoint, "asset_summary", None),
            },
            "loras": [{"name": ref.name, "weight": ref.weight} for ref in parsed.loras],
            "settings": request.model_dump(mode="json"),
            "pro_settings": self._pro_settings_payload(request, seed=seed, image=image),
            "optimization": {
                "profile_id": optimization_plan.profile_id if optimization_plan is not None else None,
                "pipeline_kind": self._pipeline_kind_for_request(request).value,
                "model_family": self._model_family_for_checkpoint(checkpoint).value,
            },
            "runtime": self._runtime_flags_payload(),
            "receipt": {
                "job_id": str(result.job_id),
                "elapsed_seconds": round(elapsed, 6),
                "step_count": step_count,
                "steps_per_second": round(step_count / elapsed, 6) if elapsed > 0 else None,
                "image_count": len(result.images),
                "images_per_second": round(len(result.images) / elapsed, 6) if elapsed > 0 else None,
                "batch_size": int(request.batch_size),
                "batch_count": int(request.batch_count),
            },
        }

    def _write_generation_receipt(
        self,
        image_path: Path,
        *,
        metadata_payload: dict[str, Any],
        infotext: str,
    ) -> str | None:
        receipt = {
            "schema": "aiwf.generation-receipt.v1",
            "image_path": str(image_path),
            "infotext": infotext,
            **metadata_payload,
        }
        receipt_path = image_path.with_suffix(f"{image_path.suffix}.receipt.json")
        try:
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
            return str(receipt_path)
        except OSError:
            return None

    def _save_generation_result(
        self,
        result: GenerationResult,
        request: GenerationRequest,
        checkpoint,
        optimization_plan: OptimizationPlan | None,
    ) -> None:
        subdir = self._output_subdir(request.mode)
        model_name = getattr(checkpoint, "name", None) or getattr(checkpoint, "id", None)
        artifacts = []
        saved_images = []
        training_enabled = bool(getattr(request, "training_metadata", False))
        for index, image in enumerate(result.images):
            infotext = result.infotexts[index] if index < len(result.infotexts) else ""
            seed = result.seeds[index] if index < len(result.seeds) else None
            if training_enabled and not (infotext or "").strip():
                infotext = self.metadata.build_infotext(
                    request,
                    seed if seed is not None else request.seed,
                    checkpoint,
                    output_width=getattr(image, "width", None),
                    output_height=getattr(image, "height", None),
                )
            infotext = self._enrich_saved_infotext(
                infotext,
                request,
                checkpoint,
                optimization_plan,
            )
            if index < len(result.infotexts):
                result.infotexts[index] = infotext
            elif infotext:
                result.infotexts.append(infotext)
            caption = None
            extra_text = None
            extra_payload = None
            filename_stem = None
            format_override = None
            generation_payload = self._generation_metadata_payload(
                result,
                request,
                checkpoint,
                seed=seed,
                index=index,
                image=image,
                optimization_plan=optimization_plan,
            )
            extra_text = {
                "aiwf_generation": json.dumps(generation_payload, sort_keys=True),
                "aiwf_generation_settings": json.dumps(generation_payload["pro_settings"], sort_keys=True),
                "aiwf_generation_receipt": json.dumps(generation_payload["receipt"], sort_keys=True),
            }
            extra_payload = {
                "generation_metadata_schema": "aiwf.generation.v1",
                "generation": generation_payload,
            }
            if training_enabled:
                caption = self._training_caption(
                    request,
                    checkpoint,
                    seed=seed,
                    index=index,
                    image=image,
                )
                training_payload = self._training_metadata_payload(
                    request,
                    checkpoint,
                    caption=caption,
                    seed=seed,
                    index=index,
                    image=image,
                    optimization_plan=optimization_plan,
                )
                extra_text.update({
                    "full_prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "aiwf_training": json.dumps(training_payload, sort_keys=True),
                })
                extra_payload.update({
                    "for_ai_training": True,
                    "training_metadata_schema": "aiwf.training.v1",
                    "training": training_payload,
                })
                filename_stem = self._training_filename_stem(request, checkpoint, index + 1)
                format_override = "png"
            if self.settings.embed_metadata or request.tags or training_enabled:
                image = self.metadata.embed(
                    image,
                    infotext,
                    tags=request.tags,
                    caption=caption,
                    extra_text=extra_text,
                    extra_payload=extra_payload,
                )
            artifact = self.store.save(
                image,
                infotext,
                subdir,
                seed=seed,
                index=index,
                model_name=model_name,
                filename_stem=filename_stem,
                format_override=format_override,
            )
            if self.settings.embed_metadata or training_enabled:
                receipt_path = self._write_generation_receipt(
                    Path(artifact.path),
                    metadata_payload=generation_payload,
                    infotext=infotext,
                )
                artifact = artifact.model_copy(
                    update={
                        "receipt_path": receipt_path,
                        "metadata": generation_payload,
                    }
                )
            artifacts.append(artifact)
            saved_images.append(image)
        if request.save_before_hires and result.before_hires_images:
            first_pass_subdir = f"{subdir}/hires-first-pass"
            for index, image in enumerate(result.before_hires_images):
                infotext = result.infotexts[index] if index < len(result.infotexts) else ""
                infotext = f"{infotext}\nHires first pass: true".strip()
                seed = result.seeds[index] if index < len(result.seeds) else None
                artifact = self.store.save(
                    image,
                    infotext,
                    first_pass_subdir,
                    seed=seed,
                    index=index,
                    model_name=model_name,
                )
                artifacts.append(artifact)
        if getattr(self.settings, "save_grid", False) and len(saved_images) > 1:
            grid_info = result.infotexts[0] if result.infotexts else ""
            grid_artifact = self.store.save_grid(saved_images, subdir, infotext=grid_info)
            if grid_artifact is not None:
                artifacts.append(grid_artifact)
        result.artifacts = artifacts

    def _save_cancelled_preview(
        self,
        job: JobRecord,
        request: GenerationRequest,
        checkpoint,
        optimization_plan: OptimizationPlan | None,
        preview: Image.Image | None,
    ) -> None:
        if preview is None or not request.save_interrupted or not self.settings.save_images or not request.save_images:
            return
        infotext = self._enrich_saved_infotext(
            "Interrupted generation preview",
            request,
            checkpoint,
            optimization_plan,
        )
        result = GenerationResult(
            job_id=job.id,
            images=[preview],
            seeds=[request.seed],
            infotexts=[infotext],
            mode=request.mode,
        )
        subdir = f"{self._output_subdir(request.mode)}/interrupted"
        model_name = getattr(checkpoint, "name", None) or getattr(checkpoint, "id", None)
        image = self.metadata.embed(preview, infotext, tags=request.tags) if self.settings.embed_metadata else preview
        artifact = self.store.save(image, infotext, subdir, seed=request.seed, index=0, model_name=model_name)
        result.artifacts = [artifact]
        job.result = result

    def _archive_failed_generation(
        self,
        job: JobRecord,
        exc: BaseException,
        *,
        preview: Image.Image | None,
        checkpoint=None,
        stage: str = "image_generation",
    ) -> None:
        if self.failure_archive is None:
            return
        try:
            checkpoint_id = getattr(checkpoint, "id", None) if checkpoint is not None else None
            self.failure_archive.archive_failure(
                kind="image",
                stage=stage,
                request=job.request,
                error=exc,
                preview=preview,
                extra={
                    "job_id": str(job.id),
                    "job_state": job.state.value,
                    "checkpoint_id": checkpoint_id,
                },
            )
        except Exception:
            trace_exception_safe("generation.failure_archive", exc, job_id=str(job.id))

    def list_checkpoints(self):
        return self.backend.list_checkpoints()

    def refresh_checkpoint_catalog(self):
        self.refresh_model_library()
        return self.backend.list_checkpoints()

    def refresh_model_library(self) -> tuple[int, int]:
        from aiwf.infrastructure.model_inventory import invalidate_model_inventory_cache

        invalidate_model_inventory_cache()
        for invalidate_name in (
            "invalidate_checkpoints",
            "invalidate_loras",
            "invalidate_embeddings",
            "invalidate_vaes",
        ):
            invalidate = getattr(self.backend, invalidate_name, None)
            if callable(invalidate):
                invalidate()
        checkpoints = self.backend.list_checkpoints()
        loras = self.backend.list_loras()
        return len(checkpoints), len(loras)

    def list_embeddings(self):
        return self.backend.list_embeddings()

    def refresh_embedding_catalog(self):
        invalidate = getattr(self.backend, "invalidate_embeddings", None)
        if callable(invalidate):
            invalidate()
        return self.backend.list_embeddings()

    def refresh_vae_catalog(self):
        invalidate = getattr(self.backend, "invalidate_vaes", None)
        if callable(invalidate):
            invalidate()
        return self.backend.list_vaes()

    def list_samplers(self):
        return self.backend.list_samplers()

    def list_loras(self):
        return self.backend.list_loras()

    def list_vaes(self):
        return self.backend.list_vaes()

    def list_flux_text_encoders(self):
        lister = getattr(self.backend, "list_flux_text_encoders", None)
        return lister() if callable(lister) else []

    def set_flux_text_encoder(self, path: str | None) -> None:
        setter = getattr(self.backend, "set_flux_text_encoder", None)
        if callable(setter):
            setter(path)

    def resolve_checkpoint(self, checkpoint_id: str | None = None):
        return self.backend.resolve_checkpoint(checkpoint_id)

    def remember_checkpoint_selection(self, checkpoint_id: str | None = None):
        checkpoint = self.backend.resolve_checkpoint(checkpoint_id)
        self._persist_last_checkpoint(checkpoint.id)
        return checkpoint

    def load_checkpoint(self, checkpoint_id: str | None = None):
        tenant_job_id = f"image_load_{threading.get_ident()}"
        if self.supervisor is not None:
            switch = self.supervisor.request_switch(
                EngineSwitchRequest(
                    target=EngineTenant.IMAGE,
                    reason="Load image checkpoint",
                    job_id=tenant_job_id,
                )
            )
            if not switch.ok:
                raise RuntimeError(f"GPU busy: {switch.message}")
        try:
            checkpoint = self.backend.load_checkpoint(checkpoint_id)
            self._persist_last_checkpoint(checkpoint.id)
            return checkpoint
        finally:
            if self.supervisor is not None:
                self.supervisor.request_switch(
                    EngineSwitchRequest(
                        target=EngineTenant.IDLE,
                        reason="Image checkpoint load complete",
                        job_id=tenant_job_id,
                    )
                )

    def _start_stall_watchdog(self, job_id, tenant_job_id: str) -> threading.Event:
        """Guard against an image job that goes silent forever.

        Image generation runs cooperatively on an in-process thread with no hard
        kill switch — unlike the subprocess-based video/training engines, which
        have a heartbeat-timeout watchdog in EngineSupervisor that can SIGTERM/
        SIGKILL a stuck worker. A blocking call inside diffusers/accelerate (e.g.
        moving a large text encoder during CPU offload) cannot be interrupted from
        another thread in Python, so we cannot kill it the same way. What we *can*
        do is stop it from blocking the rest of the app forever: if a job reports
        no progress at all for STALL_TIMEOUT_SECONDS, force-fail the job record and
        force-release the GPU tenant lock so other jobs (image or video) can run.
        The original thread may still be alive in the background after this; it is
        harmless once it loses the tenant lock and its result is discarded (see
        JobQueue.run_next's "late return after force fail" handling).

        Returns a stop Event the caller must set() once the job finishes normally,
        so the watchdog thread exits instead of polling forever.
        """
        stop_event = threading.Event()

        def _watch() -> None:
            while not stop_event.wait(_STALL_POLL_INTERVAL_SECONDS):
                elapsed = self.queue.seconds_since_progress(job_id)
                if elapsed is None or elapsed < STALL_TIMEOUT_SECONDS:
                    continue
                reason = (
                    f"Generation stalled — no progress for {elapsed:.0f}s "
                    "(likely stuck in prompt encoding/model loading with no "
                    "cancellation point reachable). Forcing this job to fail so "
                    "the GPU is not blocked indefinitely."
                )
                trace_safe(
                    "generation.stall_watchdog_triggered",
                    reason,
                    job_id=str(job_id),
                    elapsed_seconds=elapsed,
                )
                failed = self.queue.force_fail_stalled(job_id, reason)
                if failed and self.supervisor is not None:
                    self.supervisor.request_switch(
                        EngineSwitchRequest(
                            target=EngineTenant.IDLE,
                            reason=f"Force-released stalled image job {job_id}",
                            job_id=tenant_job_id,
                        )
                    )
                return

        threading.Thread(target=_watch, daemon=True, name=f"stall-watchdog-{job_id}").start()
        return stop_event

    def _loading_model_message(self, checkpoint) -> str:
        title = getattr(checkpoint, "title", None) or getattr(checkpoint, "id", None) or "selected model"
        warm = getattr(self.backend, "is_checkpoint_warm", None)
        if callable(warm) and warm(getattr(checkpoint, "id", None)) is True:
            return f"Using warm model: {title}"
        return f"Loading image model: {title}"

    def submit(
        self,
        request: GenerationRequest,
        init_images: list[Image.Image] | None = None,
        mask_images: list[Image.Image] | None = None,
        control_images: list[Image.Image] | None = None,
        image_postprocess=None,
    ) -> JobRecord:
        record = JobRecord(request=request)
        self.queue.enqueue(record)

        def worker(job: JobRecord) -> GenerationResult:
            tenant_job_id = str(job.id)
            tenant_acquired = False
            active = None
            optimization_plan: OptimizationPlan | None = None
            latest_preview: Image.Image | None = None
            watchdog_stop = self._start_stall_watchdog(job.id, tenant_job_id)
            try:
                if self.supervisor is not None:
                    # Image jobs are GPU tenants; postprocessors borrow the same
                    # tenant below so they do not race a video/training handoff.
                    switch = self.supervisor.request_switch(
                        EngineSwitchRequest(
                            target=EngineTenant.IMAGE,
                            reason=f"Image generation {job.id}",
                            job_id=tenant_job_id,
                        )
                    )
                    if not switch.ok:
                        raise RuntimeError(f"GPU busy: {switch.message}")
                    tenant_acquired = True
                job.request = self._resolve_prompts(job.request)
                job.request = self._apply_default_negative(job.request)
                job.request = self._apply_generation_settings(job.request)
                self.events.publish(BeforeGenerate(job.id, job.request))
                active = self.backend.resolve_checkpoint(job.request.checkpoint_id)
                self._persist_last_checkpoint(active.id)
                job.request = self._guard_distilled_cfg(job.request, active)
                nonlocal init_images, mask_images
                job.request, init_images, mask_images = self._route_inpaint_checkpoint_for_txt2img(
                    job.request, active, init_images, mask_images
                )
                optimization_plan = self._resolve_optimization_plan(job.request, active)

                def on_progress(
                    step: int,
                    total: int,
                    message: str,
                    preview: Image.Image | None = None,
                    completed_batch: list[Image.Image] | None = None,
                    batch_seeds: list[int] | None = None,
                ) -> None:
                    nonlocal latest_preview
                    if preview is not None:
                        latest_preview = preview
                    self.queue.update_progress(job.id, step, total, message, preview if preview is not None else latest_preview)
                    self.events.publish(JobProgressed(job.id, step, total, message))
                    _print_generation_progress(job.id, step, total, message)
                    logger.info(
                        "Generation progress: job=%s step=%s/%s message=%s",
                        job.id,
                        step,
                        total,
                        message,
                    )

                on_progress(0, max(1, int(job.request.steps)), self._loading_model_message(active))
                _gen_t0 = time.perf_counter()
                preview_every = self.settings.live_preview_interval()
                if job.request.save_interrupted and preview_every == 0:
                    preview_every = 1
                try:
                    result = self.backend.generate(
                        job.request,
                        init_images=init_images,
                        mask_images=mask_images,
                        control_images=control_images,
                        on_progress=on_progress,
                        should_cancel=lambda: self.queue.should_cancel(job.id),
                        preview_every_n_steps=preview_every,
                    )
                except GenerationCancelledError:
                    self._save_cancelled_preview(job, job.request, active, optimization_plan, latest_preview)
                    raise
                result.elapsed_seconds = time.perf_counter() - _gen_t0
                self._persist_model_settings(active.id, job.request)
                trace_model_throughput(
                    kind=str(job.request.mode.value),
                    model_id=getattr(active, "id", None),
                    model_name=getattr(active, "title", None),
                    app_version=__version__,
                    elapsed_seconds=result.elapsed_seconds,
                    units=max(1, int(job.request.steps)),
                    units_label="steps",
                    image_count=len(result.images),
                    batch_size=int(job.request.batch_size),
                )

                if self.queue.is_abandoned(job.id):
                    # A stall watchdog already force-failed this job and
                    # released the GPU tenant lock while we were stuck inside
                    # the blocking generate() call. The lock is gone, so any
                    # post-processing here would fail trying to "borrow" a
                    # tenant we no longer own. Discard quietly instead of
                    # crashing on the way out.
                    trace_safe(
                        "generation.late_return_discarded",
                        "Worker finished after being force-failed for stalling; discarding result",
                        job_id=str(job.id),
                    )
                    return result

                if image_postprocess is not None:
                    if self.supervisor is None:
                        result.images = [image_postprocess(img) for img in result.images]
                    else:
                        with self.supervisor.borrow_active_tenant(EngineTenant.IMAGE, job_id=tenant_job_id):
                            result.images = [image_postprocess(img) for img in result.images]

                if self.settings.save_images and job.request.save_images:
                    self._save_generation_result(result, job.request, active, optimization_plan)

                self.events.publish(AfterGenerate(job.id, result))
                self._write_generation_log(
                    job=job,
                    result=result,
                    checkpoint=active,
                    optimization_plan=optimization_plan,
                )
                return result
            except GenerationCancelledError as exc:
                self._archive_failed_generation(
                    job,
                    exc,
                    preview=latest_preview,
                    checkpoint=active,
                    stage="image_cancelled",
                )
                raise
            except Exception as exc:
                self._archive_failed_generation(
                    job,
                    exc,
                    preview=latest_preview,
                    checkpoint=active,
                    stage="image_generation",
                )
                raise
            finally:
                watchdog_stop.set()
                if self.supervisor is not None and tenant_acquired and not self.queue.is_abandoned(job.id):
                    self.supervisor.request_switch(
                        EngineSwitchRequest(
                            target=EngineTenant.IDLE,
                            reason=f"Image generation {job.id} complete",
                            job_id=tenant_job_id,
                        )
                    )

        self.queue.run_next(worker, block=True)
        finished = self.queue.get(record.id)
        assert finished is not None
        return finished

    def submit_streaming(
        self,
        request: GenerationRequest,
        init_images: list[Image.Image] | None = None,
        mask_images: list[Image.Image] | None = None,
        control_images: list[Image.Image] | None = None,
        image_postprocess=None,
    ) -> Iterator[
        tuple[Literal["progress"], int, int, str, Image.Image | None]
        | tuple[Literal["done"], JobRecord]
    ]:
        """Run generation on a worker thread and yield progress for Gradio streaming."""
        record = JobRecord(request=request)
        self.queue.enqueue(record)
        trace_safe(
            "generation.submit_streaming",
            "Streaming job enqueued",
            job_id=str(record.id),
            mode=request.mode.value,
            checkpoint_id=request.checkpoint_id,
        )
        progress_q: queue.Queue = queue.Queue()

        def worker(job: JobRecord) -> GenerationResult:
            tenant_job_id = str(job.id)
            tenant_acquired = False
            active = None
            optimization_plan: OptimizationPlan | None = None
            latest_preview: Image.Image | None = None
            watchdog_stop = self._start_stall_watchdog(job.id, tenant_job_id)
            try:
                if self.supervisor is not None:
                    switch = self.supervisor.request_switch(
                        EngineSwitchRequest(
                            target=EngineTenant.IMAGE,
                            reason=f"Image generation {job.id}",
                            job_id=tenant_job_id,
                        )
                    )
                    if not switch.ok:
                        raise RuntimeError(f"GPU busy: {switch.message}")
                    tenant_acquired = True
                job.request = self._resolve_prompts(job.request)
                job.request = self._apply_default_negative(job.request)
                job.request = self._apply_generation_settings(job.request)
                self.events.publish(BeforeGenerate(job.id, job.request))
                active = self.backend.resolve_checkpoint(job.request.checkpoint_id)
                self._persist_last_checkpoint(active.id)
                job.request = self._guard_distilled_cfg(job.request, active)
                nonlocal init_images, mask_images
                job.request, init_images, mask_images = self._route_inpaint_checkpoint_for_txt2img(
                    job.request, active, init_images, mask_images
                )
                optimization_plan = self._resolve_optimization_plan(job.request, active)
                preview_every = self.settings.live_preview_interval()
                if job.request.save_interrupted and preview_every == 0:
                    preview_every = 1

                def on_progress(
                    step: int,
                    total: int,
                    message: str,
                    preview: Image.Image | None = None,
                    completed_batch: list[Image.Image] | None = None,
                    batch_seeds: list[int] | None = None,
                ) -> None:
                    nonlocal latest_preview
                    if preview is not None:
                        latest_preview = preview
                    self.queue.update_progress(job.id, step, total, message, preview if preview is not None else latest_preview)
                    self.events.publish(JobProgressed(job.id, step, total, message))
                    _print_generation_progress(job.id, step, total, message)
                    logger.info(
                        "Generation progress: job=%s step=%s/%s message=%s",
                        job.id,
                        step,
                        total,
                        message,
                    )
                    if completed_batch:
                        progress_q.put(("batch_images", list(completed_batch), list(batch_seeds or [])))
                    else:
                        progress_q.put(("progress", step, total, message, preview))

                on_progress(0, max(1, int(job.request.steps)), self._loading_model_message(active))
                _gen_t0 = time.perf_counter()
                try:
                    result = self.backend.generate(
                        job.request,
                        init_images=init_images,
                        mask_images=mask_images,
                        control_images=control_images,
                        on_progress=on_progress,
                        should_cancel=lambda: self.queue.should_cancel(job.id),
                        preview_every_n_steps=preview_every,
                    )
                except GenerationCancelledError:
                    self._save_cancelled_preview(job, job.request, active, optimization_plan, latest_preview)
                    raise
                result.elapsed_seconds = time.perf_counter() - _gen_t0
                self._persist_model_settings(active.id, job.request)
                trace_model_throughput(
                    kind=str(job.request.mode.value),
                    model_id=getattr(active, "id", None),
                    model_name=getattr(active, "title", None),
                    app_version=__version__,
                    elapsed_seconds=result.elapsed_seconds,
                    units=max(1, int(job.request.steps)),
                    units_label="steps",
                    image_count=len(result.images),
                    batch_size=int(job.request.batch_size),
                )

                if self.queue.is_abandoned(job.id):
                    # Stall watchdog already force-failed this job and
                    # released the GPU tenant lock while generate() was
                    # stuck. Don't try to borrow a tenant we no longer own —
                    # discard the late result quietly.
                    trace_safe(
                        "generation.late_return_discarded",
                        "Worker finished after being force-failed for stalling; discarding result",
                        job_id=str(job.id),
                    )
                    return result

                if image_postprocess is not None:
                    if self.supervisor is None:
                        result.images = [image_postprocess(img) for img in result.images]
                    else:
                        with self.supervisor.borrow_active_tenant(EngineTenant.IMAGE, job_id=tenant_job_id):
                            result.images = [image_postprocess(img) for img in result.images]

                if self.settings.save_images and job.request.save_images:
                    total = max(1, int(job.request.steps))
                    on_progress(total, total, "Saving output", None)
                    self._save_generation_result(result, job.request, active, optimization_plan)

                self.events.publish(AfterGenerate(job.id, result))
                self._write_generation_log(
                    job=job,
                    result=result,
                    checkpoint=active,
                    optimization_plan=optimization_plan,
                )
                return result
            except GenerationCancelledError as exc:
                self._archive_failed_generation(
                    job,
                    exc,
                    preview=latest_preview,
                    checkpoint=active,
                    stage="image_cancelled",
                )
                raise
            except Exception as exc:
                self._archive_failed_generation(
                    job,
                    exc,
                    preview=latest_preview,
                    checkpoint=active,
                    stage="image_generation",
                )
                raise
            finally:
                watchdog_stop.set()
                if self.supervisor is not None and tenant_acquired and not self.queue.is_abandoned(job.id):
                    self.supervisor.request_switch(
                        EngineSwitchRequest(
                            target=EngineTenant.IDLE,
                            reason=f"Image generation {job.id} complete",
                            job_id=tenant_job_id,
                        )
                    )

        done = threading.Event()

        def _run_worker() -> None:
            try:
                self.queue.run_next(worker, block=True)
            except Exception as exc:
                trace_exception_safe(
                    "generation.streaming_worker",
                    exc,
                    job_id=str(record.id),
                    mode=request.mode.value,
                )
            finally:
                done.set()

        thread = threading.Thread(target=_run_worker, daemon=True)
        thread.start()

        while not done.is_set() or not progress_q.empty():
            try:
                item = progress_q.get(timeout=0.15)
                yield item
                continue
            except queue.Empty:
                pass
            if not done.is_set():
                # The worker thread is the only thing that flips `done`, and it
                # may never return if it is stuck inside an uncancellable blocking
                # call (see _start_stall_watchdog). If the stall watchdog already
                # force-failed this job, stop waiting on the dead thread — the
                # job is finished from the queue's point of view even though the
                # underlying thread may still be alive in the background.
                current = self.queue.get(record.id)
                if current is not None and current.state != JobState.RUNNING:
                    break

        # Don't block the UI on a thread that may be permanently stuck; the
        # watchdog (if it fired) has already released the job's slot and GPU
        # tenant lock, so there is nothing further to wait for here.
        thread.join(timeout=1.0)

        finished = self.queue.get(record.id)
        if finished is not None:
            yield ("done", finished)

    def interrupt(self, job_id=None) -> None:
        self.queue.request_cancel(job_id)

    def get_job(self, job_id):
        return self.queue.get(job_id)

    def active_job(self):
        return self.queue.active()

    def pending_count(self) -> int:
        return self.queue.pending_count()

    def recent_jobs(self, limit: int = 20):
        return self.queue.list_recent(limit)
