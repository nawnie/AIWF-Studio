from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
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
                    "model_id": unit.model_id,
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
                    "fp8": bool(getattr(flags, "fp8", False)),
                    "medvram": bool(getattr(flags, "medvram", False)),
                    "lowvram": bool(getattr(flags, "lowvram", False)),
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
        if getattr(self.settings, "sdxl_refiner_enabled", False) and not request.sdxl_refiner_enabled:
            updates.update(
                {
                    "sdxl_refiner_enabled": True,
                    "sdxl_refiner_checkpoint_id": getattr(self.settings, "sdxl_refiner_checkpoint_id", None),
                    "sdxl_refiner_steps": getattr(self.settings, "sdxl_refiner_steps", 10),
                    "sdxl_refiner_strength": getattr(self.settings, "sdxl_refiner_strength", 0.25),
                }
            )
        elif request.sdxl_refiner_enabled and not request.sdxl_refiner_checkpoint_id:
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
        for index, image in enumerate(result.images):
            infotext = result.infotexts[index] if index < len(result.infotexts) else ""
            infotext = self._enrich_saved_infotext(
                infotext,
                request,
                checkpoint,
                optimization_plan,
            )
            if index < len(result.infotexts):
                result.infotexts[index] = infotext
            seed = result.seeds[index] if index < len(result.seeds) else None
            if self.settings.embed_metadata or request.tags:
                image = self.metadata.embed(image, infotext, tags=request.tags)
            artifact = self.store.save(
                image, infotext, subdir, seed=seed, index=index, model_name=model_name
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
        invalidate = getattr(self.backend, "invalidate_checkpoints", None)
        if callable(invalidate):
            invalidate()
        return self.backend.list_checkpoints()

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

    @staticmethod
    def _loading_model_message(checkpoint) -> str:
        title = getattr(checkpoint, "title", None) or getattr(checkpoint, "id", None) or "selected model"
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
                optimization_plan = self._resolve_optimization_plan(job.request, active)

                def on_progress(
                    step: int,
                    total: int,
                    message: str,
                    preview: Image.Image | None = None,
                ) -> None:
                    nonlocal latest_preview
                    if preview is not None:
                        latest_preview = preview
                    self.queue.update_progress(job.id, step, total, message, preview)
                    self.events.publish(JobProgressed(job.id, step, total, message))

                on_progress(0, max(1, int(job.request.steps)), self._loading_model_message(active))
                _gen_t0 = time.perf_counter()
                preview_every = 1 if job.request.save_interrupted else 0
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
                if self.supervisor is not None and tenant_acquired:
                    self.supervisor.request_switch(
                        EngineSwitchRequest(
                            target=EngineTenant.IDLE,
                            reason=f"Image generation {job.id} complete",
                            job_id=tenant_job_id,
                        )
                    )

        self.queue.run_next(worker)
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
                optimization_plan = self._resolve_optimization_plan(job.request, active)
                preview_every = self.settings.live_preview_interval()
                if job.request.save_interrupted and preview_every == 0:
                    preview_every = 1

                def on_progress(
                    step: int,
                    total: int,
                    message: str,
                    preview: Image.Image | None = None,
                ) -> None:
                    nonlocal latest_preview
                    if preview is not None:
                        latest_preview = preview
                    self.queue.update_progress(job.id, step, total, message, preview)
                    self.events.publish(JobProgressed(job.id, step, total, message))
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
                if self.supervisor is not None and tenant_acquired:
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
                raise
            finally:
                done.set()

        thread = threading.Thread(target=_run_worker, daemon=True)
        thread.start()

        while not done.is_set() or not progress_q.empty():
            try:
                item = progress_q.get(timeout=0.15)
            except queue.Empty:
                continue
            yield item

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

    def recent_jobs(self, limit: int = 20):
        return self.queue.list_recent(limit)
