from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image

from aiwf.core.domain.generation import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    JobRecord,
    JobState,
)
from aiwf.core.events.bus import EventBus
from aiwf.core.events.types import AfterGenerate, BeforeGenerate, JobProgressed
from aiwf.core.interfaces.backend import InferenceBackend
from aiwf.core.interfaces.storage import ImageStore
from aiwf.core.config.settings import UserSettings
from aiwf.services.metadata import MetadataService
from aiwf.dev.diagnostics import trace_exception_safe, trace_safe
from aiwf.services.queue import JobQueue

if TYPE_CHECKING:
    from aiwf.services.prompt_processor import PromptProcessorService


class GenerationService:
    """Application layer — UI and API talk here, never to torch/diffusers directly."""

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
    ) -> None:
        self.backend = backend
        self.store = store
        self.metadata = metadata
        self.queue = queue
        self.events = events
        self.settings = settings
        self.prompts = prompts
        self._settings_path = settings_path

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

    def load_checkpoint(self, checkpoint_id: str | None = None):
        checkpoint = self.backend.load_checkpoint(checkpoint_id)
        self._persist_last_checkpoint(checkpoint.id)
        return checkpoint

    def submit(
        self,
        request: GenerationRequest,
        init_images: list[Image.Image] | None = None,
        mask_images: list[Image.Image] | None = None,
        control_images: list[Image.Image] | None = None,
    ) -> JobRecord:
        record = JobRecord(request=request)
        self.queue.enqueue(record)

        def worker(job: JobRecord) -> GenerationResult:
            job.request = self._resolve_prompts(job.request)
            self.events.publish(BeforeGenerate(job.id, job.request))
            active = self.backend.resolve_checkpoint(job.request.checkpoint_id)
            self._persist_last_checkpoint(active.id)

            def on_progress(
                step: int,
                total: int,
                message: str,
                preview: Image.Image | None = None,
            ) -> None:
                self.queue.update_progress(job.id, step, total, message, preview)
                self.events.publish(JobProgressed(job.id, step, total, message))

            result = self.backend.generate(
                job.request,
                init_images=init_images,
                mask_images=mask_images,
                control_images=control_images,
                on_progress=on_progress,
                should_cancel=lambda: self.queue.should_cancel(job.id),
                preview_every_n_steps=0,
            )

            if self.settings.save_images and job.request.save_images:
                subdir = {
                    GenerationMode.TXT2IMG: self.settings.txt2img_output_subdir,
                    GenerationMode.IMG2IMG: self.settings.img2img_output_subdir,
                    GenerationMode.INPAINT: self.settings.inpaint_output_subdir,
                }[job.request.mode]
                checkpoint = self.backend.resolve_checkpoint(job.request.checkpoint_id)
                artifacts = []
                for index, image in enumerate(result.images):
                    infotext = result.infotexts[index] if index < len(result.infotexts) else ""
                    if self.settings.embed_metadata or job.request.tags:
                        image = self.metadata.embed(image, infotext, tags=job.request.tags)
                    artifact = self.store.save(image, infotext, subdir)
                    artifacts.append(artifact)
                result.artifacts = artifacts

            self.events.publish(AfterGenerate(job.id, result))
            return result

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
            job.request = self._resolve_prompts(job.request)
            self.events.publish(BeforeGenerate(job.id, job.request))
            active = self.backend.resolve_checkpoint(job.request.checkpoint_id)
            self._persist_last_checkpoint(active.id)
            preview_every = self.settings.live_preview_interval()

            def on_progress(
                step: int,
                total: int,
                message: str,
                preview: Image.Image | None = None,
            ) -> None:
                self.queue.update_progress(job.id, step, total, message, preview)
                self.events.publish(JobProgressed(job.id, step, total, message))
                progress_q.put(("progress", step, total, message, preview))

            result = self.backend.generate(
                job.request,
                init_images=init_images,
                mask_images=mask_images,
                control_images=control_images,
                on_progress=on_progress,
                should_cancel=lambda: self.queue.should_cancel(job.id),
                preview_every_n_steps=preview_every,
            )

            if self.settings.save_images and job.request.save_images:
                total = max(1, int(job.request.steps))
                on_progress(total, total, "Saving output", None)
                subdir = {
                    GenerationMode.TXT2IMG: self.settings.txt2img_output_subdir,
                    GenerationMode.IMG2IMG: self.settings.img2img_output_subdir,
                    GenerationMode.INPAINT: self.settings.inpaint_output_subdir,
                }[job.request.mode]
                artifacts = []
                for index, image in enumerate(result.images):
                    infotext = result.infotexts[index] if index < len(result.infotexts) else ""
                    if self.settings.embed_metadata or job.request.tags:
                        image = self.metadata.embed(image, infotext, tags=job.request.tags)
                    artifact = self.store.save(image, infotext, subdir)
                    artifacts.append(artifact)
                result.artifacts = artifacts

            self.events.publish(AfterGenerate(job.id, result))
            return result

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
