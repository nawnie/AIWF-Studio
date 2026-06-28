from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.ltx import (
    LTX_DISTILLED_CHECKPOINT,
    LTX_FULL_CHECKPOINT,
    LTX_GEMMA_REPO,
    LTX_PIPELINE_DISTILLED,
    LTX_PIPELINE_ONE_STAGE,
    LTX_SPATIAL_UPSCALER_X2,
    LtxVideoRequest,
    LtxVideoResult,
)
from aiwf.services.engine_supervisor import EngineSupervisor
from aiwf.infrastructure.video.processing import VideoProcessor
from aiwf.services.process_supervisor import ProcessSupervisor, get_process_supervisor
from aiwf.services.worker_tenant import WorkerTenantRegistry

logger = logging.getLogger(__name__)


class LtxUnavailable(RuntimeError):
    pass


class LtxService:
    def __init__(
        self,
        flags: RuntimeFlags | None = None,
        settings: UserSettings | None = None,
        *,
        registry: WorkerTenantRegistry | None = None,
        supervisor: EngineSupervisor | None = None,
        process_supervisor: ProcessSupervisor | None = None,
    ) -> None:
        self.flags = flags or RuntimeFlags()
        self.settings = settings or UserSettings()
        self.registry = registry or WorkerTenantRegistry(self.flags.data_dir)
        self.supervisor = supervisor
        self.process_supervisor = process_supervisor or get_process_supervisor()

    def models_root(self) -> Path:
        return self.flags.resolved_models_dir() / "ltx"

    def default_checkpoint_path(self, pipeline: str = LTX_PIPELINE_DISTILLED) -> Path:
        name = LTX_DISTILLED_CHECKPOINT if pipeline == LTX_PIPELINE_DISTILLED else LTX_FULL_CHECKPOINT
        return self.models_root() / "checkpoints" / name

    def default_launch_pipeline(self) -> str:
        if self.default_checkpoint_path(LTX_PIPELINE_DISTILLED).is_file():
            return LTX_PIPELINE_DISTILLED
        if self.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE).is_file():
            return LTX_PIPELINE_ONE_STAGE
        return LTX_PIPELINE_DISTILLED

    def default_launch_request(self) -> LtxVideoRequest:
        return LtxVideoRequest(pipeline=self.default_launch_pipeline())

    def default_spatial_upsampler_path(self) -> Path:
        return self.models_root() / "upscalers" / LTX_SPATIAL_UPSCALER_X2

    def default_gemma_root(self) -> Path:
        return self.models_root() / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]

    def output_dir(self) -> Path:
        return self.flags.resolved_output_dir() / "ltx-videos"

    def status_markdown(self) -> str:
        try:
            status = self.registry.status("ltx")
        except KeyError:
            return "**LTX 2.3:** not registered."

        launch_pipeline = self.default_launch_pipeline()
        lines = [status.markdown_line()]
        lines.append(f"- Default launch pipeline: `{launch_pipeline}`")
        lines.append(f"- Default checkpoint: `{self.default_checkpoint_path(launch_pipeline)}`")
        lines.append(f"- Default upscaler: `{self.default_spatial_upsampler_path()}`")
        lines.append(f"- Default Gemma root: `{self.default_gemma_root()}`")
        return "\n".join(lines)

    def generate(self, request: LtxVideoRequest) -> LtxVideoResult:
        status = self.registry.status("ltx")
        if not status.ready:
            details = "; ".join(status.messages) if status.messages else "not ready"
            raise LtxUnavailable(
                "LTX 2.3 engine is not ready. Run `scripts/bootstrap_ltx.ps1 -Enable`, "
                f"then refresh Settings. Details: {details}"
            )

        normalized = self._resolve_request(request)
        self._validate_request_paths(normalized)

        job_id = f"ltx_{uuid4().hex[:8]}"
        request_path = self._write_worker_request(job_id, normalized)
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True",
            ),
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
        command = self.registry.build_command(
            "ltx",
            request_path,
            env=env,
            cwd=status.repo_dir or self.flags.data_dir,
        )

        events: list[dict] = []
        output_path = Path(normalized["output_path"])
        error_message = ""
        try:
            if self.supervisor is not None:
                with self.supervisor.tenant_session(
                    EngineTenant.VIDEO,
                    reason="LTX 2.3 video generation",
                    job_id=job_id,
                    allow_wait=False,
                ):
                    self._run_worker(job_id, command, events)
            else:
                self._run_worker(job_id, command, events)
        except Exception as exc:
            error_message = str(exc)
            logger.exception("LTX 2.3 generation failed")

        terminal_error = _last_error(events)
        if error_message or terminal_error:
            raise LtxUnavailable(terminal_error or error_message)
        if not output_path.is_file():
            raise LtxUnavailable(f"LTX worker finished but did not create output: {output_path}")
        has_audio = False
        try:
            has_audio = VideoProcessor().probe(output_path).has_audio
        except Exception:
            logger.debug("Could not probe LTX output audio stream", exc_info=True)

        return LtxVideoResult(
            output_path=str(output_path),
            message=f"LTX 2.3 video saved to {output_path.name}" + (" with native audio" if has_audio else ""),
            events=events,
            has_audio=has_audio,
            audio_mode="native",
        )

    def _resolve_request(self, request: LtxVideoRequest) -> dict:
        pipeline = request.pipeline
        if not str(request.checkpoint_path or "").strip() and pipeline == LTX_PIPELINE_DISTILLED:
            fallback_pipeline = self.default_launch_pipeline()
            if fallback_pipeline != pipeline:
                pipeline = fallback_pipeline
        checkpoint = _resolve_path(
            request.checkpoint_path,
            self.default_checkpoint_path(pipeline),
            self.flags.data_dir,
        )
        spatial_upsampler = _resolve_path(
            request.spatial_upsampler_path,
            self.default_spatial_upsampler_path(),
            self.flags.data_dir,
        )
        gemma_root = _resolve_path(
            request.gemma_root,
            self.default_gemma_root(),
            self.flags.data_dir,
        )
        source_image = _resolve_optional_path(request.source_image_path, self.flags.data_dir)
        seed = int(request.seed)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        self.output_dir().mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.output_dir() / f"ltx23-{stamp}-{uuid4().hex[:6]}.mp4"

        payload = request.model_dump()
        payload.update(
            {
                "pipeline": pipeline,
                "checkpoint_path": str(checkpoint),
                "spatial_upsampler_path": str(spatial_upsampler),
                "gemma_root": str(gemma_root),
                "source_image_path": str(source_image) if source_image is not None else None,
                "seed": seed,
                "output_path": str(output_path),
            }
        )
        return payload

    def _validate_request_paths(self, payload: dict) -> None:
        checkpoint = Path(str(payload.get("checkpoint_path") or ""))
        gemma_root = Path(str(payload.get("gemma_root") or ""))
        if not checkpoint.is_file():
            raise LtxUnavailable(f"LTX checkpoint missing: {checkpoint}")
        if not gemma_root.exists():
            raise LtxUnavailable(f"LTX Gemma text encoder folder missing: {gemma_root}")
        if payload.get("pipeline") == LTX_PIPELINE_DISTILLED:
            upsampler = Path(str(payload.get("spatial_upsampler_path") or ""))
            if not upsampler.is_file():
                raise LtxUnavailable(f"LTX spatial upscaler missing: {upsampler}")
        source = payload.get("source_image_path")
        if source and not Path(str(source)).is_file():
            raise LtxUnavailable(f"LTX source image missing: {source}")

    def _write_worker_request(self, job_id: str, payload: dict) -> Path:
        root = self.output_dir() / "requests"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{job_id}.json"
        worker_payload = {
            "_job_id": job_id,
            "_engine": "ltx",
            "_created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "generate",
            **payload,
        }
        path.write_text(json.dumps(worker_payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _run_worker(self, job_id: str, command, events: list[dict]) -> None:  # noqa: ANN001
        for line in self.process_supervisor.start(job_id, command, check=True):
            event = _parse_event(line)
            if event is not None:
                events.append(event)


def _resolve_path(raw: str | None, default: Path, root: Path) -> Path:
    text = str(raw or "").strip()
    path = Path(text) if text else default
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_optional_path(raw: str | None, root: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_event(line: str) -> dict | None:
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) and "kind" in event else None


def _last_error(events: list[dict]) -> str:
    for event in reversed(events):
        if event.get("kind") == "error":
            return str(event.get("message") or event.get("detail") or "LTX worker failed")
    return ""
