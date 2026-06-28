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
    LTX_DIFFUSERS_2B_CHECKPOINT,
    LTX_FULL_CHECKPOINT,
    LTX_GEMMA_BACKEND_GGUF,
    LTX_GEMMA_BACKEND_HF_SAFETENSORS,
    LTX_GEMMA_REPO,
    LTX_HERETIC_Q3_GGUF,
    LTX_PIPELINE_DIFFUSERS_2B,
    LTX_PIPELINE_DISTILLED,
    LTX_PIPELINE_ONE_STAGE,
    LTX_SPATIAL_UPSCALER_X2,
    LTX_T5_TOKENIZER,
    LTX_T5XXL_FP16,
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
        if pipeline == LTX_PIPELINE_DIFFUSERS_2B:
            name = LTX_DIFFUSERS_2B_CHECKPOINT
        elif pipeline == LTX_PIPELINE_DISTILLED:
            name = LTX_DISTILLED_CHECKPOINT
        else:
            name = LTX_FULL_CHECKPOINT
        return self.models_root() / "checkpoints" / name

    def default_t5_encoder_path(self) -> Path:
        primary = self.flags.resolved_models_dir() / "flux" / "Textencoder" / LTX_T5XXL_FP16
        if primary.is_file():
            return primary
        return self.flags.resolved_models_dir() / "Textencoder" / LTX_T5XXL_FP16

    def default_launch_pipeline(self) -> str:
        if self.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B).is_file() and self.default_t5_encoder_path().is_file():
            return LTX_PIPELINE_DIFFUSERS_2B
        one_stage = self.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
        if one_stage.is_file() and not ltx_checkpoint_openability_error(one_stage):
            return LTX_PIPELINE_ONE_STAGE
        distilled = self.default_checkpoint_path(LTX_PIPELINE_DISTILLED)
        if distilled.is_file() and not ltx_checkpoint_openability_error(distilled):
            return LTX_PIPELINE_DISTILLED
        return LTX_PIPELINE_DISTILLED

    def default_launch_request(self) -> LtxVideoRequest:
        return LtxVideoRequest(pipeline=self.default_launch_pipeline())

    def default_spatial_upsampler_path(self) -> Path:
        return self.models_root() / "upscalers" / LTX_SPATIAL_UPSCALER_X2

    def default_gemma_root(self) -> Path:
        return self.models_root() / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]

    def default_gemma_gguf_path(self) -> Path:
        return self.flags.resolved_models_dir() / "LLM" / "GGUF" / LTX_HERETIC_Q3_GGUF

    def output_dir(self) -> Path:
        return self.flags.resolved_output_dir() / "ltx-videos"

    def status_markdown(self) -> str:
        try:
            status = self.registry.status("ltx")
            lines = [status.markdown_line()]
        except KeyError:
            lines = ["**LTX 2.3:** not registered."]

        launch_pipeline = self.default_launch_pipeline()
        lines.append(f"- Default launch pipeline: `{launch_pipeline}`")
        lines.append(f"- Default checkpoint: `{self.default_checkpoint_path(launch_pipeline)}`")
        lines.append(f"- Default upscaler: `{self.default_spatial_upsampler_path()}`")
        lines.append(f"- Default Gemma root: `{self.default_gemma_root()}`")
        lines.append(f"- Default Heretic GGUF: `{self.default_gemma_gguf_path()}`")
        lines.append(f"- Default LTX 2B T5XXL: `{self.default_t5_encoder_path()}`")
        return "\n".join(lines)

    def generate(self, request: LtxVideoRequest) -> LtxVideoResult:
        normalized = self._resolve_request(request)
        if normalized.get("pipeline") == LTX_PIPELINE_DIFFUSERS_2B:
            return self._generate_diffusers_2b(normalized)

        status = self.registry.status("ltx")
        if not status.ready:
            details = "; ".join(status.messages) if status.messages else "not ready"
            raise LtxUnavailable(
                "LTX 2.3 engine is not ready. Run `scripts/bootstrap_ltx.ps1 -Enable`, "
                f"then refresh Settings. Details: {details}"
            )

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

    def _generate_diffusers_2b(self, payload: dict) -> LtxVideoResult:
        self._validate_request_paths(payload)
        source = payload.get("source_image_path")
        if source:
            raise LtxUnavailable("The local LTX 2B Diffusers route is text-to-video only; clear the source image.")

        from aiwf.services.ltx_diffusers import run_ltx2b_diffusers

        job_id = f"ltx2b_{uuid4().hex[:8]}"
        output_path = Path(str(payload["output_path"]))

        def _run():
            return run_ltx2b_diffusers(
                checkpoint=Path(str(payload["checkpoint_path"])),
                t5_weights=Path(str(payload["t5_encoder_path"])),
                tokenizer_id=str(payload.get("t5_tokenizer") or LTX_T5_TOKENIZER),
                output=output_path,
                prompt=str(payload.get("prompt") or ""),
                negative_prompt=str(payload.get("negative_prompt") or ""),
                width=int(payload.get("width") or 128),
                height=int(payload.get("height") or 128),
                frames=int(payload.get("num_frames") or 9),
                fps=int(round(float(payload.get("fps") or 8))),
                steps=int(payload.get("steps") or 1),
                seed=int(payload.get("seed") or 0),
            )

        try:
            if self.supervisor is not None:
                with self.supervisor.tenant_session(
                    EngineTenant.VIDEO,
                    reason="LTX 2B Diffusers video generation",
                    job_id=job_id,
                    allow_wait=False,
                ):
                    result = _run()
            else:
                result = _run()
        except Exception as exc:
            logger.exception("LTX 2B Diffusers generation failed")
            raise LtxUnavailable(f"LTX 2B Diffusers generation failed: {exc}") from exc

        return LtxVideoResult(
            output_path=str(result.output_path),
            message=(
                f"LTX 2B Diffusers video saved to {result.output_path.name} "
                f"({result.width}x{result.height}, {result.frame_count} frames, {result.fps} fps"
                f", {'cache hit' if result.cache_hit else 'loaded pipeline'})"
            ),
            events=[
                {
                    "kind": "complete",
                    "message": "LTX 2B Diffusers generation complete",
                    "bytes": result.bytes,
                    "cache_hit": result.cache_hit,
                }
            ],
            has_audio=False,
            audio_mode="none",
        )

    def probe_gemma_gguf(self, request: LtxVideoRequest | None = None) -> list[dict]:
        """Run the isolated worker's native-GGUF text-encoder feasibility probe.

        This does not run video generation or dequantize weights. It verifies
        path wiring and reports whether the LTX worker has a backend that can
        return the full Gemma hidden-state tuple LTX requires.
        """

        status = self.registry.status("ltx")
        if not status.ready:
            details = "; ".join(status.messages) if status.messages else "not ready"
            raise LtxUnavailable(
                "LTX 2.3 engine is not ready. Run `scripts/bootstrap_ltx.ps1 -Enable`, "
                f"then refresh Settings. Details: {details}"
            )

        base_request = request or LtxVideoRequest()
        if base_request.gemma_backend != LTX_GEMMA_BACKEND_GGUF:
            base_request = base_request.model_copy(update={"gemma_backend": LTX_GEMMA_BACKEND_GGUF})
        normalized = self._resolve_request(base_request)
        self._validate_gemma_paths(normalized)

        job_id = f"ltx_probe_{uuid4().hex[:8]}"
        normalized["mode"] = "probe_gemma_gguf"
        request_path = self._write_worker_request(job_id, normalized)
        command = self.registry.build_command(
            "ltx",
            request_path,
            env={"HF_HUB_DISABLE_PROGRESS_BARS": "1"},
            cwd=status.repo_dir or self.flags.data_dir,
        )

        events: list[dict] = []
        error_message = ""
        try:
            self._run_worker(job_id, command, events)
        except Exception as exc:
            error_message = str(exc)
            logger.info("LTX Gemma GGUF probe failed: %s", exc)

        terminal_error = _last_error(events)
        if error_message or terminal_error:
            raise LtxUnavailable(terminal_error or error_message)
        return events

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
        t5_encoder_path = _resolve_path(
            request.t5_encoder_path,
            self.default_t5_encoder_path(),
            self.flags.data_dir,
        )
        spatial_upsampler = _resolve_path(
            request.spatial_upsampler_path,
            self.default_spatial_upsampler_path(),
            self.flags.data_dir,
        )
        gemma_root = _resolve_path(
            _gemma_root_text(request),
            self.default_gemma_root(),
            self.flags.data_dir,
        )
        gemma_backend = str(request.gemma_backend or LTX_GEMMA_BACKEND_HF_SAFETENSORS).strip().lower()
        gemma_gguf_raw = _gemma_gguf_text(request) if gemma_backend == LTX_GEMMA_BACKEND_GGUF else ""
        if str(request.gemma_root or "").strip().lower().endswith(".gguf"):
            gemma_backend = LTX_GEMMA_BACKEND_GGUF
            gemma_gguf_raw = _gemma_gguf_text(request)
        gemma_gguf_path = _resolve_optional_path(gemma_gguf_raw, self.flags.data_dir)
        if gemma_backend == LTX_GEMMA_BACKEND_GGUF and gemma_gguf_path is None:
            gemma_gguf_path = self.default_gemma_gguf_path().resolve()
        source_image = _resolve_optional_path(request.source_image_path, self.flags.data_dir)
        seed = int(request.seed)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        self.output_dir().mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = "ltx2b" if pipeline == LTX_PIPELINE_DIFFUSERS_2B else "ltx23"
        output_path = self.output_dir() / f"{prefix}-{stamp}-{uuid4().hex[:6]}.mp4"

        payload = request.model_dump()
        payload.update(
            {
                "pipeline": pipeline,
                "checkpoint_path": str(checkpoint),
                "t5_encoder_path": str(t5_encoder_path),
                "t5_tokenizer": str(request.t5_tokenizer or LTX_T5_TOKENIZER),
                "spatial_upsampler_path": str(spatial_upsampler),
                "gemma_root": str(gemma_root),
                "gemma_backend": gemma_backend,
                "gemma_gguf_path": str(gemma_gguf_path) if gemma_gguf_path is not None else "",
                "source_image_path": str(source_image) if source_image is not None else None,
                "seed": seed,
                "output_path": str(output_path),
            }
        )
        return payload

    def _validate_request_paths(self, payload: dict) -> None:
        checkpoint = Path(str(payload.get("checkpoint_path") or ""))
        if not checkpoint.is_file():
            raise LtxUnavailable(f"LTX checkpoint missing: {checkpoint}")
        if payload.get("pipeline") == LTX_PIPELINE_DIFFUSERS_2B:
            t5_encoder = Path(str(payload.get("t5_encoder_path") or ""))
            if not t5_encoder.is_file():
                raise LtxUnavailable(f"LTX 2B T5XXL text encoder missing: {t5_encoder}")
            source = payload.get("source_image_path")
            if source and not Path(str(source)).is_file():
                raise LtxUnavailable(f"LTX source image missing: {source}")
            return
        openability_error = ltx_checkpoint_openability_error(checkpoint)
        if openability_error:
            raise LtxUnavailable(openability_error)
        self._validate_gemma_paths(payload)
        if payload.get("gemma_backend") == LTX_GEMMA_BACKEND_GGUF:
            raise LtxUnavailable(_native_gemma_gguf_blocker(payload))
        if payload.get("pipeline") == LTX_PIPELINE_DISTILLED:
            upsampler = Path(str(payload.get("spatial_upsampler_path") or ""))
            if not upsampler.is_file():
                raise LtxUnavailable(f"LTX spatial upscaler missing: {upsampler}")
        source = payload.get("source_image_path")
        if source and not Path(str(source)).is_file():
            raise LtxUnavailable(f"LTX source image missing: {source}")

    def _validate_gemma_paths(self, payload: dict) -> None:
        gemma_root = Path(str(payload.get("gemma_root") or ""))
        if not gemma_root.exists():
            raise LtxUnavailable(f"LTX Gemma tokenizer/processor folder missing: {gemma_root}")
        if payload.get("gemma_backend") != LTX_GEMMA_BACKEND_GGUF:
            return
        gguf = Path(str(payload.get("gemma_gguf_path") or ""))
        if not gguf.is_file():
            raise LtxUnavailable(f"LTX Gemma GGUF file missing: {gguf}")
        if gguf.suffix.lower() != ".gguf":
            raise LtxUnavailable(f"LTX Gemma GGUF path must end in .gguf: {gguf}")

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


def ltx_checkpoint_openability_error(path: Path) -> str:
    """Return a user-facing error when a native LTX checkpoint cannot be opened.

    This stays intentionally shallow: it only opens the safetensors container
    and lists keys. It does not materialize tensor payloads or run inference.
    On Windows, very large LTX-2.3 checkpoints can fail here with pagefile
    mmap errors before the upstream worker gets a chance to fail harder.
    """

    if path.suffix.lower() != ".safetensors":
        return ""
    try:
        if path.stat().st_size < 1024 * 1024:
            return ""
    except OSError as exc:
        return f"LTX checkpoint could not be inspected: {path}. {exc}"

    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            len(list(f.keys()))
    except OSError as exc:
        return (
            f"LTX checkpoint is not safely loadable in this Windows runtime: {path}. "
            f"{exc}. Increase the Windows paging file or use the working LTX 2B Diffusers route."
        )
    except Exception as exc:
        return f"LTX checkpoint failed a shallow safetensors open check: {path}. {type(exc).__name__}: {exc}"
    return ""


def _gemma_root_text(request: LtxVideoRequest) -> str:
    text = str(request.gemma_root or "").strip()
    if text.lower().endswith(".gguf"):
        return ""
    return text


def _gemma_gguf_text(request: LtxVideoRequest) -> str:
    text = str(request.gemma_gguf_path or "").strip()
    if text:
        return text
    root_text = str(request.gemma_root or "").strip()
    if root_text.lower().endswith(".gguf"):
        return root_text
    return ""


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
    status_tail = _interesting_status_tail(events)
    for event in reversed(events):
        if event.get("kind") == "error":
            detail = str(event.get("detail") or event.get("message") or "LTX worker failed")
            if status_tail and status_tail not in detail:
                detail = f"{detail}\n\nRecent LTX output:\n{status_tail}"
            return _clip(detail)
    if status_tail:
        return _clip(status_tail)
    return ""


def _interesting_status_tail(events: list[dict]) -> str:
    needles = (
        "traceback",
        "runtimeerror",
        "cuda",
        "out of memory",
        "invalid python storage",
        "error",
        "failed",
        "unsupported",
        "probing native gemma gguf",
        "gguf metadata",
        "aiwf ltx loader",
    )
    messages = [
        str(event.get("message") or "")
        for event in events
        if event.get("kind") == "status" and any(token in str(event.get("message") or "").lower() for token in needles)
    ]
    return "\n".join(messages[-8:])


def _clip(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:].lstrip()


def _native_gemma_gguf_blocker(payload: dict) -> str:
    return (
        "Native Gemma GGUF was selected for LTX, but this worker cannot generate with it yet. "
        "LTX needs Gemma hidden states from every layer plus an attention mask; the current "
        "upstream LTX CLI only accepts a repo-shaped safetensors Gemma text encoder. "
        f"Selected GGUF: {payload.get('gemma_gguf_path')}. "
        "Run `venv\\Scripts\\python.exe scripts\\probe_ltx_runtime.py --gguf` to verify the "
        "native-GGUF blocker without dequantizing weights."
    )
