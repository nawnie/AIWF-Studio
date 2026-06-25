from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from aiwf.core.domain.video_lab import (
    MediaProbe,
    VideoLabPlan,
    VideoLabResult,
    VideoLabSettings,
)
from aiwf.infrastructure.video.ffmpeg_core import probe_media, resolve_plan_command

ProgressCallback = Callable[[float, str], None]


class VideoLabError(RuntimeError):
    pass


class VideoLabBusy(VideoLabError):
    pass


class VideoLabCancelled(VideoLabError):
    pass


_PRESET_DEFAULTS: dict[str, dict[str, object]] = {
    "old_family_film": {
        "deinterlace": True,
        "deinterlace_mode": "send_frame",
        "deinterlace_parity": "auto",
        "deinterlace_scope": "interlaced",
        "stabilize": True,
        "stabilize_radius_x": 12,
        "stabilize_radius_y": 12,
        "stabilize_edge": "mirror",
        "stabilize_block_size": 8,
        "stabilize_contrast": 110,
        "deflicker": True,
        "deflicker_size": 7,
        "deflicker_mode": "median",
        "denoise": "light",
        "denoise_luma_spatial": 1.5,
        "denoise_chroma_spatial": 1.5,
        "denoise_luma_temporal": 6.0,
        "denoise_chroma_temporal": 6.0,
        "sharpen": "light",
        "sharpen_kernel": 5,
        "sharpen_amount": 0.35,
        "scale": "keep",
        "target_fps": None,
        "motion_interpolation": False,
        "audio_cleanup": True,
        "audio_highpass_hz": 70.0,
        "audio_lowpass_hz": 12500.0,
        "audio_noise_reduction_db": 10.0,
        "audio_noise_floor_db": -55.0,
        "audio_noise_type": "vinyl",
        "audio_track_noise": True,
        "audio_normalize": True,
        "audio_target_lufs": -16.0,
        "audio_true_peak_db": -1.5,
        "audio_lra": 11.0,
        "codec": "auto",
        "container": "mp4",
        "quality": 19,
        "audio_bitrate_kbps": 192,
    },
    "web_video_cleanup": {
        "deinterlace": False,
        "stabilize": False,
        "deflicker": False,
        "denoise": "light",
        "denoise_luma_spatial": 1.25,
        "denoise_chroma_spatial": 1.25,
        "denoise_luma_temporal": 4.5,
        "denoise_chroma_temporal": 4.5,
        "sharpen": "light",
        "sharpen_kernel": 5,
        "sharpen_amount": 0.35,
        "scale": "1080p",
        "target_fps": None,
        "motion_interpolation": False,
        "audio_cleanup": False,
        "audio_normalize": True,
        "audio_target_lufs": -16.0,
        "audio_true_peak_db": -1.5,
        "audio_lra": 11.0,
        "codec": "auto",
        "container": "mp4",
        "quality": 21,
        "audio_bitrate_kbps": 192,
    },
    "generated_video_polish": {
        "deinterlace": False,
        "stabilize": False,
        "deflicker": False,
        "denoise": "light",
        "denoise_luma_spatial": 1.0,
        "denoise_chroma_spatial": 1.0,
        "denoise_luma_temporal": 4.0,
        "denoise_chroma_temporal": 4.0,
        "sharpen": "light",
        "sharpen_kernel": 5,
        "sharpen_amount": 0.30,
        "scale": "keep",
        "target_fps": 30.0,
        "motion_interpolation": False,
        "audio_cleanup": False,
        "audio_normalize": True,
        "audio_target_lufs": -14.0,
        "audio_true_peak_db": -1.0,
        "audio_lra": 9.0,
        "codec": "auto",
        "container": "mp4",
        "quality": 19,
        "audio_bitrate_kbps": 192,
    },
    "custom": {},
}


def preset_settings(name: str) -> VideoLabSettings:
    preset = name if name in _PRESET_DEFAULTS else "custom"
    return VideoLabSettings(preset=preset, **_PRESET_DEFAULTS[preset])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_for_display(command: list[str]) -> str:
    def quote(item: str) -> str:
        if not item or any(ch.isspace() for ch in item) or any(ch in item for ch in '&()[]{};,='):
            return '"' + item.replace('"', '\\"') + '"'
        return item

    return " ".join(quote(item) for item in command)


class VideoLabService:
    """FFmpeg-first video restoration and export orchestration.

    Only one deterministic Video Lab encode is allowed at a time in the current
    local Gradio process. GPU-backed stages remain separate services until they
    can share the engine supervisor without breaking cancellation semantics.
    """

    def __init__(self, output_root: str | Path) -> None:
        self.output_root = Path(output_root).expanduser().resolve() / "video-lab"
        self._lock = threading.RLock()
        self._active_process: subprocess.Popen[str] | None = None
        self._active_job_id: str | None = None
        self._cancel_requested = False

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id

    def inspect(self, input_path: str | Path) -> MediaProbe:
        return probe_media(input_path)

    def new_job_id(self) -> str:
        return f"vlab_{uuid.uuid4().hex[:12]}"

    def _job_dir(self, job_id: str) -> Path:
        day = datetime.now().strftime("%Y%m%d")
        return self.output_root / day / job_id

    def build_plan(
        self,
        input_path: str | Path,
        settings: VideoLabSettings,
        *,
        job_id: str | None = None,
    ) -> VideoLabPlan:
        source = Path(input_path).expanduser().resolve()
        probe = self.inspect(source)
        resolved_job_id = job_id or self.new_job_id()
        job_dir = self._job_dir(resolved_job_id)
        suffix = ".mkv" if settings.container == "mkv" else ".mp4"
        output_path = job_dir / f"{source.stem}_processed{suffix}"
        duration = probe.duration_seconds
        if settings.trim_end is not None:
            duration = max(0.0, settings.trim_end - settings.trim_start)
        elif settings.trim_start:
            duration = max(0.0, duration - settings.trim_start)

        warnings: list[str] = []
        if settings.deinterlace and not probe.is_interlaced:
            warnings.append("The source reports progressive frames; deinterlace is enabled by user choice.")
        if settings.motion_interpolation:
            warnings.append(
                "FFmpeg motion interpolation is CPU-heavy. The RIFE tab remains the higher-quality option "
                "until chunked RIFE is fully joined to the Video Lab graph."
            )
        if probe.has_subtitles and settings.container == "mp4":
            warnings.append("MP4 export keeps video, the first audio stream, metadata, and chapters; subtitles are omitted.")
        if probe.source != "ffprobe":
            warnings.append("ffprobe was unavailable, so metadata is limited to the OpenCV fallback.")

        plan = VideoLabPlan(
            input_path=str(source),
            output_path=str(output_path),
            job_id=resolved_job_id,
            probe=probe,
            settings=settings,
            warnings=warnings,
            selected_container=settings.container,
            expected_duration_seconds=duration,
        )
        return resolve_plan_command(plan)

    def plan_text(self, plan: VideoLabPlan) -> str:
        settings = plan.settings
        stages = ["Inspect"]
        if settings.trim_start or settings.trim_end is not None:
            stages.append("Trim")
        if settings.deinterlace:
            stages.append("Deinterlace")
        if settings.stabilize:
            stages.append("Stabilize")
        if settings.deflicker:
            stages.append("Deflicker")
        if settings.denoise != "off":
            stages.append("Denoise")
        if settings.sharpen != "off":
            stages.append("Sharpen")
        if settings.scale != "keep":
            stages.append("Resize")
        if settings.target_fps is not None:
            stages.append("Frame-rate conversion")
        if settings.audio_cleanup:
            stages.append("Audio cleanup")
        if settings.audio_normalize:
            stages.append("Loudness normalize")
        stages.extend(["Encode", "Publish manifest"])
        stage_line = " → ".join(stages)
        payload = {
            "job_id": plan.job_id,
            "stages": stage_line,
            "input": plan.input_path,
            "output": plan.output_path,
            "source": plan.probe.model_dump(mode="json"),
            "settings": plan.settings.model_dump(mode="json"),
            "selected_codec": plan.selected_codec,
            "video_filters": plan.video_filters,
            "audio_filters": plan.audio_filters,
            "warnings": plan.warnings,
            "command_preview": _command_for_display(plan.command),
        }
        return json.dumps(payload, indent=2)

    def _write_manifest(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp, path)

    def cancel_active(self) -> str:
        with self._lock:
            process = self._active_process
            job_id = self._active_job_id
            self._cancel_requested = True
        if process is None or process.poll() is not None:
            if job_id:
                return f"Cancellation queued for {job_id}."
            return "No active Video Lab process."
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        except OSError:
            pass
        return f"Cancellation requested for {job_id or 'the active Video Lab job'}."

    def execute(self, plan: VideoLabPlan, *, on_progress: ProgressCallback | None = None) -> VideoLabResult:
        with self._lock:
            if self._active_job_id is not None:
                raise VideoLabBusy(f"Video Lab is already running {self._active_job_id}.")
            self._active_job_id = plan.job_id
            self._cancel_requested = False

        output = Path(plan.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
        manifest_path = output.parent / "job.json"
        log_path = output.parent / "ffmpeg.log"
        started = time.perf_counter()
        started_at = _utc_now()
        manifest = {
            "schema": 1,
            "job_id": plan.job_id,
            "status": "running",
            "created_at": started_at,
            "updated_at": started_at,
            "plan": plan.model_dump(mode="json"),
        }
        self._write_manifest(manifest_path, manifest)

        if on_progress:
            on_progress(0.0, "Starting FFmpeg")
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                process = subprocess.Popen(
                    plan.command,
                    stdout=subprocess.PIPE,
                    stderr=log_file,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
                with self._lock:
                    self._active_process = process
                    cancel_queued = self._cancel_requested
                if cancel_queued:
                    process.terminate()

                last_percent = -1.0
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key in {"out_time_us", "out_time_ms"}:
                        try:
                            seconds = float(value) / 1_000_000.0
                        except ValueError:
                            continue
                        duration = max(0.001, plan.expected_duration_seconds)
                        percent = min(0.99, max(0.0, seconds / duration))
                        if on_progress and percent - last_percent >= 0.005:
                            last_percent = percent
                            on_progress(percent, f"Processing {seconds:.1f}s / {duration:.1f}s")
                    elif key == "progress" and value == "end":
                        if on_progress:
                            on_progress(1.0, "Finalizing output")
                return_code = process.wait()

            with self._lock:
                cancelled = self._cancel_requested
            if cancelled:
                raise VideoLabCancelled("Video Lab job cancelled.")
            if return_code != 0:
                tail = ""
                try:
                    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
                except OSError:
                    pass
                raise VideoLabError(f"FFmpeg exited with code {return_code}.\n{tail}")
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise VideoLabError("FFmpeg completed without producing a usable output file.")
            os.replace(temporary, output)
            elapsed = time.perf_counter() - started
            manifest.update(
                {
                    "status": "completed",
                    "updated_at": _utc_now(),
                    "elapsed_seconds": elapsed,
                    "output_path": str(output),
                    "output_size_bytes": output.stat().st_size,
                }
            )
            self._write_manifest(manifest_path, manifest)
            if on_progress:
                on_progress(1.0, "Done")
            return VideoLabResult(
                job_id=plan.job_id,
                output_path=str(output),
                manifest_path=str(manifest_path),
                log_path=str(log_path),
                elapsed_seconds=elapsed,
                message=f"Saved {output.name} in {elapsed:.1f}s",
                warnings=plan.warnings,
            )
        except VideoLabCancelled:
            manifest.update({"status": "cancelled", "updated_at": _utc_now()})
            self._write_manifest(manifest_path, manifest)
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:
            manifest.update({"status": "failed", "updated_at": _utc_now(), "error": str(exc)})
            self._write_manifest(manifest_path, manifest)
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            if isinstance(exc, VideoLabError):
                raise
            raise VideoLabError(str(exc)) from exc
        finally:
            with self._lock:
                self._active_process = None
                self._active_job_id = None
                self._cancel_requested = False
