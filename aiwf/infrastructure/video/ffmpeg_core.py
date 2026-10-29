from __future__ import annotations

import functools
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from aiwf.core.domain.video_lab import MediaProbe, VideoLabPlan, VideoLabSettings
from aiwf.infrastructure.video.processing import _resolve_ffmpeg


class FFmpegUnavailable(RuntimeError):
    pass


class MediaProbeError(RuntimeError):
    pass


def _resolve_ffprobe(ffmpeg: str | None = None) -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if sibling.is_file():
            return str(sibling)
    return None


def require_ffmpeg() -> str:
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise FFmpegUnavailable(
            "FFmpeg is required for Video Lab. Install FFmpeg or reinstall "
            "imageio-ffmpeg, then restart AIWF Studio."
        )
    return ffmpeg


def _ratio(value: object) -> float:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    num, den = text.split("/", 1)
    try:
        den_value = float(den)
        return float(num) / den_value if den_value else 0.0
    except ValueError:
        return 0.0


def _int(value: object) -> int:
    try:
        return int(float(str(value or 0)))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        out = float(str(value or 0))
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _rotation(stream: dict) -> int:
    tags = stream.get("tags") or {}
    raw = tags.get("rotate")
    if raw is not None:
        return _int(raw) % 360
    for item in stream.get("side_data_list") or []:
        if "rotation" in item:
            return _int(item.get("rotation")) % 360
    return 0


def _probe_with_ffprobe(path: Path, ffprobe: str) -> MediaProbe:
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaProbeError(f"ffprobe failed for {path.name}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown ffprobe error").strip()
        raise MediaProbeError(detail[-1200:])
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid JSON.") from exc

    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    subtitles = any(item.get("codec_type") == "subtitle" for item in streams)
    format_data = payload.get("format") or {}
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate"))
    duration = _float(video.get("duration")) or _float(format_data.get("duration"))
    frame_count = _int(video.get("nb_frames"))
    if frame_count <= 0 and duration > 0 and fps > 0:
        frame_count = int(round(duration * fps))

    return MediaProbe(
        path=str(path),
        duration_seconds=max(0.0, duration),
        width=max(0, _int(video.get("width"))),
        height=max(0, _int(video.get("height"))),
        fps=max(0.0, fps),
        frame_count=max(0, frame_count),
        video_codec=str(video.get("codec_name") or "unknown"),
        pixel_format=str(video.get("pix_fmt") or "unknown"),
        field_order=str(video.get("field_order") or "unknown"),
        rotation=_rotation(video),
        has_audio=bool(audio),
        audio_codec=str(audio.get("codec_name")) if audio else None,
        audio_channels=max(0, _int(audio.get("channels"))),
        audio_sample_rate=max(0, _int(audio.get("sample_rate"))),
        has_subtitles=subtitles,
        format_name=str(format_data.get("format_name") or "unknown"),
        bit_rate=max(0, _int(format_data.get("bit_rate"))),
        size_bytes=max(0, path.stat().st_size),
        source="ffprobe",
    )


def _probe_with_opencv(path: Path) -> MediaProbe:
    try:
        from aiwf.infrastructure.video.processing import VideoProcessor

        info = VideoProcessor().probe(path)
    except Exception as exc:
        raise MediaProbeError(str(exc)) from exc
    return MediaProbe(
        path=str(path),
        duration_seconds=max(0.0, info.duration_seconds),
        width=info.width,
        height=info.height,
        fps=info.fps,
        frame_count=info.frame_count,
        has_audio=info.has_audio,
        size_bytes=max(0, path.stat().st_size),
        source="opencv fallback",
    )


def probe_media(path: str | Path) -> MediaProbe:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MediaProbeError(f"Video not found: {source}")
    ffmpeg = _resolve_ffmpeg()
    ffprobe = _resolve_ffprobe(ffmpeg)
    if ffprobe:
        try:
            return _probe_with_ffprobe(source, ffprobe)
        except MediaProbeError:
            # OpenCV gives a useful basic fallback when a partial ffprobe install is present.
            return _probe_with_opencv(source)
    return _probe_with_opencv(source)


@functools.lru_cache(maxsize=4)
def _ffmpeg_listing(kind: str, executable: str) -> frozenset[str]:
    flag = "-encoders" if kind == "encoders" else "-filters"
    try:
        result = subprocess.run(
            [executable, "-hide_banner", flag],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    names: set[str] = set()
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and len(parts[0]) <= 8:
            names.add(parts[1])
    return frozenset(names)


def available_encoders(ffmpeg: str | None = None) -> frozenset[str]:
    executable = ffmpeg or require_ffmpeg()
    return _ffmpeg_listing("encoders", executable)


def available_filters(ffmpeg: str | None = None) -> frozenset[str]:
    executable = ffmpeg or require_ffmpeg()
    return _ffmpeg_listing("filters", executable)


def _select_codec(requested: str, encoders: Iterable[str], warnings: list[str]) -> str:
    supported = set(encoders)
    fallbacks = {
        "h264_nvenc": "libx264",
        "hevc_nvenc": "libx265",
        "h264": "libx264",
        "hevc": "libx265",
    }
    if requested == "auto":
        return "h264_nvenc" if "h264_nvenc" in supported else "libx264"
    selected = fallbacks.get(requested, requested)
    if selected in supported:
        return selected
    fallback = "libx265" if "hevc" in requested else "libx264"
    warnings.append(f"Encoder {selected} is unavailable; using {fallback}.")
    return fallback


def _append_filter(filters: list[str], available: set[str], name: str, expression: str, warnings: list[str]) -> None:
    if name in available:
        filters.append(expression)
    else:
        warnings.append(f"FFmpeg filter '{name}' is unavailable; that stage was skipped.")


def build_filter_graph(
    settings: VideoLabSettings,
    probe: MediaProbe,
    *,
    filter_names: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    available = set(filter_names)
    video: list[str] = []
    audio: list[str] = []
    warnings: list[str] = []

    if settings.deinterlace:
        expression = (
            "bwdif="
            f"mode={settings.deinterlace_mode}:"
            f"parity={settings.deinterlace_parity}:"
            f"deint={settings.deinterlace_scope}"
        )
        _append_filter(video, available, "bwdif", expression, warnings)
    if settings.stabilize:
        expression = (
            "deshake="
            f"rx={settings.stabilize_radius_x}:"
            f"ry={settings.stabilize_radius_y}:"
            f"edge={settings.stabilize_edge}:"
            f"blocksize={settings.stabilize_block_size}:"
            f"contrast={settings.stabilize_contrast}"
        )
        _append_filter(video, available, "deshake", expression, warnings)
    if settings.deflicker:
        _append_filter(
            video,
            available,
            "deflicker",
            f"deflicker=size={settings.deflicker_size}:mode={settings.deflicker_mode}",
            warnings,
        )
    if settings.denoise != "off":
        _append_filter(
            video,
            available,
            "hqdn3d",
            "hqdn3d="
            f"{settings.denoise_luma_spatial:g}:"
            f"{settings.denoise_chroma_spatial:g}:"
            f"{settings.denoise_luma_temporal:g}:"
            f"{settings.denoise_chroma_temporal:g}",
            warnings,
        )
    if settings.sharpen != "off":
        kernel = int(settings.sharpen_kernel)
        _append_filter(
            video,
            available,
            "unsharp",
            f"unsharp={kernel}:{kernel}:{settings.sharpen_amount:g}:{kernel}:{kernel}:0.0",
            warnings,
        )

    if settings.scale == "720p":
        video.append("scale=-2:720:flags=lanczos")
    elif settings.scale == "1080p":
        video.append("scale=-2:1080:flags=lanczos")
    elif settings.scale == "2x":
        video.append("scale=trunc(iw*2/2)*2:trunc(ih*2/2)*2:flags=lanczos")
    elif settings.scale == "custom":
        width = int(settings.custom_width or 0)
        height = int(settings.custom_height or 0)
        if settings.keep_aspect:
            if width > 0 and height > 0:
                # Fit inside the requested box without stretching, then guarantee
                # even output dimensions for the common delivery codecs.
                video.append(
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos"
                )
            elif width > 0:
                video.append(f"scale={width}:-2:flags=lanczos")
            else:
                video.append(f"scale=-2:{height}:flags=lanczos")
        else:
            width_expr = str(width) if width > 0 else "iw"
            height_expr = str(height) if height > 0 else "ih"
            video.append(f"scale={width_expr}:{height_expr}:flags=lanczos")

    if settings.target_fps is not None:
        fps = f"{settings.target_fps:g}"
        if settings.motion_interpolation:
            _append_filter(
                video,
                available,
                "minterpolate",
                f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                warnings,
            )
        else:
            video.append(f"fps={fps}")

    # Browser codecs and NVENC generally require even dimensions and square pixels.
    video.extend(["scale=trunc(iw/2)*2:trunc(ih/2)*2", "setsar=1"])

    if probe.has_audio and settings.audio_cleanup:
        cleanup_filters = (
            ("highpass", f"highpass=f={settings.audio_highpass_hz:g}"),
            ("lowpass", f"lowpass=f={settings.audio_lowpass_hz:g}"),
            (
                "afftdn",
                "afftdn="
                f"nr={settings.audio_noise_reduction_db:g}:"
                f"nf={settings.audio_noise_floor_db:g}:"
                f"nt={settings.audio_noise_type}:"
                f"tn={1 if settings.audio_track_noise else 0}:"
                "gs=3",
            ),
        )
        for name, expression in cleanup_filters:
            _append_filter(audio, available, name, expression, warnings)
    if probe.has_audio and settings.audio_normalize:
        _append_filter(
            audio,
            available,
            "loudnorm",
            "loudnorm="
            f"I={settings.audio_target_lufs:g}:"
            f"TP={settings.audio_true_peak_db:g}:"
            f"LRA={settings.audio_lra:g}",
            warnings,
        )
    return video, audio, warnings


def build_ffmpeg_command(
    *,
    ffmpeg: str,
    input_path: Path,
    temporary_output: Path,
    settings: VideoLabSettings,
    probe: MediaProbe,
    selected_codec: str,
    video_filters: list[str],
    audio_filters: list[str],
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-y", "-i", str(input_path)]
    if settings.trim_start > 0:
        command.extend(["-ss", f"{settings.trim_start:.3f}"])
    if settings.trim_end is not None:
        command.extend(["-t", f"{settings.trim_end - settings.trim_start:.3f}"])

    command.extend(["-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "0", "-map_chapters", "0"])
    if settings.container == "mkv" and probe.has_subtitles:
        command.extend(["-map", "0:s?", "-c:s", "copy"])
    if video_filters:
        command.extend(["-vf", ",".join(video_filters)])
    if audio_filters:
        command.extend(["-af", ",".join(audio_filters)])

    if "nvenc" in selected_codec:
        command.extend(
            [
                "-c:v",
                selected_codec,
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                str(settings.quality),
                "-b:v",
                "0",
            ]
        )
    else:
        command.extend(
            ["-c:v", selected_codec, "-preset", "medium", "-crf", str(settings.quality)]
        )
    command.extend(["-pix_fmt", "yuv420p"])
    if probe.has_audio:
        command.extend(["-c:a", "aac", "-b:a", f"{settings.audio_bitrate_kbps}k"])
    if settings.container == "mp4":
        command.extend(["-movflags", "+faststart"])
    command.extend(["-progress", "pipe:1", "-nostats", str(temporary_output)])
    return command


def resolve_plan_command(plan: VideoLabPlan) -> VideoLabPlan:
    """Attach local FFmpeg capabilities and a concrete command to a plan."""
    ffmpeg = require_ffmpeg()
    warnings = list(plan.warnings)
    filters, audio, filter_warnings = build_filter_graph(
        plan.settings,
        plan.probe,
        filter_names=available_filters(ffmpeg),
    )
    warnings.extend(filter_warnings)
    codec = _select_codec(plan.settings.codec, available_encoders(ffmpeg), warnings)
    output = Path(plan.output_path)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    command = build_ffmpeg_command(
        ffmpeg=ffmpeg,
        input_path=Path(plan.input_path),
        temporary_output=temporary,
        settings=plan.settings,
        probe=plan.probe,
        selected_codec=codec,
        video_filters=filters,
        audio_filters=audio,
    )
    return plan.model_copy(
        update={
            "selected_codec": codec,
            "video_filters": filters,
            "audio_filters": audio,
            "warnings": warnings,
            "command": command,
        }
    )
