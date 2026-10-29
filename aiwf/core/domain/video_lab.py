from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

VideoLabPreset = Literal[
    "old_family_film",
    "web_video_cleanup",
    "generated_video_polish",
    "custom",
]
VideoLabDenoise = Literal["off", "light", "strong", "custom"]
VideoLabSharpen = Literal["off", "light", "strong", "custom"]
VideoLabScale = Literal["keep", "720p", "1080p", "2x", "custom"]
VideoLabCodec = Literal["auto", "h264", "hevc", "h264_nvenc", "hevc_nvenc"]
VideoLabContainer = Literal["mp4", "mkv"]
VideoLabDeinterlaceMode = Literal["send_frame", "send_field"]
VideoLabFieldParity = Literal["auto", "tff", "bff"]
VideoLabDeinterlaceScope = Literal["all", "interlaced"]
VideoLabStabilizeEdge = Literal["blank", "original", "clamp", "mirror"]
VideoLabDeflickerMode = Literal["am", "gm", "hm", "qm", "cm", "pm", "median"]
VideoLabNoiseType = Literal["white", "vinyl", "shellac"]


class MediaProbe(BaseModel):
    """Normalized media metadata used by the Video Lab planner."""

    path: str
    duration_seconds: float = Field(default=0.0, ge=0.0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    fps: float = Field(default=0.0, ge=0.0)
    frame_count: int = Field(default=0, ge=0)
    video_codec: str = "unknown"
    pixel_format: str = "unknown"
    field_order: str = "unknown"
    rotation: int = 0
    has_audio: bool = False
    audio_codec: str | None = None
    audio_channels: int = 0
    audio_sample_rate: int = 0
    has_subtitles: bool = False
    format_name: str = "unknown"
    bit_rate: int = 0
    size_bytes: int = Field(default=0, ge=0)
    source: str = "ffprobe"

    @property
    def is_interlaced(self) -> bool:
        field = self.field_order.lower().strip()
        return field not in {"", "unknown", "progressive"}


class VideoLabSettings(BaseModel):
    """User-selected deterministic post-processing options.

    Presets populate these values, but the Gradio UI exposes every active
    stage's parameters so presets remain editable starting points.
    """

    preset: VideoLabPreset = "custom"
    trim_start: float = Field(default=0.0, ge=0.0)
    trim_end: float | None = Field(default=None, gt=0.0)

    deinterlace: bool = False
    deinterlace_mode: VideoLabDeinterlaceMode = "send_frame"
    deinterlace_parity: VideoLabFieldParity = "auto"
    deinterlace_scope: VideoLabDeinterlaceScope = "interlaced"

    stabilize: bool = False
    stabilize_radius_x: int = Field(default=16, ge=0, le=64)
    stabilize_radius_y: int = Field(default=16, ge=0, le=64)
    stabilize_edge: VideoLabStabilizeEdge = "mirror"
    stabilize_block_size: int = Field(default=8, ge=4, le=128)
    stabilize_contrast: int = Field(default=125, ge=1, le=255)

    deflicker: bool = False
    deflicker_size: int = Field(default=5, ge=2, le=129)
    deflicker_mode: VideoLabDeflickerMode = "pm"

    denoise: VideoLabDenoise = "off"
    denoise_luma_spatial: float = Field(default=1.5, ge=0.0, le=20.0)
    denoise_chroma_spatial: float = Field(default=1.5, ge=0.0, le=20.0)
    denoise_luma_temporal: float = Field(default=6.0, ge=0.0, le=30.0)
    denoise_chroma_temporal: float = Field(default=6.0, ge=0.0, le=30.0)

    sharpen: VideoLabSharpen = "off"
    sharpen_kernel: int = Field(default=5, ge=3, le=23)
    sharpen_amount: float = Field(default=0.45, ge=-2.0, le=5.0)

    scale: VideoLabScale = "keep"
    custom_width: int = Field(default=0, ge=0, le=16384)
    custom_height: int = Field(default=0, ge=0, le=16384)
    keep_aspect: bool = True

    target_fps: float | None = Field(default=None, ge=1.0, le=120.0)
    motion_interpolation: bool = False

    audio_cleanup: bool = False
    audio_highpass_hz: float = Field(default=70.0, ge=10.0, le=2000.0)
    audio_lowpass_hz: float = Field(default=12500.0, ge=1000.0, le=24000.0)
    audio_noise_reduction_db: float = Field(default=12.0, ge=0.01, le=97.0)
    audio_noise_floor_db: float = Field(default=-50.0, ge=-80.0, le=-20.0)
    audio_noise_type: VideoLabNoiseType = "white"
    audio_track_noise: bool = True

    audio_normalize: bool = True
    audio_target_lufs: float = Field(default=-16.0, ge=-70.0, le=-5.0)
    audio_true_peak_db: float = Field(default=-1.5, ge=-9.0, le=0.0)
    audio_lra: float = Field(default=11.0, ge=1.0, le=50.0)

    codec: VideoLabCodec = "auto"
    container: VideoLabContainer = "mp4"
    quality: int = Field(default=20, ge=14, le=36)
    audio_bitrate_kbps: int = Field(default=192, ge=64, le=512)

    @model_validator(mode="after")
    def validate_settings(self):
        if self.trim_end is not None and self.trim_end <= self.trim_start:
            raise ValueError("Trim end must be greater than trim start.")
        if self.motion_interpolation and self.target_fps is None:
            raise ValueError("Motion interpolation needs a target FPS.")
        if self.scale == "custom" and self.custom_width <= 0 and self.custom_height <= 0:
            raise ValueError("Custom resize needs a width, height, or both.")
        if self.audio_cleanup and self.audio_lowpass_hz <= self.audio_highpass_hz:
            raise ValueError("Audio low-pass frequency must be above the high-pass frequency.")
        # FFmpeg unsharp kernels must be odd. Normalizing here avoids a cryptic
        # encoder failure when settings are loaded from a hand-edited manifest.
        if self.sharpen_kernel % 2 == 0:
            self.sharpen_kernel = min(23, self.sharpen_kernel + 1)
        return self


class VideoLabPlan(BaseModel):
    """Resolved, capability-aware FFmpeg execution plan."""

    input_path: str
    output_path: str
    job_id: str
    probe: MediaProbe
    settings: VideoLabSettings
    video_filters: list[str] = Field(default_factory=list)
    audio_filters: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    selected_codec: str = "libx264"
    selected_container: VideoLabContainer = "mp4"
    command: list[str] = Field(default_factory=list)
    expected_duration_seconds: float = Field(default=0.0, ge=0.0)


class VideoLabResult(BaseModel):
    job_id: str
    output_path: str
    manifest_path: str
    log_path: str
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
