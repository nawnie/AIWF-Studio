from __future__ import annotations

from pydantic import BaseModel, Field

# Default model: the 5B TI2V variant is the most VRAM-accessible Wan 2.2 model
# and does image->video through WanImageToVideoPipeline.
WAN_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Offload strategies (low VRAM -> high VRAM): "sequential" (8 GB, slowest),
# "model" (12-16 GB), "none" (keep on GPU, fastest, needs the most VRAM).
OFFLOAD_MODES = ("sequential", "model", "none")


def snap_num_frames(n: int) -> int:
    """Wan requires num_frames of the form 4*k + 1. Snap to the nearest valid value."""
    n = max(5, int(n))
    k = round((n - 1) / 4)
    return max(5, 4 * k + 1)


def frames_for_duration_seconds(fps: int, seconds: float, *, max_seconds: int = 10) -> int:
    """Convert fps + duration to Wan-valid frame count, capped for local testing."""
    safe_fps = max(1, int(fps))
    safe_seconds = max(1.0, min(float(seconds), float(max_seconds)))
    return snap_num_frames(int(round(safe_fps * safe_seconds)) + 1)


def duration_seconds_for_frames(num_frames: int, fps: int) -> float:
    safe_fps = max(1, int(fps))
    return max(0.0, (int(num_frames) - 1) / safe_fps)


class WanI2VRequest(BaseModel):
    """Parameters for a Wan 2.2 image-to-video render."""

    prompt: str = ""
    negative_prompt: str = ""
    num_frames: int = Field(default=49, ge=5, le=257)
    steps: int = Field(default=8, ge=1, le=100)
    high_noise_steps: int = Field(default=4, ge=1, le=60)
    low_noise_steps: int = Field(default=4, ge=1, le=60)
    guidance_scale: float = Field(default=1.0, ge=1.0, le=20.0)
    width: int = Field(default=480, ge=128, le=1280)
    height: int = Field(default=480, ge=128, le=1280)
    fps: int = Field(default=16, ge=1, le=60)
    flow_shift: float = Field(default=5.0, ge=1.0, le=12.0)
    seed: int = -1
    model_id: str = WAN_TI2V_5B
    offload: str = "model"

    # Wan 2.2 I2V 14B+ (and some variants) use a two-stage (high-noise / low-noise) transformer pair.
    # Provide both for dual-stage denoising. `boundary_ratio` controls the switch point
    # (e.g. 0.875 is a common value: high-noise transformer for early / high-noise timesteps,
    # low-noise for the rest).
    high_noise_model_id: str | None = None
    low_noise_model_id: str | None = None
    boundary_ratio: float | None = Field(default=0.875, ge=0.0, le=1.0)
    high_noise_lora_id: str | None = None
    high_noise_lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    low_noise_lora_id: str | None = None
    low_noise_lora_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    vae_id: str | None = None  # explicit VAE (Wan 2.2 I2V typically wants the Wan 2.1 VAE safetensors)
    components_base: str | None = None  # resolved diffusers folder for text_encoder/tokenizer/scheduler/vae

    @property
    def max_area(self) -> int:
        return int(self.width) * int(self.height)

    def normalized_frames(self) -> int:
        return snap_num_frames(self.num_frames)

    def effective_steps(self) -> int:
        high = max(1, int(self.high_noise_steps or 0))
        low = max(1, int(self.low_noise_steps or 0))
        return max(1, high + low)

    def effective_boundary_ratio(self) -> float:
        total = self.effective_steps()
        high = max(1, int(self.high_noise_steps or 0))
        return min(1.0, max(0.0, high / total))

    def uses_dual_transformers(self) -> bool:
        return bool(self.high_noise_model_id and self.low_noise_model_id)


class WanI2VResult(BaseModel):
    output_path: str
    frame_count: int = Field(default=0, ge=0)
    fps: int = Field(default=24, ge=1)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    elapsed_seconds: float = 0.0
    message: str = ""
