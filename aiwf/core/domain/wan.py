from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# Default model: the 5B TI2V variant is the most VRAM-accessible Wan 2.2 model
# and does image->video through WanImageToVideoPipeline.
WAN_TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Offload strategies: "sequential" (lowest VRAM, slowest), "group" (block-level
# middle ground), "streamed" (one-block group offload with CUDA prefetch),
# "model" (low-VRAM quantized active-stage swap), "balanced" (active-stage swap
# with VAE resident), "resident" (keep quantized high+low stages on GPU),
# "none" (move the whole pipeline to GPU).
OFFLOAD_MODES = ("sequential", "group", "streamed", "model", "balanced", "resident", "none")

# Sigma (noise schedule) types for FlowMatchEulerDiscreteScheduler.
# Controls how denoising steps are spaced across the noise level range.
#   simple      -- uniform linear spacing (fast, default, tends to look flat at <20 steps)
#   beta        -- beta distribution spacing (smoother motion, best quality at low step counts)
#   exponential -- exponential spacing (more detail at high noise, less at low)
#   karras      -- Karras et al. schedule (preserves fine detail, familiar from SD)
SIGMA_TYPES = ("simple", "beta", "exponential", "karras")

# Sampler (solver algorithm) types.
#   unipc -- UniPCMultistepScheduler (2nd-order predictor-corrector). This is the
#            sampler the Wan2.2-TI2V-5B checkpoint actually ships calibrated for
#            (see its scheduler/scheduler_config.json: solver_order=2,
#            solver_type="bh2", predict_x0=True, flow_shift=5.0). Default for the
#            5B route; switching away from it onto a 1st/2nd-order Euler/Heun ODE
#            solver the model was never tuned against is a known source of
#            unstable motion ("warped" video), especially at low step counts.
#   euler -- FlowMatchEulerDiscreteScheduler (fast, 1 NFE per step)
#   heun  -- FlowMatchHeunDiscreteScheduler (2nd-order ODE, ~2x NFE per step)
SAMPLER_TYPES = ("unipc", "euler", "heun")

# Wan video routes:
# - fast_5b: single 5B safetensors route for the default fast path.
# - native_high_low_fp8_experimental: dual 14B high/low route for FP8
#   safetensors weights.
# - native_high_low: dual GGUF high/low route. Keep UI filtering strict so users
#   do not mix incompatible 5B, FP8 safetensors, and GGUF assets in one request.
WAN_RUNTIME_FAST_5B = "fast_5b"
WAN_RUNTIME_HIGH_LOW = "native_high_low"
WAN_RUNTIME_HIGH_LOW_FP8 = "native_high_low_fp8_experimental"
WAN_RUNTIME_MODES = (
    WAN_RUNTIME_FAST_5B,
    WAN_RUNTIME_HIGH_LOW,
    WAN_RUNTIME_HIGH_LOW_FP8,
)


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
    num_frames: int = Field(default=81, ge=5, le=257)
    steps: int = Field(default=20, ge=1, le=100)
    high_noise_steps: int = Field(default=20, ge=1, le=60)
    low_noise_steps: int = Field(default=1, ge=1, le=60)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)
    width: int = Field(default=512, ge=128, le=1280)
    height: int = Field(default=512, ge=128, le=1280)
    fps: int = Field(default=16, ge=1, le=60)
    # 5.0 matches the Wan2.2-TI2V-5B model's own shipped scheduler_config.json.
    # The previous default of 8.0 paired with the FlowMatch Euler substitution
    # (instead of the model's native UniPC) was a likely cause of warped video.
    flow_shift: float = Field(default=5.0, ge=0.5, le=25.0)
    sigma_type: str = Field(default="simple")  # simple | beta | exponential | karras
    sampler: str = Field(default="unipc")  # unipc | euler | heun
    # Temporal chunk denoise settings. This slices latent frames, not output
    # frames. It is opt-in because every chunk reruns the full transformer.
    temporal_chunks: bool = False
    chunk_size: int = Field(default=24, ge=4, le=64)
    chunk_overlap: int = Field(default=0, ge=0, le=32)
    # Route settings semantics: `steps` belongs to the single 5B fast route;
    # `high_noise_steps`, `low_noise_steps`, `boundary_ratio`, and low-stage CFG
    # only describe dual high/low routes. Filtering keeps those knobs paired with
    # compatible model families instead of silently crossing full/FP8/GGUF routes.
    # Low-noise stage CFG for Wan 2.2 dual high/low pairs (maps to diffusers'
    # `guidance_scale_2`). 1.0 = reuse `guidance_scale` for both stages. Only applied on
    # the dual transformer path; ignored by the single 5B path. (Despite the legacy field
    # name, diffusers 0.38 has no Wan `image_guidance_scale` — that arg is InstructPix2Pix
    # only — so this is plumbed through the real low-noise guidance knob.)
    image_guidance_scale: float = Field(default=1.0, ge=1.0, le=10.0)
    # Explicit text encoder path. "" = use the encoder inside components_base/text_encoder/
    # (full-precision UMT5-XXL, largest). Point to a file from models/Textencoder/ to use
    # the FP8 or GGUF variant instead.
    # Valid: umt5-xxl-*.safetensors, umt5-xxl-*.gguf, nsfw_wan_umt5-xxl_*.safetensors
    # NOT valid: t5xxl_*.safetensors -- T5-XXL is for Flux/SD3, not Wan.
    text_encoder_path: str = Field(default="")
    seed: int = -1
    runtime_mode: str = Field(default=WAN_RUNTIME_FAST_5B)
    model_id: str = WAN_TI2V_5B
    offload: str = "balanced"
    vram_reserve_enabled: bool = False
    vram_reserve_mb: int = Field(default=1536, ge=0, le=65536)

    @field_validator("text_encoder_path")
    @classmethod
    def _validate_text_encoder_path(cls, v: str) -> str:
        if not v:
            return v
        from pathlib import Path as _Path
        p = _Path(v)
        if p.suffix.lower() not in {".safetensors", ".gguf"}:
            raise ValueError(
                f"text_encoder_path must be a .safetensors or .gguf file, got {v!r}. "
                "Note: t5xxl files are T5-XXL (Flux/SD3) and are NOT compatible with Wan."
            )
        return v

    @field_validator("sigma_type")
    @classmethod
    def _validate_sigma_type(cls, v: str) -> str:
        if v not in SIGMA_TYPES:
            raise ValueError(f"sigma_type must be one of {SIGMA_TYPES}, got {v!r}")
        return v

    @field_validator("sampler")
    @classmethod
    def _validate_sampler(cls, v: str) -> str:
        if v not in SAMPLER_TYPES:
            raise ValueError(f"sampler must be one of {SAMPLER_TYPES}, got {v!r}")
        return v

    @field_validator("offload")
    @classmethod
    def _validate_offload(cls, v: str) -> str:
        if v not in OFFLOAD_MODES:
            raise ValueError(f"offload must be one of {OFFLOAD_MODES}, got {v!r}")
        return v

    @field_validator("runtime_mode")
    @classmethod
    def _validate_runtime_mode(cls, v: str) -> str:
        if v not in WAN_RUNTIME_MODES:
            raise ValueError(f"runtime_mode must be one of {WAN_RUNTIME_MODES}, got {v!r}")
        return v

    # Wan 2.2 dual routes use a two-stage (high-noise / low-noise) transformer
    # pair. Provide both files from the same route family: FP8 safetensors or GGUF.
    # `boundary_ratio` controls the switch point (e.g. 0.875 is a common value:
    # high-noise transformer for early / high-noise timesteps, low-noise for the rest).
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
        if not self.requires_dual_transformers():
            return max(1, int(self.steps or self.high_noise_steps or 1))
        high = max(1, int(self.high_noise_steps or 0))
        low = max(1, int(self.low_noise_steps or 0))
        return max(1, high + low)

    def effective_boundary_ratio(self) -> float:
        if not self.requires_dual_transformers():
            return 1.0
        total = self.effective_steps()
        high = max(1, int(self.high_noise_steps or 0))
        return min(1.0, max(0.0, high / total))

    def uses_dual_transformers(self) -> bool:
        return self.runtime_mode in {WAN_RUNTIME_HIGH_LOW, WAN_RUNTIME_HIGH_LOW_FP8} and bool(
            self.high_noise_model_id and self.low_noise_model_id
        )

    def requires_dual_transformers(self) -> bool:
        return self.runtime_mode in {WAN_RUNTIME_HIGH_LOW, WAN_RUNTIME_HIGH_LOW_FP8}


class WanI2VResult(BaseModel):
    output_path: str
    frame_count: int = Field(default=0, ge=0)
    fps: int = Field(default=16, ge=1)  # Wan renders at 16 fps; service also falls back to 16
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    elapsed_seconds: float = 0.0
    step_count: int = Field(default=0, ge=0)
    load_seconds: float = Field(default=0.0, ge=0.0)
    preprocess_seconds: float = Field(default=0.0, ge=0.0)
    prompt_encode_seconds: float = Field(default=0.0, ge=0.0)
    image_encode_seconds: float = Field(default=0.0, ge=0.0)
    latent_prepare_seconds: float = Field(default=0.0, ge=0.0)
    denoise_seconds: float = Field(default=0.0, ge=0.0)
    high_denoise_seconds: float = Field(default=0.0, ge=0.0)
    low_denoise_seconds: float = Field(default=0.0, ge=0.0)
    pipeline_seconds: float = Field(default=0.0, ge=0.0)
    pipeline_overhead_seconds: float = Field(default=0.0, ge=0.0)
    vae_decode_seconds: float = Field(default=0.0, ge=0.0)
    manual_vae_decode: bool = False
    vae_decode_chunk_frames: int = Field(default=0, ge=0)
    latent_frame_count: int = Field(default=0, ge=0)
    temporal_chunks: bool = False
    temporal_chunk_size: int = Field(default=0, ge=0)
    temporal_chunk_overlap: int = Field(default=0, ge=0)
    transformer_chunks_per_forward: int = Field(default=1, ge=1)
    transformer_forwards_per_step: int = Field(default=1, ge=1)
    video_postprocess_seconds: float = Field(default=0.0, ge=0.0)
    offload_cleanup_seconds: float = Field(default=0.0, ge=0.0)
    postprocess_seconds: float = Field(default=0.0, ge=0.0)
    video_write_seconds: float = Field(default=0.0, ge=0.0)
    steps_per_second: float | None = Field(default=None, ge=0.0)
    iterations_per_second: float | None = Field(default=None, ge=0.0)
    fp8_linear_layers: int = Field(default=0, ge=0)
    fp8_fast_mm_calls: int = Field(default=0, ge=0)
    fp8_fallback_calls: int = Field(default=0, ge=0)
    fp8_fallback_layers: int = Field(default=0, ge=0)
    fp8_fallback_reasons: list[str] = Field(default_factory=list)
    fp8_strict_mode: bool = False
    fp8_native_available: bool = False
    fp8_profile_enabled: bool = False
    fp8_backend: str = ""
    fp8_backend_metadata: dict[str, Any] = Field(default_factory=dict)
    fp8_linear_shape_count: int = Field(default=0, ge=0)
    fp8_linear_shapes: list[dict[str, Any]] = Field(default_factory=list)
    fp8_prepare_ms: float = Field(default=0.0, ge=0.0)
    fp8_scaled_mm_ms: float = Field(default=0.0, ge=0.0)
    fp8_bias_ms: float = Field(default=0.0, ge=0.0)
    fp8_fallback_ms: float = Field(default=0.0, ge=0.0)
    attention_backends: list[str] = Field(default_factory=list)
    attention_optimizations: list[str] = Field(default_factory=list)
    stage_transition_count: int = Field(default=0, ge=0)
    stage_transition_total_ms: float = Field(default=0.0, ge=0.0)
    stage_transition_h2d_ms: float = Field(default=0.0, ge=0.0)
    stage_transition_d2h_ms: float = Field(default=0.0, ge=0.0)
    stage_transition_cleanup_ms: float = Field(default=0.0, ge=0.0)
    stage_transition_events: list[dict[str, Any]] = Field(default_factory=list)
    hardware_fingerprint: dict[str, Any] = Field(default_factory=dict)
    transfer_probe: dict[str, Any] = Field(default_factory=dict)
    performance_benchmark_valid: bool = True
    performance_benchmark_notes: list[str] = Field(default_factory=list)
    cache_mode: str = ""
    vram_reserve_enabled: bool = False
    vram_reserve_mb: int = Field(default=0, ge=0)
    vram_limit_mb: int = Field(default=0, ge=0)
    vram_total_mb: int = Field(default=0, ge=0)
    vram_limit_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    message: str = ""
