from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiwf.core.domain.prompt_style import PromptStyle

VRAM_PROFILES = {"cpu", "low", "mid", "normal", "high"}


def normalize_vram_profile(value: str | None) -> str:
    normalized = (value or "normal").strip().lower().replace("-", "_")
    if normalized in {"med", "medium"}:
        normalized = "mid"
    if normalized in {"default", "balanced"}:
        normalized = "normal"
    if normalized not in VRAM_PROFILES:
        raise ValueError("vram_profile must be cpu, low, mid, normal, or high")
    return normalized


class RuntimeFlags(BaseSettings):
    """Process/runtime knobs sourced from env, CLI, and saved launch profiles.

    Defaults are intentionally local-first: no listener, no public share, API
    disabled, and private/LAN download URLs blocked unless the user opts in.
    """

    model_config = SettingsConfigDict(env_prefix="AIWF_", extra="ignore")

    port: int = 7860
    listen: bool = False
    share: bool = False
    autolaunch: bool = False
    api: bool = False
    nowebui: bool = False
    theme: str = "dark"
    gradio_auth: str | None = None
    api_cors_origins: str = ""
    api_rate_limit_per_minute: int = 0
    block_private_download_urls: bool = True
    gerror: bool = False
    genlog: bool = False
    no_half: bool = False
    fp8: bool = False
    fluxfp8: bool = False
    directml: bool = False
    # Custom engine feature flags (set via env AIWF_* or launch profile)
    cuda_graphs: bool = False
    torchao: bool = False
    torch_compile: bool = False
    channels_last: bool = False
    fp8_quant: bool = False       # TorchAO FP8 (distinct from --fp8 half-precision)
    nvenc: bool = False
    hevc: bool = False
    # Inference backend: "diffusers" (default) or "onnx"
    inference_backend: str = "diffusers"
    onnx_provider: str = "auto"   # auto | cuda | directml | cpu
    vram_profile: str = "normal"  # cpu | low | mid | normal | high
    medvram: bool = False
    lowvram: bool = False
    highvram: bool = False
    attention_backend: str = "sdpa"
    xformers: bool = False
    opt_sdp_attention: bool = False
    opt_split_attention: bool = False
    async_offload: bool = True
    pinned_memory: bool = True
    cuda_malloc: bool = False
    cpu: bool = False
    skip_install: bool = False
    skip_prepare_environment: bool = False
    # Resolve data roots late so portable checkouts can use repo-local
    # models/outputs, while launch profiles can point at shared model disks.
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3])
    models_dir: Path | None = None
    ckpt_dir: Path | None = None
    output_dir: Path | None = None
    extra_model_dirs: list[Path] = Field(default_factory=list)
    extra_ckpt_dirs: list[Path] = Field(default_factory=list)
    nvidia_vfx_sdk_root: Path | None = None
    vsr_video_effects_app: Path | None = None
    vsr_upscale_app: Path | None = None
    videofx_denoise_app: Path | None = None
    videofx_aigs_app: Path | None = None
    videofx_relight_app: Path | None = None
    vsr_model_dir: Path | None = None
    vae_path: Path | None = None
    default_checkpoint: Path | None = None

    @field_validator("attention_backend")
    @classmethod
    def validate_attention_backend(cls, value: str) -> str:
        normalized = (value or "sdpa").strip().lower().replace("-", "_")
        if normalized in {"sage", "sageattention"}:
            normalized = "sage_sdpa"
        if normalized not in {"sage_sdpa", "sdpa", "xformers", "none"}:
            raise ValueError("attention_backend must be sage_sdpa, sdpa, xformers, or none")
        return normalized

    @field_validator("vram_profile")
    @classmethod
    def validate_vram_profile(cls, value: str) -> str:
        return normalize_vram_profile(value)

    def effective_vram_profile(self) -> str:
        if self.cpu:
            return "cpu"
        if self.lowvram:
            return "low"
        if self.medvram:
            return "mid"
        if self.highvram:
            return "high"
        return normalize_vram_profile(self.vram_profile)

    def resolved_models_dir(self) -> Path:
        return (self.models_dir or self.data_dir / "models").resolve()

    def resolved_ckpt_dir(self) -> Path:
        return (self.ckpt_dir or self.resolved_models_dir() / "Stable-diffusion").resolve()

    def resolved_output_dir(self) -> Path:
        return (self.output_dir or self.data_dir / "outputs").resolve()

    def resolved_extra_model_dirs(self) -> list[Path]:
        return [path.resolve() for path in self.extra_model_dirs if path]

    def resolved_extra_ckpt_dirs(self) -> list[Path]:
        return [path.resolve() for path in self.extra_ckpt_dirs if path]


class UserSettings(BaseSettings):
    """Persisted UI and generation preferences stored in config.json.

    These are user choices, not launch-time environment detection. Keep field
    names stable so existing local installs can round-trip old config files.
    """

    model_config = SettingsConfigDict(extra="ignore")

    save_images: bool = True
    embed_metadata: bool = True
    txt2img_output_subdir: str = "txt2img-images"
    img2img_output_subdir: str = "img2img-images"
    inpaint_output_subdir: str = "inpaint-images"
    enhance_output_subdir: str = "enhanced-images"
    rife_output_subdir: str = "rife-videos"
    vsr_output_subdir: str = "vsr-videos"
    audio_output_subdir: str = "audio"
    audio_video_output_subdir: str = "audio-videos"
    workflow_output_subdir: str = "workflow-images"
    workflows_dir: str = "workflows"
    upscale_tile_size: int = Field(default=256, ge=0, le=2048)
    upscale_tile_overlap: int = Field(default=32, ge=0, le=512)
    auto_launch_browser: bool = True
    enable_live_preview: bool = True
    show_progress_every_n_steps: int = Field(default=5, ge=1, le=20)
    live_preview_decoder: str = "vae"
    live_preview_title_progress: bool = True
    recent_tags: list[str] = Field(default_factory=list)
    generation_cooldown_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    lora_aliases: dict[str, str] = Field(default_factory=dict)
    lora_defaults: dict[str, float] = Field(default_factory=dict)
    lora_keywords: dict[str, str] = Field(default_factory=dict)
    prompts_dir: str = "prompts"
    wildcards_dir: str = "wildcards"
    prompt_styles: list[PromptStyle] = Field(default_factory=list)
    accent_preset: str = "mint"
    modern_onboarding_seen: bool = False
    github_avatar_url: str = "https://github.com/nawnie.png?size=160"
    # Hidden tabs keep unfinished/heavy surfaces out of the default local UI
    # without deleting the underlying feature routes.
    hidden_tabs: list[str] = Field(
        default_factory=lambda: [
            "Audio",
            "Models",
            "Enhance",
            "Segment",
            "Face Swap",
            "RIFE",
            "Library",
            "PNG Info",
            "History",
            "Workflows",
        ]
    )

    # Generation defaults — applied as the Studio tab's initial values.
    default_sampler: str = "euler_a"
    default_scheduler: str = "automatic"
    default_steps: int = Field(default=20, ge=1, le=150)
    default_cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    default_width: int = Field(default=512, ge=64, le=2048)
    default_height: int = Field(default=512, ge=64, le=2048)
    default_clip_skip: int = Field(default=1, ge=1, le=12)
    default_hr_upscaler: str = "lanczos"

    @field_validator("default_cfg_scale")
    @classmethod
    def clamp_cfg_scale(cls, v: float) -> float:
        return max(1.0, min(20.0, float(v)))

    # Last checkpoint the user loaded in Studio — restored on next launch.
    last_checkpoint_id: str | None = None

    # Per-checkpoint remembered generation settings (steps/cfg/sampler/etc.),
    # keyed by checkpoint id. Applied as UI defaults when a model is selected,
    # falling back to architecture sane-defaults (see model_presets.py) for
    # checkpoints that have never been run before.
    model_settings: dict[str, dict[str, object]] = Field(default_factory=dict)

    # Last Wan video settings — restored when the video tab opens.
    # Restored as UI hints only; Wan request validation still checks route and
    # model compatibility before anything is loaded.
    last_wan_high: str = ""
    last_wan_low: str = ""
    last_wan_vae: str = ""
    last_wan_text_encoder: str = ""
    last_wan_offload: str = "balanced"
    last_wan_sampler: str = "unipc"
    last_wan_flow_shift: float = 5.0
    last_wan_runtime_mode: str = "fast_5b"

    # Saved image format — "png" keeps infotext metadata; jpg/webp are smaller files.
    image_format: str = "png"
    image_quality: int = Field(default=95, ge=10, le=100)

    # Saving & Output parity (A1111-style). Defaults preserve prior behavior:
    # one image per file, timestamp filenames, no grids, no sidecars.
    save_grid: bool = False
    save_sidecar_txt: bool = False
    # Filename template. Tokens: [datetime] [date] [time] [seed] [model_name]
    # [width] [height] [seq]. "[datetime]" reproduces the legacy timestamp name.
    filename_pattern: str = "[datetime]"
    # Capture intermediate / partial images.
    save_before_hires: bool = False
    save_interrupted: bool = False

    # Saved metadata controls. These enrich the infotext written to PNG text
    # chunks and optional sidecar .txt files without changing request parsing.
    metadata_include_model_hash: bool = True
    metadata_include_vae_hash: bool = True
    metadata_include_lora_hashes: bool = True
    metadata_include_app_version: bool = True
    metadata_include_optimization_profile: bool = True
    optimization_profile_id: str = "balanced_sdpa_fp16"

    # Optional SDXL refiner. This is intentionally explicit instead of tied to a
    # profile so user-visible quality controls remain under the user's control.
    sdxl_refiner_enabled: bool = False
    sdxl_refiner_checkpoint_id: str | None = None
    sdxl_refiner_steps: int = Field(default=10, ge=1, le=150)
    sdxl_refiner_strength: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("default_hr_upscaler")
    @classmethod
    def validate_default_hr_upscaler(cls, value: str) -> str:
        normalized = (value or "lanczos").strip().lower().replace(" ", "_")
        if normalized == "latent":
            normalized = "lanczos"
        if normalized not in {"lanczos", "bicubic", "nearest"}:
            raise ValueError("default_hr_upscaler must be lanczos, bicubic, or nearest")
        return normalized

    # PNG Info handoff behavior.
    pnginfo_send_to_studio: bool = True
    pnginfo_clear_after_apply: bool = True

    # Sampler/guidance safety + convenience.
    # auto_cfg_for_distilled: clamp CFG on Lightning/Hyper-SD/Turbo/LCM/TCD models
    # (they overexpose at normal CFG). use_default_negative: fill a generic quality
    # negative when the user leaves it blank; default_negative_prompt overrides the
    # built-in text (blank = use built-in).
    auto_cfg_for_distilled: bool = True
    use_default_negative: bool = True
    default_negative_prompt: str = ""

    # Gallery & viewer preferences.
    gallery_height: int = Field(default=480, ge=120, le=1200)
    gallery_columns: int = Field(default=2, ge=1, le=8)
    send_seed_on_click: bool = True
    send_size_on_click: bool = True

    # Download safety. prefer_safetensors warns before starting a .ckpt/.pt
    # download. write_download_receipts writes a companion .json alongside each
    # downloaded file so you can trace what was fetched and when.
    prefer_safetensors: bool = True
    write_download_receipts: bool = True

    # API keys for model downloads (stored locally in config.json).
    # Stored locally for download helpers only; do not copy into metadata,
    # receipts, logs, or UI previews.
    huggingface_token: str = ""
    civitai_token: str = ""

    # ONNX model root — directory containing one or more ONNX model subdirs.
    # Only used when inference_backend == "onnx".
    onnx_model_dir: str = ""

    # Video pipeline performance (Wan / LTX). These are user-adjustable and
    # applied to the process environment on startup and on every save, so the
    # next pipeline load picks them up without a restart.
    # ltx_dtype: bf16 matches how LTX-Video is distributed/calibrated; fp16 is
    # a fallback for pre-Ampere GPUs without bf16 support.
    ltx_dtype: str = "bf16"
    # ltx_cpu_offload: auto = offload only when the checkpoint is too big to
    # stay resident; model = always model-offload; none = keep fully on GPU.
    ltx_cpu_offload: str = "auto"
    # Streamed group offload overlaps Wan block transfers with compute
    # (needs pinned memory; costs a little extra VRAM headroom).
    wan_group_offload_stream: bool = True
    wan_group_offload_blocks: int = Field(default=4, ge=1, le=40)
    # Diffusers optimized GGUF CUDA kernels (needs the `kernels` package).
    gguf_cuda_kernels: bool = False

    # Advanced Wan runtime knobs. Every default mirrors the behavior the app
    # shipped with — changing these is opt-in tuning, not a migration.
    # SageAttention preference: auto = use when installed (default),
    # force = require it (warn when missing), off = plain torch SDPA.
    wan_sage_attention: str = "auto"
    # AIWF's own denoise loop vs diffusers' pipe() as a black box.
    wan_native_denoise: bool = True
    # Manual chunked VAE decode trades decode speed for lower peak VRAM.
    wan_manual_vae_decode: bool = False
    wan_vae_chunk_frames: int = Field(default=4, ge=1, le=16)
    # Streamed offload internals: CUDA record_stream + low CPU memory staging.
    wan_group_offload_record_stream: bool = True
    wan_group_offload_low_cpu_mem: bool = True
    # Minimum total VRAM (GB) before dual FP8 stages may co-reside ("resident").
    wan_resident_min_vram_gb: int = Field(default=20, ge=8, le=96)

    # User extensions (plugins/<folder>) disabled by folder name. Disabled
    # extensions stay listed in Settings but are never imported at boot.
    disabled_extensions: list[str] = Field(default_factory=list)

    @field_validator("wan_sage_attention")
    @classmethod
    def validate_wan_sage_attention(cls, value: str) -> str:
        normalized = (value or "auto").strip().lower()
        if normalized not in {"auto", "force", "off"}:
            raise ValueError("wan_sage_attention must be auto, force, or off")
        return normalized

    @field_validator("ltx_dtype")
    @classmethod
    def validate_ltx_dtype(cls, value: str) -> str:
        normalized = (value or "bf16").strip().lower()
        if normalized in {"bfloat16", "bf16"}:
            return "bf16"
        if normalized in {"float16", "fp16", "half"}:
            return "fp16"
        raise ValueError("ltx_dtype must be bf16 or fp16")

    @field_validator("ltx_cpu_offload")
    @classmethod
    def validate_ltx_cpu_offload(cls, value: str) -> str:
        normalized = (value or "auto").strip().lower()
        if normalized not in {"auto", "model", "none"}:
            raise ValueError("ltx_cpu_offload must be auto, model, or none")
        return normalized

    def apply_video_perf_env(self) -> None:
        """Push video performance preferences into the process environment.

        The Wan/LTX loaders read these env knobs at pipeline-load time, so
        applying them on save means the next generation uses the new values
        without restarting the app.
        """
        import os

        os.environ["AIWF_LTX_DTYPE"] = self.ltx_dtype
        os.environ["AIWF_LTX_CPU_OFFLOAD"] = self.ltx_cpu_offload
        os.environ["AIWF_WAN_GROUP_OFFLOAD_STREAM"] = "1" if self.wan_group_offload_stream else "0"
        os.environ["AIWF_WAN_GROUP_OFFLOAD_BLOCKS"] = str(int(self.wan_group_offload_blocks))
        # The fused GGUF kernels (Hub repo Isotr0py/ggml) only ship Linux
        # builds; enabling the flag on Windows makes diffusers' GGUF import
        # crash, so the toggle is Linux-gated regardless of the saved value.
        gguf_kernels = self.gguf_cuda_kernels and os.name != "nt"
        os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "1" if gguf_kernels else "0"
        # Advanced Wan runtime. "auto" removes the env var so the runtime's
        # own detection (use sage when installed) stays in charge.
        if self.wan_sage_attention == "auto":
            os.environ.pop("AIWF_WAN_SAGE_ATTENTION", None)
        else:
            os.environ["AIWF_WAN_SAGE_ATTENTION"] = "1" if self.wan_sage_attention == "force" else "0"
        os.environ["AIWF_WAN_NATIVE_DENOISE"] = "1" if self.wan_native_denoise else "0"
        os.environ["AIWF_WAN_MANUAL_VAE_DECODE"] = "1" if self.wan_manual_vae_decode else "0"
        os.environ["AIWF_WAN_VAE_CHUNK_FRAMES"] = str(int(self.wan_vae_chunk_frames))
        os.environ["AIWF_WAN_GROUP_OFFLOAD_RECORD_STREAM"] = "1" if self.wan_group_offload_record_stream else "0"
        os.environ["AIWF_WAN_GROUP_OFFLOAD_LOW_CPU_MEM"] = "1" if self.wan_group_offload_low_cpu_mem else "0"
        os.environ["AIWF_WAN_RESIDENT_MIN_VRAM_MB"] = str(int(self.wan_resident_min_vram_gb) * 1024)

    def apply_token_env(self) -> None:
        """Expose saved API keys to download helpers via environment variables."""
        import os

        hf = self.huggingface_token.strip()
        if hf:
            os.environ["HF_TOKEN"] = hf
            os.environ["HUGGINGFACE_TOKEN"] = hf
        civitai = self.civitai_token.strip()
        if civitai:
            os.environ["CIVITAI_API_TOKEN"] = civitai

    def live_preview_interval(self) -> int:
        """Steps between latent decode previews during streaming generation (0 = off)."""
        if not self.enable_live_preview:
            return 0
        if self.live_preview_decoder != "vae":
            return 0
        return max(1, min(20, self.show_progress_every_n_steps))

    def live_preview_summary(self) -> str:
        if not self.enable_live_preview or self.live_preview_interval() == 0:
            return "Live preview off"
        n = self.live_preview_interval()
        decoder = "VAE decode"
        if n == 1:
            return f"Live preview every step ({decoder})"
        return f"Live preview every {n} steps ({decoder})"
