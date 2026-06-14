from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiwf.core.domain.prompt_style import PromptStyle


class RuntimeFlags(BaseSettings):
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
    no_half: bool = False
    fp8: bool = False
    directml: bool = False
    medvram: bool = False
    lowvram: bool = False
    xformers: bool = False
    opt_sdp_attention: bool = False
    opt_split_attention: bool = False
    async_offload: bool = True
    pinned_memory: bool = True
    cuda_malloc: bool = True
    cpu: bool = False
    skip_install: bool = False
    skip_prepare_environment: bool = False
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3])
    models_dir: Path | None = None
    ckpt_dir: Path | None = None
    output_dir: Path | None = None
    extra_model_dirs: list[Path] = Field(default_factory=list)
    extra_ckpt_dirs: list[Path] = Field(default_factory=list)
    vae_path: Path | None = None
    default_checkpoint: Path | None = None

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
    model_config = SettingsConfigDict(extra="ignore")

    save_images: bool = True
    embed_metadata: bool = True
    txt2img_output_subdir: str = "txt2img-images"
    img2img_output_subdir: str = "img2img-images"
    inpaint_output_subdir: str = "inpaint-images"
    enhance_output_subdir: str = "enhanced-images"
    rife_output_subdir: str = "rife-videos"
    workflow_output_subdir: str = "workflow-images"
    workflows_dir: str = "workflows"
    upscale_tile_size: int = Field(default=256, ge=0, le=2048)
    upscale_tile_overlap: int = Field(default=32, ge=0, le=512)
    auto_launch_browser: bool = True
    enable_live_preview: bool = True
    show_progress_every_n_steps: int = Field(default=1, ge=1, le=20)
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
    hidden_tabs: list[str] = Field(default_factory=list)

    # Generation defaults — applied as the Studio tab's initial values.
    default_sampler: str = "euler_a"
    default_scheduler: str = "automatic"
    default_steps: int = Field(default=20, ge=1, le=150)
    default_cfg_scale: float = Field(default=7.0, ge=1.0, le=30.0)
    default_width: int = Field(default=512, ge=64, le=2048)
    default_height: int = Field(default=512, ge=64, le=2048)
    default_clip_skip: int = Field(default=1, ge=1, le=12)

    # Last checkpoint the user loaded in Studio — restored on next launch.
    last_checkpoint_id: str | None = None

    # Last Wan video settings — restored when the video tab opens.
    last_wan_high: str = ""
    last_wan_low: str = ""
    last_wan_vae: str = ""
    last_wan_text_encoder: str = ""
    last_wan_offload: str = "model"

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
    # Capture intermediate / partial images. Both require backend support to take
    # effect and are no-ops until that lands (the setting is persisted regardless).
    save_before_hires: bool = False
    save_interrupted: bool = False

    # Saved metadata controls. These enrich the infotext written to PNG text
    # chunks and optional sidecar .txt files without changing request parsing.
    metadata_include_model_hash: bool = True
    metadata_include_vae_hash: bool = True
    metadata_include_lora_hashes: bool = True
    metadata_include_app_version: bool = True

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

    # API keys for model downloads (stored locally in config.json).
    huggingface_token: str = ""
    civitai_token: str = ""

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
