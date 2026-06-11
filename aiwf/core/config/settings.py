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
    no_half: bool = False
    medvram: bool = False
    lowvram: bool = False
    xformers: bool = False
    opt_sdp_attention: bool = False
    opt_split_attention: bool = False
    cpu: bool = False
    skip_install: bool = False
    skip_prepare_environment: bool = False
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3])
    models_dir: Path | None = None
    ckpt_dir: Path | None = None
    output_dir: Path | None = None
    vae_path: Path | None = None
    default_checkpoint: Path | None = None

    def resolved_models_dir(self) -> Path:
        return (self.models_dir or self.data_dir / "models").resolve()

    def resolved_ckpt_dir(self) -> Path:
        return (self.ckpt_dir or self.resolved_models_dir() / "Stable-diffusion").resolve()

    def resolved_output_dir(self) -> Path:
        return (self.output_dir or self.data_dir / "outputs").resolve()


class UserSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    save_images: bool = True
    embed_metadata: bool = True
    txt2img_output_subdir: str = "txt2img-images"
    img2img_output_subdir: str = "img2img-images"
    inpaint_output_subdir: str = "inpaint-images"
    enhance_output_subdir: str = "enhanced-images"
    workflow_output_subdir: str = "workflow-images"
    workflows_dir: str = "workflows"
    upscale_tile_size: int = Field(default=256, ge=0, le=2048)
    upscale_tile_overlap: int = Field(default=32, ge=0, le=512)
    auto_launch_browser: bool = True
    enable_live_preview: bool = True
    show_progress_every_n_steps: int = Field(default=1, ge=1, le=20)
    recent_tags: list[str] = Field(default_factory=list)
    generation_cooldown_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    lora_aliases: dict[str, str] = Field(default_factory=dict)
    lora_defaults: dict[str, float] = Field(default_factory=dict)
    lora_keywords: dict[str, str] = Field(default_factory=dict)
    prompts_dir: str = "prompts"
    wildcards_dir: str = "wildcards"
    prompt_styles: list[PromptStyle] = Field(default_factory=list)

    def live_preview_interval(self) -> int:
        """Steps between latent decode previews during streaming generation (0 = off)."""
        if not self.enable_live_preview:
            return 0
        return max(1, min(20, self.show_progress_every_n_steps))

    def live_preview_summary(self) -> str:
        if not self.enable_live_preview:
            return "Live preview off"
        n = self.live_preview_interval()
        if n == 1:
            return "Live preview every step"
        return f"Live preview every {n} steps"