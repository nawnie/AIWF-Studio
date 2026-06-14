"""Training request domain models for AIWF Studio.

Covers:
  - Kohya LoRA training  (SD 1.x, SDXL, Flux)
  - EveryDream2 full fine-tuning / DreamBooth

Deliberately excluded:
  - Textual Inversion (TI) — use LoRA instead; TI produces weaker results
    and its training scripts (train_textual_inversion.py, train_ti.py) are
    not wired through AIWF Studio by design.

Both request types are plain Pydantic models so they can be JSON-serialised
and written to a job directory for the subprocess worker to read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

KOHYA_NETWORK_TYPES = ("LoRA", "LoCon", "LoHa", "LoKr", "DyLoRA")
"""LoRA-family network architectures supported by Kohya train_network.py."""

KOHYA_BASE_ARCHS = ("sd1", "sdxl", "flux")
"""Base model architecture families for Kohya."""

KOHYA_LR_SCHEDULERS = (
    "constant",
    "constant_with_warmup",
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "adafactor",
)

ED2_LR_SCHEDULERS = (
    "constant",
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
)

MIXED_PRECISION_OPTS = ("no", "fp16", "bf16")


# ---------------------------------------------------------------------------
# Kohya LoRA training request
# ---------------------------------------------------------------------------

class KohyaLoraRequest(BaseModel):
    """Parameters for a Kohya LoRA training job.

    Excludes all Textual Inversion fields — TI is not supported through AIWF.

    The worker translates this into a Kohya TOML config and calls the
    appropriate train_network.py script.
    """

    # ---- Identity ----
    job_name: str = Field(..., min_length=1, max_length=80,
        description="Short name used for output directories and checkpoint filenames.")

    # ---- Model ----
    base_model_path: str = Field(..., min_length=1,
        description="Path or HuggingFace ID of the base model to train on.")
    base_arch: Literal["sd1", "sdxl", "flux"] = Field(default="sdxl",
        description="Base model architecture. Selects the correct Kohya training script.")

    # ---- Dataset ----
    dataset_dir: str = Field(..., min_length=1,
        description="Directory containing the training images (and optional caption .txt files).")
    resolution: int = Field(default=1024, ge=256, le=2048,
        description="Training resolution in pixels (square). Use 512 for SD1, 1024 for SDXL/Flux.")
    caption_extension: str = Field(default=".txt",
        description="Caption file extension to look for alongside each image.")

    # ---- Output ----
    output_dir: str = Field(default="outputs/training/kohya",
        description="Directory where trained LoRA checkpoints are saved.")
    output_name: str = Field(default="",
        description="Base filename for saved checkpoints. Defaults to job_name if empty.")
    save_every_n_steps: int = Field(default=500, ge=50, le=10000)
    save_last_n_steps: int = Field(default=5, ge=1, le=50,
        description="Keep only the N most recent checkpoints.")

    # ---- Network ----
    network_module: Literal["networks.lora", "lycoris.kohya"] = Field(
        default="networks.lora",
        description="Kohya network module. Use 'lycoris.kohya' for LoCon/LoHa/LoKr/DyLoRA.")
    network_type: str = Field(default="LoRA",
        description=f"Network architecture. One of: {KOHYA_NETWORK_TYPES}")
    network_dim: int = Field(default=32, ge=1, le=512,
        description="LoRA rank. Higher = more capacity, larger file, slower training.")
    network_alpha: float = Field(default=16.0, ge=0.1, le=512.0,
        description="LoRA alpha (scaling factor). Common: dim/2 or dim.")

    # ---- Optimisation ----
    max_train_steps: int = Field(default=1500, ge=100, le=100_000)
    batch_size: int = Field(default=1, ge=1, le=64)
    learning_rate: float = Field(default=1e-4, ge=1e-8, le=1.0)
    unet_lr: float | None = Field(default=None,
        description="Separate U-Net learning rate. None = use learning_rate.")
    text_encoder_lr: float | None = Field(default=None,
        description="Separate text encoder learning rate. None = use learning_rate / 2.")
    lr_scheduler: str = Field(default="cosine_with_restarts",
        description=f"LR scheduler. One of: {KOHYA_LR_SCHEDULERS}")
    lr_warmup_steps: int = Field(default=100, ge=0, le=10_000)
    optimizer: str = Field(default="AdamW8bit",
        description="Optimizer. Options: AdamW, AdamW8bit, Lion, Prodigy, Adafactor.")
    mixed_precision: str = Field(default="bf16",
        description=f"Mixed precision mode. One of: {MIXED_PRECISION_OPTS}")
    gradient_checkpointing: bool = Field(default=True,
        description="Reduce VRAM at slight speed cost.")
    clip_grad_norm: float = Field(default=1.0, ge=0.0, le=100.0)

    # ---- Seed ----
    seed: int = Field(default=42, ge=0)

    @field_validator("network_type")
    @classmethod
    def _validate_network_type(cls, v: str) -> str:
        if v not in KOHYA_NETWORK_TYPES:
            raise ValueError(f"network_type must be one of {KOHYA_NETWORK_TYPES}, got {v!r}")
        return v

    @field_validator("lr_scheduler")
    @classmethod
    def _validate_lr_scheduler(cls, v: str) -> str:
        if v not in KOHYA_LR_SCHEDULERS:
            raise ValueError(f"lr_scheduler must be one of {KOHYA_LR_SCHEDULERS}, got {v!r}")
        return v

    @field_validator("mixed_precision")
    @classmethod
    def _validate_mixed_precision(cls, v: str) -> str:
        if v not in MIXED_PRECISION_OPTS:
            raise ValueError(f"mixed_precision must be one of {MIXED_PRECISION_OPTS}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _apply_defaults(self) -> "KohyaLoraRequest":
        if not self.output_name:
            object.__setattr__(self, "output_name", self.job_name)
        if self.unet_lr is None:
            object.__setattr__(self, "unet_lr", self.learning_rate)
        if self.text_encoder_lr is None:
            object.__setattr__(self, "text_encoder_lr", self.learning_rate / 2)
        return self

    def dataset_path(self) -> Path:
        return Path(self.dataset_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def training_script(self) -> str:
        """Return the Kohya training script name for this architecture."""
        return {
            "sd1": "sd_scripts/train_network.py",
            "sdxl": "sd_scripts/sdxl_train_network.py",
            "flux": "flux_train_network.py",
        }[self.base_arch]


# ---------------------------------------------------------------------------
# EveryDream2 full fine-tuning request
# ---------------------------------------------------------------------------

class ED2TrainingRequest(BaseModel):
    """Parameters for an EveryDream2 full model fine-tuning job.

    ED2 trains full model checkpoints (not adapters like LoRA).
    It supports DreamBooth-style training and full fine-tuning but
    does not do Textual Inversion — that design is intentional.

    The worker writes a train.json for ED2 and calls train.py.
    """

    # ---- Identity ----
    job_name: str = Field(..., min_length=1, max_length=80)

    # ---- Model ----
    base_model_path: str = Field(..., min_length=1,
        description="Path to base .ckpt or .safetensors checkpoint, or HuggingFace ID.")
    vae_path: str = Field(default="",
        description="Optional explicit VAE path. Leave empty to use the model's built-in VAE.")

    # ---- Dataset ----
    dataset_dir: str = Field(..., min_length=1,
        description="Directory containing training images (with caption .txt files).")
    resolution: int = Field(default=512, ge=256, le=1024)
    flip_p: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Probability of random horizontal flip augmentation.")

    # ---- Output ----
    output_dir: str = Field(default="outputs/training/ed2")
    save_every_n_epochs: int = Field(default=1, ge=1, le=100)
    save_last_n_epochs: int = Field(default=3, ge=1, le=50)
    ckpt_type: Literal["safetensors", "ckpt"] = Field(default="safetensors",
        description="Checkpoint format. safetensors is strongly recommended.")

    # ---- Training ----
    max_epochs: int = Field(default=20, ge=1, le=1000)
    batch_size: int = Field(default=4, ge=1, le=64)
    lr: float = Field(default=1.5e-6, ge=1e-9, le=1.0,
        description="Global learning rate for full fine-tuning.")
    lr_scheduler: str = Field(default="constant",
        description=f"LR scheduler. One of: {ED2_LR_SCHEDULERS}")
    lr_warmup_steps: int = Field(default=0, ge=0, le=10_000)
    optimizer: str = Field(default="adamw",
        description="Optimizer. Options: adamw, adamw8bit, lion, adafactor.")
    mixed_precision: str = Field(default="bf16")
    gradient_checkpointing: bool = Field(default=True)
    clip_skip: int = Field(default=2, ge=1, le=4,
        description="Number of CLIP layers to skip. 2 is standard for most SD1.x models.")
    seed: int = Field(default=42, ge=0)

    # ---- Logging ----
    log_dir: str = Field(default="outputs/logs/ed2")
    sample_steps: int = Field(default=0, ge=0,
        description="Generate sample images every N steps. 0 = disabled.")
    sample_prompts: list[str] = Field(default_factory=list,
        description="Prompts used for sample generation during training.")

    @field_validator("lr_scheduler")
    @classmethod
    def _validate_lr_scheduler(cls, v: str) -> str:
        if v not in ED2_LR_SCHEDULERS:
            raise ValueError(f"lr_scheduler must be one of {ED2_LR_SCHEDULERS}, got {v!r}")
        return v

    @field_validator("mixed_precision")
    @classmethod
    def _validate_mixed_precision(cls, v: str) -> str:
        if v not in MIXED_PRECISION_OPTS:
            raise ValueError(f"mixed_precision must be one of {MIXED_PRECISION_OPTS}, got {v!r}")
        return v

    def dataset_path(self) -> Path:
        return Path(self.dataset_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_ed2_config(self) -> dict:
        """Render a train.json dict that ED2's train.py will accept."""
        cfg: dict = {
            "model": self.base_model_path,
            "train_data_dir": str(self.dataset_path()),
            "output_dir": str(self.output_path()),
            "log_dir": self.log_dir,
            "resolution": self.resolution,
            "flip_p": self.flip_p,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "lr_scheduler": self.lr_scheduler,
            "lr_warmup_steps": self.lr_warmup_steps,
            "optimizer": self.optimizer,
            "mixed_precision": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "clip_skip": self.clip_skip,
            "seed": self.seed,
            "save_every_n_epochs": self.save_every_n_epochs,
            "save_last_n_epochs": self.save_last_n_epochs,
            "ckpt_type": self.ckpt_type,
        }
        if self.vae_path:
            cfg["vae"] = self.vae_path
        if self.sample_steps > 0 and self.sample_prompts:
            cfg["sample_steps"] = self.sample_steps
            cfg["sample_prompts"] = self.sample_prompts
        return cfg
