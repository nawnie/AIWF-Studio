from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part[:1].upper() + part[1:] for part in rest)


class RuntimeOverlayModel(BaseModel):
    """Base model with API-friendly camelCase aliases."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)


class RuntimeOverlayStatus(str, Enum):
    ACTIVE = "active"
    STARTED = "started"
    CANDIDATE = "candidate"
    ADAPTER_READY = "adapter-ready"
    INSTALLED = "installed"
    DISABLED = "disabled"


class RuntimeOverlayPatchPoint(str, Enum):
    BEFORE_PROMPT_EXPAND = "before_prompt_expand"
    BEFORE_PROMPT_ENCODE = "before_prompt_encode"
    AFTER_PROMPT_ENCODE = "after_prompt_encode"
    BEFORE_MODEL_LOAD = "before_model_load"
    AFTER_MODEL_LOAD = "after_model_load"
    BEFORE_SAMPLE = "before_sample"
    DURING_SAMPLE_PRE_CFG = "during_sample_pre_cfg"
    DURING_SAMPLE_CFG = "during_sample_cfg"
    DURING_SAMPLE_POST_CFG = "during_sample_post_cfg"
    AFTER_SAMPLE = "after_sample"
    BEFORE_VAE_DECODE = "before_vae_decode"
    AFTER_VAE_DECODE = "after_vae_decode"
    AFTER_IMAGE = "after_image"
    RECEIPT_WRITE = "receipt_write"


PATCH_POINT_ORDER: dict[str, int] = {point.value: index for index, point in enumerate(RuntimeOverlayPatchPoint)}


class RuntimeOverlayClass(str, Enum):
    PROMPT = "prompt"
    CONDITIONING = "conditioning"
    LATENT = "latent"
    MODEL = "model"
    UNET = "unet"
    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"
    SAMPLER = "sampler"
    METADATA = "metadata"
    ARTIFACT = "artifact"


class RuntimeOverlayMemoryLease(RuntimeOverlayModel):
    vram_mb: int = Field(default=0, ge=0)
    cpu_ram_mb: int = Field(default=0, ge=0)
    ssd_cache_mb: int = Field(default=0, ge=0)
    policy: str = "none"

    def add(self, other: "RuntimeOverlayMemoryLease") -> "RuntimeOverlayMemoryLease":
        return RuntimeOverlayMemoryLease(
            vram_mb=self.vram_mb + other.vram_mb,
            cpu_ram_mb=self.cpu_ram_mb + other.cpu_ram_mb,
            ssd_cache_mb=self.ssd_cache_mb + other.ssd_cache_mb,
            policy="combined",
        )


class RuntimeOverlayContract(RuntimeOverlayModel):
    """Declared model patch/overlay capability.

    Contracts describe what an overlay is allowed to touch. They do not execute
    arbitrary plugin code by themselves.
    """

    id: str
    label: str
    status: RuntimeOverlayStatus | str = RuntimeOverlayStatus.CANDIDATE
    source: str = "aiwf-core"
    families: list[str] = Field(default_factory=lambda: ["unknown"])
    targets: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    changes_pixels: bool = True
    requires_gpu: bool = True
    safe_with_compile: bool = False
    safe_with_controlnet: bool = True
    safe_with_lora: bool = True
    memory_lease: RuntimeOverlayMemoryLease = Field(default_factory=RuntimeOverlayMemoryLease)
    summary: str = ""
    receipt_fields: list[str] = Field(default_factory=list)

    @property
    def phase_index(self) -> int:
        return min((PATCH_POINT_ORDER.get(str(phase), 999) for phase in self.phases), default=999)


class RuntimeOverlayRegistry(RuntimeOverlayModel):
    schema_name: str = Field(default="aiwf.runtime-overlay.registry.v1", alias="schema")
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    classes: list[str] = Field(default_factory=lambda: [item.value for item in RuntimeOverlayClass])
    patch_points: list[str] = Field(default_factory=lambda: [item.value for item in RuntimeOverlayPatchPoint])
    transaction: dict[str, str] = Field(
        default_factory=lambda: {
            "begin": "snapshot original references, prompt cache keys, and model placement",
            "lease": "reserve declared VRAM/RAM/SSD budget or block with a clear reason",
            "apply": "apply overlays in deterministic patch-point order",
            "run": "execute the owning generation phase",
            "receipt": "write structured overlay metadata into output/project receipts",
            "rollback": "restore originals, release memory lease, and clear temporary tensors",
        }
    )
    overlays: list[RuntimeOverlayContract] = Field(default_factory=list)


class RuntimeOverlayValidateRequest(RuntimeOverlayModel):
    model_family: str = "unknown"
    pipeline_kind: str = "txt2img"
    overlays: list[Any] = Field(default_factory=list)
    compile_enabled: bool = False
    controlnet_enabled: bool = False
    lora_count: int = 0
    requested_memory_mb: int = 0
    settings: dict[str, Any] = Field(default_factory=dict)


class RuntimeOverlayPlan(RuntimeOverlayModel):
    schema_name: str = Field(default="aiwf.runtime-overlay.plan.v1", alias="schema")
    id: str = Field(default_factory=lambda: f"overlay-plan-{uuid4().hex[:8]}")
    model_family: str = "unknown"
    pipeline_kind: str = "txt2img"
    ordered_overlay_ids: list[str] = Field(default_factory=list)
    memory_lease: RuntimeOverlayMemoryLease = Field(default_factory=RuntimeOverlayMemoryLease)
    transaction_required: bool = False
    receipt_required: bool = False
    rollback_required: bool = False


class RuntimeOverlayValidationResult(RuntimeOverlayModel):
    valid: bool = True
    blocked: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    plan: RuntimeOverlayPlan = Field(default_factory=RuntimeOverlayPlan)
    ordered_overlays: list[RuntimeOverlayContract] = Field(default_factory=list)


class RuntimeOverlayReceipt(RuntimeOverlayModel):
    schema_name: str = Field(default="aiwf.runtime-overlay.receipt.v1", alias="schema")
    id: str = Field(default_factory=lambda: f"overlay-receipt-{uuid4().hex[:10]}")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    project_id: str = "default"
    job_id: str = ""
    model_family: str = "unknown"
    model_id: str = ""
    pipeline_kind: str = "txt2img"
    overlays: list[Any] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
