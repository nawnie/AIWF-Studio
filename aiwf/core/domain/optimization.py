from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProfileReadiness(str, Enum):
    PRODUCTION = "production"
    BETA = "beta"
    EXPERIMENTAL = "experimental"
    AVOID = "avoid"


class PipelineKind(str, Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"
    HIRES = "hires"
    FAST = "fast"
    ENHANCE = "enhance"
    VIDEO = "video"
    TRAINING = "training"


class ModelFamily(str, Enum):
    SD15 = "sd15"
    SD2 = "sd2"
    SDXL = "sdxl"
    SDXL_TURBO = "sdxl_turbo"
    FLUX = "flux"
    SD3 = "sd3"
    UNKNOWN = "unknown"


class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    XFORMERS = "xformers"
    FLASH = "flash"
    SAGE = "sage"
    CUDNN = "cudnn"
    MATH = "math"
    AUTO = "auto"


class MemoryFormat(str, Enum):
    CONTIGUOUS = "contiguous"
    CHANNELS_LAST = "channels_last"
    AUTO = "auto"


class CpuOffloadPolicy(str, Enum):
    NONE = "none"
    MODEL = "model"
    SEQUENTIAL = "sequential"
    GROUP = "group"


class VaeSwitch(str, Enum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


class CompileTarget(str, Enum):
    NONE = "none"
    UNET = "unet"
    VAE_DECODE = "vae_decode"
    TRANSFORMER = "transformer"
    REGIONAL = "regional"
    MULTIPLE = "multiple"


class DTypePolicy(BaseModel):
    unet: str = "fp16"
    text_encoder: str = "fp16"
    vae: str = "fp16"
    controlnet: str = "fp16"
    allow_tf32: bool = True


class AttentionPolicy(BaseModel):
    name: AttentionBackend = AttentionBackend.SDPA
    fallback: AttentionBackend = AttentionBackend.SDPA
    requires_probe: bool = False


class MemoryPolicy(BaseModel):
    memory_format: MemoryFormat = MemoryFormat.CONTIGUOUS
    cpu_offload: CpuOffloadPolicy = CpuOffloadPolicy.NONE
    device_map: str | dict[str, Any] | None = None
    max_vram_bytes: int | None = None


class VaePolicy(BaseModel):
    slicing: VaeSwitch = VaeSwitch.OFF
    tiling: VaeSwitch = VaeSwitch.OFF
    force_upcast: bool | None = None
    vae_asset_policy: str = "user_selected"


class CompilePolicy(BaseModel):
    enabled: bool = False
    target: CompileTarget = CompileTarget.NONE
    mode: str | None = None
    fullgraph: bool | None = None
    dynamic: bool | None = None
    fixed_shapes: dict[str, Any] = Field(default_factory=dict)


class QuantPolicy(BaseModel):
    enabled: bool = False
    backend: str = "none"
    target_modules: list[str] = Field(default_factory=list)
    dtype: str | None = None


class EnginePolicy(BaseModel):
    enabled: bool = False
    backend: str = "pytorch"
    engine_profile_id: str | None = None


class SchedulerPolicy(BaseModel):
    scheduler: str = "automatic"
    steps: int | None = None
    cfg_scale: float | None = None
    notes: str = ""


class QualityModifiers(BaseModel):
    freeu: bool = False
    pag: bool = False
    refiner_sdxl: bool = False
    hires_fix: bool = False
    clip_skip_visible: bool = True


class FastMethod(BaseModel):
    name: str | None = None
    scheduler: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    requires_matching_checkpoint: bool = True


class CompatibilityRule(BaseModel):
    rule_id: str
    severity: str = "warning"
    reason: str
    fallback_profile_id: str = "safe_eager_cuda"


class OptimizationProfile(BaseModel):
    profile_id: str
    profile_version: str = "2026-06-16"
    display_name: str
    readiness: ProfileReadiness
    pipeline_kind: PipelineKind = PipelineKind.TXT2IMG
    model_family: ModelFamily = ModelFamily.UNKNOWN
    dtype_policy: DTypePolicy = Field(default_factory=DTypePolicy)
    attention_backend: AttentionPolicy = Field(default_factory=AttentionPolicy)
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    vae_policy: VaePolicy = Field(default_factory=VaePolicy)
    compile_policy: CompilePolicy = Field(default_factory=CompilePolicy)
    quant_policy: QuantPolicy = Field(default_factory=QuantPolicy)
    engine_policy: EnginePolicy = Field(default_factory=EnginePolicy)
    scheduler_policy: SchedulerPolicy = Field(default_factory=SchedulerPolicy)
    quality_modifiers: QualityModifiers = Field(default_factory=QualityModifiers)
    fast_method: FastMethod = Field(default_factory=FastMethod)
    compatibility: list[CompatibilityRule] = Field(default_factory=list)
    output_changes_visible: bool = False


class CapabilityFeature(BaseModel):
    available: bool
    version: str | None = None
    reason: str = ""


class GpuCapability(BaseModel):
    name: str = ""
    compute_capability: str = ""
    vram_total_bytes: int | None = None
    cuda_runtime: str = ""
    driver_version: str = ""


class CapabilityReport(BaseModel):
    report_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    os: str = ""
    python: str = ""
    platform: str = ""
    packages: dict[str, str | None] = Field(default_factory=dict)
    gpu: GpuCapability = Field(default_factory=GpuCapability)
    features: dict[str, CapabilityFeature] = Field(default_factory=dict)


class OptimizationRequest(BaseModel):
    profile_id: str = "balanced_sdpa_fp16"
    pipeline_kind: PipelineKind = PipelineKind.TXT2IMG
    model_family: ModelFamily = ModelFamily.UNKNOWN
    width: int | None = None
    height: int | None = None
    batch_size: int = 1
    lora_count: int = 0
    lora_targets_text_encoder: bool = False
    controlnet_count: int = 0
    experimental_flags: dict[str, bool] = Field(default_factory=dict)
    fast_method: str | None = None
    vae_tiling_requested: bool = False
    strict_quality_baseline: bool = False


class PlannerDecision(BaseModel):
    key: str
    decision: str
    reason: str
    severity: str = "info"


class OptimizationPlan(BaseModel):
    requested_profile_id: str
    effective_profile: OptimizationProfile
    capability_report_id: str | None = None
    decisions: list[PlannerDecision] = Field(default_factory=list)
    blocked: bool = False
    fallback_profile_id: str | None = None

    @property
    def profile_id(self) -> str:
        return self.effective_profile.profile_id


class BenchmarkTiming(BaseModel):
    load_time_s: float | None = None
    compile_or_build_time_s: float | None = None
    first_generation_time_s: float | None = None
    steady_state_times_s: list[float] = Field(default_factory=list)
    median_time_s: float | None = None
    p90_time_s: float | None = None
    denoise_time_s: float | None = None
    prompt_encode_time_s: float | None = None
    preprocess_time_s: float | None = None
    vae_decode_time_s: float | None = None
    postprocess_time_s: float | None = None


class BenchmarkMemory(BaseModel):
    torch_max_memory_allocated_bytes: int | None = None
    torch_max_memory_reserved_bytes: int | None = None
    nvml_peak_used_bytes: int | None = None
    cpu_rss_peak_bytes: int | None = None
    oom: bool = False
    oom_stage: str = ""


class BenchmarkOutputs(BaseModel):
    image_paths: list[str] = Field(default_factory=list)
    sha256_image: str | None = None
    phash: str | None = None
    artifact_labels: list[str] = Field(default_factory=list)
    human_verdict: str = ""


class BenchmarkReceipt(BaseModel):
    receipt_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    aiwf: dict[str, str | None] = Field(default_factory=dict)
    system: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, str | None] = Field(default_factory=dict)
    gpu: GpuCapability = Field(default_factory=GpuCapability)
    model: dict[str, Any] = Field(default_factory=dict)
    pipeline: dict[str, Any] = Field(default_factory=dict)
    optimization_profile: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    timing: BenchmarkTiming = Field(default_factory=BenchmarkTiming)
    memory: BenchmarkMemory = Field(default_factory=BenchmarkMemory)
    outputs: BenchmarkOutputs = Field(default_factory=BenchmarkOutputs)
    status: str = "completed"
