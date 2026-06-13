from __future__ import annotations

from pydantic import BaseModel, Field

from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions


class PhotoRestoreOptions(BaseModel):
    """Multi-stage old-photo restoration inspired by staged BOPBTL workflows.

    Stage order (when enabled):
    1. Pad dimensions to a model-friendly multiple
    2. Detect + inpaint scratches
    3. Global quality restore (denoise, color, contrast)
    4. Face detect → enhance → blend (GFPGAN / CodeFormer)
    5. Optional upscale
    """

    pad_multiple: int = Field(default=8, ge=1, le=64)
    scratch_detection: bool = True
    scratch_inpaint: bool = True
    scratch_sensitivity: float = Field(default=0.45, ge=0.05, le=0.95)
    scratch_dilation: int = Field(default=2, ge=0, le=32)
    global_restore: bool = True
    denoise_strength: float = Field(default=0.65, ge=0.0, le=1.0)
    color_boost: float = Field(default=0.55, ge=0.0, le=1.0)
    face_restore: bool = True
    restore: RestoreOptions | None = None
    upscale: UpscaleOptions | None = None