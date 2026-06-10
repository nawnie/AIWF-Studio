from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class FaceSwapOptions(BaseModel):
    """ReActor-style face swap options."""

    source_face_index: int = Field(default=0, ge=0)  # which detected face in the source image
    target_face_index: int = Field(default=-1, ge=-1)  # -1 = swap every face in the target
    model_id: str = "inswapper_128"
    restore_face: bool = True
    restorer_id: str | None = None  # an EnhanceService restorer (GFPGAN / CodeFormer)
    restore_visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    codeformer_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class FaceSwapModelInfo(BaseModel):
    id: str
    title: str
    path: str

    @classmethod
    def from_path(cls, path: Path) -> "FaceSwapModelInfo":
        return cls(id=path.stem, title=path.stem, path=str(path))
