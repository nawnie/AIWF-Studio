from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


def parse_face_indices(value, default: list[int]) -> list[int]:
    """Parse a comma-separated face-index string like '0, 1' into [0, 1].

    Accepts an int, a string, or None. Empty/invalid input falls back to
    ``default``. Mirrors ReActor's comma-separated source/target face fields.
    """
    if value is None:
        return list(default)
    if isinstance(value, int):
        return [value]
    out: list[int] = []
    for chunk in str(value).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            continue
    return out or list(default)


class FaceSwapOptions(BaseModel):
    """ReActor-style face swap options (mirrors sd-webui-reactor controls)."""

    # Which detected faces to use. Lists support multi-face swaps; the legacy
    # single-index fields are kept for backward compatibility.
    source_face_index: int = Field(default=0, ge=0)
    target_face_index: int = Field(default=-1, ge=-1)  # -1 = swap every face
    source_faces_index: list[int] = Field(default_factory=lambda: [0])
    target_faces_index: list[int] = Field(default_factory=list)  # empty = all

    # Gender filtering: 0 = no filter, 1 = female only, 2 = male only.
    gender_source: int = Field(default=0, ge=0, le=2)
    gender_target: int = Field(default=0, ge=0, le=2)

    model_id: str = "inswapper_128"

    # Restoration (GFPGAN / CodeFormer) applied to the swapped face.
    restore_face: bool = True
    restorer_id: str | None = None
    restore_visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    codeformer_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    restore_first: bool = True  # restore before optional blend/upscale

    # Face Mask Correction — feather the swapped region to reduce edge
    # pixelation around the face contour (ReActor "Face Mask Correction").
    mask_face: bool = False


class FaceSwapModelInfo(BaseModel):
    id: str
    title: str
    path: str

    @classmethod
    def from_path(cls, path: Path) -> "FaceSwapModelInfo":
        return cls(id=path.stem, title=path.stem, path=str(path))
