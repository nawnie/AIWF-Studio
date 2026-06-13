from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image as PILImage


@dataclass
class InpaintSession:
    """Authoritative inpaint background + mask across editor visibility toggles."""

    original: PILImage.Image | None = None
    mask: PILImage.Image | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"original": self.original, "mask": self.mask}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InpaintSession:
        return cls(original=data.get("original"), mask=data.get("mask"))


@dataclass
class StudioSession:
    """Mutable Studio tab state — replaces ad-hoc closure dicts in the monolith."""

    loop_active: bool = True
    sam_mask: PILImage.Image | None = None
    inpaint: InpaintSession = field(default_factory=InpaintSession)

    @property
    def loop_ctrl(self) -> dict[str, bool]:
        return {"active": self.loop_active}

    @loop_ctrl.setter
    def loop_ctrl(self, value: dict[str, bool]) -> None:
        self.loop_active = bool(value.get("active", False))

    @property
    def sam_state(self) -> dict[str, PILImage.Image | None]:
        return {"mask": self.sam_mask}

    @property
    def inpaint_session(self) -> dict[str, Any]:
        return self.inpaint.as_dict()