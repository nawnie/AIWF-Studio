from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InfotextBridge:
    """Cross-tab clipboard for PNG info → generation tab paste."""

    pending_text: str | None = None
    pending_image: Any = None
    last_params: dict[str, Any] = field(default_factory=dict)

    def set_pending(self, text: str, params: dict[str, Any] | None = None) -> None:
        self.pending_text = text
        if params:
            self.last_params = params

    def consume_pending(self) -> str | None:
        text = self.pending_text
        self.pending_text = None
        return text

    def set_image(self, image: Any) -> None:
        self.pending_image = image

    def consume_image(self) -> Any:
        image = self.pending_image
        self.pending_image = None
        return image