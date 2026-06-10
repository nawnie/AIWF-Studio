from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InfotextBridge:
    """Cross-tab clipboard for PNG info → generation tab paste."""

    pending_text: str | None = None
    last_params: dict[str, Any] = field(default_factory=dict)

    def set_pending(self, text: str, params: dict[str, Any] | None = None) -> None:
        self.pending_text = text
        if params:
            self.last_params = params

    def consume_pending(self) -> str | None:
        text = self.pending_text
        self.pending_text = None
        return text