"""
aiwf/core/domain/engine.py

Typed domain models for GPU engine tenant management.

One GPU-heavy tenant may own the card at a time. The EngineSupervisor uses
these types to track the active tenant and validate switch requests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class EngineTenant(str, Enum):
    """Named GPU workload tenants.

    IDLE means no tenant holds the GPU lock.  All other values represent an
    active workload category.  The supervisor enforces that at most one
    non-IDLE tenant runs at a time.
    """

    IDLE = "idle"
    IMAGE = "image"
    VIDEO = "video"
    CHAT = "chat"
    LORA_TRAINING = "lora_training"
    FULL_TRAINING = "full_training"
    ENHANCE = "enhance"

    def is_gpu_heavy(self) -> bool:
        """Return True for tenants that need exclusive GPU access."""
        return self not in (EngineTenant.IDLE, EngineTenant.CHAT)

    def friendly_name(self) -> str:
        return {
            "idle": "Idle",
            "image": "Image generation",
            "video": "Video generation",
            "chat": "Chat (Ollama)",
            "lora_training": "LoRA training",
            "full_training": "Full model training",
            "enhance": "Enhance / upscale",
        }.get(self.value, self.value)


@dataclass(frozen=True)
class EngineSwitchRequest:
    """Request to switch the active GPU tenant.

    Args:
        target:      The tenant you want to activate.
        reason:      Human-readable description (shown in logs / UI).
        allow_wait:  If True, the caller is willing to wait for the current
                     tenant to finish rather than getting an immediate refusal.
                     (Not implemented yet — reserved for future queue support.)
    """

    target: EngineTenant
    reason: str = ""
    allow_wait: bool = False


@dataclass(frozen=True)
class EngineSwitchResult:
    """Result of a tenant-switch attempt.

    Args:
        ok:       True if the switch succeeded (or was a no-op).
        active:   The tenant that is now active.
        message:  Human-readable status / error.
        log_path: Path to a log file for the prior tenant, if applicable.
    """

    ok: bool
    active: EngineTenant
    message: str
    log_path: Optional[Path] = None


@dataclass
class EngineStatus:
    """Live snapshot of engine supervisor state.  Not frozen — updated in place."""

    active: EngineTenant = EngineTenant.IDLE
    activated_at: Optional[datetime] = None
    last_switch_message: str = ""
    switch_count: int = 0

    def record_switch(self, tenant: EngineTenant, message: str) -> None:
        self.active = tenant
        self.activated_at = datetime.utcnow()
        self.last_switch_message = message
        self.switch_count += 1

    def to_dict(self) -> dict:
        return {
            "active": self.active.value,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "last_switch_message": self.last_switch_message,
            "switch_count": self.switch_count,
        }
