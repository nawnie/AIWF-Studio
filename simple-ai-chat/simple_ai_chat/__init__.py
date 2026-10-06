"""Simple AI Chat: a local, VRAM-aware multi-model Qwen harness."""
from __future__ import annotations

from .jobs import Job, JobQueue
from .manager import LoadError, ModelManager
from .registry import Hardware, ModelSpec, Registry, RegistryError, Residency
from .router import Attachment, Capability, ChatRequest, Router, RoutingError, Step
from .vram import Action, Plan, Resident, VramPlanner

__all__ = [
    "Action",
    "Attachment",
    "Capability",
    "ChatRequest",
    "Hardware",
    "Job",
    "JobQueue",
    "LoadError",
    "ModelManager",
    "ModelSpec",
    "Plan",
    "Registry",
    "RegistryError",
    "Resident",
    "Residency",
    "Router",
    "RoutingError",
    "Step",
    "VramPlanner",
]

__version__ = "0.0.1"
