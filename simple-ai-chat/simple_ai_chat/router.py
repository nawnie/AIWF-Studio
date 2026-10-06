"""Capability router: turns a user request into model steps.

The router is deliberately rule-based and cheap.  It handles what can be
decided *before* any model runs (attachments, slash commands, toggles).
Everything else goes to the brain, which decides further work through tool
calls; ``model_for_tool`` maps those tool names back to registry models so
the scheduler can load them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .registry import ModelSpec, Registry


class Capability(str, Enum):
    CHAT = "chat"
    REASONING = "reasoning"
    CODE = "code"
    VISION = "vision"
    TOOLS = "tools"
    ROUTE = "route"
    IMAGE_GEN = "image_gen"
    IMAGE_EDIT = "image_edit"
    PROMPT_ENHANCE = "prompt_enhance"
    PROMPT_ENHANCE_EDIT = "prompt_enhance_edit"
    ASR = "asr"
    TTS = "tts"
    AUDIO_UNDERSTANDING = "audio_understanding"
    EMBED = "embed"
    RERANK = "rerank"
    SIMULATE = "simulate"
    WEB_SIMULATE = "web_simulate"
    MODERATE = "moderate"


SLASH_COMMANDS: dict[str, Capability] = {
    "/image": Capability.IMAGE_GEN,
    "/edit": Capability.IMAGE_EDIT,
    "/speak": Capability.TTS,
    "/sim": Capability.SIMULATE,
    "/websim": Capability.WEB_SIMULATE,
    "/listen": Capability.AUDIO_UNDERSTANDING,
}

# Tool names the brain can call -> capability that serves them.
TOOL_CAPABILITIES: dict[str, Capability] = {
    "generate_image": Capability.IMAGE_GEN,
    "edit_image": Capability.IMAGE_EDIT,
    "speak": Capability.TTS,
    "transcribe": Capability.ASR,
    "simulate_environment": Capability.SIMULATE,
    "simulate_web_page": Capability.WEB_SIMULATE,
    "analyze_audio": Capability.AUDIO_UNDERSTANDING,
    "memory_search": Capability.EMBED,
    "rerank": Capability.RERANK,
    "moderate": Capability.MODERATE,
}

PROFILE_BY_MODE = {"code": "code", "game": "game", "research": "research", "fast": "fast"}


@dataclass(frozen=True)
class Attachment:
    kind: str  # image | audio | video | file
    path: str = ""


@dataclass
class ChatRequest:
    text: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    mode: str | None = None


@dataclass(frozen=True)
class Step:
    capability: Capability
    model_id: str
    reason: str
    profile: str | None = None


class RoutingError(LookupError):
    """No enabled model provides a required capability."""


class Router:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def model_for(self, capability: Capability) -> ModelSpec:
        spec = self.registry.best_for(capability.value)
        if spec is None:
            raise RoutingError(f"no enabled model provides {capability.value!r}")
        return spec

    def model_for_tool(self, tool_name: str) -> ModelSpec:
        try:
            capability = TOOL_CAPABILITIES[tool_name]
        except KeyError as exc:
            raise RoutingError(f"unknown tool {tool_name!r}") from exc
        return self.model_for(capability)

    def _step(self, capability: Capability, reason: str, profile: str | None = None) -> Step:
        return Step(capability, self.model_for(capability).id, reason, profile)

    def _optional_step(self, capability: Capability, reason: str) -> Step | None:
        spec = self.registry.best_for(capability.value)
        return Step(capability, spec.id, reason) if spec else None

    def plan(self, request: ChatRequest) -> list[Step]:
        steps: list[Step] = []
        kinds = {a.kind for a in request.attachments}
        text = request.text.strip()
        command, _, _rest = text.partition(" ")
        command = command.lower()

        if self.registry.toggle("moderation") and text:
            step = self._optional_step(Capability.MODERATE, "moderation toggle is on")
            if step:
                steps.append(step)

        if "audio" in kinds and command != "/listen":
            steps.append(self._step(Capability.ASR, "audio attachment: transcribe first"))

        capability = SLASH_COMMANDS.get(command)
        if capability is Capability.IMAGE_EDIT and "image" not in kinds:
            capability = None  # nothing to edit; let the brain ask for an image

        if capability in (Capability.IMAGE_GEN, Capability.IMAGE_EDIT):
            if self.registry.toggle("prompt_enhancer", True):
                enhance = (
                    Capability.PROMPT_ENHANCE
                    if capability is Capability.IMAGE_GEN
                    else Capability.PROMPT_ENHANCE_EDIT
                )
                step = self._optional_step(enhance, "prompt_enhancer toggle is on")
                if step:
                    steps.append(step)
            steps.append(self._step(capability, f"{command} command"))
            return steps

        if capability is not None:
            steps.append(self._step(capability, f"{command} command"))
            if capability is Capability.TTS:
                return steps
        else:
            profile = PROFILE_BY_MODE.get(request.mode or "", "chat")
            if kinds & {"image", "video"}:
                steps.append(self._step(Capability.VISION, "image/video attachment", profile))
            else:
                steps.append(self._step(Capability.CHAT, "default: brain", profile))

        if self.registry.toggle("speak_replies"):
            step = self._optional_step(Capability.TTS, "speak_replies toggle is on")
            if step:
                steps.append(step)
        return steps
