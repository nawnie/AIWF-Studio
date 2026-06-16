"""
aiwf/services/prompt_tools.py

Agentic Prompt / Tool Workspace — Phase 7 service layer.

All tools in this module are designed to be called by an embedded assistant
(Agent MoK or compatible) operating within AIWF Studio.  They enforce the
rules in docs/LOCAL_TOOL_SECURITY.md at the call boundary:

* No shell command execution.
* All path resolution through AppContext (root paths passed at construction).
* Path traversal prevention on every user-supplied filename.
* No weight loading — safetensors reads are header-only.
* GPU tenant lock checked before generation submission.
* Every call is appended to an audit log at output_dir/.agent_tool_log.jsonl.

Design notes
------------
* No torch / diffusers / gradio imports at module level.
* All heavy dependencies (safetensors package) are imported lazily inside
  functions so that this module can be imported in environments where those
  packages are absent.
* The service is stateless across calls — it reads live disk state each time
  so that newly added checkpoints or LoRAs appear without a restart.
* ``AppContext`` is the only required collaborator; all other deps are optional
  and degrade gracefully.
"""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

_CHECKPOINT_EXTS: frozenset[str] = frozenset({".safetensors", ".ckpt"})
_LORA_EXTS: frozenset[str] = frozenset({".safetensors", ".pt"})
_PROMPT_LIB_EXTS: frozenset[str] = frozenset({".json", ".yaml", ".yml", ".txt"})

_SAFETENSORS_HEADER_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB


@dataclass(frozen=True)
class LocalFileInfo:
    """Lightweight descriptor for a checkpoint or LoRA file."""

    name: str
    filename: str
    path: Path
    size_bytes: int
    modified_at: datetime
    extension: str


@dataclass(frozen=True)
class SafetensorsHeader:
    """Parsed safetensors header metadata."""

    path: Path
    metadata: dict[str, str]
    tensor_keys: list[str]


@dataclass(frozen=True)
class RecommendedSettings:
    """Output of ``recommend_settings``."""

    width: int
    height: int
    steps: int
    cfg_scale: float
    sampler: str
    schedule: str
    rationale: str


@dataclass
class PromptDraft:
    """Output of ``build_prompt_draft``."""

    positive: str
    negative: str
    lora_tags: list[str]
    assembled: str  # positive + lora_tags joined


# ---------------------------------------------------------------------------
# Settings recommendation table
# ---------------------------------------------------------------------------

# Keyed on (architecture, goal). Architecture values match Checkpoint.architecture.
# goal: "speed" | "quality" | "balanced"
_RECOMMEND_TABLE: dict[tuple[str, str], RecommendedSettings] = {
    ("sd15", "speed"): RecommendedSettings(
        width=512, height=512, steps=20, cfg_scale=7.0,
        sampler="euler_a", schedule="automatic",
        rationale="SD 1.5 speed preset: native 512×512, 20 steps, Euler a.",
    ),
    ("sd15", "balanced"): RecommendedSettings(
        width=512, height=768, steps=28, cfg_scale=7.5,
        sampler="dpmpp_2m", schedule="karras",
        rationale="SD 1.5 balanced: 512×768 portrait, DPM++ 2M Karras, 28 steps.",
    ),
    ("sd15", "quality"): RecommendedSettings(
        width=768, height=768, steps=40, cfg_scale=8.0,
        sampler="dpmpp_2m", schedule="karras",
        rationale="SD 1.5 quality: 768×768, DPM++ 2M Karras, 40 steps, higher CFG.",
    ),
    ("sdxl", "speed"): RecommendedSettings(
        width=1024, height=1024, steps=20, cfg_scale=7.0,
        sampler="euler", schedule="automatic",
        rationale="SDXL speed preset: native 1024×1024, 20 steps, Euler.",
    ),
    ("sdxl", "balanced"): RecommendedSettings(
        width=1024, height=1024, steps=30, cfg_scale=7.5,
        sampler="dpmpp_2m", schedule="karras",
        rationale="SDXL balanced: 1024×1024, DPM++ 2M Karras, 30 steps.",
    ),
    ("sdxl", "quality"): RecommendedSettings(
        width=1216, height=832, steps=40, cfg_scale=8.0,
        sampler="dpmpp_2m", schedule="karras",
        rationale="SDXL quality: landscape 1216×832, DPM++ 2M Karras, 40 steps.",
    ),
    ("wan", "speed"): RecommendedSettings(
        width=832, height=480, steps=20, cfg_scale=6.0,
        sampler="euler", schedule="automatic",
        rationale="Wan speed: 832×480, 20 steps, Euler, low CFG.",
    ),
    ("wan", "balanced"): RecommendedSettings(
        width=832, height=480, steps=30, cfg_scale=6.0,
        sampler="unipc", schedule="automatic",
        rationale="Wan balanced: 832×480, UniPC, 30 steps.",
    ),
    ("wan", "quality"): RecommendedSettings(
        width=1280, height=720, steps=40, cfg_scale=7.0,
        sampler="unipc", schedule="automatic",
        rationale="Wan quality: 1280×720, UniPC, 40 steps.",
    ),
}

_DEFAULT_RECOMMEND = RecommendedSettings(
    width=512, height=512, steps=28, cfg_scale=7.5,
    sampler="dpmpp_2m", schedule="karras",
    rationale="Unknown architecture — generic SD 1.5 defaults.",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_resolve(root: Path, user_input: str | Path) -> Path:
    """Resolve *user_input* relative to *root* and verify it stays inside."""
    resolved = (root / user_input).resolve()
    root_resolved = root.resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise PermissionError(
            f"Path '{user_input}' resolves outside allowed root '{root_resolved}': {resolved}"
        )
    return resolved


def _file_info(path: Path) -> LocalFileInfo:
    stat = path.stat()
    return LocalFileInfo(
        name=path.stem,
        filename=path.name,
        path=path,
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        extension=path.suffix.lower(),
    )


def _read_safetensors_header_raw(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse safetensors header without loading weights.

    Returns (metadata_dict, tensor_key_list).
    Raises ValueError if the file is invalid or the header is too large.
    """
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError("File too short to be a valid safetensors file.")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > _SAFETENSORS_HEADER_MAX_BYTES:
            raise ValueError(
                f"Safetensors header length {header_len} exceeds safety cap "
                f"{_SAFETENSORS_HEADER_MAX_BYTES}."
            )
        raw_header = f.read(header_len)
    header: dict[str, Any] = json.loads(raw_header)
    metadata: dict[str, str] = {}
    tensor_keys: list[str] = []
    for k, v in header.items():
        if k == "__metadata__":
            metadata = {str(mk): str(mv) for mk, mv in v.items()}
        else:
            tensor_keys.append(k)
    return metadata, tensor_keys


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PromptToolsService:
    """
    Agentic tool service for AIWF Studio (Phase 7).

    Parameters
    ----------
    checkpoint_dir:
        Root directory that contains local checkpoint files.
    lora_dir:
        Root directory that contains local LoRA files.
    output_dir:
        Configured output directory (only write target permitted).
    prompt_library_dir:
        Directory containing prompt style / template files.  Optional.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        lora_dir: Path,
        output_dir: Path,
        prompt_library_dir: Path | None = None,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._lora_dir = lora_dir
        self._output_dir = output_dir
        self._prompt_library_dir = prompt_library_dir
        self._audit_log = output_dir / ".agent_tool_log.jsonl"

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def _audit(
        self,
        tool: str,
        args: dict[str, Any],
        result_summary: str,
        gpu_tenant: str = "unknown",
    ) -> None:
        entry = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "tool": tool,
            "args": args,
            "result_summary": result_summary,
            "gpu_tenant_at_call": gpu_tenant,
        }
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            with self._audit_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("Failed to write agent audit log: %s", exc)

    # ------------------------------------------------------------------
    # Tier 1 — Pure Read
    # ------------------------------------------------------------------

    def list_local_checkpoints(self) -> list[LocalFileInfo]:
        """Scan ``checkpoint_dir`` and return info for each checkpoint file.

        Only ``.safetensors`` and ``.ckpt`` files are returned (no traversal
        outside the configured root).
        """
        results: list[LocalFileInfo] = []
        if not self._checkpoint_dir.is_dir():
            self._audit(
                "list_local_checkpoints", {},
                f"checkpoint_dir not found: {self._checkpoint_dir}",
            )
            return results
        for path in sorted(self._checkpoint_dir.rglob("*")):
            if path.suffix.lower() in _CHECKPOINT_EXTS and path.is_file():
                try:
                    results.append(_file_info(path))
                except OSError as exc:
                    logger.debug("Skipping %s: %s", path, exc)
        self._audit(
            "list_local_checkpoints", {},
            f"ok: {len(results)} files",
        )
        return results

    def list_local_loras(self) -> list[LocalFileInfo]:
        """Scan ``lora_dir`` and return info for each LoRA file."""
        results: list[LocalFileInfo] = []
        if not self._lora_dir.is_dir():
            self._audit(
                "list_local_loras", {},
                f"lora_dir not found: {self._lora_dir}",
            )
            return results
        for path in sorted(self._lora_dir.rglob("*")):
            if path.suffix.lower() in _LORA_EXTS and path.is_file():
                try:
                    results.append(_file_info(path))
                except OSError as exc:
                    logger.debug("Skipping %s: %s", path, exc)
        self._audit(
            "list_local_loras", {},
            f"ok: {len(results)} files",
        )
        return results

    def read_safetensors_metadata(self, filename: str) -> SafetensorsHeader:
        """Read the JSON metadata header from a safetensors file.

        *filename* is resolved relative to ``checkpoint_dir`` first, then
        ``lora_dir``.  Path traversal outside either root is rejected.

        Weights are never loaded — only the header bytes are read.
        """
        # Try checkpoint dir, then lora dir
        for root in (self._checkpoint_dir, self._lora_dir):
            try:
                resolved = _safe_resolve(root, filename)
            except PermissionError as exc:
                self._audit(
                    "read_safetensors_metadata",
                    {"filename": filename},
                    f"error: {exc}",
                )
                raise
            if resolved.is_file() and resolved.suffix.lower() == ".safetensors":
                try:
                    metadata, tensor_keys = _read_safetensors_header_raw(resolved)
                except Exception as exc:
                    self._audit(
                        "read_safetensors_metadata",
                        {"filename": filename},
                        f"error reading header: {exc}",
                    )
                    raise
                self._audit(
                    "read_safetensors_metadata",
                    {"filename": filename},
                    f"ok: {len(metadata)} metadata keys, {len(tensor_keys)} tensors",
                )
                return SafetensorsHeader(
                    path=resolved,
                    metadata=metadata,
                    tensor_keys=tensor_keys,
                )
        msg = (
            f"'{filename}' not found as a .safetensors file in checkpoint_dir "
            f"or lora_dir."
        )
        self._audit("read_safetensors_metadata", {"filename": filename}, f"error: {msg}")
        raise FileNotFoundError(msg)

    def inspect_prompt_library(self) -> list[dict[str, str]]:
        """List prompt style / template files in ``prompt_library_dir``.

        Returns a list of dicts with keys: ``name``, ``filename``, ``path``.
        Returns an empty list if no prompt library directory is configured.
        """
        if not self._prompt_library_dir or not self._prompt_library_dir.is_dir():
            self._audit("inspect_prompt_library", {}, "ok: no prompt library configured")
            return []
        results: list[dict[str, str]] = []
        for path in sorted(self._prompt_library_dir.rglob("*")):
            if path.suffix.lower() in _PROMPT_LIB_EXTS and path.is_file():
                results.append({
                    "name": path.stem,
                    "filename": path.name,
                    "path": str(path),
                })
        self._audit("inspect_prompt_library", {}, f"ok: {len(results)} templates")
        return results

    # ------------------------------------------------------------------
    # Tier 2 — Compute / Compose (CPU-only)
    # ------------------------------------------------------------------

    def build_prompt_draft(
        self,
        subject: str,
        style_template: str = "",
        lora_names: list[str] | None = None,
        lora_weights: list[float] | None = None,
        negative: str = "",
    ) -> PromptDraft:
        """Assemble a generation prompt from components.

        Parameters
        ----------
        subject:
            The core subject text, e.g. ``"a photorealistic portrait of a woman"``.
        style_template:
            Optional style string containing ``{prompt}`` placeholder.  If the
            placeholder is absent, the style is appended after the subject.
        lora_names:
            Short LoRA filenames (stem only, e.g. ``"my_lora"``).  Each is
            rendered as ``<lora:name:weight>``.
        lora_weights:
            Per-LoRA weights in [0.0, 2.0].  Defaults to 1.0 for each.
        negative:
            Negative prompt text.
        """
        lora_names = lora_names or []
        lora_weights_resolved = list(lora_weights or []) + [1.0] * max(
            0, len(lora_names) - len(lora_weights or [])
        )
        lora_tags = [
            f"<lora:{name.rstrip('.safetensors').rstrip('.pt')}:{weight:.2f}>"
            for name, weight in zip(lora_names, lora_weights_resolved)
        ]

        if style_template and "{prompt}" in style_template:
            positive = style_template.replace("{prompt}", subject).strip()
        elif style_template:
            positive = f"{subject}, {style_template}".strip(", ")
        else:
            positive = subject.strip()

        lora_suffix = " ".join(lora_tags)
        assembled = (f"{positive} {lora_suffix}".strip()) if lora_suffix else positive

        draft = PromptDraft(
            positive=positive,
            negative=negative.strip(),
            lora_tags=lora_tags,
            assembled=assembled,
        )
        self._audit(
            "build_prompt_draft",
            {"subject": subject[:80], "loras": lora_names, "style": bool(style_template)},
            f"ok: assembled {len(assembled)} chars",
        )
        return draft

    def recommend_settings(
        self,
        architecture: str = "sd15",
        goal: str = "balanced",
    ) -> RecommendedSettings:
        """Return recommended generation settings for a checkpoint architecture.

        Parameters
        ----------
        architecture:
            Model architecture string — ``"sd15"``, ``"sdxl"``, or ``"wan"``.
        goal:
            Generation goal — ``"speed"``, ``"balanced"``, or ``"quality"``.
        """
        key = (architecture.lower(), goal.lower())
        result = _RECOMMEND_TABLE.get(key, _DEFAULT_RECOMMEND)
        self._audit(
            "recommend_settings",
            {"architecture": architecture, "goal": goal},
            f"ok: {result.sampler} {result.width}x{result.height} {result.steps}steps",
        )
        return result

    def generate_workflow_json(
        self,
        checkpoint_filename: str,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 28,
        cfg_scale: float = 7.5,
        sampler: str = "dpmpp_2m",
        seed: int = -1,
    ) -> dict[str, Any]:
        """Produce a minimal ComfyUI-compatible workflow JSON structure.

        This is a pure data composition — no GPU, no network, no file I/O.
        The returned dict can be serialised to JSON and sent to a ComfyUI API.

        The workflow encodes a simple txt2img flow:
        CheckpointLoaderSimple → CLIPTextEncode (×2) → KSampler → VAEDecode
        → SaveImage.
        """
        workflow: dict[str, Any] = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint_filename},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1],
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1],
                },
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": sampler,
                    "scheduler": "karras",
                    "denoise": 1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2],
                },
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "aiwf_agent_",
                },
            },
        }
        self._audit(
            "generate_workflow_json",
            {
                "checkpoint": checkpoint_filename,
                "width": width,
                "height": height,
                "steps": steps,
            },
            "ok: workflow composed",
        )
        return workflow
