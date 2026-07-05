from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import paid_worker
from aiwf.services.studio_generation_packet import validate_workflow_code_block_document

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_MEDIA_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
}


class PaidExtensionTab(BaseModel):
    id: str
    label: str
    icon: str = "plugin"
    color: str = "#8b5cf6"
    hidden: bool = False
    workspaceType: str = "empty"
    description: str = ""


class PaidExtensionTabsPayload(BaseModel):
    tabs: list[PaidExtensionTab] = Field(default_factory=list)


class PaidAgentMessage(BaseModel):
    role: str = "user"
    content: str = ""


class PaidAgentChatPayload(BaseModel):
    model: str = ""
    messages: list[PaidAgentMessage] = Field(default_factory=list)
    enabledTools: list[str] = Field(default_factory=list)
    temperature: float = 0.2


class PaidAgentPermissionsPayload(BaseModel):
    observe: bool = True
    suggest: bool = True
    draft: bool = True
    executeWithApproval: bool = False
    trustedLocal: bool = False
    allowedTools: list[str] = Field(default_factory=list)


class PaidProjectPayload(BaseModel):
    id: str = ""
    name: str = "Untitled AIWF Project"
    data: dict[str, Any] = Field(default_factory=dict)


class PaidWorkflowValidatePayload(BaseModel):
    workflow: dict[str, Any] = Field(default_factory=dict)


class PaidQueueJobPayload(BaseModel):
    kind: str = "workflow"
    label: str = "Untitled render"
    payload: dict[str, Any] = Field(default_factory=dict)


class PaidPluginManifestPayload(BaseModel):
    manifest: dict[str, Any] = Field(default_factory=dict)


class PaidExportPlanPayload(BaseModel):
    preset: str = "web-image"
    projectId: str = "default"
    assetIds: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class PaidQaAnalyzePayload(BaseModel):
    projectId: str = "default"
    prompt: str = ""
    assetPath: str = ""
    workflow: dict[str, Any] = Field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str, fallback: str = "item") -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value or "")).strip("-").lower()
    return safe or f"{fallback}-{uuid4().hex[:8]}"


def _data_root(ctx: Any) -> Path:
    flags = getattr(ctx, "flags", None)
    root = getattr(flags, "data_dir", None) or Path.cwd()
    return Path(root).resolve()


def _outputs_root(ctx: Any) -> Path:
    flags = getattr(ctx, "flags", None)
    resolver = getattr(flags, "resolved_output_dir", None)
    if callable(resolver):
        return Path(resolver()).resolve()
    return (_data_root(ctx) / "outputs").resolve()


def _paid_runtime_dir(ctx: Any) -> Path:
    root = _data_root(ctx) / "extensions" / "paid-media-center-v4"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.debug("Could not read paid media center JSON: %s", path, exc_info=True)
    return default


def _write_json(path: Path, payload: Any) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _tabs_path(ctx: Any) -> Path:
    return _paid_runtime_dir(ctx) / "tabs.json"


def _workspace_path(ctx: Any, workspace_id: str) -> Path:
    return _paid_runtime_dir(ctx) / "workspaces" / f"{_safe_id(workspace_id, 'workspace')}.json"


def _projects_root(ctx: Any) -> Path:
    path = _paid_runtime_dir(ctx) / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_dir(ctx: Any, project_id: str) -> Path:
    return _projects_root(ctx) / _safe_id(project_id, "project")


def _project_path(ctx: Any, project_id: str) -> Path:
    return _project_dir(ctx, project_id) / "project.json"


def _workflow_path(ctx: Any, workflow_id: str) -> Path:
    return _paid_runtime_dir(ctx) / "workflows" / f"{_safe_id(workflow_id, 'workflow')}.json"


def _queue_path(ctx: Any) -> Path:
    return _paid_runtime_dir(ctx) / "render_queue.json"


def _plugins_path(ctx: Any) -> Path:
    return _paid_runtime_dir(ctx) / "plugin_registry.json"


def _permissions_path(ctx: Any) -> Path:
    return _paid_runtime_dir(ctx) / "agent_permissions.json"


def _load_tabs(ctx: Any) -> list[dict[str, Any]]:
    payload = _read_json(_tabs_path(ctx), {"tabs": []})
    tabs = payload.get("tabs") if isinstance(payload, dict) else payload
    if not isinstance(tabs, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in tabs:
        if not isinstance(item, dict):
            continue
        try:
            normalized.append(PaidExtensionTab.model_validate(item).model_dump(mode="json"))
        except Exception:
            logger.debug("Skipping invalid extension tab payload", exc_info=True)
    return normalized


def _save_tabs(ctx: Any, tabs: list[PaidExtensionTab]) -> dict[str, Any]:
    payload = {
        "schema": "aiwf.paid.extensions.tabs.v4",
        "updatedAt": _now(),
        "tabs": [tab.model_dump(mode="json") for tab in tabs],
    }
    return _write_json(_tabs_path(ctx), payload)


def _default_project(project_id: str = "default", name: str = "AIWF Media Project") -> dict[str, Any]:
    return {
        "schema": "aiwf.media-project.v1",
        "id": project_id,
        "name": name,
        "createdAt": _now(),
        "updatedAt": _now(),
        "activeWorkflowId": "main",
        "scenes": [
            {"id": "scene-01", "title": "Opening image", "status": "draft", "assetIds": [], "notes": "First scene card."},
            {"id": "scene-02", "title": "Motion pass", "status": "planned", "assetIds": [], "notes": "Image to video or extend."},
        ],
        "tracks": [
            {"id": "video", "label": "Video", "type": "video", "items": []},
            {"id": "image", "label": "Image", "type": "image", "items": []},
            {"id": "audio", "label": "Audio", "type": "audio", "items": []},
            {"id": "metadata", "label": "Receipts", "type": "metadata", "items": []},
        ],
        "versions": [],
        "promptStudio": {
            "subject": "",
            "style": "",
            "camera": "",
            "lighting": "",
            "world": "",
            "negative": "",
            "assembledPrompt": "",
        },
        "characters": [],
        "ui": {"zoom": 1, "activeDock": "tracks", "hiddenTabs": []},
    }


def _load_project(ctx: Any, project_id: str) -> dict[str, Any]:
    project = _read_json(_project_path(ctx, project_id), None)
    if isinstance(project, dict):
        return project
    default = _default_project(_safe_id(project_id, "project"), "AIWF Media Project")
    return _write_json(_project_path(ctx, project_id), default)


def _save_project(ctx: Any, project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = _load_project(ctx, project_id)
    merged = {**current, **payload, "id": _safe_id(payload.get("id") or project_id, "project"), "updatedAt": _now()}
    return _write_json(_project_path(ctx, merged["id"]), merged)


def _node_registry() -> list[dict[str, Any]]:
    return [
        {"id": "generation-request", "label": "Generation Request Block", "group": "Workflow", "requires": [], "produces": ["artifact"], "compatibleEngines": ["all"], "executionRole": "code-block"},
        {"id": "template-reference", "label": "Template Reference Block", "group": "Workflow", "requires": [], "produces": ["artifact"], "compatibleEngines": ["all"], "executionRole": "code-block"},
        {"id": "imported-json", "label": "Imported JSON Block", "group": "Workflow", "requires": [], "produces": ["artifact"], "compatibleEngines": ["all"], "executionRole": "code-block"},
        {"id": "prompt", "label": "Prompt", "group": "Input", "requires": [], "produces": ["prompt"], "compatibleEngines": ["all"]},
        {"id": "model", "label": "Model Generate", "group": "Image", "requires": ["prompt"], "produces": ["image"], "compatibleEngines": ["sd15", "sdxl", "sd35", "flux", "sana", "qwen", "zimage"]},
        {"id": "auto-mask", "label": "Auto Mask", "group": "Mask", "requires": ["image"], "produces": ["mask"], "compatibleEngines": ["segment", "sam", "dino"]},
        {"id": "inpaint", "label": "Inpaint", "group": "Image", "requires": ["image", "mask"], "produces": ["image"], "compatibleEngines": ["sd15", "sdxl", "flux_fill"]},
        {"id": "upscale", "label": "Upscale", "group": "Enhance", "requires": ["image"], "produces": ["image"], "compatibleEngines": ["enhance", "vsr"]},
        {"id": "image-to-video", "label": "Image to Video", "group": "Video", "requires": ["image", "prompt"], "produces": ["video"], "compatibleEngines": ["wan", "ltx", "sana_video"]},
        {"id": "wan-model-pack", "label": "Wan Model Pack", "group": "Wan Resource", "requires": [], "produces": ["wan-pack"], "compatibleEngines": ["wan"], "executionRole": "sidecar"},
        {"id": "wan-lora-stack", "label": "Wan LoRA Stack", "group": "Wan Resource", "requires": [], "produces": ["lora-stack"], "compatibleEngines": ["wan"], "executionRole": "sidecar"},
        {"id": "wan-offload-plan", "label": "Wan Offload Plan", "group": "Runtime", "requires": [], "produces": ["offload-plan"], "compatibleEngines": ["wan"], "executionRole": "sidecar"},
        {"id": "wan-video", "label": "Wan Video", "group": "Video", "requires": ["image", "prompt", "wan-pack", "lora-stack", "offload-plan"], "produces": ["video"], "compatibleEngines": ["wan"]},
        {"id": "rife", "label": "RIFE Interpolate", "group": "Video", "requires": ["video"], "produces": ["video"], "compatibleEngines": ["rife"]},
        {"id": "vsr-video", "label": "VSR Video", "group": "Video", "requires": ["video"], "produces": ["video"], "compatibleEngines": ["vsr"]},
        {"id": "audio-score", "label": "Audio Score", "group": "Audio", "requires": ["video"], "produces": ["audio"], "compatibleEngines": ["mmaudio", "musicgen"]},
        {"id": "mux", "label": "Mux Audio", "group": "Export", "requires": ["video", "audio"], "produces": ["video", "artifact"], "compatibleEngines": ["ffmpeg"]},
        {"id": "receipt", "label": "Receipt", "group": "Data", "requires": ["image"], "produces": ["metadata"], "compatibleEngines": ["aiwf"]},
        {"id": "output", "label": "Output", "group": "Export", "requires": ["image"], "produces": ["artifact"], "compatibleEngines": ["aiwf"]},
        {"id": "video-output", "label": "Video Output", "group": "Export", "requires": ["artifact"], "produces": ["artifact"], "compatibleEngines": ["aiwf"]},
    ]


def _workflow_templates() -> list[dict[str, Any]]:
    return [
        {
            "id": "image-polish",
            "label": "Image polish",
            "summary": "Generate, upscale, receipt, export.",
            "stages": ["prompt", "model", "upscale", "receipt", "output"],
        },
        {
            "id": "inpaint-repair",
            "label": "Inpaint repair",
            "summary": "Generate, auto-mask, inpaint, export.",
            "stages": ["prompt", "model", "auto-mask", "inpaint", "receipt", "output"],
        },
        {
            "id": "wan-community-video",
            "label": "Wan community video",
            "summary": "Image, Wan high/low pack, LoRA sidecar, offload sidecar, RIFE, VSR, audio, mux.",
            "stages": ["prompt", "model", "wan-model-pack", "wan-lora-stack", "wan-offload-plan", "wan-video", "rife", "vsr-video", "audio-score", "mux", "video-output"],
        },
        {
            "id": "video-clip",
            "label": "Generic video clip",
            "summary": "Image, video, RIFE, VSR, audio, mux.",
            "stages": ["prompt", "model", "image-to-video", "rife", "vsr-video", "audio-score", "mux"],
        },
        {
            "id": "music-bed",
            "label": "Music bed",
            "summary": "Prompt-guided audio for an existing clip.",
            "stages": ["prompt", "audio-score"],
        },
        {
            "id": "social-export",
            "label": "Social export",
            "summary": "Generate, crop-safe output, metadata.",
            "stages": ["prompt", "model", "receipt", "output"],
        },
    ]


def _registry_map() -> dict[str, dict[str, Any]]:
    return {node["id"]: node for node in _node_registry()}


def _validate_workflow_code_blocks(workflow: dict[str, Any], blocks_raw: Any) -> dict[str, Any]:
    if not isinstance(blocks_raw, list):
        return {"valid": False, "errors": ["workflow.blocks must be a list"], "availableClasses": [], "blocks": []}
    registry = _registry_map()
    shared = validate_workflow_code_block_document({**workflow, "blocks": blocks_raw})
    errors: list[str] = list(shared.get("errors", []))
    available: set[str] = set()
    block_results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, block in enumerate(blocks_raw):
        if not isinstance(block, dict):
            continue
        block_id = str(block.get("id") or f"block-{index + 1}")
        if block_id in seen_ids:
            # Shared validation already reports duplicate ids; do not double-noise the UI.
            continue
        seen_ids.add(block_id)
        label = str(block.get("label") or block_id)
        node_id = str(block.get("nodeId") or block.get("node_id") or block.get("templateId") or "generation-request")
        node = registry.get(node_id)
        if node is None and f"Block {index + 1} uses unknown nodeId '{node_id}'." not in errors:
            errors.append(f"Block {index + 1} uses unknown nodeId '{node_id}'.")
        code_ok = True
        payload = block.get("payload") if isinstance(block.get("payload"), dict) else {}
        code = block.get("code")
        if isinstance(code, str) and code.strip():
            try:
                parsed = json.loads(code)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                code_ok = False
        elif not isinstance(block.get("payload"), dict):
            code_ok = False
        classes = block.get("classes") if isinstance(block.get("classes"), dict) else {}
        produces = [str(item) for item in classes.get("produces", [])] if isinstance(classes, dict) and isinstance(classes.get("produces"), list) else [str(item) for item in (node or {}).get("produces", [])]
        requires = [str(item) for item in classes.get("requires", [])] if isinstance(classes, dict) and isinstance(classes.get("requires"), list) else [str(item) for item in (node or {}).get("requires", [])]
        available.update(produces)
        packet = payload.get("packet") if isinstance(payload, dict) else None
        gate = packet.get("selectionGate") if isinstance(packet, dict) and isinstance(packet.get("selectionGate"), dict) else {}
        block_results.append({
            "id": block_id,
            "label": label,
            "nodeId": node_id,
            "valid": node is not None and code_ok and not (isinstance(gate, dict) and gate.get("normalSelectable") is False),
            "requires": requires,
            "produces": produces,
            "order": block.get("order") or index + 1,
            "selectionGate": gate,
        })
    return {
        "valid": not errors,
        "errors": errors,
        "availableClasses": sorted(available),
        "blocks": block_results,
        "mode": "linear-code-blocks",
        "stages": [],
        "sidecars": [],
    }


def _validate_workflow_payload(workflow: dict[str, Any]) -> dict[str, Any]:
    if "blocks" in workflow:
        return _validate_workflow_code_blocks(workflow, workflow.get("blocks"))
    stages_raw = workflow.get("stages") or workflow.get("workflow") or []
    if not isinstance(stages_raw, list):
        return {"valid": False, "errors": ["workflow.stages must be a list"], "availableClasses": []}
    registry = _registry_map()
    available: set[str] = set()
    errors: list[str] = []
    sidecar_results: list[dict[str, Any]] = []
    sidecars_raw = workflow.get("sidecars") or workflow.get("resources") or []
    if sidecars_raw and not isinstance(sidecars_raw, list):
        errors.append("workflow.sidecars must be a list when provided")
        sidecars_raw = []
    for index, sidecar in enumerate(sidecars_raw):
        if not isinstance(sidecar, dict):
            continue
        node_id = str(sidecar.get("templateId") or sidecar.get("nodeId") or sidecar.get("id") or "")
        node = registry.get(node_id, {})
        raw_produces = sidecar.get("produces") if isinstance(sidecar.get("produces"), list) else node.get("produces", [])
        produces = [str(item) for item in raw_produces]
        available.update(produces)
        sidecar_results.append({
            "id": str(sidecar.get("uid") or sidecar.get("id") or f"sidecar-{index + 1}"),
            "nodeId": node_id,
            "label": sidecar.get("label") or node.get("label", node_id),
            "role": sidecar.get("role") or node.get("executionRole", "sidecar"),
            "produces": produces,
        })
    stage_results: list[dict[str, Any]] = []
    for index, stage in enumerate(stages_raw):
        if isinstance(stage, str):
            node_id = stage
            stage_id = f"stage-{index + 1}"
        elif isinstance(stage, dict):
            node_id = str(stage.get("templateId") or stage.get("nodeId") or stage.get("id") or "")
            stage_id = str(stage.get("uid") or stage.get("id") or f"stage-{index + 1}")
        else:
            node_id = ""
            stage_id = f"stage-{index + 1}"
        node = registry.get(node_id)
        if node is None:
            error = f"Stage {index + 1} uses unknown node '{node_id}'."
            errors.append(error)
            stage_results.append({"id": stage_id, "nodeId": node_id, "valid": False, "missing": [], "error": error})
            continue
        requires = [str(item) for item in node.get("requires", [])]
        produces = [str(item) for item in node.get("produces", [])]
        missing = [item for item in requires if item not in available]
        if missing:
            errors.append(f"{node['label']} is missing: {', '.join(missing)}")
        else:
            available.update(produces)
        stage_results.append({
            "id": stage_id,
            "nodeId": node_id,
            "label": node.get("label", node_id),
            "valid": not missing,
            "missing": missing,
            "requires": requires,
            "produces": produces,
            "availableAfter": sorted(available),
        })
    if not stages_raw:
        errors.append("Workflow has no stages.")
    return {"valid": not errors, "errors": errors, "availableClasses": sorted(available), "stages": stage_results, "sidecars": sidecar_results}


def _ollama_url(ctx: Any) -> str:
    settings = getattr(ctx, "settings", None)
    raw = getattr(settings, "ollama_url", "") if settings is not None else ""
    return str(raw or _DEFAULT_OLLAMA_URL).rstrip("/")


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"response": body}


def _get_json(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def _agent_tools() -> list[dict[str, str]]:
    return [
        {"id": "project-reader", "label": "Project Reader", "group": "Studio", "status": "read-only", "description": "Read project scenes, workflows, receipts, and assets."},
        {"id": "workflow-json", "label": "Workflow JSON", "group": "Studio", "status": "available", "description": "Create, inspect, validate, and explain Pipeline Atlas workflows."},
        {"id": "prompt-refiner", "label": "Prompt Refiner", "group": "Create", "status": "available", "description": "Improve prompt structure and negative prompt coverage."},
        {"id": "asset-curator", "label": "Asset Curator", "group": "Media", "status": "draft-only", "description": "Suggest tags, variants, and scene placement for assets."},
        {"id": "plugin-manager", "label": "Plugin Manager", "group": "Extensions", "status": "safe", "description": "Draft extension manifests and empty workspaces."},
        {"id": "log-viewer", "label": "Log Viewer", "group": "Runtime", "status": "read-only", "description": "Summarize logs and suggest troubleshooting steps."},
        {"id": "patch-draft", "label": "Patch Draft", "group": "Code", "status": "draft-only", "description": "Draft code changes for user review. Does not write files by itself."},
    ]


def _asset_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    for kind, suffixes in _MEDIA_EXTENSIONS.items():
        if suffix in suffixes:
            return kind
    return None


def _output_asset_url(ctx: Any, path: Path) -> str:
    root = _outputs_root(ctx)
    try:
        rel = path.resolve().relative_to(root)
        return f"/api/pro/outputs/{quote(rel.as_posix())}"
    except ValueError:
        return ""


def _scan_assets(ctx: Any, limit: int = 160) -> list[dict[str, Any]]:
    root = _outputs_root(ctx)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not path.is_file():
            continue
        kind = _asset_kind(path)
        if kind is None:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append({
            "id": uuid4().hex[:12],
            "kind": kind,
            "name": path.name,
            "path": str(path),
            "url": _output_asset_url(ctx, path),
            "sizeBytes": stat.st_size,
            "modifiedAt": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "tags": [kind, path.parent.name],
            "projectId": "default",
        })
        if len(rows) >= limit:
            break
    return rows


def _scan_receipts(ctx: Any, limit: int = 80) -> list[dict[str, Any]]:
    root = _outputs_root(ctx)
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    for suffix in ("*.json", "*.jsonl", "*.txt"):
        candidates.extend(root.rglob(suffix))
    rows: list[dict[str, Any]] = []
    for path in sorted(set(candidates), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append({
            "id": uuid4().hex[:12],
            "name": path.name,
            "path": str(path),
            "sizeBytes": stat.st_size,
            "modifiedAt": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "kind": "receipt" if path.suffix.lower() in {".json", ".jsonl"} else "metadata",
        })
        if len(rows) >= limit:
            break
    return rows


def _load_queue(ctx: Any) -> dict[str, Any]:
    payload = _read_json(_queue_path(ctx), {"schema": "aiwf.render-queue.v1", "updatedAt": _now(), "jobs": []})
    if not isinstance(payload, dict):
        return {"schema": "aiwf.render-queue.v1", "updatedAt": _now(), "jobs": []}
    payload.setdefault("jobs", [])
    return payload


def _save_queue(ctx: Any, queue: dict[str, Any]) -> dict[str, Any]:
    queue["updatedAt"] = _now()
    return _write_json(_queue_path(ctx), queue)


def _worker_receipts_path(ctx: Any) -> Path:
    return _paid_runtime_dir(ctx) / "worker_receipts.json"


def _worker_logs_dir(ctx: Any) -> Path:
    return _paid_runtime_dir(ctx) / "job_logs"


def _get_worker(ctx: Any) -> paid_worker.QueueWorker:
    return paid_worker.get_worker(ctx, _queue_path(ctx), _worker_receipts_path(ctx), _worker_logs_dir(ctx))


_CONTEXT_TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".css", ".html", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".ps1", ".bat", ".sh"}
_CONTEXT_MAX_BYTES = 256 * 1024
_CONTEXT_MAX_ENTRIES = 400


def _context_roots(ctx: Any) -> dict[str, Path]:
    """Read-only roots the agent may browse. Nothing outside these resolves."""
    roots: dict[str, Path] = {"data": _data_root(ctx), "outputs": _outputs_root(ctx)}
    repo = getattr(getattr(ctx, "flags", None), "repo_dir", None)
    if repo:
        roots["repo"] = Path(repo).resolve()
    return roots


def _resolve_context_path(ctx: Any, root_key: str, rel_path: str) -> Path:
    roots = _context_roots(ctx)
    root = roots.get(root_key)
    if root is None or not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown context root '{root_key}'.")
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path escapes the context root.") from exc
    return candidate


def _load_plugins(ctx: Any) -> dict[str, Any]:
    payload = _read_json(_plugins_path(ctx), {"schema": "aiwf.plugin-registry.v1", "updatedAt": _now(), "plugins": []})
    if not isinstance(payload, dict):
        return {"schema": "aiwf.plugin-registry.v1", "updatedAt": _now(), "plugins": []}
    payload.setdefault("plugins", [])
    return payload


def _export_presets() -> list[dict[str, Any]]:
    return [
        {"id": "web-image", "label": "Web image", "outputs": ["png", "webp", "jpg"], "summary": "Optimized still image export."},
        {"id": "print-image", "label": "Print image", "outputs": ["png", "tiff-later"], "summary": "High-quality still export path."},
        {"id": "youtube-clip", "label": "YouTube clip", "outputs": ["mp4-h264"], "summary": "16:9 H.264 with audio."},
        {"id": "reels-vertical", "label": "TikTok / Reels", "outputs": ["mp4-h264", "9:16"], "summary": "Vertical social export."},
        {"id": "audio-stem", "label": "Audio stem", "outputs": ["wav", "mp3"], "summary": "Audio stem or master export."},
        {"id": "project-archive", "label": "Project archive", "outputs": ["aiwf.zip"], "summary": "Portable AIWF project archive."},
    ]


def build_paid_extension_router(ctx: Any) -> APIRouter:
    router = APIRouter(prefix="/api/pro")

    @router.get("/extensions/manifest")
    def extension_manifest() -> dict[str, Any]:
        return {
            "schema": "aiwf.paid.extensions.manifest.v4",
            "name": "AIWF Paid Media Center Extension Host",
            "version": "4.0.0",
            "capabilities": [
                "left-rail-tabs", "empty-workspaces", "hide-show-tabs", "agent-tools", "workflow-json",
                "project-files", "autosave", "render-queue", "asset-library", "version-tree", "export-center",
                "plugin-manifests", "permission-gates", "model-family-support-matrix",
            ],
            "endpoints": [
                "/api/pro/projects", "/api/pro/projects/{project_id}", "/api/pro/projects/{project_id}/autosave",
                "/api/pro/workflows/node-registry", "/api/pro/workflows/templates", "/api/pro/workflows/validate",
                "/api/pro/queue", "/api/pro/assets/library", "/api/pro/receipts", "/api/pro/export/presets",
                "/api/pro/plugins/registry", "/api/pro/agent/permissions", "/api/pro/model-families",
            ],
        }

    @router.get("/model-families")
    def model_families() -> dict[str, Any]:
        from aiwf.core.config.settings import RuntimeFlags, UserSettings
        from aiwf.services.model_family_support import build_model_family_matrix

        flags = getattr(ctx, "flags", None) or RuntimeFlags()
        settings = getattr(ctx, "settings", None) or UserSettings()
        return build_model_family_matrix(flags, settings)

    @router.get("/extensions/tabs")
    def extension_tabs() -> dict[str, Any]:
        return {"tabs": _load_tabs(ctx)}

    @router.post("/extensions/tabs")
    def save_extension_tabs(payload: PaidExtensionTabsPayload) -> dict[str, Any]:
        return _save_tabs(ctx, payload.tabs)

    @router.post("/extensions/register-tab")
    def register_extension_tab(tab: PaidExtensionTab) -> dict[str, Any]:
        current = [PaidExtensionTab.model_validate(item) for item in _load_tabs(ctx)]
        without_existing = [item for item in current if item.id != tab.id]
        return _save_tabs(ctx, [*without_existing, tab])

    @router.get("/extensions/workspaces/{workspace_id}")
    def extension_workspace(workspace_id: str) -> dict[str, Any]:
        path = _workspace_path(ctx, workspace_id)
        payload = _read_json(path, None)
        if isinstance(payload, dict):
            return payload
        return {
            "schema": "aiwf.paid.extensions.workspace.v4",
            "id": workspace_id,
            "title": workspace_id.replace("-", " ").title(),
            "type": "empty",
            "createdAt": _now(),
            "blocks": [],
        }

    @router.post("/extensions/workspaces/{workspace_id}")
    def save_extension_workspace(workspace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload = {**payload, "id": workspace_id, "updatedAt": _now()}
        return _write_json(_workspace_path(ctx, workspace_id), payload)

    @router.get("/projects")
    def list_projects() -> dict[str, Any]:
        projects = []
        for path in sorted(_projects_root(ctx).glob("*/project.json")):
            payload = _read_json(path, None)
            if isinstance(payload, dict):
                projects.append({
                    "id": payload.get("id", path.parent.name),
                    "name": payload.get("name", path.parent.name),
                    "updatedAt": payload.get("updatedAt", ""),
                    "sceneCount": len(payload.get("scenes", []) if isinstance(payload.get("scenes"), list) else []),
                })
        if not projects:
            project = _load_project(ctx, "default")
            projects.append({"id": project["id"], "name": project["name"], "updatedAt": project.get("updatedAt", ""), "sceneCount": len(project.get("scenes", []))})
        return {"projects": projects}

    @router.post("/projects")
    def create_project(payload: PaidProjectPayload) -> dict[str, Any]:
        project_id = _safe_id(payload.id or payload.name, "project")
        project = _default_project(project_id, payload.name)
        project.update(payload.data or {})
        return _write_json(_project_path(ctx, project_id), project)

    @router.get("/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        return _load_project(ctx, project_id)

    @router.post("/projects/{project_id}")
    def save_project(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _save_project(ctx, project_id, payload)

    @router.post("/projects/{project_id}/autosave")
    def autosave_project(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        autosave = {"schema": "aiwf.media-project.autosave.v1", "projectId": project_id, "savedAt": _now(), "payload": payload}
        return _write_json(_project_dir(ctx, project_id) / "autosave.json", autosave)

    @router.get("/projects/{project_id}/versions")
    def project_versions(project_id: str) -> dict[str, Any]:
        project = _load_project(ctx, project_id)
        versions = project.get("versions") if isinstance(project.get("versions"), list) else []
        return {"versions": versions}

    @router.post("/projects/{project_id}/versions")
    def add_project_version(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        project = _load_project(ctx, project_id)
        versions = project.get("versions") if isinstance(project.get("versions"), list) else []
        version = {"id": payload.get("id") or f"version-{uuid4().hex[:8]}", "createdAt": _now(), **payload}
        project["versions"] = [version, *versions]
        _save_project(ctx, project_id, project)
        return {"version": version, "versions": project["versions"]}

    @router.get("/workflows/node-registry")
    def workflow_node_registry() -> dict[str, Any]:
        return {"classes": ["prompt", "image", "mask", "video", "audio", "metadata", "artifact"], "nodes": _node_registry()}

    @router.get("/workflows/templates")
    def workflow_templates() -> dict[str, Any]:
        return {"templates": _workflow_templates()}

    @router.post("/workflows/validate")
    def validate_workflow(payload: PaidWorkflowValidatePayload) -> dict[str, Any]:
        return _validate_workflow_payload(payload.workflow)

    @router.get("/workflows/{workflow_id}")
    def load_workflow(workflow_id: str) -> dict[str, Any]:
        workflow = _read_json(_workflow_path(ctx, workflow_id), None)
        if isinstance(workflow, dict):
            return workflow
        template = _workflow_templates()[0]
        return {"schema": "aiwf.pipeline-atlas.workflow.v1", "id": workflow_id, "label": template["label"], "stages": template["stages"]}

    @router.post("/workflows/{workflow_id}")
    def save_workflow(workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload = {**payload, "id": workflow_id, "updatedAt": _now(), "validation": _validate_workflow_payload(payload)}
        return _write_json(_workflow_path(ctx, workflow_id), payload)

    @router.get("/queue")
    def render_queue() -> dict[str, Any]:
        return _load_queue(ctx)

    @router.post("/queue/jobs")
    def add_queue_job(payload: PaidQueueJobPayload) -> dict[str, Any]:
        queue = _load_queue(ctx)
        job = {
            "id": f"job-{uuid4().hex[:8]}",
            "kind": payload.kind,
            "label": payload.label,
            "status": "queued",
            "progress": 0,
            "createdAt": _now(),
            "updatedAt": _now(),
            "payload": payload.payload,
        }
        queue["jobs"] = [job, *(queue.get("jobs") or [])]
        _save_queue(ctx, queue)
        return {"job": job, "queue": queue}

    @router.post("/queue/jobs/{job_id}/{action}")
    def update_queue_job(job_id: str, action: str) -> dict[str, Any]:
        queue = _load_queue(ctx)
        status_map = {"pause": "paused", "resume": "queued", "cancel": "cancelled", "retry": "queued", "complete": "completed", "start": "running"}
        if action not in status_map:
            raise HTTPException(status_code=422, detail="action must be pause, resume, cancel, retry, start, or complete")
        for job in queue.get("jobs", []):
            if isinstance(job, dict) and job.get("id") == job_id:
                job["status"] = status_map[action]
                job["updatedAt"] = _now()
                if action == "complete":
                    job["progress"] = 100
                _save_queue(ctx, queue)
                return {"job": job, "queue": queue}
        raise HTTPException(status_code=404, detail="Queue job not found")

    @router.get("/queue/worker")
    def queue_worker_status() -> dict[str, Any]:
        return {"worker": _get_worker(ctx).status()}

    @router.post("/queue/worker/start")
    def queue_worker_start() -> dict[str, Any]:
        worker = _get_worker(ctx)
        started = worker.start()
        return {"started": started, "worker": worker.status()}

    @router.post("/queue/worker/stop")
    def queue_worker_stop() -> dict[str, Any]:
        worker = _get_worker(ctx)
        stopped = worker.stop()
        return {"stopped": stopped, "worker": worker.status()}

    @router.post("/queue/worker/run-next")
    def queue_worker_run_next() -> dict[str, Any]:
        worker = _get_worker(ctx)
        if worker.is_running:
            raise HTTPException(status_code=409, detail="Worker loop is running; single-step is disabled.")
        job = worker.run_next()
        return {"ran": job is not None, "job": job, "worker": worker.status()}

    @router.get("/queue/jobs/{job_id}/log")
    def queue_job_log(job_id: str) -> dict[str, Any]:
        path = _worker_logs_dir(ctx) / f"{_safe_id(job_id, 'job')}.log"
        if not path.is_file():
            return {"jobId": job_id, "log": ""}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not read job log: {exc}") from exc
        return {"jobId": job_id, "log": text[-40_000:]}

    @router.get("/assets/library")
    def asset_library() -> dict[str, Any]:
        return {"root": str(_outputs_root(ctx)), "assets": _scan_assets(ctx), "tags": ["image", "video", "audio", "mask", "receipt", "export"]}

    @router.get("/receipts")
    def receipts() -> dict[str, Any]:
        return {"receipts": _scan_receipts(ctx)}

    @router.post("/qa/analyze")
    def qa_analyze(payload: PaidQaAnalyzePayload) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        prompt = payload.prompt.strip()
        if len(prompt) < 12:
            checks.append({"id": "prompt-short", "status": "warn", "label": "Prompt is very short", "suggestion": "Add subject, style, camera, lighting, and intent."})
        if payload.workflow:
            validation = _validate_workflow_payload(payload.workflow)
            checks.append({"id": "workflow-classes", "status": "pass" if validation["valid"] else "fail", "label": "Workflow type contracts", "suggestion": "; ".join(validation.get("errors", [])) or "Classes line up."})
        if not payload.assetPath:
            checks.append({"id": "asset-missing", "status": "info", "label": "No active asset selected", "suggestion": "Select an output to enable visual QA checks."})
        checks.append({"id": "metadata", "status": "pass", "label": "Receipt plan", "suggestion": "Save JSON receipt before export for reproducibility."})
        score = max(0, 100 - sum(20 for item in checks if item["status"] == "fail") - sum(8 for item in checks if item["status"] == "warn"))
        return {"score": score, "checks": checks, "analyzedAt": _now()}

    @router.get("/export/presets")
    def export_presets() -> dict[str, Any]:
        return {"presets": _export_presets()}

    @router.post("/export/plan")
    def export_plan(payload: PaidExportPlanPayload) -> dict[str, Any]:
        preset = next((item for item in _export_presets() if item["id"] == payload.preset), _export_presets()[0])
        plan = {
            "schema": "aiwf.export-plan.v1",
            "id": f"export-{uuid4().hex[:8]}",
            "projectId": payload.projectId,
            "preset": preset,
            "assetIds": payload.assetIds,
            "settings": payload.settings,
            "createdAt": _now(),
            "status": "planned",
        }
        path = _paid_runtime_dir(ctx) / "exports" / f"{plan['id']}.json"
        return _write_json(path, plan)

    @router.get("/plugins/registry")
    def plugin_registry() -> dict[str, Any]:
        return _load_plugins(ctx)

    @router.post("/plugins/install-manifest")
    def install_plugin_manifest(payload: PaidPluginManifestPayload) -> dict[str, Any]:
        manifest = payload.manifest
        plugin_id = _safe_id(str(manifest.get("id") or manifest.get("name") or "plugin"), "plugin")
        plugin = {
            "id": plugin_id,
            "name": str(manifest.get("name") or plugin_id),
            "version": str(manifest.get("version") or "0.1.0"),
            "manifest": manifest,
            "installedAt": _now(),
            "permissions": manifest.get("permissions") if isinstance(manifest.get("permissions"), list) else [],
            "enabled": True,
        }
        registry = _load_plugins(ctx)
        existing = [item for item in registry.get("plugins", []) if isinstance(item, dict) and item.get("id") != plugin_id]
        registry["plugins"] = [plugin, *existing]
        registry["updatedAt"] = _now()
        _write_json(_plugins_path(ctx), registry)
        ui = manifest.get("ui") if isinstance(manifest.get("ui"), dict) else {}
        if ui.get("leftRail"):
            tab = PaidExtensionTab(id=plugin_id, label=plugin["name"], workspaceType="empty", description="Plugin workspace")
            register_extension_tab(tab)
        return {"plugin": plugin, "registry": registry}

    @router.get("/agent/permissions")
    def agent_permissions() -> dict[str, Any]:
        payload = _read_json(_permissions_path(ctx), None)
        if isinstance(payload, dict):
            return payload
        default = PaidAgentPermissionsPayload(allowedTools=["project-reader", "workflow-json", "prompt-refiner", "log-viewer"]).model_dump(mode="json")
        return _write_json(_permissions_path(ctx), {"schema": "aiwf.agent.permissions.v1", "updatedAt": _now(), **default})

    @router.post("/agent/permissions")
    def save_agent_permissions(payload: PaidAgentPermissionsPayload) -> dict[str, Any]:
        return _write_json(_permissions_path(ctx), {"schema": "aiwf.agent.permissions.v1", "updatedAt": _now(), **payload.model_dump(mode="json")})

    @router.get("/agent/tools")
    def agent_tools() -> dict[str, Any]:
        return {"tools": _agent_tools()}

    @router.get("/agent/ollama/models")
    def ollama_models() -> dict[str, Any]:
        url = f"{_ollama_url(ctx)}/api/tags"
        try:
            payload = _get_json(url, timeout=3.0)
        except RuntimeError as exc:
            return {"available": False, "error": f"Ollama unavailable at {url}: {exc}", "models": []}
        models = []
        for item in payload.get("models", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            models.append({"id": str(item.get("name") or item.get("model") or ""), "name": str(item.get("name") or item.get("model") or "local-model"), "size": str(item.get("size") or ""), "modifiedAt": str(item.get("modified_at") or "")})
        return {"available": True, "models": models}

    @router.post("/agent/chat")
    def agent_chat(payload: PaidAgentChatPayload) -> dict[str, Any]:
        model = payload.model.strip()
        if not model:
            raise HTTPException(status_code=422, detail="Select an Ollama model first.")
        permissions = _read_json(_permissions_path(ctx), {})
        system_hint = (
            "You are AIWF Studio Agent inside a local AI media creation center. "
            "You can inspect project state, draft workflows, draft patches, explain logs, and propose plugins. "
            "Do not claim to write files, execute destructive commands, install plugins, or run heavy jobs without explicit user approval. "
            f"Enabled tools: {', '.join(payload.enabledTools) or 'none'}. "
            f"Permission profile: {json.dumps(permissions)[:900]}."
        )
        messages = [{"role": "system", "content": system_hint}]
        for message in payload.messages[-20:]:
            role = message.role if message.role in {"system", "user", "assistant", "tool"} else "user"
            if message.content.strip():
                messages.append({"role": role, "content": message.content})
        request_payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": float(payload.temperature)}}
        started = time.perf_counter()
        try:
            result = _post_json(f"{_ollama_url(ctx)}/api/chat", request_payload, timeout=180.0)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama chat failed: {exc}") from exc
        result["elapsedSeconds"] = round(time.perf_counter() - started, 3)
        return result

    def _build_chat_messages(payload: PaidAgentChatPayload) -> list[dict[str, str]]:
        permissions = _read_json(_permissions_path(ctx), {})
        system_hint = (
            "You are AIWF Studio Agent inside a local AI media creation center. "
            "You can inspect project state, draft workflows, draft patches, explain logs, and propose plugins. "
            "Do not claim to write files, execute destructive commands, install plugins, or run heavy jobs without explicit user approval. "
            f"Enabled tools: {', '.join(payload.enabledTools) or 'none'}. "
            f"Permission profile: {json.dumps(permissions)[:900]}."
        )
        messages = [{"role": "system", "content": system_hint}]
        for message in payload.messages[-20:]:
            role = message.role if message.role in {"system", "user", "assistant", "tool"} else "user"
            if message.content.strip():
                messages.append({"role": role, "content": message.content})
        return messages

    @router.post("/agent/chat/stream")
    def agent_chat_stream(payload: PaidAgentChatPayload) -> StreamingResponse:
        model = payload.model.strip()
        if not model:
            raise HTTPException(status_code=422, detail="Select an Ollama model first.")
        request_payload = {
            "model": model,
            "messages": _build_chat_messages(payload),
            "stream": True,
            "options": {"temperature": float(payload.temperature)},
        }

        def event_stream():
            request = urllib.request.Request(
                f"{_ollama_url(ctx)}/api/chat",
                data=json.dumps(request_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300.0) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        # Forward Ollama's NDJSON chunks as-is; each line is a JSON object
                        # with message.content deltas and a final {"done": true} record.
                        yield line + "\n"
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                yield json.dumps({"error": f"Ollama stream failed: {exc}", "done": True}) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    @router.get("/agent/context/roots")
    def agent_context_roots() -> dict[str, Any]:
        roots = _context_roots(ctx)
        return {"roots": [{"key": key, "path": str(path), "exists": path.is_dir()} for key, path in roots.items()]}

    @router.get("/agent/context/tree")
    def agent_context_tree(root: str = "data", path: str = "") -> dict[str, Any]:
        base = _resolve_context_path(ctx, root, path)
        if not base.is_dir():
            raise HTTPException(status_code=404, detail="Directory not found.")
        entries: list[dict[str, Any]] = []
        try:
            children = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not list directory: {exc}") from exc
        for child in children:
            if child.name.startswith(".") or child.name in {"node_modules", "__pycache__", "venv", ".git"}:
                continue
            try:
                stat = child.stat()
            except OSError:
                continue
            entries.append({
                "name": child.name,
                "path": str(child.relative_to(_context_roots(ctx)[root])),
                "type": "dir" if child.is_dir() else "file",
                "sizeBytes": 0 if child.is_dir() else stat.st_size,
                "readable": child.is_dir() or (child.suffix.lower() in _CONTEXT_TEXT_SUFFIXES and stat.st_size <= _CONTEXT_MAX_BYTES),
            })
            if len(entries) >= _CONTEXT_MAX_ENTRIES:
                break
        return {"root": root, "path": path, "entries": entries}

    @router.get("/agent/context/file")
    def agent_context_file(root: str = "data", path: str = "") -> dict[str, Any]:
        target = _resolve_context_path(ctx, root, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="File not found.")
        if target.suffix.lower() not in _CONTEXT_TEXT_SUFFIXES:
            raise HTTPException(status_code=415, detail="Only text-like files are readable in agent context.")
        try:
            size = target.stat().st_size
            if size > _CONTEXT_MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"File exceeds context cap ({_CONTEXT_MAX_BYTES} bytes).")
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not read file: {exc}") from exc
        return {"root": root, "path": path, "sizeBytes": size, "content": text}

    return router
