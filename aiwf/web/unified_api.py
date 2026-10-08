"""Same-origin Pro API for the unified Studio workspace (/api/pro/unified/*).

The React "Studio Flow" surface calls only these routes. They delegate to
aiwf.services.unified_bridge, which makes the server-side loopback calls to
Dataset Studio, ReTrain and Qwen Chat. Contract summary (schema_version "1"):

    GET  /api/pro/unified/describe                       operations, parameters, workflow, rules (for agents)
    GET  /api/pro/unified/outputs?limit=20               recent Studio images (relative paths) for catalog-outputs
    GET  /api/pro/unified/status                         capability per app
    GET  /api/pro/unified/projects                       shared project list
    POST /api/pro/unified/projects                       {name} -> new project
    GET  /api/pro/unified/projects/{id}                  project + ledger events
    POST /api/pro/unified/projects/{id}/catalog-outputs  {output_paths} -> Dataset Studio assets
    GET  /api/pro/unified/datasets/packages              Dataset Studio ReTrain packages
    POST /api/pro/unified/retrain/import                 {project_id, package_name, manifest_sha256}
    POST /api/pro/unified/retrain/preflight              {project_id, dataset_id, manifest_sha256, model_id, settings}
    GET  /api/pro/unified/projects/{id}/qwen-context     exact text a Qwen send would include
    POST /api/pro/unified/projects/{id}/qwen-ask         {model_id, question}
    GET  /api/pro/unified/model-families                 which trained artifacts work where
    POST /api/pro/unified/images                         {prompt, aspect_ratio, quality, seed?, project_id?} -> image job
    GET  /api/pro/unified/images/{job_id}                job state and saved images
    POST /api/pro/unified/images/{job_id}/cancel         dequeue or interrupt that job only
    GET  /api/pro/unified/images/{job_id}/files/{index}  the PNG (UI display)
    GET  /api/pro/unified/setup                          save and model folders (launch.json), with drive facts
    POST /api/pro/unified/setup/folders                  {output_dir?, models_dir?, ckpt_dir?, extra_*_dirs?} (setup wizard only)

Errors come back as HTTP 4xx/5xx with detail {"code", "message"}.
There is no route here that starts training.
"""

from __future__ import annotations

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from aiwf.api.security import LOOPBACK_HOSTS
from aiwf.services import output_metadata, setup_settings, unified_contract
from aiwf.services.unified_bridge import MODEL_FAMILIES, BridgeConfig, BridgeError, UnifiedBridge


# --- request bodies (unknown fields are rejected) -----------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateProjectBody(_Strict):
    name: str = Field(min_length=1, max_length=120)


class CatalogOutputsBody(_Strict):
    # paths are relative to Studio's output folder, as in /api/pro/outputs/<path> URLs
    output_paths: list[str] = Field(min_length=1, max_length=200)


class ImportPackageBody(_Strict):
    project_id: str = Field(min_length=1, max_length=64)
    package_name: str = Field(min_length=1, max_length=180)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PreflightBody(_Strict):
    project_id: str = Field(min_length=1, max_length=64)
    dataset_id: str = Field(pattern=r"^sha256-[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1, max_length=120)
    settings: dict[str, Any] = Field(default_factory=dict)


class DownloadModelBody(_Strict):
    model_id: str = Field(min_length=1, max_length=120)


class HfTokenBody(_Strict):
    # the user's Hugging Face key; accepted from the UI only and never returned, logged or stored by Studio
    token: str = Field(min_length=1, max_length=512, repr=False)


class QwenAskBody(_Strict):
    model_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)


class RecordContextQuestionBody(_Strict):
    model_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SetupFoldersBody(_Strict):
    # any subset of the five folder settings; "" or [] means "use the default"
    output_dir: str | None = Field(default=None, max_length=1024)
    models_dir: str | None = Field(default=None, max_length=1024)
    ckpt_dir: str | None = Field(default=None, max_length=1024)
    extra_model_dirs: list[str] | None = Field(default=None, max_length=32)
    extra_ckpt_dirs: list[str] | None = Field(default=None, max_length=32)
    create_missing: bool = False


class GenerateImageBody(_Strict):
    # value checks (aspect names, quality, seed range) happen in image_jobs.py so every surface gets the same messages
    prompt: str = Field(min_length=1, max_length=2000)
    aspect_ratio: str = "1:1"
    quality: str = "draft"
    seed: int | None = None
    project_id: str | None = Field(default=None, max_length=64)


# --- helpers -----------------------------------------------------------------------
def _require_local(request: Request) -> None:
    """Cross-app writes are allowed only from this machine, even for paired phones."""
    host = request.client.host if request.client else ""
    if host not in LOOPBACK_HOSTS:
        raise HTTPException(status_code=403, detail={"code": "local_only", "message": "Cross-app actions are available only on the local machine."})


def _as_http(exc: BridgeError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message})


def _output_relative(value: str) -> str:
    """Accept either a relative output path or the /api/pro/outputs/<path> URL the UI already has."""
    text = value.strip()
    prefix = "/api/pro/outputs/"
    if text.startswith(prefix):
        text = unquote(text[len(prefix):].split("?", 1)[0])
    return text


def build_unified_router(ctx: Any, *, bridge: UnifiedBridge | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/pro/unified")
    state: dict[str, UnifiedBridge | None] = {"bridge": bridge}
    lock = threading.Lock()

    # this helper builds the bridge lazily so Pro startup never waits on sibling apps
    def get_bridge() -> UnifiedBridge:
        with lock:
            if state["bridge"] is None:
                state["bridge"] = UnifiedBridge(BridgeConfig.from_environment(Path(ctx.flags.data_dir)))
            return state["bridge"]  # type: ignore[return-value]

    def output_root() -> Path | None:
        resolved = getattr(getattr(ctx, "flags", None), "resolved_output_dir", None)
        if not callable(resolved):
            return None
        try:
            return Path(resolved()).resolve()
        except OSError:
            return None

    # --- read-only routes --------------------------------------------------------------
    # self-description for agents: every operation, its parameters, workflow and rules
    # (the CLI and MCP server are generated from the same contract module)
    @router.get("/describe")
    def describe() -> dict[str, Any]:
        return unified_contract.describe()

    # recent Studio images with paths relative to the output folder, so an agent can
    # pass them straight to catalog-outputs; provenance is read from the files here
    @router.get("/outputs")
    def list_outputs(limit: int = 20) -> dict[str, Any]:
        if not 1 <= limit <= 200:
            raise HTTPException(status_code=422, detail={"code": "invalid_limit", "message": "limit must be between 1 and 200."})
        root = output_root()
        if root is None:
            raise HTTPException(status_code=503, detail={"code": "no_output_root", "message": "Studio's output folder is not available."})
        # torch-free helpers (aiwf/services/output_metadata.py), so the standalone engine API stays light
        outputs = []
        # this loop summarizes each output without sending image bytes
        for path in output_metadata.recent_paths_from_disk(root, limit=limit):
            try:
                stat = path.stat()
                settings = output_metadata.settings_from_infotext(output_metadata.read_output_infotext(path))
            except OSError:
                continue
            outputs.append({
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                "prompt": str(settings.get("prompt", ""))[:500],
                "model": settings.get("modelName"),
                "seed": settings.get("seed"),
                "width": settings.get("width"),
                "height": settings.get("height"),
            })
        return {"schema_version": "1", "outputs": outputs}

    @router.get("/status")
    def unified_status() -> dict[str, Any]:
        return get_bridge().status(studio_output_root=output_root())

    # --- guided setup: where images are saved and where models are found (launch.json) ----------
    # Reading is open to agents; changing folders is the setup wizard's job (UI only, local only).
    # New folders apply when Pro or the engine API next starts, so this never moves a live runtime.
    @router.get("/setup")
    def get_setup() -> dict[str, Any]:
        return setup_settings.describe_setup(Path(ctx.flags.data_dir))

    @router.post("/setup/folders")
    def save_setup_folders(body: SetupFoldersBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        changes = body.model_dump(exclude_none=True, exclude={"create_missing"})
        if not changes:
            raise HTTPException(status_code=422, detail={"code": "nothing_to_save", "message": "Send at least one folder setting."})
        try:
            return setup_settings.save_folders(Path(ctx.flags.data_dir), changes, create_missing=body.create_missing)
        except setup_settings.SetupError as exc:
            detail = {"code": exc.code, "message": exc.message}
            if exc.field:
                detail["field"] = exc.field
            raise HTTPException(status_code=422, detail=detail) from exc

    @router.get("/model-families")
    def model_families() -> dict[str, Any]:
        return {"schema_version": "1", "families": MODEL_FAMILIES}

    @router.get("/projects")
    def list_projects() -> dict[str, Any]:
        try:
            return get_bridge().list_projects()
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.get("/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        try:
            return get_bridge().get_project(project_id)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.get("/datasets/packages")
    def list_packages() -> dict[str, Any]:
        try:
            return get_bridge().list_packages()
        except BridgeError as exc:
            raise _as_http(exc) from exc

    # --- model weights: catalog, Hugging Face key (UI only), downloads --------------------------
    @router.get("/models")
    def list_models(refresh: bool = False) -> dict[str, Any]:
        try:
            return get_bridge().list_models(refresh=refresh)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.get("/models/downloads/{job_id}")
    def download_status(job_id: str) -> dict[str, Any]:
        try:
            return get_bridge().download_status(job_id)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.get("/projects/{project_id}/qwen-context")
    def qwen_context(project_id: str) -> dict[str, Any]:
        try:
            return get_bridge().qwen_context(project_id)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    # --- explicit user actions (local machine only) ------------------------------------
    @router.post("/projects")
    def create_project(body: CreateProjectBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().create_project(body.name)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/projects/{project_id}/catalog-outputs")
    def catalog_outputs(project_id: str, body: CatalogOutputsBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        root = output_root()
        if root is None:
            raise HTTPException(status_code=503, detail={"code": "no_output_root", "message": "Studio's output folder is not available."})
        # provenance is read from the files on disk, never taken from the browser
        try:
            return get_bridge().catalog_outputs(
                project_id,
                [_output_relative(item) for item in body.output_paths],
                output_root=root,
                read_provenance=lambda path: output_metadata.settings_from_infotext(output_metadata.read_output_infotext(path)),
            )
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/retrain/import")
    def import_package(body: ImportPackageBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().import_package(body.project_id, body.package_name, body.manifest_sha256)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/retrain/preflight")
    def preflight(body: PreflightBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().preflight(body.project_id, body.dataset_id, body.manifest_sha256, body.model_id, body.settings)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/models/download")
    def download_model(body: DownloadModelBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().download_model(body.model_id)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/models/downloads/{job_id}/cancel")
    def cancel_download(job_id: str, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().cancel_download(job_id)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/models/hf-token")
    def save_hf_token(body: HfTokenBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().save_hf_token(body.token)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/models/hf-token/clear")
    def clear_hf_token(request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().clear_hf_token()
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/projects/{project_id}/qwen-ask")
    def qwen_ask(project_id: str, body: QwenAskBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().qwen_ask(project_id, body.model_id, body.question)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/projects/{project_id}/chat-question")
    def record_chat_question(project_id: str, body: RecordContextQuestionBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().record_context_question(project_id, body.model_id, body.question, body.context_sha256)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    # --- images: Qwen Image 2.1 on the local ComfyUI (job logic in aiwf/services/image_jobs.py) ---
    @router.post("/images")
    def generate_image(body: GenerateImageBody, request: Request) -> dict[str, Any]:
        _require_local(request)
        from aiwf.services.model_startup import pro_model_load_lock

        operation_lock = pro_model_load_lock(ctx)
        if not operation_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail={"code": "gpu_busy", "message": "Another model load or GPU operation is in progress."})
        release_state = {"released": False}
        release_state_lock = threading.Lock()

        def release_gpu_lease(_job: dict[str, Any] | None = None) -> None:
            with release_state_lock:
                if release_state["released"]:
                    return
                release_state["released"] = True
            operation_lock.release()

        transferred = False
        try:
            # Pro's own generation jobs only exist when this router runs inside Pro, where pro_api is
            # already imported. The standalone engine API (aiwf/engine_api.py) has no Pro jobs, and
            # importing pro_api there would load torch and the whole Pro runtime for nothing.
            pro_api = sys.modules.get("aiwf.web.pro_api")
            if pro_api is not None:
                pro_api._assert_pro_model_load_idle(ctx)
                if (
                    pro_api._image_generation_running(ctx)
                    or pro_api._image_generation_pending(ctx)
                    or pro_api._pro_video_job_running(ctx)
                    or pro_api._pro_workflow_runs_active(ctx)
                ):
                    raise HTTPException(status_code=409, detail={"code": "gpu_busy", "message": "Wait for active GPU generation or workflow jobs to finish before starting Qwen Image."})
            result = get_bridge().start_image_job(
                prompt=body.prompt, aspect_ratio=body.aspect_ratio, quality=body.quality,
                seed=body.seed, project_id=body.project_id, output_root=output_root(),
                on_finished=release_gpu_lease,
            )
            transferred = True
            return result
        except BridgeError as exc:
            raise _as_http(exc) from exc
        finally:
            if not transferred:
                release_gpu_lease()

    @router.get("/images/{job_id}")
    def image_status(job_id: str) -> dict[str, Any]:
        try:
            return get_bridge().image_status(job_id)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    @router.post("/images/{job_id}/cancel")
    def cancel_image(job_id: str, request: Request) -> dict[str, Any]:
        _require_local(request)
        try:
            return get_bridge().cancel_image(job_id)
        except BridgeError as exc:
            raise _as_http(exc) from exc

    # the finished PNG itself, for UIs; agents use the relative paths from image_status instead
    @router.get("/images/{job_id}/files/{index}")
    def image_file(job_id: str, index: int) -> FileResponse:
        try:
            path = get_bridge().image_file(job_id, index)
        except BridgeError as exc:
            raise _as_http(exc) from exc
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    return router
