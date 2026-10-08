"""Guarded Pro startup loading for the user's saved image model."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_LOCK_INIT_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, "_CrossProcessOperationLock"] = {}


class _CrossProcessOperationLock:
    """Thread and process lock shared by Pro and the standalone engine API."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._thread_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._handle: Any = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        started = time.monotonic()
        while True:
            if self._thread_lock.acquire(blocking=False):
                break
            if not blocking or (timeout >= 0 and time.monotonic() - started >= timeout):
                return False
            time.sleep(0.05)

        deadline = None if timeout < 0 else started + timeout
        while True:
            handle = None
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                handle = self._path.open("a+b")
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, ImportError):
                if handle is not None:
                    handle.close()
                if not blocking or (deadline is not None and time.monotonic() >= deadline):
                    self._thread_lock.release()
                    return False
                time.sleep(0.05)
                continue
            with self._state_lock:
                self._handle = handle
            return True

    def release(self) -> None:
        with self._state_lock:
            handle, self._handle = self._handle, None
        if handle is None:
            raise RuntimeError("operation lock is not held")
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._thread_lock.release()


def pro_model_load_lock(ctx: Any) -> Any:
    """Return the shared GPU-operation lock for this Studio data directory."""
    lock = getattr(ctx, "_pro_model_load_lock", None)
    if lock is None:
        with _LOCK_INIT_GUARD:
            lock = getattr(ctx, "_pro_model_load_lock", None)
            if lock is None:
                flags = getattr(ctx, "flags", None)
                data_dir = getattr(flags, "data_dir", None)
                if data_dir is not None:
                    path = (Path(data_dir).expanduser() / "_local" / "model-operation.lock").resolve()
                    key = str(path).casefold()
                    lock = _FILE_LOCKS.get(key)
                    if lock is None:
                        lock = _CrossProcessOperationLock(path)
                        _FILE_LOCKS[key] = lock
                else:
                    # Lightweight test contexts may omit RuntimeFlags. Live
                    # Pro and engine API contexts always provide data_dir.
                    lock = threading.Lock()
                setattr(ctx, "_pro_model_load_lock", lock)
    return lock


def _is_disabled() -> bool:
    return os.environ.get("AIWF_PRO_STARTUP_MODEL_LOAD", "1").strip().lower() in {
        "0", "false", "no", "off",
    }


def pro_startup_model_loading_enabled() -> bool:
    return not _is_disabled()


def pro_image_support_ids(
    ctx: Any,
    model_id: str,
    engine_id: str = "",
    architecture: str = "",
) -> list[str]:
    """Return exact locally resolved support paths for image route identity.

    Resolution is metadata-only: it never loads tensor weights. The stable
    engine/architecture identifiers remain in the identity when support files
    are absent, while resolved file paths let lifecycle state detect replacing
    a support asset in place.
    """
    support_ids = [value for value in (engine_id, architecture) if value]
    backend = getattr(getattr(ctx, "generation", None), "backend", None)
    if backend is None:
        return support_ids
    # The Diffusers backend fingerprints a configured external VAE as part of
    # checkpoint support. Include it in Pro route identity as well, so a VAE
    # replacement invalidates startup residency for every image family.
    vae_path = str(getattr(getattr(ctx, "flags", None), "vae_path", "") or "").strip()
    if vae_path:
        support_ids.append(str(Path(vae_path).expanduser()))
    try:
        checkpoint = backend.resolve_checkpoint(model_id)
    except Exception:
        return support_ids

    normalized_architecture = str(architecture or getattr(checkpoint, "architecture", "")).strip().lower()
    checkpoint_path = Path(str(getattr(checkpoint, "path", ""))).expanduser()
    if checkpoint_path.is_file():
        # Single-file checkpoints are the base model itself. Tracking only
        # their sidecars misses an in-place checkpoint replacement.
        support_ids.append(str(checkpoint_path))
    if normalized_architecture in {"flux", "flux_fill"}:
        resolver = getattr(backend, "_resolve_flux_component_paths", None)
        if callable(resolver):
            try:
                support_ids.extend(str(path) for path in resolver().values())
            except Exception:
                # The normal route preflight reports missing assets. Keep this
                # identity stable and allow the setup state to be recorded.
                pass
        # Flux prompt encoding also depends on locally resolved tokenizer
        # snapshots. Track those alongside the encoder weights so replacing a
        # tokenizer invalidates the selected route's support revision.
        for name in ("_resolve_flux_clip_tokenizer_path", "_resolve_flux_t5_tokenizer_path"):
            tokenizer_resolver = getattr(backend, name, None)
            if callable(tokenizer_resolver):
                try:
                    support_ids.append(str(tokenizer_resolver()))
                except Exception:
                    # Missing tokenizers remain a normal preflight failure.
                    pass
    elif normalized_architecture in {"flux_kontext", "flux2_klein", "z_image", "zimage"}:
        resolver = getattr(backend, "_resolve_component_dir", None)
        if callable(resolver):
            try:
                component_dir = resolver(normalized_architecture, checkpoint)
                support_ids.append(str(component_dir))
                # These component folders are small compared with model
                # weights; fingerprint their files individually so replacing
                # a nested encoder/tokenizer/VAE file invalidates lifecycle
                # state even when the folder timestamp stays unchanged.
                support_ids.extend(
                    str(path)
                    for path in component_dir.rglob("*")
                    if path.is_file()
                )
            except Exception:
                pass
        # Flux.2 Klein and Z-Image can also be installed as complete
        # Diffusers snapshots. Their snapshot-local weights/config must be
        # part of route identity just like other folder-backed models; the
        # component resolver above primarily covers split-file layouts.
        if checkpoint_path.is_dir():
            flags = getattr(ctx, "flags", None)
            if flags is not None:
                try:
                    resolved_path = checkpoint_path.resolve(strict=True)
                    from aiwf.services.model_files import configured_model_roots

                    roots = [root.resolve() for root in configured_model_roots(flags)]
                    if any(resolved_path.is_relative_to(root) for root in roots):
                        support_ids.append(str(resolved_path))
                except (OSError, RuntimeError, ValueError):
                    # Keep route identity confined to configured model roots.
                    pass
    else:
        # Folder-backed Diffusers models can have their VAE, encoder, index,
        # or shard files replaced in place. Include the confined snapshot root
        # so support_revision's metadata walk invalidates stale lifecycle state.
        flags = getattr(ctx, "flags", None)
        if checkpoint_path.is_dir() and flags is not None:
            try:
                resolved_path = checkpoint_path.resolve(strict=True)
                from aiwf.services.model_files import configured_model_roots

                roots = [root.resolve() for root in configured_model_roots(flags)]
                if any(resolved_path.is_relative_to(root) for root in roots):
                    support_ids.append(str(resolved_path))
            except (OSError, RuntimeError, ValueError):
                # Readiness reports unsafe/missing paths; lifecycle tracking
                # must not follow a snapshot link outside configured roots.
                pass
    return support_ids


def _gpu_headroom_status(ctx: Any, model: dict[str, Any]) -> str | None:
    backend = getattr(getattr(ctx, "generation", None), "backend", None)
    devices = getattr(backend, "devices", None)
    device_fn = getattr(devices, "device", None)
    try:
        device = device_fn() if callable(device_fn) else None
        if getattr(device, "type", "") != "cuda":
            return None
        import torch

        from aiwf.services.gpu_memory import measured_cuda_free_bytes

        free_bytes = measured_cuda_free_bytes(torch, device)
        if free_bytes is None:
            return "Startup loading skipped because available GPU memory could not be matched to the active CUDA device."
        size_bytes = max(0, int(model.get("sizeBytes") or 0))
        estimated_gb = size_bytes / 1024**3
        flags = getattr(ctx, "flags", None)
        low_memory_profile = bool(getattr(flags, "lowvram", False) or getattr(flags, "medvram", False))
        required_gb = 3.0 if low_memory_profile else max(3.5, min(12.0, estimated_gb * 0.75 + 1.5))
        free_gb = float(free_bytes) / 1024**3
        if free_gb < required_gb:
            return f"Startup loading skipped: {free_gb:.1f} GB VRAM is free; this model needs an estimated {required_gb:.1f} GB headroom."
    except Exception:
        # When VRAM cannot be measured, avoid a speculative eager allocation.
        return "Startup loading skipped because available GPU memory could not be checked."
    return None


def _preload_pro_image_model_locked(ctx: Any) -> dict[str, str]:
    """Load the saved Pro image model in the background without generating.

    Selection and route readiness come from Pro's normal model registry. Video
    engines are excluded because Pro loads those through separate workers.
    """
    if _is_disabled():
        return {"status": "disabled", "modelId": "", "detail": "Startup model loading is disabled by AIWF_PRO_STARTUP_MODEL_LOAD."}

    generation = getattr(ctx, "generation", None)
    backend = getattr(generation, "backend", None)
    if generation is None or backend is None:
        return {"status": "unavailable", "modelId": "", "detail": "Image generation service is unavailable."}

    saved_model_id = str(getattr(getattr(ctx, "settings", None), "last_checkpoint_id", "") or "").strip()
    try:
        from aiwf.web.pro_api import _selectable_checkpoint_payloads, _settings_defaults

        selectable, blocked = _selectable_checkpoint_payloads(ctx)
        selected_id = str(_settings_defaults(ctx).get("checkpointId") or "").strip()
    except Exception as exc:
        logger.exception("Could not resolve the Pro startup image model")
        return {"status": "failed", "modelId": "", "detail": f"Could not inspect model readiness: {exc}"}
    model_id = selected_id
    selected_fallback = bool(saved_model_id and model_id and model_id != saved_model_id)
    selection_detail = (
        f"Saved model '{saved_model_id}' is unavailable. Using ready local image model '{model_id}'."
        if selected_fallback else "Checking the saved startup image model and its support assets."
    )

    from aiwf.services.route_lifecycle import select_route

    if model_id:
        select_route(
            ctx, route="image.txt2img", model_id=model_id, setup_ready=False,
            detail=selection_detail,
        )

    image_models = [
        model for model in selectable
        if model.get("routeStatus") == "request-eligible"
        and model.get("checkpointPathStatus") == "present"
        and model.get("engineId") not in {"wan", "sana_video", "ltx"}
    ]
    if not selected_id:
        return {"status": "no-model", "modelId": "", "detail": "No locally available, supported image model is ready to load."}
    selected = next((model for model in image_models if str(model.get("id") or "") == selected_id), None)
    if selected is None:
        selected_payload = next((model for model in selectable if str(model.get("id") or "") == selected_id), None)
        if selected_payload and str(selected_payload.get("engineId") or "") in {"wan", "sana_video", "ltx"}:
            return {
                "status": "route-managed",
                "modelId": selected_id,
                "detail": "The selected video model is prepared by its video route when a video job starts.",
            }
        # Keep startup useful when the saved image checkpoint still exists but
        # its family support assets or route runtime are incomplete. A model
        # is a fallback candidate only when route preflight says it is ready
        # and the backend independently confirms that all local files needed
        # for loading are present. Selection is persisted only after residency
        # is confirmed below.
        blocked_saved = next((
            item for item in blocked if str(item.get("id") or "") == selected_id
        ), None)
        can_preload = getattr(backend, "can_preload_checkpoint_locally", None)
        fallback = next((
            model for model in image_models
            if blocked_saved is not None
            and str(model.get("id") or "") != selected_id
            and callable(can_preload)
            and can_preload(str(model.get("id") or ""))
        ), None)
        if fallback is not None:
            saved_reason = str(
                (blocked_saved or {}).get("readinessReason")
                or (blocked_saved or {}).get("reason")
                or "its required support assets are incomplete"
            )
            model_id = selected_id = str(fallback.get("id") or "")
            selected = fallback
            selected_fallback = True
            selection_detail = (
                f"Saved model '{saved_model_id or selected_id}' is not ready ({saved_reason}). "
                f"Using ready local image model '{model_id}' for startup."
            )
            select_route(
                ctx, route="image.txt2img", model_id=model_id, setup_ready=False,
                detail=selection_detail,
            )
        else:
            return {
                "status": "not-ready",
                "modelId": selected_id,
                "detail": (
                    f"Selected startup fallback '{selected_id}' is not ready for image loading."
                    if selected_fallback else "The selected default model is not ready for image startup loading."
                ),
            }

    # Dedicated inpaint checkpoints are valid generation choices, but they
    # cannot be loaded through the shared txt2img startup route. If one was
    # saved as the startup default, keep it available for Inpaint jobs and
    # load the first locally preloadable standard image model for app startup.
    architecture = str(selected.get("architecture") or "").strip().lower().replace("-", "_")
    preserve_saved_selection = False
    if architecture in {"inpaint", "sd15_inpaint", "sdxl_inpaint", "flux_fill"}:
        can_preload_for_fallback = getattr(backend, "can_preload_checkpoint_locally", None)
        fallback = next((
            model for model in image_models
            if str(model.get("id") or "") != selected_id
            and callable(can_preload_for_fallback)
            and can_preload_for_fallback(str(model.get("id") or ""))
        ), None)
        if fallback is None:
            return {
                "status": "not-ready",
                "modelId": selected_id,
                "detail": "The saved model uses a dedicated inpaint route, and no standard image model is locally preloadable for startup.",
            }
        model_id = selected_id = str(fallback.get("id") or "")
        selected = fallback
        selected_fallback = True
        preserve_saved_selection = True
        selection_detail = f"Saved model '{saved_model_id or architecture}' uses a dedicated inpaint route. Loading standard image model '{model_id}' for startup."
        select_route(
            ctx, route="image.txt2img", model_id=model_id, setup_ready=False,
            detail=selection_detail,
        )

    support_ids = pro_image_support_ids(
        ctx, model_id, str(selected.get("engineId") or ""), str(selected.get("architecture") or "")
    )
    select_route(
        ctx, route="image.txt2img", model_id=model_id, setup_ready=True,
        support_ids=support_ids,
        detail=(
            f"Startup fallback '{model_id}' and its family support assets are ready for loading."
            if selected_fallback else "Saved image model and route support assets are ready for loading."
        ),
    )

    def not_ready(detail: str) -> dict[str, str]:
        select_route(
            ctx, route="image.txt2img", model_id=selected_id, setup_ready=False,
            support_ids=support_ids, detail=detail,
        )
        return {"status": "not-ready", "modelId": selected_id, "detail": detail}

    try:
        active_job = getattr(generation, "active_job", lambda: None)()
        pending_count = int(getattr(generation, "pending_count", lambda: 0)() or 0)
        if active_job is not None or pending_count:
            return {"status": "deferred", "modelId": selected_id, "detail": "Startup loading deferred because image work is active or queued."}
        supervisor = getattr(ctx, "supervisor", None)
        tenant = getattr(supervisor, "active_tenant", None)
        tenant_value = str(getattr(tenant, "value", tenant) or "").strip().lower()
        if tenant_value not in {"", "idle", "none"}:
            return {"status": "deferred", "modelId": selected_id, "detail": f"Startup loading deferred while {tenant_value} owns the GPU."}
        is_loaded = getattr(backend, "is_checkpoint_loaded", None)
        can_preload = getattr(backend, "can_preload_checkpoint_locally", None)
        if not callable(can_preload):
            return not_ready("The image backend cannot verify local model readiness.")
        try:
            already_loaded = bool(callable(is_loaded) and is_loaded(selected_id))
        except Exception as exc:
            return not_ready(f"The image backend could not confirm model residency: {exc}")
        if already_loaded:
            from aiwf.services.route_lifecycle import confirm_route_residency

            confirm_route_residency(ctx, "image.txt2img", selected_id, resident=True)
            if selected_fallback and not preserve_saved_selection:
                remember_selection = getattr(generation, "remember_checkpoint_selection", None)
                if callable(remember_selection):
                    remember_selection(selected_id)
            return {
                "status": "loaded", "modelId": selected_id,
                "detail": (
                    f"{selection_detail} The selected startup image model is already loaded."
                    if selected_fallback else "The selected startup image model was already loaded."
                ),
            }
        try:
            locally_preloadable = bool(can_preload(selected_id))
        except Exception as exc:
            return not_ready(f"The image backend could not verify local model readiness: {exc}")
        if not locally_preloadable:
            return not_ready("The selected image model or its support assets are incomplete for startup loading.")
        headroom_issue = _gpu_headroom_status(ctx, selected)
        if headroom_issue:
            return {"status": "deferred", "modelId": selected_id, "detail": headroom_issue}
        from aiwf.services.route_lifecycle import (
            begin_route_operation,
            confirm_route_residency,
            finish_route_operation,
            mark_route_running,
        )

        token = begin_route_operation(
            ctx,
            route="image.txt2img",
            model_id=model_id,
            setup_ready=True,
            support_ids=support_ids,
            detail="Loading saved image model and required support assets.",
        )
        if not token:
            return {
                "status": "deferred",
                "modelId": model_id,
                "detail": "Startup loading deferred because the selected image route changed during readiness checks.",
            }
        mark_route_running(ctx, "image.txt2img", token, "Saved image model is loading.")
        # Startup may have selected a fallback for a stale saved ID. Do not
        # persist that choice until the backend confirms residency below.
        generation.load_checkpoint(model_id, persist_selection=False)
        confirmed_loaded = bool(callable(is_loaded) and is_loaded(model_id))
        if not confirmed_loaded:
            finish_route_operation(
                ctx, "image.txt2img", token, success=False,
                detail="Image backend did not confirm the selected startup model is resident.",
            )
            return {"status": "load-unconfirmed", "modelId": model_id, "detail": "The loader returned, but the image backend did not confirm the selected model is resident."}
        if selected_fallback and not preserve_saved_selection:
            remember_selection = getattr(generation, "remember_checkpoint_selection", None)
            if callable(remember_selection):
                remember_selection(model_id)
        finish_route_operation(ctx, "image.txt2img", token, success=True, detail="Startup image model is resident and confirmed by the backend.", resident=True)
        confirm_route_residency(ctx, "image.txt2img", model_id, resident=True)
        return {
            "status": "loaded", "modelId": model_id,
            "detail": (
                f"{selection_detail} Loaded and confirmed '{model_id}'. No generation was started."
                if selected_fallback else "Saved image model loaded. No generation was started."
            ),
        }
    except Exception as exc:
        try:
            from aiwf.services.route_lifecycle import finish_route_operation

            finish_route_operation(
                ctx, "image.txt2img", locals().get("token", ""), success=False, detail=str(exc)
            )
        except Exception:
            logger.debug("Could not update startup route lifecycle state", exc_info=True)
        logger.exception("Pro startup model load failed for %s", model_id)
        supervisor = getattr(ctx, "supervisor", None)
        tenant = getattr(supervisor, "active_tenant", None)
        tenant_value = str(getattr(tenant, "value", tenant) or "").strip().lower()
        if tenant_value not in {"", "idle", "none"} or str(exc).lower().startswith("gpu busy:"):
            return {
                "status": "deferred",
                "modelId": model_id,
                "detail": f"Startup loading deferred because GPU ownership changed during the load: {exc}",
            }
        return {"status": "failed", "modelId": model_id, "detail": f"Startup model load failed: {exc}"}


def preload_pro_image_model(ctx: Any) -> dict[str, str]:
    """Claim the per-app load slot before inspecting or loading the startup model."""
    lock = pro_model_load_lock(ctx)
    if not lock.acquire(blocking=False):
        return {
            "status": "deferred",
            "modelId": "",
            "detail": "Startup loading deferred because another model load is in progress.",
        }
    try:
        return _preload_pro_image_model_locked(ctx)
    finally:
        lock.release()
