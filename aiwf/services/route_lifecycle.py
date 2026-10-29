"""Request-scoped lifecycle receipts for independently routed model families.

The tracker records control flow and verified readiness. It deliberately does
not infer GPU residency for routes whose backends do not expose it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiwf.infrastructure.file_revision import local_file_revision


_STATE_INIT_LOCK = threading.Lock()
_SNAPSHOT_SUPPORT_CHECK_SECONDS = 2.0


def support_revision(support_ids: list[str] | tuple[str, ...] = ()) -> str:
    """Return a stable, non-path-bearing revision for selected support assets.

    Existing files include metadata and bounded content probes so same-size
    replacements with preserved timestamps invalidate route state. Paths and
    file contents are never returned or copied into lifecycle status.
    """
    normalized: list[tuple[Any, ...]] = []
    for raw_value in support_ids:
        value = str(raw_value).strip()
        if not value:
            continue
        try:
            stat = Path(value).expanduser().stat()
        except (OSError, ValueError):
            normalized.append((value, "missing"))
        else:
            path = Path(value).expanduser()
            if path.is_dir():
                # Support resolvers sometimes return a component directory.
                # Its own mtime does not change when a file is replaced in
                # place, so include a metadata-only manifest of nested files.
                try:
                    for root, dirs, files in os.walk(path, followlinks=False):
                        dirs.sort()
                        files.sort()
                        root_path = Path(root)
                        for name in files:
                            file_path = root_path / name
                            try:
                                file_stat = file_path.stat()
                            except OSError:
                                normalized.append((str(file_path), "unavailable"))
                            else:
                                normalized.append((
                                    str(file_path), "file", *local_file_revision(file_path, file_stat),
                                ))
                except OSError:
                    normalized.append((value, "directory-unavailable", *local_file_revision(path, stat)))
                else:
                    normalized.append((value, "directory", *local_file_revision(path, stat)))
            else:
                normalized.append((value, "file", *local_file_revision(path, stat)))
    normalized = sorted(set(normalized))
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _state_store(ctx: Any) -> tuple[threading.RLock, dict[str, dict[str, Any]]]:
    with _STATE_INIT_LOCK:
        lock = getattr(ctx, "_pro_route_lifecycle_lock", None)
        if lock is None:
            lock = threading.RLock()
            setattr(ctx, "_pro_route_lifecycle_lock", lock)
        states = getattr(ctx, "_pro_route_lifecycle", None)
        if states is None:
            states = {}
            setattr(ctx, "_pro_route_lifecycle", states)
        return lock, states


def _support_ids_store(ctx: Any) -> dict[str, tuple[str, ...]]:
    support_ids = getattr(ctx, "_pro_route_support_ids", None)
    if support_ids is None:
        support_ids = {}
        setattr(ctx, "_pro_route_support_ids", support_ids)
    return support_ids


def _support_revision_check_store(ctx: Any) -> dict[str, tuple[float, str]]:
    checks = getattr(ctx, "_pro_route_support_revision_checks", None)
    if checks is None:
        checks = {}
        setattr(ctx, "_pro_route_support_revision_checks", checks)
    return checks


def select_route(
    ctx: Any,
    *,
    route: str,
    model_id: str,
    setup_ready: bool,
    support_ids: list[str] | tuple[str, ...] = (),
    detail: str = "",
    resident: bool | None = None,
    failed: bool = False,
) -> dict[str, Any]:
    """Record the current selection and invalidate stale in-flight completions."""
    route_key = str(route or "").strip().lower()
    selected_id = str(model_id or "").strip()
    support_id_values = tuple(str(value).strip() for value in support_ids if str(value).strip())
    revision = support_revision(support_id_values)
    lock, states = _state_store(ctx)
    with lock:
        _support_ids_store(ctx)[route_key] = support_id_values
        _support_revision_check_store(ctx)[route_key] = (time.monotonic(), revision)
        prior = states.get(route_key)
        same_selection = bool(
            prior
            and prior.get("modelId") == selected_id
            and prior.get("supportRevision") == revision
        )
        if same_selection:
            if failed:
                prior.update(status="failed", operationId="", resident=resident, detail=detail)
            elif not setup_ready:
                prior.update(status="needs-setup", operationId="", resident=resident, detail=detail)
            elif resident is not None:
                prior.update(
                    status="loaded" if resident else ("setup-ready" if prior.get("status") == "loaded" else prior.get("status")),
                    resident=resident,
                    detail=detail or prior.get("detail", ""),
                )
            elif prior.get("status") == "needs-setup":
                prior.update(status="setup-ready", resident=resident, detail=detail)
            return dict(prior)
        state = {
            "route": route_key,
            "modelId": selected_id,
            "supportRevision": revision,
            "status": (
                "failed" if failed else
                "loaded" if setup_ready and resident is True else
                "setup-ready" if setup_ready else
                "needs-setup"
            ),
            "operationId": "",
            "resident": resident,
            "detail": detail,
        }
        states[route_key] = state
        return dict(state)


def begin_route_operation(
    ctx: Any,
    *,
    route: str,
    model_id: str,
    setup_ready: bool,
    support_ids: list[str] | tuple[str, ...] = (),
    detail: str = "Preparing the selected route and support assets.",
    resident: bool | None = None,
) -> str:
    """Start a route operation and return its stale-completion guard token."""
    selected = select_route(
        ctx,
        route=route,
        model_id=model_id,
        setup_ready=setup_ready,
        support_ids=support_ids,
        detail=detail,
        resident=resident,
    )
    if not setup_ready:
        return ""
    token = uuid4().hex
    lock, states = _state_store(ctx)
    with lock:
        current = states.get(str(route).strip().lower())
        if current is None or current.get("modelId") != selected.get("modelId") or current.get("supportRevision") != selected.get("supportRevision"):
            return ""
        current.update(status="preparing", operationId=token, detail=detail)
    return token


def mark_route_running(ctx: Any, route: str, token: str, detail: str = "Generation is running.") -> bool:
    if not token:
        return False
    lock, states = _state_store(ctx)
    with lock:
        state = states.get(str(route).strip().lower())
        if state is None or state.get("operationId") != token:
            return False
        state.update(status="running", detail=detail)
        return True


def finish_route_operation(
    ctx: Any,
    route: str,
    token: str,
    *,
    success: bool,
    detail: str,
    resident: bool | None = None,
    cancelled: bool = False,
) -> bool:
    """Finish only the current selection's operation; late results are ignored."""
    return _finish_transition(
        ctx,
        route,
        token,
        status="cancelled" if cancelled else "completed" if success else "failed",
        detail=detail,
        resident=resident,
    )


def finish_route_preparation(
    ctx: Any,
    route: str,
    token: str,
    *,
    detail: str,
    resident: bool | None = None,
) -> bool:
    """Finish preparation separately from generation completion."""
    return _finish_transition(
        ctx, route, token, status="prepared", detail=detail, resident=resident,
    )


def _finish_transition(
    ctx: Any,
    route: str,
    token: str,
    *,
    status: str,
    detail: str,
    resident: bool | None = None,
) -> bool:
    if not token:
        return False
    lock, states = _state_store(ctx)
    with lock:
        route_key = str(route).strip().lower()
        state = states.get(route_key)
        if state is None or state.get("operationId") != token:
            return False
        expected_revision = state.get("supportRevision")
        support_ids = _support_ids_store(ctx).get(route_key, ())

    # File stats, directory walks, and content probes can take time; keep them
    # outside the route-state lock so another route can still update/read state.
    current_revision = support_revision(support_ids)
    with lock:
        state = states.get(route_key)
        if state is None or state.get("operationId") != token or state.get("supportRevision") != expected_revision:
            return False
        _support_revision_check_store(ctx)[route_key] = (time.monotonic(), current_revision)
        if current_revision != expected_revision:
            state.update(
                status="needs-setup",
                supportRevision=current_revision,
                operationId="",
                resident=False,
                detail="Support assets changed while this operation was running. Recheck route readiness before continuing.",
            )
            return False
        state.update(status=status, detail=detail, operationId="")
        if resident is not None:
            state["resident"] = resident
        return True


def confirm_route_residency(ctx: Any, route: str, model_id: str, *, resident: bool) -> bool:
    """Set residency only for an identity confirmed by its owning backend."""
    lock, states = _state_store(ctx)
    with lock:
        state = states.get(str(route).strip().lower())
        if state is None or state.get("modelId") != str(model_id or "").strip():
            return False
        state["resident"] = bool(resident)
        if resident:
            state["status"] = "loaded"
            state["detail"] = "The route backend confirms this selected model is resident."
        elif state.get("status") == "loaded":
            state["status"] = "setup-ready"
            state["detail"] = "The route backend confirms the selected model is not resident."
        return True


def clear_route_residency(ctx: Any, route: str, model_id: str, *, detail: str = "") -> bool:
    """Clear a prior residency claim when the owning backend cannot reconfirm it."""
    lock, states = _state_store(ctx)
    with lock:
        state = states.get(str(route).strip().lower())
        if state is None or state.get("modelId") != str(model_id or "").strip():
            return False
        state["resident"] = None
        if state.get("status") == "loaded":
            state["status"] = "setup-ready"
        if detail:
            state["detail"] = detail
        return True


def lifecycle_snapshot(ctx: Any) -> list[dict[str, Any]]:
    lock, states = _state_store(ctx)
    with lock:
        support_ids = dict(_support_ids_store(ctx))
        checks = dict(_support_revision_check_store(ctx))
        snapshots = {key: dict(value) for key, value in states.items()}

    now = time.monotonic()
    active_statuses = {"loaded", "setup-ready", "preparing", "running", "prepared"}
    for route_key, state in snapshots.items():
        if state.get("status") not in active_statuses or not support_ids.get(route_key):
            continue
        checked_at, checked_revision = checks.get(route_key, (0.0, ""))
        if checked_revision == state.get("supportRevision") and now - checked_at < _SNAPSHOT_SUPPORT_CHECK_SECONDS:
            continue
        current_revision = support_revision(support_ids[route_key])
        with lock:
            current = states.get(route_key)
            if current is None or current.get("supportRevision") != state.get("supportRevision"):
                continue
            _support_revision_check_store(ctx)[route_key] = (time.monotonic(), current_revision)
            if current_revision != current.get("supportRevision"):
                current.update(
                    status="needs-setup",
                    supportRevision=current_revision,
                    operationId="",
                    resident=False,
                    detail="Support assets changed since this route was checked. Recheck route readiness before continuing.",
                )
            snapshots[route_key] = dict(current)
    return [snapshots[key] for key in sorted(snapshots)]
