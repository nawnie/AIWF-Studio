"""Single-tenant GPU lock for AIWF Studio.

The RTX 4070 Ti SUPER (16 GB VRAM) is a single-tenant GPU workspace.  Only
one heavy compute job should have exclusive use of the GPU at a time.

This module provides a *non-blocking* advisory lock.  Callers ask if they can
acquire the lock; the lock either grants it or returns a rejection reason that
the UI can surface to the user as a clear message::

    lock = GpuTenantLock()

    with lock.acquire("wan", job_id="job_001") as granted:
        if not granted:
            # GPU is busy — show user: lock.status_message()
            return
        # ... run Wan generation ...
    # lock released automatically on exit

Tenants (engine names that can own the GPU):
    - ``"wan"``         Wan 2.2 I2V video generation
    - ``"generation"``  image generation (SD / FLUX)
    - ``"kohya"``       Kohya LoRA training
    - ``"ed2"``         EveryDream2 full-model training
    - ``"ollama"``      Ollama chat (low-priority; pre-empted by heavy jobs)
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator

logger = logging.getLogger(__name__)

# Tenants in priority order (higher = more important; lower-priority tenants
# can be pre-empted by higher-priority ones when an Ollama callback is wired).
_TENANT_PRIORITY: dict[str, int] = {
    "ed2": 100,
    "kohya": 90,
    "wan": 80,
    "generation": 70,
    "ollama": 10,
}


@dataclass
class TenantState:
    tenant: str
    job_id: str
    acquired_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GpuTenantLock:
    """Thread-safe advisory GPU ownership lock.

    This is an *advisory* lock — it cannot actually prevent a rogue thread
    from using the GPU, but it ensures AIWF-controlled entry points honour the
    single-tenant policy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: TenantState | None = None
        # Optional callback wired by the Ollama service so we can unload Ollama
        # before granting the GPU to a heavy tenant.
        self._ollama_unload_callback: "callable | None" = None

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def is_free(self) -> bool:
        with self._lock:
            return self._state is None

    @property
    def active_tenant(self) -> str | None:
        with self._lock:
            return self._state.tenant if self._state else None

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return self._state.job_id if self._state else None

    def status_message(self) -> str:
        """Human-readable current GPU state, suitable for UI display."""
        with self._lock:
            if self._state is None:
                return "GPU is free."
            return (
                f"GPU is currently owned by: {self._state.tenant}\n"
                f"Current job: {self._state.job_id}\n"
                f"Since: {self._state.acquired_at}"
            )

    def blocked_message(self, requesting_tenant: str) -> str:
        """Message to show when *requesting_tenant* cannot start because GPU is busy."""
        with self._lock:
            if self._state is None:
                return ""
            return (
                f"GPU is currently owned by: {self._state.tenant} "
                f"(job {self._state.job_id}).\n"
                f"Action blocked: {requesting_tenant}.\n"
                "Options: wait for the current job to finish, stop it, or cancel."
            )

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def try_acquire(self, tenant: str, job_id: str) -> bool:
        """Attempt to acquire the GPU for *tenant*/*job_id*.

        Returns True if the lock was granted (GPU was free or Ollama was
        successfully pre-empted), False if the GPU is owned by another heavy
        tenant.
        """
        with self._lock:
            if self._state is None:
                self._state = TenantState(tenant=tenant, job_id=job_id)
                logger.info("[GPU] %s acquired lock for job %s", tenant, job_id)
                return True

            current = self._state
            # Allow Ollama to be pre-empted by higher-priority tenants.
            if current.tenant == "ollama" and _TENANT_PRIORITY.get(tenant, 0) > _TENANT_PRIORITY.get("ollama", 0):
                logger.info(
                    "[GPU] Pre-empting Ollama (job %s) for %s (job %s)",
                    current.job_id, tenant, job_id,
                )
                # Release the lock briefly so the callback can fire without
                # holding the lock (prevents deadlock if callback re-enters).
                cb = self._ollama_unload_callback
            else:
                cb = None

            if cb is not None:
                # Release lock, fire callback, reacquire.
                self._lock.release()
                try:
                    cb()
                except Exception:
                    logger.exception("[GPU] Ollama unload callback failed")
                finally:
                    self._lock.acquire()
                # Recheck — another thread might have slipped in.
                if self._state is None or self._state.tenant == "ollama":
                    self._state = TenantState(tenant=tenant, job_id=job_id)
                    logger.info("[GPU] %s acquired lock for job %s (post Ollama unload)", tenant, job_id)
                    return True

            logger.warning(
                "[GPU] %s (job %s) blocked: GPU owned by %s (job %s)",
                tenant, job_id, current.tenant, current.job_id,
            )
            return False

    def release(self, tenant: str, job_id: str) -> None:
        """Release the GPU lock.  Silently ignored if not held by this tenant/job."""
        with self._lock:
            if self._state and self._state.tenant == tenant and self._state.job_id == job_id:
                logger.info("[GPU] %s released lock for job %s", tenant, job_id)
                self._state = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def acquire(self, tenant: str, job_id: str) -> Generator[bool, None, None]:
        """Context manager that acquires and auto-releases the GPU tenant lock.

        Yields True if the lock was granted, False if blocked.  The caller
        should check the yielded value and bail out if False::

            with lock.acquire("wan", job_id) as granted:
                if not granted:
                    raise RuntimeError(lock.blocked_message("wan"))
                ...
        """
        granted = self.try_acquire(tenant, job_id)
        try:
            yield granted
        finally:
            if granted:
                self.release(tenant, job_id)

    # ------------------------------------------------------------------
    # Ollama integration
    # ------------------------------------------------------------------

    def register_ollama_unload(self, callback: "callable") -> None:
        """Register a callback that unloads Ollama from VRAM.

        Called automatically before granting the GPU to a higher-priority
        tenant (anything other than Ollama).  The callback should issue::

            POST /api/generate  {"model": "...", "keep_alive": 0}

        or the equivalent via the Ollama HTTP API.
        """
        self._ollama_unload_callback = callback


# ---------------------------------------------------------------------------
# Module-level singleton — shared across the whole process
# ---------------------------------------------------------------------------

_global_lock: GpuTenantLock | None = None
_init_lock = threading.Lock()


def get_gpu_lock() -> GpuTenantLock:
    """Return the process-global GPU tenant lock (created on first call)."""
    global _global_lock
    if _global_lock is None:
        with _init_lock:
            if _global_lock is None:
                _global_lock = GpuTenantLock()
    return _global_lock
