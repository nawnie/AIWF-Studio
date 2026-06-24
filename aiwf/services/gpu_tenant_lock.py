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
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator

logger = logging.getLogger(__name__)

# Tenants in priority order (higher = more important; lower-priority tenants
# can be pre-empted by higher-priority ones when an Ollama callback is wired).
_TENANT_PRIORITY: dict[str, int] = {
    "full_training": 100,
    "ed2": 100,
    "lora_training": 90,
    "kohya": 90,
    "video": 80,
    "wan": 80,
    "image": 70,
    "generation": 70,
    "enhance": 60,
    "chat": 10,
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
        self._condition = threading.Condition(threading.RLock())
        self._state: TenantState | None = None
        # Optional callback wired by the Ollama service so we can unload Ollama
        # before granting the GPU to a heavy tenant.
        self._ollama_unload_callback: "callable | None" = None

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    @property
    def is_free(self) -> bool:
        with self._condition:
            return self._state is None

    @property
    def active_tenant(self) -> str | None:
        with self._condition:
            return self._state.tenant if self._state else None

    @property
    def active_job_id(self) -> str | None:
        with self._condition:
            return self._state.job_id if self._state else None

    def status_message(self) -> str:
        """Human-readable current GPU state, suitable for UI display."""
        with self._condition:
            if self._state is None:
                return "GPU is free."
            return (
                f"GPU is currently owned by: {self._state.tenant}\n"
                f"Current job: {self._state.job_id}\n"
                f"Since: {self._state.acquired_at}"
            )

    def blocked_message(self, requesting_tenant: str) -> str:
        """Message to show when *requesting_tenant* cannot start because GPU is busy."""
        with self._condition:
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
        with self._condition:
            if self._state is None:
                self._state = TenantState(tenant=tenant, job_id=job_id)
                logger.info("[GPU] %s acquired lock for job %s", tenant, job_id)
                self._condition.notify_all()
                return True

            if self._state.tenant == tenant and self._state.job_id == job_id:
                return True

            if self._state.tenant == tenant:
                self._state = TenantState(tenant=tenant, job_id=job_id)
                logger.info("[GPU] %s re-acquired lock for job %s (same tenant)", tenant, job_id)
                self._condition.notify_all()
                return True

            current = self._state
            # Allow chat/Ollama to be pre-empted by higher-priority tenants.
            if (
                current.tenant in {"chat", "ollama"}
                and _TENANT_PRIORITY.get(tenant, 0) > _TENANT_PRIORITY.get(current.tenant, 0)
            ):
                logger.info(
                    "[GPU] Pre-empting %s (job %s) for %s (job %s)",
                    current.tenant, current.job_id, tenant, job_id,
                )
                cb = self._ollama_unload_callback
            else:
                cb = None

            if cb is not None:
                # Release the condition briefly so the callback can fire
                # without holding the lock (prevents deadlock if it re-enters).
                self._condition.release()
                unloaded = False
                try:
                    unloaded = cb() is not False
                except Exception:
                    logger.exception("[GPU] Ollama unload callback failed")
                finally:
                    self._condition.acquire()

                if not unloaded:
                    logger.warning("[GPU] %s blocked: chat/Ollama unload failed", tenant)
                    return False

                # Recheck — another thread might have acquired while the
                # callback was running. If chat/Ollama is still recorded, the
                # callback succeeded and this request owns the transition.
                if self._state is None or self._state.tenant in {"chat", "ollama"}:
                    self._state = TenantState(tenant=tenant, job_id=job_id)
                    logger.info("[GPU] %s acquired lock for job %s (post chat unload)", tenant, job_id)
                    self._condition.notify_all()
                    return True

            logger.warning(
                "[GPU] %s (job %s) blocked: GPU owned by %s (job %s)",
                tenant, job_id, current.tenant, current.job_id,
            )
            return False

    def wait_acquire(self, tenant: str, job_id: str, timeout: float | None = None) -> bool:
        """Wait until the GPU can be acquired by *tenant*/*job_id*.

        Returns False only if *timeout* elapses.  A timeout of None waits
        indefinitely.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            if self.try_acquire(tenant, job_id):
                return True
            with self._condition:
                if deadline is None:
                    self._condition.wait(timeout=0.25)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=min(0.25, remaining))

    def release(self, tenant: str, job_id: str) -> None:
        """Release the GPU lock.  Silently ignored if not held by this tenant/job."""
        with self._condition:
            if self._state and self._state.tenant == tenant and self._state.job_id == job_id:
                logger.info("[GPU] %s released lock for job %s", tenant, job_id)
                self._state = None
                self._condition.notify_all()

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
        with self._condition:
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
