"""llama-server backend (mainline llama.cpp and the PrismML fork).

Each model runs as its own ``llama-server`` child process on its own port.
That keeps the PrismML fork (needed for Bonsai's PTQ1_0/PQ2_0 weights) and
mainline llama.cpp (MTP, Qwen3-ASR, embeddings) side by side, and lets the
ModelManager free VRAM by simply stopping a process.

``launch`` keys in models.yaml become CLI flags: ``snake_case`` -> ``--kebab-case``,
``true`` -> bare flag, ``false``/``null`` -> omitted.  String values may use the
``{model}``, ``{mmproj}`` and ``{draft}`` (any ``files`` key) placeholders, which
resolve to absolute paths under the models directory.  Flag spellings move
between llama.cpp builds, so keep build-specific flags in config, not code.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..registry import ModelSpec
from .base import BackendError

logger = logging.getLogger(__name__)

# Keys with dedicated handling; everything else in ``launch`` is passed generically.
_RESERVED_LAUNCH_KEYS = frozenset({"port", "extra"})


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def resolve_files(spec: ModelSpec, models_dir: Path) -> dict[str, str]:
    base = models_dir / spec.id
    return {name: str((base / rel) if not Path(rel).is_absolute() else Path(rel)) for name, rel in spec.files.items()}


def build_command(
    spec: ModelSpec,
    *,
    binary: str | Path,
    port: int,
    models_dir: str | Path,
    host: str = "127.0.0.1",
) -> list[str]:
    """Build the llama-server argv for *spec* (pure; no filesystem access)."""
    if "model" not in spec.files:
        raise BackendError(f"{spec.id}: files.model is required for llama-server")
    files = resolve_files(spec, Path(models_dir))

    cmd: list[str] = [str(binary), "-m", files["model"], "--host", host, "--port", str(port)]
    if spec.ctx:
        cmd += ["-c", str(spec.ctx)]
    if spec.on_gpu:
        if spec.kv_type != "f16":
            cmd += ["-ctk", spec.kv_type, "-ctv", spec.kv_type]
    if "mmproj" in files:
        cmd += ["--mmproj", files["mmproj"]]
        if not spec.mmproj_on_gpu:
            cmd.append("--no-mmproj-offload")

    for key, value in spec.launch.items():
        if key in _RESERVED_LAUNCH_KEYS or value is None or value is False:
            continue
        if value is True:
            cmd.append(_flag(key))
            continue
        if isinstance(value, str):
            try:
                value = value.format(**files)
            except KeyError as exc:
                raise BackendError(f"{spec.id}: launch.{key} references missing file {exc}") from exc
        cmd += [_flag(key), str(value)]

    cmd += [str(arg) for arg in spec.launch.get("extra", []) or []]
    return cmd


@dataclass
class _Running:
    process: subprocess.Popen
    port: int
    log_path: Path


class LlamaServerBackend:
    """Spawns one llama-server process per model.

    ``binary`` is the llama-server executable for this backend flavour
    (PrismML fork or mainline).  Ports come from ``launch.port`` or are
    assigned sequentially from ``base_port``.
    """

    def __init__(
        self,
        binary: str | Path,
        models_dir: str | Path,
        *,
        base_port: int = 8100,
        host: str = "127.0.0.1",
        slot_dir: str | Path | None = None,
        log_dir: str | Path = "logs",
        ready_timeout_s: float = 180.0,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.binary = Path(binary)
        self.models_dir = Path(models_dir)
        self.host = host
        self.slot_dir = Path(slot_dir) if slot_dir else None
        self.log_dir = Path(log_dir)
        self.ready_timeout_s = ready_timeout_s
        self._popen = popen
        self._next_port = base_port
        self._ports: dict[str, int] = {}
        self._running: dict[str, _Running] = {}

    def port_for(self, spec: ModelSpec) -> int:
        if spec.id not in self._ports:
            explicit = spec.launch.get("port")
            if explicit:
                self._ports[spec.id] = int(explicit)
            else:
                self._ports[spec.id] = self._next_port
                self._next_port += 1
        return self._ports[spec.id]

    def base_url(self, model_id: str) -> str:
        return f"http://{self.host}:{self._ports[model_id]}"

    # Backend protocol ------------------------------------------------------

    def load(self, spec: ModelSpec) -> None:
        if spec.id in self._running:
            return
        if not self.binary.exists():
            raise BackendError(f"llama-server binary not found: {self.binary}")
        model_path = Path(resolve_files(spec, self.models_dir)["model"])
        if not model_path.exists():
            raise BackendError(f"{spec.id}: model file not found: {model_path}")

        port = self.port_for(spec)
        cmd = build_command(spec, binary=self.binary, port=port, models_dir=self.models_dir, host=self.host)
        if self.slot_dir is not None and "--slot-save-path" not in cmd:
            self.slot_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--slot-save-path", str(self.slot_dir)]
        logger.info("[llama] starting %s: %s", spec.id, " ".join(cmd))
        # Log to a file, not a pipe: llama-server is chatty and a full pipe would block it.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{spec.id}.log"
        with open(log_path, "wb") as log:
            process = self._popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        self._running[spec.id] = _Running(process, port, log_path)
        try:
            self._wait_ready(spec.id)
        except BackendError:
            self.unload(spec.id)
            raise

    def unload(self, model_id: str) -> None:
        running = self._running.pop(model_id, None)
        if running is None:
            return
        running.process.terminate()
        try:
            running.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            running.process.kill()
            running.process.wait(timeout=10)

    def save_state(self, model_id: str) -> bool:
        return self._slot_action(model_id, "save")

    def restore_state(self, model_id: str) -> bool:
        return self._slot_action(model_id, "restore")

    # Helpers ---------------------------------------------------------------

    def _wait_ready(self, model_id: str) -> None:
        running = self._running[model_id]
        deadline = time.monotonic() + self.ready_timeout_s
        url = f"{self.base_url(model_id)}/health"
        while time.monotonic() < deadline:
            if running.process.poll() is not None:
                tail = running.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise BackendError(f"{model_id}: llama-server exited early (log: {running.log_path})\n{tail}")
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        raise BackendError(f"{model_id}: llama-server not ready after {self.ready_timeout_s:.0f}s")

    def _slot_action(self, model_id: str, action: str) -> bool:
        """POST /slots/0?action=save|restore (needs llama-server --slot-save-path)."""
        if model_id not in self._running or self.slot_dir is None:
            return False
        body: dict[str, Any] = {"filename": f"{model_id}.slot.bin"}
        request = urllib.request.Request(
            f"{self.base_url(model_id)}/slots/0?action={action}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("[llama] slot %s failed for %s: %s", action, model_id, exc)
            return False
