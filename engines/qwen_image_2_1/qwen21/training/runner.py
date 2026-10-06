"""Run a trainer subprocess and turn its console output into progress events."""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")  # tqdm "  12/2000 ["
_LOSS_RE = re.compile(r"loss[:=]\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)", re.IGNORECASE)
_EPOCH_RE = re.compile(r"[Ee]poch\s+(\d+)")


@dataclass
class Progress:
    step: int = 0
    total: int = 0
    loss: float | None = None
    epoch: int | None = None
    message: str = ""


def parse_progress(line: str, previous: Progress | None = None) -> Progress | None:
    """Return an updated Progress if the line carries step/loss info, else None."""
    m = _STEP_RE.search(line)
    loss = _LOSS_RE.search(line)
    epoch = _EPOCH_RE.search(line)
    if not (m or loss or epoch):
        return None
    prog = Progress(**(previous.__dict__ if previous else {}))
    if m:
        prog.step, prog.total = int(m.group(1)), int(m.group(2))
    if loss:
        try:
            prog.loss = float(loss.group(1))
        except ValueError:
            pass
    if epoch:
        prog.epoch = int(epoch.group(1))
    prog.message = line.strip()[-160:]
    return prog


class TrainingRunner:
    """Streams a subprocess; callbacks fire from a background thread."""

    def __init__(self, cmd: list[str], cwd: str | Path, env: dict[str, str] | None = None,
                 on_line: Callable[[str], None] | None = None,
                 on_progress: Callable[[Progress], None] | None = None,
                 on_exit: Callable[[int], None] | None = None,
                 log_path: str | Path | None = None) -> None:
        self.cmd, self.cwd = cmd, Path(cwd)
        self.env = dict(os.environ)
        self.env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        if env:
            self.env.update(env)
        self.on_line, self.on_progress, self.on_exit = on_line, on_progress, on_exit
        self.log_path = Path(log_path) if log_path else None
        self.process: subprocess.Popen[str] | None = None
        self.progress = Progress()
        self.started_at = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("already started")
        self.cwd.mkdir(parents=True, exist_ok=True)
        creation = {}
        if os.name == "nt":
            creation["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.started_at = time.time()
        self.process = subprocess.Popen(self.cmd, cwd=str(self.cwd), env=self.env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                        bufsize=1, **creation)
        self._thread = threading.Thread(target=self._pump, name="qwen21-trainer", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        log = self.log_path.open("a", encoding="utf-8") if self.log_path else None
        try:
            for raw in self.process.stdout:
                for line in raw.replace("\r", "\n").split("\n"):
                    if not line.strip():
                        continue
                    if log:
                        log.write(line + "\n")
                        log.flush()
                    if self.on_line:
                        self.on_line(line)
                    prog = parse_progress(line, self.progress)
                    if prog is not None:
                        self.progress = prog
                        if self.on_progress:
                            self.on_progress(prog)
        finally:
            if log:
                log.close()
            rc = self.process.wait()
            if self.on_exit:
                self.on_exit(rc)

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            if os.name == "nt":
                import signal

                self.process.send_signal(signal.CTRL_BREAK_EVENT)  # lets ai-toolkit finish its checkpoint
                time.sleep(2)
            self.process.terminate()
        except Exception:
            pass

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self.process is None else self.process.poll()
