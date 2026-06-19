"""Tests for aiwf/services/process_supervisor.py."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from aiwf.core.domain.worker import WorkerCommand
from aiwf.services.process_supervisor import ProcessSupervisor, get_process_supervisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _echo_cmd(*lines: str) -> WorkerCommand:
    """Build a WorkerCommand that prints each line then exits."""
    script = "; ".join(f'print({l!r})' for l in lines)
    return WorkerCommand(
        args=[sys.executable, "-c", script],
        cwd=Path("/tmp"),
        env={},
        name="echo-test",
    )


def _sleep_cmd(seconds: float) -> WorkerCommand:
    """Build a WorkerCommand that sleeps."""
    return WorkerCommand(
        args=[sys.executable, "-c", f"import time; time.sleep({seconds})"],
        cwd=Path("/tmp"),
        env={},
        name="sleep-test",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcessSupervisor:

    def setup_method(self):
        self.sup = ProcessSupervisor()

    def test_start_echo_yields_lines(self):
        cmd = _echo_cmd("hello", "world")
        lines = list(self.sup.start("echo", cmd))
        assert lines == ["hello", "world"]

    def test_worker_not_running_after_finish(self):
        cmd = _echo_cmd("done")
        list(self.sup.start("echo", cmd))
        assert not self.sup.is_running("echo")

    def test_start_check_raises_after_streaming_failed_worker(self):
        cmd = WorkerCommand(
            args=[
                sys.executable,
                "-c",
                "import sys; print('about to fail', flush=True); sys.exit(3)",
            ],
            cwd=Path("/tmp"),
            env={},
            name="fail-test",
        )
        lines = []

        with pytest.raises(RuntimeError, match="exited with code 3"):
            for line in self.sup.start("fail-test", cmd, check=True):
                lines.append(line)

        assert lines == ["about to fail"]
        assert not self.sup.is_running("fail-test")

    def test_running_workers_empty_after_finish(self):
        cmd = _echo_cmd("line1")
        list(self.sup.start("echo", cmd))
        assert "echo" not in self.sup.running_workers()

    def test_double_start_raises(self):
        """Starting a worker that's already running should raise RuntimeError."""
        started = threading.Event()
        ready = threading.Event()

        long_cmd = WorkerCommand(
            args=[sys.executable, "-c",
                  "import time; print('started', flush=True); time.sleep(10)"],
            cwd=Path("/tmp"),
            env={},
            name="long-job",
        )

        collected = []

        def _runner():
            for line in self.sup.start("long-job", long_cmd):
                collected.append(line)
                started.set()  # signal after first line

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        started.wait(timeout=5)  # wait until worker is alive

        with pytest.raises(RuntimeError, match="already running"):
            list(self.sup.start("long-job", long_cmd))

        self.sup.stop("long-job")
        t.join(timeout=5)

    def test_stop_terminates_running_process(self):
        long_cmd = _sleep_cmd(30)
        started = threading.Event()
        lines = []

        def _runner():
            for line in self.sup.start("sleep-job", long_cmd):
                lines.append(line)
                started.set()
            started.set()  # ensure set even if no output

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        # Give the process time to start
        time.sleep(0.3)

        status = self.sup.stop("sleep-job")
        t.join(timeout=5)

        assert not self.sup.is_running("sleep-job")
        assert "stopped" in status.lower()

    def test_stop_non_existent_worker(self):
        status = self.sup.stop("nonexistent")
        assert "not registered" in status.lower()

    def test_get_pid_while_running(self):
        started = threading.Event()
        long_cmd = WorkerCommand(
            args=[sys.executable, "-c",
                  "import time; print('alive', flush=True); time.sleep(10)"],
            cwd=Path("/tmp"),
            env={},
            name="pid-test",
        )

        def _runner():
            for _ in self.sup.start("pid-test", long_cmd):
                started.set()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        started.wait(timeout=5)

        pid = self.sup.get_pid("pid-test")
        assert pid is not None and pid > 0

        self.sup.stop("pid-test")
        t.join(timeout=5)

    def test_env_variable_passed_to_child(self):
        cmd = WorkerCommand(
            args=[sys.executable, "-c",
                  "import os; print(os.environ.get('AIWF_TEST_VAR', 'MISSING'))"],
            cwd=Path("/tmp"),
            env={"AIWF_TEST_VAR": "hello_from_env"},
            name="env-test",
        )
        lines = list(self.sup.start("env-test", cmd))
        assert lines == ["hello_from_env"]

    def test_stop_all_clears_workers(self):
        """stop_all() should terminate all running workers."""
        long_cmd = _sleep_cmd(30)

        threads = []
        for name in ("a", "b"):
            cmd = WorkerCommand(
                args=[sys.executable, "-c",
                      f"import time; print({name!r}, flush=True); time.sleep(30)"],
                cwd=Path("/tmp"),
                env={},
                name=name,
            )
            started = threading.Event()

            def _runner(n=name, c=cmd, ev=started):
                for _ in self.sup.start(n, c):
                    ev.set()
                ev.set()

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            threads.append((t, started))

        for t, ev in threads:
            ev.wait(timeout=5)

        time.sleep(0.2)
        self.sup.stop_all()

        for t, _ in threads:
            t.join(timeout=5)

        assert self.sup.running_workers() == []


class TestGetProcessSupervisor:
    def test_singleton(self):
        s1 = get_process_supervisor()
        s2 = get_process_supervisor()
        assert s1 is s2

    def test_returns_process_supervisor_instance(self):
        assert isinstance(get_process_supervisor(), ProcessSupervisor)
