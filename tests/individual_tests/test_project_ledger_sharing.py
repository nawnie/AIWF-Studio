"""The shared project ledger under concurrent use (aiwf/services/unified_bridge.py).

On Windows a reader can hit PermissionError while os.replace swaps the ledger file,
and the swap can hit it while a reader holds the file open. These tests pin the fix:
reads and writes in one process share the ledger lock, and both sides retry that
transient error briefly, so a project read during an image job's append never fails.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from aiwf.services import unified_bridge
from aiwf.services.unified_bridge import BridgeError, ProjectLedger


def test_reads_and_appends_from_many_threads_never_fail(tmp_path: Path) -> None:
    ledger = ProjectLedger(tmp_path)
    project_id = ledger.create("sharing")["project_id"]
    failures: list[BaseException] = []

    # this worker appends events while the others keep reading the same file
    def writer() -> None:
        try:
            for index in range(60):
                ledger.append(project_id, "test_event", {"index": index})
        except BaseException as exc:   # recorded, then asserted below
            failures.append(exc)

    def reader() -> None:
        try:
            for _ in range(150):
                ledger.get(project_id)
                ledger.list()
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(2)] + [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not failures, failures[:3]
    assert len(ledger.get(project_id)["events"]) == 120


def test_transient_sharing_violations_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = ProjectLedger(tmp_path)
    project_id = ledger.create("retry")["project_id"]
    monkeypatch.setattr(unified_bridge, "_SHARING_RETRY_SECONDS", 0)

    # the swap fails twice as if another process were reading, then succeeds
    real_replace = os.replace
    replace_failures = {"left": 2}

    def flaky_replace(source, target):
        if replace_failures["left"]:
            replace_failures["left"] -= 1
            raise PermissionError(13, "The process cannot access the file")
        return real_replace(source, target)

    monkeypatch.setattr(unified_bridge.os, "replace", flaky_replace)
    ledger.append(project_id, "after_retry", {})
    assert replace_failures["left"] == 0

    # a read fails twice as if the file were mid-swap, then succeeds
    real_read = Path.read_text
    read_failures = {"left": 2}

    def flaky_read(self, *args, **kwargs):
        if self.name == f"{project_id}.json" and read_failures["left"]:
            read_failures["left"] -= 1
            raise PermissionError(13, "The process cannot access the file")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read)
    assert ledger.get(project_id)["events"][-1]["kind"] == "after_retry"
    assert read_failures["left"] == 0


def test_a_lasting_permission_problem_still_reports_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = ProjectLedger(tmp_path)
    project_id = ledger.create("locked")["project_id"]
    monkeypatch.setattr(unified_bridge, "_SHARING_RETRY_SECONDS", 0)

    def always_denied(self, *args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "read_text", always_denied)
    with pytest.raises(BridgeError) as raised:
        ledger.get(project_id)
    assert raised.value.code == "project_unreadable"
