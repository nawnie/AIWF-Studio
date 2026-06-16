from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from aiwf.core.domain.worker import WorkerCommand
from aiwf.services.process_supervisor import ProcessSupervisor, get_process_supervisor
from aiwf.services.worker_tenant import WorkerTenantRegistry


@dataclass(frozen=True)
class WorkerProbeResult:
    engine: str
    ok: bool
    request_path: Path
    events: tuple[dict, ...]
    raw_lines: tuple[str, ...]
    message: str


class WorkerProbeService:
    def __init__(
        self,
        repo_root: Path | str | None = None,
        *,
        registry: WorkerTenantRegistry | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        self.registry = registry or WorkerTenantRegistry(repo_root)
        self.repo_root = self.registry.repo_root
        self.supervisor = supervisor or get_process_supervisor()

    def probe(self, engine: str) -> WorkerProbeResult:
        job_id = f"{engine}-probe-{uuid4().hex[:8]}"
        request_path = self._write_request(engine, job_id)
        command = self.registry.build_command(engine, request_path)
        return self._run_probe(engine, request_path, command)

    def _write_request(self, engine: str, job_id: str) -> Path:
        root = self.repo_root / "outputs" / "worker-probes"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{job_id}.json"
        payload = {
            "_job_id": job_id,
            "_engine": engine,
            "_created_at": datetime.utcnow().isoformat(),
            "mode": "probe",
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _run_probe(self, engine: str, request_path: Path, command: WorkerCommand) -> WorkerProbeResult:
        events: list[dict] = []
        raw_lines: list[str] = []
        ok = False
        message = "Worker probe did not emit a terminal event."
        try:
            for line in self.supervisor.start(engine, command):
                raw_lines.append(line)
                event = _parse_event(line)
                if event is None:
                    continue
                events.append(event)
                kind = str(event.get("kind") or "")
                if kind == "complete":
                    ok = True
                    message = str(event.get("message") or "Worker probe complete.")
                elif kind == "error":
                    ok = False
                    message = str(event.get("message") or event.get("detail") or "Worker probe failed.")
        except Exception as exc:
            ok = False
            message = str(exc)
        return WorkerProbeResult(
            engine=engine,
            ok=ok,
            request_path=request_path,
            events=tuple(events),
            raw_lines=tuple(raw_lines),
            message=message,
        )


def _parse_event(line: str) -> dict | None:
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or "kind" not in event:
        return None
    return event
