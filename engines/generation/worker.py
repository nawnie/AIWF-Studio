"""Generation engine worker entry point.

This worker runs in the main AIWF venv (shared with the UI) and dispatches
to the appropriate generation service based on ``request["_engine"]``.

Supported engines routed here:
  - "wan"        → aiwf.services.wan.WanService

Future engines (add as they are implemented):
  - "ltx"        → aiwf.services.ltx.LtxService
  - "generation" → aiwf.services.generation.GenerationService (SD image gen)

Invocation (by EngineSupervisor):
    venv/Scripts/python.exe engines/generation/worker.py <path/to/request.json>

The request.json must include:
    "_job_id"  : str   — assigned by the supervisor
    "_engine"  : str   — which sub-engine to call ("wan", "ltx", etc.)
    ... engine-specific fields (see domain models) ...
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so aiwf.* imports work.
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from aiwf.engine_workers.base import WorkerContext, emit_status, emit_error, run_worker


def main(ctx: WorkerContext) -> None:
    engine = ctx.request.get("_engine", "")
    emit_status(ctx.job_id, f"Generation worker dispatching to engine: {engine!r}")

    if engine == "wan":
        _run_wan(ctx)
    else:
        raise ValueError(
            f"Unknown engine {engine!r}. "
            f"Supported: 'wan'. Set _engine in the job request."
        )


def _run_wan(ctx: WorkerContext) -> None:
    """Invoke the Wan I2V service in-process (same venv, subprocess for isolation)."""
    from aiwf.core.domain.wan import WanI2VRequest
    from aiwf.services.wan import WanService

    req_data = {k: v for k, v in ctx.request.items() if not k.startswith("_")}
    outputs_root = ctx.request.get("_outputs_root", str(_ROOT / "outputs"))

    request = WanI2VRequest(**req_data)
    service = WanService()
    result = service.generate(request, outputs_root=outputs_root)

    from aiwf.engine_workers.base import emit_artifact, emit_complete
    emit_artifact(ctx.job_id, path=result.output_path)
    emit_complete(ctx.job_id, message=result.message or "Wan generation complete")


if __name__ == "__main__":
    run_worker(main)
