from __future__ import annotations

from importlib import metadata

from aiwf.engine_workers.base import WorkerContext, emit_complete, emit_status, run_worker


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "MISSING"


def main(ctx: WorkerContext) -> None:
    mode = str(ctx.request.get("mode") or "probe")
    emit_status(ctx.job_id, f"Wan worker mode: {mode}")
    emit_status(
        ctx.job_id,
        "Runtime packages: "
        f"torch={_version('torch')}, "
        f"diffusers={_version('diffusers')}, "
        f"transformers={_version('transformers')}, "
        f"gguf={_version('gguf')}, "
        f"sageattention={_version('sageattention')}",
    )
    if mode != "probe":
        raise RuntimeError(
            "Wan isolated worker is installed, but generation dispatch has not been migrated yet. "
            "Use mode='probe' for readiness checks."
        )
    emit_complete(ctx.job_id, "Wan worker probe complete.")


if __name__ == "__main__":
    run_worker(main)
