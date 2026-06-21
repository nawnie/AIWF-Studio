from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from aiwf.core.domain.ltx import LTX_PIPELINE_DISTILLED
from aiwf.engine_workers.base import WorkerContext, emit_artifact, emit_complete, emit_status, run_worker


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "MISSING"


def _require_package(module: str, package: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(
            f"{package} is not importable in this LTX runtime. "
            "Run `scripts/bootstrap_ltx.ps1 -Enable`, then probe the engine again."
        )


def main(ctx: WorkerContext) -> None:
    mode = str(ctx.request.get("mode") or "probe")
    emit_status(ctx.job_id, f"LTX 2.3 worker mode: {mode}")
    emit_status(
        ctx.job_id,
        "Runtime packages: "
        f"torch={_version('torch')}, "
        f"ltx-core={_version('ltx-core')}, "
        f"ltx-pipelines={_version('ltx-pipelines')}, "
        f"transformers={_version('transformers')}",
    )
    _require_package("ltx_core", "ltx-core")
    _require_package("ltx_pipelines", "ltx-pipelines")

    if mode == "probe":
        emit_complete(ctx.job_id, "LTX 2.3 worker probe complete.")
        return
    if mode != "generate":
        raise RuntimeError(f"Unsupported LTX worker mode: {mode}")
    _run_generation(ctx)


def _run_generation(ctx: WorkerContext) -> None:
    request = ctx.request
    module = (
        "ltx_pipelines.distilled"
        if str(request.get("pipeline") or LTX_PIPELINE_DISTILLED) == LTX_PIPELINE_DISTILLED
        else "ltx_pipelines.ti2vid_one_stage"
    )
    output_path = Path(str(request["output_path"])).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        "-m",
        module,
        "--gemma-root",
        str(request["gemma_root"]),
        "--prompt",
        str(request.get("prompt") or ""),
        "--output-path",
        str(output_path),
        "--seed",
        str(int(request.get("seed") or 0)),
        "--height",
        str(int(request.get("height") or 512)),
        "--width",
        str(int(request.get("width") or 512)),
        "--num-frames",
        str(int(request.get("num_frames") or 81)),
        "--frame-rate",
        str(float(request.get("fps") or 25.0)),
    ]

    if module == "ltx_pipelines.distilled":
        args.extend(
            [
                "--distilled-checkpoint-path",
                str(request["checkpoint_path"]),
                "--spatial-upsampler-path",
                str(request["spatial_upsampler_path"]),
            ]
        )
    else:
        args.extend(
            [
                "--checkpoint-path",
                str(request["checkpoint_path"]),
                "--negative-prompt",
                str(request.get("negative_prompt") or ""),
                "--num-inference-steps",
                str(int(request.get("steps") or 20)),
            ]
        )

    source_image = str(request.get("source_image_path") or "").strip()
    if source_image:
        args.extend(["--image", source_image, "0", str(float(request.get("image_strength") or 0.8))])

    offload = str(request.get("offload") or "none").strip().lower()
    if offload and offload != "none":
        args.extend(["--offload", offload])

    quantization = str(request.get("quantization") or "").strip().lower()
    if quantization:
        args.extend(["--quantization", quantization])

    if bool(request.get("enhance_prompt")):
        args.append("--enhance-prompt")

    max_batch_size = int(request.get("max_batch_size") or 1)
    if max_batch_size > 1:
        args.extend(["--max-batch-size", str(max_batch_size)])

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    emit_status(ctx.job_id, f"Running {module} -> {output_path}")
    _run_child(ctx, args, env)

    if not output_path.is_file():
        raise RuntimeError(f"LTX pipeline exited without creating output: {output_path}")
    emit_artifact(ctx.job_id, path=str(output_path))
    emit_complete(ctx.job_id, f"LTX 2.3 video complete: {output_path.name}")


def _run_child(ctx: WorkerContext, args: list[str], env: dict[str, str]) -> None:
    popen_kwargs = {
        "args": args,
        "cwd": str(Path.cwd()),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(**popen_kwargs)
    assert proc.stdout is not None
    for raw in proc.stdout:
        text = raw.strip()
        if text:
            emit_status(ctx.job_id, text[:1200])
    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"LTX pipeline exited with code {proc.returncode}")


if __name__ == "__main__":
    run_worker(main)
