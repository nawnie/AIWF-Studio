from __future__ import annotations

import logging
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.generation import GenerationRequest
from aiwf.core.domain.models import Checkpoint
from aiwf.infrastructure.diffusers.checkpoints import diffusers_dir_has_required_local_files, missing_diffusers_local_files
from aiwf.infrastructure.diffusers.model_arch import is_qwen_nunchaku_architecture
from aiwf.services.model_files import configured_model_roots, resolve_model_asset

logger = logging.getLogger(__name__)

_DEFAULT_TRANSFORMER_NAME = "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
_RUNTIME_PROBE_LOCK = threading.Lock()
_RUNTIME_PROBE_CACHE: dict[str, tuple[float, int, int, str]] = {}


def clear_qwen_nunchaku_runtime_probe_cache() -> None:
    with _RUNTIME_PROBE_LOCK:
        _RUNTIME_PROBE_CACHE.clear()


def _runtime_dependency_issue(python_exe: Path) -> str:
    """Verify isolated imports and pinned CUDA ABI without loading any model weights."""
    try:
        stat = python_exe.stat()
        cache_key = str(python_exe.resolve())
        with _RUNTIME_PROBE_LOCK:
            cached = _RUNTIME_PROBE_CACHE.get(cache_key)
            if cached and cached[:3] == (stat.st_mtime_ns, stat.st_size, int(time.monotonic() // 15)):
                return cached[3]
        code = (
            "import torch, diffusers, transformers, nunchaku; "
            "from diffusers import QwenImagePipeline; "
            "from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel; "
            "assert torch.version.cuda == '13.0', f'expected CUDA 13.0 torch, got {torch.version.cuda}'; "
            "assert torch.__version__.startswith('2.11.'), f'expected torch 2.11, got {torch.__version__}'"
        )
        result = subprocess.run(
            [str(python_exe), "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        issue = "" if result.returncode == 0 else (
            f"isolated runtime import check failed: {(result.stderr or result.stdout or '').strip()[-1200:]}"
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        issue = f"isolated runtime import check could not complete: {exc}"
        cache_key = str(python_exe)
        stat = None
    with _RUNTIME_PROBE_LOCK:
        if stat is not None:
            _RUNTIME_PROBE_CACHE[cache_key] = (
                stat.st_mtime_ns, stat.st_size, int(time.monotonic() // 15), issue
            )
    return issue


@dataclass(frozen=True)
class QwenNunchakuStatus:
    ready: bool
    python_exe: Path
    runner_script: Path
    base_dir: Path
    transformer_path: Path
    messages: tuple[str, ...]


class QwenNunchakuUnavailable(RuntimeError):
    pass


class QwenNunchakuService:
    def __init__(self, flags: RuntimeFlags | None = None) -> None:
        self.flags = flags or RuntimeFlags()

    def engine_root(self) -> Path:
        return self.flags.data_dir.resolve() / "engines" / "qwen_nunchaku"

    def python_exe(self) -> Path:
        if os.name == "nt":
            return self.engine_root() / ".venv" / "Scripts" / "python.exe"
        return self.engine_root() / ".venv" / "bin" / "python"

    def runner_script(self) -> Path:
        return self.engine_root() / "run_qwen_lightning.py"

    def downloads_base_dir(self) -> Path:
        return self.flags.data_dir.resolve() / "downloads" / "qwen_nunchaku" / "base"

    def models_base_dir(self) -> Path:
        return self.flags.resolved_models_dir() / "qwen-image" / "Diffusers" / "Qwen-Image"

    def base_dir(self) -> Path:
        candidate = resolve_model_asset(
            self.flags,
            (
                Path("qwen-image") / "Diffusers" / "Qwen-Image",
                Path("Diffusers") / "Qwen-Image",
                Path("Qwen-Image"),
            ),
            predicate=lambda path: (path / "model_index.json").is_file()
            and diffusers_dir_has_required_local_files(path),
        )
        if (candidate / "model_index.json").is_file() and diffusers_dir_has_required_local_files(candidate):
            return candidate
        downloaded = self.downloads_base_dir()
        if (downloaded / "model_index.json").is_file() and diffusers_dir_has_required_local_files(downloaded):
            return downloaded
        return self.models_base_dir()

    def models_root(self) -> Path:
        return self.flags.resolved_models_dir() / "qwen-image" / "Nunchaku"

    def output_dir(self) -> Path:
        return self.flags.resolved_output_dir() / "qwen-nunchaku"

    def default_transformer_path(self) -> Path:
        model_preferred = resolve_model_asset(
            self.flags,
            (
                Path("qwen-image") / "Nunchaku" / _DEFAULT_TRANSFORMER_NAME,
                Path("Nunchaku") / _DEFAULT_TRANSFORMER_NAME,
            ),
            fallback=self.models_root() / _DEFAULT_TRANSFORMER_NAME,
        )
        if model_preferred.is_file() and model_preferred.stat().st_size > 0:
            return model_preferred
        download_preferred = (
            self.flags.data_dir.resolve()
            / "downloads"
            / "qwen_nunchaku"
            / "transformer"
            / _DEFAULT_TRANSFORMER_NAME
        )
        if download_preferred.is_file() and download_preferred.stat().st_size > 0:
            return download_preferred
        for root in configured_model_roots(self.flags):
            for folder in (root / "qwen-image" / "Nunchaku", root / "Nunchaku"):
                try:
                    candidates = sorted(folder.glob("*.safetensors"), key=lambda path: path.name.lower())
                except OSError:
                    continue
                for candidate in candidates:
                    try:
                        resolved = candidate.resolve(strict=True)
                        resolved.relative_to(root)
                        if resolved.is_file() and resolved.stat().st_size > 0:
                            return resolved
                    except (OSError, RuntimeError, ValueError):
                        continue
        return model_preferred

    def status(self, transformer_path: str | Path | None = None) -> QwenNunchakuStatus:
        transformer = Path(transformer_path).resolve() if transformer_path else self.default_transformer_path()
        python_exe = self.python_exe()
        runner_script = self.runner_script()
        base_dir = self.base_dir()
        messages: list[str] = []
        if not python_exe.is_file():
            messages.append(f"engine runtime missing: {python_exe}")
        else:
            runtime_issue = _runtime_dependency_issue(python_exe)
            if runtime_issue:
                messages.append(runtime_issue)
        if not runner_script.is_file():
            messages.append(f"runner missing: {runner_script}")
        if not base_dir.is_dir():
            messages.append(f"base components missing: {base_dir}")
        elif not (base_dir / "model_index.json").is_file():
            messages.append(f"base components missing model_index.json: {base_dir}")
        elif not diffusers_dir_has_required_local_files(base_dir):
            missing = missing_diffusers_local_files(base_dir, limit=20)
            if missing:
                shard_names = ", ".join(str(path.relative_to(base_dir)) for path in missing)
                messages.append(f"base components incomplete; missing local shard files under {base_dir}: {shard_names}")
            else:
                messages.append(f"base components incomplete; missing local shard files under: {base_dir}")
        try:
            transformer_ready = transformer.is_file() and transformer.stat().st_size > 0
        except OSError:
            transformer_ready = False
        if not transformer_ready:
            messages.append(f"transformer missing: {transformer}")
        return QwenNunchakuStatus(
            ready=not messages,
            python_exe=python_exe,
            runner_script=runner_script,
            base_dir=base_dir,
            transformer_path=transformer,
            messages=tuple(messages),
        )

    @staticmethod
    def _headroom_issue() -> str | None:
        """Check the isolated CUDA worker's device without relying on Studio Torch."""
        try:
            from aiwf.services.gpu_memory import nvidia_smi_free_bytes

            free_bytes = nvidia_smi_free_bytes()
        except Exception:
            free_bytes = None
        if free_bytes is None:
            return "Qwen Nunchaku launch deferred because available GPU memory could not be verified."
        free_gb = float(free_bytes) / (1024**3)
        required_gb = 8.0
        if free_gb < required_gb:
            return (
                f"Qwen Nunchaku launch deferred: {free_gb:.1f} GB VRAM is free; "
                f"the selected route needs at least {required_gb:.1f} GB headroom."
            )
        return None

    def matches_checkpoint(self, checkpoint: Checkpoint) -> bool:
        return is_qwen_nunchaku_architecture(getattr(checkpoint, "architecture", ""))

    def generate(
        self,
        checkpoint: Checkpoint,
        request: GenerationRequest,
        *,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        seed: int,
        should_cancel=None,
    ) -> tuple[Image.Image, Path]:
        status = self.status(checkpoint.path)
        if not status.ready:
            details = "; ".join(status.messages) if status.messages else "runtime not ready"
            raise QwenNunchakuUnavailable(details)
        headroom_issue = self._headroom_issue()
        if headroom_issue:
            raise QwenNunchakuUnavailable(headroom_issue)

        output_dir = self.output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"qwen-nunchaku-{stamp}-{uuid4().hex[:6]}.png"

        args = [
            str(status.python_exe),
            str(status.runner_script),
            "--base-dir",
            str(status.base_dir),
            "--transformer",
            str(status.transformer_path),
            "--output",
            str(output_path),
            "--prompt",
            prompt,
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
            "--steps",
            str(int(steps)),
            "--cfg",
            str(float(request.cfg_scale)),
            "--blocks-on-gpu",
            "4",
            "--seed",
            str(int(seed)),
        ]
        negative_prompt = (request.negative_prompt or "").strip()
        if negative_prompt:
            args.extend(["--negative-prompt", negative_prompt])

        env = {
            **os.environ,
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTHONUNBUFFERED": "1",
        }
        logger.info("Running Qwen Nunchaku sidecar: %s", " ".join(args))
        proc = subprocess.Popen(
            args,
            cwd=str(self.flags.data_dir.resolve()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            while True:
                if should_cancel and should_cancel():
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise GenerationCancelledError()
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            stdout, _stderr = proc.communicate()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

        if proc.returncode not in (0, None):
            raise QwenNunchakuUnavailable(
                f"Qwen Nunchaku sidecar failed with code {proc.returncode}: {(stdout or '').strip()}"
            )
        if not output_path.is_file():
            raise QwenNunchakuUnavailable(f"Qwen Nunchaku sidecar did not create output: {output_path}")

        with Image.open(output_path) as image:
            output = image.convert("RGB").copy()
        return output, output_path

    @staticmethod
    def suggested_seed(request: GenerationRequest, *, batch_index: int, image_index: int) -> int:
        if request.seed >= 0 and batch_index == 0 and image_index == 0:
            return int(request.seed)
        return random.randint(0, 2**32 - 1)
