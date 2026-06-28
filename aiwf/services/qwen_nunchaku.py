from __future__ import annotations

import logging
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.generation import GenerationRequest
from aiwf.core.domain.models import Checkpoint
from aiwf.infrastructure.diffusers.model_arch import is_qwen_nunchaku_architecture

logger = logging.getLogger(__name__)

_DEFAULT_TRANSFORMER_NAME = "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"


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
        for candidate in (self.models_base_dir(), self.downloads_base_dir()):
            if (candidate / "model_index.json").is_file():
                return candidate
        return self.models_base_dir()

    def models_root(self) -> Path:
        return self.flags.resolved_models_dir() / "qwen-image" / "Nunchaku"

    def output_dir(self) -> Path:
        return self.flags.resolved_output_dir() / "qwen-nunchaku"

    def default_transformer_path(self) -> Path:
        model_preferred = self.models_root() / _DEFAULT_TRANSFORMER_NAME
        if model_preferred.is_file():
            return model_preferred
        download_preferred = (
            self.flags.data_dir.resolve()
            / "downloads"
            / "qwen_nunchaku"
            / "transformer"
            / _DEFAULT_TRANSFORMER_NAME
        )
        if download_preferred.is_file():
            return download_preferred
        candidates = sorted(self.models_root().glob("*.safetensors"), key=lambda path: path.name.lower())
        return candidates[0] if candidates else model_preferred

    def status(self, transformer_path: str | Path | None = None) -> QwenNunchakuStatus:
        transformer = Path(transformer_path).resolve() if transformer_path else self.default_transformer_path()
        python_exe = self.python_exe()
        runner_script = self.runner_script()
        base_dir = self.base_dir()
        messages: list[str] = []
        if not python_exe.is_file():
            messages.append(f"engine runtime missing: {python_exe}")
        if not runner_script.is_file():
            messages.append(f"runner missing: {runner_script}")
        if not base_dir.is_dir():
            messages.append(f"base components missing: {base_dir}")
        elif not (base_dir / "model_index.json").is_file():
            messages.append(f"base components missing model_index.json: {base_dir}")
        if not transformer.is_file():
            messages.append(f"transformer missing: {transformer}")
        return QwenNunchakuStatus(
            ready=not messages,
            python_exe=python_exe,
            runner_script=runner_script,
            base_dir=base_dir,
            transformer_path=transformer,
            messages=tuple(messages),
        )

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
