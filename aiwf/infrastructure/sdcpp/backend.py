from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
from uuid import uuid4

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.errors import GenerationCancelledError, ModelNotFoundError
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult
from aiwf.core.domain.models import Checkpoint, EmbeddingInfo, LoraInfo, SamplerInfo, VaeInfo
from aiwf.core.infotext import format_infotext
from aiwf.core.interfaces.backend import ProgressCallback
from aiwf.infrastructure.diffusers.checkpoints import scan_from_flags
from aiwf.infrastructure.diffusers.embeddings import scan_embeddings
from aiwf.infrastructure.diffusers.loras import scan_loras
from aiwf.infrastructure.diffusers.vae import scan_vaes

logger = logging.getLogger(__name__)


_SDCPP_SAMPLERS = [
    SamplerInfo(id="euler", label="Euler", family="sdcpp"),
    SamplerInfo(id="euler_a", label="Euler a", family="sdcpp"),
    SamplerInfo(id="heun", label="Heun", family="sdcpp"),
    SamplerInfo(id="dpm2", label="DPM2", family="sdcpp"),
    SamplerInfo(id="dpmpp_2m", label="DPM++ 2M", family="sdcpp", supports_karras=True),
    SamplerInfo(id="dpmpp_2m_karras", label="DPM++ 2M Karras", family="sdcpp", supports_karras=True),
    SamplerInfo(id="lcm", label="LCM", family="sdcpp"),
    SamplerInfo(id="tcd", label="TCD", family="sdcpp"),
]

_SAMPLER_TO_SDCPP = {
    "euler": "euler",
    "euler_a": "euler_a",
    "heun": "heun",
    "dpm2": "dpm2",
    "dpmpp_2m": "dpm++2m",
    "dpmpp_2m_karras": "dpm++2m",
    "lcm": "lcm",
    "tcd": "tcd",
    "ddim": "ddim_trailing",
}

_SCHEDULER_TO_SDCPP = {
    "automatic": None,
    "uniform": "discrete",
    "karras": "karras",
    "exponential": "exponential",
    "sgm_uniform": "sgm_uniform",
    "beta": "beta",
}

_UPSCALER_TO_SDCPP = {
    "lanczos": "Lanczos",
    "nearest": "Nearest",
    "bicubic": "Latent (bicubic)",
}

_STEP_PATTERNS = (
    re.compile(r"(?:step|sampling)\D+(\d+)\D+(\d+)", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*/\s*(\d+)\b"),
)


def _env_value(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or "").strip()


def _path_text(path: str | Path | None) -> str:
    return str(Path(path).expanduser().resolve()) if path else ""


class StableDiffusionCppBackend:
    """Subprocess bridge to stable-diffusion.cpp's sd-cli.

    This intentionally starts as a thin CLI adapter instead of binding the C API.
    It lets AIWF keep the production UI, queue, logs, model scan, metadata, and
    output handling while letting stable-diffusion.cpp own the inference runtime.
    Once the route is proven, the same interface can be replaced with an embedded
    library binding without touching the UI.
    """

    def __init__(self, flags: RuntimeFlags) -> None:
        self.flags = flags
        self._checkpoints: list[Checkpoint] | None = None
        self._loras: list[LoraInfo] | None = None
        self._vaes: list[VaeInfo] | None = None
        self._embeddings: list[EmbeddingInfo] | None = None
        self._loaded_checkpoint_id: str | None = None
        self._process_lock = threading.Lock()
        self._executable_cache: str | None = None

    @property
    def executable(self) -> str:
        return self._find_executable() or "<missing sd-cli>"

    def _find_executable(self) -> str | None:
        if self._executable_cache:
            return self._executable_cache

        explicit = _env_value("AIWF_SDCPP_BINARY")
        if explicit:
            candidate = Path(explicit).expanduser()
            if candidate.is_file():
                self._executable_cache = str(candidate.resolve())
                return self._executable_cache

        tools_root = self.flags.data_dir / "tools" / "stable-diffusion.cpp"
        candidates = [
            tools_root / "bin" / "sd-cli.exe",
            tools_root / "bin" / "sd-cli",
            tools_root / "build" / "bin" / "Release" / "sd-cli.exe",
            tools_root / "build" / "bin" / "sd-cli.exe",
            tools_root / "build" / "bin" / "sd-cli",
        ]
        for candidate in candidates:
            if candidate.is_file():
                self._executable_cache = str(candidate.resolve())
                return self._executable_cache

        for name in ("sd-cli.exe", "sd-cli", "sd.exe", "sd"):
            found = shutil.which(name)
            if found:
                self._executable_cache = found
                return self._executable_cache
        return None

    def _require_executable(self) -> str:
        executable = self._find_executable()
        if executable:
            return executable
        raise RuntimeError(
            "stable-diffusion.cpp sd-cli was not found. Set AIWF_SDCPP_BINARY to sd-cli.exe "
            "or place stable-diffusion.cpp under tools/stable-diffusion.cpp/bin/."
        )

    def invalidate_checkpoints(self) -> None:
        self._checkpoints = None

    def invalidate_loras(self) -> None:
        self._loras = None

    def invalidate_vaes(self) -> None:
        self._vaes = None

    def invalidate_embeddings(self) -> None:
        self._embeddings = None

    def list_checkpoints(self) -> list[Checkpoint]:
        if self._checkpoints is None:
            self._checkpoints = scan_from_flags(self.flags)
        return list(self._checkpoints)

    def list_samplers(self) -> list[SamplerInfo]:
        return list(_SDCPP_SAMPLERS)

    def list_loras(self) -> list[LoraInfo]:
        if self._loras is None:
            self._loras = scan_loras(self.flags)
        return list(self._loras)

    def list_vaes(self) -> list[VaeInfo]:
        if self._vaes is None:
            self._vaes = scan_vaes(self.flags)
        return list(self._vaes)

    def list_embeddings(self) -> list[EmbeddingInfo]:
        if self._embeddings is None:
            self._embeddings = scan_embeddings(self.flags)
        return list(self._embeddings)

    def resolve_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise ModelNotFoundError("No selectable models found for stable-diffusion.cpp")

        candidates = []
        if checkpoint_id:
            candidates.append(str(checkpoint_id))
        if self._loaded_checkpoint_id:
            candidates.append(self._loaded_checkpoint_id)
        if self.flags.default_checkpoint:
            candidates.append(str(self.flags.default_checkpoint))

        for candidate in candidates:
            lowered = candidate.lower()
            for item in checkpoints:
                fields = (
                    item.id,
                    item.title,
                    item.filename,
                    item.path,
                    Path(item.path).name,
                    Path(item.path).stem,
                )
                if any(str(field).lower() == lowered for field in fields):
                    return item
        return checkpoints[0]

    def load_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        checkpoint = self.resolve_checkpoint(checkpoint_id)
        self._loaded_checkpoint_id = checkpoint.id
        return checkpoint

    def is_checkpoint_warm(self, checkpoint_id: str | None = None) -> bool:
        return bool(checkpoint_id and checkpoint_id == self._loaded_checkpoint_id)

    def unload(self) -> None:
        self._loaded_checkpoint_id = None

    def _resolve_vae_path(self, request: GenerationRequest) -> str | None:
        if not request.vae_id:
            return _path_text(self.flags.vae_path) or None
        lowered = request.vae_id.lower()
        for vae in self.list_vaes():
            if lowered in {vae.id.lower(), vae.title.lower(), vae.filename.lower(), vae.path.lower()}:
                return vae.path
        return _path_text(self.flags.vae_path) or None

    def _save_input_image(self, image: Image.Image, directory: Path, name: str) -> str:
        path = directory / name
        image.convert("RGB").save(path)
        return str(path)

    def _save_mask_image(self, image: Image.Image, directory: Path, name: str) -> str:
        path = directory / name
        image.convert("L").save(path)
        return str(path)

    def _build_command(
        self,
        *,
        executable: str,
        checkpoint: Checkpoint,
        request: GenerationRequest,
        output_path: Path,
        work_dir: Path,
        init_images: list[Image.Image] | None,
        mask_images: list[Image.Image] | None,
        control_images: list[Image.Image] | None,
        preview_path: Path | None,
        preview_every_n_steps: int,
    ) -> list[str]:
        seed = int(request.seed if request.seed >= 0 else -1)
        sampler = _SAMPLER_TO_SDCPP.get(request.sampler, _SAMPLER_TO_SDCPP["euler_a"])
        scheduler = _SCHEDULER_TO_SDCPP.get((request.scheduler or "automatic").lower())

        cmd = [
            executable,
            "-m",
            checkpoint.path,
            "-p",
            request.prompt,
            "-n",
            request.negative_prompt or "",
            "-o",
            str(output_path),
            "-W",
            str(int(request.width)),
            "-H",
            str(int(request.height)),
            "--steps",
            str(int(request.steps)),
            "--cfg-scale",
            str(float(request.cfg_scale)),
            "-s",
            str(seed),
            "--sampling-method",
            sampler,
        ]

        image_total = max(1, int(request.batch_size) * int(request.batch_count))
        if image_total > 1:
            cmd.extend(["-b", str(image_total)])
        if scheduler:
            cmd.extend(["--scheduler", scheduler])
        if request.clip_skip > 0:
            cmd.extend(["--clip-skip", str(int(request.clip_skip))])

        vae_path = self._resolve_vae_path(request)
        if vae_path:
            cmd.extend(["--vae", vae_path])

        if request.mode in (GenerationMode.IMG2IMG, GenerationMode.INPAINT) and init_images:
            cmd.extend(["-i", self._save_input_image(init_images[0], work_dir, "init.png")])
            cmd.extend(["--strength", str(float(request.denoising_strength))])
        if request.mode == GenerationMode.INPAINT and mask_images:
            cmd.extend(["--mask", self._save_mask_image(mask_images[0], work_dir, "mask.png")])
        if control_images:
            cmd.extend(["--control-image", self._save_input_image(control_images[0], work_dir, "control.png")])

        if request.enable_hr and request.mode == GenerationMode.TXT2IMG:
            upscaler = _UPSCALER_TO_SDCPP.get(request.hr_upscaler, "Lanczos")
            cmd.extend(
                [
                    "--hires",
                    "--hires-scale",
                    str(float(request.hr_scale)),
                    "--hires-steps",
                    str(int(request.hr_steps)),
                    "--hires-denoising-strength",
                    str(float(request.hr_denoising_strength)),
                    "--hires-upscaler",
                    upscaler,
                ]
            )

        if preview_path is not None and preview_every_n_steps > 0:
            cmd.extend(
                [
                    "--preview",
                    _env_value("AIWF_SDCPP_PREVIEW", "vae") or "vae",
                    "--preview-path",
                    str(preview_path),
                    "--preview-interval",
                    str(max(1, int(preview_every_n_steps))),
                ]
            )

        backend = _env_value("AIWF_SDCPP_BACKEND", "cuda0" if not self.flags.cpu else "cpu")
        params_backend = _env_value("AIWF_SDCPP_PARAMS_BACKEND")
        max_vram = _env_value("AIWF_SDCPP_MAX_VRAM", "0")
        if backend:
            cmd.extend(["--backend", backend])
        if params_backend:
            cmd.extend(["--params-backend", params_backend])
        if max_vram:
            cmd.extend(["--max-vram", max_vram])
        if _env_value("AIWF_SDCPP_OFFLOAD_TO_CPU") in {"1", "true", "yes", "on"}:
            cmd.append("--offload-to-cpu")
        if _env_value("AIWF_SDCPP_STREAM_LAYERS") in {"1", "true", "yes", "on"}:
            cmd.append("--stream-layers")
        if _env_value("AIWF_SDCPP_DIFFUSION_FA", "1") not in {"0", "false", "no", "off"}:
            cmd.append("--diffusion-fa")
        if _env_value("AIWF_SDCPP_VAE_TILING") in {"1", "true", "yes", "on"}:
            cmd.append("--vae-tiling")
        if _env_value("AIWF_SDCPP_MMAP", "1") not in {"0", "false", "no", "off"}:
            cmd.append("--mmap")

        extra = _env_value("AIWF_SDCPP_EXTRA_ARGS")
        if extra:
            cmd.extend(shlex.split(extra))
        return cmd

    @staticmethod
    def _parse_step(line: str) -> tuple[int, int] | None:
        for pattern in _STEP_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            try:
                step = int(match.group(1))
                total = int(match.group(2))
            except ValueError:
                continue
            if total > 0 and 0 <= step <= total:
                return step, total
        return None

    def _collect_outputs(self, output_path: Path) -> list[Path]:
        if "%" in output_path.name:
            pattern = re.sub(r"%0?\d*d", "*", output_path.name)
            return sorted(output_path.parent.glob(pattern))
        return [output_path] if output_path.is_file() else sorted(output_path.parent.glob("output*.png"))

    def generate(
        self,
        request: GenerationRequest,
        init_images: list[Image.Image] | None = None,
        mask_images: list[Image.Image] | None = None,
        control_images: list[Image.Image] | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        preview_every_n_steps: int = 0,
    ) -> GenerationResult:
        executable = self._require_executable()
        checkpoint = self.resolve_checkpoint(request.checkpoint_id)
        self._loaded_checkpoint_id = checkpoint.id

        image_total = max(1, int(request.batch_size) * int(request.batch_count))
        output_template = "output_%03d.png" if image_total > 1 else "output.png"
        total_steps = max(1, int(request.steps))
        if request.enable_hr and request.mode == GenerationMode.TXT2IMG:
            total_steps += max(1, int(request.hr_steps))

        output_root = self.flags.resolved_output_dir() / "_sdcpp_tmp"
        output_root.mkdir(parents=True, exist_ok=True)

        with self._process_lock, tempfile.TemporaryDirectory(prefix="job-", dir=output_root) as temp_name:
            work_dir = Path(temp_name)
            output_path = work_dir / output_template
            preview_path = work_dir / "preview.png" if preview_every_n_steps > 0 else None
            cmd = self._build_command(
                executable=executable,
                checkpoint=checkpoint,
                request=request,
                output_path=output_path,
                work_dir=work_dir,
                init_images=init_images,
                mask_images=mask_images,
                control_images=control_images,
                preview_path=preview_path,
                preview_every_n_steps=preview_every_n_steps,
            )

            if on_progress is not None:
                on_progress(0, total_steps, "Launching stable-diffusion.cpp", None, None, None)

            logger.info("Launching stable-diffusion.cpp: %s", " ".join(shlex.quote(part) for part in cmd))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            lines: list[str] = []
            last_step = 0
            last_preview_mtime = 0.0
            reader_done = threading.Event()

            def reader() -> None:
                nonlocal last_step, total_steps
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue
                    lines.append(line)
                    del lines[:-80]
                    logger.info("stable-diffusion.cpp: %s", line)
                    parsed = self._parse_step(line)
                    if parsed is None:
                        continue
                    last_step, parsed_total = parsed
                    total_steps = max(total_steps, parsed_total)
                    if on_progress is not None:
                        on_progress(last_step, total_steps, line[-160:], None, None, None)
                reader_done.set()

            threading.Thread(target=reader, daemon=True, name="sdcpp-output-reader").start()

            while process.poll() is None:
                if should_cancel is not None and should_cancel():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise GenerationCancelledError("stable-diffusion.cpp generation cancelled")
                if preview_path is not None and preview_path.is_file():
                    try:
                        mtime = preview_path.stat().st_mtime
                        if mtime > last_preview_mtime:
                            last_preview_mtime = mtime
                            with Image.open(preview_path) as preview_img:
                                preview = preview_img.convert("RGB").copy()
                            if on_progress is not None:
                                step = min(total_steps, max(last_step + 1, 1))
                                on_progress(step, total_steps, "stable-diffusion.cpp preview", preview, None, None)
                    except OSError:
                        pass
                time.sleep(0.2)

            reader_done.wait(timeout=1.0)
            if process.returncode != 0:
                tail = "\n".join(lines[-20:])
                raise RuntimeError(f"stable-diffusion.cpp failed with exit code {process.returncode}\n{tail}")

            output_files = self._collect_outputs(output_path)
            if not output_files:
                tail = "\n".join(lines[-20:])
                raise RuntimeError(f"stable-diffusion.cpp completed but produced no output image\n{tail}")

            images: list[Image.Image] = []
            seeds: list[int] = []
            infotexts: list[str] = []
            base_seed = int(request.seed if request.seed >= 0 else 0)
            for index, path in enumerate(output_files):
                with Image.open(path) as image:
                    img = image.convert("RGB").copy()
                images.append(img)
                seed = base_seed + index if base_seed >= 0 else base_seed
                seeds.append(seed)
                infotexts.append(
                    format_infotext(
                        request,
                        seed,
                        checkpoint,
                        output_width=img.width,
                        output_height=img.height,
                    )
                )

            if on_progress is not None:
                on_progress(total_steps, total_steps, "stable-diffusion.cpp complete", images[0], images, seeds)

            return GenerationResult(
                job_id=uuid4(),
                images=images,
                seeds=seeds,
                infotexts=infotexts,
                mode=request.mode,
            )
