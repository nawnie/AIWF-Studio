"""AIWF-owned Python and stable-diffusion.cpp image jobs for the native app.

Discovery is metadata-only and stays Torch-free. The AIWF Diffusers context is
created lazily when a Python job starts; C++ jobs use an owned sd-cli process.
"""

from __future__ import annotations

import math
import hashlib
import importlib.metadata
import importlib.util
import logging
import os
import random
import re
import struct
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from aiwf.core.domain.errors import GenerationCancelledError, ModelNotFoundError
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.services.image_contract import (
    ASPECT_RATIOS,
    MAX_IMAGE_BYTES,
    MAX_JOBS_KEPT,
    MAX_PROMPT_CHARS,
    MAX_SEED,
    QUALITY_MEGAPIXELS,
)
from aiwf.services.unified_bridge import BridgeError, SCHEMA_VERSION, _utc_now

logger = logging.getLogger(__name__)

CPP_BACKEND_ID = "cpp"
CPP_BACKEND_LABEL = "AIWF C++ (stable-diffusion.cpp)"
PYTHON_BACKEND_ID = "python"
PYTHON_BACKEND_LABEL = "AIWF Python (Diffusers)"
PYTHON_JOB_ID_PREFIX = "imgpy-"
PYTHON_JOB_ID_SUFFIX_LENGTH = 12
_PYTHON_JOB_ID = re.compile(r"^imgpy-[0-9a-f]{12}$")
PYTHON_MODEL_ARCHITECTURES = {
    "sd15", "inpaint", "sdxl", "sdxl_inpaint", "sd35", "flux", "flux_kontext",
    "flux2_klein", "z_image", "krea2", "anima", "qwen_image", "qwen_image_nunchaku", "sana",
}
PYTHON_UNSUPPORTED_ARCHITECTURES = {
    "flux_fill", "flux2", "longcat_image", "qwen_image_edit", "qwen_image_edit_plus",
    "sdxl_refiner", "sana_video", "unknown",
}
CPP_FAMILIES = {"sd15", "sdxl"}
CPP_CHECKPOINT_EXTENSIONS = {".ckpt", ".pt", ".pth", ".safetensors", ".gguf"}
CPP_REQUIRED_FLAGS = (
    "--model",
    "--output",
    "--width",
    "--height",
    "--steps",
    "--cfg-scale",
    "--sampling-method",
    "--backend",
)
CPP_JOB_ID_PREFIX = "imgcpp-"
CPP_JOB_ID_SUFFIX_LENGTH = 12
_CPP_JOB_ID = re.compile(r"^imgcpp-[0-9a-f]{12}$")
_PYTHON_REQUIRED_DISTRIBUTIONS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "safetensors": "safetensors",
}
_PYTHON_CUDA13_DISTRIBUTIONS = {
    "torch": "2.13.0+cu132",
    "torchvision": "0.28.0+cu132",
}


def _opaque_model_id(backend_id: str, checkpoint: Any) -> str:
    resolved = Path(str(checkpoint.path)).expanduser().resolve()
    key = os.path.normcase(str(resolved)).casefold().encode("utf-8", errors="replace")
    digest = hashlib.sha256(key).hexdigest()[:24]
    return f"model-{backend_id}-{digest}"


def _checkpoint_model(checkpoint: Any, backend_id: str) -> dict[str, Any]:
    family = str(getattr(checkpoint, "architecture", "unknown") or "unknown").casefold()
    from aiwf.infrastructure.diffusers.model_arch import architecture_label

    return {
        "id": _opaque_model_id(backend_id, checkpoint),
        "label": str(getattr(checkpoint, "title", "") or getattr(checkpoint, "filename", "") or "Model"),
        "family": family,
        "family_label": architecture_label(family),
        "capabilities": ["text_to_image"],
    }


def _aiwf_image_checkpoints(flags: Any, backend_id: str) -> list[tuple[dict[str, Any], Any]]:
    """Use AIWF's configured image catalog without importing Torch or loading weights."""
    from aiwf.infrastructure.diffusers.checkpoints import scan_from_flags

    try:
        checkpoints = scan_from_flags(flags, allow_torch_load=False)
    except TypeError as exc:
        # An older source tree must fail closed instead of falling back to a
        # scanner that may torch.load a small .ckpt just to classify it.
        raise RuntimeError("AIWF model discovery does not support Torch-free scanning.") from exc
    rows: list[tuple[dict[str, Any], Any]] = []
    for checkpoint in checkpoints:
        family = str(getattr(checkpoint, "architecture", "unknown") or "unknown").casefold()
        if family not in PYTHON_MODEL_ARCHITECTURES or family in PYTHON_UNSUPPORTED_ARCHITECTURES:
            continue
        path = Path(str(getattr(checkpoint, "path", "")))
        if not path.exists():
            continue
        # This Python SD 1.5 / SDXL single-file route does not load GGUF.
        # Keep the existing separately supported Flux GGUF route intact.
        if path.is_file() and path.suffix.casefold() == ".gguf" and family in {"sd15", "sdxl"}:
            continue
        if family == "qwen_image_nunchaku" and importlib.util.find_spec("nunchaku") is None:
            continue
        rows.append((_checkpoint_model(checkpoint, backend_id), checkpoint))
    return rows


def _python_dependency_status() -> tuple[bool, dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module, distribution in _PYTHON_REQUIRED_DISTRIBUTIONS.items():
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(distribution)
            continue
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    for distribution, expected_version in _PYTHON_CUDA13_DISTRIBUTIONS.items():
        installed_version = versions.get(distribution)
        if installed_version is not None and installed_version != expected_version:
            missing.append(f"{distribution}=={expected_version} (found {installed_version})")
    transformer_version = versions.get("transformers", "")
    match = re.match(r"^(\d+)\.(\d+)", transformer_version)
    if match and int(match.group(1)) >= 5:
        missing.append("transformers>=4.44,<5 (installed version is unsupported)")
    return not missing, versions, missing


def _pe_imported_dlls(path: Path) -> set[str]:
    """Read a PE import table without loading or executing the image."""
    with path.open("rb") as stream:
        dos = stream.read(64)
        if len(dos) < 64 or dos[:2] != b"MZ":
            raise ValueError("not a Windows PE executable")
        pe_offset = struct.unpack_from("<I", dos, 0x3C)[0]
        stream.seek(pe_offset)
        if stream.read(4) != b"PE\0\0":
            raise ValueError("invalid PE signature")
        coff = stream.read(20)
        if len(coff) != 20:
            raise ValueError("truncated PE header")
        section_count = struct.unpack_from("<H", coff, 2)[0]
        optional_size = struct.unpack_from("<H", coff, 16)[0]
        optional_offset = pe_offset + 24
        stream.seek(optional_offset)
        optional = stream.read(optional_size)
        if len(optional) != optional_size:
            raise ValueError("truncated PE optional header")
        magic = struct.unpack_from("<H", optional, 0)[0]
        directories_offset = 112 if magic == 0x20B else 96 if magic == 0x10B else None
        if directories_offset is None or len(optional) < directories_offset + 16:
            raise ValueError("unsupported PE optional header")
        import_rva, import_size = struct.unpack_from("<II", optional, directories_offset + 8)
        if not import_rva or not import_size:
            return set()

        section_offset = optional_offset + optional_size
        stream.seek(section_offset)
        sections = []
        for _ in range(section_count):
            section = stream.read(40)
            if len(section) != 40:
                raise ValueError("truncated PE section table")
            virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from("<IIII", section, 8)
            sections.append((virtual_address, max(virtual_size, raw_size), raw_size, raw_pointer))

        def offset_for_rva(rva: int) -> int:
            for virtual_address, span, raw_size, raw_pointer in sections:
                delta = rva - virtual_address
                if 0 <= delta < span and delta < raw_size:
                    return raw_pointer + delta
            raise ValueError("PE import points outside file-backed sections")

        descriptor_offset = offset_for_rva(import_rva)
        imports: set[str] = set()
        for index in range(1024):
            stream.seek(descriptor_offset + index * 20)
            descriptor = stream.read(20)
            if len(descriptor) != 20:
                raise ValueError("truncated PE import table")
            name_rva = struct.unpack_from("<I", descriptor, 12)[0]
            if not name_rva:
                return imports
            stream.seek(offset_for_rva(name_rva))
            raw_name = stream.read(260).split(b"\0", 1)[0]
            if not raw_name:
                raise ValueError("empty PE import name")
            imports.add(raw_name.decode("ascii", errors="strict").casefold())
        raise ValueError("PE import table is too long")


def _make_native_cpp_backend(flags: Any) -> Any:
    """Create a lightweight sd.cpp adapter with a CUDA 13-only resolver."""
    from aiwf.infrastructure.sdcpp.backend import StableDiffusionCppBackend

    class NativeCuda13Backend(StableDiffusionCppBackend):
        def __init__(self, runtime_flags: Any) -> None:
            # The adapter does not use DeviceManager; supplying a sentinel avoids
            # importing Torch just to enumerate models or inspect the CLI.
            super().__init__(runtime_flags, devices=object())
            self._native_resolver_error: str | None = None
            self._native_probe: tuple[bool, str] | None = None

        @property
        def resolver_error(self) -> str | None:
            self._find_executable()
            return self._native_resolver_error

        def _find_executable(self) -> str | None:
            if self._executable_cache:
                return self._executable_cache

            configured = os.environ.get("AIWF_SDCPP_BINARY", "").strip()
            if configured:
                candidate = Path(configured).expanduser()
                if not candidate.is_absolute():
                    self._native_resolver_error = "AIWF_SDCPP_BINARY must be an absolute path to the CUDA 13 sd-cli.exe."
                    return None
                candidates = [candidate]
            else:
                root = Path(self.flags.data_dir).expanduser()
                candidates = [
                    root / "dependencies" / "stable-diffusion.cpp-cuda13" / "bin" / "sd-cli.exe",
                    # This repo's installer and local build guide own this
                    # checkout path. The import/probe checks below still gate
                    # it to the CUDA 13 executable before offering C++.
                    root / "tools" / "stable-diffusion.cpp" / "build" / "bin" / "Release" / "sd-cli.exe",
                ]

            for candidate in candidates:
                try:
                    resolved = candidate.resolve(strict=True)
                    if not resolved.is_file() or resolved.suffix.casefold() != ".exe":
                        if configured:
                            self._native_resolver_error = "AIWF_SDCPP_BINARY must point to an existing CUDA 13 sd-cli.exe."
                        continue
                    imports = _pe_imported_dlls(resolved)
                except FileNotFoundError:
                    if configured:
                        self._native_resolver_error = "AIWF_SDCPP_BINARY does not point to an existing CUDA 13 sd-cli.exe."
                    continue
                except (OSError, ValueError, UnicodeError) as exc:
                    self._native_resolver_error = f"The stable-diffusion.cpp executable could not be inspected ({type(exc).__name__})."
                    continue
                if "cublas64_13.dll" not in imports:
                    self._native_resolver_error = "The stable-diffusion.cpp executable does not import cublas64_13.dll, so it is not the verified CUDA 13 build."
                    continue
                self._executable_cache = str(resolved)
                self._native_resolver_error = None
                return self._executable_cache

            if not self._native_resolver_error:
                self._native_resolver_error = "No CUDA 13 stable-diffusion.cpp executable was found. Configure AIWF_SDCPP_BINARY or install the CUDA 13 package."
            return None

        def probe_cli(self) -> tuple[bool, str]:
            if self._native_probe is not None:
                return self._native_probe
            executable = self._find_executable()
            if not executable:
                self._native_probe = (False, self._native_resolver_error or "CUDA 13 sd-cli is unavailable.")
                return self._native_probe

            child_env = os.environ.copy()
            current_path = child_env.get("PATH", "")
            child_env["PATH"] = str(Path(executable).parent) + (os.pathsep + current_path if current_path else "")
            kwargs: dict[str, Any] = {
                "args": [executable, "--help"],
                "cwd": str(Path(executable).parent),
                "env": child_env,
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": 10,
                "check": False,
            }
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                kwargs["startupinfo"] = startupinfo
            try:
                result = subprocess.run(**kwargs)
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._native_probe = (False, f"CUDA 13 sd-cli dependency/help probe failed ({type(exc).__name__}).")
                return self._native_probe
            help_text = result.stdout + "\n" + result.stderr
            missing = [flag for flag in CPP_REQUIRED_FLAGS if flag not in help_text]
            if result.returncode != 0:
                reason = f"CUDA 13 sd-cli --help exited with code {result.returncode}."
                self._native_probe = (False, reason)
            elif missing:
                self._native_probe = (False, "CUDA 13 sd-cli is missing required options: " + ", ".join(missing) + ".")
            else:
                self._native_probe = (True, "CUDA 13 sd-cli dependencies and text-to-image options passed the headless probe.")
            return self._native_probe

    return NativeCuda13Backend(flags)


def _supported_checkpoints(backend: Any) -> list[Any]:
    from aiwf.infrastructure.diffusers.model_arch import gguf_has_shallow_header
    invalidate = getattr(backend, "invalidate_checkpoints", None)
    if callable(invalidate):
        invalidate()
    checkpoints = backend.list_checkpoints()
    return [
        checkpoint
        for checkpoint in checkpoints
        if str(getattr(checkpoint, "architecture", "")).casefold() in CPP_FAMILIES
        and str(getattr(checkpoint, "kind", "checkpoint")).casefold() == "checkpoint"
        and Path(str(getattr(checkpoint, "path", ""))).is_file()
        and Path(str(getattr(checkpoint, "path", ""))).suffix.casefold() in CPP_CHECKPOINT_EXTENSIONS
        and (Path(str(getattr(checkpoint, "path", ""))).suffix.casefold() != ".gguf"
             or gguf_has_shallow_header(Path(str(getattr(checkpoint, "path", "")))))
    ]


def _image_dimensions(aspect_ratio: str, quality: str) -> tuple[int, int]:
    ratio_w, ratio_h = (int(part) for part in aspect_ratio.split(":"))
    ratio = ratio_w / ratio_h
    pixels = QUALITY_MEGAPIXELS[quality] * 1_000_000
    width = max(64, int(math.sqrt(pixels * ratio) / 8 + 0.5) * 8)
    height = max(64, int(math.sqrt(pixels / ratio) / 8 + 0.5) * 8)
    scale = min(1.0, 2048 / max(width, height))
    if scale < 1.0:
        width = max(64, int(width * scale / 8) * 8)
        height = max(64, int(height * scale / 8) * 8)
    return width, height


class AiwfPythonImageJobs:
    """Native job facade over AIWF's own Diffusers GenerationService."""

    def __init__(
        self,
        *,
        flags: Any | None,
        on_done: Callable[[dict[str, Any]], None] | None = None,
        context_factory: Callable[[Any], Any] | None = None,
        model_records_factory: Callable[[Any, str], list[tuple[dict[str, Any], Any]]] | None = None,
        dependency_status_factory: Callable[[], tuple[bool, dict[str, str], list[str]]] | None = None,
        poll_seconds: float = 0.1,
    ) -> None:
        self._flags = flags
        self._on_done = on_done
        self._context_factory = context_factory or self._build_context
        self._model_records_factory = model_records_factory or _aiwf_image_checkpoints
        self._dependency_status_factory = dependency_status_factory or _python_dependency_status
        self._poll_seconds = max(0.02, poll_seconds)
        self._context: Any | None = None
        self._context_lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _build_context(flags: Any) -> Any:
        from aiwf.bootstrap import build_context

        python_flags = flags.model_copy(update={"inference_backend": "diffusers"})
        return build_context(python_flags)

    def _get_context(self) -> Any:
        if self._flags is None:
            raise BridgeError(503, "python_backend_unavailable", "The engine API did not receive AIWF model-folder settings for the Python backend.")
        with self._context_lock:
            if self._context is None:
                try:
                    self._context = self._context_factory(self._flags)
                except Exception as exc:  # noqa: BLE001 - show actionable selected-engine setup state
                    raise BridgeError(
                        503,
                        "python_backend_start_failed",
                        f"AIWF Python could not initialize in this Studio venv ({type(exc).__name__}). Close Studio, run .\\scripts\\provision_python_backend.ps1 -StudioRoot ., then restart.",
                    ) from exc
            return self._context

    def _model_records(self) -> list[tuple[dict[str, Any], Any]]:
        if self._flags is None:
            return []
        return self._model_records_factory(self._flags, PYTHON_BACKEND_ID)

    def readiness(self) -> dict[str, Any]:
        if self._flags is None:
            return {
                "available": False,
                "reason": "The engine API did not receive AIWF model-folder settings for the Python backend.",
                "models": [],
                "runtime_info": {},
            }
        dependency_ok, versions, missing = self._dependency_status_factory()
        try:
            records = self._model_records()
            models = [row for row, _checkpoint in records]
        except Exception as exc:  # noqa: BLE001 - local inventory errors are visible, not fallback routes
            return {
                "available": False,
                "reason": f"AIWF model discovery failed ({type(exc).__name__}); review the Setup model folders and retry.",
                "models": [],
                "runtime_info": {"engine": "AIWF Diffusers", "package_versions": versions},
            }

        runtime_info: dict[str, Any] = {"engine": "AIWF Diffusers", "package_versions": versions}
        if not dependency_ok:
            missing_text = ", ".join(missing)
            return {
                "available": False,
                "reason": (
                    "AIWF Python inference packages are missing or unsupported in this Studio venv: "
                    f"{missing_text}. Close Studio, run .\\scripts\\provision_python_backend.ps1 -StudioRoot ., then restart."
                ),
                "models": models,
                "runtime_info": runtime_info,
            }
        if not models:
            return {
                "available": False,
                "reason": "AIWF Python packages are installed, but Setup's model folders contain no supported text-to-image model. Add an AIWF-recognized Diffusers image model or SD checkpoint, then rescan.",
                "models": [],
                "runtime_info": runtime_info,
            }
        return {
            "available": True,
            "reason": (
                f"AIWF Python packages and {len(models)} local text-to-image model(s) were found. "
                "Discovery checks package metadata and model files only; it does not load weights or verify generation."
            ),
            "models": models,
            "runtime_info": runtime_info,
        }

    def model_rows(self) -> list[dict[str, Any]]:
        return [row for row, _checkpoint in self._model_records()]

    def unload_for_backend_switch(self) -> None:
        """Release AIWF's retained Diffusers pipelines before a C++ CUDA job."""
        with self._lock:
            active = [
                job_id for job_id, job in self._jobs.items()
                if job["state"] not in {"done", "failed", "cancelled"}
            ]
        if active:
            raise BridgeError(409, "python_job_active", "Wait for the active AIWF Python image job to finish before switching to C++.")

        with self._context_lock:
            context = self._context
            if context is None:
                return
            generation = getattr(context, "generation", None)
            backend = getattr(generation, "backend", None)
            unload = getattr(backend, "unload", None)
            if not callable(unload):
                raise BridgeError(
                    503,
                    "python_backend_cannot_unload",
                    "AIWF Python is holding a generation context that cannot safely release its image pipeline; C++ generation was not started.",
                )
            try:
                # This is the DiffusersBackend's owning cleanup path: it drops
                # pipeline references before running its CUDA synchronization/cache cleanup.
                unload()
            except Exception as exc:  # noqa: BLE001 - fail closed before starting another GPU backend
                raise BridgeError(
                    503,
                    "python_backend_unload_failed",
                    f"AIWF Python could not release its loaded model ({type(exc).__name__}); C++ generation was not started.",
                ) from exc

    def start(
        self,
        *,
        prompt: str,
        aspect_ratio: str,
        quality: str,
        seed: int | None,
        model_id: str,
        project_id: str | None,
        output_root: Path | None,
        on_finished: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        text = (prompt or "").strip()
        if not text or len(text) > MAX_PROMPT_CHARS:
            raise BridgeError(422, "invalid_prompt", f"Describe the image in 1-{MAX_PROMPT_CHARS} characters.")
        if aspect_ratio not in ASPECT_RATIOS:
            raise BridgeError(422, "invalid_aspect_ratio", "Aspect ratio must be one of " + ", ".join(ASPECT_RATIOS) + ".")
        if quality not in QUALITY_MEGAPIXELS:
            raise BridgeError(422, "invalid_quality", "Quality must be draft or standard.")
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise BridgeError(422, "invalid_seed", f"Seed must be a whole number from 0 to {MAX_SEED}.")
        if output_root is None or not Path(output_root).is_dir():
            raise BridgeError(503, "no_output_root", "Studio's output folder is not available, so the image would have nowhere to go.")

        try:
            records = self._model_records()
        except Exception as exc:  # noqa: BLE001 - inventory errors should be actionable
            raise BridgeError(503, "python_model_discovery_failed", f"AIWF model discovery failed ({type(exc).__name__}); review Setup's model folders and retry.") from exc
        selected = next(((row, checkpoint) for row, checkpoint in records if row["id"] == model_id), None)
        if selected is None:
            raise BridgeError(422, "model_not_supported", "The selected model ID is not in AIWF's current text-to-image model catalog.")

        readiness = self.readiness()
        if not readiness["available"]:
            raise BridgeError(503, "python_backend_unavailable", str(readiness["reason"]))
        row, checkpoint = selected
        width, height = _image_dimensions(aspect_ratio, quality)
        request = GenerationRequest(
            mode=GenerationMode.TXT2IMG,
            prompt=text,
            steps=20 if quality == "draft" else 30,
            cfg_scale=7.0,
            width=width,
            height=height,
            seed=seed,
            sampler="euler_a",
            checkpoint_id=str(checkpoint.id),
            pipeline_backend="native-python",
            native_model_id=model_id,
        )
        job_id = PYTHON_JOB_ID_PREFIX + uuid.uuid4().hex[:PYTHON_JOB_ID_SUFFIX_LENGTH]
        job = {
            "job_id": job_id,
            "backend_id": PYTHON_BACKEND_ID,
            "model_id": model_id,
            "model_family": str(checkpoint.architecture).casefold(),
            "model": str(checkpoint.title),
            "request": request,
            "checkpoint_id": str(checkpoint.id),
            "state": "queued",
            "prompt": text,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "seed": seed,
            "project_id": project_id,
            "output_root": Path(output_root).resolve(),
            "queue_position": None,
            "created_at": _utc_now(),
            "created_monotonic": time.monotonic(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "images": [],
            "progress": 0,
            "message": "Queued for AIWF Python.",
            "cancel_requested": False,
            "cancel_event": threading.Event(),
            "generation_job_id": None,
            "on_finished": on_finished,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()
        worker = threading.Thread(target=self._run_guarded, args=(job_id,), name=f"python-image-job-{job_id}", daemon=True)
        try:
            worker.start()
        except Exception as exc:
            with self._lock:
                self._jobs.pop(job_id, None)
            raise BridgeError(503, "python_job_start_failed", "The AIWF Python image job could not be started.") from exc
        return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}

    def status(self, job_id: str) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "job": self._view(self._get(job_id))}

    def file(self, job_id: str, index: int) -> Path:
        job = self._get(job_id)
        images = job["images"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(images):
            raise BridgeError(404, "image_not_found", "This job has no image with that number.")
        root = Path(job["output_root"]).resolve()
        path = (root / images[index]["relative_path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise BridgeError(404, "image_not_found", "The saved image is no longer in Studio's output folder.")
        return path

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        with self._lock:
            if job["state"] not in {"done", "failed", "cancelled"}:
                job["cancel_requested"] = True
                job["cancel_event"].set()
                generation_job_id = job.get("generation_job_id")
                context = self._context
            else:
                generation_job_id = None
                context = None
        if generation_job_id is not None and context is not None:
            context.generation.interrupt(generation_job_id)
        return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}

    def _run_guarded(self, job_id: str) -> None:
        try:
            self._run(job_id)
        except Exception as exc:  # noqa: BLE001 - failures become a visible terminal job state
            message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            self._finish(job_id, "failed", {"code": "python_generation_failed", "message": f"AIWF Python generation failed: {message[:400]}"})

    def _run(self, job_id: str) -> None:
        job = self._get(job_id)
        with self._lock:
            if job["state"] in {"done", "failed", "cancelled"}:
                return
            if job["cancel_event"].is_set():
                cancelled = True
            else:
                cancelled = False
                job["state"] = "running"
                job["started_at"] = _utc_now()
                job["message"] = "Starting AIWF Python."
        if cancelled:
            self._finish(job_id, "cancelled", None)
            return

        context = self._get_context()
        if job["cancel_event"].is_set():
            self._finish(job_id, "cancelled", None)
            return
        generation = context.generation
        monitor_done = threading.Event()

        def monitor() -> None:
            while not monitor_done.wait(self._poll_seconds):
                with self._lock:
                    inner_id = job.get("generation_job_id")
                    cancelled_requested = job["cancel_event"].is_set()
                if inner_id is None:
                    continue
                if cancelled_requested:
                    try:
                        generation.interrupt(inner_id)
                    except Exception:
                        pass
                try:
                    current = generation.get_job(inner_id)
                except Exception:
                    current = None
                progress = getattr(current, "progress", None)
                if progress is not None:
                    with self._lock:
                        job["progress"] = int(getattr(progress, "percent", 0) or 0)
                        job["message"] = str(getattr(progress, "message", "") or job["message"])

        monitor_thread = threading.Thread(target=monitor, name=f"python-image-status-{job_id}", daemon=True)
        monitor_thread.start()

        def on_job_created(record: Any) -> None:
            with self._lock:
                job["generation_job_id"] = record.id
                cancel_now = job["cancel_event"].is_set()
            if cancel_now:
                generation.interrupt(record.id)

        try:
            record = generation.submit(job["request"], on_job_created=on_job_created)
        finally:
            monitor_done.set()
            monitor_thread.join(timeout=1.0)
        state = str(getattr(getattr(record, "state", None), "value", getattr(record, "state", ""))).casefold()
        if state == "cancelled" or job["cancel_event"].is_set() and state != "completed":
            self._finish(job_id, "cancelled", None)
            return
        if state != "completed" or getattr(record, "result", None) is None:
            self._finish(job_id, "failed", {"code": "generation_failed", "message": str(getattr(record, "error", None) or "AIWF Python returned no completed generation result.")[:400]})
            return
        result = record.result
        images = list(getattr(result, "images", []) or [])
        if not images:
            self._finish(job_id, "failed", {"code": "no_image", "message": "AIWF Python completed without returning an image."})
            return
        try:
            saved = self._collect_saved_images(job, record, images)
        except (OSError, ValueError) as exc:
            self._finish(job_id, "failed", {"code": "save_failed", "message": f"AIWF Python completed but its output could not be verified ({type(exc).__name__})."})
            return
        with self._lock:
            job["images"] = saved
            job["progress"] = 100
            job["message"] = "Image generation complete."
        self._finish(job_id, "done", None)

    @staticmethod
    def _collect_saved_images(job: dict[str, Any], record: Any, images: list[Image.Image]) -> list[dict[str, Any]]:
        root = Path(job["output_root"]).resolve()
        artifacts = list(getattr(getattr(record, "result", None), "artifacts", []) or [])
        artifact_paths = [Path(str(getattr(item, "path", ""))).resolve() for item in artifacts[: len(images)]]
        saved: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            path = artifact_paths[index] if index < len(artifact_paths) else None
            if path is None or not path.is_file():
                path = AiwfPythonImageJobs._save_image(job, index, image)
            if root not in path.parents or path.suffix.casefold() != ".png":
                raise ValueError("native Python output is outside Studio's PNG output contract")
            size = path.stat().st_size
            if size > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds the Studio file size limit")
            saved.append({
                "index": index,
                "relative_path": path.relative_to(root).as_posix(),
                "width": int(getattr(image, "width", 0)),
                "height": int(getattr(image, "height", 0)),
                "size_bytes": size,
            })
        return saved

    @staticmethod
    def _save_image(job: dict[str, Any], index: int, image: Image.Image) -> Path:
        day = datetime.now().strftime("%Y-%m-%d")
        root = Path(job["output_root"]).resolve()
        folder = root / "aiwf-python" / day
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"aiwf-python-{job['job_id']}-{index}.png"
        metadata = PngInfo()
        metadata.add_text("parameters", str(job["prompt"]))
        metadata.add_text("aiwf_backend_id", PYTHON_BACKEND_ID)
        metadata.add_text("aiwf_model_id", str(job["model_id"]))
        metadata.add_text("aiwf_model_family", str(job["model_family"]))
        image.convert("RGB").save(target, format="PNG", pnginfo=metadata)
        return target

    def _finish(self, job_id: str, state: str, error: dict[str, str] | None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] in {"done", "failed", "cancelled"} or job.get("_finishing"):
                return
            job["_finishing"] = True
            finished_at = _utc_now()
            snapshot = {**job, "state": state, "error": error, "finished_at": finished_at, "queue_position": None}
            snapshot.pop("_finishing", None)
            snapshot.pop("cancel_event", None)
            snapshot.pop("request", None)
            snapshot.pop("generation_job_id", None)
            on_finished = job.pop("on_finished", None)
            snapshot.pop("on_finished", None)
        if callable(on_finished):
            try:
                on_finished(snapshot)
            except Exception:
                pass
        warnings: list[dict[str, str]] = []
        if self._on_done is not None:
            try:
                self._on_done(snapshot)
            except Exception:  # noqa: BLE001 - saved output remains a successful job
                logger.exception("Native Python image job %s completed, but project history could not be recorded", job_id)
                warnings.append({
                    "code": "project_history_not_saved",
                    "message": "Image was saved, but project history could not be updated. Check project storage.",
                })
        with self._lock:
            job["state"], job["error"], job["finished_at"], job["queue_position"] = state, error, finished_at, None
            job["warnings"] = warnings
            job.pop("_finishing", None)

    def _get(self, job_id: str) -> dict[str, Any]:
        if not isinstance(job_id, str) or not _PYTHON_JOB_ID.fullmatch(job_id):
            raise BridgeError(422, "invalid_job_id", "AIWF Python image job IDs look like imgpy- followed by 12 hex characters.")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise BridgeError(404, "job_not_found", "No AIWF Python image job with that ID (jobs are kept until the engine API restarts).")
        return job

    def _prune_locked(self) -> None:
        if len(self._jobs) <= MAX_JOBS_KEPT:
            return
        finished = [job_id for job_id, job in self._jobs.items() if job["state"] in {"done", "failed", "cancelled"}]
        for job_id in finished[: len(self._jobs) - MAX_JOBS_KEPT]:
            del self._jobs[job_id]

    def _view(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": job["job_id"],
                "backend_id": job["backend_id"],
                "model_id": job["model_id"],
                "model_family": job["model_family"],
                "state": job["state"],
                "model": job["model"],
                "prompt": job["prompt"][:300],
                "aspect_ratio": job["aspect_ratio"],
                "quality": job["quality"],
                "seed": job["seed"],
                "project_id": job["project_id"],
                "queue_position": job["queue_position"],
                "created_at": job["created_at"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "elapsed_seconds": round(time.monotonic() - job["created_monotonic"], 1),
                "progress": int(job.get("progress", 0)),
                "message": str(job.get("message", "")),
                "error": job["error"],
                "warnings": list(job.get("warnings", [])),
                "cancel_requested": bool(job.get("cancel_requested")),
                "images": [
                    {**image, "url": f"/api/pro/unified/images/{job['job_id']}/files/{image['index']}"}
                    for image in job["images"]
                ],
            }


class StableDiffusionCppImageJobs:
    """Image job facade around sd.cpp with owned subprocess cancellation."""

    def __init__(
        self,
        *,
        flags: Any | None,
        on_done: Callable[[dict[str, Any]], None] | None = None,
        backend_factory: Callable[[Any], Any] = _make_native_cpp_backend,
        poll_seconds: float = 0.1,
    ) -> None:
        self._flags = flags
        self._backend_factory = backend_factory
        self._on_done = on_done
        self._poll_seconds = max(0.02, poll_seconds)
        self._backend: Any | None = None
        self._backend_lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _get_backend(self) -> Any:
        if self._flags is None:
            raise BridgeError(503, "cpp_backend_unavailable", "The engine API did not receive model-folder settings for the C++ backend.")
        with self._backend_lock:
            if self._backend is None:
                try:
                    self._backend = self._backend_factory(self._flags)
                except Exception as exc:  # noqa: BLE001 - unavailable backend is reported to the native UI
                    raise BridgeError(503, "cpp_backend_unavailable", f"The C++ backend could not be initialized ({type(exc).__name__}).") from exc
            return self._backend

    def model_rows(self) -> list[dict[str, Any]]:
        try:
            return [_checkpoint_model(item, CPP_BACKEND_ID) for item in _supported_checkpoints(self._get_backend())]
        except BridgeError:
            return []
        except Exception:  # noqa: BLE001 - a damaged/unreadable model index should not break the app
            return []

    def readiness(self) -> dict[str, Any]:
        if self._flags is None:
            return {"available": False, "reason": "The engine API did not receive model-folder settings for the C++ backend.", "models": []}
        try:
            backend = self._get_backend()
            models = [_checkpoint_model(item, CPP_BACKEND_ID) for item in _supported_checkpoints(backend)]
            executable = backend.executable
            if not executable or str(executable).startswith("<missing"):
                return {
                    "available": False,
                    "reason": getattr(backend, "resolver_error", None) or "No CUDA 13 stable-diffusion.cpp executable is available.",
                    "models": models,
                    "runtime_info": {},
                }
            probe_ok, probe_reason = backend.probe_cli()
            runtime_info = {"cuda_runtime": "CUDA 13", "cli_probe": "passed" if probe_ok else "failed"}
            if not probe_ok:
                return {"available": False, "reason": probe_reason, "models": models, "runtime_info": runtime_info}
            if not models:
                return {
                    "available": False,
                    "reason": "No supported SD 1.5 or SDXL checkpoint files were found in the folders selected in Setup.",
                    "models": [],
                    "runtime_info": runtime_info,
                }
            return {
                "available": True,
                "reason": "The C++ runtime command and supported checkpoint files were found. This check does not load a model or verify driver compatibility, available memory, or generation.",
                "models": models,
                "runtime_info": runtime_info,
            }
        except BridgeError as exc:
            return {"available": False, "reason": exc.message, "models": [], "runtime_info": {}}
        except Exception as exc:  # noqa: BLE001 - keep the selector usable when inventory checks fail
            return {
                "available": False,
                "reason": f"C++ backend readiness check failed ({type(exc).__name__}).",
                "models": [],
                "runtime_info": {},
            }

    def validate_selection(self, model_id: str) -> Any:
        """Validate the executable and opaque model selection without starting a job."""
        readiness = self.readiness()
        if not readiness["available"]:
            raise BridgeError(503, "cpp_backend_unavailable", str(readiness["reason"]))
        backend = self._get_backend()
        selected = next(
            (
                item
                for item in _supported_checkpoints(backend)
                if _opaque_model_id(CPP_BACKEND_ID, item) == model_id
            ),
            None,
        )
        if selected is None:
            raise BridgeError(422, "model_not_supported", "The selected model is not a supported installed SD 1.5 or SDXL checkpoint for the C++ backend.")
        try:
            checkpoint = backend.resolve_checkpoint(str(selected.id))
        except (ModelNotFoundError, ValueError) as exc:
            raise BridgeError(422, "model_not_found", "The selected C++ model is no longer present in the configured model folders.") from exc
        if str(checkpoint.architecture).casefold() not in CPP_FAMILIES or str(checkpoint.kind).casefold() != "checkpoint":
            raise BridgeError(422, "model_not_supported", "The selected C++ model does not support this text-to-image workflow.")
        return checkpoint

    def start(
        self,
        *,
        prompt: str,
        aspect_ratio: str,
        quality: str,
        seed: int | None,
        model_id: str,
        project_id: str | None,
        output_root: Path | None,
        validated_checkpoint: Any | None = None,
        on_finished: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        text = (prompt or "").strip()
        if not text or len(text) > MAX_PROMPT_CHARS:
            raise BridgeError(422, "invalid_prompt", f"Describe the image in 1-{MAX_PROMPT_CHARS} characters.")
        if aspect_ratio not in ASPECT_RATIOS:
            raise BridgeError(422, "invalid_aspect_ratio", "Aspect ratio must be one of " + ", ".join(ASPECT_RATIOS) + ".")
        if quality not in QUALITY_MEGAPIXELS:
            raise BridgeError(422, "invalid_quality", "Quality must be draft or standard.")
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise BridgeError(422, "invalid_seed", f"Seed must be a whole number from 0 to {MAX_SEED}.")
        if output_root is None or not Path(output_root).is_dir():
            raise BridgeError(503, "no_output_root", "Studio's output folder is not available, so the image would have nowhere to go.")

        checkpoint = validated_checkpoint if validated_checkpoint is not None else self.validate_selection(model_id)

        width, height = _image_dimensions(aspect_ratio, quality)
        request = GenerationRequest(
            mode=GenerationMode.TXT2IMG,
            prompt=text,
            steps=20 if quality == "draft" else 30,
            cfg_scale=7.0,
            width=width,
            height=height,
            seed=seed,
            sampler="euler_a",
            checkpoint_id=checkpoint.id,
            pipeline_backend="sdcpp",
        )
        job = {
            "job_id": CPP_JOB_ID_PREFIX + uuid.uuid4().hex[:CPP_JOB_ID_SUFFIX_LENGTH],
            "backend_id": CPP_BACKEND_ID,
            "model_id": str(model_id),
            "model": str(checkpoint.title),
            "model_family": str(checkpoint.architecture).casefold(),
            "request": request,
            "state": "queued",
            "prompt": text,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "seed": seed,
            "project_id": project_id,
            "output_root": Path(output_root).resolve(),
            "queue_position": None,
            "created_at": _utc_now(),
            "created_monotonic": time.monotonic(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "images": [],
            "cancel_requested": False,
            "cancel_event": threading.Event(),
            "on_finished": on_finished,
        }
        with self._lock:
            self._jobs[job["job_id"]] = job
            self._prune_locked()
        worker = threading.Thread(target=self._run_guarded, args=(job["job_id"],), name=f"cpp-image-job-{job['job_id']}", daemon=True)
        try:
            worker.start()
        except Exception as exc:
            with self._lock:
                self._jobs.pop(job["job_id"], None)
            raise BridgeError(503, "cpp_job_start_failed", "The C++ image job could not be started.") from exc
        return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}

    def status(self, job_id: str) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "job": self._view(self._get(job_id))}

    def file(self, job_id: str, index: int) -> Path:
        job = self._get(job_id)
        images = job["images"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(images):
            raise BridgeError(404, "image_not_found", "This job has no image with that number.")
        root = Path(job["output_root"]).resolve()
        path = (root / images[index]["relative_path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise BridgeError(404, "image_not_found", "The saved image is no longer in Studio's output folder.")
        return path

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        with self._lock:
            if job["state"] not in {"done", "failed", "cancelled"}:
                job["cancel_requested"] = True
                job["cancel_event"].set()
        return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}

    def _run_guarded(self, job_id: str) -> None:
        try:
            self._run(job_id)
        except Exception as exc:  # noqa: BLE001 - failures become a visible terminal job state
            self._finish(job_id, "failed", {"code": "cpp_generation_failed", "message": self._failure_message(exc)})

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] in {"done", "failed", "cancelled"}:
                return
            if job["cancel_event"].is_set():
                cancelled = True
            else:
                cancelled = False
                job["state"] = "running"
                job["started_at"] = _utc_now()
        if cancelled:
            self._finish(job_id, "cancelled", None)
            return
        try:
            result = self._get_backend().generate(job["request"], should_cancel=job["cancel_event"].is_set)
        except GenerationCancelledError:
            self._finish(job_id, "cancelled", None)
            return
        if job["cancel_event"].is_set():
            self._finish(job_id, "cancelled", None)
            return
        images = list(getattr(result, "images", []) or [])
        if not images:
            self._finish(job_id, "failed", {"code": "no_image", "message": "The C++ backend completed without returning an image."})
            return
        saved: list[dict[str, Any]] = []
        try:
            infotexts = list(getattr(result, "infotexts", []) or [])
            for index, image in enumerate(images):
                saved.append(self._save_image(job, index, image, infotexts[index] if index < len(infotexts) else ""))
        except (OSError, ValueError) as exc:
            for item in saved:
                try:
                    (Path(job["output_root"]) / item["relative_path"]).unlink(missing_ok=True)
                except OSError:
                    pass
            self._finish(job_id, "failed", {"code": "save_failed", "message": f"The C++ image was generated but could not be saved ({type(exc).__name__})."})
            return
        with self._lock:
            current = self._jobs.get(job_id)
            if current is not None:
                current["images"] = saved
        self._finish(job_id, "done", None)

    @staticmethod
    def _failure_message(exc: Exception) -> str:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return f"The C++ image backend failed: {message[:400]}"

    @staticmethod
    def _save_image(job: dict[str, Any], index: int, image: Image.Image, infotext: str) -> dict[str, Any]:
        day = datetime.now().strftime("%Y-%m-%d")
        folder = Path(job["output_root"]) / "stable-diffusion-cpp" / day
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"sd-cpp-{job['job_id']}-{index}.png"
        metadata = PngInfo()
        metadata.add_text("parameters", infotext or str(job["prompt"]))
        metadata.add_text("aiwf_backend_id", CPP_BACKEND_ID)
        metadata.add_text("aiwf_model_id", str(job["model_id"]))
        metadata.add_text("aiwf_model_family", str(job["model_family"]))
        converted = image.convert("RGB")
        converted.save(target, format="PNG", pnginfo=metadata)
        size = target.stat().st_size
        if size > MAX_IMAGE_BYTES:
            target.unlink(missing_ok=True)
            raise ValueError("image exceeds the Studio file size limit")
        return {
            "index": index,
            "relative_path": target.relative_to(Path(job["output_root"])).as_posix(),
            "width": converted.width,
            "height": converted.height,
            "size_bytes": size,
        }

    def _finish(self, job_id: str, state: str, error: dict[str, str] | None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] in {"done", "failed", "cancelled"} or job.get("_finishing"):
                return
            job["_finishing"] = True
            finished_at = _utc_now()
            snapshot = {**job, "state": state, "error": error, "finished_at": finished_at, "queue_position": None}
            snapshot.pop("_finishing", None)
            snapshot.pop("cancel_event", None)
            on_finished = job.pop("on_finished", None)
            snapshot.pop("on_finished", None)
        if callable(on_finished):
            try:
                on_finished(snapshot)
            except Exception:  # noqa: BLE001 - cleanup callbacks must not corrupt job state
                pass
        warnings: list[dict[str, str]] = []
        if self._on_done is not None:
            try:
                self._on_done(snapshot)
            except Exception:  # noqa: BLE001 - a ledger problem must not lose the finished image
                logger.exception("Native C++ image job %s completed, but project history could not be recorded", job_id)
                warnings.append({
                    "code": "project_history_not_saved",
                    "message": "Image was saved, but project history could not be updated. Check project storage.",
                })
        with self._lock:
            job["state"], job["error"], job["finished_at"], job["queue_position"] = state, error, finished_at, None
            job["warnings"] = warnings
            job.pop("_finishing", None)

    def _get(self, job_id: str) -> dict[str, Any]:
        if not isinstance(job_id, str) or not _CPP_JOB_ID.fullmatch(job_id):
            raise BridgeError(422, "invalid_job_id", "C++ image job IDs look like imgcpp- followed by 12 hex characters.")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise BridgeError(404, "job_not_found", "No C++ image job with that ID (jobs are kept until the engine API restarts).")
        return job

    def _prune_locked(self) -> None:
        if len(self._jobs) <= MAX_JOBS_KEPT:
            return
        finished = [job_id for job_id, job in self._jobs.items() if job["state"] in {"done", "failed", "cancelled"}]
        for job_id in finished[: len(self._jobs) - MAX_JOBS_KEPT]:
            del self._jobs[job_id]

    def _view(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": job["job_id"],
                "backend_id": job["backend_id"],
                "model_id": job["model_id"],
                "model_family": job["model_family"],
                "state": job["state"],
                "model": job["model"],
                "prompt": job["prompt"][:300],
                "aspect_ratio": job["aspect_ratio"],
                "quality": job["quality"],
                "seed": job["seed"],
                "project_id": job["project_id"],
                "queue_position": job["queue_position"],
                "created_at": job["created_at"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "elapsed_seconds": round(time.monotonic() - job["created_monotonic"], 1),
                "error": job["error"],
                "warnings": list(job.get("warnings", [])),
                "cancel_requested": bool(job.get("cancel_requested")),
                "images": [
                    {**image, "url": f"/api/pro/unified/images/{job['job_id']}/files/{image['index']}"}
                    for image in job["images"]
                ],
            }
