from __future__ import annotations

import gc
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Callable

import numpy as np

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions, AudioGenerationResult, AudioMuxResult
from aiwf.core.domain.engine import EngineTenant
from aiwf.infrastructure.video.processing import VideoProcessor, _resolve_ffmpeg
from aiwf.services import audio_licenses
from aiwf.services.model_files import configured_model_roots
from aiwf.services.route_lifecycle import support_revision

logger = logging.getLogger(__name__)


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _is_complete_safetensors(path: Path) -> bool:
    """Check the safetensors header and declared payload length without loading tensors."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                return False
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 0 or header_length > 100 * 1024 * 1024:
                return False
            header_bytes = stream.read(header_length)
            if len(header_bytes) != header_length:
                return False
            header = json.loads(header_bytes.decode("utf-8"))
            tensors = [value for key, value in header.items() if key != "__metadata__"]
            if not tensors:
                return False
            offsets = sorted((int(item["data_offsets"][0]), int(item["data_offsets"][1])) for item in tensors)
            data_bytes = path.stat().st_size - 8 - header_length
            cursor = 0
            for start, end in offsets:
                if start != cursor or end < start:
                    return False
                cursor = end
            return cursor == data_bytes
    except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, struct.error):
        return False


_MINIMUM_AUDIO_SETUP_LOCK = threading.Lock()
_AUDIO_MODEL_OPERATION_LOCK = threading.Lock()
_EVENTS_DESCRIBER_READINESS_TIMEOUT_SECONDS = 1.0
_EVENTS_DESCRIBER_READINESS_CACHE_SECONDS = 2.0
_MUSICGEN_MINIMUM_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
_MUSICGEN_VARIANTS = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "melody": "facebook/musicgen-melody",
    "stereo-small": "facebook/musicgen-stereo-small",
}
_MMAUDIO_SHARED_FILES = (
    Path("ext_weights") / "synchformer_state_dict.pth",
)
_MMAUDIO_CLIP_REPO = "apple/DFN5B-CLIP-ViT-H-14-384"
_MMAUDIO_CLIP_REPO_CACHE = "models--apple--DFN5B-CLIP-ViT-H-14-384"
_MMAUDIO_CLIP_REQUIRED_FILES = (
    "open_clip_config.json",
)
_MMAUDIO_CLIP_WEIGHT_FILES = (
    "open_clip_pytorch_model.safetensors",
    "open_clip_pytorch_model.bin",
)
_MMAUDIO_CLIP_HUB_LAYOUTS = (
    Path("hub"),
    Path(".cache") / "huggingface" / "hub",
    Path("audio") / "MMAudio" / ".cache" / "huggingface" / "hub",
    Path("AIWF") / "mmaudio" / ".cache" / "huggingface" / "hub",
    Path("AIWF") / "audio" / "MMAudio" / ".cache" / "huggingface" / "hub",
)
_MMAUDIO_CLIP_ALLOW_PATTERNS = (
    "open_clip_config.json",
    "open_clip_pytorch_model.safetensors",
    "open_clip_pytorch_model.bin",
)
_MMAUDIO_16K_FILES = (
    Path("weights") / "mmaudio_small_16k.pth",
    Path("ext_weights") / "v1-16.pth",
    Path("ext_weights") / "best_netG.pt",
    *_MMAUDIO_SHARED_FILES,
)
# ---- commercially licensed engines (aiwf/services/audio_licenses.py) ----------------------------------
ACESTEP_MODEL_ID = "acestep:1.5-turbo"
MOSS_SFX_MODEL_ID = "moss-sfx:v2.0"
# video soundtrack: a vision model finds the sounds, MOSS-SoundEffect makes them (aiwf/services/video_soundtrack.py)
EVENTS_MODEL_ID = "events:moss-sfx"
# the files that must be complete before a render may start (largest weights of each engine)
_ACESTEP_REQUIRED_FILES = (
    "acestep-v15-turbo/model.safetensors",
    "acestep-v15-turbo/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "Qwen3-Embedding-0.6B/model.safetensors",
)
ACESTEP_MIN_DURATION_SECONDS = 10.0
_MOSS_SFX_REQUIRED_FILES = (
    "model_index.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/vae_128d_48k.pth",
)


def _last_json_line(text: str) -> dict[str, Any]:
    """The last line of a worker's output that parses as a JSON object ({} when there is none)."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return {}


_MMAUDIO_INSTALLABLE_VARIANTS = (
    "small_16k",
    "large_44k_v2",
    "large_44k",
    "medium_44k",
    "small_44k",
)


def _mmaudio_variant_files(variant: str) -> tuple[Path, ...]:
    """Files required by the upstream MMAudio demo for one checkpoint variant."""
    if variant == "small_16k":
        return (
            Path("weights") / "mmaudio_small_16k.pth",
            *_MMAUDIO_16K_FILES[1:],
        )
    if variant in _MMAUDIO_INSTALLABLE_VARIANTS:
        return (
            Path("weights") / f"mmaudio_{variant}.pth",
            Path("ext_weights") / "v1-44.pth",
            *_MMAUDIO_SHARED_FILES,
        )
    return ()


_MMAUDIO_SHARED_LAYOUTS = (
    Path(),
    Path("MMAudio"),
    Path("audio") / "MMAudio",
    Path("models") / "MMAudio",
    Path("models") / "audio" / "MMAudio",
    Path("engines") / "audio" / "MMAudio",
)


def _mmaudio_clip_cache_ready(cache_root: Path) -> bool:
    """Check for the upstream OpenCLIP HF snapshot without loading model weights."""
    try:
        resolved_root = cache_root.resolve(strict=True)
        repo_root = (resolved_root / _MMAUDIO_CLIP_REPO_CACHE).resolve(strict=True)
        repo_root.relative_to(resolved_root)
        revision_path = repo_root / "refs" / "main"
        if not _mmaudio_cache_file_ready(revision_path, resolved_root):
            return False
        revision = revision_path.read_text(encoding="utf-8").strip()
        if len(revision) != 40 or any(char not in "0123456789abcdefABCDEF" for char in revision):
            return False
        snapshot = (repo_root / "snapshots" / revision).resolve(strict=True)
        snapshot.relative_to(resolved_root)
        required_paths = [snapshot / name for name in _MMAUDIO_CLIP_REQUIRED_FILES]
        weight_paths = [snapshot / name for name in _MMAUDIO_CLIP_WEIGHT_FILES]
        if all(_mmaudio_cache_file_ready(path, resolved_root) for path in required_paths):
            return any(_mmaudio_cache_file_ready(path, resolved_root) for path in weight_paths)
    except (OSError, RuntimeError, ValueError):
        return False
    return False


def _mmaudio_cache_file_ready(path: Path, cache_root: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(cache_root)
    except (OSError, RuntimeError, ValueError):
        return False
    return _is_nonempty_file(resolved)


def _find_mmaudio_clip_hub_cache(model_roots: list[Path]) -> Path | None:
    """Find a complete DFN5B OpenCLIP snapshot under configured model roots."""
    for root in model_roots:
        try:
            resolved_root = root.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        for layout in _MMAUDIO_CLIP_HUB_LAYOUTS:
            candidate = resolved_root / layout
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except (OSError, RuntimeError, ValueError):
                continue
            if _mmaudio_clip_cache_ready(resolved):
                return resolved
    return None


class AudioUnavailable(RuntimeError):
    """Raised when optional audio generation dependencies or tools are missing."""


class AudioLicenseBlocked(AudioUnavailable):
    """Raised when a non-commercial model is requested while research mode is off."""


class AudioGenerationService:
    """Optional local text-to-audio, video-conditioned audio, and video muxing."""

    def __init__(
        self,
        flags: RuntimeFlags,
        settings: UserSettings,
        devices=None,
        supervisor=None,
        unload_image_models: Callable[[], Any] | None = None,
    ) -> None:
        self.flags = flags
        self.settings = settings
        self.devices = devices
        self.supervisor = supervisor
        self.unload_image_models = unload_image_models
        self._model: Any | None = None
        self._model_key: tuple[str, str, str] | None = None
        self._events_describer_cache_lock = threading.Lock()
        self._events_describer_model_cache: tuple[float, str | None] | None = None

    @contextmanager
    def _gpu_tenant(self, reason: str):
        if self.supervisor is None:
            yield
            return
        try:
            with self.supervisor.tenant_session(EngineTenant.AUDIO, reason=reason):
                yield
        except RuntimeError as exc:
            raise AudioUnavailable(f"GPU busy: {exc}") from exc

    @staticmethod
    def _resolve_ffprobe(ffmpeg: str) -> str | None:
        ffmpeg_path = Path(ffmpeg)
        probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        sibling_probe = ffmpeg_path.with_name(probe_name)
        return str(sibling_probe) if sibling_probe.is_file() else shutil.which("ffprobe")

    def folder_help(self) -> str:
        return (
            "Music uses Transformers MusicGen. Sound effects and video-conditioned audio use an isolated "
            "MMAudio engine. AudioCraft stays out of the shared Studio environment because its current release "
            "pins an older PyTorch stack. Use the minimum setup button before the first audio run."
        )

    # ---- commercial-use policy (aiwf/services/audio_licenses.py) ----------------------------------
    # By default only models whose licences allow commercial use are offered or run. Research mode
    # (UserSettings.allow_noncommercial_audio_models) brings back the non-commercial ones, labeled.
    def research_mode(self) -> bool:
        return bool(getattr(self.settings, "allow_noncommercial_audio_models", False))

    def _require_licensed(self, model_id: str) -> None:
        if not audio_licenses.allowed(model_id, research_mode=self.research_mode()):
            raise AudioLicenseBlocked(audio_licenses.blocked_message(model_id))

    def _offered(self, choices: list[tuple[str, str]]) -> list[tuple[str, str]]:
        # this loop drops models the policy does not allow and marks non-commercial ones in research mode
        offered = []
        for label, model_id in choices:
            if not audio_licenses.allowed(model_id, research_mode=self.research_mode()):
                continue
            if not audio_licenses.commercial_ok(model_id):
                label = f"{label} · non-commercial ({audio_licenses.license_for(model_id)['license']})"
            offered.append((label, model_id))
        return offered

    def music_model_choices(self) -> list[tuple[str, str]]:
        return self._offered([
            ("ACE-Step 1.5 turbo", ACESTEP_MODEL_ID),
            ("MusicGen small (minimum)", "facebook/musicgen-small"),
            ("MusicGen medium", "facebook/musicgen-medium"),
            ("MusicGen melody", "facebook/musicgen-melody"),
            ("MusicGen stereo small", "facebook/musicgen-stereo-small"),
        ])

    def sfx_model_choices(self) -> list[tuple[str, str]]:
        # AudioGen currently has no isolated in-app AudioCraft installer or
        # runtime. Keep it out of selectable choices until setup can complete.
        return self._offered([("MOSS-SoundEffect v2.0", MOSS_SFX_MODEL_ID), *self._mmaudio_choices()])

    def video_audio_model_choices(self) -> list[tuple[str, str]]:
        return self._offered([("Video soundtrack: describe scene + MOSS-SoundEffect", EVENTS_MODEL_ID), *self._mmaudio_choices()])

    @staticmethod
    def _mmaudio_choices() -> list[tuple[str, str]]:
        return [
            ("MMAudio small 16k (minimum)", "mmaudio:small_16k"),
            ("MMAudio large 44k v2 (install separately)", "mmaudio:large_44k_v2"),
            ("MMAudio large 44k (install separately)", "mmaudio:large_44k"),
            ("MMAudio medium 44k (install separately)", "mmaudio:medium_44k"),
            ("MMAudio small 44k (install separately)", "mmaudio:small_44k"),
        ]

    # ---- commercially licensed engines: ACE-Step 1.5 (music) and MOSS-SoundEffect v2.0 -----------------
    # Each runs in its own environment through AIWF's worker script (engines/<engine>/aiwf_worker.py),
    # one subprocess per render, with weights in Studio's model library. Installed only by its own
    # Install button (scripts/bootstrap_commercial_audio.py).
    def _engine_spec(self, model_id: str) -> dict[str, Any] | None:
        repo = self.flags.data_dir.resolve()
        library = self.flags.resolved_models_dir() / "audio"
        if str(model_id).startswith("acestep:"):
            return {
                "engine": "acestep", "label": "ACE-Step 1.5",
                "python": repo / "engines" / "acestep" / "ACE-Step-1.5" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
                "worker": repo / "engines" / "acestep" / "aiwf_worker.py",
                "weights": library / "ACE-Step-1.5",
                "files": _ACESTEP_REQUIRED_FILES,
            }
        if str(model_id).startswith(("moss-sfx:", "events:")):
            return {
                "engine": "moss-sfx", "label": "MOSS-SoundEffect v2.0",
                "python": repo / "engines" / "moss_sfx" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
                "worker": repo / "engines" / "moss_sfx" / "aiwf_worker.py",
                "weights": library / "MOSS-SoundEffect-v2.0",
                "files": _MOSS_SFX_REQUIRED_FILES,
            }
        return None

    def _events_describer_available_model(self) -> str | None:
        """Return the local commercial vision model, caching probes for a short interval."""
        with self._events_describer_cache_lock:
            now = monotonic()
            cached = self._events_describer_model_cache
            if cached is not None and now - cached[0] < _EVENTS_DESCRIBER_READINESS_CACHE_SECONDS:
                return cached[1]

            try:
                from aiwf.services.video_soundtrack import ChatDescriber

                model = ChatDescriber().available_model(timeout=_EVENTS_DESCRIBER_READINESS_TIMEOUT_SECONDS)
            except Exception:
                model = None
            self._events_describer_model_cache = (monotonic(), model)
            return model

    def commercial_engine_missing(self, model_id: str) -> list[str]:
        """What is still missing for this engine (empty when it can render)."""
        spec = self._engine_spec(model_id)
        if spec is None:
            return [f"Unknown engine for {model_id}"]
        missing = [] if spec["python"].is_file() else [f"{spec['label']} environment ({spec['python']})"]
        if str(model_id).startswith("events:") and not _resolve_ffmpeg():
            missing.append("ffmpeg (needed to read video frames)")
        missing += [str(spec["weights"] / name) for name in spec["files"] if not _is_nonempty_file(spec["weights"] / name)]
        if str(model_id).startswith("events:"):
            # Avoid a local service probe until the event route's own files and
            # frame reader are ready; it cannot be selected or rendered yet.
            if missing:
                return missing
            if self._events_describer_available_model() is None:
                missing.append("Qwen2.5-VL-7B vision model in the local chat engine")
        return missing

    def commercial_engine_ready(self, model_id: str) -> bool:
        return not self.commercial_engine_missing(model_id)

    def install_commercial_engine(self, engine: str) -> dict[str, Any]:
        """Install ACE-Step 1.5 or MOSS-SoundEffect (code, environment, weights) and return the new status."""
        if engine not in {"acestep", "moss-sfx"}:
            raise AudioUnavailable(f"Unknown audio engine: {engine or '(empty)'}")
        if not _AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False):
            raise AudioUnavailable("An audio render or model setup operation is already running.")
        if not _MINIMUM_AUDIO_SETUP_LOCK.acquire(blocking=False):
            _AUDIO_MODEL_OPERATION_LOCK.release()
            raise AudioUnavailable("Another audio setup operation is already running.")
        try:
            root = self.flags.data_dir.resolve()
            script = root / "scripts" / "bootstrap_commercial_audio.py"
            command = [sys.executable, str(script), "--engine", engine, "--repo", str(root),
                       "--models-dir", str(self.flags.resolved_models_dir()), "--json"]
            try:
                result = subprocess.run(command, cwd=str(root), capture_output=True, text=True,
                                        timeout=6 * 60 * 60, env=self._mmaudio_install_environment())
            except (OSError, subprocess.SubprocessError) as exc:
                raise AudioUnavailable(f"Could not install {engine}: {exc}") from exc
            if result.returncode != 0:
                raise AudioUnavailable((result.stderr or result.stdout or f"{engine} installation failed").strip()[-5000:])
            status = self.setup_status(deep=False)
            status["receipt"] = _last_json_line(result.stdout)
            return status
        finally:
            _MINIMUM_AUDIO_SETUP_LOCK.release()
            _AUDIO_MODEL_OPERATION_LOCK.release()

    def _run_engine_worker(self, model_id: str, job: dict[str, Any], *, label: str) -> dict[str, Any]:
        """Run one render job in the engine's own environment and return its JSON reply."""
        spec = self._engine_spec(model_id)
        missing = self.commercial_engine_missing(model_id)
        if spec is None or missing:
            raise AudioUnavailable(f"{label} is not installed yet. Install it from Audio setup first. Missing: {', '.join(missing[:3])}")
        with tempfile.TemporaryDirectory(prefix=f"aiwf-{spec['engine']}-") as scratch:
            job_path = Path(scratch) / "job.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            try:
                completed = subprocess.run([str(spec["python"]), str(spec["worker"]), "render", str(job_path)],
                                           cwd=str(spec["worker"].parent), capture_output=True, text=True,
                                           encoding="utf-8", errors="replace",   # workers print UTF-8, not cp1252
                                           timeout=60 * 60, env=env)
            except (OSError, subprocess.SubprocessError) as exc:
                raise AudioUnavailable(f"{label} could not run: {exc}") from exc
        reply = _last_json_line(completed.stdout)
        if completed.returncode != 0 or not reply.get("ok"):
            detail = reply.get("error") or (completed.stderr or completed.stdout or "no output").strip()[-1500:]
            raise AudioUnavailable(f"{label} failed: {detail}")
        return reply

    def _generate_acestep(self, options: AudioGenerationOptions, dest: Path) -> int:
        free_gb = self._free_vram_gb()
        if free_gb is not None and free_gb < 3.5:
            raise AudioUnavailable(f"Music generation deferred: {free_gb:.1f} GB VRAM is free; ACE-Step needs at least 3.5 GB.")
        item = {
            "caption": options.prompt.strip(),
            "lyrics": "",
            "duration": max(
                ACESTEP_MIN_DURATION_SECONDS,
                float(options.duration_seconds),
            ),  # ACE-Step renders 10-600 s
            "seed": int(options.seed),
            "steps": 8,                                                # the turbo model is tuned for 8 steps
            "output": str(dest),
        }
        # with less than 7 GB free the weights wait in system RAM and move to the GPU per stage
        job = {"checkpoints": str(self._engine_spec(ACESTEP_MODEL_ID)["weights"]), "device": "cuda",
               "offload": free_gb is not None and free_gb < 7.0, "items": [item]}
        reply = self._run_engine_worker(ACESTEP_MODEL_ID, job, label="ACE-Step 1.5")
        return int(reply["results"][0]["sample_rate"])

    def render_sound_effects(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Render several MOSS-SoundEffect clips with one model load (used by sound effects and video soundtracks)."""
        free_gb = self._free_vram_gb()
        if free_gb is not None and free_gb < 8.0:
            raise AudioUnavailable(f"Sound effects deferred: {free_gb:.1f} GB VRAM is free; MOSS-SoundEffect needs at least 8 GB.")
        job = {"model_dir": str(self._engine_spec(MOSS_SFX_MODEL_ID)["weights"]), "device": "cuda", "items": items}
        return self._run_engine_worker(MOSS_SFX_MODEL_ID, job, label="MOSS-SoundEffect")["results"]

    def _generate_event_soundtrack(self, video: Path, options: AudioGenerationOptions, dest: Path) -> int:
        """Describe the video, make each sound with MOSS-SoundEffect, and mix them onto one track."""
        from scipy.io import wavfile

        from aiwf.services import video_soundtrack as soundtrack

        ffmpeg = _resolve_ffmpeg()
        if not ffmpeg:
            raise AudioUnavailable("ffmpeg is required to read frames from the video.")
        duration = float(options.duration_seconds)
        describer = getattr(self, "describer", None) or soundtrack.ChatDescriber()
        # step 1-2: frames, then the vision model lists the sounds
        try:
            frames = soundtrack.extract_frames(ffmpeg, video, duration)
            events = soundtrack.plan_events(describer, frames, duration, hint=options.prompt)
        except Exception as exc:  # model missing, chat engine off, unusable answer
            raise AudioUnavailable(f"Could not work out the video's sounds: {exc}") from exc
        # the vision model no longer needs the GPU; free it so MOSS-SoundEffect has room
        release = getattr(describer, "release", None)
        if callable(release):
            release()
        # step 3: one MOSS-SoundEffect model load renders every sound
        with tempfile.TemporaryDirectory(prefix="aiwf-soundtrack-") as scratch:
            items = [{"prompt": event.prompt, "seconds": min(30.0, event.seconds), "seed": int(options.seed) + index if int(options.seed) >= 0 else -1,
                      "steps": 50, "cfg_scale": 4.0, "output": str(Path(scratch) / f"event{index}.wav")}
                     for index, event in enumerate(events)]
            rendered = self.render_sound_effects(items)
            clips = []
            sample_rate = int(rendered[0]["sample_rate"])
            for event, result in zip(events, rendered):
                rate, data = wavfile.read(result["output"])
                data = data.astype(np.float32) / (2147483648.0 if data.dtype == np.int32 else 32768.0 if data.dtype == np.int16 else 1.0)
                clips.append((event, data if data.ndim == 1 else data.mean(axis=1)))
        # step 4: place and mix
        track = soundtrack.mix_events(clips, duration, sample_rate)
        wavfile.write(str(dest), sample_rate, track)
        self.last_soundtrack_events = [event.__dict__ for event in events]
        return sample_rate

    def _generate_moss_sfx(self, options: AudioGenerationOptions, dest: Path) -> int:
        results = self.render_sound_effects([{
            "prompt": options.prompt.strip(),
            "seconds": min(30.0, float(options.duration_seconds)),     # MOSS renders up to 30 s
            "seed": int(options.seed),
            "steps": max(25, min(100, int(options.steps) * 2)),        # 50 steps by default
            "cfg_scale": float(options.cfg_coef),
            "negative_prompt": options.negative_prompt,
            "output": str(dest),
        }])
        return int(results[0]["sample_rate"])

    def _free_vram_gb(self) -> float | None:
        try:
            from aiwf.services.gpu_memory import nvidia_smi_free_bytes

            free_bytes = nvidia_smi_free_bytes()
        except Exception:
            return None
        return None if free_bytes is None else float(free_bytes) / (1024**3)

    def available_video_audio_model_choices(self) -> list[tuple[str, str]]:
        return [
            choice
            for choice in self.video_audio_model_choices()
            if (self.commercial_engine_ready(choice[1]) if choice[1].startswith("events:")
                else self._mmaudio_variant_ready(self._mmaudio_variant(choice[1])))
        ]

    def setup_status(self, *, deep: bool = False) -> dict[str, Any]:
        root = self.flags.data_dir.resolve()
        musicgen_root = self._minimum_musicgen_root()
        musicgen_missing = self._musicgen_missing_files("small")
        dependency_modules = ("torch", "transformers", "scipy", "huggingface_hub")
        missing_dependencies = [name for name in dependency_modules if importlib.util.find_spec(name) is None]

        mmaudio_root = self._mmaudio_root()
        mmaudio_python = self._audio_engine_python()
        mmaudio_demo = mmaudio_root / "demo.py"
        mmaudio_missing = [
            str(path) for path in _MMAUDIO_16K_FILES
            if not _is_nonempty_file(mmaudio_root / path)
        ]
        if not self._mmaudio_clip_hub_cache():
            mmaudio_missing.append(f"Hugging Face cache for {_MMAUDIO_CLIP_REPO} (open_clip_config.json and model weights)")

        mmaudio_import_error = (
            self._mmaudio_runtime_import_error() if deep else ""
        )
        mmaudio_import_ok = (
            mmaudio_python.is_file()
            and mmaudio_demo.is_file()
            and (not deep or not mmaudio_import_error)
        )

        from aiwf.services.audio_lab import AudioLabService

        lab_status = AudioLabService(self.flags.resolved_output_dir()).status(deep=deep)
        music_ready = not missing_dependencies and not musicgen_missing
        mmaudio_ready = mmaudio_import_ok and not mmaudio_missing
        lab_ready = bool(lab_status.installed)
        ffmpeg = _resolve_ffmpeg()
        ffprobe = self._resolve_ffprobe(ffmpeg) if ffmpeg else None
        mux_ready = bool(ffmpeg and ffprobe)
        research_mode = self.research_mode()
        mux_missing = []
        if not ffmpeg:
            mux_missing.append("ffmpeg")
        if ffmpeg and not ffprobe:
            mux_missing.append("ffprobe")
        # Commercial-safe minimum setup skips MusicGen/MMAudio. Here minimumReady
        # covers shared Audio Lab and mux tooling; per-engine readiness remains
        # in musicReady, sfxReady, and videoAudioReady.
        minimum_ready = lab_ready and mux_ready and (
            not research_mode or (music_ready and mmaudio_ready)
        )
        if minimum_ready and deep:
            message = "Minimum Audio runtime checks passed. Models are not loaded until selected for generation."
        elif minimum_ready:
            message = "Minimum Audio files and dependencies are detected; runtime checks have not run."
        elif _MINIMUM_AUDIO_SETUP_LOCK.locked():
            message = "Minimum Audio setup is running. Keep Studio open until it finishes."
        else:
            message = "Download the minimum audio models and isolated dependencies before the first run."

        return {
            "minimumReady": minimum_ready,
            "runtimeChecksPerformed": bool(deep),
            "installing": _MINIMUM_AUDIO_SETUP_LOCK.locked(),
            "musicReady": music_ready or self.commercial_engine_ready(ACESTEP_MODEL_ID),
            "musicDependenciesReady": not missing_dependencies,
            "sfxReady": mmaudio_ready or self.commercial_engine_ready(MOSS_SFX_MODEL_ID),
            # Video-audio routes must finish by muxing generated audio back
            # into the source video; model/runtime readiness alone is not enough.
            "videoAudioReady": (mmaudio_ready or self.commercial_engine_ready(EVENTS_MODEL_ID)) and mux_ready,
            "labReady": lab_ready,
            "muxReady": mux_ready,
            "message": message,
            "estimatedDownload": "Up to about 10 GB on a clean install; existing files are reused.",
            # the commercial-use policy, so every surface can show it the same way
            "researchMode": research_mode,
            "licenseNotice": (
                "Research mode is on: non-commercial models (MusicGen, MMAudio; CC-BY-NC 4.0) are available and "
                "labeled. Do not use their output commercially."
                if research_mode else
                "Commercial-safe mode: only audio models whose licences allow commercial use are offered. "
                "Non-commercial models (MusicGen, MMAudio) are hidden; Audio settings can enable them for research."
            ),
            "licenses": {
                model_id: audio_licenses.license_for(model_id)
                for model_id in ("facebook/musicgen-small", "mmaudio:small_16k", ACESTEP_MODEL_ID, MOSS_SFX_MODEL_ID, EVENTS_MODEL_ID)
            },
            "defaults": {
                "music": ACESTEP_MODEL_ID,
                "sfx": MOSS_SFX_MODEL_ID,
                "videoAudio": EVENTS_MODEL_ID,
            },
            "components": [
                *[
                    {
                        "id": f"musicgen-{variant}",
                        "label": f"MusicGen {variant.replace('-', ' ').title()}",
                        "ready": self._musicgen_variant_ready(variant) and not missing_dependencies,
                        "path": str(self._musicgen_root(variant)),
                        "missing": [*missing_dependencies, *self._musicgen_missing_files(variant)],
                    }
                    for variant in _MUSICGEN_VARIANTS
                ],
                {
                    "id": "mmaudio-small-16k",
                    "label": "MMAudio Small 16 kHz",
                    "ready": mmaudio_ready,
                    "sharedReady": self._mmaudio_variant_shared_ready("small_16k"),
                    "path": str(mmaudio_root),
                    "missing": mmaudio_missing,
                    "error": mmaudio_import_error,
                },
                *[
                    {
                        "id": f"mmaudio-{variant.replace('_', '-')}",
                        "label": f"MMAudio {variant.replace('_', ' ').title()}",
                        "ready": self._mmaudio_variant_ready(variant),
                        "sharedReady": self._mmaudio_variant_shared_ready(variant),
                        "path": str(mmaudio_root),
                        "missing": [
                            str(path)
                            for path in _mmaudio_variant_files(variant)
                            if not _is_nonempty_file(mmaudio_root / path)
                        ] + (
                            [f"Hugging Face cache for {_MMAUDIO_CLIP_REPO} (open_clip_config.json and model weights)"]
                            if not self._mmaudio_clip_hub_cache()
                            else []
                        ),
                    }
                    for variant in _MMAUDIO_INSTALLABLE_VARIANTS
                    if variant != "small_16k"
                ],
                *[
                    {
                        "id": engine_id,
                        "label": label,
                        "ready": self.commercial_engine_ready(model_id),
                        "path": str(self._engine_spec(model_id)["weights"]),
                        "missing": self.commercial_engine_missing(model_id),
                    }
                    for engine_id, label, model_id in (
                        ("acestep", "ACE-Step 1.5 (music, MIT)", ACESTEP_MODEL_ID),
                        ("moss-sfx", "MOSS-SoundEffect v2.0 (sound effects, Apache-2.0)", MOSS_SFX_MODEL_ID),
                    )
                ],
                {
                    "id": "audio-lab",
                    "label": "Audio Lab DSP",
                    "ready": lab_ready,
                    "path": str(lab_status.python_path or root / "engines" / "audio_lab" / ".venv"),
                    "missing": [] if lab_ready else [lab_status.message],
                },
                {
                    "id": "ffmpeg",
                    "label": "FFmpeg audio mux",
                    "ready": mux_ready,
                    "path": str(ffmpeg or ""),
                    "missing": mux_missing,
                },
            ],
        }

    def install_musicgen_variant(self, variant: str) -> dict[str, Any]:
        """Install one allowlisted MusicGen model into Studio's local model tree."""
        self._require_licensed(_MUSICGEN_VARIANTS.get(str(variant or "").strip(), "facebook/musicgen-"))
        normalized = str(variant or "").strip()
        repo_id = _MUSICGEN_VARIANTS.get(normalized)
        if not repo_id:
            raise AudioUnavailable(f"Unsupported MusicGen variant: {normalized or '(empty)'}")
        if not _AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False):
            raise AudioUnavailable("An audio render or model setup operation is already running.")
        if not _MINIMUM_AUDIO_SETUP_LOCK.acquire(blocking=False):
            _AUDIO_MODEL_OPERATION_LOCK.release()
            raise AudioUnavailable("Another audio setup operation is already running.")
        try:
            if self._musicgen_variant_ready(normalized):
                return {
                    "variant": normalized,
                    "modelId": repo_id,
                    "installed": True,
                    "path": str(self._musicgen_root(normalized)),
                }
            try:
                from huggingface_hub import snapshot_download

                destination = self._musicgen_install_root(normalized)
                destination.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(destination),
                    allow_patterns=[*_MUSICGEN_MINIMUM_FILES, "special_tokens_map.json"],
                )
            except Exception as exc:
                raise AudioUnavailable(f"Could not install {repo_id}: {exc}") from exc
            if not self._musicgen_variant_ready(normalized):
                raise AudioUnavailable(f"{repo_id} installation finished, but its required files are incomplete.")
            return {"variant": normalized, "modelId": repo_id, "installed": True, "path": str(destination)}
        finally:
            _MINIMUM_AUDIO_SETUP_LOCK.release()
            _AUDIO_MODEL_OPERATION_LOCK.release()

    def install_mmaudio_variant(self, variant: str) -> dict[str, Any]:
        """Download one allowlisted MMAudio checkpoint into the isolated engine."""
        normalized = str(variant or "").strip()
        self._require_licensed(f"mmaudio:{normalized}")
        if normalized not in _MMAUDIO_INSTALLABLE_VARIANTS:
            raise AudioUnavailable(f"Unsupported MMAudio variant: {normalized or '(empty)'}")
        if not _AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False):
            raise AudioUnavailable("An audio render or model setup operation is already running.")
        if not _MINIMUM_AUDIO_SETUP_LOCK.acquire(blocking=False):
            _AUDIO_MODEL_OPERATION_LOCK.release()
            raise AudioUnavailable("Another audio setup operation is already running.")
        try:
            root = self._mmaudio_root()
            python = self._audio_engine_python()
            if not (root / "demo.py").is_file() or not python.is_file():
                raise AudioUnavailable("Install the minimum Audio setup before adding an MMAudio variant.")
            if self._mmaudio_variant_ready(normalized):
                return {"variant": normalized, "installed": True, "path": str(root)}
            self._install_mmaudio_clip_assets(python)
            self._import_shared_mmaudio_assets(normalized)
            command = (
                "from mmaudio.eval_utils import all_model_cfg; "
                f"all_model_cfg[{normalized!r}].download_if_needed()"
            )
            try:
                result = subprocess.run(
                    [str(python), "-c", command],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=4 * 60 * 60,
                    env=self._mmaudio_install_environment(),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise AudioUnavailable(f"Could not install MMAudio {normalized}: {exc}") from exc
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "MMAudio variant installation failed").strip()
                raise AudioUnavailable(detail[-5000:])
            if not self._mmaudio_variant_ready(normalized):
                raise AudioUnavailable(f"MMAudio {normalized} installation finished, but its required files are incomplete.")
            return {"variant": normalized, "installed": True, "path": str(root)}
        finally:
            _MINIMUM_AUDIO_SETUP_LOCK.release()
            _AUDIO_MODEL_OPERATION_LOCK.release()

    def install_minimum(self) -> dict[str, Any]:
        if not _AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False):
            raise AudioUnavailable("An audio render or model setup operation is already running.")
        if not _MINIMUM_AUDIO_SETUP_LOCK.acquire(blocking=False):
            _AUDIO_MODEL_OPERATION_LOCK.release()
            raise AudioUnavailable("Minimum Audio setup is already running.")
        try:
            root = self.flags.data_dir.resolve()
            script = root / "scripts" / "bootstrap_audio_minimum.py"
            if not script.is_file():
                raise AudioUnavailable(f"Minimum Audio setup script is missing: {script}")
            command = [
                sys.executable,
                str(script),
                "--repo",
                str(root),
                "--models-dir",
                str(self.flags.resolved_models_dir()),
                "--json",
            ]
            # a commercial install never downloads the non-commercial MusicGen/MMAudio weights
            if not self.research_mode():
                command.append("--commercial-only")
            command.extend(
                argument
                for model_root in self.flags.resolved_extra_model_dirs()
                for argument in ("--extra-model-dir", str(model_root))
            )
            result = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=4 * 60 * 60,
                env=self._mmaudio_install_environment(),
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "Minimum Audio setup failed").strip()
                raise AudioUnavailable(detail[-5000:])
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            receipt: dict[str, Any] = {}
            for line in reversed(lines):
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, dict):
                    receipt = decoded
                    break
            status = self.setup_status(deep=True)
            status["receipt"] = receipt
            return status
        finally:
            _MINIMUM_AUDIO_SETUP_LOCK.release()
            _AUDIO_MODEL_OPERATION_LOCK.release()

    def video_audio_status(self) -> str:
        status = self.setup_status(deep=False)
        root = self._mmaudio_root()
        if not self.research_mode():
            return (
                "Commercial-safe mode: MMAudio (CC-BY-NC 4.0) is not used. Turn on research mode in Audio settings "
                "to use it for non-commercial work."
            )
        if status["videoAudioReady"]:
            return (
                f"Video audio ready: MMAudio Small 16 kHz at {root}. "
                "MMAudio checkpoints are CC-BY-NC 4.0; use for non-commercial work unless licensed otherwise."
            )
        return (
            f"Video audio needs the minimum MMAudio setup at {root}. "
            "Use the Download minimum audio models & dependencies button first."
        )

    def output_path(self, *, stem: str = "audio", suffix: str = ".wav") -> Path:
        root = self.flags.resolved_output_dir() / getattr(self.settings, "audio_output_subdir", "audio")
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nonce = uuid.uuid4().hex
        return root / f"{stem}_{stamp}_{nonce}{suffix}"

    def video_output_path(self, input_video: str | Path) -> Path:
        root = self.flags.resolved_output_dir() / getattr(self.settings, "audio_video_output_subdir", "audio-videos")
        root.mkdir(parents=True, exist_ok=True)
        stem = Path(input_video).stem or "video"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nonce = uuid.uuid4().hex
        return root / f"{stem}_audio_{stamp}_{nonce}.mp4"
    @staticmethod
    def _require_output_file(path: Path, label: str) -> None:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                raise AudioUnavailable(f"{label} did not produce a non-empty output file.")
        except OSError as exc:
            raise AudioUnavailable(f"{label} output could not be verified: {exc}") from exc

    @contextmanager
    def _staged_output(self, destination: Path, label: str):
        staged = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.partial{destination.suffix}"
        )
        try:
            yield staged
            self._require_output_file(staged, label)
            try:
                os.replace(staged, destination)
            except OSError as exc:
                raise AudioUnavailable(f"{label} output could not be published: {exc}") from exc
        finally:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove staged audio output %s", staged, exc_info=True)

    def generate(
        self,
        options: AudioGenerationOptions,
        *,
        output_path: str | Path | None = None,
    ) -> AudioGenerationResult:
        if not _AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False):
            raise AudioUnavailable("Audio setup or another audio render is already running. Try again when it finishes.")
        try:
            return self._generate_with_model_lock(options, output_path=output_path)
        finally:
            _AUDIO_MODEL_OPERATION_LOCK.release()

    def prepare(self, *, kind: str, model_id: str) -> dict[str, Any]:
        """Prepare an installed audio model without generating output or installing weights.

        MusicGen exposes an in-process model object, so its residency can be
        confirmed after loading. MMAudio is an isolated per-render subprocess;
        this operation only verifies its exact setup and reports residency as
        unknown rather than claiming that weights were loaded.
        """
        normalized_kind = str(kind or "").strip().lower()
        normalized_model = str(model_id or "").strip()
        if normalized_kind not in {"music", "sfx"}:
            raise AudioUnavailable("Audio kind must be 'music' or 'sfx'.")
        self._require_licensed(normalized_model)
        if normalized_kind == "sfx" and normalized_model.startswith("events:"):
            choices = self.video_audio_model_choices()
        else:
            choices = self.music_model_choices() if normalized_kind == "music" else self.sfx_model_choices()
        if normalized_model not in {model for _, model in choices}:
            raise AudioUnavailable(f"Choose a supported {normalized_kind} model.")
        if normalized_model == "facebook/audiogen-medium":
            raise AudioUnavailable("AudioGen is unavailable because AudioCraft is not installed in the shared Studio runtime.")
        if not _AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False):
            raise AudioUnavailable("Audio setup or another audio render is already running. Try again when it finishes.")
        try:
            if normalized_model.startswith(("acestep:", "moss-sfx:", "events:")):
                missing = self.commercial_engine_missing(normalized_model)
                if missing:
                    raise AudioUnavailable(f"{normalized_model} is not installed completely: {', '.join(missing[:3])}")
                with self._gpu_tenant("Prepare commercial audio engine"):
                    self._release_image_models()
                    self.unload()
                return {"kind": normalized_kind, "modelId": normalized_model, "ready": True, "resident": None,
                        "detail": "Installed and ready; this engine loads its weights per render, so residency is unreported."}
            if normalized_model.startswith("facebook/musicgen-"):
                variant = normalized_model.removeprefix("facebook/musicgen-")
                setup = self.setup_status(deep=False)
                if not self._musicgen_variant_ready(variant):
                    raise AudioUnavailable(f"{normalized_model} is not installed. Install it from the Audio model setup panel first.")
                if not setup.get("musicDependenciesReady", setup.get("musicReady", False)):
                    raise AudioUnavailable("Install the minimum Audio setup before preparing MusicGen runtime dependencies.")
                with self._gpu_tenant("Audio model preparation"):
                    self._release_image_models()
                    headroom_issue = self._audio_headroom_issue(normalized_model)
                    if headroom_issue:
                        raise AudioUnavailable(headroom_issue)
                    source = self._musicgen_model_source(normalized_model)
                    self._load_transformers_musicgen(normalized_model, source)
                    resident = self._musicgen_model_is_resident(normalized_model)
                    if not resident:
                        raise AudioUnavailable("MusicGen loader returned without confirming the selected model is resident.")
                    return {"kind": normalized_kind, "modelId": normalized_model, "ready": True, "resident": True}

            variant = normalized_model.split(":", 1)[1]
            if not self._mmaudio_variant_ready(variant):
                raise AudioUnavailable(f"MMAudio {variant} is not installed completely. Install this variant in Audio Studio first.")
            import_error = self._mmaudio_runtime_import_error()
            if import_error:
                raise AudioUnavailable(f"MMAudio runtime is not ready: {import_error}")
            with self._gpu_tenant("Prepare MMAudio and release prior in-process models"):
                self._release_image_models()
                self.unload()
            return {"kind": normalized_kind, "modelId": normalized_model, "ready": True, "resident": None,
                    "detail": "MMAudio is installed and ready; its isolated runtime loads weights per render, so residency is unreported."}
        finally:
            _AUDIO_MODEL_OPERATION_LOCK.release()

    def _generate_with_model_lock(
        self,
        options: AudioGenerationOptions,
        *,
        output_path: str | Path | None = None,
    ) -> AudioGenerationResult:
        prompt = (options.prompt or "").strip()
        if not prompt:
            raise AudioUnavailable("Enter an audio prompt first.")
        if str(options.kind or "").lower() == "video_audio":
            raise AudioUnavailable("Video-conditioned audio needs a target video.")
        self._require_licensed(options.model_id)
        generation_options = options
        if str(options.model_id).startswith("acestep:"):
            generation_options = options.model_copy(
                update={
                    "duration_seconds": max(
                        ACESTEP_MIN_DURATION_SECONDS,
                        float(options.duration_seconds),
                    )
                }
            )
        mmaudio_text = options.kind == "sfx" and str(options.model_id or "").startswith("mmaudio:")
        suffix = ".flac" if mmaudio_text else ".wav"
        dest = Path(output_path) if output_path else self.output_path(stem=self._safe_stem(prompt), suffix=suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with self._staged_output(dest, "Audio generation") as staged_dest:
            with self._gpu_tenant("Audio generation"):
                self._release_image_models()
                if options.seed is not None and int(options.seed) >= 0:
                    self._set_seed(int(options.seed))
                try:
                    if str(options.model_id).startswith("acestep:"):
                        sample_rate = self._generate_acestep(generation_options, staged_dest)
                    elif str(options.model_id).startswith("moss-sfx:"):
                        sample_rate = self._generate_moss_sfx(options, staged_dest)
                    elif mmaudio_text:
                        sample_rate = self._generate_mmaudio_text_audio(options, staged_dest)
                    elif options.kind == "sfx":
                        sample_rate = self._generate_audiocraft(options, staged_dest)
                    else:
                        sample_rate = self._generate_transformers_musicgen(options, staged_dest)
                finally:
                    self._park_cached_model_on_cpu()

        infotext = (
            f"Audio {options.kind}: {options.model_id}, {generation_options.duration_seconds:.1f}s, "
            f"licence: {audio_licenses.short_label(options.model_id)}"
        )
        return AudioGenerationResult(
            output_path=str(dest),
            prompt=prompt,
            model_id=options.model_id,
            kind=options.kind,
            duration_seconds=float(generation_options.duration_seconds),
            sample_rate=sample_rate,
            message=f"Saved {generation_options.duration_seconds:.1f}s audio -> {dest}",
            infotext=infotext,
            license=audio_licenses.license_for(options.model_id),
        )

    def generate_for_video(
        self,
        video_path: str | Path,
        options: AudioGenerationOptions,
        *,
        duration_seconds: float | None = None,
    ) -> AudioGenerationResult:
        if duration_seconds is None or duration_seconds <= 0:
            info = VideoProcessor().probe(video_path)
            duration_seconds = info.duration_seconds or options.duration_seconds
        safe_duration = min(120.0, max(1.0, float(duration_seconds)))
        next_options = options.model_copy(update={"duration_seconds": safe_duration})
        if str(next_options.kind or "").lower() == "video_audio":
            return self.generate_video_audio(video_path, next_options)
        return self.generate(next_options)

    def generate_video_audio(
        self,
        video_path: str | Path,
        options: AudioGenerationOptions,
        *,
        output_path: str | Path | None = None,
    ) -> AudioGenerationResult:
        if not _AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False):
            raise AudioUnavailable("Audio setup or another audio render is already running. Try again when it finishes.")
        try:
            return self._generate_video_audio_with_model_lock(video_path, options, output_path=output_path)
        finally:
            _AUDIO_MODEL_OPERATION_LOCK.release()

    def _generate_video_audio_with_model_lock(
        self,
        video_path: str | Path,
        options: AudioGenerationOptions,
        *,
        output_path: str | Path | None = None,
    ) -> AudioGenerationResult:
        prompt = (options.prompt or "").strip()
        # the event soundtrack works from the scene itself; its prompt is only an optional hint
        if not prompt and not str(options.model_id).startswith("events:"):
            raise AudioUnavailable("Enter an audio prompt first.")
        src_video = Path(video_path)
        if not src_video.is_file():
            raise AudioUnavailable(f"Video not found: {src_video}")
        self._require_licensed(options.model_id)
        stem = f"{src_video.stem}_{self._safe_stem(prompt)}"
        # the event soundtrack is mixed here as a float WAV; MMAudio's worker writes FLAC
        dest_suffix = ".wav" if str(options.model_id).startswith("events:") else ".flac"
        dest = Path(output_path) if output_path else self.output_path(stem=stem, suffix=dest_suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with self._staged_output(dest, "Video audio generation") as staged_dest:
            with self._gpu_tenant("Video audio generation"):
                self._release_image_models()
                if str(options.model_id).startswith("events:"):
                    sample_rate = self._generate_event_soundtrack(src_video, options, staged_dest)
                else:
                    sample_rate = self._generate_mmaudio_video_audio(src_video, options, staged_dest)

        infotext = (
            f"Video audio {options.model_id}: {options.duration_seconds:.1f}s, "
            f"steps {int(options.steps)}, CFG {float(options.cfg_coef):.2f}, "
            f"licence: {audio_licenses.short_label(options.model_id)}"
        )
        return AudioGenerationResult(
            output_path=str(dest),
            prompt=prompt,
            model_id=options.model_id,
            kind="video_audio",
            duration_seconds=float(options.duration_seconds),
            sample_rate=sample_rate,
            message=f"Saved video-conditioned audio -> {dest}",
            infotext=infotext,
            license=audio_licenses.license_for(options.model_id),
        )

    @staticmethod
    def _validate_mux_container(path: Path, ffprobe: str) -> None:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=format_name,duration:stream=codec_type",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise AudioUnavailable("Audio mux output is not a readable media container.")
            probe_data = json.loads(result.stdout or "{}")
            if not isinstance(probe_data, dict):
                raise AudioUnavailable("Audio mux container validation returned malformed data.")
            container = probe_data.get("format")
            streams = probe_data.get("streams")
            if not isinstance(container, dict) or not isinstance(streams, list):
                raise AudioUnavailable("Audio mux container validation returned malformed data.")
            if any(not isinstance(stream, dict) for stream in streams):
                raise AudioUnavailable("Audio mux container validation returned malformed streams.")
            format_name = str(container.get("format_name") or "").strip()
            try:
                duration = float(container.get("duration"))
            except (TypeError, ValueError) as exc:
                raise AudioUnavailable("Audio mux output must report a positive finite duration.") from exc
            if not math.isfinite(duration) or duration <= 0:
                raise AudioUnavailable("Audio mux output must report a positive finite duration.")
            stream_types = {str(item.get("codec_type") or "") for item in streams}
            if not format_name or not {"video", "audio"}.issubset(stream_types):
                raise AudioUnavailable("Audio mux output must contain a readable container with video and audio streams.")
        except AudioUnavailable:
            raise
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            raise AudioUnavailable(f"Audio mux container could not be validated: {exc}") from exc

    def mux_audio(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> AudioMuxResult:
        ffmpeg = _resolve_ffmpeg()
        if ffmpeg is None:
            raise AudioUnavailable("ffmpeg is required to mux generated audio into video.")
        src_video = Path(video_path)
        src_audio = Path(audio_path)
        if not src_video.is_file():
            raise AudioUnavailable(f"Video not found: {src_video}")
        if not src_audio.is_file():
            raise AudioUnavailable(f"Audio not found: {src_audio}")
        ffprobe = self._resolve_ffprobe(ffmpeg)
        if ffprobe is None:
            raise AudioUnavailable("ffprobe is required to validate the muxed video container.")
        dest = Path(output_path) if output_path else self.video_output_path(src_video)
        dest.parent.mkdir(parents=True, exist_ok=True)
        staged = dest.with_name(f".{dest.stem}.{uuid.uuid4().hex}.partial{dest.suffix or '.mp4'}")
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(src_video),
            "-i",
            str(src_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-af",
            "apad",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(staged),
        ]
        try:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            except subprocess.TimeoutExpired as exc:
                raise AudioUnavailable("Audio mux exceeded its 60-minute time limit.") from exc
            except OSError as exc:
                raise AudioUnavailable(f"Audio mux could not start ffmpeg: {exc}") from exc
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                raise AudioUnavailable(f"Audio mux failed: {detail}")
            self._require_output_file(staged, "Audio mux")
            self._validate_mux_container(staged, ffprobe)
            try:
                os.replace(staged, dest)
            except OSError as exc:
                raise AudioUnavailable(f"Audio mux output could not be published: {exc}") from exc
        finally:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove staged audio mux output %s", staged, exc_info=True)
        return AudioMuxResult.saved(src_video, src_audio, dest)

    def generate_and_mux(
        self,
        video_path: str | Path,
        options: AudioGenerationOptions,
        *,
        duration_seconds: float | None = None,
    ) -> tuple[AudioGenerationResult, AudioMuxResult]:
        audio = self.generate_for_video(video_path, options, duration_seconds=duration_seconds)
        muxed = self.mux_audio(video_path, audio.output_path)
        return audio, muxed

    def unload(self) -> None:
        self._model = None
        self._model_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def release_cached_model_for_modality_switch(self) -> bool:
        """Drop the in-process audio cache only while audio owns the model lock and GPU tenant.

        The method is intentionally nonblocking: an active render/setup or another GPU
        tenant keeps ownership, so the caller must defer its route switch or let the
        normal tenant scheduler serialize the work.
        """
        if self._model is None:
            return True
        if self.supervisor is None or not callable(getattr(self.supervisor, "tenant_session", None)):
            return False
        if not _AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False):
            return False
        try:
            with self.supervisor.tenant_session(
                EngineTenant.AUDIO,
                reason="Release cached audio model before switching modalities",
            ):
                self.unload()
            return self._model is None and self._model_key is None
        except Exception:
            logger.info("Could not release cached audio model for modality switch", exc_info=True)
            return False
        finally:
            _AUDIO_MODEL_OPERATION_LOCK.release()

    def _release_image_models(self) -> None:
        """Free the shared image backend before audio work claims GPU memory."""
        if not callable(self.unload_image_models):
            return
        try:
            self.unload_image_models()
        except Exception as exc:
            logger.exception("Could not unload the image backend before audio work")
            raise AudioUnavailable(f"Could not release the image model before audio work: {exc}") from exc

    def _generate_audiocraft(self, options: AudioGenerationOptions, dest: Path) -> int:
        raise AudioUnavailable(
            "AudioCraft generation has no explicit local model installer in Studio. "
            "Choose an installed MusicGen or MMAudio model so generation does not download weights implicitly."
        )

    def _load_audiocraft_model(self, model_cls, model_id: str, kind: str):
        del model_cls, model_id, kind
        raise AudioUnavailable(
            "AudioCraft model loading is disabled until Studio provides an explicit local installer and readiness check."
        )

    def _generate_transformers_musicgen(self, options: AudioGenerationOptions, dest: Path) -> int:
        try:
            import scipy.io.wavfile
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
        except Exception as exc:
            raise AudioUnavailable(
                "MusicGen needs `transformers`, `scipy`, and `torch`. Run the minimum Audio setup first."
            ) from exc

        if float(options.duration_seconds) > 30.0:
            raise AudioUnavailable(
                "The minimum MusicGen backend supports clips up to 30 seconds per render. "
                "Use Audio Lab to arrange or crossfade longer pieces."
            )
        model_id = options.model_id or "facebook/musicgen-small"
        headroom_issue = self._audio_headroom_issue(
            model_id,
            model_resident=self._musicgen_model_is_resident(model_id),
        )
        if headroom_issue:
            raise AudioUnavailable(headroom_issue)
        model_source = self._musicgen_model_source(model_id)
        processor, model = self._load_transformers_musicgen(model_id, model_source)
        device = self._device_string()
        if str(model.device) != device:
            model.to(device)
        inputs = processor(text=[options.prompt], padding=True, return_tensors="pt").to(model.device)
        token_rate = 50
        max_new_tokens = max(16, int(float(options.duration_seconds) * token_rate))
        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=float(options.cfg_coef),
                temperature=float(options.temperature),
                top_k=int(options.top_k),
                max_new_tokens=max_new_tokens,
            )
        sample_rate = int(model.config.audio_encoder.sampling_rate)
        audio = audio_values[0].detach().cpu().float().numpy()
        if audio.ndim == 2:
            audio = np.swapaxes(audio, 0, 1)
        audio = np.asarray(audio, dtype=np.float32)
        scipy.io.wavfile.write(str(dest), sample_rate, audio)
        return sample_rate

    def _load_transformers_musicgen(self, model_id: str, model_source: str):
        try:
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
        except Exception as exc:
            raise AudioUnavailable(
                "MusicGen needs `transformers` and `torch`. Run the minimum Audio setup first."
            ) from exc
        key = self._musicgen_cache_key(model_id, model_source)
        if self._model is None or self._model_key != key:
            previous_model = self._model
            previous_key = self._model_key
            if previous_model is not None:
                self._park_cached_model_on_cpu()
            try:
                processor = AutoProcessor.from_pretrained(model_source, local_files_only=True)
                device = self._device_string()
                load_options: dict[str, Any] = {"low_cpu_mem_usage": True, "use_safetensors": True}
                if device == "cuda":
                    load_options.update(dtype=torch.float16, attn_implementation="sdpa")
                    torch.backends.cuda.matmul.allow_tf32 = True
                model = MusicgenForConditionalGeneration.from_pretrained(model_source, local_files_only=True, **load_options)
                model.to(device)
            except Exception:
                # Keep the previous cache available on CPU if switching fails.
                # Its key remains accurate, so readiness for the requested model
                # cannot be reported as resident.
                self._model = previous_model
                self._model_key = previous_key
                raise
            self._model = (processor, model)
            self._model_key = key
        else:
            # Generation parks the cached weights on CPU after every render.
            # Re-entering prepare for the same model must put them back on the
            # selected device before the caller can report residency.
            model = self._model[1]
            device = self._device_string()
            if str(getattr(model, "device", "")) != device:
                model.to(device)
        return self._model

    def _musicgen_model_is_resident(self, model_id: str) -> bool:
        cached = self._model
        expected_key = self._musicgen_current_cache_key(model_id)
        if self._model_key != expected_key or not isinstance(cached, tuple) or len(cached) != 2:
            return False
        model = cached[1]
        if model is None:
            return False
        device = self._device_string()
        try:
            return str(model.device) == device
        except Exception:
            return False

    def musicgen_model_is_parked_on_cpu(self, model_id: str) -> bool:
        """Confirm that this selected MusicGen cache remains available on CPU."""
        cached = self._model
        expected_key = self._musicgen_current_cache_key(model_id)
        if self._model_key != expected_key or not isinstance(cached, tuple) or len(cached) != 2:
            return False
        model = cached[1]
        if model is None:
            return False
        try:
            return str(model.device).strip().lower() == "cpu"
        except Exception:
            return False

    @staticmethod
    def _musicgen_cache_key(model_id: str, model_source: str) -> tuple[Any, ...]:
        source_path = Path(str(model_source)).expanduser()
        if not source_path.exists():
            return ("transformers", "music", model_id)
        return ("transformers", "music", model_id, support_revision([str(source_path)]))

    def _musicgen_current_cache_key(self, model_id: str) -> tuple[Any, ...]:
        try:
            model_source = self._musicgen_model_source(model_id)
        except AudioUnavailable:
            return ("transformers", "music", model_id)
        return self._musicgen_cache_key(model_id, model_source)

    def _generate_mmaudio_text_audio(self, options: AudioGenerationOptions, dest: Path) -> int:
        root = self._mmaudio_root()
        demo = root / "demo.py"
        python = self._audio_engine_python()
        if not demo.is_file() or not python.is_file():
            raise AudioUnavailable("MMAudio is not installed. Run the minimum Audio setup first.")
        variant = self._mmaudio_variant(options.model_id)
        if variant not in _MMAUDIO_INSTALLABLE_VARIANTS:
            raise AudioUnavailable(f"Unsupported MMAudio variant: {variant}")
        if not self._mmaudio_variant_ready(variant):
            raise AudioUnavailable(f"MMAudio {variant} is not installed completely. Install this variant in Audio Studio first.")
        import_error = self._mmaudio_runtime_import_error()
        if import_error:
            raise AudioUnavailable(f"MMAudio runtime is not ready: {import_error}")
        headroom_issue = self._audio_headroom_issue(options.model_id, external_worker=True)
        if headroom_issue:
            raise AudioUnavailable(headroom_issue)
        run_dir = dest.parent / f"{dest.stem}_mmaudio_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)
        seed = int(options.seed) if options.seed is not None and int(options.seed) >= 0 else random.randint(0, 2**31 - 1)
        command = [
            str(python),
            str(self.flags.data_dir.resolve() / "scripts" / "run_mmaudio_offline.py"),
            str(demo),
            "--variant",
            variant,
            "--prompt",
            options.prompt,
            "--negative_prompt",
            options.negative_prompt or "",
            "--duration",
            f"{float(options.duration_seconds):.3f}",
            "--cfg_strength",
            f"{float(options.cfg_coef):.3f}",
            "--num_steps",
            str(int(options.steps)),
            "--seed",
            str(seed),
            "--output",
            str(run_dir),
            "--skip_video_composite",
        ]
        result = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=3600,
            env=self._mmaudio_offline_environment(),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise AudioUnavailable(f"MMAudio sound generation failed: {detail}")
        source = self._find_mmaudio_audio_output(run_dir, expected_stem=None)
        if source.resolve() != dest.resolve():
            if dest.exists():
                dest.unlink()
            shutil.move(str(source), str(dest))
        return 44100 if "44k" in variant else 16000

    def _generate_mmaudio_video_audio(self, video_path: Path, options: AudioGenerationOptions, dest: Path) -> int:
        # A finished video conditions audio generation through the isolated MMAudio bridge.
        # The subprocess exits after each render, so its CUDA allocation cannot leak into
        # the next image or video job.
        root = self._mmaudio_root()
        demo = root / "demo.py"
        python = self._audio_engine_python()
        if not demo.is_file():
            raise AudioUnavailable(f"MMAudio demo.py not found. Install MMAudio at {root}.")
        if not python.is_file():
            raise AudioUnavailable(f"MMAudio engine Python not found: {python}")

        variant = self._mmaudio_variant(options.model_id)
        if variant not in _MMAUDIO_INSTALLABLE_VARIANTS:
            raise AudioUnavailable(f"Unsupported MMAudio variant: {variant}")
        if not self._mmaudio_variant_ready(variant):
            raise AudioUnavailable(f"MMAudio {variant} is not installed completely. Install this variant in Audio Studio first.")
        import_error = self._mmaudio_runtime_import_error()
        if import_error:
            raise AudioUnavailable(f"MMAudio runtime is not ready: {import_error}")
        headroom_issue = self._audio_headroom_issue(options.model_id, external_worker=True)
        if headroom_issue:
            raise AudioUnavailable(headroom_issue)
        run_dir = dest.parent / f"{dest.stem}_mmaudio_{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)
        seed = int(options.seed) if options.seed is not None and int(options.seed) >= 0 else random.randint(0, 2**31 - 1)
        command = [
            str(python),
            str(self.flags.data_dir.resolve() / "scripts" / "run_mmaudio_offline.py"),
            str(demo),
            "--variant",
            variant,
            "--video",
            str(video_path),
            "--prompt",
            options.prompt,
            "--negative_prompt",
            options.negative_prompt or "",
            "--duration",
            f"{float(options.duration_seconds):.3f}",
            "--cfg_strength",
            f"{float(options.cfg_coef):.3f}",
            "--num_steps",
            str(int(options.steps)),
            "--seed",
            str(seed),
            "--output",
            str(run_dir),
            "--skip_video_composite",
        ]
        result = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=3600,
            env=self._mmaudio_offline_environment(),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise AudioUnavailable(f"MMAudio video audio failed: {detail}")
        source = self._find_mmaudio_audio_output(run_dir, expected_stem=video_path.stem)
        if source.resolve() != dest.resolve():
            if dest.exists():
                dest.unlink()
            shutil.move(str(source), str(dest))
        return 44100 if "44k" in variant else 16000

    @staticmethod
    def _find_mmaudio_audio_output(run_dir: Path, *, expected_stem: str | None) -> Path:
        expected = run_dir / f"{expected_stem}.flac" if expected_stem else None
        if expected is not None and expected.is_file() and expected.stat().st_size > 0:
            return expected
        candidates = sorted(
            (
                path
                for path in run_dir.glob("*.flac")
                if path.is_file() and path.stat().st_size > 0
            ),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        if len(candidates) == 1:
            return candidates[0]
        found = ", ".join(path.name for path in candidates[:5]) or "none"
        expected_text = str(expected) if expected is not None else "one generated .flac file"
        raise AudioUnavailable(
            f"MMAudio did not create expected audio: {expected_text}. "
            f"Found {len(candidates)} .flac file(s): {found}"
        )

    def _minimum_musicgen_root(self) -> Path:
        return self._musicgen_root("small")

    def _musicgen_root(self, variant: str) -> Path:
        if variant not in _MUSICGEN_VARIANTS:
            raise AudioUnavailable(f"Unsupported MusicGen variant: {variant or '(empty)'}")
        relative_candidates = (
            Path("audio") / "MusicGen" / f"musicgen-{variant}",
            Path("MusicGen") / f"musicgen-{variant}",
        )
        for root in configured_model_roots(self.flags):
            for relative in relative_candidates:
                candidate = root / relative
                try:
                    resolved = candidate.resolve(strict=False)
                    resolved.relative_to(root)
                except (OSError, RuntimeError, ValueError):
                    continue
                if self._musicgen_files_ready(resolved):
                    return resolved
        return self._musicgen_install_root(variant)

    def _musicgen_install_root(self, variant: str) -> Path:
        """Return Studio's writable destination, never a configured shared root."""
        if variant not in _MUSICGEN_VARIANTS:
            raise AudioUnavailable(f"Unsupported MusicGen variant: {variant or '(empty)'}")
        return self.flags.resolved_models_dir() / "audio" / "MusicGen" / f"musicgen-{variant}"

    @staticmethod
    def _musicgen_files_ready(root: Path) -> bool:
        return all(
            _is_nonempty_file(root / name) if name != "model.safetensors" else _is_complete_safetensors(root / name)
            for name in _MUSICGEN_MINIMUM_FILES
        )

    def _musicgen_variant_ready(self, variant: str) -> bool:
        if variant not in _MUSICGEN_VARIANTS:
            return False
        return self._musicgen_files_ready(self._musicgen_root(variant))

    def _musicgen_missing_files(self, variant: str) -> list[str]:
        if variant not in _MUSICGEN_VARIANTS:
            return ["unsupported MusicGen variant"]
        root = self._musicgen_root(variant)
        missing = []
        for name in _MUSICGEN_MINIMUM_FILES:
            path = root / name
            valid = _is_complete_safetensors(path) if name == "model.safetensors" else _is_nonempty_file(path)
            if not valid:
                missing.append(name if name != "model.safetensors" or not path.exists() else "model.safetensors (invalid or truncated)")
        return missing

    def model_support_paths(self, model_id: str) -> list[str]:
        """Return exact variant files used to fingerprint setup lifecycle state.

        Include expected paths even when files are absent so installation or a
        replacement invalidates an earlier readiness receipt. No weights are
        opened or loaded here.
        """
        normalized = str(model_id or "").strip()
        variant = next((key for key, repo_id in _MUSICGEN_VARIANTS.items() if repo_id == normalized), None)
        if variant is not None:
            root = self._musicgen_root(variant)
            return [str(root / name) for name in _MUSICGEN_MINIMUM_FILES]
        if normalized.startswith("mmaudio:"):
            variant = normalized.split(":", 1)[1]
            if variant not in _MMAUDIO_INSTALLABLE_VARIANTS:
                return []
            root = self._mmaudio_root()
            cache = self._mmaudio_clip_hub_cache() or self._mmaudio_clip_install_cache()
            snapshots = self._mmaudio_clip_snapshots(cache)
            if not snapshots:
                snapshots = [cache / _MMAUDIO_CLIP_REPO_CACHE / "snapshots" / "main"]
            cache_files = [
                str(snapshot / filename)
                for snapshot in snapshots
                for filename in (*_MMAUDIO_CLIP_REQUIRED_FILES, *_MMAUDIO_CLIP_WEIGHT_FILES)
            ]
            return [str(root / relative) for relative in _mmaudio_variant_files(variant)] + cache_files
        return []

    def _musicgen_model_source(self, model_id: str) -> str:
        variant = next((key for key, repo_id in _MUSICGEN_VARIANTS.items() if repo_id == model_id), None)
        if variant is None:
            raise AudioUnavailable(f"Unsupported MusicGen model: {model_id}")
        root = self._musicgen_root(variant)
        if not self._musicgen_variant_ready(variant):
            raise AudioUnavailable(f"{model_id} is not installed. Install it from the Audio model setup panel first.")
        return str(root)

    def _mmaudio_variant_ready(self, variant: str) -> bool:
        if variant not in _MMAUDIO_INSTALLABLE_VARIANTS:
            return False
        root = self._mmaudio_root()
        if not (root / "demo.py").is_file() or not self._audio_engine_python().is_file():
            return False
        required = _mmaudio_variant_files(variant)
        return all(_is_nonempty_file(root / path) for path in required) and bool(self._mmaudio_clip_hub_cache())

    def _mmaudio_clip_snapshots(self, cache_root: Path) -> list[Path]:
        snapshots_root = cache_root / _MMAUDIO_CLIP_REPO_CACHE / "snapshots"
        try:
            resolved_cache = cache_root.resolve(strict=True)
            resolved_snapshots = snapshots_root.resolve(strict=True)
            resolved_snapshots.relative_to(resolved_cache)
            return [
                resolved
                for item in resolved_snapshots.iterdir()
                if self._confined_path(item, resolved_cache) is not None
                for resolved in [item.resolve(strict=True)]
            ]
        except (OSError, RuntimeError, ValueError):
            return []

    @staticmethod
    def _confined_path(path: Path, root: Path) -> Path | None:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root.resolve(strict=True))
            return resolved
        except (OSError, RuntimeError, ValueError):
            return None

    def _mmaudio_clip_hub_cache(self) -> Path | None:
        return _find_mmaudio_clip_hub_cache(configured_model_roots(self.flags))

    def _mmaudio_clip_install_cache(self) -> Path:
        roots = configured_model_roots(self.flags)
        if not roots:
            raise AudioUnavailable("No configured model root is available for the MMAudio CLIP cache.")
        root = roots[0]
        candidate = root / "hub"
        try:
            candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        except (OSError, RuntimeError, ValueError) as exc:
            raise AudioUnavailable(f"Unsafe MMAudio CLIP cache destination: {candidate}") from exc
        return candidate

    def _mmaudio_install_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["HF_HUB_CACHE"] = str(self._mmaudio_clip_hub_cache() or self._mmaudio_clip_install_cache())
        env["HF_HUB_OFFLINE"] = "0"
        return env

    def _mmaudio_offline_environment(self) -> dict[str, str]:
        cache = self._mmaudio_clip_hub_cache()
        if cache is None:
            raise AudioUnavailable(f"MMAudio CLIP encoder is missing from configured model roots ({_MMAUDIO_CLIP_REPO}).")
        env = os.environ.copy()
        env["HF_HUB_CACHE"] = str(cache)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    def _install_mmaudio_clip_assets(self, python: Path) -> Path:
        existing = self._mmaudio_clip_hub_cache()
        if existing is not None:
            return existing
        cache = self._mmaudio_clip_install_cache()
        program = (
            "import sys; from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={_MMAUDIO_CLIP_REPO!r}, cache_dir=sys.argv[1], "
            f"allow_patterns={list(_MMAUDIO_CLIP_ALLOW_PATTERNS)!r})"
        )
        try:
            result = subprocess.run(
                [str(python), "-c", program, str(cache)],
                cwd=str(self._mmaudio_root()),
                capture_output=True,
                text=True,
                timeout=4 * 60 * 60,
                env=self._mmaudio_install_environment(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AudioUnavailable(f"Could not install MMAudio CLIP encoder {_MMAUDIO_CLIP_REPO}: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "MMAudio CLIP encoder installation failed").strip()
            raise AudioUnavailable(detail[-5000:])
        installed = self._mmaudio_clip_hub_cache()
        if installed is None:
            raise AudioUnavailable(f"MMAudio CLIP encoder {_MMAUDIO_CLIP_REPO} setup finished, but its config or weights are missing.")
        return installed

    def _mmaudio_runtime_import_error(self) -> str:
        """Probe the isolated MMAudio environment before claiming setup-ready."""
        root = self._mmaudio_root()
        python = self._audio_engine_python()
        if not python.is_file():
            return f"Audio engine Python is missing: {python}"
        if not (root / "demo.py").is_file():
            return f"MMAudio entrypoint is missing: {root / 'demo.py'}"
        try:
            result = subprocess.run(
                [str(python), str(root / "demo.py"), "--help"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=120,
                env=self._mmaudio_offline_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return str(exc) or "MMAudio import check failed."
        if result.returncode == 0:
            return ""
        return (result.stderr or result.stdout or "MMAudio demo import check failed").strip()[-1200:]

    def _mmaudio_shared_asset_sources(self, variant: str) -> dict[Path, Path]:
        """Find variant assets under configured roots without following escapes.

        The upstream demo resolves all checkpoint paths relative to its working
        directory and may overwrite files whose checksums differ. Shared roots
        therefore serve as read-only import sources; assets are copied into the
        Studio-owned engine directory before the demo is allowed to use them.
        """
        required = _mmaudio_variant_files(variant)
        if not required:
            return {}
        found: dict[Path, Path] = {}
        for root in configured_model_roots(self.flags):
            for relative in required:
                if relative in found:
                    continue
                for layout in _MMAUDIO_SHARED_LAYOUTS:
                    candidate = root / layout / relative
                    try:
                        resolved = candidate.resolve(strict=True)
                        resolved.relative_to(root)
                        if _is_nonempty_file(resolved):
                            found[relative] = resolved
                            break
                    except (OSError, RuntimeError, ValueError):
                        continue
        return found

    def _mmaudio_variant_shared_ready(self, variant: str) -> bool:
        required = _mmaudio_variant_files(variant)
        found = self._mmaudio_shared_asset_sources(variant)
        return bool(required) and all(path in found for path in required)

    def _import_shared_mmaudio_assets(self, variant: str) -> list[str]:
        """Copy discovered shared weights into Studio-owned paths, never write shared roots."""
        root = self._mmaudio_root().resolve()
        for shared_root in configured_model_roots(self.flags):
            try:
                root.relative_to(shared_root.resolve())
            except (OSError, RuntimeError, ValueError):
                continue
            raise AudioUnavailable(
                f"MMAudio install directory is inside a configured read-only shared model root: {root}"
            )
        sources = self._mmaudio_shared_asset_sources(variant)
        copied: list[str] = []
        for relative, source in sources.items():
            destination = root / relative
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                resolved_destination = destination.resolve(strict=False)
                resolved_destination.relative_to(root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise AudioUnavailable(f"Unsafe MMAudio install destination: {destination}") from exc
            if _is_nonempty_file(resolved_destination):
                continue
            staged = resolved_destination.with_name(f".{resolved_destination.name}.{uuid.uuid4().hex}.partial")
            try:
                shutil.copyfile(source, staged)
                os.replace(staged, resolved_destination)
            except OSError as exc:
                raise AudioUnavailable(f"Could not import shared MMAudio asset {relative}: {exc}") from exc
            finally:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove partial MMAudio import %s", staged, exc_info=True)
            copied.append(str(relative))
        return copied

    def _park_cached_model_on_cpu(self) -> None:
        cached = self._model
        model = cached[1] if isinstance(cached, tuple) and len(cached) == 2 else cached
        if model is None or not hasattr(model, "to"):
            return
        try:
            model.to("cpu")
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Could not park the cached audio model on CPU", exc_info=True)

    def _mmaudio_root(self) -> Path:
        return self.flags.data_dir.resolve() / "engines" / "audio" / "MMAudio"

    def _audio_engine_python(self) -> Path:
        if os.name == "nt":
            return self.flags.data_dir.resolve() / "engines" / "audio" / ".venv" / "Scripts" / "python.exe"
        return self.flags.data_dir.resolve() / "engines" / "audio" / ".venv" / "bin" / "python"

    @staticmethod
    def _mmaudio_variant(model_id: str) -> str:
        text = str(model_id or "").strip()
        if text.startswith("mmaudio:"):
            return text.split(":", 1)[1] or "large_44k_v2"
        return text or "large_44k_v2"

    def _device_string(self) -> str:
        if self.devices is not None:
            try:
                return str(self.devices.device())
            except Exception:
                pass
        try:
            import torch

            return "cuda" if torch.cuda.is_available() and not self.flags.cpu else "cpu"
        except Exception:
            return "cpu"

    def _audio_headroom_issue(
        self,
        model_id: str,
        *,
        model_resident: bool = False,
        external_worker: bool = False,
    ) -> str | None:
        """Refuse an audio model load when device-wide VRAM is not safely available."""
        device_name = self._device_string().strip().lower()
        if not external_worker and not device_name.startswith("cuda"):
            return None
        try:
            from aiwf.services.gpu_memory import measured_cuda_free_bytes

            if external_worker:
                from aiwf.services.gpu_memory import nvidia_smi_free_bytes

                free_bytes = nvidia_smi_free_bytes()
            else:
                import torch

                if not torch.cuda.is_available():
                    return "Audio model loading deferred because CUDA is unavailable to PyTorch."
                device = torch.device(device_name)
                free_bytes = measured_cuda_free_bytes(torch, device)
        except Exception:
            free_bytes = None
        if free_bytes is None:
            return "Audio model loading deferred because available GPU memory could not be verified."

        normalized = str(model_id or "").strip().lower()
        if normalized.startswith("facebook/musicgen-"):
            required_gb = 2.0 if model_resident else (5.0 if normalized != "facebook/musicgen-small" else 3.5)
        else:
            variant = self._mmaudio_variant(normalized)
            required_gb = {
                "small_16k": 6.0,
                "small_44k": 8.0,
                "medium_44k": 10.0,
                "large_44k": 12.0,
                "large_44k_v2": 12.0,
            }.get(variant, 8.0)
        free_gb = float(free_bytes) / (1024**3)
        if free_gb < required_gb:
            return (
                f"Audio model loading deferred: {free_gb:.1f} GB VRAM is free; "
                f"the selected route needs at least {required_gb:.1f} GB headroom."
            )
        return None

    @staticmethod
    def _set_seed(seed: int) -> None:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    @staticmethod
    def _safe_stem(prompt: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in prompt.strip().lower())
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return (cleaned[:48] or "audio").strip("_") or "audio"
