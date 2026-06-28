from __future__ import annotations

import gc
import logging
import os
import random
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions, AudioGenerationResult, AudioMuxResult
from aiwf.core.domain.engine import EngineTenant
from aiwf.infrastructure.video.processing import VideoProcessor, _resolve_ffmpeg

logger = logging.getLogger(__name__)


class AudioUnavailable(RuntimeError):
    """Raised when optional audio generation dependencies or tools are missing."""


class AudioGenerationService:
    """Optional local text-to-audio, video-conditioned audio, and video muxing."""

    def __init__(self, flags: RuntimeFlags, settings: UserSettings, devices=None, supervisor=None) -> None:
        self.flags = flags
        self.settings = settings
        self.devices = devices
        self.supervisor = supervisor
        self._model: Any | None = None
        self._model_key: tuple[str, str, str] | None = None

    @contextmanager
    def _gpu_tenant(self, reason: str):
        if self.supervisor is None:
            yield
            return
        try:
            with self.supervisor.tenant_session(EngineTenant.VIDEO, reason=reason):
                yield
        except RuntimeError as exc:
            raise AudioUnavailable(f"GPU busy: {exc}") from exc

    def folder_help(self) -> str:
        return (
            "Audio generation is optional. Video-conditioned audio uses an isolated MMAudio engine when installed. "
            "Standalone music uses Transformers MusicGen. "
            "`torchaudio` is installed by `launch.py` with the CUDA torch stack. "
            "AudioCraft is kept out of the shared venv because current releases pin older torch packages."
        )

    def music_model_choices(self) -> list[tuple[str, str]]:
        return [
            ("MusicGen small", "facebook/musicgen-small"),
            ("MusicGen medium", "facebook/musicgen-medium"),
            ("MusicGen melody", "facebook/musicgen-melody"),
            ("MusicGen stereo small", "facebook/musicgen-stereo-small"),
        ]

    def sfx_model_choices(self) -> list[tuple[str, str]]:
        return [
            ("AudioGen medium", "facebook/audiogen-medium"),
        ]

    def video_audio_model_choices(self) -> list[tuple[str, str]]:
        return [
            ("MMAudio large 44k v2", "mmaudio:large_44k_v2"),
            ("MMAudio large 44k", "mmaudio:large_44k"),
            ("MMAudio medium 44k", "mmaudio:medium_44k"),
            ("MMAudio small 44k", "mmaudio:small_44k"),
            ("MMAudio small 16k", "mmaudio:small_16k"),
        ]

    def video_audio_status(self) -> str:
        root = self._mmaudio_root()
        demo = root / "demo.py"
        python = self._audio_engine_python()
        if demo.is_file() and python.is_file():
            return (
                f"Video audio ready: MMAudio at {root}. "
                "MMAudio checkpoints are CC-BY-NC 4.0; use for non-commercial work unless licensed otherwise."
            )
        return (
            f"Video audio needs MMAudio installed at {root} with engine Python at {python}. "
            "This route is video-conditioned audio for generated Wan clips."
        )

    def output_path(self, *, stem: str = "audio", suffix: str = ".wav") -> Path:
        root = self.flags.resolved_output_dir() / getattr(self.settings, "audio_output_subdir", "audio")
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"{stem}_{stamp}{suffix}"

    def video_output_path(self, input_video: str | Path) -> Path:
        root = self.flags.resolved_output_dir() / getattr(self.settings, "audio_video_output_subdir", "audio-videos")
        root.mkdir(parents=True, exist_ok=True)
        stem = Path(input_video).stem or "video"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"{stem}_audio_{stamp}.mp4"

    def generate(
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
        dest = Path(output_path) if output_path else self.output_path(stem=self._safe_stem(prompt))
        dest.parent.mkdir(parents=True, exist_ok=True)

        with self._gpu_tenant("Audio generation"):
            if options.seed is not None and int(options.seed) >= 0:
                self._set_seed(int(options.seed))
            try:
                if options.kind == "sfx":
                    sample_rate = self._generate_audiocraft(options, dest)
                else:
                    sample_rate = self._generate_audiocraft(options, dest)
            except AudioUnavailable:
                if options.kind == "sfx":
                    raise
                sample_rate = self._generate_transformers_musicgen(options, dest)

        infotext = f"Audio {options.kind}: {options.model_id}, {options.duration_seconds:.1f}s"
        return AudioGenerationResult(
            output_path=str(dest),
            prompt=prompt,
            model_id=options.model_id,
            kind=options.kind,
            duration_seconds=float(options.duration_seconds),
            sample_rate=sample_rate,
            message=f"Saved {options.duration_seconds:.1f}s audio -> {dest}",
            infotext=infotext,
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
        prompt = (options.prompt or "").strip()
        if not prompt:
            raise AudioUnavailable("Enter an audio prompt first.")
        src_video = Path(video_path)
        if not src_video.is_file():
            raise AudioUnavailable(f"Video not found: {src_video}")
        stem = f"{src_video.stem}_{self._safe_stem(prompt)}"
        dest = Path(output_path) if output_path else self.output_path(stem=stem, suffix=".flac")
        dest.parent.mkdir(parents=True, exist_ok=True)

        with self._gpu_tenant("Video audio generation"):
            sample_rate = self._generate_mmaudio_video_audio(src_video, options, dest)

        infotext = (
            f"Video audio {options.model_id}: {options.duration_seconds:.1f}s, "
            f"steps {int(options.steps)}, CFG {float(options.cfg_coef):.2f}"
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
        )

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
        dest = Path(output_path) if output_path else self.video_output_path(src_video)
        dest.parent.mkdir(parents=True, exist_ok=True)
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
            str(dest),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise AudioUnavailable(f"Audio mux failed: {detail}")
        if not dest.is_file() or dest.stat().st_size <= 0:
            raise AudioUnavailable("Audio mux did not produce an output video.")
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

    def _generate_audiocraft(self, options: AudioGenerationOptions, dest: Path) -> int:
        try:
            import torchaudio
            from audiocraft.models import AudioGen, MusicGen
        except Exception as exc:
            raise AudioUnavailable(
                "AudioCraft is not installed in the shared Studio venv because current releases pin "
                "older torch packages. Use music generation, which falls back to Transformers MusicGen."
            ) from exc

        kind = "sfx" if options.kind == "sfx" else "music"
        model_cls = AudioGen if kind == "sfx" else MusicGen
        model_id = options.model_id or ("facebook/audiogen-medium" if kind == "sfx" else "facebook/musicgen-small")
        model = self._load_audiocraft_model(model_cls, model_id, kind)
        params = {
            "duration": float(options.duration_seconds),
            "temperature": float(options.temperature),
        }
        if kind == "music":
            params["cfg_coef"] = float(options.cfg_coef)
            params["top_k"] = int(options.top_k)
        model.set_generation_params(**params)
        wav = model.generate([options.prompt])
        sample_rate = int(getattr(model, "sample_rate", 32000 if kind == "music" else 16000))
        torchaudio.save(str(dest), wav[0].detach().cpu(), sample_rate=sample_rate)
        return sample_rate

    def _load_audiocraft_model(self, model_cls, model_id: str, kind: str):
        key = ("audiocraft", kind, model_id)
        if self._model is not None and self._model_key == key:
            return self._model
        device = self._device_string()
        try:
            model = model_cls.get_pretrained(model_id, device=device)
        except TypeError:
            model = model_cls.get_pretrained(model_id)
        self._model = model
        self._model_key = key
        return model

    def _generate_transformers_musicgen(self, options: AudioGenerationOptions, dest: Path) -> int:
        try:
            import scipy.io.wavfile
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
        except Exception as exc:
            raise AudioUnavailable(
                "MusicGen fallback needs `transformers`, `scipy`, and `torch`; AudioCraft is recommended."
            ) from exc

        model_id = options.model_id or "facebook/musicgen-small"
        key = ("transformers", "music", model_id)
        if self._model is None or self._model_key != key:
            processor = AutoProcessor.from_pretrained(model_id)
            model = MusicgenForConditionalGeneration.from_pretrained(model_id)
            model.to(self._device_string())
            self._model = (processor, model)
            self._model_key = key
        processor, model = self._model
        inputs = processor(text=[options.prompt], padding=True, return_tensors="pt").to(model.device)
        token_rate = 50
        max_new_tokens = max(16, int(float(options.duration_seconds) * token_rate))
        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=float(options.cfg_coef),
                temperature=float(options.temperature),
                max_new_tokens=max_new_tokens,
            )
        sample_rate = int(model.config.audio_encoder.sampling_rate)
        audio = audio_values[0].detach().cpu().float().numpy()
        if audio.ndim == 2:
            audio = np.swapaxes(audio, 0, 1)
        audio = np.asarray(audio, dtype=np.float32)
        scipy.io.wavfile.write(str(dest), sample_rate, audio)
        return sample_rate

    def _generate_mmaudio_video_audio(self, video_path: Path, options: AudioGenerationOptions, dest: Path) -> int:
        # VAP is internal shorthand for this post-processing route: a finished video
        # conditions audio generation through the isolated local MMAudio CLI bridge.
        # Come back here when the MVP is proven: parse MMAudio's real output metadata
        # instead of inferring sample rate from the variant name, and expose a small
        # install/status probe so the UI can distinguish "not installed" from "installed
        # but missing checkpoints".
        root = self._mmaudio_root()
        demo = root / "demo.py"
        python = self._audio_engine_python()
        if not demo.is_file():
            raise AudioUnavailable(f"MMAudio demo.py not found. Install MMAudio at {root}.")
        if not python.is_file():
            raise AudioUnavailable(f"MMAudio engine Python not found: {python}")

        variant = self._mmaudio_variant(options.model_id)
        run_dir = dest.parent / f"{dest.stem}_mmaudio"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed = int(options.seed) if options.seed is not None and int(options.seed) >= 0 else random.randint(0, 2**31 - 1)
        command = [
            str(python),
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
        result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, timeout=3600)
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
    def _find_mmaudio_audio_output(run_dir: Path, *, expected_stem: str) -> Path:
        expected = run_dir / f"{expected_stem}.flac"
        if expected.is_file() and expected.stat().st_size > 0:
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
        raise AudioUnavailable(
            f"MMAudio did not create expected audio: {expected}. "
            f"Found {len(candidates)} .flac file(s): {found}"
        )

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
