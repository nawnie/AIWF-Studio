from __future__ import annotations

import gc
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.core.domain.sana_video import (
    SANA_VIDEO_MODEL_REPO_480P,
    SANA_VIDEO_PIPELINE_I2V,
    SanaVideoRequest,
    SanaVideoResult,
    resolve_sana_video_path,
    sana_video_model_folder_name,
)
from aiwf.infrastructure.video.processing import VideoProcessor
from aiwf.services.audio import AudioGenerationService, AudioUnavailable

logger = logging.getLogger(__name__)


class SanaVideoUnavailable(RuntimeError):
    pass


class SanaVideoService:
    def __init__(self, flags: RuntimeFlags | None = None, settings: UserSettings | None = None, devices=None, supervisor=None) -> None:
        self.flags = flags or RuntimeFlags()
        self.settings = settings or UserSettings()
        self.devices = devices
        self.supervisor = supervisor

    def models_root(self) -> Path:
        return self.flags.resolved_models_dir() / "sana-video" / "Diffusers"

    def default_model_path(self) -> Path:
        return self.models_root() / sana_video_model_folder_name(SANA_VIDEO_MODEL_REPO_480P)

    def output_dir(self) -> Path:
        root = self.flags.resolved_output_dir() / "sana-videos"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def status_markdown(self) -> str:
        model = self.default_model_path()
        model_index = model / "model_index.json"
        runtime_ok = self.runtime_available()
        lines = [
            "**Sana video:** "
            + ("runtime ready" if runtime_ok else "Diffusers Sana video classes missing"),
            f"- Default model: `{model}`",
            "- Audio: no native Sana audio path detected; optional MMAudio post-process can attach audio after video.",
        ]
        if not model_index.is_file():
            lines.append("- Model snapshot missing. Download the 480p SANA-Video Diffusers folder first.")
        return "\n".join(lines)

    @staticmethod
    def runtime_available() -> bool:
        try:
            import diffusers

            return hasattr(diffusers, "SanaVideoPipeline") and hasattr(diffusers, "SanaImageToVideoPipeline")
        except Exception:
            return False

    def generate(self, request: SanaVideoRequest) -> SanaVideoResult:
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise SanaVideoUnavailable("Enter a Sana video prompt first.")
        if not self.runtime_available():
            raise SanaVideoUnavailable("Installed Diffusers does not expose SanaVideoPipeline.")

        model_path = resolve_sana_video_path(request.model_path, self.default_model_path(), self.flags.data_dir)
        if not (model_path / "model_index.json").is_file():
            raise SanaVideoUnavailable(f"Sana video model folder missing model_index.json: {model_path}")

        source_image = resolve_sana_video_path(request.source_image_path, Path(), self.flags.data_dir) if request.source_image_path else None
        if request.wants_image_to_video and (source_image is None or not source_image.is_file()):
            raise SanaVideoUnavailable(f"Sana image-to-video source image missing: {source_image}")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.output_dir() / f"sana-video-{stamp}-{uuid4().hex[:6]}.mp4"
        video_only_path = output_path

        pipe = None
        try:
            pipe = self._load_pipeline(model_path, image_to_video=request.wants_image_to_video)
            frames = self._run_pipeline(pipe, request, source_image=source_image)
            self._export_frames(frames, output_path, fps=float(request.fps))
        finally:
            if pipe is not None:
                del pipe
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        audio_path = ""
        has_audio = VideoProcessor().probe(output_path).has_audio
        if request.generate_audio:
            audio_prompt = (request.audio_prompt or request.prompt or "").strip()
            audio_service = AudioGenerationService(self.flags, self.settings, self.devices, self.supervisor)
            try:
                audio, muxed = audio_service.generate_and_mux(
                    output_path,
                    AudioGenerationOptions(
                        prompt=audio_prompt,
                        kind="video_audio",
                        model_id=request.audio_model_id,
                        duration_seconds=max(1.0, float(request.frames) / max(float(request.fps), 1.0)),
                        cfg_coef=float(request.audio_cfg),
                        steps=int(request.audio_steps),
                        seed=int(request.seed),
                    ),
                    duration_seconds=max(1.0, float(request.frames) / max(float(request.fps), 1.0)),
                )
            except AudioUnavailable as exc:
                raise SanaVideoUnavailable(f"Sana video rendered, but audio generation failed: {exc}") from exc
            audio_path = audio.output_path
            video_only_path = output_path
            output_path = Path(muxed.output_path)
            has_audio = True

        return SanaVideoResult(
            output_path=str(output_path),
            message=f"Sana video saved to {output_path.name}" + (" with audio" if has_audio else ""),
            frames=int(request.frames),
            fps=float(request.fps),
            width=int(request.width),
            height=int(request.height),
            has_audio=has_audio,
            audio_path=audio_path,
            video_only_path=str(video_only_path) if Path(video_only_path) != Path(output_path) else "",
            infotext=(
                f"Sana video {request.pipeline}: {request.width}x{request.height}, "
                f"{request.frames} frames, {request.steps} steps, CFG {request.cfg_scale:.2f}"
            ),
        )

    def _load_pipeline(self, model_path: Path, *, image_to_video: bool):
        import torch
        from diffusers import SanaImageToVideoPipeline, SanaVideoPipeline

        pipeline_cls = SanaImageToVideoPipeline if image_to_video else SanaVideoPipeline
        dtype = self._dtype()
        kwargs = {"torch_dtype": dtype, "local_files_only": True}
        try:
            pipe = pipeline_cls.from_pretrained(str(model_path), **kwargs)
        except TypeError:
            kwargs.pop("torch_dtype", None)
            kwargs["dtype"] = dtype
            pipe = pipeline_cls.from_pretrained(str(model_path), **kwargs)
        device = self._device()
        pipe = pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        if hasattr(pipe, "vae"):
            try:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
            except Exception:
                logger.debug("Could not enable Sana video VAE tiling/slicing", exc_info=True)
        return pipe

    def _run_pipeline(self, pipe, request: SanaVideoRequest, *, source_image: Path | None):  # noqa: ANN001
        import torch

        generator = torch.Generator(device=self._device() if self._device().type == "cuda" else "cpu")
        seed = int(request.seed)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        generator.manual_seed(seed)

        kwargs = dict(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            num_inference_steps=int(request.steps),
            guidance_scale=float(request.cfg_scale),
            height=int(request.height),
            width=int(request.width),
            frames=int(request.frames),
            generator=generator,
            output_type="pil",
            clean_caption=False,
            use_resolution_binning=bool(request.use_resolution_binning),
            max_sequence_length=int(request.max_sequence_length),
        )
        if source_image is not None:
            kwargs["image"] = Image.open(source_image).convert("RGB")
        output = pipe(**kwargs)
        frames = getattr(output, "frames", output[0] if isinstance(output, tuple) else output)
        if isinstance(frames, list) and frames and isinstance(frames[0], list):
            return frames[0]
        return frames

    @staticmethod
    def _export_frames(frames, output_path: Path, *, fps: float) -> None:  # noqa: ANN001
        try:
            from diffusers.utils import export_to_video

            export_to_video(frames, str(output_path), fps=int(round(fps)))
        except Exception:
            from aiwf.infrastructure.video.processing import write_frames

            write_frames(frames, output_path, fps=fps)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise SanaVideoUnavailable(f"Sana video export did not create output: {output_path}")

    def _device(self):
        if self.devices is not None:
            try:
                return self.devices.device()
            except Exception:
                pass
        try:
            import torch

            return torch.device("cuda" if torch.cuda.is_available() and not self.flags.cpu else "cpu")
        except Exception:
            import torch

            return torch.device("cpu")

    def _dtype(self):
        try:
            import torch

            device = self._device()
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if device.type == "cuda":
                return torch.float16
            return torch.float32
        except Exception:
            import torch

            return torch.float32
