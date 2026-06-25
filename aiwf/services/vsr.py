from __future__ import annotations

import logging
import os
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.core.domain.vsr import (
    VideoFxAigsOptions,
    VideoFxDenoiseOptions,
    VideoFxRelightOptions,
    VsrOptions,
    VsrResult,
)
from aiwf.infrastructure.video import VideoError, VideoProcessor

logger = logging.getLogger(__name__)
VSR_CLEANUP_SAME_RES_MODES = set(range(8, 16))
VIDEOFX_TENANT = EngineTenant.VIDEO


class VsrUnavailable(RuntimeError):
    """Raised when NVIDIA RTX VSR cannot run in this environment."""


@dataclass(frozen=True)
class VsrInstallInfo:
    app_path: Path | None
    sdk_root: Path | None
    model_dir: Path | None
    upscale_app_path: Path | None = None
    denoise_app_path: Path | None = None
    aigs_app_path: Path | None = None
    relight_app_path: Path | None = None
    feature_names: tuple[str, ...] = ()
    model_count: int = 0
    compute_capability: str | None = None
    install_script: Path | None = None

    @property
    def available(self) -> bool:
        return self.app_path is not None and self.app_path.is_file()

    @property
    def upscale_available(self) -> bool:
        return self.upscale_app_path is not None and self.upscale_app_path.is_file()

    @property
    def denoise_available(self) -> bool:
        return self.denoise_app_path is not None and self.denoise_app_path.is_file()

    @property
    def aigs_available(self) -> bool:
        return self.aigs_app_path is not None and self.aigs_app_path.is_file()

    @property
    def relight_available(self) -> bool:
        return self.relight_app_path is not None and self.relight_app_path.is_file()

    @property
    def sdk_runtime_available(self) -> bool:
        if self.sdk_root is None:
            return False
        return (
            (self.sdk_root / "bin" / "NVVideoEffects.dll").is_file()
            and (self.sdk_root / "bin" / "NVCVImage.dll").is_file()
            and (self.sdk_root / "nvvfx" / "include" / "nvVideoEffects.h").is_file()
        )

    @property
    def models_installed(self) -> bool:
        return self.model_count > 0


class VsrService:
    """NVIDIA Video Effects SDK wrapper.

    The first implementation uses NVIDIA's Video Effects SDK sample binary
    (`VideoEffectsApp.exe`) instead of binding the C API directly. That keeps
    the Python app boot-safe while we validate the SDK install path locally.
    Failures are raised as VsrUnavailable so higher-level video flows can
    soft-fail optional cleanup/upscale stages and preserve the prior video.
    """

    def __init__(self, flags: RuntimeFlags, settings: UserSettings, supervisor=None) -> None:
        self.flags = flags
        self.settings = settings
        self.supervisor = supervisor
        self.vsr_model = None

    def upscale_image(self, img: Image.Image, options: VsrOptions) -> Image.Image:
        """Upscales a single PIL Image natively using PyTorch tensors, mirroring ComfyUI."""
        import torch
        import numpy as np
        from PIL import Image

        # 1. Convert PIL Image to a batch tensor: Shape (Batch=1, Channels, Height, Width)
        # ComfyUI natively represents images as [B, H, W, C] normalized to 0.0 - 1.0
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Shape: [1, H, W, C]
        
        # Move tensor to the appropriate execution device (GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)
        
        # 2. Replicate frames if the VSR model expects a temporal sequence (e.g., minimum 3 frames)
        # Many video models require a temporal window to calculate optical flow or attention
        temporal_window = 3 
        vsr_input = img_tensor.repeat(temporal_window, 1, 1, 1)  # Shape: [3, H, W, C]
        
        # 3. Run the VSR processing pipeline directly in memory
        with torch.inference_mode():
            # Replace this placeholder with your model's forward pass
            # e.g., upscaled_sequence = self.vsr_model(vsr_input)
            upscaled_sequence = vsr_input  
            
        # 4. Extract the primary upscaled frame (usually the middle or first frame)
        target_frame = upscaled_sequence[0].cpu()
        
        # 5. Convert back to a standard PIL Image
        output_np = (target_frame.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(output_np)

    def install_info(self) -> VsrInstallInfo:
        app_path = self._resolve_app_path()
        upscale_app_path = self._resolve_upscale_app_path()
        denoise_app_path = self._resolve_denoise_app_path()
        aigs_app_path = self._resolve_aigs_app_path()
        relight_app_path = self._resolve_relight_app_path()
        sdk_root = self._resolve_sdk_root(
            app_path or upscale_app_path or denoise_app_path or aigs_app_path or relight_app_path
        )
        model_dir = self._resolve_model_dir(sdk_root)
        return VsrInstallInfo(
            app_path=app_path,
            upscale_app_path=upscale_app_path,
            denoise_app_path=denoise_app_path,
            aigs_app_path=aigs_app_path,
            relight_app_path=relight_app_path,
            sdk_root=sdk_root,
            model_dir=model_dir,
            feature_names=self._installed_feature_names(sdk_root),
            model_count=self._model_count(model_dir),
            compute_capability=self._compute_capability(sdk_root),
            install_script=self._feature_install_script(sdk_root),
        )

    def available(self) -> bool:
        return self.install_info().available

    def denoise_available(self) -> bool:
        return self.install_info().denoise_available

    def aigs_available(self) -> bool:
        return self.install_info().aigs_available

    def relight_available(self) -> bool:
        return self.install_info().relight_available

    def relighting_hdr_choices(self) -> list[tuple[str, str]]:
        info = self.install_info()
        dirs: list[Path] = []
        if info.relight_app_path is not None:
            dirs.append(info.relight_app_path.parent)
        if info.sdk_root is not None:
            dirs.extend(
                [
                    info.sdk_root,
                    info.sdk_root / "bin",
                    info.sdk_root / "assets",
                    info.sdk_root / "samples",
                ]
            )
        seen: set[str] = set()
        choices: list[tuple[str, str]] = []
        for directory in dirs:
            if not directory.is_dir():
                continue
            try:
                files = list(directory.glob("*.hdr")) + list(directory.glob("*.exr"))
            except OSError:
                continue
            for path in sorted(files, key=lambda item: item.name.lower()):
                key = str(path).lower()
                if key in seen:
                    continue
                seen.add(key)
                choices.append((path.stem, str(path)))
        return choices

    def folder_help(self) -> str:
        info = self.install_info()
        if info.available or info.upscale_available or info.denoise_available or info.aigs_available or info.relight_available:
            details = []
            if info.available:
                details.append(f"Detected NVIDIA Video Effects runner: `{info.app_path}`")
            if info.upscale_available:
                details.append(f"Detected NVIDIA Upscale runner: `{info.upscale_app_path}`")
            if info.denoise_available:
                details.append(f"Detected NVIDIA Denoise runner: `{info.denoise_app_path}`")
            if info.aigs_available:
                details.append(f"Detected NVIDIA AI Green Screen runner: `{info.aigs_app_path}`")
            if info.relight_available:
                details.append(f"Detected NVIDIA Relighting runner: `{info.relight_app_path}`")
            if info.sdk_root is not None:
                details.append(f"SDK root: `{info.sdk_root}`")
            if info.feature_names:
                details.append(f"Features: {', '.join(info.feature_names)}")
            if info.model_dir is not None:
                details.append(f"Models: {info.model_count} package(s) in `{info.model_dir}`")
            return "  \n".join(details)
        if info.sdk_runtime_available:
            install_hint = self._feature_install_hint(info)
            parts = [
                f"Detected NVIDIA Video Effects SDK core at `{info.sdk_root}`.",
                "AIWF still needs `VideoEffectsApp.exe` for the current VSR runner.",
            ]
            if info.feature_names:
                parts.append(f"Installed features: {', '.join(info.feature_names)}.")
            else:
                parts.append("No VideoFX feature packs are installed yet.")
            if info.model_dir is not None:
                parts.append(f"Models: {info.model_count} package(s) in `{info.model_dir}`.")
            if install_hint:
                parts.append(f"Feature install command: `{install_hint}`")
            parts.append("Build NVIDIA-Maxine/VFX-SDK-Samples, then set `AIWF_VSR_VIDEO_EFFECTS_APP` to the built sample executable.")
            return "  \n".join(parts)
        return (
            "NVIDIA VideoFX needs the NVIDIA Video Effects SDK runtime and "
            "one or more built sample apps. Install the SDK, or set "
            "`AIWF_VSR_VIDEO_EFFECTS_APP`, `AIWF_VSR_UPSCALE_APP`, "
            "`AIWF_VIDEOFX_DENOISE_APP`, `AIWF_VIDEOFX_AIGS_APP`, "
            "or `AIWF_VIDEOFX_RELIGHT_APP`."
        )

    def _output_path(self, input_video: str | Path, suffix: str = "vsr") -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(input_video).stem or "video"
        sub = getattr(self.settings, "vsr_output_subdir", "vsr-videos")
        root = self.flags.resolved_output_dir() / sub
        root.mkdir(parents=True, exist_ok=True)
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in suffix).strip("_") or "vsr"
        return root / f"{stem}_{safe_suffix}_{stamp}.mp4"

    def upscale(
        self,
        input_video: str | Path,
        options: VsrOptions,
        *,
        output_path: str | Path | None = None,
    ) -> VsrResult:
        src = Path(input_video)
        if not src.is_file():
            raise VsrUnavailable(f"Video not found: {src}")

        info = self.install_info()
        selected_effect = options.effect if options.effect in {"SuperRes", "Cleanup", "Upscale"} else "SuperRes"
        if selected_effect in {"SuperRes", "Cleanup"} and (not info.available or info.app_path is None):
            raise VsrUnavailable(self.folder_help())
        if selected_effect == "Upscale" and (not info.upscale_available or info.upscale_app_path is None):
            raise VsrUnavailable(self.folder_help())

        processor = VideoProcessor()
        try:
            input_info = processor.probe(src)
        except VideoError as exc:
            raise VsrUnavailable(str(exc)) from exc
        if input_info.width <= 0 or input_info.height <= 0:
            raise VsrUnavailable("Could not read input video dimensions for VSR.")

        mode = max(0, min(19, int(options.mode)))
        scale = self._valid_scale(float(options.scale))
        if selected_effect == "Cleanup" or mode in VSR_CLEANUP_SAME_RES_MODES:
            scale = 1.0
        target_height = int(round(input_info.height * scale))
        suffix = "upscale" if selected_effect == "Upscale" else ("cleanup" if scale == 1.0 and mode >= 8 else "vsr")
        dest = Path(output_path) if output_path else self._output_path(src, suffix=suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)

        command = self._command(
            info,
            src,
            dest,
            effect=selected_effect,
            target_height=target_height,
            mode=mode,
            strength=float(options.strength),
            codec=options.codec,
        )
        self._run_sdk_command(command, info, label="NVIDIA VideoFX VSR", dest=dest)

        try:
            output_info = processor.probe(dest)
        except VideoError as exc:
            raise VsrUnavailable(f"NVIDIA VideoFX VSR output is not readable: {dest}") from exc

        infotext = (
            f"NVIDIA VideoFX {selected_effect} mode {mode} {scale:g}x "
            f"({input_info.width}x{input_info.height}->{output_info.width}x{output_info.height})"
        )
        return self._result(dest, input_info=input_info, output_info=output_info, infotext=infotext)

    def denoise(
        self,
        input_video: str | Path,
        options: VideoFxDenoiseOptions,
        *,
        output_path: str | Path | None = None,
    ) -> VsrResult:
        src = Path(input_video)
        if not src.is_file():
            raise VsrUnavailable(f"Video not found: {src}")

        info = self.install_info()
        if not info.denoise_available or info.denoise_app_path is None:
            raise VsrUnavailable(self.folder_help())

        processor = VideoProcessor()
        input_info = self._probe_video(processor, src, "NVIDIA VideoFX Denoise input")
        dest = Path(output_path) if output_path else self._output_path(src, suffix="denoise")
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = self._denoise_command(info, src, dest, options)

        self._run_sdk_command(command, info, label="NVIDIA VideoFX Denoise", dest=dest)
        output_info = self._probe_video(processor, dest, "NVIDIA VideoFX Denoise output")
        return self._result(
            dest,
            input_info=input_info,
            output_info=output_info,
            infotext=(
                f"NVIDIA VideoFX Denoise strength {float(options.strength):.2f} "
                f"({input_info.width}x{input_info.height}->{output_info.width}x{output_info.height})"
            ),
        )

    def aigs(
        self,
        input_video: str | Path,
        options: VideoFxAigsOptions,
        *,
        output_path: str | Path | None = None,
    ) -> VsrResult:
        src = Path(input_video)
        if not src.is_file():
            raise VsrUnavailable(f"Video not found: {src}")

        info = self.install_info()
        if not info.aigs_available or info.aigs_app_path is None:
            raise VsrUnavailable(self.folder_help())

        processor = VideoProcessor()
        input_info = self._probe_video(processor, src, "NVIDIA VideoFX AI Green Screen input")
        if min(input_info.width, input_info.height) < 288 or max(input_info.width, input_info.height) < 512:
            raise VsrUnavailable(
                "NVIDIA AI Green Screen requires video at least 512x288 or 288x512. "
                "Upscale first, or generate a larger video."
            )

        suffix_by_comp = {
            0: "aigs-matte",
            1: "aigs-mask",
            2: "aigs-green",
            3: "aigs-white",
            4: "aigs-input",
            5: "aigs-bg",
            6: "aigs-blur",
        }
        comp_mode = max(0, min(6, int(options.comp_mode)))
        dest = Path(output_path) if output_path else self._output_path(
            src,
            suffix=suffix_by_comp.get(comp_mode, "aigs"),
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = self._aigs_command(info, src, dest, options)

        self._run_sdk_command(command, info, label="NVIDIA VideoFX AI Green Screen", dest=dest)
        output_info = self._probe_video(processor, dest, "NVIDIA VideoFX AI Green Screen output")
        return self._result(
            dest,
            input_info=input_info,
            output_info=output_info,
            infotext=(
                f"NVIDIA VideoFX AI Green Screen comp {comp_mode} "
                f"({input_info.width}x{input_info.height}->{output_info.width}x{output_info.height})"
            ),
        )

    def relight(
        self,
        input_video: str | Path,
        options: VideoFxRelightOptions,
        *,
        output_path: str | Path | None = None,
    ) -> VsrResult:
        src = Path(input_video)
        if not src.is_file():
            raise VsrUnavailable(f"Video not found: {src}")

        info = self.install_info()
        if not info.relight_available or info.relight_app_path is None:
            raise VsrUnavailable(self.folder_help())

        hdr_path = self._resolve_relight_hdr(options.hdr_file, info)
        if hdr_path is None:
            raise VsrUnavailable("NVIDIA Relighting requires an HDR/EXR illumination file.")

        processor = VideoProcessor()
        input_info = self._probe_video(processor, src, "NVIDIA VideoFX Relighting input")
        if min(input_info.width, input_info.height) < 288 or max(input_info.width, input_info.height) < 512:
            raise VsrUnavailable(
                "NVIDIA Relighting requires video at least 512x288 or 288x512. "
                "Upscale first, or generate a larger video."
            )

        dest = Path(output_path) if output_path else self._output_path(src, suffix="relight")
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = self._relight_command(info, src, dest, options, hdr_path)

        self._run_sdk_command(command, info, label="NVIDIA VideoFX Relighting", dest=dest)
        output_info = self._probe_video(processor, dest, "NVIDIA VideoFX Relighting output")
        return self._result(
            dest,
            input_info=input_info,
            output_info=output_info,
            infotext=(
                f"NVIDIA VideoFX Relighting hdr {hdr_path.name} "
                f"({input_info.width}x{input_info.height}->{output_info.width}x{output_info.height})"
            ),
        )

    def _command(
        self,
        info: VsrInstallInfo,
        src: Path,
        dest: Path,
        *,
        effect: str,
        target_height: int,
        mode: int,
        strength: float,
        codec: str,
    ) -> list[str]:
        selected_effect = effect if effect in {"SuperRes", "Upscale"} else "SuperRes"
        if selected_effect == "Upscale":
            if info.upscale_app_path is None:
                raise VsrUnavailable("NVIDIA UpscalePipelineApp.exe was not found.")
            command = [
                str(info.upscale_app_path),
                f"--in_file={src}",
                f"--out_file={dest}",
                f"--resolution={max(1, int(target_height))}",
                "--show=false",
                f"--codec={codec or 'H264'}",
                f"--upscale_strength={max(0.0, min(1.0, float(strength))):.3f}",
            ]
            if info.model_dir is not None and info.model_dir.is_dir():
                command.append(f"--model_dir={info.model_dir}")
            return command

        if info.app_path is None:
            raise VsrUnavailable("NVIDIA VideoEffectsApp.exe was not found.")
        command = [
            str(info.app_path),
            f"--in_file={src}",
            f"--out_file={dest}",
            "--effect=VideoSuperRes",
            f"--resolution={max(1, int(target_height))}",
            "--show=false",
            f"--codec={codec or 'H264'}",
        ]
        command.append(f"--mode={max(0, min(19, int(mode)))}")
        if info.model_dir is not None and info.model_dir.is_dir():
            command.append(f"--model_dir={info.model_dir}")
        return command

    def _denoise_command(
        self,
        info: VsrInstallInfo,
        src: Path,
        dest: Path,
        options: VideoFxDenoiseOptions,
    ) -> list[str]:
        if info.denoise_app_path is None:
            raise VsrUnavailable("NVIDIA DenoiseEffectApp.exe was not found.")
        command = [
            str(info.denoise_app_path),
            f"--in_file={src}",
            f"--out_file={dest}",
            "--show=false",
            "--progress=true",
            f"--codec={options.codec or 'avc1'}",
            f"--strength={max(0.0, min(1.0, float(options.strength))):.3f}",
        ]
        if info.model_dir is not None and info.model_dir.is_dir():
            command.append(f"--model_dir={info.model_dir}")
        return command

    def _aigs_command(
        self,
        info: VsrInstallInfo,
        src: Path,
        dest: Path,
        options: VideoFxAigsOptions,
    ) -> list[str]:
        if info.aigs_app_path is None:
            raise VsrUnavailable("NVIDIA AigsEffectApp.exe was not found.")
        comp_mode = max(0, min(6, int(options.comp_mode)))
        command = [
            str(info.aigs_app_path),
            f"--in_file={src}",
            f"--out_file={dest}",
            "--show=false",
            "--progress=true",
            f"--codec={options.codec or 'avc1'}",
            f"--mode={max(0, min(1, int(options.mode)))}",
            f"--comp_mode={comp_mode}",
            f"--blur_strength={max(0.0, min(1.0, float(options.blur_strength))):.3f}",
        ]
        if options.background_file and comp_mode == 5:
            command.append(f"--bg_file={options.background_file}")
        if options.cuda_graph:
            command.append("--cuda_graph=true")
        if info.model_dir is not None and info.model_dir.is_dir():
            command.append(f"--model_dir={info.model_dir}")
        return command

    def _relight_command(
        self,
        info: VsrInstallInfo,
        src: Path,
        dest: Path,
        options: VideoFxRelightOptions,
        hdr_path: Path,
    ) -> list[str]:
        if info.relight_app_path is None:
            raise VsrUnavailable("NVIDIA RelightingEffectApp.exe was not found.")
        bg_mode = max(0, min(4, int(options.background_mode)))
        command = [
            str(info.relight_app_path),
            f"--in_file={src}",
            f"--out_file={dest}",
            f"--in_hdr={hdr_path}",
            "--show=false",
            f"--codec={options.codec or 'avc1'}",
            f"--bg_mode={bg_mode}",
            f"--pan={float(options.pan_degrees):.3f}",
            f"--vfov={max(1.0, min(179.0, float(options.vfov_degrees))):.3f}",
        ]
        if options.autorotate:
            command.append("--autorotate=true")
            command.append(f"--rotation_rate={float(options.rotation_rate):.3f}")
        if options.background and bg_mode in {3, 4}:
            command.append(f"--in_bg={options.background}")
        if info.model_dir is not None and info.model_dir.is_dir():
            command.append(f"--model_dir={info.model_dir}")
        return command

    def _run_sdk_command(self, command: list[str], info: VsrInstallInfo, *, label: str, dest: Path) -> None:
        tenant_job_id = f"videofx_{uuid.uuid4().hex[:8]}"
        if self.supervisor is not None:
            # VideoFX sample apps are external processes but still consume the
            # same GPU tenant as Wan, so gate them through the supervisor.
            switch = self.supervisor.request_switch(
                EngineSwitchRequest(
                    target=VIDEOFX_TENANT,
                    reason=label,
                    job_id=tenant_job_id,
                )
            )
            if not switch.ok:
                raise VsrUnavailable(f"GPU busy: {switch.message}")

        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(command[0]).parent),
                env=self._sdk_env(info),
                capture_output=True,
                text=True,
                timeout=7200,
            )
        finally:
            if self.supervisor is not None:
                self.supervisor.request_switch(
                    EngineSwitchRequest(
                        target=EngineTenant.IDLE,
                        reason=f"{label} complete",
                        job_id=tenant_job_id,
                    )
                )

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise VsrUnavailable(f"{label} failed ({completed.returncode}): {detail}")
        if not dest.is_file() or dest.stat().st_size <= 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise VsrUnavailable(f"{label} did not produce an output file. {detail}")

    @staticmethod
    def _probe_video(processor: VideoProcessor, path: Path, label: str):
        try:
            info = processor.probe(path)
        except VideoError as exc:
            raise VsrUnavailable(f"{label} is not readable: {path}") from exc
        if info.width <= 0 or info.height <= 0:
            raise VsrUnavailable(f"Could not read dimensions for {label}: {path}")
        return info

    @staticmethod
    def _result(dest: Path, *, input_info, output_info, infotext: str) -> VsrResult:
        return VsrResult(
            output_path=str(dest),
            input_width=input_info.width,
            input_height=input_info.height,
            output_width=output_info.width,
            output_height=output_info.height,
            fps=output_info.fps,
            frame_count=output_info.frame_count,
            infotext=infotext,
            message=(
                f"Saved {output_info.frame_count} frame(s) at "
                f"{output_info.width}x{output_info.height}, {output_info.fps:.1f} fps -> {dest}"
            ),
        )

    def _sdk_env(self, info: VsrInstallInfo) -> dict[str, str]:
        env = os.environ.copy()
        path_entries = self._candidate_path_entries(info)
        if path_entries:
            env["PATH"] = os.pathsep.join([str(p) for p in path_entries] + [env.get("PATH", "")])
        return env

    def _candidate_path_entries(self, info: VsrInstallInfo) -> list[Path]:
        entries: list[Path] = []
        if info.app_path is not None:
            entries.append(info.app_path.parent)
            entries.extend(self._sample_dependency_dirs(info.app_path))
        if info.upscale_app_path is not None:
            entries.append(info.upscale_app_path.parent)
            entries.extend(self._sample_dependency_dirs(info.upscale_app_path))
        if info.denoise_app_path is not None:
            entries.append(info.denoise_app_path.parent)
            entries.extend(self._sample_dependency_dirs(info.denoise_app_path))
        if info.aigs_app_path is not None:
            entries.append(info.aigs_app_path.parent)
            entries.extend(self._sample_dependency_dirs(info.aigs_app_path))
        if info.relight_app_path is not None:
            entries.append(info.relight_app_path.parent)
            entries.extend(self._sample_dependency_dirs(info.relight_app_path))
        if info.sdk_root is not None:
            entries.extend(
                [
                    info.sdk_root,
                    info.sdk_root / "bin",
                    info.sdk_root / "lib",
                    info.sdk_root / "redist",
                    info.sdk_root / "samples" / "external" / "opencv" / "bin",
                ]
            )
            entries.extend(self._feature_dirs(info.sdk_root))
            entries.extend(path / "bin" for path in self._feature_dirs(info.sdk_root))
        return [p for p in entries if p.is_dir()]

    @staticmethod
    def _valid_scale(scale: float) -> float:
        allowed = (1.0, 1.3333333, 1.5, 2.0, 3.0, 4.0)
        return min(allowed, key=lambda item: abs(item - scale))

    def _resolve_app_path(self) -> Path | None:
        explicit = os.environ.get("AIWF_VSR_VIDEO_EFFECTS_APP", "").strip()
        if explicit:
            path = Path(explicit)
            return path if path.is_file() else None
        return self._resolve_sample_app_path("VideoEffectsApp")

    def _resolve_upscale_app_path(self) -> Path | None:
        explicit = os.environ.get("AIWF_VSR_UPSCALE_APP", "").strip()
        if explicit:
            path = Path(explicit)
            return path if path.is_file() else None
        return self._resolve_sample_app_path("UpscalePipelineApp")

    def _resolve_denoise_app_path(self) -> Path | None:
        explicit = (
            os.environ.get("AIWF_VIDEOFX_DENOISE_APP", "").strip()
            or os.environ.get("AIWF_VSR_DENOISE_APP", "").strip()
        )
        if explicit:
            path = Path(explicit)
            return path if path.is_file() else None
        return self._resolve_sample_app_path("DenoiseEffectApp")

    def _resolve_aigs_app_path(self) -> Path | None:
        explicit = (
            os.environ.get("AIWF_VIDEOFX_AIGS_APP", "").strip()
            or os.environ.get("AIWF_AIGS_APP", "").strip()
        )
        if explicit:
            path = Path(explicit)
            return path if path.is_file() else None
        return self._resolve_sample_app_path("AigsEffectApp")

    def _resolve_relight_app_path(self) -> Path | None:
        explicit = (
            os.environ.get("AIWF_VIDEOFX_RELIGHT_APP", "").strip()
            or os.environ.get("AIWF_RELIGHT_APP", "").strip()
        )
        if explicit:
            path = Path(explicit)
            return path if path.is_file() else None
        return self._resolve_sample_app_path("RelightingEffectApp")

    def _resolve_relight_hdr(self, hdr_file: str | None, info: VsrInstallInfo) -> Path | None:
        if hdr_file:
            path = Path(hdr_file)
            if path.is_file():
                return path
        for _label, value in self.relighting_hdr_choices():
            path = Path(value)
            if path.is_file() and path.stem.lower() == "default":
                return path
        for _label, value in self.relighting_hdr_choices():
            path = Path(value)
            if path.is_file():
                return path
        if info.relight_app_path is not None:
            default = info.relight_app_path.parent / "Default.hdr"
            if default.is_file():
                return default
        return None

    def _resolve_sample_app_path(self, app_name: str) -> Path | None:
        exe_name = f"{app_name}.exe"
        for root in self._candidate_roots():
            candidates = [
                root / "build" / "apps" / app_name / "Release" / exe_name,
                root / "build" / "apps" / app_name / exe_name,
                root / "samples" / app_name / exe_name,
                root / "samples" / app_name / "bin" / exe_name,
                root / app_name / exe_name,
                root / exe_name,
                root / "bin" / exe_name,
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            if root.is_dir():
                try:
                    found = next(root.rglob(exe_name), None)
                except OSError:
                    found = None
                if found is not None and found.is_file():
                    return found
        return None

    def _resolve_sdk_root(self, app_path: Path | None) -> Path | None:
        explicit = os.environ.get("AIWF_NVIDIA_VFX_SDK_ROOT", "").strip()
        if explicit and Path(explicit).is_dir():
            return Path(explicit)
        for root in self._candidate_roots():
            if self._is_vfx_sdk_root(root):
                return root
        if app_path is not None:
            for parent in [app_path.parent, *app_path.parents]:
                if self._is_vfx_sdk_root(parent):
                    return parent
        if app_path is not None:
            return app_path.parent
        for root in self._candidate_roots():
            if root.is_dir():
                return root
        return None

    def _resolve_model_dir(self, sdk_root: Path | None) -> Path | None:
        explicit = os.environ.get("AIWF_VSR_MODEL_DIR", "").strip()
        if explicit and Path(explicit).is_dir():
            return Path(explicit)
        if sdk_root is None:
            return None
        for candidate in (
            sdk_root / "models",
            sdk_root / "bin" / "models",
            sdk_root / "samples" / "models",
            sdk_root / "VideoEffectsApp" / "models",
        ):
            if candidate.is_dir():
                return candidate
        return None

    def _candidate_roots(self) -> Sequence[Path]:
        anchor = self.flags.data_dir.anchor if self.flags.data_dir.anchor else ""
        roots: list[str] = [
            os.environ.get("AIWF_NVIDIA_VFX_SDK_ROOT", "").strip(),
            r"C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects",
            r"C:\Program Files\NVIDIA Corporation\NVIDIA VFX SDK",
            str(Path(anchor) / "VideoFX") if anchor else "",
            str(self.flags.data_dir.parent / "VideoFX"),
            str(Path(anchor) / "sdks" / "nvidia" / "nvidia-vfx-sdk-samples") if anchor else "",
            str(Path(anchor) / "sdks" / "nvidia" / "nvidia-vfx-sdk") if anchor else "",
            str(self.flags.data_dir / "engines" / "nvidia-vfx-sdk"),
            str(self.flags.data_dir / "engines" / "rtx-video-sdk"),
            str(self.flags.data_dir / "engines" / "nvidia-vfx-sdk-samples"),
        ]
        candidates: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            if not root:
                continue
            path = Path(root)
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(path)
        return candidates

    @staticmethod
    def _is_vfx_sdk_root(root: Path) -> bool:
        return (
            root.is_dir()
            and (root / "bin" / "NVVideoEffects.dll").is_file()
            and (root / "nvvfx" / "include" / "nvVideoEffects.h").is_file()
        )

    def _feature_install_script(self, sdk_root: Path | None) -> Path | None:
        if sdk_root is None:
            return None
        script = sdk_root / "features" / "install_feature.ps1"
        return script if script.is_file() else None

    def _feature_dirs(self, sdk_root: Path | None) -> list[Path]:
        if sdk_root is None:
            return []
        feature_root = sdk_root / "features"
        if not feature_root.is_dir():
            return []
        try:
            return [path for path in feature_root.iterdir() if path.is_dir()]
        except OSError:
            return []

    def _installed_feature_names(self, sdk_root: Path | None) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in self._feature_dirs(sdk_root)))

    @staticmethod
    def _sample_dependency_dirs(app_path: Path) -> list[Path]:
        dirs: list[Path] = []
        for parent in app_path.parents:
            candidate = parent / "external" / "opencv" / "bin"
            if candidate.is_dir():
                dirs.append(candidate)
        return dirs

    @staticmethod
    def _model_count(model_dir: Path | None) -> int:
        if model_dir is None or not model_dir.is_dir():
            return 0
        try:
            return sum(1 for path in model_dir.rglob("*.trtpkg") if path.is_file())
        except OSError:
            return 0

    def _compute_capability(self, sdk_root: Path | None) -> str | None:
        explicit = os.environ.get("AIWF_NVIDIA_SM", "").strip()
        if explicit:
            return explicit
        if sdk_root is None:
            return None
        probe = sdk_root / "features" / "compute_capability.exe"
        if not probe.is_file():
            return None
        try:
            completed = subprocess.run(
                [str(probe)],
                cwd=str(probe.parent),
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        value = (completed.stdout or "").strip().splitlines()
        return value[-1].strip() if value else None

    def _feature_install_hint(self, info: VsrInstallInfo) -> str:
        if info.install_script is None:
            return ""
        gpu = info.compute_capability or "<gpu>"
        features = (
            "nvvfxvideosuperres,nvvfxtransfer,nvvfxupscale,nvvfxdenoising,"
            "nvvfxgreenscreen,nvvfxbackgroundblur,nvvfxrelighting,nvvfxaigsrelighting"
        )
        return (
            f"cd /d {info.install_script.parent} && "
            f"powershell -ExecutionPolicy Bypass -File .\\install_feature.ps1 "
            f"-gpu {gpu} -features {features}"
        )
