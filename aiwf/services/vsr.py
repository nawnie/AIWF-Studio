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
from aiwf.core.domain.vsr import VsrOptions, VsrResult
from aiwf.infrastructure.video import VideoError, VideoProcessor

logger = logging.getLogger(__name__)


class VsrUnavailable(RuntimeError):
    """Raised when NVIDIA RTX VSR cannot run in this environment."""


@dataclass(frozen=True)
class VsrInstallInfo:
    app_path: Path | None
    sdk_root: Path | None
    model_dir: Path | None

    @property
    def available(self) -> bool:
        return self.app_path is not None and self.app_path.is_file()


class VsrService:
    """NVIDIA RTX Video Super Resolution wrapper.

    The first implementation uses NVIDIA's Video Effects SDK sample binary
    (`VideoEffectsApp.exe`) instead of binding the C API directly. That keeps
    the Python app boot-safe while we validate the SDK install path locally.
    """

    def __init__(self, flags: RuntimeFlags, settings: UserSettings, supervisor=None) -> None:
        self.flags = flags
        self.settings = settings
        self.supervisor = supervisor

    def install_info(self) -> VsrInstallInfo:
        app_path = self._resolve_app_path()
        sdk_root = self._resolve_sdk_root(app_path)
        model_dir = self._resolve_model_dir(sdk_root)
        return VsrInstallInfo(app_path=app_path, sdk_root=sdk_root, model_dir=model_dir)

    def available(self) -> bool:
        return self.install_info().available

    def folder_help(self) -> str:
        info = self.install_info()
        if info.available:
            return f"Detected NVIDIA Video Effects SDK: `{info.app_path}`"
        return (
            "NVIDIA RTX VSR needs the NVIDIA Video Effects SDK runtime and "
            "`VideoEffectsApp.exe`. Install the SDK, or set "
            "`AIWF_VSR_VIDEO_EFFECTS_APP` to the sample executable."
        )

    def _output_path(self, input_video: str | Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(input_video).stem or "video"
        sub = getattr(self.settings, "vsr_output_subdir", "vsr-videos")
        root = self.flags.resolved_output_dir() / sub
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{stem}_vsr_{stamp}.mp4"

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
        if not info.available or info.app_path is None:
            raise VsrUnavailable(self.folder_help())

        processor = VideoProcessor()
        try:
            input_info = processor.probe(src)
        except VideoError as exc:
            raise VsrUnavailable(str(exc)) from exc
        if input_info.width <= 0 or input_info.height <= 0:
            raise VsrUnavailable("Could not read input video dimensions for VSR.")

        scale = self._valid_scale(float(options.scale))
        target_height = int(round(input_info.height * scale))
        dest = Path(output_path) if output_path else self._output_path(src)
        dest.parent.mkdir(parents=True, exist_ok=True)

        command = self._command(
            info,
            src,
            dest,
            effect=options.effect,
            target_height=target_height,
            mode=int(options.mode),
            strength=float(options.strength),
            codec=options.codec,
        )
        tenant_job_id = f"vsr_{uuid.uuid4().hex[:8]}"
        if self.supervisor is not None:
            switch = self.supervisor.request_switch(
                EngineSwitchRequest(
                    target=EngineTenant.VIDEO,
                    reason="NVIDIA RTX VSR",
                    job_id=tenant_job_id,
                )
            )
            if not switch.ok:
                raise VsrUnavailable(f"GPU busy: {switch.message}")

        try:
            completed = subprocess.run(
                command,
                cwd=str(info.app_path.parent),
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
                        reason="NVIDIA RTX VSR complete",
                        job_id=tenant_job_id,
                    )
                )

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise VsrUnavailable(f"NVIDIA RTX VSR failed ({completed.returncode}): {detail}")
        if not dest.is_file() or dest.stat().st_size <= 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise VsrUnavailable(f"NVIDIA RTX VSR did not produce an output file. {detail}")

        try:
            output_info = processor.probe(dest)
        except VideoError as exc:
            raise VsrUnavailable(f"NVIDIA RTX VSR output is not readable: {dest}") from exc

        infotext = (
            f"NVIDIA RTX VSR {options.effect} {scale:g}x "
            f"({input_info.width}x{input_info.height}->{output_info.width}x{output_info.height})"
        )
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
        assert info.app_path is not None
        selected_effect = effect if effect in {"SuperRes", "Upscale"} else "SuperRes"
        command = [
            str(info.app_path),
            f"--in_file={src}",
            f"--out_file={dest}",
            f"--effect={selected_effect}",
            f"--resolution={max(1, int(target_height))}",
            "--show=false",
            f"--codec={codec or 'H264'}",
        ]
        if selected_effect == "SuperRes":
            command.append(f"--mode={1 if int(mode) else 0}")
        else:
            command.append(f"--strength={max(0.0, min(1.0, float(strength))):.3f}")
        if info.model_dir is not None and info.model_dir.is_dir():
            command.append(f"--model_dir={info.model_dir}")
        return command

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
        return [p for p in entries if p.is_dir()]

    @staticmethod
    def _valid_scale(scale: float) -> float:
        allowed = (1.3333333, 1.5, 2.0, 3.0, 4.0)
        return min(allowed, key=lambda item: abs(item - scale))

    def _resolve_app_path(self) -> Path | None:
        explicit = os.environ.get("AIWF_VSR_VIDEO_EFFECTS_APP", "").strip()
        if explicit:
            path = Path(explicit)
            return path if path.is_file() else None
        for root in self._candidate_roots():
            candidates = [
                root / "samples" / "VideoEffectsApp" / "VideoEffectsApp.exe",
                root / "samples" / "VideoEffectsApp" / "bin" / "VideoEffectsApp.exe",
                root / "VideoEffectsApp.exe",
                root / "bin" / "VideoEffectsApp.exe",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            if root.is_dir():
                try:
                    found = next(root.rglob("VideoEffectsApp.exe"), None)
                except OSError:
                    found = None
                if found is not None and found.is_file():
                    return found
        return None

    def _resolve_sdk_root(self, app_path: Path | None) -> Path | None:
        explicit = os.environ.get("AIWF_NVIDIA_VFX_SDK_ROOT", "").strip()
        if explicit and Path(explicit).is_dir():
            return Path(explicit)
        if app_path is not None:
            for parent in [app_path.parent, *app_path.parents]:
                if (
                    (parent / "models").is_dir()
                    or (parent / "samples" / "VideoEffectsApp").is_dir()
                    or parent.name.lower() in {"oss", "nvidia video effects", "nvidia-vfx-sdk", "rtx-video-sdk"}
                ):
                    return parent
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
        roots = [
            os.environ.get("AIWF_NVIDIA_VFX_SDK_ROOT", "").strip(),
            r"C:\Program Files\NVIDIA Corporation\NVIDIA Video Effects",
            r"C:\Program Files\NVIDIA Corporation\NVIDIA VFX SDK",
            str(self.flags.data_dir / "engines" / "nvidia-vfx-sdk"),
            str(self.flags.data_dir / "engines" / "rtx-video-sdk"),
        ]
        return [Path(root) for root in roots if root]
