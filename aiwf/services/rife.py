from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.rife import RifeOptions, RifeResult
from aiwf.infrastructure.rife import RifeUnavailable, interpolate_video_file
from aiwf.infrastructure.rife.backend import list_rife_checkpoints, resolve_vfi_root
from aiwf.infrastructure.torch.devices import DeviceManager

logger = logging.getLogger(__name__)


class RifeService:
    def __init__(self, flags: RuntimeFlags, settings: UserSettings, devices: DeviceManager) -> None:
        self.flags = flags
        self.settings = settings
        self.devices = devices

    def folder_help(self) -> str:
        root = resolve_vfi_root()
        models_dir = self.flags.resolved_models_dir() / "rife"
        lines = [
            f"RIFE uses **ComfyUI-Frame-Interpolation** (Practical-RIFE). "
            f"Set `AIWF_RIFE_VFI_ROOT` if it is not auto-detected.",
            f"Local checkpoint folder (optional): `{models_dir}`",
        ]
        if root:
            lines.append(f"Detected VFI pack: `{root}`")
        else:
            lines.append(
                "VFI pack **not detected** — install "
                "[ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) "
                "or point `AIWF_RIFE_VFI_ROOT` at it."
            )
        return "  \n".join(lines)

    def list_checkpoints(self) -> list[str]:
        return list_rife_checkpoints(resolve_vfi_root())

    def default_checkpoint(self) -> str:
        choices = self.list_checkpoints()
        for preferred in ("rife47.pth", "rife49.pth", "rife46.pth"):
            if preferred in choices:
                return preferred
        return choices[0] if choices else "rife47.pth"

    def _output_path(self, input_video: str | Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(input_video).stem
        sub = getattr(self.settings, "rife_output_subdir", "rife-videos")
        return self.flags.resolved_output_dir() / sub / f"{stem}_rife_{stamp}.mp4"

    def interpolate(
        self,
        input_video: str | Path,
        options: RifeOptions,
        *,
        output_path: str | Path | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> RifeResult:
        dest = Path(output_path) if output_path else self._output_path(input_video)
        try:
            out, in_frames, out_frames, in_fps, out_fps, width, height = interpolate_video_file(
                input_video,
                dest,
                ckpt_name=options.ckpt_name,
                multiplier=options.multiplier,
                scale_factor=options.scale_factor,
                fast_mode=options.fast_mode,
                ensemble=options.ensemble,
                clear_cache_every_n_frames=options.clear_cache_every_n_frames,
                max_input_frames=options.max_input_frames,
                device=self.devices.device(),
                on_progress=on_progress,
            )
        except RifeUnavailable:
            raise
        except Exception as exc:
            logger.exception("RIFE interpolation failed")
            raise RifeUnavailable(str(exc)) from exc

        infotext = (
            f"RIFE {options.ckpt_name} x{options.multiplier} "
            f"({in_frames}→{out_frames} frames, {in_fps:.1f}→{out_fps:.1f} fps)"
        )
        return RifeResult(
            output_path=str(out),
            input_frames=in_frames,
            output_frames=out_frames,
            input_fps=in_fps,
            output_fps=out_fps,
            width=width,
            height=height,
            infotext=infotext,
            message=f"Saved {out_frames} frames at {width}x{height}, {out_fps:.1f} fps → {out}",
        )