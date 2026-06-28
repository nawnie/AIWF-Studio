from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings


@dataclass(frozen=True)
class PipelineInfo:
    id: str
    label: str
    kind: str
    engine: str
    summary: str
    ready: bool = True
    message: str = "ready"
    launch_backend: str | None = None


class PipelineRegistry:
    """User-facing pipeline catalog.

    Engines are isolated runtimes. Pipelines are the method a runtime uses:
    Diffusers, ONNX Runtime, GGUF, or future backends. This registry is metadata
    only and must stay import-safe for Settings and startup.
    """

    def __init__(self, flags: RuntimeFlags, settings: UserSettings) -> None:
        self.flags = flags
        self.settings = settings

    def image_pipelines(self) -> list[PipelineInfo]:
        qwen_nunchaku = self._qwen_nunchaku_pipeline()
        return [
            PipelineInfo(
                id="diffusers",
                label="Diffusers pipeline (default)",
                kind="image",
                engine="Studio Image Engine",
                summary="Reference SD/SDXL path with broad model compatibility.",
                ready=True,
                launch_backend="diffusers",
            ),
            PipelineInfo(
                id="qwen-image",
                label="Qwen Image Diffusers pipeline",
                kind="image",
                engine="Studio Image Engine",
                summary="Full-folder Qwen Image 2512/2.x text-to-image route.",
                ready=True,
                message="available when a Qwen Image Diffusers snapshot is installed",
            ),
            qwen_nunchaku,
            PipelineInfo(
                id="sana",
                label="Sana Diffusers pipeline",
                kind="image",
                engine="Studio Image Engine",
                summary="Full-folder Sana and Sana Sprint 1024px text-to-image route.",
                ready=True,
                message="available when a Sana Diffusers snapshot is installed",
            ),
            self._onnx_pipeline(),
        ]

    def video_pipelines(self) -> list[PipelineInfo]:
        ltx2b = self._ltx2b_pipeline()
        ltx = self._ltx_pipeline()
        sana_video = self._sana_video_pipeline()
        return [
            PipelineInfo(
                id="wan-diffusers",
                label="Wan Diffusers pipeline",
                kind="video",
                engine="Wan Video Engine",
                summary="Reference Wan image-to-video path.",
                ready=True,
                message="available when Wan models are configured",
            ),
            PipelineInfo(
                id="wan-gguf",
                label="Wan GGUF pipeline",
                kind="video",
                engine="Wan Video Engine",
                summary="Quantized Wan path for lower VRAM experiments.",
                ready=True,
                message="available when GGUF high/low transformer files are configured",
            ),
            sana_video,
            ltx2b,
            ltx,
        ]

    def launch_choices(self) -> list[tuple[str, str]]:
        return [
            (pipeline.label, str(pipeline.launch_backend))
            for pipeline in self.image_pipelines()
            if pipeline.launch_backend is not None
        ]

    def status_markdown(self) -> str:
        lines = ["**Pipelines**"]
        for pipeline in [*self.image_pipelines(), *self.video_pipelines()]:
            state = "Ready" if pipeline.ready else "Needs setup"
            lines.append(f"- **{pipeline.label}:** {state} - {pipeline.message}")
            lines.append(f"  - Engine: `{pipeline.engine}`")
            lines.append(f"  - Type: `{pipeline.kind}`")
            lines.append(f"  - {pipeline.summary}")
        return "\n".join(lines)

    def _onnx_pipeline(self) -> PipelineInfo:
        root = self._onnx_root()
        ready = root.exists()
        return PipelineInfo(
            id="onnx",
            label="ONNX Runtime pipeline",
            kind="image",
            engine="Studio Image Engine",
            summary="Alternative image path using AIWF sampler math and ONNX model folders.",
            ready=ready,
            message=f"model folder found at {root}" if ready else f"model folder missing: {root}",
            launch_backend="onnx",
        )

    def _onnx_root(self) -> Path:
        raw = (self.settings.onnx_model_dir or "").strip()
        if raw:
            path = Path(raw)
            return path.resolve() if path.is_absolute() else (self.flags.data_dir / path).resolve()
        return (self.flags.resolved_models_dir() / "onnx").resolve()

    def _ltx_pipeline(self) -> PipelineInfo:
        from aiwf.core.domain.ltx import LTX_PIPELINE_ONE_STAGE, LtxVideoRequest
        from aiwf.services.pipeline_preflight import preflight_ltx_pipeline

        preflight = preflight_ltx_pipeline(
            self.flags,
            self.settings,
            request=LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE),
        )
        if preflight.ok:
            selected = preflight.metadata.get("selected_pipeline", "default")
            message = f"ready via isolated LTX worker ({selected})"
            if preflight.warnings:
                message = f"{message}; {preflight.warnings[0]}"
        else:
            blocking = [item.message for item in preflight.items if not item.ok]
            message = "; ".join(blocking)
        return PipelineInfo(
            id="ltx-2.3",
            label="LTX 2.3 worker pipeline",
            kind="video",
            engine="LTX 2.3 Video Engine",
            summary="Optional Lightricks LTX 2.3 text/image-to-video path in engines/ltx/.venv.",
            ready=preflight.ok,
            message=message or "enable/install the LTX worker in Settings",
        )

    def _ltx2b_pipeline(self) -> PipelineInfo:
        from aiwf.core.domain.ltx import LTX_PIPELINE_DIFFUSERS_2B, LtxVideoRequest
        from aiwf.services.pipeline_preflight import preflight_ltx_pipeline

        preflight = preflight_ltx_pipeline(
            self.flags,
            self.settings,
            request=LtxVideoRequest(pipeline=LTX_PIPELINE_DIFFUSERS_2B),
        )
        if preflight.ok:
            message = f"ready with {preflight.metadata.get('checkpoint_path', '')}"
        else:
            blocking = [item.message for item in preflight.items if not item.ok]
            message = "; ".join(blocking)
        return PipelineInfo(
            id="ltx-2b-diffusers",
            label="LTX 2B Diffusers pipeline",
            kind="video",
            engine="Studio Video Engine",
            summary="Local LTX 0.9.5 2B text-to-video route using Diffusers and local T5XXL weights.",
            ready=preflight.ok,
            message=message or "install the LTX 2B checkpoint and T5XXL text encoder",
        )

    def _sana_video_pipeline(self) -> PipelineInfo:
        from aiwf.services.pipeline_preflight import preflight_sana_video_pipeline

        preflight = preflight_sana_video_pipeline(self.flags, self.settings)
        installed = preflight.metadata.get("model_installed") == "true"
        if not preflight.ok:
            blocking = [item.message for item in preflight.items if not item.ok]
            message = "; ".join(blocking)
        elif installed:
            message = f"ready with model at {preflight.metadata.get('model_path', '')}"
        else:
            message = f"available when the SANA-Video snapshot is installed at {preflight.metadata.get('model_path', '')}"
        return PipelineInfo(
            id="sana-video",
            label="Sana Video Diffusers pipeline",
            kind="video",
            engine="Studio Video Engine",
            summary="Diffusers SANA-Video text/image-to-video route. Audio is a post-process path.",
            ready=preflight.ok and installed,
            message=message,
        )

    def _qwen_nunchaku_pipeline(self) -> PipelineInfo:
        from aiwf.services.qwen_nunchaku import QwenNunchakuService

        status = QwenNunchakuService(self.flags).status()
        message = "ready via isolated Qwen Nunchaku runtime" if status.ready else "; ".join(status.messages)
        return PipelineInfo(
            id="qwen-nunchaku",
            label="Qwen Image Nunchaku pipeline",
            kind="image",
            engine="Qwen Nunchaku Engine",
            summary="Single-transformer safetensors Qwen Image Lightning route with shared base components.",
            ready=status.ready,
            message=message or "install/download the Nunchaku runtime and transformer",
        )
