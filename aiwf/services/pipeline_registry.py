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
            self._onnx_pipeline(),
        ]

    def video_pipelines(self) -> list[PipelineInfo]:
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
