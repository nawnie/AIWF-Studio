from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.infrastructure.diffusers.checkpoints import diffusers_dir_has_required_local_files


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
            self._diffusers_snapshot_pipeline(
                id="qwen-image",
                label="Qwen Image Diffusers pipeline",
                summary="Full-folder Qwen Image 2512/2.x text-to-image route.",
                subdir=Path("qwen-image") / "Diffusers",
                class_tokens=("qwenimagepipeline", "qwen image", "qwen-image", "qwenimage"),
                install_label="a complete Qwen Image Diffusers snapshot",
            ),
            qwen_nunchaku,
            self._diffusers_snapshot_pipeline(
                id="sana",
                label="Sana Diffusers pipeline",
                summary="Full-folder Sana and Sana Sprint 1024px text-to-image route.",
                subdir=Path("sana") / "Diffusers",
                class_tokens=("sanasprintpipeline", "sanapipeline", "sana"),
                install_label="a complete Sana Diffusers snapshot",
            ),
            self._onnx_pipeline(),
        ]

    def video_pipelines(self) -> list[PipelineInfo]:
        ltx2b = self._ltx2b_pipeline()
        ltx = self._ltx_pipeline()
        sana_video = self._sana_video_pipeline()
        return [
            self._wan_diffusers_pipeline(),
            self._wan_gguf_pipeline(),
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

    def _diffusers_snapshot_pipeline(
        self,
        *,
        id: str,
        label: str,
        summary: str,
        subdir: Path,
        class_tokens: tuple[str, ...],
        install_label: str,
    ) -> PipelineInfo:
        root = (self.flags.resolved_models_dir() / subdir).resolve()
        ready, message = self._diffusers_snapshot_status(root, class_tokens, install_label)
        return PipelineInfo(
            id=id,
            label=label,
            kind="image",
            engine="Studio Image Engine",
            summary=summary,
            ready=ready,
            message=message,
        )

    def _diffusers_snapshot_status(
        self,
        root: Path,
        class_tokens: tuple[str, ...],
        install_label: str,
    ) -> tuple[bool, str]:
        if not root.exists():
            return False, f"install {install_label} under {root}"

        candidates: list[Path] = []
        unreadable: list[Path] = []
        for index_path in sorted(root.rglob("model_index.json"), key=lambda item: str(item).lower()):
            folder = index_path.parent
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                unreadable.append(folder)
                continue
            class_name = str(payload.get("_class_name") or "")
            text = f"{class_name} {folder.name}".lower().replace("_", "")
            if class_tokens and not any(token.replace("_", "").lower() in text for token in class_tokens):
                continue
            candidates.append(folder)

        if not candidates:
            if unreadable:
                return False, f"Diffusers snapshot model_index.json is unreadable at {unreadable[0]}"
            return False, f"install {install_label} under {root}"

        incomplete: list[Path] = []
        for folder in candidates:
            if diffusers_dir_has_required_local_files(folder):
                return True, f"ready with model at {folder}"
            incomplete.append(folder)

        return False, f"incomplete Diffusers snapshot; missing local shard files under {incomplete[0]}"

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

    def _wan_diffusers_pipeline(self) -> PipelineInfo:
        from aiwf.services.pipeline_preflight import preflight_wan_pipeline

        preflight = preflight_wan_pipeline(self.flags, self.settings)
        if preflight.ok:
            message = f"ready with {preflight.metadata.get('model_id', '')}"
        else:
            blocking = [item.message for item in preflight.items if not item.ok]
            message = "; ".join(blocking)
        failure_hint = self._latest_failure_hint(kind="video", stage_token="wan", runtime_modes=("fast_5b",))
        if failure_hint:
            message = f"{message}; {failure_hint}" if message else failure_hint
        return PipelineInfo(
            id="wan-diffusers",
            label="Wan Diffusers pipeline",
            kind="video",
            engine="Wan Video Engine",
            summary="Reference Wan image-to-video path.",
            ready=preflight.ok,
            message=message or "install/configure local Wan fast 5B model, components, and VAE",
        )

    def _wan_gguf_pipeline(self) -> PipelineInfo:
        from aiwf.core.domain.wan import WAN_RUNTIME_HIGH_LOW, WanI2VRequest
        from aiwf.services.wan import WanService

        service = WanService(self.flags, self.settings)
        high, low = self._first_wan_gguf_pair(service.list_local_models())
        if not high or not low:
            return PipelineInfo(
                id="wan-gguf",
                label="Wan GGUF pipeline",
                kind="video",
                engine="Wan Video Engine",
                summary="Quantized Wan path for lower VRAM experiments.",
                ready=False,
                message="install/configure matched Wan GGUF high and low transformer files",
            )
        preflight = service.preflight(
            WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW, high_noise_model_id=high, low_noise_model_id=low)
        )
        message = (
            f"ready with high={Path(high).name}, low={Path(low).name}"
            if preflight.ok
            else preflight.message() or "Wan GGUF pair preflight failed"
        )
        failure_hint = self._latest_failure_hint(kind="video", stage_token="wan", runtime_modes=(WAN_RUNTIME_HIGH_LOW,))
        if failure_hint:
            message = f"{message}; {failure_hint}"
        return PipelineInfo(
            id="wan-gguf",
            label="Wan GGUF pipeline",
            kind="video",
            engine="Wan Video Engine",
            summary="Quantized Wan path for lower VRAM experiments.",
            ready=preflight.ok,
            message=message,
        )

    @staticmethod
    def _first_wan_gguf_pair(models: list[str]) -> tuple[str | None, str | None]:
        high = None
        low = None
        for item in models:
            name = Path(item).name.lower()
            if not name.endswith(".gguf"):
                continue
            if high is None and "high" in name:
                high = item
            elif low is None and "low" in name:
                low = item
            if high and low:
                break
        return high, low

    def _latest_failure_hint(
        self,
        *,
        kind: str,
        stage_token: str,
        runtime_modes: tuple[str, ...] = (),
    ) -> str:
        index_path = self.flags.resolved_output_dir() / "failures" / "index.jsonl"
        if not index_path.is_file():
            return ""
        try:
            lines = index_path.read_text(encoding="utf-8").splitlines()[-200:]
        except OSError:
            return ""
        wanted_modes = {item.lower() for item in runtime_modes if item}
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("kind") or "").lower() != kind:
                continue
            if stage_token and stage_token.lower() not in str(payload.get("stage") or "").lower():
                continue
            request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
            extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
            runtime_mode = str(request.get("runtime_mode") or extra.get("runtime_mode") or "").lower()
            if wanted_modes and runtime_mode not in wanted_modes:
                continue
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            error_type = str(error.get("type") or "Error")
            message = " ".join(str(error.get("message") or "").split())
            if len(message) > 140:
                message = f"{message[:137]}..."
            created_at = str(payload.get("created_at") or "").replace("T", " ").replace("Z", " UTC")
            prefix = f"last runtime failure {created_at}: " if created_at else "last runtime failure: "
            return f"{prefix}{error_type}: {message}".strip()
        return ""

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
