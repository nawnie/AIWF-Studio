from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

from aiwf.core.bridge import InfotextBridge
from aiwf.core.config.launch import LaunchSettings, launch_settings_path, load_launch_settings, save_launch_settings
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.events.bus import EventBus
from aiwf.core.events.types import AppStarted
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.infrastructure.storage.filesystem import FilesystemImageStore
from aiwf.infrastructure.torch.devices import DeviceManager
from aiwf.plugins.registry import PluginRegistry
from aiwf.services.enhance import EnhanceService
from aiwf.services.faceswap import FaceSwapService
from aiwf.services.controlnet import ControlNetService
from aiwf.services.generation import GenerationService
from aiwf.services.metadata import MetadataService
from aiwf.services.model_catalog import ModelCatalogService
from aiwf.services.model_download import ModelDownloadService
from aiwf.services.plot import PlotService
from aiwf.services.prompt_processor import PromptProcessorService
from aiwf.services.queue import JobQueue
from aiwf.services.tags import TagService
from aiwf.services.segment import SegmentService
from aiwf.services.workflow import WorkflowService


@dataclass
class AppContext:
    """Composition root — the only place that wires dependencies."""

    flags: RuntimeFlags
    settings: UserSettings
    events: EventBus
    plugins: PluginRegistry
    generation: GenerationService
    enhance: EnhanceService
    controlnet: ControlNetService
    faceswap: FaceSwapService
    plots: PlotService
    models: ModelCatalogService
    model_download: ModelDownloadService
    prompts: PromptProcessorService
    tags: TagService
    workflows: WorkflowService
    segment: SegmentService
    infotext_bridge: InfotextBridge
    settings_path: Path
    launch_settings_path: Path
    runtime_port: int | None = None

    def save_settings(self) -> None:
        self.settings_path.write_text(
            self.settings.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def load_settings(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if data.get("show_progress_every_n_steps", 1) == 0:
                data["enable_live_preview"] = False
                data["show_progress_every_n_steps"] = 1
            loaded = UserSettings.model_validate(data)
            # Update in place: services hold a reference to this settings object,
            # so replacing it would leave them reading stale defaults.
            for name in UserSettings.model_fields:
                setattr(self.settings, name, getattr(loaded, name))
            self.settings.apply_token_env()
        except (json.JSONDecodeError, ValueError):
            pass

    def load_launch_settings(self) -> LaunchSettings | None:
        return load_launch_settings(self.launch_settings_path)

    def save_launch_settings(self, launch: LaunchSettings, *, project_root: Path | None = None) -> None:
        save_launch_settings(self.launch_settings_path, launch)
        if project_root is not None:
            from aiwf.core.config.launch import write_webui_settings_bat

            write_webui_settings_bat(project_root, launch)


def _check_transformers_compat() -> None:
    try:
        import transformers

        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if version >= (5, 0):
            logger.error(
                "transformers %s is incompatible with diffusers checkpoint loading. "
                "Run: pip install \"transformers>=4.44,<5\"",
                transformers.__version__,
            )
    except Exception:
        pass


def build_context(flags: RuntimeFlags | None = None) -> AppContext:
    _check_transformers_compat()
    flags = flags or RuntimeFlags()
    flags.data_dir = flags.data_dir.resolve()

    models_dir = flags.resolved_models_dir()
    for directory in (
        models_dir,
        flags.resolved_ckpt_dir(),
        flags.resolved_output_dir(),
        models_dir / "RealESRGAN",
        models_dir / "GFPGAN",
        models_dir / "Codeformer",
        models_dir / "ControlNet",
        models_dir / "insightface",
        flags.data_dir / "prompts",
        flags.data_dir / "wildcards",
        flags.data_dir / "workflows",
        models_dir / "sam",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    events = EventBus()
    devices = DeviceManager(flags)
    devices.log_status()
    backend = DiffusersBackend(flags, devices)
    metadata = MetadataService()
    queue = JobQueue(events)
    settings_path = flags.data_dir / "config.json"
    settings = UserSettings()
    store = FilesystemImageStore(flags.resolved_output_dir(), settings=settings)

    generation = GenerationService(backend, store, metadata, queue, events, settings)
    enhance = EnhanceService(flags, settings, devices, store)
    controlnet = ControlNetService(flags)
    controlnet.ensure_dir()
    faceswap = FaceSwapService(flags)
    faceswap.ensure_dir()
    plots = PlotService(generation)
    models = ModelCatalogService(generation, flags, settings)
    model_download = ModelDownloadService(flags)
    model_download.ensure_dirs()
    prompts = PromptProcessorService(flags, settings, models)
    prompts.ensure_dirs()
    generation.prompts = prompts
    segment = SegmentService(flags, settings, devices)
    segment.ensure_default_models()
    workflows = WorkflowService(flags, settings, generation, enhance, segment)
    workflows.ensure_dir()
    ctx = AppContext(
        flags=flags,
        settings=settings,
        events=events,
        plugins=PluginRegistry(),
        generation=generation,
        enhance=enhance,
        controlnet=controlnet,
        faceswap=faceswap,
        plots=plots,
        models=models,
        model_download=model_download,
        prompts=prompts,
        tags=TagService(settings, flags.resolved_output_dir()),
        workflows=workflows,
        segment=segment,
        infotext_bridge=InfotextBridge(),
        settings_path=settings_path,
        launch_settings_path=launch_settings_path(flags.data_dir),
    )
    ctx.load_settings()
    if ctx.prompts.ensure_default_styles():
        ctx.save_settings()
    ctx.plugins.discover(flags.data_dir / "plugins", ctx)
    events.publish(AppStarted())
    return ctx
