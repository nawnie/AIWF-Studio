from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from aiwf.core.bridge import InfotextBridge
from aiwf.core.config.launch import LaunchSettings, launch_settings_path, load_launch_settings, save_launch_settings
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.events.bus import EventBus
from aiwf.core.events.types import AppStarted
from aiwf.infrastructure.storage.filesystem import FilesystemImageStore
from aiwf.plugins.registry import PluginRegistry
from aiwf.services.engine_supervisor import EngineSupervisor, get_supervisor
from aiwf.services.faceswap import FaceSwapService
from aiwf.services.controlnet import ControlNetService
from aiwf.services.failure_archive import FailureArchiveService
from aiwf.services.generation import GenerationService
from aiwf.services.genlog import GenerationLogService
from aiwf.services.metadata import MetadataService
from aiwf.services.model_catalog import ModelCatalogService
from aiwf.services.benchmark_receipts import BenchmarkReceiptService
from aiwf.services.optimization import CapabilityDetector, OptimizationPlanner
from aiwf.services.optimization_diagnostics import OptimizationDiagnosticsService
from aiwf.services.plot import PlotService
from aiwf.services.prompt_processor import PromptProcessorService
from aiwf.services.queue import JobQueue
from aiwf.services.tags import TagService
from aiwf.services.workflow import WorkflowService

@dataclass
class AppContext:
    """Composition root — the only place that wires dependencies."""

    flags: RuntimeFlags
    settings: UserSettings
    events: EventBus
    plugins: PluginRegistry
    supervisor: EngineSupervisor
    generation: GenerationService
    genlog: GenerationLogService
    failure_archive: FailureArchiveService
    enhance: EnhanceService
    controlnet: ControlNetService
    faceswap: FaceSwapService
    plots: PlotService
    models: ModelCatalogService
    model_download: ModelDownloadService
    capabilities: CapabilityDetector
    optimization_planner: OptimizationPlanner
    benchmark_receipts: BenchmarkReceiptService
    optimization_diagnostics: OptimizationDiagnosticsService
    prompts: PromptProcessorService
    tags: TagService
    workflows: WorkflowService
    segment: SegmentService
    infotext_bridge: InfotextBridge
    settings_path: Path
    launch_settings_path: Path
    runtime_port: int | None = None
    dev: Any = None  # DevDiagnostics when install_dev_diagnostics runs

    def save_settings(self) -> None:
        self.settings_path.write_text(
            self.settings.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def load_settings(self) -> None:
        _load_user_settings(self.settings, self.settings_path)

    def load_launch_settings(self) -> LaunchSettings | None:
        return load_launch_settings(self.launch_settings_path)

    def save_launch_settings(self, launch: LaunchSettings, *, project_root: Path | None = None) -> None:
        save_launch_settings(self.launch_settings_path, launch)


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


def _load_user_settings(settings: UserSettings, settings_path: Path) -> None:
    if not settings_path.exists():
        return
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        if data.get("show_progress_every_n_steps", 1) == 0:
            data["enable_live_preview"] = False
            data["show_progress_every_n_steps"] = 1
        loaded = UserSettings.model_validate(data)
        for name in UserSettings.model_fields:
            setattr(settings, name, getattr(loaded, name))
        settings.apply_token_env()
    except (json.JSONDecodeError, ValueError):
        pass


def _create_device_manager(flags: RuntimeFlags):
    from aiwf.infrastructure.torch.devices import DeviceManager

    return DeviceManager(flags)


def _create_diffusers_backend(flags: RuntimeFlags, devices):
    from aiwf.infrastructure.diffusers.backend import DiffusersBackend

    return DiffusersBackend(flags, devices)


def _create_onnx_backend(flags: RuntimeFlags, settings: UserSettings):
    from aiwf.infrastructure.onnx.backend import ONNXBackend

    onnx_root = Path(settings.onnx_model_dir) if settings.onnx_model_dir else flags.resolved_models_dir() / "onnx"
    backend = ONNXBackend(
        models_root=onnx_root,
        provider=flags.onnx_provider,  # type: ignore[arg-type]
        device_id=0,
    )
    logger.info("Inference backend: ONNX Runtime (provider=%s, models=%s)", flags.onnx_provider, onnx_root)
    return backend


def _create_inference_backend(flags: RuntimeFlags, settings: UserSettings, devices):
    if flags.inference_backend == "onnx":
        return _create_onnx_backend(flags, settings)
    backend = _create_diffusers_backend(flags, devices)
    logger.info("Inference backend: Diffusers")
    return backend


def _create_enhance_service(flags: RuntimeFlags, settings: UserSettings, devices, store, *, supervisor):
    from aiwf.services.enhance import EnhanceService

    return EnhanceService(flags, settings, devices, store, supervisor=supervisor)


def _create_model_download_service(flags: RuntimeFlags):
    from aiwf.services.model_download import ModelDownloadService

    return ModelDownloadService(flags)


def _create_segment_service(flags: RuntimeFlags, settings: UserSettings, devices, *, supervisor):
    from aiwf.services.segment import SegmentService

    return SegmentService(flags, settings, devices, supervisor=supervisor)


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

    # Sort any models the user dropped into "models to sort" into their correct
    # folders by reading file headers, before the model library is indexed.
    try:
        from aiwf.infrastructure.model_sorter import sort_inbox_on_startup
        from aiwf.infrastructure.model_inventory import invalidate_model_inventory_cache

        sort_actions = sort_inbox_on_startup(flags)
        if any(action.moved for action in sort_actions):
            invalidate_model_inventory_cache()
    except Exception:
        logger.warning("Startup model sort failed; continuing without it.", exc_info=True)

    events = EventBus()
    devices = _create_device_manager(flags)
    devices.log_status()
    from aiwf.infrastructure.torch.attention import _ensure_cuda_attention_bootstrapped, describe_best_attention_stack

    _ensure_cuda_attention_bootstrapped()
    logger.info("Attention stack: %s", describe_best_attention_stack(flags))
    settings_path = flags.data_dir / "config.json"
    settings = UserSettings()
    _load_user_settings(settings, settings_path)
    supervisor = get_supervisor()

    # Propagate engine feature flags to environment so sub-modules pick them up.
    _engine_env = {
        "AIWF_CUDA_GRAPHS":    ("1" if flags.cuda_graphs    else "0"),
        "AIWF_TORCHAO":        ("1" if flags.torchao        else "0"),
        "AIWF_TORCH_COMPILE":  ("1" if flags.torch_compile  else "0"),
        "AIWF_CHANNELS_LAST":  ("1" if flags.channels_last  else "0"),
        "AIWF_FP8":            ("1" if flags.fp8_quant      else "0"),
        "AIWF_NVENC":          ("1" if flags.nvenc          else "0"),
        "AIWF_HEVC":           ("1" if flags.hevc           else "0"),
    }
    _path_env = {
        "AIWF_NVIDIA_VFX_SDK_ROOT": flags.nvidia_vfx_sdk_root,
        "AIWF_VSR_VIDEO_EFFECTS_APP": flags.vsr_video_effects_app,
        "AIWF_VSR_UPSCALE_APP": flags.vsr_upscale_app,
        "AIWF_VIDEOFX_DENOISE_APP": flags.videofx_denoise_app,
        "AIWF_VIDEOFX_AIGS_APP": flags.videofx_aigs_app,
        "AIWF_VIDEOFX_RELIGHT_APP": flags.videofx_relight_app,
        "AIWF_VSR_MODEL_DIR": flags.vsr_model_dir,
    }
    import os as _os
    for _k, _v in _engine_env.items():
        _os.environ.setdefault(_k, _v)
    for _k, _v in _path_env.items():
        if _v is not None:
            _os.environ.setdefault(_k, str(_v))

    backend = _create_inference_backend(flags, settings, devices)
    metadata = MetadataService()
    queue = JobQueue(events)
    store = FilesystemImageStore(flags.resolved_output_dir(), settings=settings)
    failure_archive = FailureArchiveService(flags.resolved_output_dir())
    genlog = GenerationLogService(flags.resolved_output_dir(), enabled=flags.genlog)

    capabilities = CapabilityDetector()
    optimization_planner = OptimizationPlanner()
    benchmark_receipts = BenchmarkReceiptService(flags.resolved_output_dir())
    optimization_diagnostics = OptimizationDiagnosticsService(
        flags=flags,
        settings=settings,
        detector=capabilities,
        planner=optimization_planner,
        output_dir=flags.resolved_output_dir(),
    )

    generation = GenerationService(
        backend,
        store,
        metadata,
        queue,
        events,
        settings,
        settings_path=settings_path,
        supervisor=supervisor,
        optimization_planner=optimization_planner,
        failure_archive=failure_archive,
        genlog=genlog,
    )
    enhance = _create_enhance_service(flags, settings, devices, store, supervisor=supervisor)
    controlnet = ControlNetService(flags)
    controlnet.ensure_dir()
    faceswap = FaceSwapService(flags, supervisor=supervisor)
    faceswap.ensure_dir()
    plots = PlotService(generation)
    models = ModelCatalogService(generation, flags, settings)
    model_download = _create_model_download_service(flags)
    model_download.ensure_dirs()
    prompts = PromptProcessorService(flags, settings, models)
    prompts.ensure_dirs()
    generation.prompts = prompts
    segment = _create_segment_service(flags, settings, devices, supervisor=supervisor)
    workflows = WorkflowService(flags, settings, generation, enhance, segment)
    workflows.ensure_dir()
    ctx = AppContext(
        flags=flags,
        settings=settings,
        events=events,
        plugins=PluginRegistry(),
        supervisor=supervisor,
        generation=generation,
        genlog=genlog,
        failure_archive=failure_archive,
        enhance=enhance,
        controlnet=controlnet,
        faceswap=faceswap,
        plots=plots,
        models=models,
        model_download=model_download,
        capabilities=capabilities,
        optimization_planner=optimization_planner,
        benchmark_receipts=benchmark_receipts,
        optimization_diagnostics=optimization_diagnostics,
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
    from aiwf.dev.diagnostics import install_dev_diagnostics

    ctx.dev = install_dev_diagnostics(ctx)
    events.publish(AppStarted())
    return ctx
