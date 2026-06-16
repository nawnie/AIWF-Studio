from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobRecord, JobState
from aiwf.plugins.registry import PluginRegistry
from aiwf.services.controlnet import ControlNetService
from aiwf.services.plot import PlotAxis, PlotRequest, PlotService


def test_controlnet_service_scans_models(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    model_dir = flags.resolved_models_dir() / "ControlNet"
    model_dir.mkdir(parents=True)
    (model_dir / "control.safetensors").write_bytes(b"fake")
    (model_dir / "ignore.txt").write_text("nope", encoding="utf-8")

    service = ControlNetService(flags)

    models = service.list_models()
    assert [model.id for model in models] == ["control"]
    assert service.list_modules()[0] == "none"


class FakeGeneration:
    def __init__(self):
        self.requests = []

    def submit(self, request, init_images=None, mask_images=None):
        self.requests.append(request)
        image = Image.new("RGB", (16, 16), "purple")
        return JobRecord(
            request=request,
            state=JobState.COMPLETED,
            result=GenerationResult(
                job_id=uuid4(),
                images=[image],
                seeds=[request.seed],
                infotexts=[f"seed {request.seed}"],
                mode=request.mode,
            ),
        )


def test_plot_service_runs_axis_combinations():
    generation = FakeGeneration()
    service = PlotService(generation)

    result = service.run(
        PlotRequest(
            base=GenerationRequest(mode=GenerationMode.TXT2IMG, prompt="cat"),
            axes=[PlotAxis(field="seed", values=[1, 2])],
        )
    )

    assert [request.seed for request in generation.requests] == [1, 2]
    assert result.labels == ["seed=1", "seed=2"]
    assert result.grid is not None


def test_plugin_registry_records_new_style_plugin_metadata(tmp_path: Path):
    plugin_dir = tmp_path / "plugins" / "demo"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "\n".join(
            [
                "class DemoPlugin:",
                "    name = 'Demo'",
                "    version = '1.2.3'",
                "    def on_load(self, ctx):",
                "        ctx.loaded_demo = True",
                "plugin = DemoPlugin()",
            ]
        ),
        encoding="utf-8",
    )
    ctx = type("Ctx", (), {})()
    registry = PluginRegistry()

    registry.discover(tmp_path / "plugins", ctx)

    assert ctx.loaded_demo is True
    assert registry.loaded == ["demo"]
    assert registry.list_plugins()[0].name == "Demo"
    assert registry.list_plugins()[0].version == "1.2.3"
