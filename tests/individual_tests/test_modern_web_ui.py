import json
from pathlib import Path
from types import SimpleNamespace

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.models import Checkpoint, SamplerInfo
from aiwf.web.modern.app import create_modern_web_ui
from aiwf.web.modern.dataset_reference import filter_records, load_manifest, resolve_dataset_dir


class _Devices:
    def describe(self):
        return "CPU (test)"


class _Generation:
    backend = SimpleNamespace(devices=_Devices())

    def list_checkpoints(self):
        return [
            Checkpoint(
                id="ckpt-a",
                title="Test Checkpoint",
                filename="test.safetensors",
                path="models/test.safetensors",
            )
        ]

    def list_loras(self):
        return []

    def list_vaes(self):
        return []

    def list_samplers(self):
        return [SamplerInfo(id="euler_a", label="Euler a")]

    def recent_jobs(self, _limit=12):
        return []


def _ctx(tmp_path: Path):
    settings = UserSettings()
    return SimpleNamespace(
        flags=RuntimeFlags(data_dir=tmp_path, theme="dark"),
        settings=settings,
        settings_path=tmp_path / "config.json",
        generation=_Generation(),
        save_settings=lambda: (tmp_path / "config.json").write_text(settings.model_dump_json(), encoding="utf-8"),
    )


def test_modern_user_settings_defaults():
    settings = UserSettings()

    assert settings.modern_onboarding_seen is False
    assert settings.github_avatar_url == "https://github.com/nawnie.png?size=160"


def test_create_modern_web_ui_smoke(tmp_path):
    demo, theme, css, js = create_modern_web_ui(_ctx(tmp_path))

    labels = {
        (component.get("props") or {}).get("label")
        for component in demo.config.get("components", [])
    }
    assert "Prompt" in labels
    assert "Checkpoint" in labels
    assert "Reference search" in labels
    assert theme is not None
    assert "modern-sidebar" in css
    assert js == ""


def test_dataset_reference_resolves_latest_pointer(tmp_path):
    dataset = tmp_path / "dataset"
    captions = dataset / "training_image_caption"
    captions.mkdir(parents=True)
    (captions / "fig.png").write_bytes(b"png")
    (captions / "fig.txt").write_text("caption", encoding="utf-8")
    (dataset / "manifest.csv").write_text(
        "id,image_id,type,chapter_title,caption,file_name,caption_file,split,status,notes\n"
        "id1,fig1,diagram,Layout,Blocks layout,training_image_caption/fig.png,training_image_caption/fig.txt,train,ok,note\n",
        encoding="utf-8",
    )
    pointer = tmp_path / "latest.json"
    pointer.write_text(json.dumps({"dataset_dir": str(dataset)}), encoding="utf-8")

    assert resolve_dataset_dir(pointer) == dataset
    records = load_manifest(dataset)
    assert len(records) == 1
    assert records[0].chapter_title == "Layout"
    assert filter_records(records, query="blocks", asset_type="diagram") == records
    assert filter_records(records, query="missing", asset_type="diagram") == []
