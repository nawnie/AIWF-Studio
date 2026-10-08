import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_audio_minimum.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_audio_minimum", SCRIPT_PATH)
bootstrap = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(bootstrap)


def _write_complete_musicgen(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in bootstrap.MUSICGEN_FILES:
        target = path / name
        if name == "model.safetensors":
            header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            target.write_bytes(struct.pack("<Q", len(header)) + header + b"data")
        else:
            target.write_text("{}", encoding="utf-8")


def _write_mmaudio_clip_cache(cache_root: Path) -> None:
    snapshot = (
        cache_root
        / "models--apple--DFN5B-CLIP-ViT-H-14-384"
        / "snapshots"
        / ("a" * 40)
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    refs = cache_root / "models--apple--DFN5B-CLIP-ViT-H-14-384" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text("a" * 40, encoding="utf-8")
    (snapshot / "open_clip_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "open_clip_pytorch_model.safetensors").write_bytes(b"clip weights")


def test_minimum_musicgen_reuses_a_complete_extra_model_root(tmp_path):
    shared_models = tmp_path / "shared"
    expected = shared_models / "audio" / "MusicGen" / "musicgen-small"
    _write_complete_musicgen(expected)

    result = bootstrap._download_musicgen(tmp_path / "studio", tmp_path / "studio-models", [shared_models])

    assert result == expected
    assert not (tmp_path / "studio-models" / "audio" / "MusicGen" / "musicgen-small").exists()


def test_minimum_musicgen_reuses_model_without_optional_special_tokens_map(tmp_path):
    from aiwf.core.config.settings import RuntimeFlags, UserSettings
    from aiwf.services.audio import AudioGenerationService

    shared_models = tmp_path / "shared"
    expected = shared_models / "audio" / "MusicGen" / "musicgen-small"
    _write_complete_musicgen(expected)
    (expected / "special_tokens_map.json").unlink()

    assert bootstrap._find_musicgen_small([shared_models]) == expected
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path / "studio", extra_model_dirs=[shared_models]),
        UserSettings(),
    )
    assert service._musicgen_variant_ready("small") is True


def test_minimum_musicgen_downloads_missing_model_into_configured_models_root(tmp_path, monkeypatch):
    models_dir = tmp_path / "configured-models"
    calls = []

    def fake_snapshot_download(*, repo_id, local_dir, allow_patterns):
        calls.append((repo_id, Path(local_dir), allow_patterns))
        _write_complete_musicgen(Path(local_dir))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))

    result = bootstrap._download_musicgen(tmp_path / "studio", models_dir, [])

    expected = models_dir / "audio" / "MusicGen" / "musicgen-small"
    assert result == expected
    assert calls == [("facebook/musicgen-small", expected, list(bootstrap.MUSICGEN_FILES))]


def test_minimum_mmaudio_imports_shared_assets_and_skips_weight_download(tmp_path, monkeypatch):
    studio = tmp_path / "studio"
    shared_models = tmp_path / "shared-models"
    source_root = shared_models / "MMAudio"
    from aiwf.services.audio import _is_nonempty_file, _mmaudio_variant_files

    shared_bytes = {}
    for relative in _mmaudio_variant_files("small_16k"):
        source = source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        contents = f"shared asset: {relative}".encode()
        source.write_bytes(contents)
        shared_bytes[source] = contents

    engine_root = studio / "engines" / "audio" / "MMAudio"
    engine_python = studio / "engines" / "audio" / ".venv" / "Scripts" / "python.exe"
    calls = []
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: "powershell.exe")

    def fake_run(command, *, cwd, label):
        calls.append(label)
        if "isolated MMAudio engine" in label:
            engine_root.mkdir(parents=True, exist_ok=True)
            (engine_root / "demo.py").write_text("", encoding="utf-8")
            engine_python.parent.mkdir(parents=True, exist_ok=True)
            engine_python.write_text("", encoding="utf-8")
        elif "CLIP encoder" in label:
            _write_mmaudio_clip_cache(Path(command[-1]))

    monkeypatch.setattr(bootstrap, "_run", fake_run)

    result_root, result_python = bootstrap._install_mmaudio(studio, [shared_models])

    assert result_root == engine_root
    assert result_python == engine_python
    assert calls == [
        "Installing or repairing the isolated MMAudio engine",
        "Installing MMAudio CLIP encoder apple/DFN5B-CLIP-ViT-H-14-384",
    ]
    assert all(_is_nonempty_file(engine_root / relative) for relative in _mmaudio_variant_files("small_16k"))
    assert all(source.read_bytes() == contents for source, contents in shared_bytes.items())


def test_audio_service_passes_configured_roots_to_minimum_bootstrap(tmp_path, monkeypatch):
    from aiwf.core.config.settings import RuntimeFlags, UserSettings
    from aiwf.services import audio as audio_module
    from aiwf.services.audio import AudioGenerationService

    studio = tmp_path / "studio"
    models = tmp_path / "models"
    shared = tmp_path / "shared-models"
    (studio / "scripts").mkdir(parents=True)
    (studio / "scripts" / "bootstrap_audio_minimum.py").write_text("", encoding="utf-8")
    service = AudioGenerationService(
        RuntimeFlags(data_dir=studio, models_dir=models, extra_model_dirs=[shared]),
        UserSettings(),
    )
    captured = {}
    monkeypatch.setattr(
        audio_module.subprocess,
        "run",
        lambda command, **_kwargs: captured.update(command=command) or SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr=""),
    )
    monkeypatch.setattr(service, "setup_status", lambda *, deep: {"deep": deep})

    result = service.install_minimum()
    assert result == {"deep": True, "receipt": {"ok": True}}
    command = captured["command"]
    assert command[command.index("--models-dir") + 1] == str(models.resolve())
    assert command[command.index("--extra-model-dir") + 1] == str(shared.resolve())


def test_minimum_mmaudio_runs_its_installer_when_shared_variant_is_missing(tmp_path, monkeypatch):
    studio = tmp_path / "studio"
    engine_root = studio / "engines" / "audio" / "MMAudio"
    engine_python = studio / "engines" / "audio" / ".venv" / "Scripts" / "python.exe"
    calls = []
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: "powershell.exe")

    def fake_run(command, *, cwd, label):
        calls.append((command, label))
        if "isolated MMAudio engine" in label:
            engine_root.mkdir(parents=True, exist_ok=True)
            (engine_root / "demo.py").write_text("", encoding="utf-8")
            engine_python.parent.mkdir(parents=True, exist_ok=True)
            engine_python.write_text("", encoding="utf-8")
        elif "CLIP encoder" in label:
            _write_mmaudio_clip_cache(Path(command[-1]))

    monkeypatch.setattr(bootstrap, "_run", fake_run)

    bootstrap._install_mmaudio(studio, [tmp_path / "empty-shared-root"])

    assert [label for _command, label in calls] == [
        "Installing or repairing the isolated MMAudio engine",
        "Installing MMAudio CLIP encoder apple/DFN5B-CLIP-ViT-H-14-384",
        "Downloading missing MMAudio Small 16 kHz model assets",
    ]


def test_minimum_mmaudio_reuses_existing_clip_hub_snapshot_without_download(tmp_path, monkeypatch):
    from aiwf.services.audio import _find_mmaudio_clip_hub_cache

    cache_root = tmp_path / "shared-models" / "hub"
    _write_mmaudio_clip_cache(cache_root)

    result = bootstrap._install_mmaudio_clip_assets(tmp_path / "engine-python.exe", [cache_root.parent])

    assert result == cache_root.resolve()
    assert _find_mmaudio_clip_hub_cache([cache_root.parent]) == result


def test_minimum_mmaudio_reuses_audio_mmaudio_hub_layout_without_download(tmp_path, monkeypatch):
    expected = tmp_path / "audio" / "MMAudio" / ".cache" / "huggingface" / "hub"
    _write_mmaudio_clip_cache(expected)
    monkeypatch.setattr(bootstrap, "_run", lambda *_args, **_kwargs: pytest.fail("unexpected CLIP download"))

    result = bootstrap._install_mmaudio_clip_assets(tmp_path / "engine-python.exe", [tmp_path])

    assert result == expected.resolve()


def test_minimum_mmaudio_refuses_to_write_inside_a_configured_model_root(tmp_path):
    studio = tmp_path / "studio"
    engine_root = studio / "engines" / "audio" / "MMAudio"

    with pytest.raises(RuntimeError, match="inside a configured read-only model root"):
        bootstrap._import_shared_mmaudio([studio.resolve()], engine_root)

    assert not engine_root.exists()
