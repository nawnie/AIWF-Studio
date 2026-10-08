from __future__ import annotations

from pathlib import Path
import json
import struct
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions, AudioGenerationResult
from aiwf.services.audio import AudioGenerationService, AudioUnavailable

_audio_headroom_issue = AudioGenerationService._audio_headroom_issue


@pytest.fixture(autouse=True)
def _keep_audio_route_tests_independent_of_workstation_vram(monkeypatch):
    monkeypatch.setattr(AudioGenerationService, "_audio_headroom_issue", lambda self, *args, **kwargs: None)


def _write_safetensors_fixture(path: Path) -> None:
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"data")


def _write_minimum_mmaudio(root: Path) -> tuple[Path, Path]:
    engine = root / "engines" / "audio" / "MMAudio"
    engine.mkdir(parents=True, exist_ok=True)
    (engine / "demo.py").write_text("print('demo')", encoding="utf-8")
    for relative in (
        Path("weights") / "mmaudio_small_16k.pth",
        Path("ext_weights") / "v1-16.pth",
        Path("ext_weights") / "best_netG.pt",
        Path("ext_weights") / "synchformer_state_dict.pth",
    ):
        path = engine / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
    python = root / "engines" / "audio" / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    _write_mmaudio_clip_cache(root / "models" / "hub")
    return engine, python


def _write_mmaudio_clip_cache(cache_root: Path) -> Path:
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
    return cache_root


def _write_shared_mmaudio_variant(root: Path, variant: str) -> dict[Path, Path]:
    from aiwf.services.audio import _mmaudio_variant_files

    assets = {}
    for relative in _mmaudio_variant_files(variant):
        path = root / "MMAudio" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"shared:{variant}:{relative}".encode())
        assets[relative] = path
    return assets


def test_audio_options_defaults():
    opts = AudioGenerationOptions(prompt="calm synth score")
    assert opts.kind == "music"
    assert opts.model_id == "facebook/musicgen-small"
    assert opts.duration_seconds == 8.0


def test_audio_model_support_paths_are_exact_to_selected_variant(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())

    musicgen = service.model_support_paths("facebook/musicgen-medium")
    mmaudio = service.model_support_paths("mmaudio:large_44k_v2")

    assert any(Path(path).name == "model.safetensors" and "musicgen-medium" in path for path in musicgen)
    assert all("musicgen-small" not in path for path in musicgen)
    assert any(Path(path).name == "mmaudio_large_44k_v2.pth" for path in mmaudio)
    assert all("mmaudio_small_16k.pth" not in path for path in mmaudio)
    assert service.model_support_paths("mmaudio:unknown") == []


def test_musicgen_cache_identity_changes_when_local_model_snapshot_changes(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    snapshot = tmp_path / "models" / "audio" / "MusicGen" / "musicgen-small"
    snapshot.mkdir(parents=True)
    weight = snapshot / "model.safetensors"
    weight.write_bytes(b"old weights")
    before = service._musicgen_cache_key("facebook/musicgen-small", str(snapshot))
    original_stat = weight.stat()
    weight.write_bytes(b"replacement weights")
    import os
    os.utime(weight, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))

    after = service._musicgen_cache_key("facebook/musicgen-small", str(snapshot))

    assert before != after


def test_prepare_musicgen_loads_and_swaps_cached_variant_without_generation(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    loaded = []

    class FakeModel:
        device = "cpu"

    def fake_load(model_id, _source):
        model = FakeModel()
        service._model = (object(), model)
        service._model_key = ("transformers", "music", model_id)
        loaded.append(model_id)
        return service._model

    monkeypatch.setattr(service, "_musicgen_variant_ready", lambda _variant: True)
    monkeypatch.setattr(service, "_musicgen_model_source", lambda model_id: f"local:{model_id}")
    monkeypatch.setattr(service, "setup_status", lambda **_kwargs: {"musicDependenciesReady": True})
    monkeypatch.setattr(service, "_load_transformers_musicgen", fake_load)
    monkeypatch.setattr(service, "_device_string", lambda: "cpu")

    first = service.prepare(kind="music", model_id="facebook/musicgen-small")
    second = service.prepare(kind="music", model_id="facebook/musicgen-medium")

    assert first["resident"] is True
    assert second["resident"] is True
    assert loaded == ["facebook/musicgen-small", "facebook/musicgen-medium"]
    assert service._model_key == ("transformers", "music", "facebook/musicgen-medium")
    assert not hasattr(service._model[1], "generate")


def test_prepare_musicgen_defers_before_loader_when_headroom_is_low(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    load_calls = []
    monkeypatch.setattr(service, "_musicgen_variant_ready", lambda _variant: True)
    monkeypatch.setattr(service, "_musicgen_model_source", lambda _model_id: "local:musicgen-small")
    monkeypatch.setattr(service, "setup_status", lambda **_kwargs: {"musicDependenciesReady": True})
    monkeypatch.setattr(service, "_device_string", lambda: "cuda")
    monkeypatch.setattr(
        service,
        "_audio_headroom_issue",
        lambda *_args, **_kwargs: "Audio model loading deferred: 2.4 GB VRAM is free; the selected route needs at least 3.5 GB headroom.",
    )
    monkeypatch.setattr(service, "_load_transformers_musicgen", lambda *_args: load_calls.append("load"))

    with pytest.raises(AudioUnavailable, match="2.4 GB VRAM is free"):
        service.prepare(kind="music", model_id="facebook/musicgen-small")

    assert load_calls == []


def test_audio_headroom_uses_device_wide_memory_measurement(tmp_path: Path, monkeypatch):
    import sys

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda _name: SimpleNamespace(type="cuda", index=0),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(AudioGenerationService, "_device_string", lambda _self: "cuda:0")
    monkeypatch.setattr("aiwf.services.gpu_memory.measured_cuda_free_bytes", lambda *_args: int(2.4 * 1024**3))
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())

    issue = _audio_headroom_issue(service, "facebook/musicgen-small")

    assert issue is not None
    assert "2.4 GB VRAM is free" in issue


def test_external_mmaudio_headroom_uses_nvidia_smi_even_when_studio_device_is_cpu(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path, cpu=True), UserSettings())
    monkeypatch.setattr(service, "_device_string", lambda: "cpu")
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.nvidia_smi_free_bytes",
        lambda: int(2.4 * 1024**3),
    )

    issue = _audio_headroom_issue(service, "mmaudio:small_16k", external_worker=True)

    assert issue is not None
    assert "2.4 GB VRAM is free" in issue


def test_mmaudio_render_defers_before_subprocess_when_headroom_is_low(tmp_path: Path, monkeypatch):
    engine, python = _write_minimum_mmaudio(tmp_path)
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(service, "_mmaudio_root", lambda: engine)
    monkeypatch.setattr(service, "_audio_engine_python", lambda: python)
    monkeypatch.setattr(service, "_mmaudio_runtime_import_error", lambda: "")
    monkeypatch.setattr(
        service,
        "_audio_headroom_issue",
        lambda *_args, **_kwargs: "Audio model loading deferred: 2.4 GB VRAM is free; the selected route needs at least 6.0 GB headroom.",
    )
    monkeypatch.setattr("aiwf.services.audio.subprocess.run", lambda *_args, **_kwargs: pytest.fail("MMAudio subprocess launched"))

    with pytest.raises(AudioUnavailable, match="2.4 GB VRAM is free"):
        service._generate_mmaudio_text_audio(
            AudioGenerationOptions(prompt="wind", kind="sfx", model_id="mmaudio:small_16k"),
            tmp_path / "audio.flac",
        )


def test_video_conditioned_mmaudio_defers_before_subprocess_when_headroom_is_low(tmp_path: Path, monkeypatch):
    engine, python = _write_minimum_mmaudio(tmp_path)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fixture")
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(service, "_mmaudio_root", lambda: engine)
    monkeypatch.setattr(service, "_audio_engine_python", lambda: python)
    monkeypatch.setattr(service, "_mmaudio_runtime_import_error", lambda: "")
    monkeypatch.setattr(
        service,
        "_audio_headroom_issue",
        lambda *_args, **_kwargs: "Audio model loading deferred: 2.4 GB VRAM is free; the selected route needs at least 6.0 GB headroom.",
    )
    monkeypatch.setattr("aiwf.services.audio.subprocess.run", lambda *_args, **_kwargs: pytest.fail("MMAudio subprocess launched"))

    with pytest.raises(AudioUnavailable, match="2.4 GB VRAM is free"):
        service._generate_mmaudio_video_audio(
            video,
            AudioGenerationOptions(prompt="wind", kind="video_audio", model_id="mmaudio:small_16k"),
            tmp_path / "video_audio.flac",
        )


def test_prepare_musicgen_restores_same_cached_model_after_render_parks_it_on_cpu(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    loads = []

    class FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

    model = FakeModel()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(_source, **_kwargs):
            return object()

    class FakeMusicgen:
        @staticmethod
        def from_pretrained(_source, **_kwargs):
            loads.append("facebook/musicgen-small")
            return model

    fake_torch = SimpleNamespace(
        float16=object(),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoProcessor=FakeProcessor, MusicgenForConditionalGeneration=FakeMusicgen),
    )
    monkeypatch.setattr(service, "_musicgen_variant_ready", lambda _variant: True)
    monkeypatch.setattr(service, "_musicgen_model_source", lambda _model_id: "local:musicgen-small")
    monkeypatch.setattr(service, "setup_status", lambda **_kwargs: {"musicDependenciesReady": True})
    monkeypatch.setattr(service, "_device_string", lambda: "cuda")

    first = service.prepare(kind="music", model_id="facebook/musicgen-small")
    assert first["resident"] is True
    assert model.device == "cuda"

    service._park_cached_model_on_cpu()
    assert model.device == "cpu"

    second = service.prepare(kind="music", model_id="facebook/musicgen-small")

    assert second["resident"] is True
    assert model.device == "cuda"
    assert loads == ["facebook/musicgen-small"]


def test_musicgen_failed_variant_switch_keeps_previous_cache_and_allows_retry(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    load_attempts = []

    class FakeModel:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        @staticmethod
        def from_pretrained(source, **_kwargs):
            return f"processor:{source}"

    class FakeMusicgen:
        @staticmethod
        def from_pretrained(source, **_kwargs):
            load_attempts.append(source)
            if source == "local:facebook/musicgen-medium" and load_attempts.count(source) == 1:
                raise RuntimeError("injected model-load failure")
            return FakeModel()

    fake_torch = SimpleNamespace(
        float16=object(),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoProcessor=FakeProcessor, MusicgenForConditionalGeneration=FakeMusicgen),
    )
    monkeypatch.setattr(service, "_device_string", lambda: "cpu")

    old_cache = service._load_transformers_musicgen("facebook/musicgen-small", "local:facebook/musicgen-small")
    old_key = service._model_key

    with pytest.raises(RuntimeError, match="injected model-load failure"):
        service._load_transformers_musicgen("facebook/musicgen-medium", "local:facebook/musicgen-medium")

    assert service._model is old_cache
    assert service._model_key == old_key
    assert service._musicgen_model_is_resident("facebook/musicgen-small") is True
    assert service._musicgen_model_is_resident("facebook/musicgen-medium") is False

    processor, model = service._load_transformers_musicgen("facebook/musicgen-medium", "local:facebook/musicgen-medium")

    assert processor == "processor:local:facebook/musicgen-medium"
    assert model.device == "cpu"
    assert service._musicgen_model_is_resident("facebook/musicgen-medium") is True
    assert load_attempts == ["local:facebook/musicgen-small", "local:facebook/musicgen-medium", "local:facebook/musicgen-medium"]


def test_musicgen_cpu_parking_confirmation_requires_matching_cached_model(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())

    class FakeModel:
        device = "cpu"

    service._model = (object(), FakeModel())
    service._model_key = ("transformers", "music", "facebook/musicgen-small")

    assert service.musicgen_model_is_parked_on_cpu("facebook/musicgen-small") is True
    assert service.musicgen_model_is_parked_on_cpu("facebook/musicgen-medium") is False


def test_prepare_mmaudio_reports_ready_without_claiming_residency_or_loading(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(service, "_mmaudio_variant_ready", lambda variant: variant == "small_16k")
    monkeypatch.setattr(service, "_mmaudio_runtime_import_error", lambda: "")

    result = service.prepare(kind="sfx", model_id="mmaudio:small_16k")

    assert result["ready"] is True
    assert result["resident"] is None
    assert "per render" in result["detail"]
    assert service._model is None
    assert service._model_key is None


def test_prepare_mmaudio_releases_prior_musicgen_and_image_models(tmp_path: Path, monkeypatch):
    from contextlib import contextmanager

    events = []

    class Supervisor:
        @contextmanager
        def tenant_session(self, tenant, **_kwargs):
            events.append(("tenant", tenant.value))
            yield

    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path),
        UserSettings(),
        supervisor=Supervisor(),
        unload_image_models=lambda: events.append(("image-unload", True)),
    )
    service._model = object()
    service._model_key = ("transformers", "music", "facebook/musicgen-small")
    monkeypatch.setattr(service, "_mmaudio_variant_ready", lambda variant: variant == "small_16k")
    monkeypatch.setattr(service, "_mmaudio_runtime_import_error", lambda: "")

    result = service.prepare(kind="sfx", model_id="mmaudio:small_16k")

    assert result["resident"] is None
    assert service._model is None and service._model_key is None
    assert events == [("tenant", "audio"), ("image-unload", True)]


def test_prepare_mmaudio_rejects_broken_isolated_import_before_claiming_ready(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(service, "_mmaudio_variant_ready", lambda variant: variant == "small_16k")
    monkeypatch.setattr(service, "_mmaudio_runtime_import_error", lambda: "ModuleNotFoundError: mmaudio")

    with pytest.raises(AudioUnavailable, match="MMAudio runtime is not ready: ModuleNotFoundError: mmaudio"):
        service.prepare(kind="sfx", model_id="mmaudio:small_16k")


def test_mmaudio_runtime_probe_executes_demo_help_in_isolated_environment(tmp_path: Path, monkeypatch):
    engine, python = _write_minimum_mmaudio(tmp_path)
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr="", stdout="usage")

    monkeypatch.setattr("aiwf.services.audio.subprocess.run", fake_run)

    assert service._mmaudio_runtime_import_error() == ""
    assert observed["command"] == [str(python), str(engine / "demo.py"), "--help"]
    assert observed["kwargs"]["cwd"] == str(engine)


@pytest.mark.parametrize("conditioned_on_video", [False, True])
def test_mmaudio_render_refuses_broken_runtime_before_launching_demo(tmp_path: Path, monkeypatch, conditioned_on_video: bool):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    engine = tmp_path / "MMAudio"
    engine.mkdir()
    (engine / "demo.py").write_text("", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr(service, "_mmaudio_root", lambda: engine)
    monkeypatch.setattr(service, "_audio_engine_python", lambda: python)
    monkeypatch.setattr(service, "_mmaudio_variant_ready", lambda _variant: True)
    monkeypatch.setattr(service, "_mmaudio_runtime_import_error", lambda: "missing runtime module")
    monkeypatch.setattr("aiwf.services.audio.subprocess.run", lambda *_args, **_kwargs: pytest.fail("demo launched"))
    options = AudioGenerationOptions(
        prompt="rain ambience",
        kind="video_audio" if conditioned_on_video else "sfx",
        model_id="mmaudio:small_16k",
    )

    with pytest.raises(AudioUnavailable, match="MMAudio runtime is not ready: missing runtime module"):
        if conditioned_on_video:
            service._generate_mmaudio_video_audio(tmp_path / "clip.mp4", options, tmp_path / "audio.wav")
        else:
            service._generate_mmaudio_text_audio(options, tmp_path / "audio.wav")


@pytest.mark.parametrize(("kind", "model_id"), [("music", "mmaudio:small_16k"), ("sfx", "facebook/musicgen-small"), ("video_audio", "mmaudio:small_16k"), ("sfx", "facebook/audiogen-medium")])
def test_prepare_rejects_unsupported_kind_model_pairs(tmp_path: Path, kind: str, model_id: str):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())

    with pytest.raises(AudioUnavailable):
        service.prepare(kind=kind, model_id=model_id)


def test_musicgen_minimum_does_not_accept_zero_byte_placeholder_weights(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    root = service._minimum_musicgen_root()
    root.mkdir(parents=True)
    from aiwf.services.audio import _MUSICGEN_MINIMUM_FILES

    for name in _MUSICGEN_MINIMUM_FILES:
        (root / name).touch()
    with pytest.raises(AudioUnavailable, match="not installed"):
        service._musicgen_model_source("facebook/musicgen-small")

    for name in _MUSICGEN_MINIMUM_FILES:
        if name == "model.safetensors":
            _write_safetensors_fixture(root / name)
        else:
            (root / name).write_bytes(b"valid fixture")
    assert service._musicgen_model_source("facebook/musicgen-small") == str(root)


def test_musicgen_variants_install_to_local_roots_and_generation_never_falls_back_to_hub(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    from aiwf.services.audio import _MUSICGEN_MINIMUM_FILES

    calls = []

    def fake_snapshot_download(*, repo_id, local_dir, allow_patterns):
        assert set(allow_patterns) == {*_MUSICGEN_MINIMUM_FILES, "special_tokens_map.json"}
        assert not any(name.endswith(".bin") for name in allow_patterns)
        calls.append((repo_id, local_dir))
        root = Path(local_dir)
        for name in _MUSICGEN_MINIMUM_FILES:
            if name == "model.safetensors":
                _write_safetensors_fixture(root / name)
            else:
                (root / name).write_bytes(b"fixture weights")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    receipt = service.install_musicgen_variant("medium")

    assert receipt["modelId"] == "facebook/musicgen-medium"
    assert receipt["installed"] is True
    assert Path(receipt["path"]) == service._musicgen_root("medium")
    assert calls == [("facebook/musicgen-medium", str(service._musicgen_root("medium")))]
    assert service._musicgen_model_source("facebook/musicgen-medium") == str(service._musicgen_root("medium"))
    with pytest.raises(AudioUnavailable, match="Unsupported MusicGen variant"):
        service.install_musicgen_variant("../../outside")
    with pytest.raises(AudioUnavailable, match="not installed"):
        service._musicgen_model_source("facebook/musicgen-melody")


def test_musicgen_readiness_rejects_truncated_safetensors(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    root = service._minimum_musicgen_root()
    root.mkdir(parents=True)
    from aiwf.services.audio import _MUSICGEN_MINIMUM_FILES

    for name in _MUSICGEN_MINIMUM_FILES:
        (root / name).write_bytes(b"fixture")
    _write_safetensors_fixture(root / "model.safetensors")
    assert service._musicgen_variant_ready("small") is True
    (root / "model.safetensors").write_bytes((root / "model.safetensors").read_bytes()[:-1])
    assert service._musicgen_variant_ready("small") is False
    assert "model.safetensors (invalid or truncated)" in service._musicgen_missing_files("small")


def test_musicgen_discovers_complete_model_from_configured_shared_root(tmp_path: Path):
    shared_root = tmp_path / "shared-models"
    shared_musicgen = shared_root / "audio" / "MusicGen" / "musicgen-small"
    shared_musicgen.mkdir(parents=True)
    from aiwf.services.audio import _MUSICGEN_MINIMUM_FILES

    for name in _MUSICGEN_MINIMUM_FILES:
        if name == "model.safetensors":
            _write_safetensors_fixture(shared_musicgen / name)
        else:
            (shared_musicgen / name).write_bytes(b"fixture")
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path / "studio", extra_model_dirs=[shared_root]),
        UserSettings(),
    )

    assert service._musicgen_variant_ready("small") is True
    assert service._musicgen_model_source("facebook/musicgen-small") == str(shared_musicgen)


def test_musicgen_install_reuses_shared_model_without_writing_to_it(tmp_path: Path, monkeypatch):
    shared_root = tmp_path / "shared-models"
    shared_musicgen = shared_root / "audio" / "MusicGen" / "musicgen-small"
    shared_musicgen.mkdir(parents=True)
    from aiwf.services.audio import _MUSICGEN_MINIMUM_FILES

    for name in _MUSICGEN_MINIMUM_FILES:
        if name == "model.safetensors":
            _write_safetensors_fixture(shared_musicgen / name)
        else:
            (shared_musicgen / name).write_bytes(b"fixture")
    before = {path.name: path.read_bytes() for path in shared_musicgen.iterdir()}
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path / "studio", extra_model_dirs=[shared_root]),
        UserSettings(),
    )
    calls = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs)),
    )

    receipt = service.install_musicgen_variant("small")

    assert receipt["path"] == str(shared_musicgen)
    assert calls == []
    assert {path.name: path.read_bytes() for path in shared_musicgen.iterdir()} == before


def test_musicgen_install_writes_to_primary_root_when_shared_variant_missing(tmp_path: Path, monkeypatch):
    shared_root = tmp_path / "shared-models"
    shared_root.mkdir()
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path / "studio", extra_model_dirs=[shared_root]),
        UserSettings(),
    )
    from aiwf.services.audio import _MUSICGEN_MINIMUM_FILES

    calls = []

    def fake_snapshot_download(*, repo_id, local_dir, allow_patterns):
        assert set(allow_patterns) == {*_MUSICGEN_MINIMUM_FILES, "special_tokens_map.json"}
        calls.append((repo_id, local_dir))
        root = Path(local_dir)
        for name in _MUSICGEN_MINIMUM_FILES:
            if name == "model.safetensors":
                _write_safetensors_fixture(root / name)
            else:
                (root / name).write_bytes(b"fixture")

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake_snapshot_download))
    receipt = service.install_musicgen_variant("medium")

    expected = service.flags.resolved_models_dir() / "audio" / "MusicGen" / "musicgen-medium"
    assert Path(receipt["path"]) == expected
    assert calls == [("facebook/musicgen-medium", str(expected))]


def test_audio_generation_and_install_share_one_nonblocking_model_lock(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    from aiwf.services import audio as audio_service

    assert audio_service._AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(AudioUnavailable, match="render or model setup"):
            service.install_musicgen_variant("medium")
        with pytest.raises(AudioUnavailable, match="setup or another audio render"):
            service.generate(AudioGenerationOptions(prompt="test"))
    finally:
        audio_service._AUDIO_MODEL_OPERATION_LOCK.release()


def test_video_conditioned_audio_and_install_share_one_model_lock(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    from aiwf.services import audio as audio_service

    assert audio_service._AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(AudioUnavailable, match="setup or another audio render"):
            service.generate_video_audio(
                video,
                AudioGenerationOptions(prompt="room tone", kind="video_audio", model_id="mmaudio:small_16k"),
            )
        with pytest.raises(AudioUnavailable, match="render or model setup"):
            service.install_mmaudio_variant("large_44k_v2")
    finally:
        audio_service._AUDIO_MODEL_OPERATION_LOCK.release()


@pytest.mark.parametrize("tenant_available", [True, False])
def test_modality_switch_evicts_cached_audio_only_under_audio_tenant(tmp_path: Path, tenant_available: bool):
    from aiwf.core.domain.engine import EngineTenant

    events = []

    class FakeSupervisor:
        @contextmanager
        def tenant_session(self, tenant, *, reason):
            events.append((tenant, reason))
            if not tenant_available:
                raise RuntimeError("GPU is owned by another tenant")
            yield "audio-eviction"

    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path), UserSettings(), supervisor=FakeSupervisor()
    )
    service._model = (object(), SimpleNamespace(to=lambda _device: None))
    service._model_key = ("transformers", "music", "facebook/musicgen-small")
    unloads = []

    def fake_unload():
        unloads.append(True)
        service._model = None
        service._model_key = None

    service.unload = fake_unload

    released = service.release_cached_model_for_modality_switch()

    assert events and events[0][0] is EngineTenant.AUDIO
    assert released is tenant_available
    assert unloads == ([True] if tenant_available else [])
    if not tenant_available:
        assert service._model is not None
        assert service._model_key == ("transformers", "music", "facebook/musicgen-small")


def test_modality_switch_does_not_evict_during_audio_operation(tmp_path: Path):
    from aiwf.services import audio as audio_service

    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    service._model = (object(), SimpleNamespace())
    service._model_key = ("transformers", "music", "facebook/musicgen-small")
    assert audio_service._AUDIO_MODEL_OPERATION_LOCK.acquire(blocking=False)
    try:
        assert service.release_cached_model_for_modality_switch() is False
        assert service._model is not None
        assert service._model_key == ("transformers", "music", "facebook/musicgen-small")
    finally:
        audio_service._AUDIO_MODEL_OPERATION_LOCK.release()


def test_audiocraft_model_download_path_is_explicitly_blocked(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    with pytest.raises(AudioUnavailable, match="no explicit local model installer"):
        service._generate_audiocraft(
            AudioGenerationOptions(prompt="test", kind="sfx", model_id="facebook/audiogen-medium"),
            tmp_path / "output.wav",
        )


def test_audio_generation_missing_prompt_raises(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())

    with pytest.raises(AudioUnavailable, match="audio prompt"):
        service.generate(AudioGenerationOptions(prompt=""))


def test_audio_mux_builds_ffmpeg_command(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    out = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    captured = {}

    def fake_run(command, **_kwargs):
        if command[0] == "ffmpeg":
            captured["command"] = command
            Path(command[-1]).write_bytes(b"muxed")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        captured["probe"] = command
        return SimpleNamespace(
            returncode=0,
            stdout='{"format":{"format_name":"mov,mp4,m4a,3gp,3g2,mj2","duration":"1.0"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}',
            stderr="",
        )

    with patch("aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"), patch(
        "aiwf.services.audio.shutil.which", return_value="ffprobe"
    ), patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        result = service.mux_audio(video, audio, output_path=out)

    assert captured["command"][:5] == ["ffmpeg", "-y", "-i", str(video), "-i"]
    assert str(audio) in captured["command"]
    assert captured["command"][captured["command"].index("-af") + 1] == "apad"
    assert "-shortest" in captured["command"]
    assert captured["command"][-1] != str(out)
    assert captured["command"][-1].endswith(".partial.mp4")
    assert captured["probe"][-1] == captured["command"][-1]
    assert "format=format_name,duration:stream=codec_type" in captured["probe"]
    assert result.output_path == str(out)
    assert out.read_bytes() == b"muxed"
    assert list(tmp_path.glob(".*.partial.mp4")) == []


@pytest.mark.parametrize(
    ("exit_code", "write_bytes", "probe_stdout", "error"),
    [
        (1, b"partial", "", "Audio mux failed"),
        (0, None, "", "did not produce a non-empty output file"),
        (0, b"", "", "did not produce a non-empty output file"),
        (0, b"partial", '{"format":{},"streams":[]}', "positive finite duration"),
        (0, b"partial", "null", "malformed data"),
        (0, b"partial", "[]", "malformed data"),
        (0, b"partial", '{"format":{"format_name":"mp4","duration":"1.0"},"streams":[null]}', "malformed streams"),
        (0, b"partial", '{"format":{"format_name":"mp4","duration":"1.0"},"streams":[{"codec_type":"video"}]}', "readable container with video and audio"),
        (0, b"partial", '{"format":{"format_name":"mp4","duration":"1.0"},"streams":[{"codec_type":"audio"}]}', "readable container with video and audio"),
        (0, b"partial", '{"format":{"format_name":"mp4","duration":"0"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}', "positive finite duration"),
        (0, b"partial", '{"format":{"format_name":"mp4","duration":"NaN"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}', "positive finite duration"),
        (0, b"partial", '{"format":{"format_name":"mp4","duration":"inf"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}', "positive finite duration"),
        (0, b"partial", '{"format":{"format_name":"mp4","duration":"-1"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}', "positive finite duration"),
        (0, b"partial", '{"format":{"format_name":"mp4","duration":"N/A"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}', "positive finite duration"),
        (0, b"partial", '{"format":{"format_name":"mp4"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}', "positive finite duration"),
    ],
)
def test_audio_mux_failure_preserves_existing_destination(
    tmp_path: Path, exit_code: int, write_bytes: bytes | None, probe_stdout: str, error: str
):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video, audio, out = tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    out.write_bytes(b"previous mux")

    def fake_run(command, **_kwargs):
        if command[0] == "ffmpeg":
            if write_bytes is not None:
                Path(command[-1]).write_bytes(write_bytes)
            return SimpleNamespace(returncode=exit_code, stdout="", stderr="fixture mux failure")
        return SimpleNamespace(returncode=0, stdout=probe_stdout, stderr="")

    with patch("aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"), patch(
        "aiwf.services.audio.shutil.which", return_value="ffprobe"
    ), patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        with pytest.raises(AudioUnavailable, match=error):
            service.mux_audio(video, audio, output_path=out)

    assert out.read_bytes() == b"previous mux"
    assert list(tmp_path.glob(".*.partial.mp4")) == []


def test_audio_mux_fails_before_running_without_ffprobe(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video, audio, out = tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    out.write_bytes(b"previous mux")
    with patch("aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"), patch.object(
        service, "_resolve_ffprobe", return_value=None
    ), patch("aiwf.services.audio.subprocess.run") as run:
        with pytest.raises(AudioUnavailable, match="ffprobe is required"):
            service.mux_audio(video, audio, output_path=out)
    run.assert_not_called()
    assert out.read_bytes() == b"previous mux"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("timeout", "60-minute time limit"),
        ("launch", "could not start ffmpeg"),
    ],
)
def test_audio_mux_process_failure_preserves_destination_and_cleans_stage(
    tmp_path: Path, failure: str, message: str
):
    import subprocess

    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video, audio, out = tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    out.write_bytes(b"previous mux")
    error = subprocess.TimeoutExpired("ffmpeg", timeout=3600) if failure == "timeout" else FileNotFoundError("ffmpeg")
    with patch("aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"), patch.object(
        service, "_resolve_ffprobe", return_value="ffprobe"
    ), patch("aiwf.services.audio.subprocess.run", side_effect=error):
        with pytest.raises(AudioUnavailable, match=message):
            service.mux_audio(video, audio, output_path=out)
    assert out.read_bytes() == b"previous mux"
    assert list(tmp_path.glob(".*.partial.mp4")) == []


def test_audio_mux_probe_timeout_preserves_destination(tmp_path: Path):
    import subprocess

    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video, audio, out = tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    out.write_bytes(b"previous mux")

    def fake_run(command, **_kwargs):
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"partial mux")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(command, timeout=60)

    with patch("aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"), patch(
        "aiwf.services.audio.shutil.which", return_value="ffprobe"
    ), patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        with pytest.raises(AudioUnavailable, match="could not be validated"):
            service.mux_audio(video, audio, output_path=out)

    assert out.read_bytes() == b"previous mux"
    assert list(tmp_path.glob(".*.partial.mp4")) == []


def test_audio_mux_replace_failure_preserves_destination(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video, audio, out = tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    out.write_bytes(b"previous mux")

    def fake_run(command, **_kwargs):
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"new mux")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout='{"format":{"format_name":"mp4","duration":"1.0"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}',
            stderr="",
        )

    with patch("aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"), patch(
        "aiwf.services.audio.shutil.which", return_value="ffprobe"
    ), patch("aiwf.services.audio.subprocess.run", side_effect=fake_run), patch(
        "aiwf.services.audio.os.replace", side_effect=OSError("fixture replace failure")
    ):
        with pytest.raises(AudioUnavailable, match="could not be published"):
            service.mux_audio(video, audio, output_path=out)

    assert out.read_bytes() == b"previous mux"
    assert list(tmp_path.glob(".*.partial.mp4")) == []


def test_repeated_audio_muxes_get_unique_destinations(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video, audio = tmp_path / "video.mp4", tmp_path / "audio.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    def fake_run(command, **_kwargs):
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"muxed")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout='{"format":{"format_name":"mp4","duration":"1.0"},"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}',
            stderr="",
        )

    with patch("aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"), patch(
        "aiwf.services.audio.shutil.which", return_value="ffprobe"
    ), patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        first = service.mux_audio(video, audio)
        second = service.mux_audio(video, audio)

    assert first.output_path != second.output_path
    assert Path(first.output_path).read_bytes() == b"muxed"
    assert Path(second.output_path).read_bytes() == b"muxed"
    assert list((tmp_path / "outputs" / "audio-videos").glob(".*.partial.mp4")) == []


def test_video_audio_builds_mmaudio_command(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    service._mmaudio_runtime_import_error = lambda: ""
    video = tmp_path / "clip.mp4"
    out = tmp_path / "sound.flac"
    video.write_bytes(b"video")
    engine, python = _write_minimum_mmaudio(tmp_path)
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        captured["kwargs"] = _kwargs
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "clip.flac").write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        result = service.generate_video_audio(
            video,
            AudioGenerationOptions(
                prompt="cloth movement and footsteps",
                kind="video_audio",
                model_id="mmaudio:small_16k",
                duration_seconds=5,
                cfg_coef=4.5,
                steps=12,
                seed=123,
            ),
            output_path=out,
        )

    command = captured["command"]
    assert command[0] == str(python)
    assert Path(command[1]).name == "run_mmaudio_offline.py"
    assert command[2] == str(engine / "demo.py")
    assert captured["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert command[command.index("--variant") + 1] == "small_16k"
    assert command[command.index("--video") + 1] == str(video)
    assert command[command.index("--num_steps") + 1] == "12"
    assert "--skip_video_composite" in command
    assert result.output_path == str(out)
    assert result.kind == "video_audio"
    assert out.read_bytes() == b"audio"


def test_video_audio_accepts_single_alternate_mmaudio_flac(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    service._mmaudio_runtime_import_error = lambda: ""
    video = tmp_path / "clip.mp4"
    out = tmp_path / "sound.flac"
    video.write_bytes(b"video")
    engine = tmp_path / "engines" / "audio" / "MMAudio"
    engine.mkdir(parents=True)
    (engine / "demo.py").write_text("print('demo')", encoding="utf-8")
    python = tmp_path / "engines" / "audio" / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    for relative in (
        Path("weights") / "mmaudio_large_44k_v2.pth",
        Path("ext_weights") / "v1-44.pth",
        Path("ext_weights") / "synchformer_state_dict.pth",
    ):
        path = engine / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture weights")
    _write_mmaudio_clip_cache(tmp_path / "models" / "hub")

    def fake_run(command, **_kwargs):
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "mmaudio_output.flac").write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        result = service.generate_video_audio(
            video,
            AudioGenerationOptions(
                prompt="cloth movement",
                kind="video_audio",
                model_id="mmaudio:large_44k_v2",
                duration_seconds=5,
            ),
            output_path=out,
        )

    assert result.output_path == str(out)
    assert out.read_bytes() == b"audio"


def test_mmaudio_minimum_is_the_safe_default():
    service = AudioGenerationService(RuntimeFlags(), UserSettings())

    assert service.sfx_model_choices()[0][1] == "mmaudio:small_16k"
    assert service.video_audio_model_choices()[0][1] == "mmaudio:small_16k"


def test_mmaudio_variants_are_individually_installed_and_allowlisted(tmp_path: Path, monkeypatch):
    engine, _python = _write_minimum_mmaudio(tmp_path)
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    calls = []

    def fake_run(command, *, cwd, capture_output, text, timeout, env):
        calls.append((command, cwd, timeout))
        assert "all_model_cfg['large_44k_v2'].download_if_needed()" in command[-1]
        for relative in (
            Path("weights") / "mmaudio_large_44k_v2.pth",
            Path("ext_weights") / "v1-44.pth",
            Path("ext_weights") / "synchformer_state_dict.pth",
        ):
            path = engine / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture weights")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("aiwf.services.audio.subprocess.run", fake_run)
    receipt = service.install_mmaudio_variant("large_44k_v2")

    assert receipt["installed"] is True
    assert service._mmaudio_variant_ready("large_44k_v2") is True
    assert calls and calls[0][1] == str(engine)
    assert calls[0][2] == 4 * 60 * 60


@pytest.mark.parametrize(
    "variant",
    ["small_16k", "large_44k_v2", "large_44k", "medium_44k", "small_44k"],
)
def test_mmaudio_discovers_shared_assets_for_every_supported_variant(tmp_path: Path, variant: str):
    from aiwf.services.audio import _mmaudio_variant_files

    shared_root = tmp_path / "shared-models"
    expected = _write_shared_mmaudio_variant(shared_root, variant)
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path / "studio", extra_model_dirs=[shared_root]),
        UserSettings(),
    )

    assert service._mmaudio_variant_shared_ready(variant) is True
    assert set(service._mmaudio_shared_asset_sources(variant)) == set(_mmaudio_variant_files(variant))
    sources = service._mmaudio_shared_asset_sources(variant)
    assert all(source == expected[relative].resolve() for relative, source in sources.items())
    assert service._mmaudio_variant_ready(variant) is False


def test_mmaudio_clip_readiness_requires_complete_configured_hub_snapshot(tmp_path: Path):
    from aiwf.services.audio import _find_mmaudio_clip_hub_cache

    shared_root = tmp_path / "shared-models"
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path / "studio", extra_model_dirs=[shared_root]),
        UserSettings(),
    )
    assert _find_mmaudio_clip_hub_cache([shared_root]) is None
    assert service._mmaudio_variant_ready("small_16k") is False

    cache = _write_mmaudio_clip_cache(shared_root / "hub")
    assert _find_mmaudio_clip_hub_cache([shared_root]) == cache.resolve()


def test_mmaudio_variant_status_reports_missing_shared_clip_cache(tmp_path: Path, monkeypatch):
    from aiwf.core.domain.audio_lab import AudioEngineStatus
    from aiwf.services.audio import _mmaudio_variant_files

    engine, python = _write_minimum_mmaudio(tmp_path)
    for relative in _mmaudio_variant_files("large_44k_v2"):
        path = engine / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture weights")
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr("aiwf.services.audio.importlib.util.find_spec", lambda _name: object())
    monkeypatch.setattr(service, "_mmaudio_clip_hub_cache", lambda: None)
    monkeypatch.setattr("aiwf.services.audio._resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        "aiwf.services.audio_lab.AudioLabService.status",
        lambda _self, *, deep=False: AudioEngineStatus(installed=True, message="fixture"),
    )

    status = service.setup_status(deep=False)
    component = next(item for item in status["components"] if item["id"] == "mmaudio-large-44k-v2")

    assert python.is_file()
    assert component["ready"] is False
    assert component["missing"] == [
        "Hugging Face cache for apple/DFN5B-CLIP-ViT-H-14-384 (open_clip_config.json and model weights)"
    ]


def test_mmaudio_clip_discovers_hub_cache_under_audio_mmaudio_layout(tmp_path: Path):
    from aiwf.services.audio import _find_mmaudio_clip_hub_cache

    shared_root = tmp_path / "shared-models"
    expected = shared_root / "audio" / "MMAudio" / ".cache" / "huggingface" / "hub"
    _write_mmaudio_clip_cache(expected)

    assert _find_mmaudio_clip_hub_cache([shared_root]) == expected.resolve()


def test_mmaudio_clip_cache_rejects_weight_symlink_escape(tmp_path: Path):
    from aiwf.services.audio import _find_mmaudio_clip_hub_cache

    shared_root = tmp_path / "shared-models"
    snapshot = _write_mmaudio_clip_cache(shared_root / "hub") / "models--apple--DFN5B-CLIP-ViT-H-14-384" / "snapshots" / ("a" * 40)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside weight")
    weight = snapshot / "open_clip_pytorch_model.safetensors"
    weight.unlink()
    try:
        weight.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not available in this test environment")

    assert _find_mmaudio_clip_hub_cache([shared_root]) is None


@pytest.mark.parametrize(
    "variant",
    ["small_16k", "large_44k_v2", "large_44k", "medium_44k", "small_44k"],
)
def test_mmaudio_installer_imports_shared_assets_into_studio_engine_without_writing_shared_root(
    tmp_path: Path, monkeypatch, variant: str,
):
    from aiwf.services.audio import _mmaudio_variant_files

    engine, _python = _write_minimum_mmaudio(tmp_path / "studio")
    shared_root = tmp_path / "shared-models"
    shared_assets = _write_shared_mmaudio_variant(shared_root, variant)
    before = {path: path.read_bytes() for path in shared_assets.values()}
    for relative in _mmaudio_variant_files(variant):
        (engine / relative).unlink(missing_ok=True)
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path / "studio", extra_model_dirs=[shared_root]),
        UserSettings(),
    )
    calls = []

    def fake_run(command, *, cwd, capture_output, text, timeout, env):
        calls.append((command, cwd, timeout))
        assert f"all_model_cfg[{variant!r}].download_if_needed()" in command[-1]
        assert all((engine / relative).is_file() for relative in shared_assets)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("aiwf.services.audio.subprocess.run", fake_run)
    receipt = service.install_mmaudio_variant(variant)

    assert receipt["installed"] is True
    assert service._mmaudio_variant_ready(variant) is True
    assert calls and calls[0][1] == str(engine)
    assert {path: path.read_bytes() for path in shared_assets.values()} == before
    sources = service._mmaudio_shared_asset_sources(variant)
    assert all(
        (engine / relative).read_bytes() == before[source]
        for relative, source in sources.items()
    )


def test_mmaudio_shared_asset_discovery_rejects_symlink_escape(tmp_path: Path):
    shared_root = tmp_path / "shared-models"
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside weight")
    escaped = shared_root / "MMAudio" / "weights" / "mmaudio_large_44k_v2.pth"
    escaped.parent.mkdir(parents=True)
    try:
        escaped.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not available in this test environment")
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path / "studio", extra_model_dirs=[shared_root]),
        UserSettings(),
    )

    assert Path("weights") / "mmaudio_large_44k_v2.pth" not in service._mmaudio_shared_asset_sources("large_44k_v2")


def test_mmaudio_variant_install_rejects_unknown_key_without_subprocess(tmp_path: Path, monkeypatch):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    calls = []
    monkeypatch.setattr("aiwf.services.audio.subprocess.run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(AudioUnavailable, match="Unsupported MMAudio variant"):
        service.install_mmaudio_variant("../../unexpected")
    assert calls == []


def test_mmaudio_generation_blocks_variant_when_its_weights_are_missing(tmp_path: Path, monkeypatch):
    _write_minimum_mmaudio(tmp_path)
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    calls = []
    monkeypatch.setattr("aiwf.services.audio.subprocess.run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(AudioUnavailable, match="large_44k_v2 is not installed completely"):
        service._generate_mmaudio_text_audio(
            AudioGenerationOptions(prompt="wind and rain", kind="sfx", model_id="mmaudio:large_44k_v2"),
            tmp_path / "out.flac",
        )
    assert calls == []


@pytest.mark.parametrize("video_audio", [False, True])
def test_audio_releases_image_backend_before_audio_work(tmp_path: Path, monkeypatch, video_audio: bool):
    events = []
    service = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path),
        UserSettings(),
        unload_image_models=lambda: events.append("unload-image"),
    )
    if video_audio:
        video = tmp_path / "source.mp4"
        video.write_bytes(b"video")

        def generate(_video, _options, destination):
            events.append("audio")
            destination.write_bytes(b"audio")
            return 44100

        monkeypatch.setattr(service, "_generate_mmaudio_video_audio", generate)
        service.generate_video_audio(
            video,
            AudioGenerationOptions(prompt="wind", kind="video_audio", model_id="mmaudio:small_16k"),
        )
    else:
        def generate(_options, destination):
            events.append("audio")
            destination.write_bytes(b"audio")
            return 32000

        monkeypatch.setattr(service, "_generate_transformers_musicgen", generate)
        service.generate(AudioGenerationOptions(prompt="soft piano", kind="music"))
    assert events == ["unload-image", "audio"]


def test_mmaudio_text_to_sound_uses_isolated_engine(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    service._mmaudio_runtime_import_error = lambda: ""
    engine, python = _write_minimum_mmaudio(tmp_path)
    output = tmp_path / "sound.flac"
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        captured["kwargs"] = _kwargs
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "footsteps.flac").write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        result = service.generate(
            AudioGenerationOptions(
                prompt="footsteps",
                kind="sfx",
                model_id="mmaudio:small_16k",
                duration_seconds=4,
                seed=7,
            ),
            output_path=output,
        )

    command = captured["command"]
    assert command[0] == str(python)
    assert Path(command[1]).name == "run_mmaudio_offline.py"
    assert command[2] == str(engine / "demo.py")
    assert captured["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert "--video" not in command
    assert command[command.index("--variant") + 1] == "small_16k"
    assert result.sample_rate == 16000
    assert output.read_bytes() == b"audio"


def test_setup_status_requires_real_minimum_assets(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    _write_minimum_mmaudio(tmp_path)
    musicgen = tmp_path / "models" / "audio" / "MusicGen" / "musicgen-small"
    musicgen.mkdir(parents=True)
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if name == "model.safetensors":
            _write_safetensors_fixture(musicgen / name)
        else:
            (musicgen / name).write_bytes(b"asset")

    lab_status = SimpleNamespace(installed=True, python_path=str(tmp_path / "audio-lab-python"), message="ready")
    with patch("aiwf.services.audio_lab.AudioLabService.status", return_value=lab_status), patch(
        "aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"
    ):
        with patch.object(service, "_resolve_ffprobe", return_value="ffprobe"):
            status = service.setup_status(deep=False)
        with patch.object(service, "_resolve_ffprobe", return_value=None):
            no_probe_status = service.setup_status(deep=False)

    assert no_probe_status["minimumReady"] is False
    assert no_probe_status["muxReady"] is False
    assert no_probe_status["videoAudioReady"] is False
    mux_component = next(item for item in no_probe_status["components"] if item["id"] == "ffmpeg")
    assert mux_component["ready"] is False
    assert mux_component["missing"] == ["ffprobe"]
    assert status["minimumReady"] is True
    assert status["runtimeChecksPerformed"] is False
    assert "runtime checks have not run" in status["message"]
    assert status["musicReady"] is True
    assert status["sfxReady"] is True
    assert status["videoAudioReady"] is True
    assert status["labReady"] is True
    assert status["muxReady"] is True
    assert all(
        component["ready"]
        for component in status["components"]
        if component["id"] in {"musicgen-small", "mmaudio-small-16k", "audio-lab", "ffmpeg"}
    )


def test_setup_status_rejects_empty_mmaudio_support_weight(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    engine, _python = _write_minimum_mmaudio(tmp_path)
    (engine / "ext_weights" / "synchformer_state_dict.pth").write_bytes(b"")

    lab_status = SimpleNamespace(installed=True, python_path=str(tmp_path / "audio-lab-python"), message="ready")
    with patch("aiwf.services.audio_lab.AudioLabService.status", return_value=lab_status), patch(
        "aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"
    ), patch.object(service, "_resolve_ffprobe", return_value="ffprobe"):
        status = service.setup_status(deep=False)

    mmaudio = next(item for item in status["components"] if item["id"] == "mmaudio-small-16k")
    assert status["sfxReady"] is False
    assert status["videoAudioReady"] is False
    assert mmaudio["ready"] is False
    assert str(Path("ext_weights") / "synchformer_state_dict.pth") in mmaudio["missing"]


def test_generate_for_video_clamps_probed_duration(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video = tmp_path / "short.mp4"
    video.write_bytes(b"video")
    captured = {}

    def fake_generate_video_audio(video_path, options):
        captured["video_path"] = video_path
        captured["duration_seconds"] = options.duration_seconds
        return AudioGenerationResult(
            output_path=str(tmp_path / "sound.flac"),
            prompt=options.prompt,
            model_id=options.model_id,
            kind="video_audio",
            duration_seconds=options.duration_seconds,
        )

    service.generate_video_audio = fake_generate_video_audio

    with patch(
        "aiwf.services.audio.VideoProcessor.probe",
        return_value=SimpleNamespace(duration_seconds=0.25),
    ):
        result = service.generate_for_video(
            video,
            AudioGenerationOptions(
                prompt="room tone",
                kind="video_audio",
                model_id="mmaudio:small_16k",
            ),
        )

    assert captured["video_path"] == video
    assert captured["duration_seconds"] == 1.0
    assert result.duration_seconds == 1.0


def test_default_audio_and_mux_exports_do_not_collide_within_one_second(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    fixed_time = SimpleNamespace(strftime=lambda _fmt: "20261004_040506")

    with patch("aiwf.services.audio.datetime") as clock, patch(
        "aiwf.services.audio.uuid.uuid4",
        side_effect=[
            SimpleNamespace(hex="a" * 32),
            SimpleNamespace(hex="b" * 32),
            SimpleNamespace(hex="c" * 32),
            SimpleNamespace(hex="d" * 32),
        ],
    ):
        clock.now.return_value = fixed_time
        audio_first = service.output_path(stem="same_prompt", suffix=".wav")
        audio_first.write_bytes(b"first render")
        audio_second = service.output_path(stem="same_prompt", suffix=".wav")
        mux_first = service.video_output_path("same_video.mp4")
        mux_first.write_bytes(b"first mux")
        mux_second = service.video_output_path("same_video.mp4")

    assert audio_first != audio_second
    assert audio_first.read_bytes() == b"first render"
    assert not audio_second.exists()
    assert mux_first != mux_second
    assert mux_first.read_bytes() == b"first mux"
    assert not mux_second.exists()


@pytest.mark.parametrize("mode", ["text", "video"])
def test_mmaudio_runs_do_not_reuse_stale_outputs(tmp_path: Path, mode: str):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    service._mmaudio_runtime_import_error = lambda: ""
    engine, _python = _write_minimum_mmaudio(tmp_path)
    destination = tmp_path / "sound.flac"
    destination.write_bytes(b"previous export")
    stale_run = destination.parent / "sound_mmaudio"
    stale_run.mkdir()
    stale_audio = stale_run / ("clip.flac" if mode == "video" else "old.flac")
    stale_audio.write_bytes(b"stale output")
    captured = {}

    def fake_run(command, **_kwargs):
        captured["output_dir"] = Path(command[command.index("--output") + 1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.audio.uuid.uuid4", return_value=SimpleNamespace(hex="newrun")), patch(
        "aiwf.services.audio.subprocess.run", side_effect=fake_run
    ):
        with pytest.raises(AudioUnavailable, match="did not create expected audio"):
            if mode == "video":
                video = tmp_path / "clip.mp4"
                video.write_bytes(b"video")
                service.generate_video_audio(
                    video,
                    AudioGenerationOptions(prompt="room tone", kind="video_audio", model_id="mmaudio:small_16k"),
                    output_path=destination,
                )
            else:
                service.generate(
                    AudioGenerationOptions(prompt="room tone", kind="sfx", model_id="mmaudio:small_16k"),
                    output_path=destination,
                )

    assert captured["output_dir"] != stale_run
    assert captured["output_dir"].is_dir()
    assert stale_audio.read_bytes() == b"stale output"
    assert destination.read_bytes() == b"previous export"
    assert engine.is_dir()

@pytest.mark.parametrize("mode", ["audio", "video"])
@pytest.mark.parametrize("written_bytes", [None, b""])
def test_generation_rejects_missing_or_empty_output(tmp_path: Path, mode: str, written_bytes: bytes | None):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    destination = tmp_path / ("render.wav" if mode == "audio" else "render.flac")
    destination.write_bytes(b"existing export")

    def fake_generator(*args):
        if written_bytes is not None:
            Path(args[-1]).write_bytes(written_bytes)
        return 16000

    options = AudioGenerationOptions(
        prompt="fixture tone",
        kind="music" if mode == "audio" else "video_audio",
        model_id="facebook/musicgen-small" if mode == "audio" else "mmaudio:small_16k",
    )
    with pytest.raises(AudioUnavailable, match="non-empty output file"):
        if mode == "audio":
            with patch.object(service, "_generate_transformers_musicgen", side_effect=fake_generator):
                service.generate(options, output_path=destination)
        else:
            video = tmp_path / "fixture.mp4"
            video.write_bytes(b"video")
            with patch.object(service, "_generate_mmaudio_video_audio", side_effect=fake_generator):
                service.generate_video_audio(video, options, output_path=destination)

    assert destination.read_bytes() == b"existing export"
    assert list(tmp_path.glob(".*.partial.*")) == []


@pytest.mark.parametrize("mode", ["audio", "video"])
def test_generation_publishes_only_verified_staged_output(tmp_path: Path, mode: str):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    destination = tmp_path / ("render.wav" if mode == "audio" else "render.flac")
    destination.write_bytes(b"old export")
    seen = {}

    def fake_generator(*args):
        staged = Path(args[-1])
        seen["staged"] = staged
        assert staged != destination
        assert staged.parent == destination.parent
        assert staged.suffix == destination.suffix
        staged.write_bytes(b"verified new audio")
        return 16000

    options = AudioGenerationOptions(
        prompt="fixture tone",
        kind="music" if mode == "audio" else "video_audio",
        model_id="facebook/musicgen-small" if mode == "audio" else "mmaudio:small_16k",
    )
    if mode == "audio":
        with patch.object(service, "_generate_transformers_musicgen", side_effect=fake_generator):
            result = service.generate(options, output_path=destination)
    else:
        video = tmp_path / "fixture.mp4"
        video.write_bytes(b"video")
        with patch.object(service, "_generate_mmaudio_video_audio", side_effect=fake_generator):
            result = service.generate_video_audio(video, options, output_path=destination)

    assert result.output_path == str(destination)
    assert destination.read_bytes() == b"verified new audio"
    assert not seen["staged"].exists()
    assert list(tmp_path.glob(".*.partial.*")) == []


def test_generation_preserves_destination_when_atomic_publish_fails(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    destination = tmp_path / "render.wav"
    destination.write_bytes(b"old export")

    def fake_generator(*args):
        Path(args[-1]).write_bytes(b"new verified audio")
        return 16000

    options = AudioGenerationOptions(prompt="fixture tone", kind="music", model_id="facebook/musicgen-small")
    with patch.object(service, "_generate_transformers_musicgen", side_effect=fake_generator), patch(
        "aiwf.services.audio.os.replace", side_effect=OSError("fixture publish failure")
    ):
        with pytest.raises(AudioUnavailable, match="could not be published"):
            service.generate(options, output_path=destination)

    assert destination.read_bytes() == b"old export"
    assert list(tmp_path.glob(".*.partial.*")) == []


def test_generation_cleans_partial_stage_when_backend_raises(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    destination = tmp_path / "render.wav"
    destination.write_bytes(b"old export")

    def broken_generator(*args):
        Path(args[-1]).write_bytes(b"partial bytes")
        raise RuntimeError("fixture backend failure")

    options = AudioGenerationOptions(prompt="fixture tone", kind="music", model_id="facebook/musicgen-small")
    with patch.object(service, "_generate_transformers_musicgen", side_effect=broken_generator):
        with pytest.raises(RuntimeError, match="fixture backend failure"):
            service.generate(options, output_path=destination)

    assert destination.read_bytes() == b"old export"
    assert list(tmp_path.glob(".*.partial.*")) == []
