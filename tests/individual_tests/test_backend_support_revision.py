from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.infrastructure.diffusers import backend as backend_module


@pytest.mark.parametrize(
    ("architecture", "loader_name"),
    [
        ("flux", "_load_flux_checkpoint"),
        ("flux2_klein", "_load_flux2_klein_checkpoint"),
        ("z_image", "_load_z_image_checkpoint"),
        ("qwen_image", "_load_qwen_image_checkpoint"),
        ("sana", "_load_sana_checkpoint"),
        ("krea2", "_load_krea2_checkpoint"),
        ("flux_kontext", "_load_flux_kontext_checkpoint"),
    ],
)
def test_same_path_checkpoint_reuses_pipeline_only_when_support_revision_matches(
    tmp_path,
    monkeypatch,
    architecture: str,
    loader_name: str,
) -> None:
    checkpoint = SimpleNamespace(path=str(tmp_path / "checkpoint"), architecture=architecture, title="fixture")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend.devices = SimpleNamespace(dtype=lambda _no_half: object())
    backend._active = checkpoint
    backend._txt2img = object()
    backend._active_support_revision = "same-support"
    monkeypatch.setattr(backend, "_checkpoint_support_revision", lambda _checkpoint: "same-support")
    unload_calls: list[bool] = []
    monkeypatch.setattr(backend, "unload", lambda **kwargs: unload_calls.append(bool(kwargs.get("keep_flux_encoders"))))

    loaded = getattr(backend, loader_name)(checkpoint)

    assert loaded is checkpoint
    assert unload_calls == []


@pytest.mark.parametrize(
    ("architecture", "loader_name"),
    [
        ("flux", "_load_flux_checkpoint"),
        ("flux2_klein", "_load_flux2_klein_checkpoint"),
        ("z_image", "_load_z_image_checkpoint"),
        ("qwen_image", "_load_qwen_image_checkpoint"),
        ("sana", "_load_sana_checkpoint"),
        ("krea2", "_load_krea2_checkpoint"),
        ("flux_kontext", "_load_flux_kontext_checkpoint"),
    ],
)
def test_same_path_checkpoint_invalidates_pipeline_when_support_revision_changes(
    tmp_path,
    monkeypatch,
    architecture: str,
    loader_name: str,
) -> None:
    checkpoint = SimpleNamespace(path=str(tmp_path / "checkpoint"), architecture=architecture, title="fixture")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._active = checkpoint
    backend._txt2img = object()
    backend._active_support_revision = "old-support"
    monkeypatch.setattr(backend, "_checkpoint_support_revision", lambda _checkpoint: "new-support")

    class ReloadRequired(Exception):
        pass

    unload_calls: list[bool] = []

    def unload(**kwargs):
        unload_calls.append(bool(kwargs.get("keep_flux_encoders")))
        raise ReloadRequired()

    monkeypatch.setattr(backend, "unload", unload)

    with pytest.raises(ReloadRequired):
        getattr(backend, loader_name)(checkpoint)

    assert unload_calls == [False]


def test_selected_vae_same_id_reloads_after_file_revision_changes(tmp_path, monkeypatch):
    from aiwf.core.domain.models import VaeInfo

    vae_path = tmp_path / "custom.safetensors"
    vae_path.write_bytes(b"first")
    vae = VaeInfo(id="custom", title="Custom", filename=vae_path.name, path=str(vae_path))
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend.devices = SimpleNamespace(dtype=lambda _no_half: object())
    pipe = SimpleNamespace(vae=None)
    loaded = []

    class FakeVae:
        def to(self, _device):
            return self

    monkeypatch.setattr(backend, "list_vaes", lambda: [vae])
    monkeypatch.setattr(backend, "_prepare_vae_compute", lambda value, _dtype: value)
    monkeypatch.setattr(backend, "_execution_device", lambda _pipe: "cpu")
    monkeypatch.setattr(backend_module, "AutoencoderKL", SimpleNamespace(
        from_single_file=lambda path, **_kwargs: loaded.append(path) or FakeVae(),
    ))
    monkeypatch.setattr("aiwf.infrastructure.diffusers.backend.apply_image_pipeline_optimizations", lambda *_args, **_kwargs: None)

    backend._apply_vae(pipe, "custom")
    backend._apply_vae(pipe, "custom")
    assert len(loaded) == 1

    vae_path.write_bytes(b"replacement with a different revision")
    backend._apply_vae(pipe, "custom")

    assert len(loaded) == 2


def test_selected_custom_vae_can_switch_back_to_checkpoint_base(tmp_path, monkeypatch):
    from aiwf.core.domain.models import VaeInfo

    vae_path = tmp_path / "custom.safetensors"
    vae_path.write_bytes(b"custom")
    custom = VaeInfo(id="custom", title="Custom", filename=vae_path.name, path=str(vae_path))
    base_vae = object()
    pipe = SimpleNamespace(vae=base_vae)
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend.devices = SimpleNamespace(dtype=lambda _no_half: object())

    class FakeVae:
        def to(self, _device):
            return self

    monkeypatch.setattr(backend, "list_vaes", lambda: [custom])
    monkeypatch.setattr(backend, "_prepare_vae_compute", lambda value, _dtype: value)
    monkeypatch.setattr(backend, "_execution_device", lambda _pipe: "cpu")
    monkeypatch.setattr(backend_module, "AutoencoderKL", SimpleNamespace(
        from_single_file=lambda *_args, **_kwargs: FakeVae(),
    ))
    monkeypatch.setattr("aiwf.infrastructure.diffusers.backend.apply_image_pipeline_optimizations", lambda *_args, **_kwargs: None)

    backend._apply_vae(pipe, "custom")
    assert pipe.vae is not base_vae

    backend._apply_vae(pipe, None)

    assert pipe.vae is base_vae
    assert pipe._aiwf_vae_id is None


def test_missing_selected_vae_restores_base_and_fails_clearly(tmp_path, monkeypatch):
    from aiwf.core.domain.models import VaeInfo

    vae_path = tmp_path / "custom.safetensors"
    vae_path.write_bytes(b"custom")
    custom = VaeInfo(id="custom", title="Custom", filename=vae_path.name, path=str(vae_path))
    base_vae = object()
    pipe = SimpleNamespace(vae=base_vae)
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend.devices = SimpleNamespace(dtype=lambda _no_half: object())

    class FakeVae:
        def to(self, _device):
            return self

    monkeypatch.setattr(backend, "list_vaes", lambda: [custom])
    monkeypatch.setattr(backend, "_prepare_vae_compute", lambda value, _dtype: value)
    monkeypatch.setattr(backend, "_execution_device", lambda _pipe: "cpu")
    monkeypatch.setattr(backend_module, "AutoencoderKL", SimpleNamespace(
        from_single_file=lambda *_args, **_kwargs: FakeVae(),
    ))
    monkeypatch.setattr("aiwf.infrastructure.diffusers.backend.apply_image_pipeline_optimizations", lambda *_args, **_kwargs: None)
    backend._apply_vae(pipe, "custom")
    monkeypatch.setattr(backend, "list_vaes", lambda: [])

    with pytest.raises(ValueError, match="restored the checkpoint's base VAE"):
        backend._apply_vae(pipe, "missing")

    assert pipe.vae is base_vae
    assert pipe._aiwf_vae_id is None


@pytest.mark.parametrize("architecture", ["flux", "flux2_klein", "z_image", "qwen_image", "sana"])
def test_backend_support_revision_tracks_nested_file_replacement(tmp_path, architecture: str) -> None:
    support = tmp_path / "support"
    nested = support / "text_encoder" / "model.safetensors"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"old")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    checkpoint = SimpleNamespace(path=str(support), architecture=architecture)

    if architecture == "flux":
        backend._resolve_flux_component_paths = lambda: {"t5xxl": nested}
        backend._resolve_flux_clip_tokenizer_path = lambda: support
        backend._resolve_flux_t5_tokenizer_path = lambda: support
    elif architecture in {"flux2_klein", "z_image"}:
        backend._resolve_component_dir = lambda _architecture, _checkpoint: support
    before = backend._checkpoint_support_revision(checkpoint)
    nested.write_bytes(b"replacement content")
    after = backend._checkpoint_support_revision(checkpoint)

    assert before != after


def test_backend_support_revision_detects_same_size_replacement_with_restored_timestamp(tmp_path) -> None:
    support = tmp_path / "support"
    nested = support / "text_encoder" / "model.safetensors"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"first")
    original = nested.stat()
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    checkpoint = SimpleNamespace(path=str(tmp_path / "model.safetensors"), architecture="flux2_klein")
    backend._resolve_component_dir = lambda _architecture, _checkpoint: support

    before = backend._checkpoint_support_revision(checkpoint)
    nested.write_bytes(b"other")
    import os

    os.utime(nested, ns=(original.st_atime_ns, original.st_mtime_ns))
    after = backend._checkpoint_support_revision(checkpoint)

    assert before != after


def test_flux_support_revision_detects_newly_resolved_assets(tmp_path) -> None:
    vae = tmp_path / "ae.safetensors"
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    checkpoint = SimpleNamespace(path=str(tmp_path / "flux.safetensors"), architecture="flux")
    backend._resolve_flux_component_paths = lambda: {"vae": vae}
    backend._resolve_flux_clip_tokenizer_path = lambda: tmp_path / "clip-tokenizer"
    backend._resolve_flux_t5_tokenizer_path = lambda: tmp_path / "t5-tokenizer"

    missing_revision = backend._checkpoint_support_revision(checkpoint)
    vae.write_bytes(b"vae")
    (tmp_path / "clip-tokenizer").mkdir()
    (tmp_path / "t5-tokenizer").mkdir()
    resolved_revision = backend._checkpoint_support_revision(checkpoint)

    assert missing_revision != resolved_revision


def test_flux_encoder_support_revision_is_shared_across_different_transformers(tmp_path) -> None:
    clip = tmp_path / "clip.safetensors"
    t5 = tmp_path / "t5.safetensors"
    vae = tmp_path / "ae.safetensors"
    for path in (clip, t5, vae):
        path.write_bytes(b"support")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._resolve_flux_component_paths = lambda: {"clip_l": clip, "t5xxl": t5, "vae": vae}
    backend._resolve_flux_clip_tokenizer_path = lambda: tmp_path / "clip-tokenizer"
    backend._resolve_flux_t5_tokenizer_path = lambda: tmp_path / "t5-tokenizer"
    first_path = tmp_path / "flux-a.safetensors"
    second_path = tmp_path / "flux-b.safetensors"
    first_path.write_bytes(b"transformer-a")
    second_path.write_bytes(b"transformer-b")
    first = SimpleNamespace(path=str(first_path), architecture="flux")
    second = SimpleNamespace(path=str(second_path), architecture="flux")

    encoder_revision_before = backend._flux_encoder_support_revision()
    first_revision = backend._checkpoint_support_revision(first)
    encoder_revision_after = backend._flux_encoder_support_revision()
    second_revision = backend._checkpoint_support_revision(second)

    assert first_revision != second_revision
    assert encoder_revision_before == encoder_revision_after


def test_same_path_flux_inpaint_switch_distinguishes_flux_fill_architecture(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "shared-flux-checkpoint"
    checkpoint_path.mkdir()
    support_paths = [tmp_path / name for name in ("ae.safetensors", "clip_l.safetensors", "t5xxl.safetensors")]
    for path in support_paths:
        path.write_bytes(b"support")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._resolve_flux_component_paths = lambda: {
        "vae": support_paths[0], "clip_l": support_paths[1], "t5xxl": support_paths[2],
    }
    backend._resolve_flux_clip_tokenizer_path = lambda: tmp_path / "clip-tokenizer"
    backend._resolve_flux_t5_tokenizer_path = lambda: tmp_path / "t5-tokenizer"
    base_flux = SimpleNamespace(path=str(checkpoint_path), architecture="flux", title="Flux")
    flux_fill = SimpleNamespace(path=str(checkpoint_path), architecture="flux_fill", title="Flux Fill")
    base_revision = backend._checkpoint_support_revision(base_flux)
    fill_revision = backend._checkpoint_support_revision(flux_fill)
    assert base_revision != fill_revision

    cached_base_pipeline = object()
    expected_fill_pipeline = object()
    backend._inpaint = cached_base_pipeline
    backend._inpaint_active = base_flux
    backend._inpaint_support_revision = base_revision
    calls = []
    monkeypatch.setattr(
        backend, "_load_flux_fill_pipeline",
        lambda checkpoint: calls.append(checkpoint) or expected_fill_pipeline,
    )

    loaded = backend._load_inpaint_checkpoint(flux_fill)

    assert loaded is expected_fill_pipeline
    assert calls == [flux_fill]


@pytest.mark.parametrize("architecture", ["sd15", "sdxl", "sd35", "flux", "flux2_klein", "z_image", "qwen_image", "sana", "krea2", "flux_kontext"])
def test_backend_loaded_state_is_false_when_support_revision_changes(tmp_path, monkeypatch, architecture: str) -> None:
    checkpoint = SimpleNamespace(path=str(tmp_path / "checkpoint"), architecture=architecture)
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._active = checkpoint
    backend._txt2img = object()
    backend._active_support_revision = "old-support"
    backend._resolve_checkpoint = lambda _checkpoint_id: checkpoint
    monkeypatch.setattr(backend, "_checkpoint_support_revision", lambda _checkpoint: "new-support")

    assert backend.is_checkpoint_loaded("fixture") is False


def test_polled_status_support_revision_is_throttled_but_expires(tmp_path, monkeypatch) -> None:
    import time

    checkpoint = SimpleNamespace(path=str(tmp_path / "checkpoint"), architecture="sdxl")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    calls: list[str] = []
    monkeypatch.setattr(backend, "_checkpoint_support_revision", lambda _checkpoint: calls.append("probe") or "revision-a")

    assert backend._checkpoint_support_revision_for_status(checkpoint) == "revision-a"
    assert backend._checkpoint_support_revision_for_status(checkpoint) == "revision-a"
    assert calls == ["probe"]

    key, _checked_at, revision = backend._status_support_revision_cache
    backend._status_support_revision_cache = (key, time.monotonic() - 2.0, revision)

    assert backend._checkpoint_support_revision_for_status(checkpoint) == "revision-a"
    assert calls == ["probe", "probe"]


def test_only_status_poll_uses_cached_revision_generation_check_stays_fresh(tmp_path, monkeypatch) -> None:
    checkpoint = SimpleNamespace(path=str(tmp_path / "checkpoint"), architecture="sdxl")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._active = checkpoint
    backend._txt2img = object()
    backend._active_support_revision = "revision-a"
    backend._resolve_checkpoint = lambda _checkpoint_id: checkpoint
    revision = ["revision-a"]
    monkeypatch.setattr(backend, "_checkpoint_support_revision", lambda _checkpoint: revision[0])

    assert backend.is_checkpoint_loaded_for_status("fixture") is True
    revision[0] = "revision-b"

    # The UI may report stale residency until its short throttle expires.
    assert backend.is_checkpoint_loaded_for_status("fixture") is True
    # Calls used for generation admission always verify against fresh support.
    assert backend.is_checkpoint_loaded("fixture") is False


def test_qwen_nunchaku_sentinel_does_not_inherit_a_previous_support_revision(tmp_path) -> None:
    checkpoint = SimpleNamespace(path=str(tmp_path / "checkpoint"), architecture="qwen_image_nunchaku", title="fixture")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._active = checkpoint
    backend._txt2img = backend._QWEN_NUNCHAKU_SENTINEL
    backend._active_support_revision = "previous-route-revision"
    backend._resolve_checkpoint = lambda _checkpoint_id: checkpoint
    backend._qwen_nunchaku = SimpleNamespace(status=lambda _path: SimpleNamespace(ready=True))

    backend._load_qwen_nunchaku_checkpoint(checkpoint)

    assert backend._active_support_revision is None
    assert backend.is_checkpoint_loaded("fixture") is True


def test_qwen_nunchaku_residency_rechecks_runtime_support_readiness(tmp_path) -> None:
    checkpoint = SimpleNamespace(path=str(tmp_path / "checkpoint"), architecture="qwen_image_nunchaku", title="fixture")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._active = checkpoint
    backend._txt2img = backend._QWEN_NUNCHAKU_SENTINEL
    backend._resolve_checkpoint = lambda _checkpoint_id: checkpoint
    backend._qwen_nunchaku = SimpleNamespace(status=lambda _path: SimpleNamespace(ready=False))

    assert backend.is_checkpoint_loaded("fixture") is False


def test_inpaint_pipeline_cache_invalidates_when_support_revision_changes(tmp_path, monkeypatch) -> None:
    checkpoint = SimpleNamespace(path=str(tmp_path / "flux.safetensors"), architecture="flux", title="fixture", id="fixture")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    cached = object()
    backend._inpaint = cached
    backend._inpaint_active = checkpoint
    backend._inpaint_support_revision = "old-support"
    monkeypatch.setattr(backend, "_checkpoint_support_revision", lambda _checkpoint: "new-support")

    class ReloadRequired(Exception):
        pass

    monkeypatch.setattr(backend, "_load_flux_inpaint_pipeline", lambda _checkpoint: (_ for _ in ()).throw(ReloadRequired()))

    with pytest.raises(ReloadRequired):
        backend._load_inpaint_checkpoint(checkpoint)

    assert backend._inpaint is None
    assert backend._inpaint_active is None
    assert backend._inpaint_support_revision is None


def test_refiner_pipeline_cache_invalidates_when_support_revision_changes(tmp_path, monkeypatch) -> None:
    checkpoint = SimpleNamespace(path=str(tmp_path / "refiner.safetensors"), architecture="sdxl", title="fixture")
    backend = DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())
    backend._refiner = object()
    backend._refiner_active = checkpoint
    backend._refiner_support_revision = "old-support"
    backend._resolve_checkpoint = lambda _checkpoint_id: checkpoint
    monkeypatch.setattr(backend, "_checkpoint_support_revision", lambda _checkpoint: "new-support")

    class ReloadRequired(Exception):
        pass

    monkeypatch.setattr(backend, "_dtype_for_architecture", lambda _architecture: (_ for _ in ()).throw(ReloadRequired()))

    with pytest.raises(ReloadRequired):
        backend._load_refiner_checkpoint("fixture")

    assert backend._refiner is None
    assert backend._refiner_active is None
    assert backend._refiner_support_revision is None
