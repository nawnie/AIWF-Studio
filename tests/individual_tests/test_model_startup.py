from __future__ import annotations

from types import SimpleNamespace
import os
import subprocess
import sys
import threading

import pytest

from aiwf.services.model_startup import _gpu_headroom_status, preload_pro_image_model, pro_image_support_ids
from aiwf.services.route_lifecycle import support_revision


def test_model_operation_lock_is_shared_across_contexts_and_processes(tmp_path):
    from aiwf.services.model_startup import pro_model_load_lock

    first = SimpleNamespace(flags=SimpleNamespace(data_dir=tmp_path))
    second = SimpleNamespace(flags=SimpleNamespace(data_dir=tmp_path))
    lock = pro_model_load_lock(first)
    assert pro_model_load_lock(second) is lock
    assert lock.acquire(blocking=False) is True

    child_code = (
        "from pathlib import Path; from types import SimpleNamespace; "
        "from aiwf.services.model_startup import pro_model_load_lock; "
        f"ctx=SimpleNamespace(flags=SimpleNamespace(data_dir=Path({str(tmp_path)!r}))); "
        "lock=pro_model_load_lock(ctx); acquired=lock.acquire(blocking=False); "
        "print(acquired); lock.release() if acquired else None"
    )
    blocked = subprocess.run([sys.executable, "-c", child_code], capture_output=True, text=True, check=True)
    assert blocked.stdout.strip() == "False"

    released = threading.Event()
    thread = threading.Thread(target=lambda: (lock.release(), released.set()))
    thread.start()
    thread.join(timeout=2)
    assert released.is_set()
    available = subprocess.run([sys.executable, "-c", child_code], capture_output=True, text=True, check=True)
    assert available.stdout.strip() == "True"


def _model(model_id: str, *, engine_id: str = "sdxl") -> dict[str, object]:
    return {
        "id": model_id,
        "title": model_id,
        "engineId": engine_id,
        "routeStatus": "request-eligible",
        "checkpointPathStatus": "present",
        "sizeBytes": 1024,
    }


def _context(*, saved_id: str, active_tenant: str = "idle"):
    loaded: list[str] = []
    settings = SimpleNamespace(last_checkpoint_id=saved_id)
    backend = SimpleNamespace(
        devices=SimpleNamespace(device=lambda: SimpleNamespace(type="cpu")),
        can_preload_checkpoint_locally=lambda _model_id: True,
        is_checkpoint_loaded=lambda model_id: model_id in loaded,
    )
    generation = SimpleNamespace(
        backend=backend,
        active_job=lambda: None,
        pending_count=lambda: 0,
        load_checkpoint=lambda model_id, **_kwargs: loaded.append(model_id),
        remember_checkpoint_selection=lambda model_id: setattr(settings, "last_checkpoint_id", model_id),
    )
    ctx = SimpleNamespace(
        generation=generation,
        settings=settings,
        supervisor=SimpleNamespace(active_tenant=active_tenant),
        flags=SimpleNamespace(lowvram=False, medvram=False),
    )
    return ctx, loaded


def test_gpu_headroom_uses_lower_device_wide_nvidia_smi_reading(monkeypatch):
    import sys

    device = SimpleNamespace(type="cuda", index=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(mem_get_info=lambda _device: (int(14.7 * 1024**3), int(16 * 1024**3)), current_device=lambda: 0)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, GPU-abc, 2442\n"),
    )
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=SimpleNamespace(devices=SimpleNamespace(device=lambda: device))),
        flags=SimpleNamespace(lowvram=False, medvram=False),
    )

    issue = _gpu_headroom_status(ctx, {"sizeBytes": 1024})

    assert issue is not None
    assert "2.4 GB VRAM is free" in issue


def test_gpu_headroom_uses_torch_reading_when_nvidia_smi_is_unavailable(monkeypatch):
    import sys

    device = SimpleNamespace(type="cuda", index=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(mem_get_info=lambda _device: (int(8 * 1024**3), int(16 * 1024**3)), current_device=lambda: 0)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr("aiwf.services.gpu_memory.subprocess.run", unavailable)
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=SimpleNamespace(devices=SimpleNamespace(device=lambda: device))),
        flags=SimpleNamespace(lowvram=False, medvram=False),
    )

    assert _gpu_headroom_status(ctx, {"sizeBytes": 1024}) is None


def test_gpu_headroom_maps_cuda_visible_device_to_physical_nvidia_smi_index(monkeypatch):
    import sys

    device = SimpleNamespace(type="cuda", index=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(mem_get_info=lambda _device: (int(14 * 1024**3), int(16 * 1024**3)), current_device=lambda: 0)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, GPU-a, 2442\n1, GPU-b, 8000\n"),
    )
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=SimpleNamespace(devices=SimpleNamespace(device=lambda: device))),
        flags=SimpleNamespace(lowvram=False, medvram=False),
    )

    assert _gpu_headroom_status(ctx, {"sizeBytes": 1024}) is None


def test_gpu_headroom_fails_closed_when_cuda_visible_uuid_cannot_be_matched(monkeypatch):
    import sys

    device = SimpleNamespace(type="cuda", index=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(mem_get_info=lambda _device: (int(14 * 1024**3), int(16 * 1024**3)), current_device=lambda: 0)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-wrong")
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, GPU-other, 2442\n"),
    )
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=SimpleNamespace(devices=SimpleNamespace(device=lambda: device))),
        flags=SimpleNamespace(lowvram=False, medvram=False),
    )

    issue = _gpu_headroom_status(ctx, {"sizeBytes": 1024})

    assert issue is not None
    assert "could not be matched" in issue


def test_gpu_headroom_matches_nvidia_smi_by_device_uuid(monkeypatch):
    import sys

    device = SimpleNamespace(type="cuda", index=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            mem_get_info=lambda _device: (int(14 * 1024**3), int(16 * 1024**3)),
            current_device=lambda: 0,
            get_device_properties=lambda _device: SimpleNamespace(uuid="62f42542-b1ff"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-62f42542-b1ff")
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, GPU-62f42542-b1ff, 2442\n1, GPU-other, 12000\n"),
    )
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=SimpleNamespace(devices=SimpleNamespace(device=lambda: device))),
        flags=SimpleNamespace(lowvram=False, medvram=False),
    )

    issue = _gpu_headroom_status(ctx, {"sizeBytes": 1024})

    assert issue is not None
    assert "2.4 GB VRAM is free" in issue


def test_gpu_headroom_fails_closed_when_torch_and_cuda_visible_uuids_disagree(monkeypatch):
    import sys

    device = SimpleNamespace(type="cuda", index=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            mem_get_info=lambda _device: (int(14 * 1024**3), int(16 * 1024**3)),
            current_device=lambda: 0,
            get_device_properties=lambda _device: SimpleNamespace(uuid="torch-device"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-visible-device")
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, GPU-visible-device, 12000\n"),
    )
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=SimpleNamespace(devices=SimpleNamespace(device=lambda: device))),
        flags=SimpleNamespace(lowvram=False, medvram=False),
    )

    issue = _gpu_headroom_status(ctx, {"sizeBytes": 1024})

    assert issue is not None
    assert "could not be matched" in issue


def test_startup_loads_saved_supported_image_model_without_generation(monkeypatch):
    saved = _model("saved-sdxl")
    other = _model("first-sd15", engine_id="sd15")
    video = _model("wan-video", engine_id="wan")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([other, video, saved], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "saved-sdxl"})
    ctx, loaded = _context(saved_id="saved-sdxl")

    state = preload_pro_image_model(ctx)

    assert state["status"] == "loaded"
    assert state["modelId"] == "saved-sdxl"
    assert loaded == ["saved-sdxl"]
    route_state = next(row for row in ctx._pro_route_lifecycle.values() if row["route"] == "image.txt2img")
    assert route_state["status"] == "loaded"
    assert route_state["resident"] is True
    assert not hasattr(ctx.generation, "generate")


def test_image_support_identity_includes_resolved_flux_assets(tmp_path):
    clip = tmp_path / "clip_l.safetensors"
    t5 = tmp_path / "t5xxl.safetensors"
    vae = tmp_path / "ae.safetensors"
    clip_tokenizer = tmp_path / "tokenizers" / "clip"
    t5_tokenizer = tmp_path / "tokenizers" / "t5"
    for path in (clip, t5, vae):
        path.write_bytes(b"model")
    clip_tokenizer.mkdir(parents=True)
    t5_tokenizer.mkdir(parents=True)
    backend = SimpleNamespace(
        resolve_checkpoint=lambda _model_id: SimpleNamespace(architecture="flux"),
        _resolve_flux_component_paths=lambda: {"clip_l": clip, "t5xxl": t5, "vae": vae},
        _resolve_flux_clip_tokenizer_path=lambda: clip_tokenizer,
        _resolve_flux_t5_tokenizer_path=lambda: t5_tokenizer,
    )
    ctx = SimpleNamespace(generation=SimpleNamespace(backend=backend))

    support_ids = pro_image_support_ids(ctx, "flux-model", "flux", "flux")

    assert "flux" in support_ids
    assert {str(clip), str(t5), str(vae), str(clip_tokenizer), str(t5_tokenizer)} <= set(support_ids)


def test_flux_image_support_identity_detects_tokenizer_replacement(tmp_path):
    tokenizers = [tmp_path / "clip-tokenizer", tmp_path / "t5-tokenizer"]
    files = []
    for root in tokenizers:
        root.mkdir()
        file_path = root / "tokenizer.json"
        file_path.write_text("old", encoding="utf-8")
        files.append(file_path)
    backend = SimpleNamespace(
        resolve_checkpoint=lambda _model_id: SimpleNamespace(architecture="flux"),
        _resolve_flux_component_paths=lambda: {},
        _resolve_flux_clip_tokenizer_path=lambda: tokenizers[0],
        _resolve_flux_t5_tokenizer_path=lambda: tokenizers[1],
    )
    ctx = SimpleNamespace(generation=SimpleNamespace(backend=backend))

    before = support_revision(pro_image_support_ids(ctx, "flux-model", "flux", "flux"))
    original_stat = files[0].stat()
    files[0].write_text("replacement tokenizer", encoding="utf-8")
    os.utime(files[0], ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
    after = support_revision(pro_image_support_ids(ctx, "flux-model", "flux", "flux"))

    assert before != after


def test_single_file_image_support_identity_detects_checkpoint_replacement(tmp_path):
    checkpoint_path = tmp_path / "flux-kontext-Q5_K_M.gguf"
    checkpoint_path.write_bytes(b"old checkpoint")
    checkpoint = SimpleNamespace(path=str(checkpoint_path), architecture="flux")
    backend = SimpleNamespace(
        resolve_checkpoint=lambda _model_id: checkpoint,
        _resolve_flux_component_paths=lambda: {},
        _resolve_flux_clip_tokenizer_path=lambda: None,
        _resolve_flux_t5_tokenizer_path=lambda: None,
    )
    ctx = SimpleNamespace(generation=SimpleNamespace(backend=backend))

    before_ids = pro_image_support_ids(ctx, "flux-kontext", "diffusers", "flux")
    before = support_revision(before_ids)
    assert str(checkpoint_path) in before_ids
    original_stat = checkpoint_path.stat()
    checkpoint_path.write_bytes(b"replacement checkpoint")
    os.utime(checkpoint_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
    after = support_revision(pro_image_support_ids(ctx, "flux-kontext", "diffusers", "flux"))

    assert before != after


def test_image_support_identity_tracks_configured_external_vae_for_startup(tmp_path):
    checkpoint_path = tmp_path / "sdxl.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")
    vae_path = tmp_path / "vae.safetensors"
    vae_path.write_bytes(b"old vae")
    backend = SimpleNamespace(
        resolve_checkpoint=lambda _model_id: SimpleNamespace(path=str(checkpoint_path), architecture="sdxl"),
    )
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=backend),
        flags=SimpleNamespace(vae_path=str(vae_path)),
    )

    before_ids = pro_image_support_ids(ctx, "sdxl-model", "diffusers", "sdxl")
    assert str(vae_path) in before_ids
    before = support_revision(before_ids)
    original_stat = vae_path.stat()
    vae_path.write_bytes(b"replacement vae")
    os.utime(vae_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
    after = support_revision(pro_image_support_ids(ctx, "sdxl-model", "diffusers", "sdxl"))

    assert before != after


def test_image_support_identity_detects_nested_component_replacement(tmp_path):
    component_dir = tmp_path / "components"
    nested_weight = component_dir / "text_encoder" / "model.safetensors"
    nested_weight.parent.mkdir(parents=True)
    nested_weight.write_bytes(b"old")
    backend = SimpleNamespace(
        resolve_checkpoint=lambda _model_id: SimpleNamespace(architecture="flux2_klein"),
        _resolve_component_dir=lambda _architecture, _checkpoint: component_dir,
    )
    ctx = SimpleNamespace(generation=SimpleNamespace(backend=backend))

    before = support_revision(pro_image_support_ids(ctx, "klein-model", "flux2", "flux2_klein"))
    original_stat = nested_weight.stat()
    nested_weight.write_bytes(b"new-content")
    os.utime(nested_weight, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
    after = support_revision(pro_image_support_ids(ctx, "klein-model", "flux2", "flux2_klein"))

    assert before != after


def test_flux_kontext_gguf_support_identity_tracks_resolved_components(tmp_path):
    checkpoint_path = tmp_path / "flux-kontext-Q5_K_M.gguf"
    checkpoint_path.write_bytes(b"kontext checkpoint")
    checkpoint = SimpleNamespace(path=str(checkpoint_path), architecture="flux_kontext")
    component_dir = tmp_path / "flux-kontext-components"
    component_weight = component_dir / "text_encoder" / "model.safetensors"
    component_weight.parent.mkdir(parents=True)
    component_weight.write_bytes(b"old")
    calls = []

    def resolve_component_dir(architecture, resolved_checkpoint):
        calls.append((architecture, resolved_checkpoint))
        return component_dir

    backend = SimpleNamespace(
        resolve_checkpoint=lambda _model_id: checkpoint,
        _resolve_component_dir=resolve_component_dir,
    )
    ctx = SimpleNamespace(generation=SimpleNamespace(backend=backend))

    before_ids = pro_image_support_ids(ctx, "flux-kontext", "diffusers", "flux_kontext")
    before = support_revision(before_ids)
    assert str(component_dir) in before_ids
    assert str(component_weight) in before_ids
    original_stat = component_weight.stat()
    component_weight.write_bytes(b"replacement component weight")
    os.utime(component_weight, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
    after_ids = pro_image_support_ids(ctx, "flux-kontext", "diffusers", "flux_kontext")
    after = support_revision(after_ids)

    assert before != after
    assert len(calls) == 2
    assert all(architecture == "flux_kontext" and resolved_checkpoint is checkpoint for architecture, resolved_checkpoint in calls)


def test_folder_backed_image_support_identity_detects_snapshot_file_replacement(tmp_path):
    from aiwf.core.config.settings import RuntimeFlags

    snapshot = tmp_path / "models" / "qwen-image" / "Diffusers" / "Qwen-Image"
    component = snapshot / "vae" / "diffusion_pytorch_model.safetensors"
    component.parent.mkdir(parents=True)
    component.write_bytes(b"old")
    checkpoint = SimpleNamespace(path=str(snapshot), architecture="qwen_image")
    backend = SimpleNamespace(resolve_checkpoint=lambda _model_id: checkpoint)
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=backend),
        flags=RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"),
    )

    before = support_revision(pro_image_support_ids(ctx, "qwen-image", "qwen", "qwen_image"))
    original_stat = component.stat()
    component.write_bytes(b"new-content")
    os.utime(component, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
    after = support_revision(pro_image_support_ids(ctx, "qwen-image", "qwen", "qwen_image"))

    assert before != after


@pytest.mark.parametrize("architecture", ["flux2_klein", "z_image"])
def test_flux2_and_zimage_snapshot_identity_tracks_snapshot_file_replacement(tmp_path, architecture):
    from aiwf.core.config.settings import RuntimeFlags

    snapshot = tmp_path / "models" / architecture / "Diffusers" / "snapshot"
    component = snapshot / "transformer" / "diffusion_pytorch_model.safetensors"
    component.parent.mkdir(parents=True)
    component.write_bytes(b"old snapshot weight")
    checkpoint = SimpleNamespace(path=str(snapshot), architecture=architecture)
    # A split-file resolver can also return a directory for these families;
    # it must not replace identity coverage for the selected full snapshot.
    backend = SimpleNamespace(
        resolve_checkpoint=lambda _model_id: checkpoint,
        _resolve_component_dir=lambda _architecture, _checkpoint: tmp_path / "split-components",
    )
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=backend),
        flags=RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"),
    )

    before_ids = pro_image_support_ids(ctx, "snapshot-model", "diffusers", architecture)
    assert str(snapshot.resolve()) in before_ids
    before = support_revision(before_ids)
    original_stat = component.stat()
    component.write_bytes(b"replacement snapshot weight")
    os.utime(component, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
    after = support_revision(pro_image_support_ids(ctx, "snapshot-model", "diffusers", architecture))

    assert before != after


def test_folder_backed_image_support_identity_rejects_snapshot_escape(tmp_path):
    from aiwf.core.config.settings import RuntimeFlags

    outside = tmp_path / "outside" / "Qwen-Image"
    outside.mkdir(parents=True)
    (outside / "model_index.json").write_text("{}", encoding="utf-8")
    backend = SimpleNamespace(
        resolve_checkpoint=lambda _model_id: SimpleNamespace(path=str(outside), architecture="qwen_image")
    )
    ctx = SimpleNamespace(
        generation=SimpleNamespace(backend=backend),
        flags=RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"),
    )

    support_ids = pro_image_support_ids(ctx, "qwen-image", "qwen", "qwen_image")

    assert str(outside.resolve()) not in support_ids


def test_backend_preload_gate_rejects_flux_kontext_with_empty_components(tmp_path):
    from types import SimpleNamespace

    from aiwf.infrastructure.diffusers.backend import DiffusersBackend

    snapshot = tmp_path / "FLUX.1-Kontext-dev"
    for component in ("scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "transformer", "vae"):
        (snapshot / component).mkdir(parents=True)
    (snapshot / "model_index.json").write_text('{"_class_name":"FluxKontextPipeline"}', encoding="utf-8")
    backend = SimpleNamespace(
        _resolve_checkpoint=lambda _model_id: SimpleNamespace(path=str(snapshot), architecture="flux_kontext"),
    )

    assert DiffusersBackend.can_preload_checkpoint_locally(backend, "kontext") is False


def test_startup_defers_if_route_selection_changes_before_load(monkeypatch):
    saved = _model("saved-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([saved], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "saved-sdxl"})
    monkeypatch.setattr("aiwf.services.route_lifecycle.begin_route_operation", lambda *_args, **_kwargs: "")
    ctx, loaded = _context(saved_id="saved-sdxl")

    state = preload_pro_image_model(ctx)

    assert state["status"] == "deferred"
    assert "route changed" in state["detail"]
    assert loaded == []


def test_startup_skips_when_gpu_tenant_is_busy(monkeypatch):
    model = _model("saved-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([model], []))
    ctx, loaded = _context(saved_id="saved-sdxl", active_tenant="audio")

    state = preload_pro_image_model(ctx)

    assert state["status"] == "deferred"
    assert "audio" in state["detail"]
    assert loaded == []


def test_startup_load_contention_is_retryable_deferred_not_failed(monkeypatch):
    model = _model("saved-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([model], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "saved-sdxl"})
    ctx, loaded = _context(saved_id="saved-sdxl")

    def contend(_model_id, **_kwargs):
        raise RuntimeError("GPU busy: video route acquired the GPU")

    ctx.generation.load_checkpoint = contend

    state = preload_pro_image_model(ctx)

    assert state["status"] == "deferred"
    assert "GPU ownership changed" in state["detail"]
    assert loaded == []


def test_startup_exception_before_model_load_returns_failure_without_secondary_error(monkeypatch):
    model = _model("saved-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([model], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "saved-sdxl"})
    ctx, loaded = _context(saved_id="saved-sdxl")
    ctx.generation.active_job = lambda: (_ for _ in ()).throw(RuntimeError("inspection failed"))

    state = preload_pro_image_model(ctx)

    assert state["status"] == "failed"
    assert state["modelId"] == "saved-sdxl"
    assert "inspection failed" in state["detail"]
    assert loaded == []


def test_startup_does_not_load_an_unselected_image_model_when_saved_selection_is_video(monkeypatch):
    video = _model("wan-video", engine_id="wan")
    image = _model("unselected-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([video, image], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "wan-video"})
    ctx, loaded = _context(saved_id="wan-video")

    state = preload_pro_image_model(ctx)

    assert state["status"] == "route-managed"
    assert state["modelId"] == "wan-video"
    assert loaded == []


def test_startup_reports_saved_ltx_as_route_managed_without_image_fallback(monkeypatch):
    ltx = _model("ltx-one-stage", engine_id="ltx")
    image = _model("unselected-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([ltx, image], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "ltx-one-stage"})
    ctx, loaded = _context(saved_id="ltx-one-stage")

    state = preload_pro_image_model(ctx)

    assert state["status"] == "route-managed"
    assert state["modelId"] == "ltx-one-stage"
    assert "video route" in state["detail"]
    assert loaded == []


def test_startup_loads_ready_local_fallback_when_selected_model_support_is_incomplete(monkeypatch):
    saved = _model("saved-sdxl")
    fallback = _model("complete-sd15", engine_id="sd15")
    blocked_saved = {**saved, "routeStatus": "blocked", "readinessReason": "missing SDXL VAE"}
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([fallback], [blocked_saved]))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "saved-sdxl"})
    ctx, loaded = _context(saved_id="saved-sdxl")
    ctx.generation.backend.can_preload_checkpoint_locally = lambda model_id: model_id == "complete-sd15"

    state = preload_pro_image_model(ctx)

    assert state["status"] == "loaded"
    assert state["modelId"] == "complete-sd15"
    assert "missing SDXL VAE" in state["detail"]
    assert loaded == ["complete-sd15"]
    assert ctx.settings.last_checkpoint_id == "complete-sd15"


def test_startup_loads_ready_fallback_when_saved_selection_is_stale(monkeypatch):
    fallback = _model("first-selectable-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([fallback], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "first-selectable-sdxl"})
    ctx, loaded = _context(saved_id="removed-saved-model")

    state = preload_pro_image_model(ctx)

    assert state["status"] == "loaded"
    assert state["modelId"] == "first-selectable-sdxl"
    assert "removed-saved-model" in state["detail"]
    assert loaded == ["first-selectable-sdxl"]
    assert ctx.settings.last_checkpoint_id == "first-selectable-sdxl"


@pytest.mark.parametrize("architecture", ["inpaint", "sdxl_inpaint", "flux_fill"])
def test_startup_uses_standard_image_fallback_for_saved_dedicated_inpaint_model(monkeypatch, architecture):
    inpaint = _model("saved-inpaint")
    inpaint["architecture"] = architecture
    fallback = _model("ready-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([inpaint, fallback], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "saved-inpaint"})
    ctx, loaded = _context(saved_id="saved-inpaint")

    state = preload_pro_image_model(ctx)

    assert state["status"] == "loaded"
    assert state["modelId"] == "ready-sdxl"
    assert "saved-inpaint" in state["detail"]
    assert loaded == ["ready-sdxl"]
    assert ctx.settings.last_checkpoint_id == "saved-inpaint"


def test_startup_does_not_persist_fallback_until_residency_is_confirmed(monkeypatch):
    fallback = _model("first-selectable-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([fallback], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "first-selectable-sdxl"})
    ctx, loaded = _context(saved_id="removed-saved-model")
    ctx.generation.backend.is_checkpoint_loaded = lambda _model_id: False
    ctx.generation.remember_checkpoint_selection = lambda model_id: setattr(ctx.settings, "last_checkpoint_id", model_id)

    state = preload_pro_image_model(ctx)

    assert state["status"] == "load-unconfirmed"
    assert ctx.settings.last_checkpoint_id == "removed-saved-model"
    assert loaded == ["first-selectable-sdxl"]


def test_startup_preserves_saved_selection_with_missing_inventory_path(monkeypatch, tmp_path):
    fallback = _model("first-selectable-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([fallback], []))
    monkeypatch.setattr("aiwf.web.pro_api._is_checkpoint_id_selectable", lambda _ctx, _model_id: False)
    ctx, loaded = _context(saved_id="deleted-saved-model")
    ctx.generation.list_checkpoints = lambda: [
        SimpleNamespace(id="deleted-saved-model", path=str(tmp_path / "deleted.safetensors"))
    ]

    state = preload_pro_image_model(ctx)

    assert state["status"] == "not-ready"
    assert state["modelId"] == "deleted-saved-model"
    assert loaded == []


def test_startup_preserves_present_model_with_fixable_missing_support_assets(monkeypatch, tmp_path):
    fallback = _model("first-selectable-sdxl")
    saved_path = tmp_path / "flux-dev.safetensors"
    saved_path.write_bytes(b"synthetic checkpoint")
    saved = SimpleNamespace(id="saved-flux", path=str(saved_path), architecture="flux")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([fallback], []))
    monkeypatch.setattr("aiwf.web.pro_api._is_checkpoint_id_selectable", lambda _ctx, _model_id: False)
    monkeypatch.setattr(
        "aiwf.web.pro_api._runtime_checkpoint_block",
        lambda _ctx, _checkpoint: {"status": "missing-assets", "reason": "Flux support assets missing"},
    )
    ctx, loaded = _context(saved_id="saved-flux")
    ctx.generation.list_checkpoints = lambda: [saved]

    state = preload_pro_image_model(ctx)

    assert state["status"] == "not-ready"
    assert state["modelId"] == "saved-flux"
    assert loaded == []


@pytest.mark.parametrize("preload_check", ["missing", "false", "error", "residency-error"])
def test_startup_not_ready_downgrades_route_lifecycle_from_setup_ready(monkeypatch, preload_check):
    from aiwf.services.route_lifecycle import lifecycle_snapshot

    selected = _model("saved-sdxl")
    monkeypatch.setattr("aiwf.web.pro_api._selectable_checkpoint_payloads", lambda _ctx: ([selected], []))
    monkeypatch.setattr("aiwf.web.pro_api._settings_defaults", lambda _ctx: {"checkpointId": "saved-sdxl"})
    ctx, loaded = _context(saved_id="saved-sdxl")
    if preload_check == "missing":
        del ctx.generation.backend.can_preload_checkpoint_locally
    elif preload_check == "error":
        def preload_check_error(_model_id):
            raise RuntimeError("support probe failed")
        ctx.generation.backend.can_preload_checkpoint_locally = preload_check_error
    elif preload_check == "residency-error":
        def residency_check_error(_model_id):
            raise RuntimeError("residency probe failed")
        ctx.generation.backend.is_checkpoint_loaded = residency_check_error
    else:
        ctx.generation.backend.can_preload_checkpoint_locally = lambda _model_id: False

    state = preload_pro_image_model(ctx)

    route_state = next(row for row in lifecycle_snapshot(ctx) if row["route"] == "image.txt2img")
    assert state["status"] == "not-ready"
    assert route_state["status"] == "needs-setup"
    assert route_state["resident"] is None
    assert loaded == []
