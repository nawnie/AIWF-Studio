from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace

from aiwf.services.route_lifecycle import (
    begin_route_operation,
    confirm_route_residency,
    finish_route_operation,
    lifecycle_snapshot,
    mark_route_running,
    select_route,
    support_revision,
)


def test_route_operation_can_transition_running_to_completed_or_failed_without_claiming_residency():
    ctx = SimpleNamespace()
    token = begin_route_operation(ctx, route="video.wan", model_id="wan-v2", setup_ready=True)

    assert mark_route_running(ctx, "video.wan", token)
    assert lifecycle_snapshot(ctx)[0]["status"] == "running"
    assert finish_route_operation(ctx, "video.wan", token, success=True, detail="Output verified.")
    assert lifecycle_snapshot(ctx)[0]["status"] == "completed"
    assert lifecycle_snapshot(ctx)[0]["resident"] is None
    assert not finish_route_operation(ctx, "video.wan", token, success=False, detail="Late failure.")


def test_route_operation_can_record_cancellation_separately_from_failure():
    ctx = SimpleNamespace()
    token = begin_route_operation(ctx, route="video.sana", model_id="sana-480p", setup_ready=True)
    assert finish_route_operation(ctx, "video.sana", token, success=False, cancelled=True, detail="Cancelled.")
    assert lifecycle_snapshot(ctx)[0]["status"] == "cancelled"


def test_switching_model_or_support_invalidates_old_completion():
    ctx = SimpleNamespace()
    token = begin_route_operation(
        ctx,
        route="image.txt2img",
        model_id="model-a",
        support_ids=["clip-a"],
        setup_ready=True,
    )

    select_route(
        ctx,
        route="image.txt2img",
        model_id="model-b",
        support_ids=["clip-a"],
        setup_ready=True,
    )
    assert not finish_route_operation(ctx, "image.txt2img", token, success=True, detail="stale")

    token = begin_route_operation(
        ctx,
        route="image.txt2img",
        model_id="model-b",
        support_ids=["clip-a"],
        setup_ready=True,
    )
    select_route(
        ctx,
        route="image.txt2img",
        model_id="model-b",
        support_ids=["clip-b"],
        setup_ready=True,
    )
    assert not finish_route_operation(ctx, "image.txt2img", token, success=True, detail="stale")


def test_same_selection_readiness_downgrade_clears_inflight_operation():
    ctx = SimpleNamespace()
    token = begin_route_operation(ctx, route="audio.music", model_id="musicgen", setup_ready=True)

    state = select_route(ctx, route="audio.music", model_id="musicgen", setup_ready=False, detail="Assets missing.")

    assert state["status"] == "needs-setup"
    assert not finish_route_operation(ctx, "audio.music", token, success=True, detail="stale")


def test_residency_is_only_reported_from_backend_confirmation_and_clears():
    ctx = SimpleNamespace()
    select_route(ctx, route="image.txt2img", model_id="checkpoint-a", setup_ready=True)

    assert lifecycle_snapshot(ctx)[0]["resident"] is None
    assert confirm_route_residency(ctx, "image.txt2img", "checkpoint-a", resident=True)
    assert lifecycle_snapshot(ctx)[0]["status"] == "loaded"
    assert confirm_route_residency(ctx, "image.txt2img", "checkpoint-a", resident=False)
    assert lifecycle_snapshot(ctx)[0]["status"] == "setup-ready"
    assert lifecycle_snapshot(ctx)[0]["resident"] is False


def test_selection_can_record_backend_confirmed_residency_on_existing_route():
    ctx = SimpleNamespace()
    select_route(ctx, route="video.sana.720p", model_id="sana-720p", setup_ready=True)

    loaded = select_route(
        ctx,
        route="video.sana.720p",
        model_id="sana-720p",
        setup_ready=True,
        resident=True,
        detail="Pipeline loaded by owning backend.",
    )

    assert loaded["status"] == "loaded"
    assert loaded["resident"] is True
    assert loaded["detail"] == "Pipeline loaded by owning backend."


def test_concurrent_first_use_shares_one_state_store():
    ctx = SimpleNamespace()

    def select(index):
        return select_route(ctx, route=f"audio.{index}", model_id=f"model-{index}", setup_ready=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(select, range(32)))

    assert len(lifecycle_snapshot(ctx)) == 32


def test_support_revision_changes_when_asset_is_replaced_in_place(tmp_path):
    asset = tmp_path / "text_encoder.safetensors"
    asset.write_bytes(b"first")
    first_revision = support_revision([str(asset)])

    asset.write_bytes(b"replacement")
    os.utime(asset, ns=(asset.stat().st_atime_ns, asset.stat().st_mtime_ns + 1_000_000))

    assert support_revision([str(asset)]) != first_revision
    assert str(asset) not in support_revision([str(asset)])


def test_support_revision_detects_same_size_replacement_with_restored_timestamp(tmp_path):
    asset = tmp_path / "text_encoder.safetensors"
    asset.write_bytes(b"first")
    original = asset.stat()
    first_revision = support_revision([str(asset)])

    asset.write_bytes(b"other")
    os.utime(asset, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert support_revision([str(asset)]) != first_revision


def test_finish_rejects_operation_if_support_changes_midflight(tmp_path):
    ctx = SimpleNamespace()
    asset = tmp_path / "text_encoder.safetensors"
    asset.write_bytes(b"first")
    original = asset.stat()
    token = begin_route_operation(
        ctx,
        route="image.txt2img",
        model_id="model-a",
        support_ids=[str(asset)],
        setup_ready=True,
    )

    asset.write_bytes(b"other")
    os.utime(asset, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert not finish_route_operation(ctx, "image.txt2img", token, success=True, detail="stale support")
    state = lifecycle_snapshot(ctx)[0]
    assert state["status"] == "needs-setup"
    assert state["resident"] is False
    assert state["operationId"] == ""


def test_snapshot_invalidates_loaded_state_after_support_changes(tmp_path, monkeypatch):
    import aiwf.services.route_lifecycle as route_lifecycle

    ctx = SimpleNamespace()
    asset = tmp_path / "text_encoder.safetensors"
    asset.write_bytes(b"first")
    original = asset.stat()
    select_route(
        ctx,
        route="video.sana",
        model_id="sana-base",
        support_ids=[str(asset)],
        setup_ready=True,
        resident=True,
    )

    asset.write_bytes(b"other")
    os.utime(asset, ns=(original.st_atime_ns, original.st_mtime_ns))
    monkeypatch.setattr(route_lifecycle, "_SNAPSHOT_SUPPORT_CHECK_SECONDS", 0)

    state = lifecycle_snapshot(ctx)[0]

    assert state["status"] == "needs-setup"
    assert state["resident"] is False
    assert "Support assets changed" in state["detail"]


def test_support_revision_tracks_nested_files_in_component_directory(tmp_path):
    component_dir = tmp_path / "components"
    component = component_dir / "text_encoder" / "model.safetensors"
    component.parent.mkdir(parents=True)
    component.write_bytes(b"old")
    directory_stat = component_dir.stat()
    first_revision = support_revision([str(component_dir)])

    component.write_bytes(b"new weights")
    original_stat = component.stat()
    component.touch()
    # Keep the directory timestamp fixed: replacing an existing file does not
    # reliably update it, and status must still notice the changed component.
    os.utime(component, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
    os.utime(component_dir, ns=(directory_stat.st_atime_ns, directory_stat.st_mtime_ns))

    assert support_revision([str(component_dir)]) != first_revision
