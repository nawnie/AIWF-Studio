from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

import aiwf

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AIWF_ROOT = str(_REPO_ROOT / "aiwf")
if _AIWF_ROOT not in aiwf.__path__:
    aiwf.__path__.insert(0, _AIWF_ROOT)
import aiwf.infrastructure

_INFRASTRUCTURE_ROOT = str(Path(_AIWF_ROOT) / "infrastructure")
if _INFRASTRUCTURE_ROOT not in aiwf.infrastructure.__path__:
    aiwf.infrastructure.__path__.insert(0, _INFRASTRUCTURE_ROOT)

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.checkpoints import diffusers_dir_has_required_local_files, scan_from_flags
from aiwf.infrastructure.diffusers.loras import scan_loras
from aiwf.infrastructure import model_inventory
from aiwf.infrastructure.model_header import (
    ARCH_FLUX2_KLEIN_LORA,
    ARCH_FLUX_KONTEXT_TRANSFORMER,
    ARCH_FLUX2_KLEIN_TRANSFORMER,
    ARCH_FLUX_LORA,
    ARCH_FLUX_VAE,
    ARCH_LTX_AUDIO_VAE,
    ARCH_LTX_LORA,
    ARCH_LTX_TRANSFORMER,
    ARCH_LTX_VAE,
    ARCH_SD_CHECKPOINT,
    ARCH_SD35_CHECKPOINT,
    ARCH_T5XXL_ENCODER,
    ARCH_UMT5_ENCODER,
    ARCH_Z_IMAGE_TRANSFORMER,
    ARCH_LONGCAT_IMAGE,
    ROLE_TEXT_ENCODER,
    ROLE_VAE,
    read_model_info,
)
from aiwf.infrastructure.model_inventory import get_model_inventory, inventory_path, scan_and_write_model_inventory, scan_model_inventory_report
from aiwf.infrastructure.model_sorter import plan_inbox_sort, plan_model_reorganize, reorganize_models, sort_inbox_models


def test_diffusers_snapshot_readiness_requires_referenced_tokenizer_and_scheduler(tmp_path: Path):
    snapshot = tmp_path / "qwen"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text(json.dumps({
        "_class_name": "QwenImagePipeline",
        "transformer": ["diffusers", "QwenImageTransformer2DModel"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
    }), encoding="utf-8")

    assert not diffusers_dir_has_required_local_files(snapshot)

    scheduler = snapshot / "scheduler"
    scheduler.mkdir()
    (scheduler / "scheduler_config.json").write_text("{}", encoding="utf-8")
    tokenizer = snapshot / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "spiece.model").write_bytes(b"tokenizer")

    transformer = snapshot / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    assert not diffusers_dir_has_required_local_files(snapshot)
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    assert diffusers_dir_has_required_local_files(snapshot)


def _write_safetensors_header(path: Path, tensors: dict, metadata: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = dict(tensors)
    if metadata:
        header["__metadata__"] = metadata
    body_size = 0
    for item in tensors.values():
        offsets = item.get("data_offsets", [0, 0])
        body_size = max(body_size, int(offsets[1]))
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + (b"\0" * body_size))


def test_inventory_finds_misplaced_sdxl_lora_and_writes_manifest(tmp_path: Path):
    models = tmp_path / "models"
    misplaced = models / "Stable-diffusion" / "style.safetensors"
    _write_safetensors_header(
        misplaced,
        {
            "lora_unet_down_blocks_0.lora_down.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {
            "ss_network_module": "networks.lora",
            "ss_base_model_version": "sdxl_base_v1-0",
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    loras = scan_loras(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == "style.safetensors")
    assert record.family == "lora"
    assert record.architecture == "sdxl"
    assert record.recommended_subdir == "Loras/SDXL"
    assert record.should_move is True
    assert inventory_path(flags).is_file()
    assert [lora.filename for lora in loras] == ["style.safetensors"]
    assert loras[0].architecture == "sdxl"
    assert checkpoints == []


def test_inventory_keeps_torchscript_inpaint_asset_out_of_checkpoint_placement(tmp_path: Path):
    import zipfile

    models = tmp_path / "models"
    asset = models / "inpaint" / "big-lama.pt"
    asset.parent.mkdir(parents=True)
    with zipfile.ZipFile(asset, "w") as archive:
        archive.writestr("big-lama/code/__torch__/lama.py", b"scripted module")
        archive.writestr("big-lama/constants.pkl", b"constants")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    record = next(item for item in records if item.filename == asset.name)

    assert record.family == "runtime_asset"
    assert record.architecture == "unknown"
    assert record.should_move is False
    assert record.recommended_subdir == "inpaint"
    assert record.header_identifiers["format_marker"] == "torchscript archive; weights not opened"


@pytest.mark.parametrize(
    ("relative_path", "expected_family"),
    [
        (Path("audio/MusicGen/musicgen-small/model.safetensors"), "audio"),
        (Path("LLM/GGUF/model.gguf"), "llm"),
    ],
)
def test_inventory_uses_modality_folder_without_parsing_large_weight_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: Path, expected_family: str
):
    models = tmp_path / "models"
    asset = models / relative_path
    asset.parent.mkdir(parents=True)
    if asset.suffix == ".safetensors":
        _write_safetensors_header(
            asset,
            {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        )
    else:
        asset.write_bytes(b"GGUF")

    def unexpected_shapes(_path):
        raise AssertionError("The modality path is sufficient; do not parse model tensor tables")

    monkeypatch.setattr(model_inventory, "_safetensors_tensor_shapes", unexpected_shapes)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    record = model_inventory.classify_model_file(asset, model_inventory.model_inventory_roots(flags))

    assert record is not None
    assert record.family == expected_family
    assert record.architecture == expected_family
    assert record.current_subdir == record.recommended_subdir
    assert record.should_move is False


def test_inventory_scan_report_shows_shared_and_missing_roots_without_moving_files(tmp_path: Path):
    models = tmp_path / "models"
    checkpoints = tmp_path / "checkpoints"
    shared = tmp_path / "shared-models"
    missing = tmp_path / "offline-models"
    component = shared / "FluxPipeline"
    component.mkdir(parents=True)
    (component / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}), encoding="utf-8")
    flags = RuntimeFlags(
        data_dir=tmp_path,
        models_dir=models,
        ckpt_dir=checkpoints,
        extra_model_dirs=[shared, missing],
    )

    report = scan_model_inventory_report(flags)
    roots = {root["path"]: root for root in report["roots"]}

    assert report["inventoryCount"] == 1
    assert roots[str(shared.resolve())]["status"] == "scanned"
    assert roots[str(shared.resolve())]["assetCount"] == 1
    assert roots[str(shared.resolve())]["familyCounts"] == {"runtime_asset": 1}
    assert roots[str(missing.resolve())]["status"] == "missing"
    assert len(report["assets"]) == 1
    assert report["assets"][0]["family"] == "runtime_asset"
    assert report["assets"][0]["path"] == str(component.resolve())
    assert report["assetsTruncated"] == 0
    assert (component / "model_index.json").is_file()


def test_inventory_scan_report_honors_unlimited_proposal_limit(tmp_path: Path, monkeypatch):
    from aiwf.infrastructure.model_inventory import ModelInventoryRecord

    models = tmp_path / "models"
    models.mkdir()
    for index in range(260):
        (models / f"asset-{index:03}.safetensors").write_bytes(b"fixture")

    def classify(path, _roots):
        return ModelInventoryRecord(
            path=str(path.resolve()), filename=path.name, family="fixture", architecture="fixture",
            current_subdir="", recommended_subdir="fixture", should_move=False,
        )

    monkeypatch.setattr(model_inventory, "classify_model_file", classify)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    capped = scan_model_inventory_report(flags)
    unlimited = scan_model_inventory_report(flags, proposal_limit=None)

    assert len(capped["assets"]) == 250
    assert capped["assetsTruncated"] == 10
    assert len(unlimited["assets"]) == 260
    assert unlimited["assetsTruncated"] == 0


def test_inventory_placement_candidates_match_the_action_that_can_handle_them(tmp_path: Path, monkeypatch):
    from aiwf.infrastructure.model_inventory import ModelInventoryRecord

    models = tmp_path / "models"
    extra = tmp_path / "shared-models"
    primary_asset = models / "Stable-diffusion" / "flux-base.safetensors"
    shared_asset = extra / "flux-fill.safetensors"
    unknown_primary = models / "mystery.gguf"
    unknown_shared = extra / "unknown.gguf"
    diffusers_folder = models / "unsorted" / "Qwen-Image"
    primary_asset.parent.mkdir(parents=True)
    extra.mkdir()
    primary_asset.write_bytes(b"primary")
    shared_asset.write_bytes(b"shared")
    unknown_primary.write_bytes(b"unknown primary")
    unknown_shared.write_bytes(b"unknown shared")
    diffusers_folder.mkdir(parents=True)
    (diffusers_folder / "model_index.json").write_text('{"_class_name":"QwenImagePipeline"}', encoding="utf-8")
    records = [
        ModelInventoryRecord(
            path=str(asset.resolve()), filename=asset.name, family="flux", architecture="flux",
            current_subdir="Stable-diffusion" if asset == primary_asset else "",
            recommended_subdir="flux/Components", should_move=True,
        )
        for asset in (primary_asset, shared_asset)
    ]
    records.extend(
        ModelInventoryRecord(
            path=str(asset.resolve()), filename=asset.name, family="unknown", architecture="unknown",
            current_subdir="", recommended_subdir="misc",
            should_move=True, header_identifiers={"fallback_marker": "extension"},
        )
        for asset in (unknown_primary, unknown_shared)
    )
    records.append(ModelInventoryRecord(
        path=str(diffusers_folder.resolve()), filename=diffusers_folder.name, family="runtime_asset",
        architecture="qwen_image", current_subdir="unsorted", recommended_subdir="qwen-image/Diffusers",
        should_move=True, header_identifiers={"model_index": "QwenImagePipeline"},
    ))
    monkeypatch.setattr(model_inventory, "scan_and_write_model_inventory", lambda _flags, **_kwargs: records)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models, extra_model_dirs=[extra])

    report = scan_model_inventory_report(flags)

    placements = {item["path"]: item["placement"] for item in report["assets"]}
    assert placements[str(primary_asset.resolve())] == "reorganize-candidate"
    assert placements[str(shared_asset.resolve())] == "candidate"
    assert placements[str(unknown_primary.resolve())] == "manual-review"
    assert placements[str(unknown_shared.resolve())] == "manual-review"
    assert placements[str(diffusers_folder.resolve())] == "reorganize-check"


def test_inventory_report_prioritizes_actionable_placements_before_in_place_assets(tmp_path: Path, monkeypatch):
    from aiwf.infrastructure.model_inventory import ModelInventoryRecord

    models = tmp_path / "models"
    extra = tmp_path / "shared-models"
    models.mkdir()
    extra.mkdir()
    records = []
    for index in range(60):
        asset = models / f"a-in-place-{index:03}.safetensors"
        asset.write_bytes(b"fixture")
        records.append(ModelInventoryRecord(
            path=str(asset.resolve()), filename=asset.name, family="flux", architecture="flux",
            current_subdir="flux/Safetensors", recommended_subdir="flux/Safetensors", should_move=False,
        ))
    candidate = extra / "z-copy-candidate.safetensors"
    candidate.write_bytes(b"fixture")
    records.append(ModelInventoryRecord(
        path=str(candidate.resolve()), filename=candidate.name, family="flux", architecture="flux",
        current_subdir="", recommended_subdir="flux/Safetensors", should_move=True,
        header_identifiers={"tensor_role": "transformer"},
    ))
    monkeypatch.setattr(model_inventory, "scan_and_write_model_inventory", lambda _flags, **_kwargs: records)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models, extra_model_dirs=[extra])

    report = scan_model_inventory_report(flags, proposal_limit=10)

    assert report["assets"][0]["placement"] == "candidate"
    assert report["assets"][0]["path"] == str(candidate.resolve())
    assert all(item["placement"] == "in-place" for item in report["assets"][1:])
    assert report["assetsTruncated"] == 51


def test_inventory_does_not_offer_shared_copy_when_roots_overlap(tmp_path: Path, monkeypatch):
    from aiwf.infrastructure.model_inventory import ModelInventoryRecord

    extra = tmp_path / "shared"
    models = extra / "models"
    source = extra / "flux-base.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture")
    record = ModelInventoryRecord(
        path=str(source.resolve()), filename=source.name, family="flux", architecture="flux",
        current_subdir="", recommended_subdir="flux/GGUF", should_move=True,
        header_identifiers={"filename_marker": "flux gguf"},
    )
    monkeypatch.setattr(model_inventory, "scan_and_write_model_inventory", lambda _flags, **_kwargs: [record])
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models, extra_model_dirs=[extra])

    report = scan_model_inventory_report(flags)

    assert report["assets"][0]["placement"] == "manual-review"
    assert "roots overlap" in report["assets"][0]["placementReason"]


def test_inventory_scan_report_marks_roots_partial_when_child_walk_fails(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    checkpoints = tmp_path / "checkpoints"
    models.mkdir()
    checkpoints.mkdir()
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models, ckpt_dir=checkpoints)
    original_walk = model_inventory.os.walk

    def walk_with_child_error(path, *, onerror=None):
        if Path(path).resolve() == models.resolve():
            if onerror is not None:
                onerror(PermissionError(13, "permission denied", str(models / "locked")))
            yield str(models), [], []
            return
        yield from original_walk(path, onerror=onerror)

    monkeypatch.setattr(model_inventory.os, "walk", walk_with_child_error)

    report = model_inventory.scan_model_inventory_report(flags)
    roots = {root["path"]: root for root in report["roots"]}

    assert roots[str(models.resolve())]["status"] == "partial"
    assert roots[str(models.resolve())]["errorCount"] == 1
    assert "permission denied" in roots[str(models.resolve())]["errors"][0]


def test_inventory_cache_rescans_when_nested_model_asset_is_removed(tmp_path: Path):
    models = tmp_path / "models"
    removed = models / "upscale_models" / "4xBHI_dat2_multiblurjpg.safetensors"
    removed.parent.mkdir(parents=True)
    removed.write_bytes(b"fixture only")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    first = get_model_inventory(flags, force_rescan=True)
    assert any(Path(item.path) == removed.resolve() for item in first)
    model_root_mtime = models.stat().st_mtime_ns

    removed.unlink()
    assert models.stat().st_mtime_ns == model_root_mtime

    second = get_model_inventory(flags)
    assert all(Path(item.path) != removed.resolve() for item in second)


def test_inventory_cache_rescans_when_asset_is_added_to_nested_existing_folder(tmp_path: Path):
    models = tmp_path / "models"
    nested = models / "flux" / "Textencoder"
    original = nested / "clip_l.safetensors"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"fixture only")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    first = get_model_inventory(flags, force_rescan=True)
    assert any(Path(item.path) == original.resolve() for item in first)
    model_root_mtime = models.stat().st_mtime_ns

    added = nested / "t5xxl_fp8_e4m3fn.safetensors"
    added.write_bytes(b"fixture only")
    assert models.stat().st_mtime_ns == model_root_mtime

    second = get_model_inventory(flags)
    assert any(Path(item.path) == added.resolve() for item in second)


def test_inventory_cache_rescans_when_nested_model_file_is_replaced(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    path = models / "checkpoints" / "model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old model")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    original_classify = model_inventory.classify_model_file
    classify_calls = {"count": 0}

    def classify(*args, **kwargs):
        classify_calls["count"] += 1
        return original_classify(*args, **kwargs)

    monkeypatch.setattr(model_inventory, "classify_model_file", classify)

    first = get_model_inventory(flags, force_rescan=True)
    old_mtime = path.stat().st_mtime_ns
    old_parent_mtime = path.parent.stat().st_mtime_ns
    classify_calls["count"] = 0

    path.write_bytes(b"new model has different bytes")
    assert path.parent.stat().st_mtime_ns == old_parent_mtime
    assert path.stat().st_mtime_ns != old_mtime

    second = get_model_inventory(flags)
    assert any(Path(item.path) == path.resolve() for item in first)
    assert any(Path(item.path) == path.resolve() for item in second)
    assert classify_calls["count"] > 0


def test_upscaler_in_shared_models_root_is_not_classified_as_a_generation_checkpoint(tmp_path: Path):
    from aiwf.infrastructure.diffusers.model_arch import UNET_INPUT_KEY

    models = tmp_path / "models"
    upscaler = models / "upscale_models" / "Upscalers" / "4xBHI_dat2_multiblurjpg.safetensors"
    _write_safetensors_header(
        upscaler,
        {UNET_INPUT_KEY: {"dtype": "F16", "shape": [4, 4, 3, 3], "data_offsets": [0, 288]}},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    record = next(item for item in records if Path(item.path) == upscaler.resolve())

    assert record.family == "upscaler"
    assert record.recommended_subdir == "upscale_models"
    assert all(item.path != str(upscaler.resolve()) for item in scan_from_flags(flags))


def test_inventory_cache_rescans_after_classifier_version_changes(tmp_path: Path):
    models = tmp_path / "models"
    lora = models / "Loras" / "Flux" / "styles" / "furry_lora.safetensors"
    _write_safetensors_header(
        lora,
        {
            "lora_unet_down_blocks_0.lora_down.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"ss_network_module": "networks.lora"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    stale_record = {
        "path": str(lora.resolve()),
        "filename": lora.name,
        "family": "runtime_asset",
        "architecture": "flux",
        "current_subdir": "Loras/Flux/styles",
        "recommended_subdir": "Loras/Flux",
        "should_move": True,
        "header_identifiers": {},
        "metadata": {},
    }
    cached = inventory_path(flags)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        json.dumps(
            {
                "schema_version": 7,
                "generated_at": "2026-10-06T18:21:00+00:00",
                "roots": [str(root) for root in [models.resolve()]],
                "assets": [stale_record],
            }
        ),
        encoding="utf-8",
    )

    records = get_model_inventory(flags)

    record = next(item for item in records if item.filename == lora.name)
    assert record.family == "lora"
    assert record.architecture == "flux"
    assert json.loads(cached.read_text(encoding="utf-8"))["schema_version"] == model_inventory.MODEL_INVENTORY_VERSION


def test_unknown_safetensor_goes_to_sort_bucket_not_sd15_catalog(tmp_path: Path):
    models = tmp_path / "models"
    unknown = models / "Stable-diffusion" / "mystery.safetensors"
    _write_safetensors_header(unknown, {})
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == "mystery.safetensors")
    assert record.family == "checkpoint"
    assert record.architecture == "unknown"
    assert record.recommended_subdir == "models to sort"
    assert record.should_move is True
    assert checkpoints == []


def test_inventory_report_marks_unknown_sort_bucket_as_manual_review(tmp_path: Path):
    unknown = tmp_path / "models" / "Stable-diffusion" / "mystery.safetensors"
    _write_safetensors_header(unknown, {})
    report = scan_model_inventory_report(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    proposal = next(item for item in report["assets"] if item["filename"] == unknown.name)
    assert proposal["recommendedSubdir"] == "models to sort"
    assert proposal["placement"] == "manual-review"


def test_wan_substring_in_unknown_filename_does_not_trigger_auto_sort(tmp_path: Path):
    models = tmp_path / "models"
    candidate = models / "models to sort" / "want_to_test.safetensors"
    _write_safetensors_header(candidate, {})
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = reorganize_models(flags)

    assert candidate.is_file()
    assert not any(action.moved for action in actions)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == candidate.name)
    assert record.family == "checkpoint"
    assert record.architecture == "unknown"
    assert record.recommended_subdir == "models to sort"


def test_diffusers_pipeline_inbox_plan_is_safe_and_apply_places_whole_snapshot(tmp_path: Path):
    models = tmp_path / "models"
    source = models / "models to sort" / "Qwen-Image-2512"
    (source / "transformer").mkdir(parents=True)
    (source / "model_index.json").write_text(json.dumps({
        "_class_name": "QwenImagePipeline",
        "transformer": ["diffusers", "QwenImageTransformer2DModel"],
    }), encoding="utf-8")
    (source / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (source / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    planned = plan_inbox_sort(flags)

    assert source.is_dir()
    action = next(item for item in planned if item.filename == source.name)
    assert action.status == "moved"
    assert action.dest_subdir == "qwen-image/Diffusers"

    applied = sort_inbox_models(flags)

    assert not source.exists()
    target = models / "qwen-image" / "Diffusers" / source.name
    assert (target / "model_index.json").is_file()
    assert (target / "transformer" / "diffusion_pytorch_model.safetensors").read_bytes() == b"fixture"
    assert any(item.filename == source.name and item.moved for item in applied)


def test_model_sorter_leaves_linked_inbox_file_in_place(tmp_path: Path):
    models = tmp_path / "models"
    inbox = models / "models to sort"
    inbox.mkdir(parents=True)
    external = tmp_path / "outside.safetensors"
    _write_safetensors_header(
        external,
        {"model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
        {"ss_base_model_version": "sd_v1-5"},
    )
    linked = inbox / external.name
    try:
        linked.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"filesystem does not permit symlink creation: {exc}")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = sort_inbox_models(flags)

    assert linked.is_symlink()
    assert external.is_file()
    assert not (models / "Stable-diffusion" / external.name).exists()
    assert any(action.filename == linked.name and action.status == "left" for action in actions)


def test_model_sorter_rejects_linked_inbox_directory(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    external_inbox = tmp_path / "outside-inbox"
    external_inbox.mkdir()
    (external_inbox / "placeholder.txt").write_text("leave outside", encoding="utf-8")
    inbox = models / "models to sort"
    try:
        inbox.symlink_to(external_inbox, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"filesystem does not permit directory symlink creation: {exc}")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = sort_inbox_models(flags)

    assert (external_inbox / "placeholder.txt").is_file()
    assert any(action.filename == inbox.name and action.status == "left" for action in actions)


def test_model_sorter_places_flux_teacher_support_assets_for_auto_discovery(tmp_path: Path):
    models = tmp_path / "models"
    inbox = models / "models to sort"
    inbox.mkdir(parents=True)
    clip = inbox / "clip_l.safetensors"
    t5 = inbox / "t5xxl_fp8_e4m3fn.safetensors"
    vae = inbox / "ae.safetensors"
    _write_safetensors_header(
        clip,
        {"cond_stage_model.transformer.resblocks.0.attn.in_proj_weight": {
            "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
        }},
    )
    _write_safetensors_header(
        t5,
        {"encoder.block.0.layer.0.SelfAttention.q.weight": {
            "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
        }},
    )
    _write_safetensors_header(
        vae,
        {"encoder.conv_in.weight": {
            "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
        }},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = sort_inbox_models(flags)

    placements = {action.filename: action.dest_subdir for action in actions if action.moved}
    assert placements == {
        "clip_l.safetensors": "Textencoder",
        "t5xxl_fp8_e4m3fn.safetensors": "flux/Textencoder",
        "ae.safetensors": "flux/VAE",
    }
    assert (models / "Textencoder" / clip.name).is_file()
    assert (models / "flux" / "Textencoder" / t5.name).is_file()
    assert (models / "flux" / "VAE" / vae.name).is_file()

    from types import SimpleNamespace

    from aiwf.infrastructure.diffusers.backend import DiffusersBackend

    backend = DiffusersBackend(flags, SimpleNamespace())
    components = backend._resolve_flux_component_paths()
    assert components == {
        "vae": (models / "flux" / "VAE" / vae.name).resolve(),
        "clip_l": (models / "Textencoder" / clip.name).resolve(),
        "t5xxl": (models / "flux" / "Textencoder" / t5.name).resolve(),
    }


def test_inventory_ignores_family_names_above_configured_model_root(tmp_path: Path):
    from aiwf.infrastructure.model_inventory import classify_model_dir

    # A workspace can legitimately put its configured model root under a
    # folder with a family-like name; only paths inside `models` are layout.
    models = tmp_path / "controlnet" / "models"
    snapshot = models / "my-sdxl-snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    roots = model_inventory.model_inventory_roots(flags)

    record = classify_model_dir(snapshot, roots)

    assert record is not None
    assert record.family == "checkpoint"
    assert record.architecture == "sdxl"


def test_inventory_file_family_uses_only_model_root_relative_layout(tmp_path: Path):
    models = tmp_path / "vae" / "models"
    checkpoint = models / "Stable-diffusion" / "sdxl.safetensors"
    _write_safetensors_header(
        checkpoint,
        {"model.diffusion_model.input_blocks.0.0.weight": {
            "dtype": "F16", "shape": [320, 4, 3, 3], "data_offsets": [0, 23040],
        }},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    roots = model_inventory.model_inventory_roots(flags)

    record = model_inventory.classify_model_file(checkpoint, roots)

    assert record is not None
    assert record.family == "checkpoint"


def test_inventory_distinguishes_controlnet_annotators_from_controlnet_weights(tmp_path: Path):
    models = tmp_path / "models"
    annotator = models / "ControlNet" / "body_pose_model.pth"
    annotator.parent.mkdir(parents=True)
    annotator.write_bytes(b"fixture")
    controlnet = models / "ControlNet" / "control_v11p_sd15_canny.safetensors"
    _write_safetensors_header(
        controlnet,
        {"control_model.input_blocks.0.0.weight": {
            "dtype": "F16", "shape": [320, 4, 3, 3], "data_offsets": [0, 23040],
        }},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    roots = model_inventory.model_inventory_roots(flags)

    annotator_record = model_inventory.classify_model_file(annotator, roots)
    controlnet_record = model_inventory.classify_model_file(controlnet, roots)

    assert annotator_record is not None
    assert annotator_record.family == "preprocessor"
    assert annotator_record.recommended_subdir == "ControlNet/Annotators"
    assert controlnet_record is not None
    assert controlnet_record.family == "controlnet"


def test_inventory_identifies_rife_frame_interpolation_models(tmp_path: Path):
    models = tmp_path / "models"
    rife = models / "frame_interpolation" / "rife_v4.26.safetensors"
    rife.parent.mkdir(parents=True)
    _write_safetensors_header(
        rife,
        {"conv0.weight": {"dtype": "F16", "shape": [16, 3, 3, 3], "data_offsets": [0, 864]}},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    roots = model_inventory.model_inventory_roots(flags)

    record = model_inventory.classify_model_file(rife, roots)

    assert record is not None
    assert record.family == "runtime_asset"
    assert record.architecture == "rife"
    assert record.recommended_subdir == "frame_interpolation"
    assert not record.should_move


def test_inventory_does_not_label_lama_inpainting_weights_as_controlnet(tmp_path: Path):
    models = tmp_path / "models"
    lama = models / "ControlNet" / "ControlNetLama.pth"
    lama.parent.mkdir(parents=True)
    lama.write_bytes(b"fixture")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    roots = model_inventory.model_inventory_roots(flags)

    record = model_inventory.classify_model_file(lama, roots)

    assert record is not None
    assert record.family == "runtime_asset"
    assert record.architecture == "lama"
    assert record.recommended_subdir == "ControlNet"
    assert not record.should_move


def test_diffusers_destination_ignores_ancestor_named_diffusers(tmp_path: Path):
    models = tmp_path / "Diffusers" / "models"
    snapshot = models / "bonsai-image-ternary-4B-gemlite-2bit"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"}), encoding="utf-8"
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    roots = model_inventory.model_inventory_roots(flags)

    record = model_inventory.classify_model_dir(snapshot, roots)

    assert record is not None
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "flux2/Components"


def test_invalid_safetensors_architecture_ignores_family_names_above_model_root(tmp_path: Path):
    models = tmp_path / "flux2" / "models"
    invalid = models / "models to sort" / "broken.safetensors"
    invalid.parent.mkdir(parents=True)
    invalid.write_bytes(b"truncated")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    roots = model_inventory.model_inventory_roots(flags)

    record = model_inventory.classify_model_file(invalid, roots)

    assert record is not None
    assert record.family == "invalid_asset"
    assert record.architecture == "unknown"


def test_diffusers_sort_uses_exact_supported_pipeline_family_destinations(tmp_path: Path):
    models = tmp_path / "models"
    inbox = models / "models to sort"
    expected = {
        "FluxKontextPipeline": "flux/Components/FLUX.1-Kontext-dev",
        "Flux2KleinPipeline": "flux2/Components",
        "Krea2Pipeline": "krea2/Diffusers",
        "QwenImagePipeline": "qwen-image/Diffusers",
        "QwenImage21Pipeline": "qwen-image/Diffusers",
        "SanaPipeline": "sana/Diffusers",
        "SanaSprintPipeline": "sana/Diffusers",
        "SanaImageToVideoPipeline": "sana-video/Diffusers",
        "SanaVideoPipeline": "sana-video/Diffusers",
        "StableDiffusionPipeline": "Stable-diffusion",
        "StableDiffusionInpaintPipeline": "Stable-diffusion",
        "StableDiffusion3Pipeline": "Stable-diffusion",
        "StableDiffusionXLPipeline": "Stable-diffusion",
        "StableDiffusionXLInpaintPipeline": "Stable-diffusion",
        "WanPipeline": "wan/Diffusers",
        "WanImageToVideoPipeline": "wan/Diffusers",
        "WanVideoPipeline": "wan/Diffusers",
        "ZImagePipeline": "z-image/Components",
    }
    for index, (class_name, _) in enumerate(expected.items()):
        name = f"snapshot-{index}-{class_name}"
        folder = inbox / name
        folder.mkdir(parents=True)
        (folder / "model_index.json").write_text(json.dumps({
            "_class_name": class_name,
            "transformer": ["diffusers", "GenericTransformer"],
        }), encoding="utf-8")
        transformer = folder / "transformer"
        transformer.mkdir()
        (transformer / "config.json").write_text("{}", encoding="utf-8")
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    unsupported_pipeline_names = ("QwenImageEditPipeline", "QwenImageEditPlusPipeline")
    unsupported_names = set()
    for index, class_name in enumerate(unsupported_pipeline_names, start=len(expected)):
        name = f"snapshot-{index}-{class_name}"
        unsupported_names.add(name)
        folder = inbox / name
        folder.mkdir(parents=True)
        (folder / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = plan_inbox_sort(flags)

    planned = {item.filename: item.dest_subdir for item in actions if item.status == "moved"}
    special_incomplete = {
        "FluxKontextPipeline", "Flux2KleinPipeline", "ZImagePipeline",
        "QwenImageEditPipeline", "QwenImageEditPlusPipeline",
    }
    assert planned == {
        f"snapshot-{index}-{class_name}": destination
        for index, (class_name, destination) in enumerate(expected.items())
        if class_name not in special_incomplete
    }
    left = {item.filename for item in actions if item.status == "left"}
    assert {f"snapshot-{index}-{name}" for index, name in enumerate(expected) if name in special_incomplete} <= left
    assert unsupported_names <= left


@pytest.mark.parametrize(
    ("parent_name", "class_name", "family", "architecture"),
    [
        ("want_to_test", "SanaPipeline", "runtime_asset", "sana"),
        ("ltx experiments", "SanaVideoPipeline", "runtime_asset", "sana_video"),
        ("ltx experiments", "WanImageToVideoPipeline", "wan", "wan"),
        ("want_to_test", "LTXPipeline", "runtime_asset", "ltx"),
        ("flux test", "StableDiffusionPipeline", "checkpoint", "sd15"),
        ("wan test", "StableDiffusionPipeline", "checkpoint", "sd15"),
        ("flux test", "StableDiffusionXLInpaintPipeline", "checkpoint", "sdxl"),
    ],
)
def test_diffusers_pipeline_class_overrides_unrelated_parent_family_tokens(
    tmp_path: Path, parent_name: str, class_name: str, family: str, architecture: str,
):
    models = tmp_path / "models"
    snapshot = models / parent_name / class_name
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": class_name}), encoding="utf-8"
    )
    roots = model_inventory.model_inventory_roots(RuntimeFlags(data_dir=tmp_path, models_dir=models))

    record = model_inventory.classify_model_dir(snapshot, roots)

    assert record is not None
    assert record.family == family
    assert record.architecture == architecture


def test_sana_pipeline_is_sorted_correctly_under_wan_named_parent(tmp_path: Path):
    models = tmp_path / "models"
    snapshot = models / "models to sort" / "want_to_test" / "SanaPipeline snapshot"
    transformer = snapshot / "transformer"
    transformer.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "SanaPipeline", "transformer": ["diffusers", "GenericTransformer"]}),
        encoding="utf-8",
    )
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")

    actions = plan_inbox_sort(RuntimeFlags(data_dir=tmp_path, models_dir=models))

    moved = [item for item in actions if item.status == "moved"]
    assert len(moved) == 1
    assert moved[0].dest_subdir == "sana/Diffusers"


def test_incomplete_flux2_pipeline_in_diffusers_tree_is_left_in_place(tmp_path: Path):
    models = tmp_path / "models"
    snapshot = models / "Diffusers" / "bonsai-image-ternary-4B-gemlite-2bit"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"}), encoding="utf-8"
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = plan_model_reorganize(flags)

    action = next(item for item in actions if item.filename == snapshot.name)
    assert action.status == "left"
    assert "incomplete" in action.reason
    assert snapshot.is_dir()


def test_flux2_sort_routes_valid_support_and_full_snapshots_separately(tmp_path: Path):
    declarations = {
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3ForCausalLM"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "transformer": ["diffusers", "Flux2Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
    }
    configs = {
        "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000},
        "text_encoder/config.json": {"model_type": "qwen3", "hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 16},
        "vae/config.json": {"_class_name": "AutoencoderKLFlux2", "in_channels": 3, "out_channels": 3, "latent_channels": 32},
        "tokenizer/tokenizer_config.json": {"tokenizer_class": "Qwen2TokenizerFast"},
        "tokenizer/tokenizer.json": {"model": {"type": "BPE", "vocab": {"x": 0}}},
    }
    models = tmp_path / "models"
    inbox = models / "models to sort"
    sources = {"support": inbox / "flux2-support", "full": inbox / "flux2-full"}
    for name, source in sources.items():
        source.mkdir(parents=True)
        (source / "model_index.json").write_text(
            json.dumps({"_class_name": "Flux2KleinPipeline", **declarations}), encoding="utf-8"
        )
        for relative, data in configs.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        _write_safetensors_header(source / "text_encoder" / "model.safetensors", {
            "weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        })
        _write_safetensors_header(source / "vae" / "diffusion_pytorch_model.safetensors", {
            "weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        })
    transformer = sources["full"] / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text(
        json.dumps({"_class_name": "Flux2Transformer2DModel", "in_channels": 128, "num_layers": 5}),
        encoding="utf-8",
    )
    _write_safetensors_header(transformer / "diffusion_pytorch_model.safetensors", {
        "weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
    })
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = plan_inbox_sort(flags)
    planned = {item.filename: item.dest_subdir for item in actions if item.status == "moved"}

    assert planned == {"flux2-support": "flux2/Components", "flux2-full": "flux2/Diffusers"}


def test_zimage_sort_routes_valid_support_and_full_snapshots_separately(tmp_path: Path):
    declarations = {
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3Model"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    configs = {
        "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000},
        "text_encoder/config.json": {"model_type": "qwen3", "hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 16},
        "vae/config.json": {"_class_name": "AutoencoderKL", "in_channels": 3, "out_channels": 3, "latent_channels": 4},
        "tokenizer/tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
        "tokenizer/tokenizer.json": {"model": {"type": "BPE", "vocab": {"x": 0}}},
    }
    models = tmp_path / "models"
    inbox = models / "models to sort"
    sources = {"support": inbox / "zimage-support", "full": inbox / "zimage-full"}
    for source in sources.values():
        source.mkdir(parents=True)
        (source / "model_index.json").write_text(
            json.dumps({"_class_name": "ZImagePipeline", **declarations}), encoding="utf-8"
        )
        for relative, data in configs.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        _write_safetensors_header(source / "text_encoder" / "model.safetensors", {
            "weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        })
        _write_safetensors_header(source / "vae" / "diffusion_pytorch_model.safetensors", {
            "weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        })
    transformer = sources["full"] / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text(
        json.dumps({"_class_name": "ZImageTransformer2DModel", "in_channels": 16, "dim": 8, "n_layers": 1}),
        encoding="utf-8",
    )
    _write_safetensors_header(transformer / "diffusion_pytorch_model.safetensors", {
        "weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
    })
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = plan_inbox_sort(flags)
    planned = {item.filename: item.dest_subdir for item in actions if item.status == "moved"}

    assert planned == {"zimage-support": "z-image/Components", "zimage-full": "z-image/Diffusers"}


def test_complete_flux2_diffusers_pipeline_is_not_recommended_for_unet_folder(tmp_path: Path):
    models = tmp_path / "models"
    snapshot = models / "Diffusers" / "bonsai-image-ternary-4B-gemlite-2bit"
    transformer = snapshot / "transformer"
    transformer.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"}), encoding="utf-8"
    )
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"metadata-only fixture")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    record = next(
        item for item in scan_and_write_model_inventory(flags)
        if item.filename == snapshot.name
    )

    assert record.family == "runtime_asset"
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "flux2/Diffusers"
    assert record.should_move is True


def test_incomplete_flux2_diffusers_pipeline_stays_in_diffusers_tree(tmp_path: Path):
    models = tmp_path / "models"
    snapshot = models / "Diffusers" / "bonsai-image-ternary-4B-gemlite-2bit"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"}), encoding="utf-8"
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    record = next(
        item for item in scan_and_write_model_inventory(flags)
        if item.filename == snapshot.name
    )

    assert record.family == "runtime_asset"
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "flux2/Diffusers"
    assert record.should_move is True


def test_diffusers_sort_leaves_unknown_incomplete_and_conflicting_folders(tmp_path: Path):
    models = tmp_path / "models"
    inbox = models / "models to sort"
    unknown = inbox / "unknown-pipeline"
    unknown.mkdir(parents=True)
    (unknown / "model_index.json").write_text(json.dumps({"_class_name": "UnknownPipeline"}), encoding="utf-8")
    incomplete = inbox / "Qwen-Image-incomplete"
    (incomplete / "transformer").mkdir(parents=True)
    (incomplete / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}), encoding="utf-8")
    (incomplete / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}}), encoding="utf-8"
    )
    conflict = inbox / "Qwen-Image-conflict"
    conflict.mkdir()
    (conflict / "model_index.json").write_text(json.dumps({
        "_class_name": "QwenImagePipeline",
        "transformer": ["diffusers", "QwenImageTransformer2DModel"],
    }), encoding="utf-8")
    (conflict / "transformer").mkdir()
    (conflict / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (conflict / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    existing = models / "qwen-image" / "Diffusers" / conflict.name
    existing.mkdir(parents=True)
    (existing / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}), encoding="utf-8")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = sort_inbox_models(flags)

    assert unknown.is_dir()
    assert incomplete.is_dir()
    assert conflict.is_dir()
    assert any(item.filename == incomplete.name and item.status == "left" for item in actions)
    assert any(item.filename == conflict.name and item.status == "conflict" for item in actions)


def test_inventory_excludes_reactor_face_embedding_from_checkpoints(tmp_path: Path):
    models = tmp_path / "models"
    face = models / "reactor" / "faces" / "person.safetensors"
    _write_safetensors_header(
        face,
        {
            "embedding": {"dtype": "F32", "shape": [512], "data_offsets": [0, 2048]},
            "bbox": {"dtype": "F32", "shape": [4], "data_offsets": [2048, 2064]},
            "kps": {"dtype": "F32", "shape": [5, 2], "data_offsets": [2064, 2104]},
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == "person.safetensors")
    assert record.family == "face_embedding"
    assert record.recommended_subdir == "reactor/faces"
    assert checkpoints == []


def test_controlnet_and_wan_loras_do_not_enter_image_lora_catalog(tmp_path: Path):
    models = tmp_path / "models"
    control_lora = models / "controlnet" / "sai_xl_canny_128lora.safetensors"
    wan_lora = models / "Loras" / "Wan" / "motion_wan_rank16.safetensors"
    tensor = {
        "lora_unet_down_blocks_0.lora_down.weight": {
            "dtype": "F16",
            "shape": [4, 4],
            "data_offsets": [0, 32],
        }
    }
    _write_safetensors_header(control_lora, tensor, {"ss_base_model_version": "sdxl_base_v1-0"})
    _write_safetensors_header(wan_lora, tensor, {"ss_base_model_version": "wan2.2"})
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    image_loras = scan_loras(flags)

    by_name = {record.filename: record for record in records}
    assert by_name["sai_xl_canny_128lora.safetensors"].family == "controlnet"
    assert by_name["motion_wan_rank16.safetensors"].family == "lora"
    assert by_name["motion_wan_rank16.safetensors"].architecture == "wan"
    assert image_loras == []


def test_sd35_diffusers_folder_is_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    sd35 = models / "Stable-diffusion" / "stable-diffusion-3.5-medium"
    sd35.mkdir(parents=True)
    (sd35 / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusion3Pipeline"}),
        encoding="utf-8",
    )
    (sd35 / "transformer").mkdir()
    _write_safetensors_header(
        sd35 / "transformer" / "diffusion_pytorch_model.safetensors",
        {
            "transformer_blocks.0.attn.add_q_proj.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    folder_record = next(item for item in records if item.path == str(sd35.resolve()))
    assert folder_record.family == "checkpoint"
    assert folder_record.architecture == "sd35"
    assert folder_record.current_subdir == "Stable-diffusion"
    assert folder_record.recommended_subdir == "Stable-diffusion"
    assert folder_record.should_move is False
    assert [checkpoint.id for checkpoint in checkpoints] == ["stable-diffusion-3.5-medium"]
    assert checkpoints[0].architecture == "sd35"


def test_sd35_all_in_one_metadata_is_not_misclassified_as_vae(tmp_path: Path):
    models = tmp_path / "models"
    checkpoint = models / "Stable-diffusion" / "sd3.5_large_fp8_scaled.safetensors"
    _write_safetensors_header(
        checkpoint,
        {},
        {
            "modelspec.architecture": "stable-diffusion-v3.5-large",
            "modelspec.description": (
                "SD3.5-large all-in-one checkpoint with text encoder and vae weights."
            ),
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == checkpoint.name)
    assert record.family == "checkpoint"
    assert record.architecture == "sd35"
    assert record.recommended_subdir == "Stable-diffusion"
    assert [item.id for item in checkpoints] == ["sd3.5_large_fp8_scaled"]
    assert checkpoints[0].architecture == "sd35"


def test_model_header_labels_sd3_joint_blocks_as_sd35(tmp_path: Path):
    model = tmp_path / "models" / "Stable-diffusion" / "sd3_medium.safetensors"
    _write_safetensors_header(
        model,
        {
            "model.diffusion_model.joint_blocks.0.attn.qkv.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )

    info = read_model_info(model)

    assert info.arch == ARCH_SD35_CHECKPOINT
    assert "SD3.5" in info.display_name


def test_flux_gguf_is_runtime_asset_and_selectable_flux_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    flux = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    flux.parent.mkdir(parents=True)
    flux.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == flux.name)
    assert record.family == "runtime_asset"
    assert record.architecture == "flux"
    assert record.recommended_subdir == "flux/GGUF"
    assert [checkpoint.id for checkpoint in checkpoints] == ["flux1-dev-Q5_K_M"]
    assert checkpoints[0].architecture == "flux"
    assert checkpoints[0].kind == "flux"


def test_reorganize_moves_confident_gguf_without_overwriting(tmp_path: Path):
    models = tmp_path / "models"
    misplaced = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = reorganize_models(flags)

    moved = [action for action in actions if action.status == "moved"]
    assert [action.filename for action in moved] == ["flux1-dev-Q5_K_M.gguf"]
    assert moved[0].dest_subdir == "flux/GGUF"
    assert not misplaced.exists()
    assert (models / "flux" / "GGUF" / "flux1-dev-Q5_K_M.gguf").is_file()


def test_reorganize_applies_only_moves_in_reviewed_plan(tmp_path: Path):
    models = tmp_path / "models"
    reviewed = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    reviewed.parent.mkdir(parents=True)
    reviewed.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    plan = plan_model_reorganize(flags)
    action = next(item for item in plan if item.status == "moved")
    approved = {
        (
            str(Path(action.source).resolve()).casefold(),
            action.family.casefold(),
            action.architecture.casefold(),
            action.dest_subdir.replace("\\", "/").casefold(),
        )
    }
    added_after_review = reviewed.with_name("flux1-schnell-Q5_K_M.gguf")
    added_after_review.write_bytes(b"GGUF")

    actions = reorganize_models(flags, approved_moves=approved)

    moved = [item for item in actions if item.status == "moved"]
    assert [item.filename for item in moved] == [reviewed.name]
    assert (models / "flux" / "GGUF" / reviewed.name).is_file()
    assert added_after_review.is_file()
    assert not (models / "flux" / "GGUF" / added_after_review.name).exists()


def test_reorganize_never_moves_approved_asset_through_destination_symlink(tmp_path: Path):
    models = tmp_path / "models"
    outside = tmp_path / "outside"
    misplaced = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    misplaced.parent.mkdir(parents=True)
    outside.mkdir()
    misplaced.write_bytes(b"GGUF")
    try:
        (models / "flux").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"filesystem does not permit symlink creation: {exc}")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = reorganize_models(flags)

    assert misplaced.is_file()
    assert not (outside / "GGUF" / misplaced.name).exists()
    assert any(
        action.filename == misplaced.name
        and action.status == "left"
        and "escapes" in action.reason
        for action in actions
    )


def test_sort_inbox_never_moves_diffusers_snapshot_through_destination_symlink(tmp_path: Path):
    models = tmp_path / "models"
    outside = tmp_path / "outside"
    source = models / "models to sort" / "Qwen-Image-2512"
    (source / "transformer").mkdir(parents=True)
    (source / "model_index.json").write_text(
        json.dumps({
            "_class_name": "QwenImagePipeline",
            "transformer": ["diffusers", "QwenImageTransformer2DModel"],
        }), encoding="utf-8"
    )
    (source / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (source / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    outside.mkdir()
    try:
        (models / "qwen-image").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"filesystem does not permit symlink creation: {exc}")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = sort_inbox_models(flags)

    assert source.is_dir()
    assert not (outside / "Diffusers" / source.name).exists()
    assert any(
        action.filename == source.name
        and action.status == "left"
        and "escapes" in action.reason
        for action in actions
    )


def test_reorganize_leaves_model_file_symlink_and_target_untouched(tmp_path: Path):
    models = tmp_path / "models"
    source_target = models / "Stable-diffusion" / "target.safetensors"
    source_target.parent.mkdir(parents=True)
    _write_safetensors_header(
        source_target,
        {"model.diffusion_model.input_blocks.0.0.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
        {"ss_base_model_version": "sd_v1-5"},
    )
    linked = models / "models to sort" / "linked.safetensors"
    linked.parent.mkdir(parents=True)
    try:
        linked.symlink_to(source_target)
    except OSError as exc:
        pytest.skip(f"filesystem does not permit symlink creation: {exc}")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = reorganize_models(flags)

    assert linked.is_symlink()
    assert source_target.is_file()
    assert not (models / "Stable-diffusion" / "linked.safetensors").exists()
    assert any(action.filename == linked.name and action.status == "left" for action in actions)


def test_flux2_klein_gguf_is_runtime_asset_and_selectable_flux2_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    klein = models / "Stable-diffusion" / "fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf"
    klein.parent.mkdir(parents=True)
    klein.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(klein)
    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == klein.name)
    assert info.arch == ARCH_FLUX2_KLEIN_TRANSFORMER
    assert record.family == "runtime_asset"
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "flux2/GGUF"
    assert [checkpoint.id for checkpoint in checkpoints] == ["fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM"]
    assert checkpoints[0].architecture == "flux2_klein"
    assert checkpoints[0].kind == "flux2"


def test_comfy_saved_flux2_klein_safetensor_is_selectable_runtime_asset(tmp_path: Path):
    models = tmp_path / "models"
    klein = models / "flux" / "UNet" / "snofsSexNudesAndOtherFunStuff_v14Distilled.safetensors"
    _write_safetensors_header(
        klein,
        {
            "model.diffusion_model.double_blocks.0.img_attn.qkv.weight": {
                "dtype": "F8_E4M3",
                "shape": [1],
                "data_offsets": [0, 1],
            }
        },
        metadata={
            "prompt": json.dumps(
                {
                    "1": {
                        "class_type": "UNETLoader",
                        "inputs": {"unet_name": "flux-2-klein-9b.safetensors"},
                    }
                }
            )
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(klein)
    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == klein.name)
    assert info.arch == ARCH_FLUX2_KLEIN_TRANSFORMER
    assert record.family == "runtime_asset"
    assert record.architecture == "flux2_klein"
    assert record.current_subdir == "flux/UNet"
    assert record.recommended_subdir == "flux2/UNet"
    assert record.should_move is True
    assert [checkpoint.id for checkpoint in checkpoints] == ["snofsSexNudesAndOtherFunStuff_v14Distilled"]
    assert checkpoints[0].architecture == "flux2_klein"
    assert checkpoints[0].kind == "flux2"


def test_z_image_gguf_is_runtime_asset_and_selectable_z_image_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    z_image = models / "flux" / "GGUF" / "fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf"
    z_image.parent.mkdir(parents=True)
    z_image.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(z_image)
    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == z_image.name)
    assert info.arch == ARCH_Z_IMAGE_TRANSFORMER
    assert record.family == "runtime_asset"
    assert record.architecture == "z_image"
    assert record.current_subdir == "flux/GGUF"
    assert record.recommended_subdir == "z-image/GGUF"
    assert record.should_move is True
    if os.name == "nt":
        # Z-Image GGUF is blocked on Windows: the fused GGUF CUDA kernels are
        # Linux-only and the fallback dequant path pages out 16 GB GPUs.
        assert checkpoints == []
    else:
        assert [checkpoint.id for checkpoint in checkpoints] == ["fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4"]
        assert checkpoints[0].architecture == "z_image"
        assert checkpoints[0].kind == "z-image"


def test_qwen_and_sana_diffusers_dirs_are_selectable_runtime_assets(tmp_path: Path):
    models = tmp_path / "models"
    specs = [
        (
            models / "qwen-image" / "Diffusers" / "Qwen-Image-2512",
            "QwenImagePipeline",
            "qwen_image",
            "qwen-image/Diffusers",
            "qwen-image",
        ),
        (
            models / "sana" / "Diffusers" / "Sana_Sprint_1.6B_1024px_diffusers",
            "SanaSprintPipeline",
            "sana",
            "sana/Diffusers",
            "sana",
        ),
        (
            models / "krea2" / "Diffusers" / "Krea-2-Turbo",
            "Krea2Pipeline",
            "krea2",
            "krea2/Diffusers",
            "krea2",
        ),
    ]
    for root, class_name, _, _, _ in specs:
        root.mkdir(parents=True)
        (root / "model_index.json").write_text(json.dumps({
            "_class_name": class_name,
            "transformer": ["diffusers", "GenericTransformer"],
        }), encoding="utf-8")
        (root / "transformer").mkdir()
        (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
        (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    by_path = {Path(record.path): record for record in records}
    by_kind = {checkpoint.kind: checkpoint for checkpoint in checkpoints}

    for root, _, architecture, recommended_subdir, kind in specs:
        record = by_path[root.resolve()]
        assert record.family == "runtime_asset"
        assert record.architecture == architecture
        assert record.recommended_subdir == recommended_subdir
        assert by_kind[kind].architecture == architecture
        assert by_kind[kind].file_count == 3
        assert by_kind[kind].asset_summary.startswith("folder, 3 files")
        assert by_kind[kind].asset_summary in by_kind[kind].title


@pytest.mark.parametrize(
    ("class_name", "architecture"),
    [
        ("QwenImageEditPipeline", "qwen_image_edit"),
        ("QwenImageEditPlusPipeline", "qwen_image_edit_plus"),
    ],
)
def test_unsupported_qwen_edit_pipelines_keep_distinct_label_and_are_not_selectable(
    tmp_path: Path, class_name: str, architecture: str
):
    models = tmp_path / "models"
    root = models / "staging" / class_name
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
    (root / "transformer.safetensors").write_bytes(b"weights")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    record = next(item for item in records if Path(item.path) == root.resolve())

    assert record.family == "runtime_asset"
    assert record.architecture == architecture
    assert record.recommended_subdir == "models to sort"
    assert not any(checkpoint.id == class_name for checkpoint in checkpoints)


def test_incomplete_qwen_diffusers_dir_is_not_selectable(tmp_path: Path):
    models = tmp_path / "models"
    root = models / "qwen-image" / "Diffusers" / "Qwen-Image"
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}), encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "transformer_blocks.0.attn.to_q.weight": "diffusion_pytorch_model-00001-of-00009.safetensors"
                },
            }
        ),
        encoding="utf-8",
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if Path(item.path) == root.resolve())
    assert record.family == "runtime_asset"
    assert record.architecture == "qwen_image"
    assert checkpoints == []


def test_sana_video_diffusers_dir_is_video_runtime_asset_not_image_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    root = models / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "SanaVideoPipeline"}), encoding="utf-8")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    record = next(item for item in records if Path(item.path) == root.resolve())

    assert record.family == "runtime_asset"
    assert record.architecture == "sana_video"
    assert record.recommended_subdir == "sana-video/Diffusers"
    assert record.should_move is False
    assert all(checkpoint.id != "SANA-Video_2B_480p_diffusers" for checkpoint in checkpoints)


def test_qwen_nunchaku_transformer_is_tracked_but_not_selectable_in_v1(tmp_path: Path):
    models = tmp_path / "models"
    transformer = models / "qwen-image" / "Nunchaku" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
    transformer.parent.mkdir(parents=True)
    _write_safetensors_header(
        transformer,
        {
            "transformer_blocks.0.attn.to_q.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == transformer.name)
    assert record.family == "runtime_asset"
    assert record.architecture == "qwen_image_nunchaku"
    assert record.current_subdir == "qwen-image/Nunchaku"
    assert record.recommended_subdir == "qwen-image/Nunchaku"
    assert all(checkpoint.id != "svdq-int4_r32-qwen-image-lightningv1.0-4steps" for checkpoint in checkpoints)


def test_flux2_and_z_image_component_dirs_are_support_assets_not_checkpoints(tmp_path: Path):
    models = tmp_path / "models"
    component_specs = [
        (
            models / "flux2" / "Components" / "FLUX.2-klein-4B",
            "Flux2KleinPipeline",
            "flux2_klein",
            "flux2/Components",
        ),
        (
            models / "z-image" / "Components" / "Z-Image-Turbo",
            "ZImagePipeline",
            "z_image",
            "z-image/Components",
        ),
    ]
    for component, class_name, _, _ in component_specs:
        (component / "text_encoder").mkdir(parents=True)
        (component / "model_index.json").write_text(
            json.dumps({"_class_name": class_name}),
            encoding="utf-8",
        )
        (component / "text_encoder" / "model-00001-of-00002.safetensors").write_bytes(
            b"not-a-real-safetensors-header"
        )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    by_path = {Path(record.path): record for record in records}

    for component, _, architecture, recommended_subdir in component_specs:
        record = by_path[component.resolve()]
        assert record.family == "text_encoder"
        assert record.architecture == architecture
        assert record.recommended_subdir == recommended_subdir
        assert record.should_move is False
    assert all(record.filename != "model-00001-of-00002.safetensors" for record in records)
    assert checkpoints == []


def test_flux_lora_header_overrides_wrong_folder(tmp_path: Path):
    models = tmp_path / "models"
    flux_lora = models / "Loras" / "Wan" / "flux_motion_rank16.safetensors"
    _write_safetensors_header(
        flux_lora,
        {
            "lora_transformer_double_blocks_0_img_attn_qkv.lora_down.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"ss_base_model_version": "flux1-dev"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(flux_lora)
    records = scan_and_write_model_inventory(flags)
    image_loras = scan_loras(flags)

    record = next(item for item in records if item.filename == flux_lora.name)
    assert info.arch == ARCH_FLUX_LORA
    assert record.family == "lora"
    assert record.architecture == "flux"
    assert record.recommended_subdir == "Loras/Flux"
    assert [item.filename for item in image_loras] == [flux_lora.name]
    assert image_loras[0].architecture == "flux"


def test_flux2_klein_lora_metadata_routes_to_flux2_lora_folder(tmp_path: Path):
    models = tmp_path / "models"
    klein_lora = models / "Loras" / "party_time_v2.0_klein9b.safetensors"
    _write_safetensors_header(
        klein_lora,
        {
            "lora_transformer_double_blocks_0_img_attn_qkv.lora_A.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"modelspec.architecture": "flux2-klein-9b/lora"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(klein_lora)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == klein_lora.name)

    assert info.arch == ARCH_FLUX2_KLEIN_LORA
    assert info.display_name.startswith("Flux.2 Klein LoRA")
    assert record.family == "lora"
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "Loras/Flux2"


def test_flux_adapter_formats_are_not_misclassified_as_checkpoints(tmp_path: Path):
    models = tmp_path / "models"
    flux_lora = models / "Loras" / "Flux" / "styles" / "furry_lora.safetensors"
    klein_lokr = models / "Loras" / "Flux2" / "Realism_Engine_Klein_V2.safetensors"
    _write_safetensors_header(
        flux_lora,
        {
            "double_blocks.0.processor.proj_lora1.down.weight": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
            },
            "double_blocks.0.processor.proj_lora1.up.weight": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [32, 64],
            },
        },
    )
    _write_safetensors_header(
        klein_lokr,
        {
            "lora_transformer_double_blocks_0_img_attn.lokr_w1": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
            },
            "lora_transformer_double_blocks_0_img_attn.lokr_w2": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [32, 64],
            },
        },
        {"ss_base_model_version": "flux2_klein_9b"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    by_path = {Path(record.path): record for record in records}

    assert by_path[flux_lora.resolve()].family == "lora"
    assert by_path[flux_lora.resolve()].architecture == "flux"
    assert by_path[flux_lora.resolve()].recommended_subdir == "Loras/Flux"
    assert by_path[klein_lokr.resolve()].family == "lora"
    assert by_path[klein_lokr.resolve()].architecture == "flux2_klein"
    assert by_path[klein_lokr.resolve()].recommended_subdir == "Loras/Flux2"
    assert "Flux.2 Klein LoRA" in read_model_info(klein_lokr).display_name
    assert checkpoints == []


def test_f2k_lokr_in_generic_lora_folder_keeps_flux2_identity(tmp_path: Path):
    models = tmp_path / "models"
    adapter = models / "Loras" / "F2K_style.safetensors"
    kontext_adapter = models / "Loras" / "kontext_style.safetensors"
    ambiguous_adapter = models / "Loras" / "notF2K_style.safetensors"
    _write_safetensors_header(
        adapter,
        {
            "lora_transformer_double_blocks_0_img_attn.lokr_w1": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
            },
        },
    )
    _write_safetensors_header(
        kontext_adapter,
        {
            "lora_transformer_double_blocks_0_img_attn.lora_down.weight": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
            },
        },
    )
    _write_safetensors_header(
        ambiguous_adapter,
        {
            "lora_transformer_double_blocks_0_img_attn.lokr_w1": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
            },
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = {item.filename: item for item in scan_and_write_model_inventory(flags)}
    record = records[adapter.name]
    assert record.family == "lora"
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "Loras/Flux2"
    kontext_record = records[kontext_adapter.name]
    assert kontext_record.family == "lora"
    assert kontext_record.architecture == "flux_kontext"
    assert kontext_record.recommended_subdir == "Loras/FluxKontext"
    assert "Flux Kontext LoRA" in read_model_info(kontext_adapter).display_name
    assert records[ambiguous_adapter.name].architecture != "flux2_klein"


def test_outer_revolutionf2k_directory_does_not_relabel_generic_flux_model(tmp_path: Path):
    models = tmp_path / "RevolutionF2K" / "models"
    model = models / "flux-model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == model.name)
    assert record.family == "runtime_asset"
    assert record.architecture == "flux"


def test_structured_flux_architecture_overrides_f2k_alias_in_filename(tmp_path: Path):
    models = tmp_path / "models"
    adapter = models / "Loras" / "RevolutionF2K_style.safetensors"
    _write_safetensors_header(
        adapter,
        {
            "lora_transformer_double_blocks_0_img_attn.lokr_w1": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
            },
        },
        {"modelspec.architecture": "flux/lora"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(adapter)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == adapter.name)

    assert info.arch == ARCH_FLUX_LORA
    assert record.architecture == "flux"


def test_flux2_klein_and_flux_kontext_names_override_generic_flux_gguf_header(tmp_path: Path):
    models = tmp_path / "models"
    kontext = models / "unet" / "flux1-kontext-dev-Q5_K_M.gguf"
    mixed_label_kontext = models / "unet" / "flux2-kontext-dev-Q5_K_M.gguf"
    klein = models / "flux" / "GGUF" / "unstableRevolutionF2K_AlphaF2K4BQ80.gguf"
    kontext.parent.mkdir(parents=True)
    klein.parent.mkdir(parents=True)
    # Invalid-but-present GGUF files exercise the filename fallback path.
    kontext.write_bytes(b"GGUF")
    mixed_label_kontext.write_bytes(b"GGUF")
    klein.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    by_path = {Path(record.path): record for record in records}

    assert by_path[kontext.resolve()].family == "runtime_asset"
    assert by_path[kontext.resolve()].architecture == "flux_kontext"
    assert by_path[mixed_label_kontext.resolve()].architecture == "flux_kontext"
    assert by_path[klein.resolve()].family == "runtime_asset"
    assert by_path[klein.resolve()].architecture == "flux2_klein"
    assert ARCH_FLUX_KONTEXT_TRANSFORMER == "flux-kontext-transformer"
    assert ARCH_FLUX2_KLEIN_TRANSFORMER == "flux2-klein-transformer"


def test_generic_flux2_inventory_is_not_classified_as_klein(tmp_path: Path):
    models = tmp_path / "models"
    generic = models / "flux2" / "GGUF" / "flux2-base-Q4_K_M.gguf"
    generic.parent.mkdir(parents=True)
    generic.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    record = next(item for item in scan_and_write_model_inventory(flags) if Path(item.path) == generic.resolve())

    assert record.architecture == "flux2"
    assert record.family == "runtime_asset"
    assert record.recommended_subdir == "models to sort"


def test_longcat_gguf_is_labeled_separately_and_excluded_from_flux_picker(tmp_path: Path):
    models = tmp_path / "models"
    longcat = models / "longcat" / "LongCat-Image-Edit-Turbo-Q5_K_M.gguf"
    longcat.parent.mkdir(parents=True)
    longcat.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    by_path = {Path(record.path): record for record in records}

    assert by_path[longcat.resolve()].family == "runtime_asset"
    assert by_path[longcat.resolve()].architecture == ARCH_LONGCAT_IMAGE
    assert by_path[longcat.resolve()].recommended_subdir == "longcat"
    assert scan_from_flags(flags) == []


def test_placeholder_adapter_title_falls_back_to_filename_without_duplicate_role(tmp_path: Path):
    models = tmp_path / "models"
    wan_lora = models / "wan" / "Safetensor" / "WAN2.2_I2V-T2V_Flatchested_High.safetensors"
    _write_safetensors_header(
        wan_lora,
        {
            "diffusion_model.blocks.0.cross_attn.k.lora_A.weight": {
                "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
            }
        },
        {
            "modelspec.title": "Your High Noise LoRA Name Here",
            "ss_base_model_version": "wan2.2",
            "ss_network_module": "networks.lora_wan",
        },
    )

    info = read_model_info(wan_lora)
    record = next(item for item in scan_and_write_model_inventory(
        RuntimeFlags(data_dir=tmp_path, models_dir=models)
    ) if item.filename == wan_lora.name)

    assert "Your High Noise" not in info.display_name
    assert "Flatchested High" in info.display_name
    assert "LoRA LoRA" not in info.display_name
    assert record.family == "lora"
    assert record.architecture == "wan"


def test_ltx_headers_route_gguf_lora_and_vaes_to_ltx_folders(tmp_path: Path):
    models = tmp_path / "models"
    gguf = models / "ltx" / "GGUF" / "ltx23DISTILLEDGGUF_q2k.gguf"
    lora = models / "Stable-diffusion" / "ltx23_ltx2322bDistilled.safetensors"
    video_vae = models / "Stable-diffusion" / "ltx23FP4_ltx23VideoVae.safetensors"
    audio_vae = models / "Stable-diffusion" / "ltx23FP4_ltx23AudioVae.safetensors"
    text_encoder = models / "ltx" / "text_encoder" / "gemma_3_12B_it_fp4_mixed.safetensors"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    _write_safetensors_header(
        lora,
        {
            "diffusion_model.transformer_blocks.0.attn.q_proj.lora_A.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"modelspec.title": "LTX 2.3 Distilled LoRA"},
    )
    _write_safetensors_header(
        video_vae,
        {
            "encoder.conv_in.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"config": '{"class_name":"CausalVideoAutoencoder"}', "modelspec.title": "LTX Video VAE"},
    )
    _write_safetensors_header(
        audio_vae,
        {
            "audio_vae.encoder.conv_in.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"modelspec.title": "LTX Audio VAE"},
    )
    _write_safetensors_header(
        text_encoder,
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    by_name = {item.filename: item for item in scan_and_write_model_inventory(flags)}
    info_by_name = {path.name: read_model_info(path) for path in (gguf, lora, video_vae, audio_vae, text_encoder)}

    assert info_by_name[gguf.name].arch == ARCH_LTX_TRANSFORMER
    assert by_name[gguf.name].family == "runtime_asset"
    assert by_name[gguf.name].recommended_subdir == "ltx/GGUF"
    assert info_by_name[lora.name].arch == ARCH_LTX_LORA
    assert by_name[lora.name].family == "lora"
    assert by_name[lora.name].recommended_subdir == "ltx/loras"
    assert info_by_name[video_vae.name].arch == ARCH_LTX_VAE
    assert by_name[video_vae.name].family == "vae"
    assert by_name[video_vae.name].recommended_subdir == "ltx/vae"
    assert info_by_name[audio_vae.name].arch == ARCH_LTX_AUDIO_VAE
    assert by_name[audio_vae.name].family == "vae"
    assert by_name[audio_vae.name].recommended_subdir == "ltx/audio_vae"
    assert info_by_name[text_encoder.name].role == ROLE_TEXT_ENCODER
    assert by_name[text_encoder.name].family == "text_encoder"
    assert by_name[text_encoder.name].recommended_subdir == "ltx/text_encoder"


def test_thumbnail_metadata_does_not_drive_ltx_detection(tmp_path: Path):
    path = tmp_path / "models" / "Stable-diffusion" / "sd15-with-thumbnail.safetensors"
    _write_safetensors_header(
        path,
        {
            "model.diffusion_model.input_blocks.0.0.weight": {
                "dtype": "F16",
                "shape": [320, 4, 3, 3],
                "data_offsets": [0, 23040],
            }
        },
        {
            "modelspec.title": "Stable Diffusion v1.5",
            "modelspec.thumbnail": "data:image/jpeg;base64,this-value-mentions-ltx-by-accident",
        },
    )

    info = read_model_info(path)

    assert info.arch == ARCH_SD_CHECKPOINT


def test_wan_lora_header_overrides_wrong_flux_folder_and_stays_out_of_image_loras(tmp_path: Path):
    models = tmp_path / "models"
    wan_lora = models / "Loras" / "Flux" / "wan_motion_rank16.safetensors"
    _write_safetensors_header(
        wan_lora,
        {
            "diffusion_model.blocks.0.cross_attn.k.lora_A.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"ss_base_model_version": "wan2.2"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    image_loras = scan_loras(flags)

    record = next(item for item in records if item.filename == wan_lora.name)
    assert record.family == "lora"
    assert record.architecture == "wan"
    assert record.recommended_subdir == "Loras/Wan"
    assert image_loras == []


def test_t5xxl_and_umT5_headers_stay_separate(tmp_path: Path):
    models = tmp_path / "models"
    t5xxl = models / "Textencoder" / "t5xxl_fp8_e4m3fn.safetensors"
    umt5 = models / "Textencoder" / "umt5-xxl_fp8_e4m3fn.safetensors"
    tensor = {
        "encoder.block.0.layer.0.SelfAttention.q.weight": {
            "dtype": "F16",
            "shape": [4, 4],
            "data_offsets": [0, 32],
        }
    }
    _write_safetensors_header(t5xxl, tensor, {})
    _write_safetensors_header(umt5, tensor, {})
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    t5_info = read_model_info(t5xxl)
    umt5_info = read_model_info(umt5)
    records = {item.filename: item for item in scan_and_write_model_inventory(flags)}

    assert t5_info.arch == ARCH_T5XXL_ENCODER
    assert t5_info.is_t5xxl() is True
    assert records[t5xxl.name].family == "text_encoder"
    assert records[t5xxl.name].architecture == "flux"
    assert records[t5xxl.name].recommended_subdir == "flux/Textencoder"
    assert umt5_info.arch == ARCH_UMT5_ENCODER
    assert umt5_info.is_t5xxl() is False
    assert records[umt5.name].architecture == "wan"
    assert records[umt5.name].recommended_subdir == "Textencoder"


def test_clip_encoder_under_noncanonical_flux_named_folder_is_not_flux_support(tmp_path: Path):
    models = tmp_path / "models"
    clip = models / "Flux test data" / "clip_l.safetensors"
    _write_safetensors_header(
        clip,
        {"text_model.encoder.layers.0.weight": {
            "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
        }},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    record = next(item for item in scan_and_write_model_inventory(flags) if item.path == str(clip.resolve()))

    assert record.family == "text_encoder"
    assert record.architecture == "unknown"
    assert record.recommended_subdir == "Textencoder"


def test_clip_encoder_under_canonical_flux_component_folder_keeps_flux_hint(tmp_path: Path):
    models = tmp_path / "models"
    clip = models / "flux" / "Textencoder" / "clip_l.safetensors"
    _write_safetensors_header(
        clip,
        {"text_model.encoder.layers.0.weight": {
            "dtype": "F16", "shape": [4, 4], "data_offsets": [0, 32],
        }},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    record = next(item for item in scan_and_write_model_inventory(flags) if item.path == str(clip.resolve()))

    assert record.family == "text_encoder"
    assert record.architecture == "flux"
    assert record.recommended_subdir == "flux/Textencoder"


def test_qwen_text_encoder_stays_in_textencoder_folder(tmp_path: Path):
    models = tmp_path / "models"
    qwen = models / "Textencoder" / "qwen_3_8b_fp8mixed.safetensors"
    _write_safetensors_header(
        qwen,
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F8_E4M3",
                "shape": [4, 4],
                "data_offsets": [0, 16],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(qwen)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == qwen.name)

    assert info.role == ROLE_TEXT_ENCODER
    assert record.family == "text_encoder"
    assert record.architecture == "unknown"
    assert record.recommended_subdir == "Textencoder"


def test_flux2_vae_role_overrides_transformer_filename(tmp_path: Path):
    models = tmp_path / "models"
    vae = models / "VAE" / "flux2-vae.safetensors"
    _write_safetensors_header(
        vae,
        {
            "encoder.conv_in.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [0, 64],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(vae)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == vae.name)

    assert info.role == ROLE_VAE
    assert record.family == "vae"
    assert record.architecture == "flux2"
    assert record.recommended_subdir == "VAE"


def test_flux_ae_vae_is_not_treated_as_wan_vae(tmp_path: Path):
    models = tmp_path / "models"
    ae = models / "VAE" / "ae.safetensors"
    _write_safetensors_header(
        ae,
        {
            "encoder.conv_in.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(ae)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == ae.name)

    assert info.arch == ARCH_FLUX_VAE
    assert record.family == "vae"
    assert record.architecture == "flux"
    assert record.recommended_subdir == "flux/VAE"


def test_embeddings_folder_does_not_enter_checkpoint_catalog(tmp_path: Path):
    models = tmp_path / "models"
    embedding = models / "embeddings" / "EasyNegative.safetensors"
    _write_safetensors_header(
        embedding,
        {
            "emb_params": {
                "dtype": "F16",
                "shape": [77, 768],
                "data_offsets": [0, 118272],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == "EasyNegative.safetensors")
    assert record.family == "embedding"
    assert record.recommended_subdir == "embeddings"
    assert checkpoints == []


def test_ip_adapter_weights_do_not_enter_checkpoint_catalog(tmp_path: Path):
    models = tmp_path / "models"
    adapter = models / "ipadapter" / "SDXL" / "ip-adapter-plus_sdxl_vit-h.safetensors"
    _write_safetensors_header(
        adapter,
        {"image_proj.latents": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == adapter.name)
    assert record.family == "ip_adapter"
    assert record.recommended_subdir == "ipadapter"
    assert checkpoints == []


def test_truncated_qwen_safetensors_are_marked_invalid_and_not_selectable(tmp_path: Path):
    models = tmp_path / "models"
    truncated = models / "staging-qwen_image_edit_2511_int8_convrot.safetensors"
    truncated.parent.mkdir(parents=True)
    header = json.dumps(
        {
            "transformer_blocks.0.attn.weight": {
                "dtype": "F16",
                "shape": [100, 100],
                "data_offsets": [0, 20000],
            }
        }
    ).encode("utf-8")
    truncated.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == truncated.name)
    assert record.family == "invalid_asset"
    assert record.architecture == "qwen_image_edit"
    assert record.header_identifiers["integrity"] == "invalid_safetensors_tensor_ranges_or_framing"
    assert checkpoints == []


def test_llm_folder_weights_do_not_enter_checkpoint_catalog(tmp_path: Path):
    models = tmp_path / "models"
    gguf = models / "LLM" / "GGUF" / "gemma-3-12b-it-heretic" / "gemma-3-12b-it-heretic-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    by_name = {record.filename: record for record in records}

    assert by_name[gguf.name].family == "llm"
    assert by_name[gguf.name].recommended_subdir == "LLM/GGUF/gemma-3-12b-it-heretic"
    assert by_name[gguf.name].should_move is False
    assert checkpoints == []


def test_lora_architecture_can_come_from_parent_folder(tmp_path: Path):
    models = tmp_path / "models"
    lora = models / "Loras" / "SDXL" / "folder_tagged_style.safetensors"
    _write_safetensors_header(
        lora,
        {
            "lora_unet_down_blocks_0.lora_down.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"ss_network_module": "networks.lora"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    image_loras = scan_loras(flags)

    record = next(item for item in records if item.filename == "folder_tagged_style.safetensors")
    assert record.family == "lora"
    assert record.architecture == "sdxl"
    assert record.recommended_subdir == "Loras/SDXL"
    assert image_loras[0].architecture == "sdxl"


def test_ltx_assets_are_recommended_to_ltx_folders(tmp_path: Path):
    models = tmp_path / "models"
    checkpoint = models / "Stable-diffusion" / "ltx-2.3-22b-distilled-1.1.safetensors"
    upscaler = models / "upscalers" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    for path in (checkpoint, upscaler):
        _write_safetensors_header(
            path,
            {
                "transformer_blocks.0.weight": {
                    "dtype": "F16",
                    "shape": [4, 4],
                    "data_offsets": [0, 32],
                }
            },
        )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = {item.filename: item for item in scan_and_write_model_inventory(flags)}

    assert records[checkpoint.name].family == "runtime_asset"
    assert records[checkpoint.name].architecture == "ltx"
    assert records[checkpoint.name].recommended_subdir == "ltx/checkpoints"
    assert records[upscaler.name].recommended_subdir == "ltx/upscalers"
