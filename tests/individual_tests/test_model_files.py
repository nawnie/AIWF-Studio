from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.checkpoints import (
    diffusers_dir_has_required_local_files,
    missing_diffusers_local_files,
)
from aiwf.services.model_files import (
    configured_model_roots,
    indexed_safetensors_shards_ready,
    indexed_weight_shards_ready,
    nonempty_json_object,
    resolve_model_asset,
)


def test_indexed_safetensors_requires_nonempty_confined_shards(tmp_path: Path):
    component = tmp_path / "text_encoder"
    component.mkdir()
    index = component / "model.safetensors.index.json"
    shard = component / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    index.write_text(json.dumps({"weight_map": {"tensor": shard.name}}), encoding="utf-8")

    assert indexed_safetensors_shards_ready(component, index) is True

    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    index.write_text(json.dumps({"weight_map": {"tensor": "../outside.safetensors"}}), encoding="utf-8")
    assert indexed_safetensors_shards_ready(component, index) is False

    index.write_text(json.dumps({"weight_map": {"tensor": str(outside)}}), encoding="utf-8")
    assert indexed_safetensors_shards_ready(component, index) is False


def test_indexed_safetensors_rejects_missing_and_empty_shards(tmp_path: Path):
    component = tmp_path / "text_encoder"
    component.mkdir()
    index = component / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"tensor": "missing.safetensors"}}), encoding="utf-8")
    assert indexed_safetensors_shards_ready(component, index) is False

    (component / "missing.safetensors").touch()
    assert indexed_safetensors_shards_ready(component, index) is False


def test_indexed_safetensors_rejects_non_safetensors_files(tmp_path: Path):
    component = tmp_path / "text_encoder"
    component.mkdir()
    metadata = component / "config.json"
    metadata.write_text('{"hidden_size": 8}', encoding="utf-8")
    index = component / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"tensor": metadata.name}}), encoding="utf-8")

    assert indexed_safetensors_shards_ready(component, index) is False


@pytest.mark.parametrize("shard_name", ["../outside.safetensors", "C:/outside.safetensors", "config.json"])
def test_indexed_diffusers_shards_reject_unsafe_or_non_weight_paths(tmp_path: Path, shard_name: str):
    component = tmp_path / "transformer"
    component.mkdir()
    index = component / "diffusion_pytorch_model.safetensors.index.json"
    (tmp_path / "outside.safetensors").write_bytes(b"outside")
    (component / "config.json").write_text('{"hidden_size": 8}', encoding="utf-8")
    index.write_text(json.dumps({"weight_map": {"weight": shard_name}}), encoding="utf-8")

    assert indexed_weight_shards_ready(component, index) is False


@pytest.mark.parametrize("payload", ["not json", "{}", '{"weight_map": {}}', '{"weight_map": {"w": 4}}'])
def test_indexed_diffusers_shards_reject_malformed_indexes(tmp_path: Path, payload: str):
    component = tmp_path / "transformer"
    component.mkdir()
    index = component / "model.safetensors.index.json"
    index.write_text(payload, encoding="utf-8")

    assert indexed_weight_shards_ready(component, index) is False


@pytest.mark.parametrize("suffix", [".safetensors", ".bin", ".pt", ".onnx"])
def test_indexed_diffusers_shards_accept_supported_nested_formats(tmp_path: Path, suffix: str):
    component = tmp_path / "transformer"
    nested = component / "weights" / "part-00001"
    nested.mkdir(parents=True)
    shard = nested / f"model{suffix}"
    shard.write_bytes(b"weights")
    index = component / "model.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "weights/part-00001/" + shard.name}}), encoding="utf-8")

    assert indexed_weight_shards_ready(component, index) is True


def test_indexed_diffusers_shards_reject_symlink_escape(tmp_path: Path):
    component = tmp_path / "transformer"
    component.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    link = component / "linked.safetensors"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("File symlinks are unavailable in this Windows context")
    index = component / "model.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": link.name}}), encoding="utf-8")

    assert indexed_weight_shards_ready(component, index) is False


def test_component_metadata_must_be_nonempty_json_object(tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    assert nonempty_json_object(metadata) is False
    metadata.write_text("", encoding="utf-8")
    assert nonempty_json_object(metadata) is False
    metadata.write_text('{"key": "value"}', encoding="utf-8")
    assert nonempty_json_object(metadata) is True


def test_diffusers_readiness_rejects_traversal_and_empty_weight_maps(tmp_path: Path):
    model = tmp_path / "Sana"
    component = model / "text_encoder"
    component.mkdir(parents=True)
    (model / "model_index.json").write_text('{"_class_name": "SanaPipeline"}', encoding="utf-8")
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    index = component / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}), encoding="utf-8")

    assert diffusers_dir_has_required_local_files(model) is False
    assert missing_diffusers_local_files(model) == [index]

    index.write_text('{"weight_map": {}}', encoding="utf-8")
    assert diffusers_dir_has_required_local_files(model) is False
    assert missing_diffusers_local_files(model) == [index]


def test_configured_model_roots_are_deduplicated_and_preserve_priority(tmp_path: Path):
    primary = tmp_path / "primary"
    shared = tmp_path / "shared"
    flags = RuntimeFlags(
        data_dir=tmp_path / "app",
        models_dir=primary,
        extra_model_dirs=[shared, primary],
    )

    assert configured_model_roots(flags) == [primary.resolve(), shared.resolve()]


def test_resolve_model_asset_prefers_primary_nonempty_candidate(tmp_path: Path):
    primary = tmp_path / "primary"
    shared = tmp_path / "shared"
    relative = Path("flux") / "Textencoder" / "t5.safetensors"
    primary_path = primary / relative
    shared_path = shared / relative
    primary_path.parent.mkdir(parents=True)
    shared_path.parent.mkdir(parents=True)
    primary_path.touch()
    shared_path.write_bytes(b"shared")
    flags = RuntimeFlags(data_dir=tmp_path / "app", models_dir=primary, extra_model_dirs=[shared])

    assert resolve_model_asset(flags, (relative,)) == shared_path.resolve()

    primary_path.write_bytes(b"primary")
    assert resolve_model_asset(flags, (relative,)) == primary_path.resolve()


def test_resolve_model_asset_rejects_traversal_and_escaping_link(tmp_path: Path):
    primary = tmp_path / "primary"
    shared = tmp_path / "shared"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    link = primary / "linked.safetensors"
    primary.mkdir()
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("File symlinks are unavailable in this Windows context")
    flags = RuntimeFlags(data_dir=tmp_path / "app", models_dir=primary, extra_model_dirs=[shared])

    fallback = primary / "expected.safetensors"
    assert resolve_model_asset(flags, ("../outside.safetensors", "linked.safetensors"), fallback=fallback) == fallback
