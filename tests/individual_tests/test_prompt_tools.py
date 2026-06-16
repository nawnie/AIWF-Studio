"""tests/test_prompt_tools.py

Tests for aiwf.services.prompt_tools (Phase 7 agentic tool layer).

All tests are pure-Python, require no GPU, and use tmp_path fixtures to
avoid touching the real checkpoint/LoRA directories.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from aiwf.services.prompt_tools import (
    LocalFileInfo,
    PromptDraft,
    PromptToolsService,
    RecommendedSettings,
    SafetensorsHeader,
    _read_safetensors_header_raw,
    _safe_resolve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fake_safetensors(path: Path, metadata: dict[str, str], tensor_keys: list[str]) -> None:
    """Write a minimal valid .safetensors file with the given header."""
    # Build the JSON header dict
    header_dict: dict = {"__metadata__": metadata}
    # Add stub tensor entries (dtype/shape/offsets are arbitrary here)
    offset = 0
    for k in tensor_keys:
        header_dict[k] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    raw = json.dumps(header_dict).encode("utf-8")
    length_bytes = struct.pack("<Q", len(raw))
    # Append the data section (just zeros equal to total offsets)
    path.write_bytes(length_bytes + raw + b"\x00" * offset)


@pytest.fixture()
def checkpoint_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture()
def lora_dir(tmp_path: Path) -> Path:
    d = tmp_path / "loras"
    d.mkdir()
    return d


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture()
def svc(checkpoint_dir: Path, lora_dir: Path, output_dir: Path) -> PromptToolsService:
    return PromptToolsService(
        checkpoint_dir=checkpoint_dir,
        lora_dir=lora_dir,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# _safe_resolve
# ---------------------------------------------------------------------------

class TestSafeResolve:
    def test_valid_filename(self, tmp_path: Path) -> None:
        root = tmp_path / "models"
        root.mkdir()
        (root / "model.safetensors").write_bytes(b"")
        resolved = _safe_resolve(root, "model.safetensors")
        assert resolved == (root / "model.safetensors").resolve()

    def test_traversal_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "models"
        root.mkdir()
        with pytest.raises(PermissionError, match="resolves outside"):
            _safe_resolve(root, "../../etc/passwd")


# ---------------------------------------------------------------------------
# _read_safetensors_header_raw
# ---------------------------------------------------------------------------

class TestReadSafetensorsHeaderRaw:
    def test_reads_metadata_and_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "model.safetensors"
        _make_fake_safetensors(
            path,
            metadata={"ss_base_model_version": "sd_v1", "modelspec.title": "Test Model"},
            tensor_keys=["model.weight", "model.bias"],
        )
        metadata, tensor_keys = _read_safetensors_header_raw(path)
        assert metadata["ss_base_model_version"] == "sd_v1"
        assert metadata["modelspec.title"] == "Test Model"
        assert "model.weight" in tensor_keys
        assert "model.bias" in tensor_keys

    def test_empty_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "model.safetensors"
        _make_fake_safetensors(path, metadata={}, tensor_keys=["layer.weight"])
        metadata, tensor_keys = _read_safetensors_header_raw(path)
        assert metadata == {}
        assert "layer.weight" in tensor_keys

    def test_too_short_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.safetensors"
        path.write_bytes(b"\x00\x01")  # less than 8 bytes
        with pytest.raises(ValueError, match="too short"):
            _read_safetensors_header_raw(path)

    def test_header_size_cap(self, tmp_path: Path) -> None:
        path = tmp_path / "huge_header.safetensors"
        # Claim a header of 11 MB
        huge_len = 11 * 1024 * 1024
        path.write_bytes(struct.pack("<Q", huge_len) + b"\x00" * 8)
        with pytest.raises(ValueError, match="safety cap"):
            _read_safetensors_header_raw(path)


# ---------------------------------------------------------------------------
# list_local_checkpoints
# ---------------------------------------------------------------------------

class TestListLocalCheckpoints:
    def test_finds_safetensors(
        self, svc: PromptToolsService, checkpoint_dir: Path
    ) -> None:
        (checkpoint_dir / "sd15.safetensors").write_bytes(b"fake")
        (checkpoint_dir / "sdxl.ckpt").write_bytes(b"fake")
        (checkpoint_dir / "readme.txt").write_bytes(b"ignore me")
        results = svc.list_local_checkpoints()
        exts = {r.extension for r in results}
        filenames = {r.filename for r in results}
        assert ".safetensors" in exts
        assert ".ckpt" in exts
        assert "readme.txt" not in filenames

    def test_empty_dir(self, svc: PromptToolsService) -> None:
        assert svc.list_local_checkpoints() == []

    def test_missing_dir(self, output_dir: Path, lora_dir: Path) -> None:
        svc = PromptToolsService(
            checkpoint_dir=Path("/nonexistent/checkpoints"),
            lora_dir=lora_dir,
            output_dir=output_dir,
        )
        assert svc.list_local_checkpoints() == []

    def test_returns_localfileinfo(
        self, svc: PromptToolsService, checkpoint_dir: Path
    ) -> None:
        (checkpoint_dir / "model.safetensors").write_bytes(b"data")
        results = svc.list_local_checkpoints()
        assert len(results) == 1
        info = results[0]
        assert isinstance(info, LocalFileInfo)
        assert info.name == "model"
        assert info.size_bytes == 4

    def test_audit_log_written(
        self, svc: PromptToolsService, output_dir: Path
    ) -> None:
        svc.list_local_checkpoints()
        log = output_dir / ".agent_tool_log.jsonl"
        assert log.is_file()
        entry = json.loads(log.read_text().strip().splitlines()[-1])
        assert entry["tool"] == "list_local_checkpoints"


# ---------------------------------------------------------------------------
# list_local_loras
# ---------------------------------------------------------------------------

class TestListLocalLoras:
    def test_finds_lora_files(
        self, svc: PromptToolsService, lora_dir: Path
    ) -> None:
        (lora_dir / "my_lora.safetensors").write_bytes(b"x")
        (lora_dir / "old_lora.pt").write_bytes(b"x")
        (lora_dir / "notes.md").write_bytes(b"ignore")
        results = svc.list_local_loras()
        exts = {r.extension for r in results}
        assert ".safetensors" in exts
        assert ".pt" in exts
        assert len(results) == 2


# ---------------------------------------------------------------------------
# read_safetensors_metadata
# ---------------------------------------------------------------------------

class TestReadSafetensorsMetadata:
    def test_reads_checkpoint_metadata(
        self, svc: PromptToolsService, checkpoint_dir: Path
    ) -> None:
        path = checkpoint_dir / "v1.safetensors"
        _make_fake_safetensors(
            path,
            metadata={"modelspec.title": "My SD 1.5"},
            tensor_keys=["unet.weight"],
        )
        result = svc.read_safetensors_metadata("v1.safetensors")
        assert isinstance(result, SafetensorsHeader)
        assert result.metadata["modelspec.title"] == "My SD 1.5"
        assert "unet.weight" in result.tensor_keys

    def test_reads_lora_metadata(
        self, svc: PromptToolsService, lora_dir: Path
    ) -> None:
        path = lora_dir / "char_lora.safetensors"
        _make_fake_safetensors(path, metadata={"ss_output_name": "char"}, tensor_keys=[])
        result = svc.read_safetensors_metadata("char_lora.safetensors")
        assert result.metadata["ss_output_name"] == "char"

    def test_not_found_raises(self, svc: PromptToolsService) -> None:
        with pytest.raises(FileNotFoundError):
            svc.read_safetensors_metadata("nonexistent.safetensors")

    def test_traversal_rejected(self, svc: PromptToolsService) -> None:
        with pytest.raises(PermissionError):
            svc.read_safetensors_metadata("../../etc/passwd")


# ---------------------------------------------------------------------------
# build_prompt_draft
# ---------------------------------------------------------------------------

class TestBuildPromptDraft:
    def test_subject_only(self, svc: PromptToolsService) -> None:
        draft = svc.build_prompt_draft("a cat on a mat")
        assert isinstance(draft, PromptDraft)
        assert draft.positive == "a cat on a mat"
        assert draft.assembled == "a cat on a mat"
        assert draft.lora_tags == []

    def test_style_with_placeholder(self, svc: PromptToolsService) -> None:
        draft = svc.build_prompt_draft(
            "a sunset", style_template="best quality photo of {prompt}, 8k"
        )
        assert draft.positive == "best quality photo of a sunset, 8k"

    def test_style_without_placeholder(self, svc: PromptToolsService) -> None:
        draft = svc.build_prompt_draft("a dog", style_template="oil painting")
        assert "a dog" in draft.positive
        assert "oil painting" in draft.positive

    def test_lora_tags_appended(self, svc: PromptToolsService) -> None:
        draft = svc.build_prompt_draft(
            "a warrior", lora_names=["hero_v2", "armor_detail"], lora_weights=[0.8, 1.0]
        )
        assert "<lora:hero_v2:0.80>" in draft.assembled
        assert "<lora:armor_detail:1.00>" in draft.assembled
        assert len(draft.lora_tags) == 2

    def test_default_lora_weight(self, svc: PromptToolsService) -> None:
        draft = svc.build_prompt_draft("portrait", lora_names=["face_fix"])
        assert "<lora:face_fix:1.00>" in draft.assembled

    def test_negative_preserved(self, svc: PromptToolsService) -> None:
        draft = svc.build_prompt_draft("landscape", negative="blurry, low quality")
        assert draft.negative == "blurry, low quality"


# ---------------------------------------------------------------------------
# recommend_settings
# ---------------------------------------------------------------------------

class TestRecommendSettings:
    @pytest.mark.parametrize("arch,goal", [
        ("sd15", "speed"), ("sd15", "balanced"), ("sd15", "quality"),
        ("sdxl", "speed"), ("sdxl", "balanced"), ("sdxl", "quality"),
        ("wan", "speed"), ("wan", "balanced"), ("wan", "quality"),
    ])
    def test_known_combos(
        self, svc: PromptToolsService, arch: str, goal: str
    ) -> None:
        result = svc.recommend_settings(arch, goal)
        assert isinstance(result, RecommendedSettings)
        assert result.width > 0
        assert result.height > 0
        assert result.steps > 0
        assert result.sampler != ""

    def test_unknown_architecture_returns_default(self, svc: PromptToolsService) -> None:
        result = svc.recommend_settings("flux", "balanced")
        assert isinstance(result, RecommendedSettings)
        assert result.rationale.startswith("Unknown architecture")

    def test_case_insensitive(self, svc: PromptToolsService) -> None:
        r1 = svc.recommend_settings("SDXL", "QUALITY")
        r2 = svc.recommend_settings("sdxl", "quality")
        assert r1 == r2

    def test_sdxl_speed_native_resolution(self, svc: PromptToolsService) -> None:
        result = svc.recommend_settings("sdxl", "speed")
        assert result.width == 1024
        assert result.height == 1024


# ---------------------------------------------------------------------------
# generate_workflow_json
# ---------------------------------------------------------------------------

class TestGenerateWorkflowJson:
    def test_output_is_dict(self, svc: PromptToolsService) -> None:
        wf = svc.generate_workflow_json("v1.safetensors", "a forest")
        assert isinstance(wf, dict)

    def test_required_nodes_present(self, svc: PromptToolsService) -> None:
        wf = svc.generate_workflow_json("v1.safetensors", "a city at night")
        class_types = {v["class_type"] for v in wf.values()}
        assert "CheckpointLoaderSimple" in class_types
        assert "KSampler" in class_types
        assert "VAEDecode" in class_types
        assert "SaveImage" in class_types
        assert "CLIPTextEncode" in class_types

    def test_checkpoint_name_in_loader(self, svc: PromptToolsService) -> None:
        wf = svc.generate_workflow_json("sdxl_base.safetensors", "portrait")
        loader = next(
            v for v in wf.values() if v["class_type"] == "CheckpointLoaderSimple"
        )
        assert loader["inputs"]["ckpt_name"] == "sdxl_base.safetensors"

    def test_prompt_in_positive_encoder(self, svc: PromptToolsService) -> None:
        wf = svc.generate_workflow_json("model.safetensors", "my prompt text")
        # First CLIPTextEncode should carry the positive prompt
        text_encoders = [
            v for v in wf.values() if v["class_type"] == "CLIPTextEncode"
        ]
        positive_texts = [t["inputs"]["text"] for t in text_encoders]
        assert "my prompt text" in positive_texts

    def test_dimensions_in_latent(self, svc: PromptToolsService) -> None:
        wf = svc.generate_workflow_json(
            "model.safetensors", "sky", width=1024, height=768
        )
        latent = next(
            v for v in wf.values() if v["class_type"] == "EmptyLatentImage"
        )
        assert latent["inputs"]["width"] == 1024
        assert latent["inputs"]["height"] == 768

    def test_is_json_serialisable(self, svc: PromptToolsService) -> None:
        wf = svc.generate_workflow_json("model.safetensors", "test")
        serialised = json.dumps(wf)
        assert len(serialised) > 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_multiple_calls_append(
        self, svc: PromptToolsService, output_dir: Path
    ) -> None:
        svc.list_local_checkpoints()
        svc.list_local_loras()
        svc.recommend_settings("sd15", "speed")
        log = output_dir / ".agent_tool_log.jsonl"
        lines = [l for l in log.read_text().strip().splitlines() if l]
        assert len(lines) == 3
        tools = [json.loads(l)["tool"] for l in lines]
        assert "list_local_checkpoints" in tools
        assert "list_local_loras" in tools
        assert "recommend_settings" in tools

    def test_error_calls_logged(
        self, svc: PromptToolsService, output_dir: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            svc.read_safetensors_metadata("no_such_file.safetensors")
        log = output_dir / ".agent_tool_log.jsonl"
        entry = json.loads(log.read_text().strip().splitlines()[-1])
        assert "error" in entry["result_summary"]
