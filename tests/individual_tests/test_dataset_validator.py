"""Tests for aiwf/services/training/dataset_validator.py."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aiwf.services.training.dataset_validator import (
    DatasetValidator,
    ValidationResult,
    _check_base_model,
    _check_dataset_dir,
    _check_output_dir,
)

validator = DatasetValidator()


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_passed_has_no_errors(self):
        r = ValidationResult.passed()
        assert r.ok and not r.errors

    def test_failed_has_errors(self):
        r = ValidationResult.failed(["bad thing"])
        assert not r.ok and r.errors == ["bad thing"]

    def test_merge_combines_errors(self):
        a = ValidationResult.passed(warnings=["w1"])
        b = ValidationResult.failed(["e1"], warnings=["w2"])
        a.merge(b)
        assert not a.ok
        assert "e1" in a.errors
        assert "w1" in a.warnings and "w2" in a.warnings


# ---------------------------------------------------------------------------
# _check_dataset_dir
# ---------------------------------------------------------------------------

class TestCheckDatasetDir:
    def test_missing_dir_is_error(self, tmp_path):
        errs = _check_dataset_dir(str(tmp_path / "nonexistent"), warnings=[])
        assert any("does not exist" in e for e in errs)

    def test_not_a_directory_is_error(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        errs = _check_dataset_dir(str(f), warnings=[])
        assert any("not a directory" in e for e in errs)

    def test_empty_dir_is_error(self, tmp_path):
        errs = _check_dataset_dir(str(tmp_path), warnings=[])
        assert any("No image files" in e for e in errs)

    def test_images_without_captions_warns(self, tmp_path):
        # 3 images, 1 missing caption = 33% missing → warning, not error
        for i in range(3):
            (tmp_path / f"img{i}.png").write_bytes(b"\x89PNG")
        (tmp_path / "img0.txt").write_text("cat")
        (tmp_path / "img1.txt").write_text("dog")
        # img2.txt intentionally absent
        warnings: list[str] = []
        errs = _check_dataset_dir(str(tmp_path), require_captions=True, warnings=warnings)
        assert not errs
        assert any("caption" in w.lower() for w in warnings)

    def test_no_captions_majority_missing_is_error(self, tmp_path):
        # Create 3 images, 0 captions → 100% missing → error
        for i in range(3):
            (tmp_path / f"img{i}.png").write_bytes(b"\x89PNG")
        warnings: list[str] = []
        errs = _check_dataset_dir(str(tmp_path), require_captions=True, warnings=warnings)
        assert any("missing" in e for e in errs)

    def test_valid_dataset_with_captions(self, tmp_path):
        img = tmp_path / "cat.png"
        img.write_bytes(b"\x89PNG")
        (tmp_path / "cat.txt").write_text("a fluffy cat")
        warnings: list[str] = []
        errs = _check_dataset_dir(str(tmp_path), require_captions=True, warnings=warnings)
        assert not errs

    def test_none_input_returns_error(self):
        errs = _check_dataset_dir(None, warnings=[])
        assert errs

    def test_empty_subdir_warns(self, tmp_path):
        (tmp_path / "img.png").write_bytes(b"\x89PNG")
        (tmp_path / "empty_sub").mkdir()
        warnings: list[str] = []
        _check_dataset_dir(str(tmp_path), warnings=warnings)
        assert any("empty" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# _check_output_dir
# ---------------------------------------------------------------------------

class TestCheckOutputDir:
    def test_creates_missing_dir(self, tmp_path):
        new_dir = tmp_path / "new" / "output"
        errs = _check_output_dir(str(new_dir), [])
        assert not errs
        assert new_dir.exists()

    def test_existing_dir_is_ok(self, tmp_path):
        errs = _check_output_dir(str(tmp_path), [])
        assert not errs

    def test_file_path_is_error(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        errs = _check_output_dir(str(f), [])
        assert any("not a directory" in e for e in errs)


# ---------------------------------------------------------------------------
# _check_base_model
# ---------------------------------------------------------------------------

class TestCheckBaseModel:
    def test_empty_path_is_error(self):
        assert _check_base_model("")

    def test_missing_local_safetensors_is_error(self, tmp_path):
        errs = _check_base_model(str(tmp_path / "model.safetensors"))
        assert any("not found" in e for e in errs)

    def test_existing_local_file_is_ok(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"fake")
        errs = _check_base_model(str(f))
        assert not errs

    def test_hf_id_is_ok_when_allowed(self):
        errs = _check_base_model("stabilityai/stable-diffusion-xl-base-1.0", allow_hf_id=True)
        assert not errs

    def test_hf_id_blocked_when_not_allowed(self):
        errs = _check_base_model("stabilityai/stable-diffusion-xl-base-1.0", allow_hf_id=False)
        assert errs


# ---------------------------------------------------------------------------
# DatasetValidator.validate_kohya
# ---------------------------------------------------------------------------

class TestValidateKohya:
    def _req(self, tmp_path: Path) -> dict:
        ds = tmp_path / "dataset"
        ds.mkdir()
        (ds / "img.png").write_bytes(b"\x89PNG")
        (ds / "img.txt").write_text("a cat")
        mdl = tmp_path / "model.safetensors"
        mdl.write_bytes(b"fake")
        return {
            "job_name":        "test_lora",
            "base_model_path": str(mdl),
            "dataset_dir":     str(ds),
            "output_dir":      str(tmp_path / "out"),
            "resolution":      1024,
            "max_train_steps": 1000,
            "caption_extension": ".txt",
        }

    def test_valid_request_passes(self, tmp_path):
        r = validator.validate_kohya(self._req(tmp_path))
        assert r.ok

    def test_missing_dataset_fails(self, tmp_path):
        req = self._req(tmp_path)
        req["dataset_dir"] = str(tmp_path / "nonexistent")
        r = validator.validate_kohya(req)
        assert not r.ok

    def test_missing_model_fails(self, tmp_path):
        req = self._req(tmp_path)
        req["base_model_path"] = str(tmp_path / "missing.safetensors")
        r = validator.validate_kohya(req)
        assert not r.ok

    def test_non_multiple_of_64_resolution_warns(self, tmp_path):
        req = self._req(tmp_path)
        req["resolution"] = 900
        r = validator.validate_kohya(req)
        assert r.ok   # warning, not error
        assert any("64" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# DatasetValidator.validate_ed2
# ---------------------------------------------------------------------------

class TestValidateED2:
    def _req(self, tmp_path: Path) -> dict:
        ds = tmp_path / "dataset"
        ds.mkdir()
        (ds / "img.png").write_bytes(b"\x89PNG")
        (ds / "img.txt").write_text("a dog")
        mdl = tmp_path / "model.safetensors"
        mdl.write_bytes(b"fake")
        return {
            "job_name":        "test_ed2",
            "base_model_path": str(mdl),
            "dataset_dir":     str(ds),
            "output_dir":      str(tmp_path / "out"),
            "max_epochs":      10,
            "lr":              1.5e-6,
        }

    def test_valid_request_passes(self, tmp_path):
        r = validator.validate_ed2(self._req(tmp_path))
        assert r.ok

    def test_high_lr_warns(self, tmp_path):
        req = self._req(tmp_path)
        req["lr"] = 1e-3
        r = validator.validate_ed2(req)
        assert r.ok
        assert any("lr" in w.lower() or "learning rate" in w.lower() for w in r.warnings)

    def test_missing_vae_warns(self, tmp_path):
        req = self._req(tmp_path)
        req["vae_path"] = str(tmp_path / "nonexistent.safetensors")
        r = validator.validate_ed2(req)
        assert r.ok   # warning, not error
        assert any("vae" in w.lower() for w in r.warnings)


class TestValidateLLM:
    def _req(self, tmp_path: Path) -> dict:
        ds = tmp_path / "dataset.jsonl"
        ds.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Write Python."},
                        {"role": "assistant", "content": "Use functions."},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        model = tmp_path / "model"
        model.mkdir()
        return {
            "job_name": "llm_job",
            "base_model_path": str(model),
            "dataset_path": str(ds),
            "dataset_format": "messages",
            "output_dir": str(tmp_path / "out"),
            "method": "qlora",
            "max_seq_length": 1024,
        }

    def test_messages_jsonl_passes(self, tmp_path):
        result = validator.validate_llm(self._req(tmp_path))

        assert result.ok
        assert any("qlora" in warning.lower() for warning in result.warnings)

    def test_prompt_completion_passes(self, tmp_path):
        ds = tmp_path / "prompt_completion.jsonl"
        ds.write_text(json.dumps({"prompt": "p", "completion": "c"}) + "\n", encoding="utf-8")
        req = self._req(tmp_path)
        req["dataset_path"] = str(ds)
        req["dataset_format"] = "prompt_completion"

        result = validator.validate_llm(req)

        assert result.ok

    def test_invalid_shape_fails(self, tmp_path):
        ds = tmp_path / "bad.jsonl"
        ds.write_text(json.dumps({"foo": "bar"}) + "\n", encoding="utf-8")
        req = self._req(tmp_path)
        req["dataset_path"] = str(ds)

        result = validator.validate_llm(req)

        assert not result.ok
        assert any("supported LLM dataset shape" in error for error in result.errors)

    def test_full_method_warns_about_vram(self, tmp_path):
        req = self._req(tmp_path)
        req["method"] = "full"
        req["max_seq_length"] = 4096

        result = validator.validate_llm(req)

        assert result.ok
        assert any("Full fine-tuning" in warning for warning in result.warnings)
        assert any("max_seq_length=4096" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# DatasetValidator.validate_dataset_dir  (standalone)
# ---------------------------------------------------------------------------

class TestValidateDatasetDir:
    def test_valid_dir(self, tmp_path):
        (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff")
        r = validator.validate_dataset_dir(tmp_path)
        assert r.ok

    def test_empty_dir(self, tmp_path):
        r = validator.validate_dataset_dir(tmp_path)
        assert not r.ok

    def test_missing_dir(self, tmp_path):
        r = validator.validate_dataset_dir(tmp_path / "gone")
        assert not r.ok
