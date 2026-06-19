"""Tests for kohya_config.py and ed2_config.py — pure config builders."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiwf.services.training.kohya_config import (
    build_kohya_toml,
    training_script_for_arch,
    write_kohya_toml,
    _render_toml,
    _toml_value,
)
from aiwf.services.training.ed2_config import (
    build_ed2_config,
    write_ed2_config,
)


# ---------------------------------------------------------------------------
# TOML serialiser primitives
# ---------------------------------------------------------------------------

class TestTomlValue:
    def test_bool_true(self):  assert _toml_value(True)  == "true"
    def test_bool_false(self): assert _toml_value(False) == "false"
    def test_int(self):        assert _toml_value(42)    == "42"
    def test_string(self):     assert _toml_value("hi")  == '"hi"'
    def test_string_escapes_backslash(self):
        assert _toml_value("C:\\path") == '"C:\\\\path"'
    def test_string_escapes_doublequote(self):
        assert _toml_value('say "hello"') == '"say \\"hello\\""'
    def test_float(self):
        v = _toml_value(1e-4)
        assert "1e" in v or "0.0001" in v
    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _toml_value([1, 2, 3])


class TestRenderToml:
    def test_produces_section_headers(self):
        out = _render_toml({"my_section": {"a": 1, "b": True}})
        assert "[my_section]" in out
        assert "a = 1" in out
        assert "b = true" in out

    def test_multiple_sections(self):
        out = _render_toml({
            "s1": {"x": "hello"},
            "s2": {"y": 42},
        })
        assert "[s1]" in out and "[s2]" in out


# ---------------------------------------------------------------------------
# build_kohya_toml
# ---------------------------------------------------------------------------

class TestBuildKohyaToml:
    def _base_req(self) -> dict:
        return {
            "job_name":        "test_lora",
            "base_model_path": "models/base.safetensors",
            "base_arch":       "sdxl",
            "dataset_dir":     "datasets/cats",
            "output_dir":      "outputs/kohya",
            "resolution":      1024,
            "max_train_steps": 1500,
            "batch_size":      1,
            "learning_rate":   1e-4,
            "lr_scheduler":    "cosine_with_restarts",
            "lr_warmup_steps": 100,
            "optimizer":       "AdamW8bit",
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "clip_grad_norm":  1.0,
            "network_dim":     32,
            "network_alpha":   16.0,
            "network_module":  "networks.lora",
            "save_every_n_steps": 500,
            "save_last_n_steps":  5,
            "seed":            42,
            "caption_extension": ".txt",
        }

    def test_returns_string(self):
        toml = build_kohya_toml(self._base_req())
        assert isinstance(toml, str) and len(toml) > 100

    def test_contains_required_sections(self):
        toml = build_kohya_toml(self._base_req())
        assert "[model_arguments]" in toml
        assert "[dataset_arguments]" in toml
        assert "[training_arguments]" in toml
        assert "[network_arguments]" in toml

    def test_base_model_path_present(self):
        toml = build_kohya_toml(self._base_req())
        assert "models/base.safetensors" in toml

    def test_dataset_dir_present(self):
        toml = build_kohya_toml(self._base_req())
        assert "datasets/cats" in toml

    def test_network_dim_present(self):
        toml = build_kohya_toml(self._base_req())
        assert "network_dim = 32" in toml

    def test_mixed_precision_bf16(self):
        toml = build_kohya_toml(self._base_req())
        assert "bf16" in toml

    def test_output_name_defaults_to_job_name(self):
        req = self._base_req()
        req.pop("output_name", None)   # ensure it's absent
        toml = build_kohya_toml(req)
        assert "test_lora" in toml

    def test_backslash_paths_escaped(self):
        req = self._base_req()
        req["dataset_dir"] = "C:\\Users\\shawn\\training"
        toml = build_kohya_toml(req)
        assert "\\\\" in toml   # backslash must be escaped in TOML

    def test_accepts_dict(self):
        toml = build_kohya_toml(self._base_req())
        assert toml

    def test_accepts_object_with_attributes(self):
        class Obj:
            job_name = "obj_lora"
            base_model_path = "model.safetensors"
            base_arch = "sdxl"
            dataset_dir = "ds"
            output_dir = "out"
            output_name = ""
            resolution = 512
            max_train_steps = 500
            batch_size = 1
            learning_rate = 1e-4
            unet_lr = None
            text_encoder_lr = None
            lr_scheduler = "constant"
            lr_warmup_steps = 0
            optimizer = "AdamW8bit"
            mixed_precision = "bf16"
            gradient_checkpointing = True
            clip_grad_norm = 1.0
            network_dim = 16
            network_alpha = 8.0
            network_module = "networks.lora"
            save_every_n_steps = 500
            save_last_n_steps = 3
            seed = 42
            caption_extension = ".txt"

        toml = build_kohya_toml(Obj())
        assert "[model_arguments]" in toml


# ---------------------------------------------------------------------------
# write_kohya_toml
# ---------------------------------------------------------------------------

class TestWriteKohyaToml:
    def test_creates_file(self, tmp_path):
        req = {
            "job_name": "x", "base_model_path": "m", "base_arch": "sdxl",
            "dataset_dir": "d", "output_dir": str(tmp_path / "out"),
        }
        dest = tmp_path / "sub" / "config.toml"
        result = write_kohya_toml(req, dest)
        assert result == dest
        assert dest.exists()
        assert "[model_arguments]" in dest.read_text()


# ---------------------------------------------------------------------------
# training_script_for_arch
# ---------------------------------------------------------------------------

class TestTrainingScriptForArch:
    def test_sd1(self):  assert "train_network.py" in training_script_for_arch("sd1")
    def test_sdxl(self): assert "sdxl_train_network.py" in training_script_for_arch("sdxl")
    def test_flux(self): assert "flux_train_network.py" in training_script_for_arch("flux")
    def test_unknown_falls_back_to_sdxl(self):
        assert "sdxl" in training_script_for_arch("unknown")


# ---------------------------------------------------------------------------
# build_ed2_config
# ---------------------------------------------------------------------------

class TestBuildED2Config:
    def _base_req(self) -> dict:
        return {
            "job_name":        "test_ed2",
            "base_model_path": "models/sd15.safetensors",
            "dataset_dir":     "datasets/dogs",
            "output_dir":      "outputs/ed2",
            "resolution":      512,
            "max_epochs":      20,
            "batch_size":      4,
            "lr":              1.5e-6,
            "lr_scheduler":    "constant",
            "lr_warmup_steps": 0,
            "optimizer":       "adamw",
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "clip_skip":       2,
            "seed":            42,
            "save_every_n_epochs": 1,
            "save_last_n_epochs":  3,
            "ckpt_type":       "safetensors",
        }

    def test_returns_dict(self):
        cfg = build_ed2_config(self._base_req())
        assert isinstance(cfg, dict)

    def test_required_keys_present(self):
        cfg = build_ed2_config(self._base_req())
        for key in ("resume_ckpt", "data_root", "save_ckpt_dir", "max_epochs", "lr"):
            assert key in cfg, f"Missing key: {key}"

    def test_model_path_mapped(self):
        cfg = build_ed2_config(self._base_req())
        assert cfg["resume_ckpt"] == "models/sd15.safetensors"

    def test_project_name_is_job_name(self):
        cfg = build_ed2_config(self._base_req())
        assert cfg["project_name"] == "test_ed2"

    def test_vae_absent_when_empty(self):
        cfg = build_ed2_config(self._base_req())
        assert "vae" not in cfg

    def test_vae_present_when_set(self):
        req = {**self._base_req(), "vae_path": "models/vae.safetensors"}
        cfg = build_ed2_config(req)
        assert cfg["vae"] == "models/vae.safetensors"

    def test_sample_prompts_absent_when_not_set(self):
        cfg = build_ed2_config(self._base_req())
        assert "sample_prompts" not in cfg
        assert cfg["sample_steps"] == 0

    def test_sample_prompts_present_when_set(self):
        req = {**self._base_req(), "sample_steps": 100, "sample_prompts": "sample_prompts.txt"}
        cfg = build_ed2_config(req)
        assert cfg["sample_prompts"] == "sample_prompts.txt"
        assert cfg["sample_steps"] == 100

    def test_is_json_serialisable(self):
        cfg = build_ed2_config(self._base_req())
        dumped = json.dumps(cfg)
        assert dumped


# ---------------------------------------------------------------------------
# write_ed2_config
# ---------------------------------------------------------------------------

class TestWriteED2Config:
    def test_creates_valid_json(self, tmp_path):
        req = {
            "job_name": "x", "base_model_path": "m",
            "dataset_dir": "d", "output_dir": str(tmp_path / "out"),
        }
        dest = tmp_path / "train.json"
        result = write_ed2_config(req, dest)
        assert result == dest
        data = json.loads(dest.read_text())
        assert data["project_name"] == "x"
