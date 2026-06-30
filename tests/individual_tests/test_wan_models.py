from __future__ import annotations

from pathlib import Path

import pytest

from aiwf.services.wan_models import (
    wan_lora_info,
    wan_lora_stage_matches,
    wan_model_pair_compatibility,
    wan_model_quant_family,
    wan_selectable_loras,
    wan_selectable_transformers,
    wan_model_stage_role,
    wan_model_storage_family,
)


def test_wan_model_helpers_classify_common_high_low_names():
    assert wan_model_stage_role("wan2.2_i2v_high_noise_fp8.safetensors") == "high"
    assert wan_model_stage_role("wan2.2_i2v_low_noise_q4.gguf") == "low"
    assert wan_model_storage_family("model.gguf") == "gguf"
    assert wan_model_storage_family("model.safetensors") == "safetensors"
    assert wan_model_quant_family("wan_low_q4_k.gguf") == "q4"
    assert wan_model_quant_family("wan_high_fp8.safetensors") == "fp8"


def test_wan_model_pair_compatibility_blocks_mismatched_format():
    check = wan_model_pair_compatibility(
        "wan2.2_i2v_high_noise_fp8.safetensors",
        "wan2.2_i2v_low_noise_q4.gguf",
    )

    assert not check.ok
    assert any("different storage formats" in error for error in check.errors)


def test_wan_model_pair_compatibility_blocks_swapped_roles():
    check = wan_model_pair_compatibility(
        "wan2.2_i2v_low_noise_q4.gguf",
        "wan2.2_i2v_high_noise_q4.gguf",
    )

    assert not check.ok
    assert any("High noise selection looks like a low-noise model" in error for error in check.errors)
    assert any("Low noise selection looks like a high-noise model" in error for error in check.errors)


def test_wan_selectable_transformers_enforces_peer_storage_quant_and_role():
    candidates = [
        "wan_a14b_high_q4.gguf",
        "wan_a14b_low_q4.gguf",
        "wan_a14b_low_q5.gguf",
        "wan_a14b_low_q4.safetensors",
        "wan_a14b_high_q4_copy.gguf",
    ]

    assert wan_selectable_transformers(
        candidates,
        runtime_mode="native_high_low",
        want_role="low",
        peer_id="wan_a14b_high_q4.gguf",
    ) == ["wan_a14b_low_q4.gguf"]


def test_wan_selectable_loras_filters_by_runtime_size_class():
    candidates = [
        "motion_a14b_rank16.safetensors",
        "motion_5b_ti2v_rank16.safetensors",
        "style_rank16.safetensors",
    ]

    assert wan_selectable_loras(candidates, runtime_mode="native_high_low") == [
        "motion_a14b_rank16.safetensors",
        "style_rank16.safetensors",
    ]
    assert wan_selectable_loras(candidates, runtime_mode="fast_5b") == [
        "motion_5b_ti2v_rank16.safetensors",
        "style_rank16.safetensors",
    ]


def test_wan_lora_info_uses_safetensors_metadata_for_size_and_stage(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    lora = tmp_path / "motion_adapter.safetensors"
    safetensors.save_file(
        {"diffusion_model.blocks.39.self_attn.q.lora_down.weight": torch.ones(1, 1)},
        lora,
        metadata={"ss_base_model_version": "Wan2.2-I2V-A14B-HighNoise"},
    )

    info = wan_lora_info(lora)

    assert info.ok is True
    assert info.size_class == "14b"
    assert info.role == "high"
    assert wan_selectable_loras([str(lora)], runtime_mode="native_high_low", stage="high") == [str(lora)]
    assert wan_selectable_loras([str(lora)], runtime_mode="fast_5b") == []
    assert wan_lora_stage_matches(lora, stage="low") is False


def test_wan_lora_info_infers_5b_from_block_keys_when_name_is_ambiguous(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    lora = tmp_path / "turbo_adapter.safetensors"
    safetensors.save_file(
        {"diffusion_model.blocks.29.self_attn.q.lora_down.weight": torch.ones(1, 1)},
        lora,
    )

    info = wan_lora_info(lora)

    assert info.size_class == "5b"
    assert wan_selectable_loras([str(lora)], runtime_mode="fast_5b") == [str(lora)]
    assert wan_selectable_loras([str(lora)], runtime_mode="native_high_low") == []
