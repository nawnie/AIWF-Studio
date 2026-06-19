from __future__ import annotations

from aiwf.services.wan_models import (
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
