from __future__ import annotations

from aiwf.services.wan_models import (
    wan_model_pair_compatibility,
    wan_model_quant_family,
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
