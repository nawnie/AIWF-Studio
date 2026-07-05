from __future__ import annotations

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.services.model_family_support import (
    build_model_family_matrix,
    detect_precision_from_name,
    precision_bucket,
    static_model_family_support,
)


def test_precision_detector_covers_current_quant_spellings() -> None:
    examples = {
        "flux2_klein_4b_int8.safetensors": "INT8",
        "Wan2.2_I2V_A14B_high_Q4_K_M.gguf": "Q4_K_M",
        "Wan_low_q5_k_s.gguf": "Q5_K_S",
        "fluxFusionV2GGUFQ4KM.gguf": "Q4_K_M",
        "ltx-2.3-22b-dev-nvfp4.safetensors": "NVFP4",
        "flux-dev-bnb-nf4.safetensors": "NF4",
        "model-f8_e4m3fn.safetensors": "FP8",
        "t5xxl_fp16.safetensors": "FP16",
        "clip_bf16.safetensors": "BF16",
    }
    for filename, expected in examples.items():
        assert detect_precision_from_name(filename) == expected


def test_precision_bucket_groups_compatible_pairs() -> None:
    assert precision_bucket("Q4_K_M") == "q4"
    assert precision_bucket("Q4_0") == "q4"
    assert precision_bucket("NF4") == "4bit"
    assert precision_bucket("FP8") == "8bit"


def test_static_family_matrix_lists_core_families_and_gaps() -> None:
    families = {family["id"]: family for family in static_model_family_support()}
    for family_id in ("flux", "flux2_klein", "wan", "ltx", "sana", "sana_video", "qwen_image", "z_image", "sdxl", "sd35"):
        assert family_id in families
    assert any(item["name"] == "INT8" and item["status"] == "missing" for item in families["flux2_klein"]["precisions"])
    assert any("T2V" in blocker for blocker in families["wan"]["blockers"])
    assert any(item["name"].startswith("NVFP4") and item["status"] == "blocked" for item in families["ltx"]["precisions"])


def test_build_model_family_matrix_is_import_light(tmp_path) -> None:
    flags = RuntimeFlags(data_dir=tmp_path)
    payload = build_model_family_matrix(flags, UserSettings())
    assert payload["schema"] == "aiwf.model-family-support.v1"
    assert payload["families"]
    assert "precisionVocabulary" in payload
