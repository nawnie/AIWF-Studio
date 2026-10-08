from __future__ import annotations

from types import SimpleNamespace

import aiwf.services.pipeline_readiness as pipeline_readiness
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


def test_family_matrix_generation_modes_match_runtime_preflight_contracts() -> None:
    families = {family["id"]: family for family in static_model_family_support()}
    wan_routes = {route["id"]: route for route in families["wan"]["routes"]}
    ltx_routes = {route["id"]: route for route in families["ltx"]["routes"]}

    assert wan_routes["wan-fast-5b"]["kind"] == "text/image-to-video (TI2V)"
    assert "A14B pairs require a source image" in families["wan"]["blockers"][0]
    assert ltx_routes["ltx-2b-diffusers"]["kind"] == "text-to-video"


def test_krea2_matrix_separates_folder_route_from_blocked_split_assets() -> None:
    krea = next(item for item in static_model_family_support() if item["id"] == "krea2")

    assert krea["status"] == "partial-supported"
    route_statuses = {route["id"]: route["status"] for route in krea["routes"]}
    assert route_statuses["krea2"] == "supported-when-folder-installed"
    assert route_statuses["krea2-split"] == "blocked-cleanly"
    assert any("no real Krea 2 generation smoke" in blocker for blocker in krea["blockers"])


def test_build_model_family_matrix_is_import_light(tmp_path) -> None:
    flags = RuntimeFlags(data_dir=tmp_path)
    payload = build_model_family_matrix(flags, UserSettings())
    assert payload["schema"] == "aiwf.model-family-support.v1"
    assert payload["families"]
    assert "precisionVocabulary" in payload


def test_matrix_attributes_image_readiness_by_architecture_and_video_route(tmp_path, monkeypatch) -> None:
    records = [
        SimpleNamespace(family="image", metadata={"architecture": "sd15"}, route="diffusers", status="working", quantization="", path="sd15.safetensors", reason="", suggested_action=""),
        SimpleNamespace(family="image", metadata={"architecture": "flux"}, route="flux-dev", status="metadata-only", quantization="", path="flux-dev.safetensors", reason="", suggested_action=""),
        SimpleNamespace(family="image", metadata={"architecture": "qwen_image"}, route="qwen-image", status="working", quantization="", path="qwen-image.safetensors", reason="", suggested_action=""),
        SimpleNamespace(family="image", metadata={"architecture": "flux2"}, route="flux2-generic", status="unsupported-no-route", quantization="", path="generic-flux2.safetensors", reason="", suggested_action=""),
        SimpleNamespace(family="image", metadata={"architecture": "flux2_klein"}, route="flux2-klein", status="working", quantization="", path="flux2-klein.safetensors", reason="", suggested_action=""),
        SimpleNamespace(family="video", metadata={"architecture": "sana"}, route="sana-video", status="working", quantization="", path="sana-video", reason="", suggested_action=""),
    ]
    monkeypatch.setattr(pipeline_readiness, "collect_pipeline_readiness", lambda *args, **kwargs: records)

    payload = build_model_family_matrix(RuntimeFlags(data_dir=tmp_path), UserSettings())
    families = {family["id"]: family for family in payload["families"]}

    assert families["sd15"]["localReadiness"] == {"working": 1}
    assert families["flux"]["localReadiness"] == {"metadata-only": 1}
    assert families["qwen_image"]["localReadiness"] == {"working": 1}
    assert families["flux2_klein"]["localReadiness"] == {"working": 1}
    assert payload["readiness"]["countsByFamily"]["flux2_generic"] == {"unsupported-no-route": 1}
    assert families["sana"]["localReadiness"] == {}
    assert families["sana_video"]["localReadiness"] == {"working": 1}


def test_matrix_blocker_limit_preserves_each_family(tmp_path, monkeypatch) -> None:
    records = [
        SimpleNamespace(
            family="image",
            metadata={"architecture": "flux"},
            route=f"flux-{index}",
            status="blocked-cleanly",
            quantization="",
            path=f"flux-{index}.safetensors",
            reason="Flux blocker",
            suggested_action="",
        )
        for index in range(81)
    ]
    records.append(
        SimpleNamespace(
            family="image",
            metadata={"architecture": "qwen_image"},
            route="qwen-image",
            status="blocked-cleanly",
            quantization="",
            path="qwen-image.safetensors",
            reason="Qwen Image blocker",
            suggested_action="",
        )
    )
    monkeypatch.setattr(pipeline_readiness, "collect_pipeline_readiness", lambda *args, **kwargs: records)

    payload = build_model_family_matrix(RuntimeFlags(data_dir=tmp_path), UserSettings())

    assert len(payload["blockedExamples"]) == 80
    assert {item["family"] for item in payload["blockedExamples"]} == {"flux", "qwen_image"}
    assert any(item["reason"] == "Qwen Image blocker" for item in payload["blockedExamples"])
