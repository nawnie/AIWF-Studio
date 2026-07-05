from __future__ import annotations

import json

from aiwf.services.studio_generation_packet import (
    build_studio_generation_packet,
    gradio_wan_workflow_json,
    model_selection_gate,
    validate_workflow_code_block_document,
    workflow_document_json_from_packet,
)


def test_wan_packet_preserves_high_low_lora_and_offload_sidecars() -> None:
    packet = build_studio_generation_packet(
        {
            "mode": "video",
            "prompt": "a fox turns toward camera",
            "modelId": "wan-route",
            "engineId": "wan",
            "width": 832,
            "height": 480,
            "frames": 81,
            "runtimeMode": "native_high_low",
            "highNoiseModelId": "wan-high-q4.gguf",
            "lowNoiseModelId": "wan-low-q4.gguf",
            "vaeId": "wan-2.1-vae.safetensors",
            "textEncoderPath": "umt5-xxl-q8_0.gguf",
            "highNoiseLoraId": "detail-high.safetensors",
            "lowNoiseLoraId": "motion-low.safetensors",
            "offload": "streamed",
        },
        model={"id": "wan-route", "name": "Wan high/low Q4", "engineId": "wan", "status": "metadata-only"},
    )
    assert packet["family"] == "wan"
    assert packet["route"] == "wan-video"
    assert packet["video"]["wan"]["runtimeMode"] == "native_high_low"
    assert packet["sidecars"]["wanModelPack"]["highNoiseModelId"] == "wan-high-q4.gguf"
    assert packet["sidecars"]["loraStack"]["entries"][1]["id"] == "motion-low.safetensors"
    assert packet["sidecars"]["offloadPlan"]["mode"] == "streamed"


def test_workflow_document_validation_blocks_broken_runtime_model() -> None:
    packet = build_studio_generation_packet(
        {"mode": "image", "modelId": "bad.safetensors", "prompt": "test"},
        model={"id": "bad.safetensors", "name": "bad", "status": "broken-runtime", "reason": "schema mismatch"},
    )
    document = json.loads(workflow_document_json_from_packet(packet))
    result = validate_workflow_code_block_document(document)
    assert not result["valid"]
    assert "schema mismatch" in "; ".join(result["errors"])


def test_gradio_wan_workflow_json_uses_linear_code_block_schema() -> None:
    raw = gradio_wan_workflow_json(
        prompt="camera pan across neon water",
        modelId="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        runtimeMode="fast_5b",
        offload="balanced",
        frames=21,
        fps=16,
    )
    document = json.loads(raw)
    assert document["schema"] == "aiwf.workflow-code-blocks.v1"
    assert document["routing"]["mode"] == "linear-code-blocks"
    packet = document["blocks"][0]["payload"]["packet"]
    assert packet["family"] == "wan"
    assert packet["video"]["wan"]["runtimeMode"] == "fast_5b"


def test_model_selection_gate_distinguishes_warning_from_block() -> None:
    assert model_selection_gate("metadata-only")["normalSelectable"] is True
    assert model_selection_gate("metadata-only")["requiresWarning"] is True
    assert model_selection_gate("unsupported-no-route")["normalSelectable"] is False


def test_pro_generate_payload_accepts_wan_high_low_fields() -> None:
    from aiwf.web.pro_api import ProGeneratePayload, _canonical_wan_runtime_mode

    payload = ProGeneratePayload.model_validate(
        {
            "mode": "video",
            "prompt": "cinematic catwalk",
            "wanRuntimeMode": "native_high_low",
            "highNoiseModelId": "wan-high-q4.gguf",
            "lowNoiseModelId": "wan-low-q4.gguf",
            "vaeId": "wan2.1_vae.safetensors",
            "textEncoderPath": "umt5-xxl-q4.gguf",
            "highNoiseLoraId": "motion-high.safetensors",
            "highNoiseLoraScale": 0.7,
            "lowNoiseLoraId": "detail-low.safetensors",
            "lowNoiseLoraScale": 0.45,
            "wanOffload": "model",
            "wanSampler": "heun",
            "wanSigmaType": "beta",
            "wanFlowShift": 6.0,
        }
    )

    assert payload.wan_runtime_mode == "native_high_low"
    assert payload.high_noise_model_id == "wan-high-q4.gguf"
    assert payload.low_noise_lora_scale == 0.45
    assert payload.wan_offload == "model"
    assert _canonical_wan_runtime_mode("high_low") == "native_high_low"
