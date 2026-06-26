from aiwf.second_gui import (
    _memory_label,
    _normalize_models,
    _normalize_sampler,
    _normalize_samplers,
    _safe_percent,
)


def test_normalize_sampler_common_labels():
    assert _normalize_sampler("Euler a") == "euler_a"
    assert _normalize_sampler("DPM++ 2M Karras") == "dpmpp_2m"
    assert _normalize_sampler("unipc") == "unipc"


def test_normalize_models_accepts_native_and_sdapi_shapes():
    models = _normalize_models(
        [
            {"id": "local-sdxl", "title": "Local SDXL", "path": "F:/models/sdxl.safetensors"},
            {"model_name": "A1111 Model", "filename": "F:/models/a1111.safetensors", "sha256": "abc"},
        ]
    )
    assert models[0]["id"] == "local-sdxl"
    assert models[0]["title"] == "Local SDXL"
    assert models[1]["id"] == "A1111 Model"
    assert models[1]["hash"] == "abc"


def test_normalize_samplers_prefers_id_but_keeps_label():
    samplers = _normalize_samplers([{"id": "euler_a", "label": "Euler a"}, {"name": "DDIM"}])
    assert samplers == [
        {"id": "euler_a", "title": "Euler a", "raw": {"id": "euler_a", "label": "Euler a"}},
        {"id": "ddim", "title": "DDIM", "raw": {"name": "DDIM"}},
    ]


def test_memory_label_formats_used_over_total():
    gib = 1024**3
    assert _memory_label(8 * gib, 16 * gib) == "8.0 / 16.0 GB"
    assert _memory_label(0, 0) == "WIP"


def test_safe_percent_handles_fraction_and_bounds():
    assert _safe_percent(0.42, fraction=True) == 42
    assert _safe_percent(120) == 100
    assert _safe_percent(-5) == 0
