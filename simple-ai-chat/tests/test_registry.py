from __future__ import annotations

import pytest

from simple_ai_chat.registry import Registry, RegistryError, Residency

from .conftest import CONFIG_DIR, spec


def test_shipped_config_loads_and_fits_budget():
    registry = Registry.from_yaml(CONFIG_DIR / "models.yaml", CONFIG_DIR / "hardware.yaml")

    assert registry.hardware.vram_budget_gib == pytest.approx(14.5)
    assert registry.best_for("chat").id == "bonsai2-27b"
    assert registry.get("bonsai2-27b").residency is Residency.PINNED
    for model in registry.enabled():
        assert model.vram_gib() <= registry.hardware.vram_budget_gib, model.id
    for capability in ("asr", "tts", "image_gen", "image_edit", "simulate", "embed", "rerank"):
        assert registry.best_for(capability) is not None, capability


def test_shipped_brain_estimate_matches_research():
    registry = Registry.from_yaml(CONFIG_DIR / "models.yaml", CONFIG_DIR / "hardware.yaml")
    brain = registry.get("bonsai2-27b")
    # 5.54 weights + 0.59 mmproj + 2.0 KV (32K f16 @ 64 KiB/token) + 1.2 overhead
    assert brain.kv_gib() == pytest.approx(2.0)
    assert brain.vram_gib() == pytest.approx(9.33)


def test_kv_and_vram_math():
    model = spec("m", weights_gib=5.0, mmproj_gib=0.5, kv_kib_per_token_f16=64, ctx=32768, overhead_gib=1.0)
    assert model.kv_gib() == pytest.approx(2.0)
    assert model.kv_gib(kv_type="q8_0") == pytest.approx(2.0 * 8.5 / 16)
    assert model.kv_gib(ctx=8192, kv_type="q4_0") == pytest.approx(0.5 * 4.5 / 16)
    assert model.vram_gib() == pytest.approx(8.5)

    no_mmproj = spec("n", weights_gib=5.0, mmproj_gib=0.5, mmproj_on_gpu=False)
    assert no_mmproj.vram_gib() == pytest.approx(5.0)
    assert spec("cpu", residency="cpu", weights_gib=3.0).vram_gib() == 0.0


def test_providers_sorted_by_rank_and_skip_disabled(small_registry):
    chat = [m.id for m in small_registry.providers("chat")]
    assert chat == ["brain", "router"]
    assert small_registry.best_for("web_simulate") is None
    assert small_registry.best_for("nonexistent") is None


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"id": "x", "backend": "nope", "capabilities": ["chat"]}, "unknown backend"),
        ({"id": "x", "backend": "llama", "capabilities": []}, "capability"),
        ({"id": "x", "backend": "llama", "capabilities": ["chat"], "kv_type": "q3"}, "kv_type"),
        ({"id": "x", "backend": "llama", "capabilities": ["chat"], "residency": "sometimes"}, "residency"),
        ({"backend": "llama", "capabilities": ["chat"]}, "id"),
    ],
)
def test_invalid_model_entries_rejected(bad, message):
    with pytest.raises(RegistryError, match=message):
        Registry.from_dicts({"models": [bad]})


def test_duplicate_ids_rejected():
    with pytest.raises(RegistryError, match="duplicate"):
        Registry([spec("a"), spec("a")])


def test_pinned_models_must_fit_budget(hardware):
    with pytest.raises(RegistryError, match="pinned"):
        Registry([spec("a", residency="pinned", weights_gib=10.0), spec("b", residency="pinned", weights_gib=5.0)],
                 hardware=hardware)
