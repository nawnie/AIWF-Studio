from __future__ import annotations

from pathlib import Path

import pytest

from simple_ai_chat.registry import Hardware, ModelSpec, Registry

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def spec(model_id: str, **overrides) -> ModelSpec:
    data = {
        "id": model_id,
        "backend": "llama",
        "capabilities": ["chat"],
        "residency": "warm",
        "weights_gib": 1.0,
    }
    data.update(overrides)
    return ModelSpec.from_dict(data)


@pytest.fixture
def hardware() -> Hardware:
    # 16 - 1 - 0.5 = 14.5 GiB budget, same as the real card.
    return Hardware(gpu_total_gib=16.0, reserved_gib=1.0, safety_margin_gib=0.5)


@pytest.fixture
def small_registry(hardware: Hardware) -> Registry:
    return Registry(
        [
            spec("brain", capabilities=["chat", "vision", "tools"], residency="pinned", priority=100, rank=1,
                 weights_gib=6.0, mmproj_gib=0.5, kv_kib_per_token_f16=64, ctx=32768, overhead_gib=1.0),
            spec("router", capabilities=["route", "chat"], priority=40, rank=50, weights_gib=1.0),
            spec("asr", capabilities=["asr"], priority=60, weights_gib=1.5),
            spec("tts", backend="qwentts", capabilities=["tts"], priority=60, weights_gib=2.5),
            spec("pe", capabilities=["prompt_enhance"], residency="transient", priority=65, weights_gib=6.0),
            spec("pe-edit", capabilities=["prompt_enhance_edit"], residency="transient", priority=65, weights_gib=6.0),
            spec("image", backend="sdcpp", capabilities=["image_gen", "image_edit"], residency="transient",
                 exclusive=True, priority=70, weights_gib=9.0, overhead_gib=2.0),
            spec("sim", capabilities=["simulate"], residency="transient", priority=50, weights_gib=5.0),
            spec("embed", capabilities=["embed"], residency="cpu", priority=30),
            spec("guard", capabilities=["moderate"], residency="cpu", priority=30),
            spec("websim", capabilities=["web_simulate"], residency="transient", enabled=False),
        ],
        hardware=hardware,
        toggles={"prompt_enhancer": True, "speak_replies": False, "moderation": False},
    )
