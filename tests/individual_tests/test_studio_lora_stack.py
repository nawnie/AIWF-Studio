from __future__ import annotations

from types import SimpleNamespace

from aiwf.core.domain.models import LoraInfo
from aiwf.web.studio.handlers.prompts import apply_lora_stack_to_prompt, strip_lora_tags


class _Models:
    def __init__(self):
        self._loras = {
            "style": LoraInfo(id="style", title="Style", filename="style.safetensors", path="style.safetensors"),
            "detail": LoraInfo(id="detail", title="Detail", filename="detail.safetensors", path="detail.safetensors"),
        }

    def find_lora(self, lora_id):
        return self._loras.get(lora_id)

    def lora_keywords(self, lora_id):
        return {"style": "painted texture", "detail": "sharp detail"}.get(lora_id, "")


def _ctx():
    return SimpleNamespace(models=_Models())


def test_strip_lora_tags_cleans_prompt_commas():
    prompt = "portrait, <lora:old:0.7>, dramatic light"

    assert strip_lora_tags(prompt) == "portrait, dramatic light"


def test_apply_lora_stack_replaces_existing_tags_and_adds_keywords():
    result = apply_lora_stack_to_prompt(
        _ctx(),
        "portrait, <lora:old:0.7>",
        "style",
        0.8,
        "detail",
        0.5,
        None,
        1.0,
        "style",
        1.2,
        True,
    )

    assert result == "portrait, <lora:style:0.8>, painted texture, <lora:detail:0.5>, sharp detail"


def test_apply_lora_stack_can_skip_keywords():
    result = apply_lora_stack_to_prompt(
        _ctx(),
        "portrait",
        "style",
        1.0,
        None,
        1.0,
        False,
    )

    assert result == "portrait, <lora:style:1>"


def test_empty_lora_stack_only_removes_lora_tags():
    result = apply_lora_stack_to_prompt(_ctx(), "portrait, <lora:old:0.7>", None, 1.0, True)

    assert result == "portrait"
