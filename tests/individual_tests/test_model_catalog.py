from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.models import Checkpoint, LoraInfo
from aiwf.infrastructure.safetensors_metadata import suggest_lora_keywords
from aiwf.services.model_catalog import ModelCatalogService


@pytest.fixture
def catalog(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path)
    settings = UserSettings(
        lora_aliases={"clearskin": "ClearSkin_v2"},
        lora_defaults={"ClearSkin_v2": 0.8},
        lora_keywords={"ClearSkin_v2": "clear skin, smooth skin"},
    )
    generation = MagicMock()
    generation.list_loras.return_value = [
        LoraInfo(
            id="ClearSkin_v2",
            title="ClearSkin_v2",
            filename="ClearSkin_v2.safetensors",
            path=str(tmp_path / "ClearSkin_v2.safetensors"),
            architecture="sdxl",
            recommended_subdir="Loras/SDXL",
        ),
        LoraInfo(
            id="DetailTweaker",
            title="DetailTweaker",
            filename="DetailTweaker.safetensors",
            path=str(tmp_path / "DetailTweaker.safetensors"),
        ),
    ]
    generation.list_checkpoints.return_value = [
        Checkpoint(
            id="test_model",
            title="test_model",
            filename="test_model.safetensors",
            path=str(tmp_path / "test_model.safetensors"),
            hash="abc123",
        )
    ]
    return ModelCatalogService(generation, flags, settings)


def test_expand_lora_keyword_with_alias_and_triggers(catalog):
    expanded = catalog.expand_prompt_keywords("portrait *lora:clearskin smiling")
    assert expanded == "portrait clear skin, smooth skin <lora:ClearSkin_v2:0.8> smiling"


def test_expand_lora_keyword_by_id(catalog):
    expanded = catalog.expand_prompt_keywords("*lora:DetailTweaker")
    assert expanded == "<lora:DetailTweaker:1>"


def test_expand_unknown_keyword_left_unchanged(catalog):
    prompt = "photo *lora:missing"
    assert catalog.expand_prompt_keywords(prompt) == prompt


def test_set_lora_config_replaces_alias(catalog):
    catalog.set_lora_config("DetailTweaker", alias="detail", strength=0.65, keywords="sharp details")
    assert catalog.settings.lora_aliases["detail"] == "DetailTweaker"
    assert catalog.settings.lora_defaults["DetailTweaker"] == 0.65
    assert catalog.settings.lora_keywords["DetailTweaker"] == "sharp details"


def test_suggest_lora_keywords_from_tag_frequency():
    metadata = {
        "ss_tag_frequency": '{"1girl": 120, "solo": 80, "portrait": 40}',
    }
    assert suggest_lora_keywords(metadata) == "1girl, solo, portrait"


def test_suggest_lora_keywords_flattens_nested_tag_frequency():
    metadata = {
        "ss_tag_frequency": '{"dataset_a": {"1girl": 120, "solo": 80}, "dataset_b": {"1girl": 30, "portrait": 40}}',
    }
    assert suggest_lora_keywords(metadata) == "1girl, solo, portrait"


def test_checkpoint_details_include_usage_help(catalog):
    checkpoint = catalog.find_checkpoint("test_model")
    text = catalog.checkpoint_details(checkpoint)
    assert "test_model" in text
    assert "How to use" in text


def test_lora_details_include_compatibility_metadata(catalog):
    lora = catalog.find_lora("ClearSkin_v2")
    text = catalog.lora_details(lora)
    assert "**Architecture:** SDXL" in text
    assert "**Recommended folder:** `Loras/SDXL`" in text
