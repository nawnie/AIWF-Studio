from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.prompt_style import PromptStyle
from aiwf.services.prompt_processor import PromptProcessorService


@pytest.fixture
def processor(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path)
    settings = UserSettings(
        prompt_styles=[
            PromptStyle(name="Portrait", prompt="portrait photo of {prompt}", negative_prompt="blurry"),
        ]
    )
    models = MagicMock()
    models.expand_prompt_keywords.side_effect = lambda text: text
    return PromptProcessorService(flags, settings, models)


def test_list_and_read_prompt_files(processor: PromptProcessorService, tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "base.txt").write_text("line one\nline two\n", encoding="utf-8")

    files = processor.list_prompt_files()
    assert files == [("base.txt", "base.txt")]

    rng_seed = processor.read_prompt_file("base.txt", rng=__import__("random").Random(0))
    assert rng_seed in {"line one", "line two"}


def test_prepare_prompt_applies_style_and_wildcard(processor: PromptProcessorService, tmp_path: Path):
    wildcards = tmp_path / "wildcards"
    wildcards.mkdir()
    (wildcards / "mood.txt").write_text("serene\n", encoding="utf-8")

    prompt, negative = processor.prepare_prompt(
        "__mood__ woman",
        negative_text="",
        style_name="Portrait",
        seed=5,
    )
    assert prompt == "portrait photo of serene woman"
    assert negative == "blurry"


def test_prepare_prompt_merges_prompt_file(processor: PromptProcessorService, tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "scene.txt").write_text("sunset beach\n", encoding="utf-8")

    prompt, _negative = processor.prepare_prompt(
        "extra detail",
        use_prompt_file=True,
        prompt_file="scene.txt",
        seed=1,
    )
    assert prompt == "sunset beach extra detail"


def test_prepare_prompt_uses_style_override(processor: PromptProcessorService):
    prompt, negative = processor.prepare_prompt(
        "cat",
        negative_text="noise",
        style_name="Portrait",
        style_override=PromptStyle(
            name="Portrait",
            prompt="edited {prompt}",
            negative_prompt="edited negative",
        ),
    )
    assert prompt == "edited cat"
    assert negative == "noise, edited negative"


def test_reset_style_to_default(processor: PromptProcessorService, tmp_path: Path):
    processor.save_style(
        PromptStyle(name="Quality — Standard", prompt="custom", negative_prompt="custom neg"),
        ctx_save=None,
    )
    from aiwf.core.domain.style_presets import get_builtin_style

    preset = processor.reset_style_to_default("Quality — Standard", ctx_save=None)
    assert preset is not None
    saved = processor.find_style("Quality — Standard")
    assert saved is not None
    assert saved.prompt == get_builtin_style("Quality — Standard").prompt


def test_save_and_delete_style(processor: PromptProcessorService):
    saved = []
    processor.save_style(
        PromptStyle(name="Anime", prompt="anime style"),
        ctx_save=lambda: saved.append(True),
    )
    assert saved == [True]
    assert processor.find_style("Anime") is not None

    processor.delete_style("Anime", ctx_save=lambda: saved.append(True))
    assert processor.find_style("Anime") is None
    assert len(saved) == 2