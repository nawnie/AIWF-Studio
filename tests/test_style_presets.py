from aiwf.core.config.settings import UserSettings
from aiwf.core.domain.prompt_style import PromptStyle, apply_prompt_style
from aiwf.core.domain.style_presets import (
    DEFAULT_PROMPT_STYLES,
    ensure_default_prompt_styles,
    get_builtin_style,
    is_builtin_style,
    style_preview_text,
)


def test_quality_standard_leads_with_photo_prefix():
    style = next(item for item in DEFAULT_PROMPT_STYLES if item.name == "Quality — Standard")
    positive, negative = apply_prompt_style(style, "a red fox", "noise")
    assert positive.startswith("a high quality photo of a red fox,")
    assert "masterpiece" in positive
    assert negative.startswith("noise,")
    assert "worst quality" in negative


def test_detail_general_puts_enhancers_after_user_prompt():
    style = next(item for item in DEFAULT_PROMPT_STYLES if item.name == "Detail — General")
    positive, _ = apply_prompt_style(style, "mountain lake", "")
    assert positive.startswith("mountain lake,")
    assert "intricate details" in positive


def test_ensure_default_prompt_styles_adds_missing_only():
    settings = UserSettings(prompt_styles=[])
    assert ensure_default_prompt_styles(settings) is True
    assert len(settings.prompt_styles) == len(DEFAULT_PROMPT_STYLES)

    assert ensure_default_prompt_styles(settings) is False


def test_ensure_default_prompt_styles_preserves_user_edits():
    stale = PromptStyle(
        name="Quality — Standard",
        prompt="my custom template {prompt}",
        negative_prompt="my custom negative",
    )
    custom = PromptStyle(name="My custom look", prompt="{prompt}, custom")
    settings = UserSettings(prompt_styles=[stale, custom])

    ensure_default_prompt_styles(settings)

    updated = next(s for s in settings.prompt_styles if s.name == "Quality — Standard")
    assert updated.prompt == "my custom template {prompt}"
    assert updated.negative_prompt == "my custom negative"
    assert len(settings.prompt_styles) == len(DEFAULT_PROMPT_STYLES) + 1
    assert any(style.name == "My custom look" for style in settings.prompt_styles)


def test_builtin_helpers():
    assert is_builtin_style("Quality — Standard")
    assert not is_builtin_style("My custom look")
    preset = get_builtin_style("Quality — Standard")
    assert preset is not None
    assert "{prompt}" in preset.prompt


def test_style_preview_text_includes_examples():
    style = DEFAULT_PROMPT_STYLES[0]
    preview = style_preview_text(style)
    assert "Positive example" in preview
    assert "woman in a garden" in preview