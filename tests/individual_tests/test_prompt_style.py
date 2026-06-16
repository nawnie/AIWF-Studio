from __future__ import annotations

from aiwf.core.domain.prompt_style import PromptStyle, apply_prompt_style


def test_apply_style_with_prompt_placeholder():
    style = PromptStyle(name="x", prompt="oil painting of {prompt}", negative_prompt="low quality")
    positive, negative = apply_prompt_style(style, "a castle", "blurry")
    assert positive == "oil painting of a castle"
    assert negative == "blurry, low quality"


def test_apply_style_appends_enhancers_after_user_prompt():
    style = PromptStyle(name="x", prompt="masterpiece", negative_prompt="ugly")
    positive, negative = apply_prompt_style(style, "cat", "noise")
    assert positive == "cat, masterpiece"
    assert negative == "noise, ugly"


def test_apply_style_without_style_returns_user_text():
    positive, negative = apply_prompt_style(None, "dog", "bad")
    assert positive == "dog"
    assert negative == "bad"