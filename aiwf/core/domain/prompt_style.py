from __future__ import annotations

from pydantic import BaseModel, Field


class PromptStyle(BaseModel):
    """Named prompt template.

    Use `{prompt}` where the user's text belongs. Place it after a lead-in
    (``a high quality photo of {prompt}, best quality``) or before enhancers
    (``{prompt}, best quality, highly detailed``).
    """

    name: str
    prompt: str = ""
    negative_prompt: str = ""


def apply_prompt_style(
    style: PromptStyle | None,
    prompt: str,
    negative_prompt: str = "",
) -> tuple[str, str]:
    """Merge a style template with the user's prompt and negative prompt."""
    if style is None:
        return prompt.strip(), negative_prompt.strip()

    positive = _merge_style_text(style.prompt, prompt)
    negative = _merge_style_text(style.negative_prompt, negative_prompt)
    return positive.strip(), negative.strip()


def _merge_style_text(template: str, user_text: str) -> str:
    template = (template or "").strip()
    user_text = (user_text or "").strip()
    if not template:
        return user_text
    if "{prompt}" in template:
        return template.replace("{prompt}", user_text).strip()
    if user_text:
        return f"{user_text}, {template}".strip()
    return template