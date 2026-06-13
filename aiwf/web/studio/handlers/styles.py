from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext


def load_style_editor(ctx: AppContext, style_name: str | None) -> tuple:
    from aiwf.core.domain.style_presets import is_builtin_style, style_preview_text

    if not style_name:
        return "", "", "", gr.update(value="", visible=False), gr.update(interactive=False)

    style = ctx.prompts.find_style(style_name)
    if style is None:
        return "", "", "", gr.update(value="", visible=False), gr.update(interactive=False)

    return (
        style.name,
        style.prompt,
        style.negative_prompt,
        gr.update(value=style_preview_text(style), visible=True),
        gr.update(interactive=is_builtin_style(style_name)),
    )


def refresh_style_preview(template_prompt: str, template_negative: str):
    from aiwf.core.domain.prompt_style import PromptStyle
    from aiwf.core.domain.style_presets import style_preview_text

    if not (template_prompt or "").strip() and not (template_negative or "").strip():
        return gr.update(value="", visible=False)
    preview_style = PromptStyle(name="", prompt=template_prompt or "", negative_prompt=template_negative or "")
    return gr.update(value=style_preview_text(preview_style), visible=True)


def apply_style_to_prompt(
    template_prompt: str,
    template_negative: str,
    prompt_text: str,
    negative_text: str,
) -> tuple[str, str]:
    from aiwf.core.domain.prompt_style import PromptStyle, apply_prompt_style

    if not (template_prompt or "").strip() and not (template_negative or "").strip():
        raise gr.Error("Select or edit a style preset first.")
    style = PromptStyle(name="", prompt=template_prompt or "", negative_prompt=template_negative or "")
    return apply_prompt_style(style, prompt_text, negative_text)


def save_prompt_style(
    ctx: AppContext,
    name: str,
    template_prompt: str,
    template_negative: str,
    selected_name: str | None,
) -> tuple:
    from aiwf.core.domain.prompt_style import PromptStyle

    clean = (name or "").strip()
    if not clean:
        raise gr.Error("Enter a preset name.")
    if not (template_prompt or "").strip() and not (template_negative or "").strip():
        raise gr.Error("Enter at least one style template (positive or negative).")
    if selected_name and selected_name != clean:
        ctx.prompts.delete_style(selected_name, ctx_save=None)
    ctx.prompts.save_style(
        PromptStyle(name=clean, prompt=template_prompt or "", negative_prompt=template_negative or ""),
        ctx_save=ctx.save_settings,
    )
    choices = ctx.prompts.style_choices()
    return gr.update(choices=choices, value=clean), clean


def reset_prompt_style(ctx: AppContext, style_name: str | None) -> tuple:
    from aiwf.core.domain.style_presets import is_builtin_style, style_preview_text

    if not style_name:
        raise gr.Error("Select a built-in preset to reset.")
    if not is_builtin_style(style_name):
        raise gr.Error("Only built-in presets can be reset to default.")
    preset = ctx.prompts.reset_style_to_default(style_name, ctx_save=ctx.save_settings)
    if preset is None:
        raise gr.Error("Preset not found.")
    return preset.prompt, preset.negative_prompt, gr.update(value=style_preview_text(preset), visible=True)


def delete_prompt_style(ctx: AppContext, name: str | None) -> tuple:
    if not name:
        raise gr.Error("Select a preset to delete.")
    ctx.prompts.delete_style(name, ctx_save=ctx.save_settings)
    return (
        gr.update(choices=ctx.prompts.style_choices(), value=None),
        "",
        "",
        "",
        gr.update(value="", visible=False),
        gr.update(interactive=False),
    )