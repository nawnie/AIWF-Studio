from __future__ import annotations

import gradio as gr


ACCENT_PRESETS = {
    "mint": {
        "primary_light": "#7fcab4",
        "primary_dark": "#5da892",
        "primary_hover_light": "#92d8c5",
        "primary_hover_dark": "#68c3aa",
        "accent": "#68c3aa",
        "accent_soft": "rgba(104,195,170,0.14)",
        "link": "#8fd8c4",
        "slider": "#68c3aa",
    },
    "amber": {
        "primary_light": "#dfb067",
        "primary_dark": "#b98842",
        "primary_hover_light": "#e9c27f",
        "primary_hover_dark": "#d09a4f",
        "accent": "#d39a4f",
        "accent_soft": "rgba(211,154,79,0.16)",
        "link": "#f0c785",
        "slider": "#d39a4f",
    },
    "ice": {
        "primary_light": "#79b8d9",
        "primary_dark": "#4e8fb0",
        "primary_hover_light": "#93c9e4",
        "primary_hover_dark": "#62a6c8",
        "accent": "#62a6c8",
        "accent_soft": "rgba(98,166,200,0.15)",
        "link": "#9ad4ef",
        "slider": "#62a6c8",
    },
}


def accent_preset_names() -> list[str]:
    return list(ACCENT_PRESETS)


def _accent(preset: str) -> dict[str, str]:
    return ACCENT_PRESETS.get(preset, ACCENT_PRESETS["mint"])


def theme_css_overrides(*, preset: str = "mint") -> str:
    palette = _accent(preset)
    bg_radial_secondary = "rgba(235, 174, 108, 0.05)"
    if preset == "amber":
        bg_radial_secondary = "rgba(211, 154, 79, 0.07)"
    elif preset == "ice":
        bg_radial_secondary = "rgba(98, 166, 200, 0.06)"
    return f"""
<style>
:root {{
    --aiwf-border-focus: color-mix(in srgb, {palette['accent']} 55%, transparent);
    --aiwf-accent: {palette['accent']};
    --aiwf-accent-bright: {palette['link']};
    --aiwf-accent-soft: {palette['accent_soft']};
    --aiwf-accent-glow: color-mix(in srgb, {palette['accent']} 30%, transparent);
    --aiwf-mode-text: {palette['link']};
}}
.aiwf-app::before {{
    background:
        radial-gradient(ellipse 70% 45% at 12% -8%, color-mix(in srgb, {palette['accent']} 9%, transparent), transparent 55%),
        radial-gradient(ellipse 55% 40% at 92% 8%, {bg_radial_secondary}, transparent 50%),
        radial-gradient(ellipse 50% 35% at 50% 100%, rgba(199, 125, 255, 0.04), transparent 55%),
        var(--aiwf-bg);
}}
</style>
""".strip()


def build_theme(*, dark: bool = True, accent_preset: str = "mint") -> gr.Theme:
    palette = _accent(accent_preset)
    """AIWF Studio visual identity — refined studio pro on obsidian."""
    if not dark:
        return (
            gr.themes.Soft(
                primary_hue=gr.themes.colors.blue,
                secondary_hue=gr.themes.colors.slate,
                neutral_hue=gr.themes.colors.gray,
                font=gr.themes.GoogleFont("DM Sans"),
                font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
            )
            .set(
                body_background_fill="#f6f7fb",
                block_background_fill="#ffffff",
                block_border_width="1px",
                block_label_text_weight="600",
                button_primary_background_fill=palette["primary_dark"],
                button_primary_text_color="#ffffff",
            )
        )

    return (
        gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("DM Sans"),
            font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        )
        .set(
            body_background_fill="#06070a",
            body_text_color="#f2f4f8",
            background_fill_primary="#0c0e14",
            background_fill_secondary="#10141c",
            block_background_fill="#10141c",
            block_border_color="rgba(255,255,255,0.08)",
            block_border_width="1px",
            block_radius="10px",
            block_label_text_weight="600",
            block_label_text_color="#8b93a8",
            block_title_text_color="#f2f4f8",
            input_background_fill="#080a10",
            input_border_color="rgba(255,255,255,0.1)",
            input_radius="10px",
            button_large_radius="10px",
            button_primary_background_fill=f"linear-gradient(180deg, {palette['primary_light']} 0%, {palette['primary_dark']} 100%)",
            button_primary_text_color="#ffffff",
            button_primary_background_fill_hover=f"linear-gradient(180deg, {palette['primary_hover_light']} 0%, {palette['primary_hover_dark']} 100%)",
            button_secondary_background_fill="#161b26",
            button_secondary_text_color="#b8c0d4",
            button_secondary_background_fill_hover="#1c2230",
            border_color_primary="rgba(255,255,255,0.1)",
            color_accent=palette["accent"],
            color_accent_soft=palette["accent_soft"],
            link_text_color=palette["link"],
            shadow_drop="0 16px 48px rgba(0,0,0,0.55)",
            checkbox_background_color="#10141c",
            slider_color=palette["slider"],
        )
    )
