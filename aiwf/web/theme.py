from __future__ import annotations

import gradio as gr


def build_theme(*, dark: bool = True) -> gr.Theme:
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
                button_primary_background_fill="#5da892",
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
            button_primary_background_fill="linear-gradient(180deg, #7fcab4 0%, #5da892 100%)",
            button_primary_text_color="#ffffff",
            button_primary_background_fill_hover="linear-gradient(180deg, #92d8c5 0%, #68c3aa 100%)",
            button_secondary_background_fill="#161b26",
            button_secondary_text_color="#b8c0d4",
            button_secondary_background_fill_hover="#1c2230",
            border_color_primary="rgba(255,255,255,0.1)",
            color_accent="#68c3aa",
            color_accent_soft="rgba(104,195,170,0.14)",
            link_text_color="#8fd8c4",
            shadow_drop="0 16px 48px rgba(0,0,0,0.55)",
            checkbox_background_color="#10141c",
            slider_color="#68c3aa",
        )
    )
