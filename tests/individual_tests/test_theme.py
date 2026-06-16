from aiwf.web.theme import accent_preset_names, theme_css_overrides


def test_accent_preset_names_include_expected_choices():
    assert accent_preset_names() == ["mint", "amber", "ice"]


def test_theme_css_overrides_contains_selected_palette():
    css = theme_css_overrides(preset="amber")
    assert "--aiwf-accent: #d39a4f;" in css
    assert "rgba(211, 154, 79, 0.07)" in css
