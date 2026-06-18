from types import SimpleNamespace

from aiwf.web.app import register_default_tabs
from aiwf.web.registry import WebRegistry
from aiwf.web.tabs.settings import TAB_VISIBILITY_CHOICES


def _noop(_ctx, _tab):
    return None


def test_visible_tabs_hides_secondary_tabs_from_settings():
    registry = WebRegistry()
    registry.tab("Image", order=1)(_noop)
    registry.tab("Models", order=2)(_noop)
    registry.tab("Settings", order=90)(_noop)

    ctx = SimpleNamespace(settings=SimpleNamespace(hidden_tabs=["Models"]))

    visible = [name for name, _builder, _order in registry.visible_tabs(ctx)]

    assert visible == ["Image", "Settings"]


def test_visible_tabs_keeps_pinned_tabs_even_if_hidden():
    registry = WebRegistry()
    registry.tab("Image", order=1)(_noop)
    registry.tab("Settings", order=90)(_noop)

    ctx = SimpleNamespace(settings=SimpleNamespace(hidden_tabs=["Image", "Settings"]))

    visible = [name for name, _builder, _order in registry.visible_tabs(ctx)]

    assert visible == ["Image", "Settings"]


def test_default_tabs_include_shipped_workspace_tabs():
    registry = WebRegistry()
    register_default_tabs(registry)

    names = [name for name, _builder, _order in registry.tabs]

    for expected in ("Image", "Models", "Segment", "Chat", "Video", "RIFE", "Training", "Settings"):
        assert expected in names

    assert names[:4] == ["Image", "Video", "Chat", "Training"]


def test_settings_visibility_choices_include_secondary_shipped_tabs():
    for expected in ("Models", "Segment", "Enhance", "Chat", "Video", "RIFE", "Training"):
        assert expected in TAB_VISIBILITY_CHOICES

    assert TAB_VISIBILITY_CHOICES[:3] == ["Video", "Chat", "Training"]
