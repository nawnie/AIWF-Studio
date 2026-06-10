from aiwf.core.config.settings import UserSettings


def test_live_preview_disabled_returns_zero_interval():
    settings = UserSettings(enable_live_preview=False, show_progress_every_n_steps=3)
    assert settings.live_preview_interval() == 0
    assert settings.live_preview_summary() == "Live preview off"


def test_live_preview_enabled_clamps_interval():
    settings = UserSettings(enable_live_preview=True, show_progress_every_n_steps=5)
    assert settings.live_preview_interval() == 5
    assert settings.live_preview_summary() == "Live preview every 5 steps"


def test_live_preview_every_step_summary():
    settings = UserSettings(enable_live_preview=True, show_progress_every_n_steps=1)
    assert settings.live_preview_interval() == 1
    assert settings.live_preview_summary() == "Live preview every step"