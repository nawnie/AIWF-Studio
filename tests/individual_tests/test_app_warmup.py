from __future__ import annotations

from types import SimpleNamespace


class _Generation:
    def __init__(self, active_job=None):
        self._active_job = active_job

    def active_job(self):
        return self._active_job


def test_foreground_model_work_active_detects_image_job():
    from aiwf.app import _foreground_model_work_active

    ctx = SimpleNamespace(
        generation=_Generation(active_job=object()),
        supervisor=SimpleNamespace(active_tenant=SimpleNamespace(value="idle")),
    )

    assert _foreground_model_work_active(ctx) is True


def test_foreground_model_work_active_detects_video_tenant():
    from aiwf.app import _foreground_model_work_active

    ctx = SimpleNamespace(
        generation=_Generation(active_job=None),
        supervisor=SimpleNamespace(active_tenant=SimpleNamespace(value="video")),
    )

    assert _foreground_model_work_active(ctx) is True


def test_foreground_model_work_active_allows_idle():
    from aiwf.app import _foreground_model_work_active

    ctx = SimpleNamespace(
        generation=_Generation(active_job=None),
        supervisor=SimpleNamespace(active_tenant=SimpleNamespace(value="idle")),
    )

    assert _foreground_model_work_active(ctx) is False
