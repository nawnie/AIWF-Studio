from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from aiwf.web.studio.generation_runner import GenerationRunner


def _runner() -> GenerationRunner:
    ctx = MagicMock()
    ctx.settings.gallery_columns = 3
    ctx.tags.recent_tag_choices.return_value = []
    service = MagicMock()
    catalogs = MagicMock()
    session = MagicMock()
    session.loop_active = True
    return GenerationRunner(ctx, service, catalogs, session)


def test_session_accumulates_batches_and_keeps_last_primary():
    runner = _runner()
    img_a = Image.new("RGB", (8, 8), color=(255, 0, 0))
    img_b = Image.new("RGB", (8, 8), color=(0, 255, 0))

    runner._extend_session_outputs([img_a], [111])
    runner._extend_session_outputs([img_b], [222])

    assert len(runner._session_images) == 2
    assert runner._session_seeds == [111, 222]
    assert runner._last_primary_image is img_b


def test_gallery_visible_for_multiple_images():
    runner = _runner()
    images = [Image.new("RGB", (4, 4), color=(1, 2, 3)) for _ in range(2)]
    update = runner._gallery_update(images)
    assert update["visible"] is True
    assert update["columns"] == 2
    assert len(update["value"]) == 2