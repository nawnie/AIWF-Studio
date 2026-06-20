from __future__ import annotations

from aiwf.infrastructure.diffusers.prompt_encode import build_prompt_kwargs


class _SD3Pipe:
    transformer = object()
    text_encoder_3 = object()

    def __call__(self, prompt=None, negative_prompt=None, clip_skip=None, **kwargs):
        return None


def test_sd3_prompt_with_emphasis_uses_native_prompt_kwargs():
    kwargs = build_prompt_kwargs(_SD3Pipe(), "(bright portrait)", "low quality", 2)

    assert kwargs == {
        "prompt": "(bright portrait)",
        "negative_prompt": "low quality",
        "clip_skip": 2,
    }
