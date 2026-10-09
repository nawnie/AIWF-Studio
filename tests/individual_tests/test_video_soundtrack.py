"""The commercial-safe video soundtrack pipeline (aiwf/services/video_soundtrack.py).

The vision model and the sound generator are replaced by fakes, so these tests check what is
ours: reading the model's answer safely, choosing only commercially licensed describers, and
placing clips on the timeline at the right sample.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aiwf.services import video_soundtrack as vs

RATE = 48000


# ---- reading the vision model's answer -----------------------------------------------------------------
def test_parse_events_accepts_a_clean_list() -> None:
    events = vs.parse_events(json.dumps([{"start": 0, "end": 9.5, "prompt": "wind over a field"},
                                         {"start": 3.2, "end": 4.4, "prompt": "a door slams"}]), 10.0)
    assert [(e.start, e.end, e.prompt) for e in events] == [(0.0, 9.5, "wind over a field"), (3.2, 4.4, "a door slams")]


def test_parse_events_survives_code_fences_and_chatter() -> None:
    answer = 'Sure! Here you go:\n```json\n[{"start": 1, "end": 3, "prompt": "footsteps on gravel"}]\n```\nAnything else?'
    assert vs.parse_events(answer, 8.0)[0].prompt == "footsteps on gravel"


def test_parse_events_clamps_times_into_the_video_and_caps_the_count() -> None:
    raw = [{"start": -4, "end": 99, "prompt": "ambience"}] + [{"start": i, "end": i + 1, "prompt": f"sound {i}"} for i in range(20)]
    events = vs.parse_events(json.dumps(raw), 12.0)
    assert len(events) == vs.MAX_EVENTS
    assert all(0.0 <= e.start < e.end <= 12.0 for e in events)


@pytest.mark.parametrize("answer", ["no json at all", "[1, 2, 3]", '[{"start": 1, "end": 2}]', "[{broken"])
def test_parse_events_rejects_unusable_answers(answer: str) -> None:
    with pytest.raises(ValueError):
        vs.parse_events(answer, 10.0)


def test_plan_events_passes_frames_and_the_persons_hint_to_the_describer() -> None:
    seen = {}

    def describer(frames, duration, prompt):
        seen.update(frames=len(frames), duration=duration, prompt=prompt)
        return '[{"start": 0, "end": 4, "prompt": "rain on a window"}]'

    frames = [(1.0, b"a"), (3.0, b"b")]
    events = vs.plan_events(describer, frames, 4.0, hint="cozy cabin")
    assert events[0].prompt == "rain on a window"
    assert seen["frames"] == 2 and "cozy cabin" in seen["prompt"] and "4.0-second" in seen["prompt"]
    with pytest.raises(ValueError):
        vs.plan_events(describer, [], 4.0)


# ---- the describer must be commercially licensed ------------------------------------------------------------
@pytest.mark.parametrize(("model", "ok"), [
    ("qwen2.5-vl-7b-abliterated", True), ("Qwen2.5-VL-7B-Instruct", True),
    ("qwen2.5-vl-3b-instruct", False),      # Qwen Research License
    ("gemma-3-12b-it-qat-vision", False),   # Google's terms: conditional, not the default
    ("llama-3.2-vision", False),
])
def test_only_apache_vision_models_are_commercial_describers(model: str, ok: bool) -> None:
    assert vs.describer_is_commercial(model) is ok


def test_chat_describer_refuses_non_loopback_addresses() -> None:
    with pytest.raises(ValueError):
        vs.ChatDescriber("http://192.168.1.10:8080")


# ---- the mix --------------------------------------------------------------------------------------------------
def _clip(seconds: float, level: float = 0.2) -> np.ndarray:
    time = np.arange(int(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * 440 * time)).astype(np.float32)


def test_mix_places_each_clip_at_its_start_and_sets_the_peak() -> None:
    clips = [(vs.SoundEvent(1.0, 2.0, "a"), _clip(1.0)), (vs.SoundEvent(3.0, 4.0, "b"), _clip(1.0, 0.1))]
    track = vs.mix_events(clips, 5.0, RATE)
    assert track.shape == (5 * RATE,) and track.dtype == np.float32
    assert np.max(np.abs(track[: int(0.9 * RATE)])) == 0.0                       # silence before the first sound
    assert np.max(np.abs(track[int(1.2 * RATE): int(1.8 * RATE)])) > 0.1          # first sound present
    assert np.max(np.abs(track[int(2.2 * RATE): int(2.8 * RATE)])) == 0.0         # gap between sounds
    assert 20 * np.log10(np.max(np.abs(track))) == pytest.approx(-3.0, abs=0.05)
    # the second clip was recorded at half the level and stays quieter after normalization
    assert np.max(np.abs(track[int(3.2 * RATE): int(3.8 * RATE)])) < np.max(np.abs(track[int(1.2 * RATE): int(1.8 * RATE)]))


def test_mix_trims_clips_to_their_event_and_the_video_end() -> None:
    long_clip = _clip(6.0)
    track = vs.mix_events([(vs.SoundEvent(4.0, 9.0, "late"), long_clip)], 5.0, RATE)
    assert track.shape == (5 * RATE,) and np.max(np.abs(track[: 4 * RATE])) == 0.0
    assert np.max(np.abs(track[4 * RATE:])) > 0.1


def test_mix_fades_clip_edges_so_there_are_no_clicks() -> None:
    track = vs.mix_events([(vs.SoundEvent(0.0, 1.0, "x"), _clip(1.0, 0.5))], 1.0, RATE)
    assert abs(float(track[0])) < 0.01 and abs(float(track[-1])) < 0.01


def test_repeated_sounds_merge_into_one_longer_event() -> None:
    raw = [{"start": 0.7, "end": 8.0, "prompt": "Ocean waves"},
           {"start": 2.0, "end": 3.3, "prompt": "waves crashing against rocks"},
           {"start": 3.3, "end": 4.7, "prompt": "waves crashing against rocks"},
           {"start": 4.7, "end": 6.0, "prompt": "Waves crashing against rocks."},
           {"start": 7.0, "end": 8.0, "prompt": "ocean waves"}]
    events = vs.parse_events(json.dumps(raw), 8.0)
    assert [(e.start, e.end, e.prompt) for e in events] == [(0.7, 8.0, "Ocean waves"), (2.0, 6.0, "waves crashing against rocks")]
