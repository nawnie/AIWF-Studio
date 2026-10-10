"""Commercial-safe soundtrack for a video: find the sounds, make each one, place them on the timeline.

MMAudio, ThinkSound and PrismAudio make audio straight from video but are licensed for non-commercial
use only, and HunyuanVideo-Foley excludes the EU, UK and South Korea. This pipeline reaches the same
goal with parts that can all be sold:

  1. extract_frames   ffmpeg takes a handful of frames across the video;
  2. plan_events      a local vision model (Qwen2.5-VL-7B, Apache-2.0) is shown the frames and returns
                      the sounds the scene would make as {start, end, prompt};
  3. (service)        MOSS-SoundEffect v2.0 (Apache-2.0) renders each prompt;
  4. mix_events       the clips are placed at their start times and summed into one track.

Sync is to the event, not to the frame: a door slam lands within the second the vision model named,
which is good for ambience, footsteps and scene sounds but not lip-sync-tight Foley.

The describer is injected (a function from frames to text), so tests need no model and the engine
that answers can be any Apache-licensed local vision model. Only the pure steps live here; running
the model is the audio service's job.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# vision models whose licences allow commercial use (prefix of the chat engine's model id).
# Qwen2.5-VL-3B (Qwen Research License) and Gemma (Google terms) are deliberately not listed.
COMMERCIAL_DESCRIBERS = ("qwen2.5-vl-7b",)

MAX_EVENTS = 8
MAX_EVENT_SECONDS = 30.0      # MOSS-SoundEffect's longest clip
MIN_EVENT_SECONDS = 0.8
FRAME_COUNT = 6

PROMPT = (
    "You are a sound designer. These {n} frames are taken evenly from a {duration:.1f}-second video "
    "(times in seconds: {times}). List the distinct sounds this scene would make, as a JSON array of "
    'objects {{"start": seconds, "end": seconds, "prompt": "a short, concrete description of one sound"}}. '
    "Include background ambience as one long event. Use at most {limit} events, keep every start/end "
    "between 0 and {duration:.1f}, and describe only sounds, never speech or music. Answer with the JSON "
    "array only.{hint}"
)


@dataclass(frozen=True)
class SoundEvent:
    start: float
    end: float
    prompt: str

    @property
    def seconds(self) -> float:
        return self.end - self.start


Describer = Callable[[list[tuple[float, bytes]], float, str], str]


def describer_is_commercial(model_id: str) -> bool:
    return str(model_id or "").lower().startswith(COMMERCIAL_DESCRIBERS)


# ---- 1. frames ---------------------------------------------------------------------------------------
def extract_frames(ffmpeg: str, video: Path, duration: float, count: int = FRAME_COUNT) -> list[tuple[float, bytes]]:
    """JPEG frames at evenly spaced times (the middle of each of `count` equal slices)."""
    frames: list[tuple[float, bytes]] = []
    with tempfile.TemporaryDirectory(prefix="aiwf-frames-") as scratch:
        for index in range(count):
            moment = duration * (index + 0.5) / count
            target = Path(scratch) / f"f{index}.jpg"
            subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-ss", f"{moment:.3f}", "-i", str(video), "-frames:v", "1",
                 "-vf", "scale=-2:448", "-q:v", "4", str(target)],
                check=False, capture_output=True,
            )
            if target.is_file() and target.stat().st_size > 0:
                frames.append((moment, target.read_bytes()))
    return frames


# ---- 2. events ----------------------------------------------------------------------------------------
def build_prompt(frames: list[tuple[float, bytes]], duration: float, hint: str = "") -> str:
    return PROMPT.format(
        n=len(frames), duration=duration, times=", ".join(f"{t:.1f}" for t, _ in frames), limit=MAX_EVENTS,
        hint=f" The person adds: {hint.strip()}" if hint.strip() else "",
    )


def parse_events(text: str, duration: float) -> list[SoundEvent]:
    """Read the model's answer into clean events: valid JSON array, clamped times, no empties."""
    body = (text or "").strip()
    body = re.sub(r"^```(?:json)?|```$", "", body, flags=re.MULTILINE).strip()
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("The vision model did not return a JSON list of sounds.")
    try:
        raw = json.loads(body[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"The vision model's sound list is not valid JSON: {exc}") from exc
    events: list[SoundEvent] = []
    # this loop keeps each well-formed event, clamping its times into the video
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or item.get("description") or "").strip()
        try:
            begin, finish = float(item.get("start", 0)), float(item.get("end", duration))
        except (TypeError, ValueError):
            continue
        begin = min(max(begin, 0.0), max(duration - MIN_EVENT_SECONDS, 0.0))
        finish = min(max(finish, begin + MIN_EVENT_SECONDS), duration)
        finish = min(finish, begin + MAX_EVENT_SECONDS)
        if prompt and finish - begin >= 0.5:
            events.append(SoundEvent(round(begin, 2), round(finish, 2), prompt[:300]))
    if not events:
        raise ValueError("The vision model listed no usable sounds for this video.")
    return _merge_repeats(sorted(events, key=lambda event: event.start))[:MAX_EVENTS]


def _merge_repeats(events: list[SoundEvent]) -> list[SoundEvent]:
    """Small vision models repeat themselves. Same sound listed again over a touching or overlapping
    stretch becomes one longer event, so it is generated once instead of five times."""
    merged: list[SoundEvent] = []
    # this loop folds each event into an earlier one with the same words that it overlaps or touches
    for event in events:
        key = re.sub(r"[^a-z0-9 ]", "", event.prompt.lower()).strip()
        for index, earlier in enumerate(merged):
            same = re.sub(r"[^a-z0-9 ]", "", earlier.prompt.lower()).strip() == key
            if same and event.start <= earlier.end + 0.25:
                merged[index] = SoundEvent(earlier.start, min(max(earlier.end, event.end), earlier.start + MAX_EVENT_SECONDS), earlier.prompt)
                break
        else:
            merged.append(event)
    return merged


def plan_events(describer: Describer, frames: list[tuple[float, bytes]], duration: float, hint: str = "") -> list[SoundEvent]:
    if not frames:
        raise ValueError("No frames could be read from the video.")
    return parse_events(describer(frames, duration, build_prompt(frames, duration, hint)), duration)


# ---- 4. mix ---------------------------------------------------------------------------------------------
def mix_events(clips: list[tuple[SoundEvent, np.ndarray]], duration: float, sample_rate: int,
               peak_dbfs: float = -3.0) -> np.ndarray:
    """Place each clip at its start time with short fades, sum them, and set the peak level.

    clips: (event, mono float32 samples at sample_rate). Returns mono float32 of exactly `duration`.
    """
    total = int(round(duration * sample_rate))
    track = np.zeros(total, dtype=np.float64)
    fade = int(0.03 * sample_rate)
    # this loop adds each clip into the track, trimmed to its event window and the video's end
    for event, clip in clips:
        begin = int(round(event.start * sample_rate))
        length = min(len(clip), int(round(event.seconds * sample_rate)), total - begin)
        if length <= 0:
            continue
        piece = clip[:length].astype(np.float64).copy()
        ramp = min(fade, length // 2)
        if ramp > 0:
            piece[:ramp] *= np.linspace(0.0, 1.0, ramp)
            piece[-ramp:] *= np.linspace(1.0, 0.0, ramp)
        track[begin:begin + length] += piece
    peak = float(np.max(np.abs(track))) if total else 0.0
    if peak > 1e-9:
        track *= (10.0 ** (peak_dbfs / 20.0)) / peak
    return track.astype(np.float32)


# ---- the chat engine as describer ------------------------------------------------------------------------------
class ChatDescriber:
    """Asks a vision model on the local llama.cpp chat engine (OpenAI-compatible, loopback only)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", key_file: Path | None = None, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.key_file = key_file or Path.home() / ".llama-chat" / "api-key.txt"
        self.timeout = timeout
        if not (self.base_url.startswith("http://127.") or self.base_url.startswith("http://localhost")):
            raise ValueError("The vision describer must be a loopback address.")

    def _headers(self) -> dict[str, str]:
        try:
            lines = [line.strip() for line in self.key_file.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.startswith("#")]
        except OSError:
            lines = []
        return {"Authorization": f"Bearer {lines[0]}"} if lines else {}

    def available_model(self, *, timeout: float = 15.0) -> str | None:
        """First commercially licensed vision model the chat engine offers."""
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        ids = [str(item.get("id", "")) for item in response.json().get("data", []) if isinstance(item, dict)]
        return next((model for model in ids if describer_is_commercial(model)), None)

    def release(self) -> None:
        """Ask the chat engine to unload the vision model (best effort; the engine keeps running)."""
        import httpx

        model = self.available_model()
        if model is None:
            return
        for base in (self.base_url, self.base_url.replace(":8080", ":8082")):
            try:
                if httpx.post(f"{base}/models/unload", headers=self._headers(), json={"model": model}, timeout=30).status_code < 300:
                    return
            except httpx.HTTPError:
                continue

    def __call__(self, frames: list[tuple[float, bytes]], duration: float, prompt: str) -> str:
        import httpx

        model = self.available_model()
        if model is None:
            raise RuntimeError(
                "No commercially licensed vision model (Qwen2.5-VL-7B) is available in the chat engine; "
                "add one to the chat model list to make video soundtracks."
            )
        content: list[dict] = [{"type": "text", "text": prompt}]
        for _moment, jpeg in frames:
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}})
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions", headers=self._headers(), timeout=self.timeout,
            json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.2, "max_tokens": 900},
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"].get("content") or "")
