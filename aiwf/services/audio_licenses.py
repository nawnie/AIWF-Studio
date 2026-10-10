"""Licences of the audio models AIWF Studio can run, and the commercial-use policy built on them.

AIWF Studio is sold as a commercial product, so by default it offers only audio models whose
licences allow commercial use without conditions. Models released for non-commercial or
research use stay available only in an explicit "research mode" that the person switches on
(UserSettings.allow_noncommercial_audio_models); the audio service refuses them otherwise, so
the UI, the Pro API, Gradio, Video Lab and agents all follow the same rule.

Every entry records what the licence says, where it was read, and when. "provenance" is what
the model's publisher states about its training data; an undocumented provenance is a residual
risk to keep in mind, not a licence restriction. This is a product policy, not legal advice.

The module is plain data with no heavy imports, so any surface can load it.
"""

from __future__ import annotations

from typing import Any

CHECKED = "2026-10-08"

# --- verdicts --------------------------------------------------------------------------
YES = "yes"                  # commercial use allowed without conditions
CONDITIONAL = "conditional"  # allowed with conditions (revenue cap, territory, ...)
NO = "no"                    # non-commercial or research only

_CC_BY_NC = {
    "license": "CC-BY-NC-4.0",
    "commercial": NO,
    "conditions": "Released for non-commercial use only. Use in research mode; do not use the output commercially.",
}

# --- one entry per selectable model ID (prefix entries cover every variant) -------------------
_MODELS: dict[str, dict[str, Any]] = {
    # music: Meta MusicGen (all released checkpoints share the licence)
    "facebook/musicgen-": {
        **_CC_BY_NC,
        "family": "MusicGen (Meta)",
        "provenance": "Meta's licensed music data; the weights themselves are non-commercial.",
        "source": "https://huggingface.co/facebook/musicgen-small",
    },
    # sound effects and video-to-audio: MMAudio
    "mmaudio:": {
        **_CC_BY_NC,
        "family": "MMAudio",
        "provenance": "Academic audio-visual datasets; weights are non-commercial.",
        "source": "https://huggingface.co/hkchengrex/MMAudio",
    },
    # music: ACE-Step 1.5 (commercial replacement for MusicGen)
    "acestep:": {
        "license": "MIT",
        "commercial": YES,
        "conditions": "",
        "family": "ACE-Step 1.5",
        "provenance": "Publisher states professionally licensed, royalty-free/public-domain and synthetic (MIDI-rendered) music.",
        "source": "https://huggingface.co/ACE-Step/Ace-Step1.5",
    },
    # sound effects: MOSS-SoundEffect v2.0 (commercial replacement for MMAudio text-to-audio)
    "moss-sfx:": {
        "license": "Apache-2.0",
        "commercial": YES,
        "conditions": "",
        "family": "MOSS-SoundEffect",
        "provenance": "Not documented on the model card.",
        "source": "https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0",
    },
    # audio for video: events found by a local vision model, each sound made by MOSS-SoundEffect
    "events:": {
        "license": "Apache-2.0 (MOSS-SoundEffect) + the vision model's licence",
        "commercial": YES,
        "conditions": "The vision model only describes the video; its licence is listed with the chosen model.",
        "family": "Two-step video audio",
        "provenance": "Sound generation as MOSS-SoundEffect; no audio is produced by the vision model.",
        "source": "https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0",
    },
}


def license_for(model_id: str) -> dict[str, Any]:
    """The licence record for a model ID; unknown models are treated as not cleared for commercial use."""
    text = str(model_id or "").strip()
    for prefix, record in _MODELS.items():
        if text.startswith(prefix):
            return {"model_id": text, "checked": CHECKED, **record}
    return {
        "model_id": text,
        "checked": CHECKED,
        "license": "unknown",
        "commercial": NO,
        "conditions": "No licence on record; not offered for commercial use.",
        "family": "unknown",
        "provenance": "unknown",
        "source": "",
    }


def commercial_ok(model_id: str) -> bool:
    """True only when the licence allows commercial use without conditions."""
    return license_for(model_id)["commercial"] == YES


def allowed(model_id: str, *, research_mode: bool) -> bool:
    """Whether this model may be offered or run under the current policy."""
    return commercial_ok(model_id) or bool(research_mode)


def short_label(model_id: str) -> str:
    """A few words for pickers and output metadata, e.g. 'MIT, commercial use OK'."""
    record = license_for(model_id)
    verdict = {YES: "commercial use OK", CONDITIONAL: "commercial use with conditions", NO: "non-commercial only"}[record["commercial"]]
    return f"{record['license']}, {verdict}"


def notice_for(model_id: str) -> str:
    """A readable project notice retaining the fields needed to audit a saved output."""
    record = license_for(model_id)
    parts = [
        str(record["license"]),
        f"commercial use: {record['commercial']}",
        f"checked: {record['checked']}",
    ]
    if record.get("conditions"):
        parts.append(f"conditions: {record['conditions']}")
    if record.get("source"):
        parts.append(f"source: {record['source']}")
    return "; ".join(parts)


def blocked_message(model_id: str) -> str:
    """The sentence shown when a non-commercial model is requested outside research mode."""
    record = license_for(model_id)
    return (
        f"{record['family']} is licensed {record['license']} for non-commercial use only, so AIWF Studio does not run it "
        "by default. Turn on 'Allow non-commercial research models' in Audio settings to use it for research, "
        "or choose a commercial-safe model."
    )
