from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

AudioLabStage = Literal[
    "trim",
    "gate",
    "filters",
    "eq",
    "compressor",
    "pitch",
    "gain",
    "pan",
    "envelope",
    "normalize",
    "limiter",
    "export",
]

AUDIO_LAB_ORDER: tuple[AudioLabStage, ...] = (
    "trim",
    "gate",
    "filters",
    "eq",
    "compressor",
    "pitch",
    "gain",
    "pan",
    "envelope",
    "normalize",
    "limiter",
    "export",
)

AUDIO_STAGE_LABELS: dict[str, str] = {
    "trim": "Trim",
    "gate": "Noise gate",
    "filters": "High / low pass",
    "eq": "Equalizer",
    "compressor": "Compressor",
    "pitch": "Pitch shift",
    "gain": "Gain",
    "pan": "Pan",
    "envelope": "Automation / fades",
    "normalize": "Loudness normalize",
    "limiter": "Limiter",
    "export": "Export",
}


class AudioLabSettings(BaseModel):
    stages: list[AudioLabStage] = Field(default_factory=lambda: ["normalize", "limiter", "export"])
    preset: str = "custom"

    trim_start_seconds: float = Field(default=0.0, ge=0.0)
    trim_end_seconds: float | None = Field(default=None, gt=0.0)

    gate_threshold_db: float = Field(default=-48.0, ge=-100.0, le=0.0)
    gate_ratio: float = Field(default=6.0, ge=1.0, le=20.0)
    gate_attack_ms: float = Field(default=5.0, ge=0.1, le=500.0)
    gate_release_ms: float = Field(default=120.0, ge=1.0, le=3000.0)

    highpass_hz: float = Field(default=30.0, ge=10.0, le=2000.0)
    lowpass_hz: float = Field(default=20000.0, ge=1000.0, le=24000.0)

    low_shelf_hz: float = Field(default=120.0, ge=20.0, le=1000.0)
    low_shelf_gain_db: float = Field(default=0.0, ge=-18.0, le=18.0)
    mid_hz: float = Field(default=1000.0, ge=100.0, le=12000.0)
    mid_gain_db: float = Field(default=0.0, ge=-18.0, le=18.0)
    mid_q: float = Field(default=0.8, ge=0.1, le=10.0)
    high_shelf_hz: float = Field(default=8000.0, ge=1000.0, le=20000.0)
    high_shelf_gain_db: float = Field(default=0.0, ge=-18.0, le=18.0)

    compressor_threshold_db: float = Field(default=-18.0, ge=-60.0, le=0.0)
    compressor_ratio: float = Field(default=3.0, ge=1.0, le=20.0)
    compressor_attack_ms: float = Field(default=15.0, ge=0.1, le=500.0)
    compressor_release_ms: float = Field(default=120.0, ge=1.0, le=3000.0)

    pitch_semitones: float = Field(default=0.0, ge=-24.0, le=24.0)
    pitch_start_seconds: float = Field(default=0.0, ge=0.0)
    pitch_end_seconds: float | None = Field(default=None, gt=0.0)

    gain_db: float = Field(default=0.0, ge=-36.0, le=24.0)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)

    fade_in_seconds: float = Field(default=0.0, ge=0.0, le=120.0)
    fade_out_seconds: float = Field(default=0.0, ge=0.0, le=120.0)
    gain_envelope: str = ""

    target_lufs: float = Field(default=-14.0, ge=-36.0, le=-5.0)
    limiter_threshold_db: float = Field(default=-1.0, ge=-20.0, le=0.0)
    limiter_release_ms: float = Field(default=100.0, ge=1.0, le=3000.0)

    export_format: Literal["wav", "flac"] = "wav"
    sample_rate: int = Field(default=0, ge=0, le=192000)

    @model_validator(mode="after")
    def normalize(self):
        selected = set(self.stages)
        self.stages = [stage for stage in AUDIO_LAB_ORDER if stage in selected]
        if "export" not in self.stages:
            self.stages.append("export")
        if "trim" in self.stages and self.trim_end_seconds is not None and self.trim_end_seconds <= self.trim_start_seconds:
            raise ValueError("Trim end must be greater than trim start.")
        if "pitch" in self.stages and self.pitch_end_seconds is not None and self.pitch_end_seconds <= self.pitch_start_seconds:
            raise ValueError("Pitch region end must be greater than its start.")
        if "filters" in self.stages and self.lowpass_hz <= self.highpass_hz:
            raise ValueError("Low-pass frequency must be above the high-pass frequency.")
        return self


class AudioLabPlan(BaseModel):
    stages: list[AudioLabStage]
    labels: list[str]
    warnings: list[str] = Field(default_factory=list)


class AudioEngineStatus(BaseModel):
    installed: bool
    python_path: str | None = None
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class AudioCommandPlan(BaseModel):
    understood: bool
    operation: str = "unknown"
    parameters: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
