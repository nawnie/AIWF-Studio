"""Audio Lab's permissive signal chain (engines/audio_lab/dsp.py), measured on test signals.

dsp.py replaces Pedalboard (GPL-3.0). Each effect must keep the region's exact length and do
what its name says, within a tolerance a mixing engineer would accept: filters attenuate where
they should, EQ boosts by the set amount at the set frequency, dynamics hold their levels, and
pitch shift moves a tone by the right interval.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

DSP_PATH = Path(__file__).resolve().parents[2] / "engines" / "audio_lab" / "dsp.py"
RATE = 48000


@pytest.fixture(scope="module")
def dsp():
    spec = importlib.util.spec_from_file_location("audio_lab_dsp", DSP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- test signals and measurements -------------------------------------------------------------------
def _tone(frequency: float, level_db: float = -12.0, seconds: float = 1.0, channels: int = 2) -> np.ndarray:
    time = np.arange(int(RATE * seconds)) / RATE
    wave = float(10 ** (level_db / 20)) * np.sin(2 * np.pi * frequency * time)
    return np.tile(wave, (channels, 1)).astype(np.float32)


def _rms_db(audio: np.ndarray, skip_seconds: float = 0.2) -> float:
    """Level of the settled part of the signal (filters and dynamics need a moment to settle)."""
    settled = audio[:, int(skip_seconds * RATE):]
    return float(20 * np.log10(np.sqrt(np.mean(settled.astype(np.float64) ** 2)) + 1e-12))


def _change_db(dsp_out: np.ndarray, source: np.ndarray) -> float:
    return _rms_db(dsp_out) - _rms_db(source)


def _peak_frequency(channel: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(channel * np.hanning(channel.size)))
    return float(np.fft.rfftfreq(channel.size, 1 / RATE)[int(np.argmax(spectrum))])


# ---- every effect keeps the exact length ----------------------------------------------------------------
def test_every_effect_keeps_the_region_length(dsp) -> None:
    rng = np.random.default_rng(7)
    audio = (0.1 * rng.standard_normal((2, 12345))).astype(np.float32)
    effects = [
        lambda a: dsp.gain(a, 3.0),
        lambda a: dsp.highpass(a, RATE, 120.0),
        lambda a: dsp.lowpass(a, RATE, 9000.0),
        lambda a: dsp.peak_eq(a, RATE, 1000.0, 4.0, 1.0),
        lambda a: dsp.low_shelf(a, RATE, 200.0, -3.0),
        lambda a: dsp.high_shelf(a, RATE, 6000.0, 2.0),
        lambda a: dsp.compressor(a, RATE, -24.0, 3.0, 5.0, 80.0),
        lambda a: dsp.noise_gate(a, RATE, -50.0, 8.0, 2.0, 60.0),
        lambda a: dsp.limiter(a, RATE, -3.0, 60.0),
        lambda a: dsp.pitch_shift(a, RATE, 3.0),
    ]
    for effect in effects:
        out = effect(audio)
        assert out.shape == audio.shape and out.dtype == np.float32 and np.isfinite(out).all()


# ---- filters and EQ ---------------------------------------------------------------------------------------
def test_highpass_and_lowpass_are_first_order(dsp) -> None:
    low, high = _tone(100.0), _tone(10000.0)
    # one decade below a first-order corner is about -20 dB; far above it passes unchanged
    assert _change_db(dsp.highpass(low, RATE, 1000.0), low) == pytest.approx(-20.0, abs=1.0)
    assert _change_db(dsp.highpass(high, RATE, 1000.0), high) == pytest.approx(0.0, abs=0.3)
    # the digital (bilinear) low-pass falls a little faster than the analog -20 dB near Nyquist
    assert -22.5 <= _change_db(dsp.lowpass(high, RATE, 1000.0), high) <= -19.5
    assert _change_db(dsp.lowpass(low, RATE, 1000.0), low) == pytest.approx(0.0, abs=0.3)


def test_peak_eq_boosts_exactly_at_its_center(dsp) -> None:
    center, away = _tone(1000.0), _tone(80.0)
    assert _change_db(dsp.peak_eq(center, RATE, 1000.0, 6.0, 1.0), center) == pytest.approx(6.0, abs=0.2)
    assert _change_db(dsp.peak_eq(center, RATE, 1000.0, -6.0, 1.0), center) == pytest.approx(-6.0, abs=0.2)
    assert _change_db(dsp.peak_eq(away, RATE, 1000.0, 6.0, 1.0), away) == pytest.approx(0.0, abs=0.5)


def test_shelves_move_only_their_side_of_the_spectrum(dsp) -> None:
    low, high = _tone(40.0, seconds=1.5), _tone(12000.0)
    assert _change_db(dsp.low_shelf(low, RATE, 300.0, 6.0), low) == pytest.approx(6.0, abs=0.5)
    assert _change_db(dsp.low_shelf(high, RATE, 300.0, 6.0), high) == pytest.approx(0.0, abs=0.3)
    assert _change_db(dsp.high_shelf(high, RATE, 3000.0, 6.0), high) == pytest.approx(6.0, abs=0.5)
    assert _change_db(dsp.high_shelf(low, RATE, 3000.0, 6.0), low) == pytest.approx(0.0, abs=0.3)


def test_gain_is_exact(dsp) -> None:
    tone = _tone(440.0)
    assert _change_db(dsp.gain(tone, -9.5), tone) == pytest.approx(-9.5, abs=0.01)


# ---- dynamics -----------------------------------------------------------------------------------------------
def test_compressor_holds_the_expected_level(dsp) -> None:
    loud, quiet = _tone(440.0, level_db=-6.0, seconds=2.0), _tone(440.0, level_db=-40.0, seconds=2.0)
    # a -6 dBFS-peak tone over a -20 dB threshold at 4:1 settles 14 * 3/4 = 10.5 dB lower
    assert _change_db(dsp.compressor(loud, RATE, -20.0, 4.0, 5.0, 100.0), loud) == pytest.approx(-10.5, abs=1.0)
    assert _change_db(dsp.compressor(quiet, RATE, -20.0, 4.0, 5.0, 100.0), quiet) == pytest.approx(0.0, abs=0.1)


def test_compressor_attack_preserves_region_start_transient(dsp) -> None:
    loud = _tone(440.0, level_db=-6.0, seconds=0.5)
    processed = dsp.compressor(loud, RATE, -20.0, 4.0, 100.0, 100.0)
    first_block = int(RATE * 0.001)

    # A long attack starts with unity gain; the same sustained tone is compressed later.
    assert np.max(np.abs(processed[:, :first_block])) >= 0.95 * np.max(np.abs(loud[:, :first_block]))
    assert np.max(np.abs(processed[:, 200 * first_block:])) < 0.5 * np.max(np.abs(loud[:, 200 * first_block:]))


def test_noise_gate_closes_on_noise_and_opens_for_signal(dsp) -> None:
    rng = np.random.default_rng(3)
    hiss = (0.001 * rng.standard_normal((2, RATE))).astype(np.float32)          # about -60 dBFS
    voice = _tone(220.0, level_db=-10.0)
    assert _change_db(dsp.noise_gate(hiss, RATE, -40.0, 10.0, 1.0, 50.0), hiss) < -30.0
    assert _change_db(dsp.noise_gate(voice, RATE, -40.0, 10.0, 1.0, 50.0), voice) == pytest.approx(0.0, abs=0.2)


def test_noise_gate_attack_fades_in_signal_at_region_start(dsp) -> None:
    voice = _tone(220.0, level_db=-10.0, seconds=0.5)
    processed = dsp.noise_gate(voice, RATE, -40.0, 10.0, 100.0, 100.0)
    first_block = int(RATE * 0.001)

    # Starting below threshold closes the gate; the configured attack governs its opening.
    assert np.max(np.abs(processed[:, :first_block])) < 0.1 * np.max(np.abs(voice[:, :first_block]))
    assert np.max(np.abs(processed[:, 200 * first_block:])) > 0.9 * np.max(np.abs(voice[:, 200 * first_block:]))


def test_limiter_never_lets_peaks_through(dsp) -> None:
    loud = _tone(100.0, level_db=0.0)
    limited = dsp.limiter(loud, RATE, -6.0, 50.0)
    assert float(np.max(np.abs(limited))) <= 10 ** (-6.0 / 20) + 1e-6
    assert _change_db(limited, loud) < -4.0


# ---- pitch and sample rate ---------------------------------------------------------------------------------
@pytest.mark.parametrize(("semitones", "expected_hz"), [(12.0, 440.0), (-12.0, 110.0), (7.0, 220.0 * 2 ** (7 / 12))])
def test_pitch_shift_moves_the_tone_by_the_interval(dsp, semitones: float, expected_hz: float) -> None:
    tone = _tone(220.0, seconds=2.0)
    shifted = dsp.pitch_shift(tone, RATE, semitones)
    assert shifted.shape == tone.shape
    assert _peak_frequency(shifted[0]) == pytest.approx(expected_hz, rel=0.01)
    # phase locking keeps the partials coherent, so the loudness stays the same
    assert _change_db(shifted, tone) == pytest.approx(0.0, abs=0.5)


def test_pitch_shift_keeps_a_chord_together(dsp) -> None:
    time = np.arange(2 * RATE) / RATE
    chord = sum(0.1 * np.sin(2 * np.pi * f * time) for f in (220.0, 277.2, 329.6))
    audio = np.tile(chord, (2, 1)).astype(np.float32)
    shifted = dsp.pitch_shift(audio, RATE, 12.0)
    spectrum = np.abs(np.fft.rfft(shifted[0, RATE // 5:] * np.hanning(shifted.shape[1] - RATE // 5)))
    freqs = np.fft.rfftfreq(shifted.shape[1] - RATE // 5, 1 / RATE)
    strongest = sorted(float(freqs[i]) for i in np.argsort(spectrum)[-200:])
    # all three chord tones moved up an octave
    for target in (440.0, 554.4, 659.2):
        assert min(abs(f - target) for f in strongest) < 3.0
    assert _change_db(shifted, audio) == pytest.approx(0.0, abs=1.0)


def test_resample_keeps_duration_and_pitch(dsp) -> None:
    tone = _tone(1000.0)
    converted = dsp.resample(tone, RATE, 44100)
    assert converted.shape == (2, 44100)
    spectrum = np.abs(np.fft.rfft(converted[0] * np.hanning(converted.shape[1])))
    assert float(np.fft.rfftfreq(converted.shape[1], 1 / 44100)[int(np.argmax(spectrum))]) == pytest.approx(1000.0, abs=2.0)
