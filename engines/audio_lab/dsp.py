"""Audio Lab's signal chain, written on NumPy and SciPy only (both BSD-licensed).

This replaces Pedalboard (GPL-3.0) so the audio engine can ship inside a commercial installer.
Every effect works on a whole region at once: audio is float32 shaped (channels, frames), and
each function returns the same shape and the same number of frames, so later automation times
and project metadata stay aligned (the same promise the Pedalboard board kept with reset=True).

The filters use the standard formulas Pedalboard's JUCE filters are built from:
- high-pass / low-pass: first-order (6 dB per octave), like Pedalboard's HighpassFilter/LowpassFilter;
- low shelf, peak and high shelf: Robert Bristow-Johnson's "Audio EQ Cookbook" biquads.
Dynamics (gate, compressor, limiter) follow the signal level at a 1 ms block rate and turn the
smoothed gain back into a per-sample curve, which keeps them fast without a per-sample Python loop.
Pitch shift is a phase vocoder (time stretch) followed by polyphase resampling back to the
original length; Pedalboard used Rubber Band here, which is GPL as well.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d

_EPS = 1e-12
_BLOCK_SECONDS = 0.001   # dynamics look at the level once per millisecond


# ---- helpers ------------------------------------------------------------------------------------------
def _as_float(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32)


def _db_to_gain(db: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def _clamp_frequency(frequency_hz: float, sample_rate: int) -> float:
    # keep every corner frequency inside the audible band the sample rate can represent
    return float(min(max(frequency_hz, 1.0), 0.49 * sample_rate))


def _fit_length(audio: np.ndarray, frames: int) -> np.ndarray:
    if audio.shape[-1] > frames:
        return audio[..., :frames]
    if audio.shape[-1] < frames:
        return np.pad(audio, [(0, 0)] * (audio.ndim - 1) + [(0, frames - audio.shape[-1])])
    return audio


# ---- gain and filters ------------------------------------------------------------------------------------
def gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return _as_float(audio * float(_db_to_gain(gain_db)))


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    sos = signal.butter(1, _clamp_frequency(cutoff_hz, sample_rate), btype="highpass", fs=sample_rate, output="sos")
    return _as_float(signal.sosfilt(sos, audio, axis=-1))


def lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    sos = signal.butter(1, _clamp_frequency(cutoff_hz, sample_rate), btype="lowpass", fs=sample_rate, output="sos")
    return _as_float(signal.sosfilt(sos, audio, axis=-1))


def _biquad(audio: np.ndarray, b: tuple[float, float, float], a: tuple[float, float, float]) -> np.ndarray:
    a0 = a[0]
    return _as_float(signal.lfilter(np.array(b) / a0, np.array(a) / a0, audio, axis=-1))


def peak_eq(audio: np.ndarray, sample_rate: int, center_hz: float, gain_db: float, q: float) -> np.ndarray:
    """Bell boost or cut around center_hz (RBJ peaking EQ)."""
    amplitude = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * _clamp_frequency(center_hz, sample_rate) / sample_rate
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    cos_w0 = math.cos(w0)
    return _biquad(
        audio,
        (1.0 + alpha * amplitude, -2.0 * cos_w0, 1.0 - alpha * amplitude),
        (1.0 + alpha / amplitude, -2.0 * cos_w0, 1.0 - alpha / amplitude),
    )


def low_shelf(audio: np.ndarray, sample_rate: int, cutoff_hz: float, gain_db: float, q: float = 0.707) -> np.ndarray:
    """Boost or cut everything below cutoff_hz (RBJ low shelf)."""
    amplitude = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * _clamp_frequency(cutoff_hz, sample_rate) / sample_rate
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    cos_w0, root = math.cos(w0), 2.0 * math.sqrt(amplitude) * alpha
    return _biquad(
        audio,
        (amplitude * ((amplitude + 1) - (amplitude - 1) * cos_w0 + root),
         2.0 * amplitude * ((amplitude - 1) - (amplitude + 1) * cos_w0),
         amplitude * ((amplitude + 1) - (amplitude - 1) * cos_w0 - root)),
        ((amplitude + 1) + (amplitude - 1) * cos_w0 + root,
         -2.0 * ((amplitude - 1) + (amplitude + 1) * cos_w0),
         (amplitude + 1) + (amplitude - 1) * cos_w0 - root),
    )


def high_shelf(audio: np.ndarray, sample_rate: int, cutoff_hz: float, gain_db: float, q: float = 0.707) -> np.ndarray:
    """Boost or cut everything above cutoff_hz (RBJ high shelf)."""
    amplitude = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * _clamp_frequency(cutoff_hz, sample_rate) / sample_rate
    alpha = math.sin(w0) / (2.0 * max(q, 0.05))
    cos_w0, root = math.cos(w0), 2.0 * math.sqrt(amplitude) * alpha
    return _biquad(
        audio,
        (amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w0 + root),
         -2.0 * amplitude * ((amplitude - 1) + (amplitude + 1) * cos_w0),
         amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w0 - root)),
        ((amplitude + 1) - (amplitude - 1) * cos_w0 + root,
         2.0 * ((amplitude - 1) - (amplitude + 1) * cos_w0),
         (amplitude + 1) - (amplitude - 1) * cos_w0 - root),
    )


# ---- dynamics -----------------------------------------------------------------------------------------------
def _block_peaks(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Peak level of each 1 ms block across all channels (stereo-linked detection)."""
    block = max(1, int(round(sample_rate * _BLOCK_SECONDS)))
    frames = audio.shape[-1]
    count = max(1, math.ceil(frames / block))
    padded = np.pad(np.abs(audio), ((0, 0), (0, count * block - frames)))
    return padded.reshape(audio.shape[0], count, block).max(axis=(0, 2)), block


def _follow(levels: np.ndarray, block_seconds: float, attack_ms: float, release_ms: float) -> np.ndarray:
    """Envelope follower: rises with the attack time, falls with the release time."""
    attack = math.exp(-block_seconds / max(attack_ms / 1000.0, block_seconds))
    release = math.exp(-block_seconds / max(release_ms / 1000.0, block_seconds))
    envelope = np.empty_like(levels)
    current = 0.0
    # this loop runs once per millisecond of audio, not once per sample
    for index, level in enumerate(levels.tolist()):
        coefficient = attack if level > current else release
        current = coefficient * current + (1.0 - coefficient) * level
        envelope[index] = current
    return envelope


def _apply_block_gain(audio: np.ndarray, gains: np.ndarray, block: int) -> np.ndarray:
    """Spread one gain per block smoothly over every sample of the region."""
    frames = audio.shape[-1]
    centers = (np.arange(gains.size) + 0.5) * block
    curve = np.interp(np.arange(frames), centers, gains).astype(np.float32)
    return _as_float(audio * curve[None, :])


def compressor(audio: np.ndarray, sample_rate: int, threshold_db: float, ratio: float,
               attack_ms: float, release_ms: float) -> np.ndarray:
    """Turn down levels above threshold_db by ratio:1."""
    if audio.size == 0:
        return _as_float(audio)
    peaks, block = _block_peaks(audio, sample_rate)
    envelope_db = 20.0 * np.log10(_follow(peaks, block / sample_rate, attack_ms, release_ms) + _EPS)
    over = np.maximum(envelope_db - float(threshold_db), 0.0)
    reduction_db = -over * (1.0 - 1.0 / max(float(ratio), 1.0))
    return _apply_block_gain(audio, _db_to_gain(reduction_db), block)


def noise_gate(audio: np.ndarray, sample_rate: int, threshold_db: float, ratio: float,
               attack_ms: float, release_ms: float) -> np.ndarray:
    """Turn down levels below threshold_db, ratio:1 downward expansion (a gate when the ratio is high)."""
    if audio.size == 0:
        return _as_float(audio)
    peaks, block = _block_peaks(audio, sample_rate)
    envelope_db = 20.0 * np.log10(_follow(peaks, block / sample_rate, attack_ms, release_ms) + _EPS)
    under = np.maximum(float(threshold_db) - envelope_db, 0.0)
    reduction_db = np.maximum(-under * (max(float(ratio), 1.0) - 1.0), -100.0)
    return _apply_block_gain(audio, _db_to_gain(reduction_db), block)


def limiter(audio: np.ndarray, sample_rate: int, threshold_db: float, release_ms: float) -> np.ndarray:
    """Keep peaks at or below threshold_db: a short look-ahead, smooth release, then a hard ceiling."""
    if audio.size == 0:
        return _as_float(audio)
    ceiling = float(_db_to_gain(threshold_db))
    peaks, block = _block_peaks(audio, sample_rate)
    needed = np.minimum(1.0, ceiling / np.maximum(peaks, _EPS))
    # look ahead about 3 ms so the gain is already down when a peak arrives
    lookahead = 3
    needed = minimum_filter1d(needed, size=2 * lookahead + 1, origin=-lookahead, mode="nearest")
    release = math.exp(-(block / sample_rate) / max(release_ms / 1000.0, block / sample_rate))
    gains = np.empty_like(needed)
    current = 1.0
    # this loop runs once per millisecond: instant attack, exponential release toward unity
    for index, want in enumerate(needed.tolist()):
        current = want if want < current else release * current + (1.0 - release) * want
        gains[index] = current
    limited = _apply_block_gain(audio, gains, block)
    return _as_float(np.clip(limited, -ceiling, ceiling))


# ---- pitch -------------------------------------------------------------------------------------------------
_N_FFT = 2048
_HOP = 512


def _stft(channel: np.ndarray) -> np.ndarray:
    padded = np.pad(channel, (_N_FFT // 2, _N_FFT // 2), mode="reflect" if channel.size > _N_FFT // 2 else "constant")
    frame_count = 1 + max(0, (padded.size - _N_FFT) // _HOP)
    padded = np.pad(padded, (0, max(0, (frame_count - 1) * _HOP + _N_FFT - padded.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, _N_FFT)[::_HOP][:frame_count]
    return np.fft.rfft(frames * signal.windows.hann(_N_FFT, sym=False), axis=-1).T   # (bins, frames)


def _istft(spectrum: np.ndarray, length: int) -> np.ndarray:
    window = signal.windows.hann(_N_FFT, sym=False)
    frames = np.fft.irfft(spectrum.T, n=_N_FFT, axis=-1) * window
    total = _N_FFT + _HOP * (frames.shape[0] - 1)
    output = np.zeros(total)
    weight = np.zeros(total)
    # overlap-add: one addition per frame, normalized by the summed window energy
    for index, frame in enumerate(frames):
        start = index * _HOP
        output[start:start + _N_FFT] += frame
        weight[start:start + _N_FFT] += window ** 2
    # Normalize by the window energy, but never by less than a tenth of its steady value: at the
    # very ends only one tapered window covers the signal, and a phase-vocoder frame is not tapered
    # itself, so dividing by a near-zero sum there would turn the last milliseconds into spikes.
    output /= np.maximum(weight, 0.1 * float(weight.max()))
    output = output[_N_FFT // 2:]
    return _fit_length(output, length)


def _time_stretch(channel: np.ndarray, stretch: float) -> np.ndarray:
    """Make the sound `stretch` times longer without changing its pitch.

    Phase vocoder with identity phase locking (Laroche and Dolson): each spectral peak carries its
    phase forward in time at its measured frequency, and the bins around a peak keep the phase
    offset they had in the analysis. A plain vocoder advances every bin on its own, which loses
    the relation between neighbouring bins of one partial; they then partly cancel each other
    (a tone came out 15 dB quieter) and the result sounds "phasey".
    """
    spectrum = _stft(channel)
    bins, frame_count = spectrum.shape
    if frame_count < 2:
        return channel.copy()
    steps = np.arange(0.0, frame_count - 1, 1.0 / stretch)
    left = np.floor(steps).astype(int)
    fraction = (steps - left)[None, :]
    magnitude = (1.0 - fraction) * np.abs(spectrum[:, left]) + fraction * np.abs(spectrum[:, left + 1])
    analysis_phase = np.angle(spectrum)
    expected = np.linspace(0.0, math.pi * _HOP, bins)
    delta = analysis_phase[:, 1:] - analysis_phase[:, :-1] - expected[:, None]
    delta -= 2.0 * math.pi * np.round(delta / (2.0 * math.pi))
    advance = expected[:, None] + delta            # true phase advance per hop, per bin and frame
    synthesis_phase = np.empty((bins, steps.size))
    running = analysis_phase[:, left[0]].copy()
    positions = np.arange(bins)
    # this loop runs once per output frame; each step is a handful of vector operations on the bins
    for index, frame in enumerate(left.tolist()):
        level = magnitude[:, index]
        peaks = np.flatnonzero((level[1:-1] >= level[:-2]) & (level[1:-1] >= level[2:])) + 1
        if peaks.size == 0:
            peaks = np.array([int(np.argmax(level))])
        # every bin belongs to its nearest peak (boundaries halfway between peaks)
        owner = peaks[np.searchsorted((peaks[:-1] + peaks[1:]) / 2.0, positions)]
        locked = running[owner] + (analysis_phase[:, frame] - analysis_phase[owner, frame])
        synthesis_phase[:, index] = locked
        running = locked + advance[:, frame]
    return _istft(magnitude * np.exp(1j * synthesis_phase), int(round(channel.size * stretch)))


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Raise or lower the pitch by semitones while keeping the exact length."""
    del sample_rate  # the shift is a ratio; the rate does not change
    audio = _as_float(audio)
    if abs(semitones) < 1e-6 or audio.shape[-1] == 0:
        return audio
    ratio = 2.0 ** (float(semitones) / 12.0)
    frames = audio.shape[-1]
    fraction = Fraction(ratio).limit_denominator(1000)
    shifted = []
    # each channel: stretch by the pitch ratio, then resample back to the original duration
    for channel in audio.astype(np.float64):
        stretched = _time_stretch(channel, ratio)
        resampled = signal.resample_poly(stretched, fraction.denominator, fraction.numerator)
        shifted.append(_fit_length(resampled, frames))
    return _as_float(np.vstack(shifted))


# ---- sample-rate conversion ------------------------------------------------------------------------------
def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase sample-rate conversion (replaces librosa's resampler, which pulls in LGPL soxr)."""
    if source_rate == target_rate or audio.shape[-1] == 0:
        return _as_float(audio)
    fraction = Fraction(int(target_rate), int(source_rate))
    converted = signal.resample_poly(audio, fraction.numerator, fraction.denominator, axis=-1)
    expected = int(round(audio.shape[-1] * target_rate / source_rate))
    return _as_float(_fit_length(converted, expected))
