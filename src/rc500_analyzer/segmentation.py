"""Onset detection and event segmentation for isolated percussive hits."""

from dataclasses import dataclass, asdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .audio import Recording
from .midi_notes import assign_notes, expected_event_count


@dataclass
class SegmentationSettings:
    onset_threshold: float = 0.3      # librosa peak_pick delta on normalised onset envelope
    min_spacing: float = 0.15         # minimum seconds between events
    pre_onset_padding: float = 0.005  # seconds kept before each onset
    max_duration: float = 5.0         # safety limit on event length (seconds)
    silence_threshold_db: float = -50.0  # tail ends when envelope < peak + this (dB)
    hop_length: int = 128

    def to_dict(self):
        return asdict(self)


@dataclass
class Event:
    onset: int   # sample index of the detected onset
    start: int   # first sample exported
    end: int     # one past last sample exported


def detect_onsets(mono: np.ndarray, sr: int, s: SegmentationSettings) -> np.ndarray:
    """Return onset sample positions (sorted)."""
    if not np.any(mono):
        return np.array([], dtype=int)
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=s.hop_length)
    if env.max() <= 0:
        return np.array([], dtype=int)
    env = env / env.max()
    wait = max(1, int(round(s.min_spacing * sr / s.hop_length)))
    frames = librosa.util.peak_pick(
        env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=s.onset_threshold, wait=wait
    )
    refined = []
    # Refine to sample accuracy: first sample in the neighbourhood above a fraction of its local peak.
    radius = 2 * s.hop_length
    for f in frames:
        centre = int(f) * s.hop_length
        lo, hi = max(0, centre - radius), min(len(mono), centre + radius)
        seg = np.abs(mono[lo:hi])
        if seg.max() <= 0:
            continue
        refined.append(lo + int(np.argmax(seg >= 0.2 * seg.max())))
    return np.array(sorted(set(refined)), dtype=int)


def find_event_end(mono: np.ndarray, onset: int, limit: int, sr: int, s: SegmentationSettings) -> int:
    """End of an event: last point where the smoothed envelope exceeds the silence threshold
    relative to the event peak, bounded by `limit` (next onset / max duration)."""
    seg = np.abs(mono[onset:limit])
    if seg.size == 0:
        return limit
    win = max(1, int(0.005 * sr))
    env = np.convolve(seg, np.ones(win) / win, mode="same")
    peak = env.max()
    if peak <= 0:
        return onset + 1
    above = np.nonzero(env >= peak * 10 ** (s.silence_threshold_db / 20.0))[0]
    return onset + int(above[-1]) + 1


def build_events(mono: np.ndarray, onsets: np.ndarray, sr: int, s: SegmentationSettings) -> list:
    pre = int(round(s.pre_onset_padding * sr))
    max_len = int(round(s.max_duration * sr))
    events = []
    for i, on in enumerate(onsets):
        next_on = int(onsets[i + 1]) - pre if i + 1 < len(onsets) else len(mono)
        limit = min(next_on, on + max_len, len(mono))
        end = find_event_end(mono, int(on), limit, sr, s)
        events.append(Event(onset=int(on), start=max(0, int(on) - pre), end=max(end, int(on) + 1)))
    return events


def segment(rec: Recording, first_note: int, last_note: int, s: SegmentationSettings):
    """Return (events, diagnostics). Does not raise on count mismatch; check `diagnostics['count_ok']`."""
    expected = expected_event_count(first_note, last_note)
    onsets = detect_onsets(rec.mono(), rec.sample_rate, s)
    events = build_events(rec.mono(), onsets, rec.sample_rate, s)
    warnings = []
    for e, nxt in zip(events, events[1:]):
        if nxt.onset - e.onset >= int(s.max_duration * rec.sample_rate):
            warnings.append(f"Event at {e.onset / rec.sample_rate:.2f}s hit the max-duration limit; tail may be truncated")
    for e in events:
        if e.onset < int(0.002 * rec.sample_rate):
            warnings.append("Recording starts immediately with a hit; check the first event")
            break
    diag = {
        "expected_events": expected,
        "detected_events": len(events),
        "count_ok": len(events) == expected,
        "warnings": warnings,
    }
    return events, diag


def export_events(rec: Recording, events, first_note: int, last_note: int, out_dir) -> list:
    """Write midi_NNN.wav files (original channels and sample rate). Requires an exact count."""
    notes = assign_notes(len(events), first_note, last_note)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for note, e in zip(notes, events):
        p = out_dir / f"midi_{note:03d}.wav"
        sf.write(str(p), rec.data[e.start:e.end], rec.sample_rate, subtype="FLOAT")
        paths.append(p)
    return paths
