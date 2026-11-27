"""WAV loading and validation. The source recording is never modified."""

from dataclasses import dataclass

import numpy as np
import soundfile as sf

CLIP_LEVEL = 0.999


@dataclass
class Recording:
    data: np.ndarray  # shape (n_samples, n_channels), float
    sample_rate: int

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.data.shape[0] / self.sample_rate

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.data))) if self.data.size else 0.0

    @property
    def clipped_samples(self) -> int:
        return int(np.sum(np.abs(self.data) >= CLIP_LEVEL))

    @property
    def is_clipped(self) -> bool:
        return self.clipped_samples > 0

    def mono(self) -> np.ndarray:
        return to_mono(self.data)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Deterministic downmix: arithmetic mean of channels."""
    data = np.asarray(data)
    if data.ndim == 1:
        return data.astype(np.float64)
    return data.mean(axis=1, dtype=np.float64)


def load_wav(path) -> Recording:
    """Load at the native sample rate, preserving channels."""
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError(f"{path} contains no audio")
    return Recording(data=data, sample_rate=int(sr))


def describe(rec: Recording) -> dict:
    return {
        "sample_rate": rec.sample_rate,
        "channels": rec.channels,
        "duration_seconds": rec.duration,
        "peak": rec.peak,
        "clipping_detected": rec.is_clipped,
        "clipped_samples": rec.clipped_samples,
    }
