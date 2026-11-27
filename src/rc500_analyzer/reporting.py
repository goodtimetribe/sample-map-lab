"""Diagnostic plots (Milestone 1: segmentation diagnostic only)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .midi_notes import assign_notes


def plot_segmentation(rec, events, first_note, last_note, path, note_labels=True):
    """Waveform with onsets labelled by MIDI number (if counts match) or by event index."""
    mono = rec.mono()
    t = np.arange(len(mono)) / rec.sample_rate
    fig, ax = plt.subplots(figsize=(20, 4), dpi=80)
    ax.plot(t, mono, lw=0.4, color="0.2")
    try:
        labels = [str(n) for n in assign_notes(len(events), first_note, last_note)]
    except ValueError:
        labels = [f"#{i}" for i in range(len(events))]
        ax.set_title("Event count mismatch: labels are event indices, NOT MIDI notes", color="red")
    top = max(float(np.max(np.abs(mono))), 1e-9)
    for i, (e, lab) in enumerate(zip(events, labels)):
        x = e.onset / rec.sample_rate
        ax.axvline(x, color="tab:red", lw=0.6, alpha=0.7)
        ax.text(x, top * (0.95 if i % 2 == 0 else 0.8), lab, fontsize=6, rotation=90, va="top", ha="right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mono mix)")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
