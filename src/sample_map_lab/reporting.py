"""Diagnostic plots (Milestone 1: segmentation diagnostic only)."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_segmentation(rec, events, addresses, path):
    """Label onsets with addresses only when the detected count matches."""
    mono = rec.mono()
    t = np.arange(len(mono)) / rec.sample_rate
    fig, ax = plt.subplots(figsize=(20, 4), dpi=80)
    ax.plot(t, mono, lw=0.4, color="0.2")
    if len(events) == len(addresses):
        labels = [address.display_label for address in addresses]
    else:
        labels = [f"#{i}" for i in range(len(events))]
        ax.set_title("Event count mismatch: labels are event indices, NOT addresses", color="red")
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


def write_segments_manifest(rec, events, addresses, paths, path):
    """Write the exact source sample boundaries used for the exported WAVs."""
    if not len(events) == len(addresses) == len(paths):
        raise ValueError("Event, address and output counts must match")
    fields = [
        "event_index", "address_type", "address_value", "display_label",
        "midi_note", "note_name", "onset_sample", "onset_seconds",
        "start_sample", "end_sample", "start_seconds", "end_seconds",
        "duration_seconds", "output_filename",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, (event, address, output) in enumerate(zip(events, addresses, paths)):
            writer.writerow({
                "event_index": i, "address_type": address.address_type,
                "address_value": address.address_value, "display_label": address.display_label,
                "midi_note": address.midi_note, "note_name": address.note_name,
                "onset_sample": event.onset, "onset_seconds": event.onset / rec.sample_rate,
                "start_sample": event.start, "end_sample": event.end,
                "start_seconds": event.start / rec.sample_rate,
                "end_seconds": event.end / rec.sample_rate,
                "duration_seconds": (event.end - event.start) / rec.sample_rate,
                "output_filename": output.name,
            })
