# Sample Map Lab
An experimental tool for analyzing MIDI-addressable and otherwise addressable
hardware sample maps. It does not assume that every device identifies samples
by MIDI note number.

## Status

The first experiment targets the BOSS RC-500. Milestone 1 provides WAV loading, onset segmentation,
event-count validation, per-note WAV export and a segmentation diagnostic image.
Later milestones (features, similarity, clustering, report, comparison) are not yet implemented.

Hardware testing suggests that MIDI notes 25..88 trigger sounds; this is an
empirical observation, not a documented BOSS sample map. MIDI-note addressing
is the model used by the current RC-500 experiment, not a general sample identity.
Note numbers are assigned purely by order of detected events.

## Install

    pip install -r requirements.txt && pip install -e .

## Run

    python -m sample_map_lab analyze recordings/rc500.wav --first-note 25 --last-note 88

Outputs: `samples/midi_025.wav` … (original sample rate/channels),
`output/segmentation.png`, and `output/segments.csv` with addresses, sample
boundaries and output filenames. End sample indices are exclusive; filenames
are relative to the samples directory.
If the detected event count differs from `last-note - first-note + 1`, nothing is exported
and the command exits non-zero; tune `--onset-threshold`, `--min-spacing`, `--pre-onset-padding`,
`--max-duration`, `--silence-threshold-db`. Each event's end is found from the energy envelope
(bounded by the next onset and `--max-duration`).
Before each analysis, the analyzer removes the prior `output/segments.csv` and
reserved `samples/midi_000.wav` through `samples/midi_127.wav` files in the
specified directories. Keep other files under different names. A failed
event-count validation still writes the diagnostic image but no mapping outputs.
If cleanup fails, the command reports an error; prior artifacts may remain.

The default segmentation parameters have only been tested against synthetic
recordings and have not yet been validated against a real RC-500 recording.

## Recording protocol

Send notes sequentially from first to last, same velocity and note length, enough silence for decays,
identical RC-500 levels/settings, rhythm reverb off, neutral tone controls, no clipping at the interface.

## Tests

    python -m pytest
