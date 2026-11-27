# sample-map-lab
Analyze, compare, and reverse-engineer MIDI-addressable hardware sample maps.

## Status

Milestone 1 of the RC-500 sample analyzer: WAV loading, onset segmentation,
event-count validation, per-note WAV export and a segmentation diagnostic image.
Later milestones (features, similarity, clustering, report, comparison) are not yet implemented.

The hypothesis that MIDI notes 25..88 map to 64 fixed sounds is **experimental**, not documented fact;
this tool exists to test it. Note numbers are assigned purely by order of detected events.

## Install

    pip install -r requirements.txt && pip install -e .

## Run

    python -m rc500_analyzer analyze recordings/rc500.wav --first-note 25 --last-note 88

Outputs: `samples/midi_025.wav` … (original sample rate/channels) and `output/segmentation.png`.
If the detected event count differs from `last-note - first-note + 1`, nothing is exported
and the command exits non-zero; tune `--onset-threshold`, `--min-spacing`, `--pre-onset-padding`,
`--max-duration`, `--silence-threshold-db`. Each event's end is found from the energy envelope
(bounded by the next onset and `--max-duration`).

## Recording protocol

Send notes sequentially from first to last, same velocity and note length, enough silence for decays,
identical RC-500 levels/settings, rhythm reverb off, neutral tone controls, no clipping at the interface.

## Tests

    python -m pytest
