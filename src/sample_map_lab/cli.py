"""Command line interface: `python -m sample_map_lab analyze recording.wav ...`"""

import argparse
import sys
from pathlib import Path

from .audio import describe, load_wav
from .midi_notes import midi_addresses
from .reporting import plot_segmentation, write_segments_manifest
from .segmentation import SegmentationSettings, export_events, segment


def build_parser() -> argparse.ArgumentParser:
    d = SegmentationSettings()
    p = argparse.ArgumentParser(prog="sample_map_lab")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="segment a recording and export one WAV per MIDI note")
    a.add_argument("wav")
    a.add_argument("--first-note", type=int, default=25)
    a.add_argument("--last-note", type=int, default=88)
    a.add_argument("--samples-dir", default="samples")
    a.add_argument("--output-dir", default="output")
    a.add_argument("--onset-threshold", type=float, default=d.onset_threshold)
    a.add_argument("--min-spacing", type=float, default=d.min_spacing, help="seconds between events")
    a.add_argument("--pre-onset-padding", type=float, default=d.pre_onset_padding, help="seconds")
    a.add_argument("--max-duration", type=float, default=d.max_duration, help="seconds, tail safety limit")
    a.add_argument("--silence-threshold-db", type=float, default=d.silence_threshold_db)
    return p


def clear_mapping_outputs(samples_dir, output_dir, input_path):
    """Remove only the filenames reserved for this analyzer's MIDI exports."""
    samples = Path(samples_dir)
    candidates = [Path(output_dir) / "segments.csv"]
    candidates.extend(samples / f"midi_{note:03d}.wav" for note in range(128))
    existing = [p for p in candidates if p.is_file() or p.is_symlink()]
    if any(p.resolve() == input_path.resolve() for p in existing):
        raise ValueError("input recording conflicts with a reserved output path; use separate directories")
    for candidate in existing:
        candidate.unlink()


def analyze(args) -> int:
    path = Path(args.wav)
    try:
        clear_mapping_outputs(args.samples_dir, args.output_dir, path)
    except (OSError, ValueError) as exc:
        print(f"error: cannot clear previous mapping outputs: {exc}", file=sys.stderr)
        return 2
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    try:
        rec = load_wav(path)
    except Exception as exc:  # unreadable / not audio
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    info = describe(rec)
    print(f"Sample rate: {info['sample_rate']} Hz")
    print(f"Channels: {info['channels']}")
    print(f"Duration: {info['duration_seconds']:.2f} s")
    print(f"Peak amplitude: {info['peak']:.4f}")
    if info["clipping_detected"]:
        print(f"WARNING: clipping detected ({info['clipped_samples']} samples at full scale)")

    settings = SegmentationSettings(
        onset_threshold=args.onset_threshold, min_spacing=args.min_spacing,
        pre_onset_padding=args.pre_onset_padding, max_duration=args.max_duration,
        silence_threshold_db=args.silence_threshold_db,
    )
    try:
        addresses = midi_addresses(args.first_note, args.last_note)
        events, diag = segment(rec, addresses, settings)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Expected events: {diag['expected_events']}")
    print(f"Detected events: {diag['detected_events']}")
    for w in diag["warnings"]:
        print(f"WARNING: {w}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_segmentation(rec, events, addresses, out / "segmentation.png")
    print(f"Segmentation diagnostic: {out / 'segmentation.png'}")

    if not diag["count_ok"]:
        print("ERROR: event count mismatch; not exporting samples to avoid assigning wrong MIDI numbers.\n"
              "Inspect the diagnostic image and adjust --onset-threshold / --min-spacing.", file=sys.stderr)
        return 1
    filenames = [f"midi_{address.midi_note:03d}.wav" for address in addresses]
    try:
        paths = export_events(rec, events, filenames, args.samples_dir)
        write_segments_manifest(rec, events, addresses, paths, out / "segments.csv")
    except (OSError, RuntimeError) as exc:
        print(f"error: cannot export mapping: {exc}", file=sys.stderr)
        try:
            clear_mapping_outputs(args.samples_dir, args.output_dir, path)
        except (OSError, ValueError) as cleanup_exc:
            print(f"error: cannot clear incomplete mapping outputs: {cleanup_exc}", file=sys.stderr)
        return 2
    print(f"Exported {len(paths)} samples to {args.samples_dir}/")
    print(f"Segmentation manifest: {out / 'segments.csv'}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        return analyze(args)
    return 2
