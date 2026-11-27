import csv
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

from sample_map_lab.addresses import SampleAddress
from sample_map_lab.audio import Recording, load_wav
from sample_map_lab.cli import main
from sample_map_lab.midi_notes import midi_addresses
from sample_map_lab.segmentation import SegmentationSettings, export_events, segment

SR = 22050


def synth(n_hits, lead=0.3, gap=0.6, stereo=True, start_at_zero=False):
    rng = np.random.default_rng(0)
    total = int((lead + gap * n_hits + 0.5) * SR)
    sig = np.zeros(total)
    for i in range(n_hits):
        on = 0 if (start_at_zero and i == 0) else int((lead + gap * i) * SR)
        n = int(0.15 * SR)
        t = np.arange(n) / SR
        hit = np.exp(-t * 30) * (rng.standard_normal(n) * 0.5 + np.sin(2 * np.pi * (100 + 20 * i) * t))
        sig[on:on + n] += 0.4 * hit / np.abs(hit).max()
    return np.stack([sig, sig * 0.5], axis=1) if stereo else sig[:, None]


def test_segment_expected_count():
    rec = Recording(synth(8), SR)
    events, diag = segment(rec, midi_addresses(25, 32), SegmentationSettings())
    assert diag["count_ok"] and len(events) == 8
    assert all(e.end > e.onset for e in events)
    assert all(a.end <= b.start + 1 for a, b in zip(events, events[1:]))


def test_segment_mismatch_reported():
    rec = Recording(synth(7), SR)
    _, diag = segment(rec, midi_addresses(25, 32), SegmentationSettings())
    assert not diag["count_ok"] and diag["detected_events"] == 7


def test_silence_gives_no_events():
    rec = Recording(np.zeros((SR, 1)), SR)
    events, diag = segment(rec, midi_addresses(25, 26), SegmentationSettings())
    assert events == [] and not diag["count_ok"]


def test_export_preserves_stereo_and_rate(tmp_path):
    rec = Recording(synth(4), SR)
    events, _ = segment(rec, midi_addresses(25, 28), SegmentationSettings())
    paths = export_events(rec, events, [f"midi_{n:03d}.wav" for n in range(25, 29)], tmp_path)
    assert [p.name for p in paths] == [f"midi_0{n}.wav" for n in range(25, 29)]
    data, sr = sf.read(paths[0])
    assert sr == SR and data.ndim == 2 and data.shape[1] == 2


def test_clipping_detected():
    data = synth(2)
    data[100, 0] = 1.0
    assert Recording(data, SR).is_clipped
    assert not Recording(synth(2), SR).is_clipped


def test_cli_end_to_end(tmp_path, capsys):
    wav = tmp_path / "r.wav"
    sf.write(str(wav), synth(5), SR, subtype="FLOAT")
    args = [str(wav), "--first-note", "25", "--last-note", "29",
            "--samples-dir", str(tmp_path / "s"), "--output-dir", str(tmp_path / "o")]
    assert main(["analyze", *args]) == 0
    assert (tmp_path / "s" / "midi_029.wav").exists()
    assert (tmp_path / "o" / "segmentation.png").exists()
    assert (tmp_path / "o" / "segments.csv").exists()
    assert load_wav(wav).channels == 2
    assert main(["analyze", str(wav), "--first-note", "25", "--last-note", "30",
                 "--samples-dir", str(tmp_path / "s2"), "--output-dir", str(tmp_path / "o2")]) == 1
    assert not (tmp_path / "s2").exists()
    assert (tmp_path / "o2" / "segmentation.png").exists()
    assert not (tmp_path / "o2" / "segments.csv").exists()
    assert "Expected events: 6" in capsys.readouterr().out


def test_non_midi_addresses_can_drive_segmentation():
    addresses = [SampleAddress(i, "patch", f"patch-{i}", f"Patch {i}") for i in range(2)]
    events, diag = segment(Recording(synth(2), SR), addresses, SegmentationSettings())
    assert len(events) == 2 and diag["count_ok"]


def test_module_cli_manifest_matches_exports(tmp_path):
    wav = tmp_path / "recording.wav"
    sf.write(wav, synth(3), SR, subtype="FLOAT")
    samples, output = tmp_path / "samples", tmp_path / "output"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-m", "sample_map_lab", "analyze", str(wav),
         "--first-note", "25", "--last-note", "27",
         "--samples-dir", str(samples), "--output-dir", str(output)],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    with (output / "segments.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rec = load_wav(wav)
    events, diag = segment(rec, midi_addresses(25, 27), SegmentationSettings())
    assert diag["count_ok"] and len(rows) == len(events) == 3
    for i, (row, event) in enumerate(zip(rows, events)):
        note = 25 + i
        assert row["event_index"] == str(i)
        assert row["address_type"] == "midi_note"
        assert row["address_value"] == row["midi_note"] == str(note)
        assert row["display_label"] == f"MIDI {note}"
        assert row["note_name"] == midi_addresses(25, 27)[i].note_name
        assert int(row["onset_sample"]) == event.onset
        assert int(row["start_sample"]) == event.start
        assert int(row["end_sample"]) == event.end
        assert float(row["onset_seconds"]) == event.onset / SR
        assert float(row["start_seconds"]) == event.start / SR
        assert float(row["end_seconds"]) == event.end / SR
        assert float(row["duration_seconds"]) == (event.end - event.start) / SR
        assert row["output_filename"] == f"midi_{note:03d}.wav"
        exported, rate = sf.read(samples / row["output_filename"], always_2d=True)
        assert rate == SR and exported.shape == (event.end - event.start, 2)
        np.testing.assert_array_equal(exported, rec.data[event.start:event.end])
