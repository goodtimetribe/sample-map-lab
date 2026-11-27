import numpy as np
import soundfile as sf

from rc500_analyzer.audio import Recording, load_wav
from rc500_analyzer.cli import main
from rc500_analyzer.segmentation import SegmentationSettings, export_events, segment

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
    events, diag = segment(rec, 25, 32, SegmentationSettings())
    assert diag["count_ok"] and len(events) == 8
    assert all(e.end > e.onset for e in events)
    assert all(a.end <= b.start + 1 for a, b in zip(events, events[1:]))


def test_segment_mismatch_reported():
    rec = Recording(synth(7), SR)
    _, diag = segment(rec, 25, 32, SegmentationSettings())
    assert not diag["count_ok"] and diag["detected_events"] == 7


def test_silence_gives_no_events():
    rec = Recording(np.zeros((SR, 1)), SR)
    events, diag = segment(rec, 25, 26, SegmentationSettings())
    assert events == [] and not diag["count_ok"]


def test_export_preserves_stereo_and_rate(tmp_path):
    rec = Recording(synth(4), SR)
    events, _ = segment(rec, 25, 28, SegmentationSettings())
    paths = export_events(rec, events, 25, 28, tmp_path)
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
    assert load_wav(wav).channels == 2
    assert main(["analyze", str(wav), "--first-note", "25", "--last-note", "30",
                 "--samples-dir", str(tmp_path / "s2"), "--output-dir", str(tmp_path / "o")]) == 1
    assert not (tmp_path / "s2").exists()
    assert "Expected events: 6" in capsys.readouterr().out
