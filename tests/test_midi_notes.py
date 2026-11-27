import pytest

from rc500_analyzer.midi_notes import assign_notes, expected_event_count, note_name


def test_expected_count():
    assert expected_event_count(25, 88) == 64
    assert expected_event_count(60, 60) == 1


def test_bad_range():
    with pytest.raises(ValueError):
        expected_event_count(88, 25)
    with pytest.raises(ValueError):
        expected_event_count(-1, 5)


def test_sequential_assignment():
    notes = assign_notes(64, 25, 88)
    assert notes[0] == 25 and notes[-1] == 88 and len(notes) == 64


def test_assignment_count_mismatch():
    with pytest.raises(ValueError):
        assign_notes(63, 25, 88)


def test_note_names():
    assert note_name(60) == "C4"
    assert note_name(60, "C3") == "C3"
    assert note_name(60, "C5") == "C5"
    assert note_name(25) == "C#1"
    assert note_name(0) == "C-1"
