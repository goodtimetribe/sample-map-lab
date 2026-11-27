"""MIDI note helpers. The integer note number is the canonical identifier;
note names are display-only and depend on an octave-numbering convention."""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Octave number shown for MIDI note 0 (C-1 in the common "C4 = 60" convention).
MIDDLE_C_CONVENTIONS = {"C3": -2, "C4": -1, "C5": 0}


def validate_range(first_note: int, last_note: int) -> None:
    if not (0 <= first_note <= 127 and 0 <= last_note <= 127):
        raise ValueError("MIDI notes must be within 0..127")
    if last_note < first_note:
        raise ValueError(f"last note ({last_note}) is lower than first note ({first_note})")


def expected_event_count(first_note: int, last_note: int) -> int:
    validate_range(first_note, last_note)
    return last_note - first_note + 1


def assign_notes(n_events: int, first_note: int, last_note: int) -> list:
    """Sequentially assign MIDI numbers to events; requires an exact count match."""
    expected = expected_event_count(first_note, last_note)
    if n_events != expected:
        raise ValueError(f"Cannot assign notes: expected {expected} events, got {n_events}")
    return list(range(first_note, last_note + 1))


def note_name(note: int, middle_c: str = "C4") -> str:
    """Render a note name. `middle_c` names MIDI 60 ('C3', 'C4' or 'C5')."""
    if not 0 <= note <= 127:
        raise ValueError("MIDI note must be within 0..127")
    if middle_c not in MIDDLE_C_CONVENTIONS:
        raise ValueError(f"Unknown convention {middle_c!r}; use one of {sorted(MIDDLE_C_CONVENTIONS)}")
    octave = note // 12 + MIDDLE_C_CONVENTIONS[middle_c]
    return f"{NOTE_NAMES[note % 12]}{octave}"
