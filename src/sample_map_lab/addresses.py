"""Identity of an expected event in an ordered sample map."""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class SampleAddress:
    index: int
    address_type: str
    address_value: Union[int, str]
    display_label: str
    midi_note: Optional[int] = None
    note_name: Optional[str] = None
