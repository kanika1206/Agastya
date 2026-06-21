from __future__ import annotations

from collections.abc import Iterable

from agastya.types import BBox, Detection

TRIPLE_RIDING_MIN = 3


def count_riders(motorcycle: BBox, persons: Iterable[Detection], min_overlap: float) -> int:
    count = 0
    for person in persons:
        person_area = person.box.area()
        if person_area <= 0.0:
            continue
        inter = motorcycle.intersection(person.box)
        if inter / person_area >= min_overlap:
            count += 1
    return count


def is_triple_riding(motorcycle: BBox, persons: Iterable[Detection], min_overlap: float) -> bool:
    return count_riders(motorcycle, persons, min_overlap) >= TRIPLE_RIDING_MIN
