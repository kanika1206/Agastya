from __future__ import annotations

import re

from agastya.types import PlateReading

_SEPARATORS = re.compile(r"[\s\-]+")
_PLATE_PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")


def normalize_plate(text: str) -> str:
    return _SEPARATORS.sub("", text).upper()


def is_valid_plate(text: str) -> bool:
    return _PLATE_PATTERN.match(normalize_plate(text)) is not None


def guard_reading(text: str, confidence: float, min_confidence: float) -> PlateReading:
    normalized = normalize_plate(text)
    abstained = confidence < min_confidence or not is_valid_plate(normalized)
    return PlateReading(text=normalized, confidence=confidence, abstained=abstained)
