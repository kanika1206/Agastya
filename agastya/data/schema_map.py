from __future__ import annotations

from agastya.schema.classes import name_to_id

_TRIPLE_MAP: dict[str, str] = {
    "with_helmet": "helmet",
    "without_helmet": "no-helmet",
    "number_plate": "license-plate",
    "Triple_riding": "triple-riding",
    "motorcycle": "motorcycle",
}

_SAFETY_MAP: dict[str, str] = {
    "helmet": "helmet",
    "no-helmet": "no-helmet",
    "number-plate": "license-plate",
    "bike": "motorcycle",
}

_SOURCES: dict[str, dict[str, str]] = {
    "triple": _TRIPLE_MAP,
    "safety": _SAFETY_MAP,
}


def map_source_label(source: str, name: str) -> int | None:
    if source not in _SOURCES:
        raise ValueError(f"unknown dataset source: {source}")
    unified = _SOURCES[source].get(name)
    if unified is None:
        return None
    return name_to_id(unified)
