from __future__ import annotations

from agastya.schema.classes import name_to_id

_IDD_MAP: dict[str, str] = {
    "motorcycle": "motorcycle",
    "rider": "rider",
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "autorickshaw": "auto-rickshaw",
}

_AICITY_MAP: dict[str, str] = {
    "motorbike": "motorcycle",
    "DHelmet": "helmet",
    "DNoHelmet": "no-helmet",
    "P1Helmet": "helmet",
    "P1NoHelmet": "no-helmet",
    "P2Helmet": "helmet",
    "P2NoHelmet": "no-helmet",
    "P0Helmet": "helmet",
    "P0NoHelmet": "no-helmet",
}

_ANPR_MAP: dict[str, str] = {
    "license-plate": "license-plate",
    "licence_plate": "license-plate",
    "plate": "license-plate",
}

_SOURCES: dict[str, dict[str, str]] = {
    "idd": _IDD_MAP,
    "aicity": _AICITY_MAP,
    "anpr": _ANPR_MAP,
}


def map_source_label(source: str, name: str) -> int | None:
    if source not in _SOURCES:
        raise ValueError(f"unknown dataset source: {source}")
    unified = _SOURCES[source].get(name)
    if unified is None:
        return None
    return name_to_id(unified)
