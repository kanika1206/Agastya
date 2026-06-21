from __future__ import annotations

from agastya.types import ViolationRecord


def build_manifest(record: ViolationRecord, model_versions: dict[str, str]) -> dict:
    plate_text: str | None = None
    if record.plate is not None and not record.plate.abstained:
        plate_text = record.plate.text
    manifest: dict = {
        "violation_type": record.violation_type,
        "confidence": record.confidence,
        "plate": plate_text,
        "model_versions": dict(model_versions),
    }
    manifest.update(record.metadata)
    return manifest
