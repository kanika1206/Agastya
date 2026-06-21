from agastya.stages.evidence.manifest import build_manifest
from agastya.types import PlateReading, ViolationRecord


def _record() -> ViolationRecord:
    return ViolationRecord(
        violation_type="no-helmet",
        confidence=0.82,
        plate=PlateReading(text="KA01AB1234", confidence=0.77),
        metadata={"camera_id": "CAM-7", "timestamp": "2026-06-16T10:00:00Z"},
    )


def test_manifest_contains_core_fields():
    manifest = build_manifest(_record(), model_versions={"detector": "yolo26-m@0.1"})
    assert manifest["violation_type"] == "no-helmet"
    assert manifest["confidence"] == 0.82
    assert manifest["plate"] == "KA01AB1234"
    assert manifest["camera_id"] == "CAM-7"
    assert manifest["model_versions"]["detector"] == "yolo26-m@0.1"


def test_manifest_is_json_serializable_and_deterministic():
    import json

    a = build_manifest(_record(), model_versions={"detector": "yolo26-m@0.1"})
    b = build_manifest(_record(), model_versions={"detector": "yolo26-m@0.1"})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_abstained_plate_is_null():
    record = ViolationRecord(
        violation_type="no-helmet",
        confidence=0.6,
        plate=PlateReading(text="", confidence=0.1, abstained=True),
    )
    manifest = build_manifest(record, model_versions={})
    assert manifest["plate"] is None
