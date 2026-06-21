from agastya.stages.evidence.audit import verify_chain
from agastya.stages.evidence.binding import verify_binding
from agastya.stages.evidence.credential import verify_credential
from agastya.stages.evidence.record import build_evidence_bundle
from agastya.types import PlateReading, ViolationRecord

_KEY = b"camera-signing-key"


def _record() -> ViolationRecord:
    return ViolationRecord(
        violation_type="no-helmet",
        confidence=0.82,
        plate=PlateReading(text="KA01AB1234", confidence=0.77),
        metadata={"camera_id": "CAM-7", "timestamp": "2026-06-19T10:00:00Z"},
    )


def test_bundle_credential_verifies():
    bundle = build_evidence_bundle(_record(), b"jpeg-bytes", {"detector": "yolo26-m@0.1"}, _KEY)
    assert verify_credential(bundle["credential"], _KEY) is True


def test_bundle_binds_pixels():
    bundle = build_evidence_bundle(_record(), b"jpeg-bytes", {}, _KEY)
    assert verify_binding(bundle["credential"]["manifest"], b"jpeg-bytes") is True


def test_bundle_audit_chain_is_valid_and_continues_prior():
    first = build_evidence_bundle(_record(), b"img-a", {}, _KEY)
    prior = tuple(first["audit_chain"])
    second = build_evidence_bundle(_record(), b"img-b", {}, _KEY, prior_audit=prior)
    chain = tuple(second["audit_chain"])
    assert len(chain) == 2
    assert verify_chain(chain) is True


def test_bundle_manifest_carries_standards():
    bundle = build_evidence_bundle(_record(), b"img", {}, _KEY)
    manifest = bundle["credential"]["manifest"]
    assert any(item["id"] == "ISO/IEC 27037:2012" for item in manifest["standards"])
