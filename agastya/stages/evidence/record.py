from __future__ import annotations

from collections.abc import Sequence

from agastya.stages.evidence.audit import AuditEntry, append
from agastya.stages.evidence.binding import bind_content
from agastya.stages.evidence.credential import sign_manifest
from agastya.stages.evidence.manifest import build_manifest
from agastya.stages.evidence.standards import attach_standards
from agastya.types import ViolationRecord


def build_evidence_bundle(
    record: ViolationRecord,
    image_bytes: bytes,
    model_versions: dict[str, str],
    signing_key: bytes,
    prior_audit: Sequence[AuditEntry] = (),
) -> dict:
    manifest = build_manifest(record, model_versions)
    manifest = attach_standards(manifest)
    manifest = bind_content(manifest, image_bytes)
    credential = sign_manifest(manifest, signing_key)
    audit = append(prior_audit, "evidence_created", {"content_hash": manifest["content_hash"]})
    return {
        "credential": credential,
        "content_hash": manifest["content_hash"],
        "evidence_root": manifest["evidence_root"],
        "audit_chain": list(audit),
    }
