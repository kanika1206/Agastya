from __future__ import annotations

import hashlib
import json

from agastya.stages.evidence.merkle import merkle_root

_DERIVED_KEYS = ("content_hash", "evidence_root")


def content_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def canonical_json(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def bind_content(manifest: dict, image_bytes: bytes) -> dict:
    bound = {key: manifest[key] for key in manifest}
    bound["content_hash"] = content_hash(image_bytes)
    bound["evidence_root"] = merkle_root([canonical_json(manifest).encode(), image_bytes])
    return bound


def verify_binding(bound: dict, image_bytes: bytes) -> bool:
    if bound.get("content_hash") != content_hash(image_bytes):
        return False
    base = {key: value for key, value in bound.items() if key not in _DERIVED_KEYS}
    expected = merkle_root([canonical_json(base).encode(), image_bytes])
    return bound.get("evidence_root") == expected
