from __future__ import annotations

import hashlib
import hmac

from agastya.stages.evidence.binding import canonical_json

ALGORITHM = "hmac-sha256"


def sign_manifest(manifest: dict, key: bytes) -> dict:
    signature = hmac.new(key, canonical_json(manifest).encode(), hashlib.sha256).hexdigest()
    return {"manifest": manifest, "signature": signature, "alg": ALGORITHM}


def verify_credential(credential: dict, key: bytes) -> bool:
    if credential.get("alg") != ALGORITHM:
        return False
    manifest = credential.get("manifest")
    if not isinstance(manifest, dict):
        return False
    expected = hmac.new(key, canonical_json(manifest).encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, str(credential.get("signature", "")))
