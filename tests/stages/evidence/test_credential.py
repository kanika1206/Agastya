from agastya.stages.evidence.credential import sign_manifest, verify_credential

_KEY = b"camera-signing-key"


def test_sign_then_verify_roundtrip():
    credential = sign_manifest({"violation_type": "no-helmet"}, _KEY)
    assert credential["alg"] == "hmac-sha256"
    assert verify_credential(credential, _KEY) is True


def test_verify_rejects_wrong_key():
    credential = sign_manifest({"violation_type": "no-helmet"}, _KEY)
    assert verify_credential(credential, b"attacker-key") is False


def test_verify_rejects_manifest_tamper():
    credential = sign_manifest({"violation_type": "no-helmet"}, _KEY)
    credential["manifest"]["violation_type"] = "triple-riding"
    assert verify_credential(credential, _KEY) is False


def test_verify_rejects_unknown_alg():
    credential = sign_manifest({"violation_type": "no-helmet"}, _KEY)
    credential["alg"] = "rot13"
    assert verify_credential(credential, _KEY) is False
