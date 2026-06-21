from agastya.stages.evidence.binding import bind_content, content_hash, verify_binding


def test_content_hash_is_deterministic():
    assert content_hash(b"pixels") == content_hash(b"pixels")


def test_content_hash_changes_with_pixels():
    assert content_hash(b"pixels") != content_hash(b"pixelt")


def test_bind_content_adds_hash_and_root():
    bound = bind_content({"violation_type": "no-helmet"}, b"img")
    assert bound["content_hash"] == content_hash(b"img")
    assert "evidence_root" in bound
    assert bound["violation_type"] == "no-helmet"


def test_verify_binding_accepts_untampered():
    bound = bind_content({"violation_type": "no-helmet", "confidence": 0.9}, b"img")
    assert verify_binding(bound, b"img") is True


def test_verify_binding_rejects_pixel_tamper():
    bound = bind_content({"violation_type": "no-helmet"}, b"img")
    assert verify_binding(bound, b"tampered") is False


def test_verify_binding_rejects_manifest_tamper():
    bound = bind_content({"violation_type": "no-helmet"}, b"img")
    bound["violation_type"] = "triple-riding"
    assert verify_binding(bound, b"img") is False
