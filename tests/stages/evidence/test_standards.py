from agastya.stages.evidence.standards import STANDARDS, attach_standards


def test_standards_reference_iso_27037():
    ids = {item["id"] for item in STANDARDS}
    assert "ISO/IEC 27037:2012" in ids
    assert "NIST SP 800-86" in ids


def test_attach_standards_adds_list_without_mutating_input():
    manifest = {"violation_type": "no-helmet"}
    bound = attach_standards(manifest)
    assert "standards" not in manifest
    assert len(bound["standards"]) == len(STANDARDS)
    assert bound["violation_type"] == "no-helmet"
