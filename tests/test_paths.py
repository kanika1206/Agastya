from agastya.paths import resolve_evidence_path, to_relative


def test_to_relative_strips_evidence_root(monkeypatch, tmp_path):
    monkeypatch.setenv("AGASTYA_EVIDENCE_ROOT", str(tmp_path))
    absolute = str(tmp_path / "web" / "assets" / "evidence" / "3.png")
    assert to_relative(absolute) == "web/assets/evidence/3.png"


def test_to_relative_leaves_relative_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("AGASTYA_EVIDENCE_ROOT", str(tmp_path))
    assert to_relative("web/assets/evidence/3.png") == "web/assets/evidence/3.png"


def test_to_relative_leaves_outside_root_absolute(monkeypatch, tmp_path):
    monkeypatch.setenv("AGASTYA_EVIDENCE_ROOT", str(tmp_path / "root"))
    outside = "/somewhere/else/9.png"
    assert to_relative(outside) == outside


def test_resolve_joins_relative_against_root(monkeypatch, tmp_path):
    monkeypatch.setenv("AGASTYA_EVIDENCE_ROOT", str(tmp_path))
    assert resolve_evidence_path("a/b.png") == str(tmp_path / "a" / "b.png")


def test_resolve_keeps_absolute_path(monkeypatch, tmp_path):
    monkeypatch.setenv("AGASTYA_EVIDENCE_ROOT", str(tmp_path))
    absolute = "/abs/c.png"
    assert resolve_evidence_path(absolute) == absolute


def test_resolve_none_returns_none():
    assert resolve_evidence_path(None) is None
