import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agastya.api.app import create_app  # noqa: E402
from agastya.stages.evidence.record import build_evidence_bundle  # noqa: E402
from agastya.store.sqlite_store import ViolationStore  # noqa: E402
from agastya.types import PlateReading, ViolationRecord  # noqa: E402

_KEY = b"camera-signing-key"


def _bundle(violation_type="no-helmet", camera_id="CAM-7", pixels=b"img"):
    record = ViolationRecord(
        violation_type=violation_type,
        confidence=0.82,
        plate=PlateReading(text="KA01AB1234", confidence=0.77),
        metadata={"camera_id": camera_id, "timestamp": "2026-06-19T10:00:00Z"},
    )
    return build_evidence_bundle(record, pixels, {"detector": "yolo26-m@0.1"}, _KEY)


@pytest.fixture
def client():
    store = ViolationStore(":memory:")
    store.save(_bundle(violation_type="no-helmet", pixels=b"a"))
    store.save(_bundle(violation_type="triple-riding", pixels=b"b"))
    app = create_app(store, signing_key=_KEY)
    yield TestClient(app)
    store.close()


def test_list_violations_envelope(client):
    response = client.get("/violations")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["meta"]["total"] == 2
    assert len(body["data"]) == 2


def test_list_filter_by_type(client):
    body = client.get("/violations", params={"violation_type": "triple-riding"}).json()
    assert body["meta"]["total"] == 1
    assert body["data"][0]["violation_type"] == "triple-riding"


def test_stats_endpoint(client):
    body = client.get("/stats").json()
    assert body["data"]["total"] == 2
    assert body["data"]["by_type"]["no-helmet"] == 1


def test_get_violation_bundle(client):
    body = client.get("/violations/1").json()
    assert body["data"]["credential"]["alg"] == "hmac-sha256"


def test_get_missing_returns_404(client):
    assert client.get("/violations/999").status_code == 404


def test_verify_valid_evidence(client):
    body = client.get("/violations/1/verify").json()
    assert body["data"]["audit_chain_valid"] is True
    assert body["data"]["credential_signature_valid"] is True
    assert body["data"]["signing_key_configured"] is True


def test_list_filter_by_plate(client):
    body = client.get("/violations", params={"plate": "01AB"}).json()
    assert body["meta"]["total"] == 2


def test_list_empty_plate_param_ignored(client):
    body = client.get("/violations", params={"plate": ""}).json()
    assert body["meta"]["total"] == 2


def test_list_sort_oldest(client):
    body = client.get("/violations", params={"sort": "oldest"}).json()
    assert body["meta"]["sort"] == "oldest"
    assert body["data"][0]["id"] < body["data"][-1]["id"]


def test_list_rejects_bad_sort(client):
    assert client.get("/violations", params={"sort": "sideways"}).status_code == 422


def test_cors_header_present(client):
    response = client.get("/violations", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_image_404_without_evidence(client):
    assert client.get("/violations/1/image").status_code == 404


def test_image_endpoint_serves_relative_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AGASTYA_EVIDENCE_ROOT", str(tmp_path))
    evidence_dir = tmp_path / "web" / "assets" / "evidence"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    store = ViolationStore(":memory:")
    vid = store.save(_bundle(pixels=b"rel"))
    store.set_image_path(vid, "web/assets/evidence/1.png")
    client = TestClient(create_app(store, signing_key=_KEY))
    response = client.get(f"/violations/{vid}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    store.close()


def test_image_endpoint_serves_jpeg(tmp_path):
    import cv2
    import numpy as np

    from agastya.config import PipelineConfig, VRAMProfile
    from agastya.ingest.runner import IngestRunner
    from agastya.pipeline import Pipeline
    from agastya.stages.stubs import StubDetector, StubGate, StubOCR, StubRestorer

    store = ViolationStore(":memory:")
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    runner = IngestRunner(
        pipeline,
        store,
        signing_key=_KEY,
        model_versions={"detector": "stub@0.1"},
        evidence_dir=str(tmp_path),
    )
    pixels = cv2.imencode(".jpg", (np.random.rand(48, 64, 3) * 255).astype(np.uint8))[1].tobytes()
    result = runner.ingest_image("img-1", pixels)
    client = TestClient(create_app(store, signing_key=_KEY))
    response = client.get(f"/violations/{result.violation_ids[0]}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    store.close()
