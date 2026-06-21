import pytest

from agastya.stages.evidence.record import build_evidence_bundle
from agastya.store.sqlite_store import ViolationStore, compute_dedup_key
from agastya.types import PlateReading, ViolationRecord

_KEY = b"camera-signing-key"


def _bundle(violation_type: str = "no-helmet", camera_id: str = "CAM-7", pixels: bytes = b"img"):
    record = ViolationRecord(
        violation_type=violation_type,
        confidence=0.82,
        plate=PlateReading(text="KA01AB1234", confidence=0.77),
        metadata={"camera_id": camera_id, "timestamp": "2026-06-19T10:00:00Z"},
    )
    return build_evidence_bundle(record, pixels, {"detector": "yolo26-m@0.1"}, _KEY)


@pytest.fixture
def store():
    store = ViolationStore(":memory:")
    yield store
    store.close()


def test_save_returns_incrementing_ids(store):
    first = store.save(_bundle(pixels=b"a"))
    second = store.save(_bundle(pixels=b"b"))
    assert second == first + 1


def test_list_returns_saved_violation(store):
    store.save(_bundle())
    items, total = store.list()
    assert total == 1
    assert items[0].violation_type == "no-helmet"
    assert items[0].plate == "KA01AB1234"
    assert items[0].camera_id == "CAM-7"


def test_list_filters_by_type(store):
    store.save(_bundle(violation_type="no-helmet", pixels=b"a"))
    store.save(_bundle(violation_type="triple-riding", pixels=b"b"))
    items, total = store.list(violation_type="triple-riding")
    assert total == 1
    assert items[0].violation_type == "triple-riding"


def test_list_paginates(store):
    for index in range(3):
        store.save(_bundle(pixels=bytes([index])), created_at=f"2026-06-19T0{index}:00:00Z")
    page, total = store.list(limit=2, offset=0)
    assert total == 3
    assert len(page) == 2


def test_get_returns_full_bundle(store):
    new_id = store.save(_bundle())
    bundle = store.get(new_id)
    assert bundle["credential"]["alg"] == "hmac-sha256"
    assert isinstance(bundle["audit_chain"], list)
    assert bundle["audit_chain"][0]["seq"] == 0


def test_get_missing_returns_none(store):
    assert store.get(999) is None


def test_dedup_key_save_is_idempotent(store):
    key = compute_dedup_key("hash-1", "no-helmet", [(1.0, 2.0, 3.0, 4.0)])
    first = store.save(_bundle(pixels=b"a"), dedup_key=key)
    second = store.save(_bundle(pixels=b"b"), dedup_key=key)
    assert second == first
    _, total = store.list()
    assert total == 1


def test_dedup_key_differs_by_boxes(store):
    key_a = compute_dedup_key("hash-1", "no-helmet", [(1.0, 2.0, 3.0, 4.0)])
    key_b = compute_dedup_key("hash-1", "no-helmet", [(5.0, 6.0, 7.0, 8.0)])
    store.save(_bundle(pixels=b"a"), dedup_key=key_a)
    store.save(_bundle(pixels=b"b"), dedup_key=key_b)
    _, total = store.list()
    assert total == 2


def test_list_filters_by_plate_substring(store):
    store.save(_bundle(pixels=b"a"))
    items, total = store.list(plate="01AB")
    assert total == 1
    assert items[0].plate == "KA01AB1234"


def test_list_filters_by_confidence_range(store):
    store.save(_bundle(pixels=b"a"))
    assert store.list(min_confidence=0.9)[1] == 0
    assert store.list(max_confidence=0.9)[1] == 1


def test_list_filters_by_created_range(store):
    store.save(_bundle(pixels=b"a"), created_at="2026-06-19T01:00:00Z")
    store.save(_bundle(pixels=b"b"), created_at="2026-06-21T01:00:00Z")
    items, total = store.list(created_from="2026-06-20T00:00:00Z")
    assert total == 1
    assert items[0].created_at.startswith("2026-06-21")


def test_list_sort_oldest_first(store):
    store.save(_bundle(pixels=b"a"), created_at="2026-06-19T01:00:00Z")
    store.save(_bundle(pixels=b"b"), created_at="2026-06-20T01:00:00Z")
    items, _ = store.list(sort="oldest")
    assert items[0].created_at.startswith("2026-06-19")


def test_set_image_path(store):
    new_id = store.save(_bundle())
    store.set_image_path(new_id, "/tmp/evidence/1.jpg")
    assert store.get(new_id)["image_path"] == "/tmp/evidence/1.jpg"


def test_stats_aggregates_by_type_and_day(store):
    store.save(_bundle(violation_type="no-helmet", pixels=b"a"), created_at="2026-06-19T01:00:00Z")
    store.save(_bundle(violation_type="no-helmet", pixels=b"b"), created_at="2026-06-19T02:00:00Z")
    store.save(_bundle(violation_type="triple-riding", pixels=b"c"), created_at="2026-06-20T01:00:00Z")
    stats = store.stats()
    assert stats["total"] == 3
    assert stats["by_type"]["no-helmet"] == 2
    assert stats["by_day"]["2026-06-19"] == 2
    assert stats["by_day"]["2026-06-20"] == 1
    assert stats["mean_confidence"] == pytest.approx(0.82)
