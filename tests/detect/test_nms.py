from agastya.detect.nms import non_max_suppression
from agastya.types import BBox, Detection


def _det(label: str, score: float, x1: float) -> Detection:
    return Detection(label=label, score=score, box=BBox(x1, 0.0, x1 + 2.0, 2.0))


def test_nms_suppresses_lower_score_overlap():
    high = _det("helmet", 0.9, 0.0)
    low = _det("helmet", 0.6, 0.2)
    kept = non_max_suppression([low, high], iou_threshold=0.5)
    assert kept == [high]


def test_nms_keeps_disjoint_boxes():
    a = _det("helmet", 0.9, 0.0)
    b = _det("helmet", 0.8, 50.0)
    kept = non_max_suppression([a, b], iou_threshold=0.5)
    assert set(kept) == {a, b}


def test_nms_is_per_label():
    helmet = _det("helmet", 0.9, 0.0)
    person = _det("person", 0.8, 0.1)
    kept = non_max_suppression([helmet, person], iou_threshold=0.5)
    assert set(kept) == {helmet, person}


def test_nms_empty_returns_empty():
    assert non_max_suppression([], iou_threshold=0.5) == []
