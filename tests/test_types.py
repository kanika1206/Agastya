import pytest

from agastya.types import BBox, Detection


def test_bbox_area():
    box = BBox(x1=0.0, y1=0.0, x2=2.0, y2=3.0)
    assert box.area() == 6.0


def test_bbox_rejects_inverted():
    with pytest.raises(ValueError):
        BBox(x1=5.0, y1=0.0, x2=1.0, y2=1.0)


def test_bbox_iou_identical_is_one():
    box = BBox(x1=0.0, y1=0.0, x2=2.0, y2=2.0)
    assert box.iou(box) == pytest.approx(1.0)


def test_bbox_iou_disjoint_is_zero():
    a = BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    b = BBox(x1=5.0, y1=5.0, x2=6.0, y2=6.0)
    assert a.iou(b) == 0.0


def test_detection_holds_class_and_score():
    det = Detection(label="helmet", score=0.9, box=BBox(0.0, 0.0, 1.0, 1.0))
    assert det.label == "helmet"
    assert det.score == 0.9
