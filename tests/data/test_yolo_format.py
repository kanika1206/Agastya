import pytest

from agastya.data.yolo_format import (
    bbox_to_yolo,
    format_label_line,
    parse_label_line,
    polygon_to_bbox_yolo,
    yolo_to_bbox,
)
from agastya.types import BBox


def test_polygon_to_bbox_yolo_spans_extremes():
    coords = [0.2, 0.3, 0.6, 0.3, 0.6, 0.7, 0.2, 0.7]
    cx, cy, w, h = polygon_to_bbox_yolo(coords)
    assert cx == pytest.approx(0.4)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(0.4)
    assert h == pytest.approx(0.4)


def test_polygon_to_bbox_yolo_rejects_odd_coords():
    with pytest.raises(ValueError):
        polygon_to_bbox_yolo([0.1, 0.2, 0.3])


def test_bbox_to_yolo_centers_and_normalizes():
    box = BBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
    cx, cy, w, h = bbox_to_yolo(box, image_width=100.0, image_height=200.0)
    assert cx == pytest.approx(0.2)
    assert cy == pytest.approx(0.2)
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.2)


def test_yolo_to_bbox_roundtrip():
    box = BBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
    cx, cy, w, h = bbox_to_yolo(box, 100.0, 200.0)
    restored = yolo_to_bbox(cx, cy, w, h, 100.0, 200.0)
    assert restored.x1 == pytest.approx(box.x1)
    assert restored.y1 == pytest.approx(box.y1)
    assert restored.x2 == pytest.approx(box.x2)
    assert restored.y2 == pytest.approx(box.y2)


def test_format_label_line_has_five_fields():
    line = format_label_line(3, 0.5, 0.5, 0.2, 0.4)
    assert line == "3 0.500000 0.500000 0.200000 0.400000"


def test_parse_label_line_roundtrip():
    class_id, cx, cy, w, h = parse_label_line("3 0.5 0.5 0.2 0.4")
    assert class_id == 3
    assert (cx, cy, w, h) == (0.5, 0.5, 0.2, 0.4)


def test_parse_rejects_malformed():
    with pytest.raises(ValueError):
        parse_label_line("3 0.5 0.5")
