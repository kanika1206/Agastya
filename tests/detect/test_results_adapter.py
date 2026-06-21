import pytest

from agastya.detect.results_adapter import boxes_to_detections
from agastya.types import BBox


def test_boxes_to_detections_maps_names():
    dets = boxes_to_detections(
        xyxy=[[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 7.0, 9.0]],
        class_ids=[2, 4],
        scores=[0.9, 0.7],
        names={2: "helmet", 4: "person"},
    )
    assert dets[0].label == "helmet"
    assert dets[0].box == BBox(0.0, 0.0, 2.0, 2.0)
    assert dets[1].label == "person"
    assert dets[1].score == 0.7


def test_boxes_to_detections_length_mismatch_raises():
    with pytest.raises(ValueError):
        boxes_to_detections(xyxy=[[0.0, 0.0, 1.0, 1.0]], class_ids=[2, 4], scores=[0.9], names={})


def test_boxes_to_detections_unknown_id_raises():
    with pytest.raises(KeyError):
        boxes_to_detections(
            xyxy=[[0.0, 0.0, 1.0, 1.0]], class_ids=[99], scores=[0.9], names={2: "helmet"}
        )
