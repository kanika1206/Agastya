from agastya.detect.sahi_merge import SlicePrediction, merge_slice_predictions, shift_detection
from agastya.types import BBox, Detection


def test_shift_detection_offsets_box():
    det = Detection(label="helmet", score=0.9, box=BBox(0.0, 0.0, 2.0, 2.0))
    shifted = shift_detection(det, offset_x=10.0, offset_y=20.0)
    assert shifted.box == BBox(10.0, 20.0, 12.0, 22.0)
    assert shifted.label == "helmet"
    assert shifted.score == 0.9


def test_merge_dedupes_overlapping_seam_detections():
    slice_a = SlicePrediction(
        offset_x=0.0,
        offset_y=0.0,
        detections=[Detection("helmet", 0.7, BBox(98.0, 0.0, 102.0, 4.0))],
    )
    slice_b = SlicePrediction(
        offset_x=100.0,
        offset_y=0.0,
        detections=[Detection("helmet", 0.9, BBox(-2.0, 0.0, 2.0, 4.0))],
    )
    merged = merge_slice_predictions([slice_a, slice_b], iou_threshold=0.3)
    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_merge_keeps_distinct_detections():
    slice_a = SlicePrediction(
        offset_x=0.0, offset_y=0.0, detections=[Detection("helmet", 0.8, BBox(0.0, 0.0, 4.0, 4.0))]
    )
    slice_b = SlicePrediction(
        offset_x=200.0,
        offset_y=0.0,
        detections=[Detection("no-helmet", 0.8, BBox(0.0, 0.0, 4.0, 4.0))],
    )
    merged = merge_slice_predictions([slice_a, slice_b], iou_threshold=0.5)
    assert len(merged) == 2
