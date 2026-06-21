from __future__ import annotations

from dataclasses import dataclass

from agastya.detect.nms import non_max_suppression
from agastya.types import BBox, Detection


@dataclass(frozen=True)
class SlicePrediction:
    offset_x: float
    offset_y: float
    detections: list[Detection]


def shift_detection(detection: Detection, offset_x: float, offset_y: float) -> Detection:
    box = detection.box
    return Detection(
        label=detection.label,
        score=detection.score,
        box=BBox(
            x1=box.x1 + offset_x,
            y1=box.y1 + offset_y,
            x2=box.x2 + offset_x,
            y2=box.y2 + offset_y,
        ),
    )


def merge_slice_predictions(
    slices: list[SlicePrediction], iou_threshold: float
) -> list[Detection]:
    globalized: list[Detection] = []
    for sl in slices:
        for det in sl.detections:
            globalized.append(shift_detection(det, sl.offset_x, sl.offset_y))
    return non_max_suppression(globalized, iou_threshold)
