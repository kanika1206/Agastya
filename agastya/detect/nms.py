from __future__ import annotations

from agastya.types import Detection


def non_max_suppression(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        suppressed = False
        for keeper in kept:
            if keeper.label != candidate.label:
                continue
            if keeper.box.iou(candidate.box) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept
