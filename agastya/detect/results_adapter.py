from __future__ import annotations

from collections.abc import Sequence

from agastya.types import BBox, Detection


def boxes_to_detections(
    xyxy: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    scores: Sequence[float],
    names: dict[int, str],
) -> list[Detection]:
    if not len(xyxy) == len(class_ids) == len(scores):
        raise ValueError("xyxy, class_ids, and scores must have equal length")
    detections: list[Detection] = []
    for coords, class_id, score in zip(xyxy, class_ids, scores):
        label = names[class_id]
        detections.append(
            Detection(
                label=label,
                score=float(score),
                box=BBox(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
            )
        )
    return detections
