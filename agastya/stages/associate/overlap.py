from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mask_overlap_ratio(person_mask: np.ndarray, motorcycle_mask: np.ndarray) -> float:
    person_area = float(person_mask.sum())
    if person_area <= 0.0:
        return 0.0
    inter = float(np.logical_and(person_mask, motorcycle_mask).sum())
    return inter / person_area


def count_overlapping_masks(
    motorcycle_mask: np.ndarray,
    person_masks: Sequence[np.ndarray],
    min_overlap: float,
) -> int:
    return sum(
        1
        for person_mask in person_masks
        if mask_overlap_ratio(person_mask, motorcycle_mask) >= min_overlap
    )
