import numpy as np

from agastya.stages.associate.overlap import count_overlapping_masks, mask_overlap_ratio


def _row(*cols: bool) -> np.ndarray:
    return np.array([list(cols)], dtype=bool)


def test_ratio_zero_for_empty_person_mask():
    person = np.zeros((1, 4), dtype=bool)
    moto = np.ones((1, 4), dtype=bool)
    assert mask_overlap_ratio(person, moto) == 0.0


def test_ratio_is_intersection_over_person_area():
    person = _row(True, True, False, False)
    moto = _row(True, False, False, False)
    assert mask_overlap_ratio(person, moto) == 0.5


def test_counts_masks_meeting_threshold():
    moto = _row(True, True, False, False)
    p_full = _row(True, True, False, False)
    p_half = _row(True, False, True, False)
    p_none = _row(False, False, True, True)
    masks = [p_full, p_half, p_none]
    assert count_overlapping_masks(moto, masks, 0.5) == 2
    assert count_overlapping_masks(moto, masks, 1.0) == 1
