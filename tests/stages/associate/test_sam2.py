import numpy as np

from agastya.stages.associate.sam2 import Sam2Associator
from agastya.types import BBox, Detection


def _person(x1: float) -> Detection:
    return Detection(label="person", score=0.9, box=BBox(x1, 0.0, x1 + 1.0, 2.0))


def _row(*cols: bool) -> np.ndarray:
    return np.array([list(cols)], dtype=bool)


class _FakeSam2(Sam2Associator):
    def __init__(self, masks: list[np.ndarray], min_overlap: float = 0.5) -> None:
        super().__init__(model="x", weights=None, device="cpu", min_overlap=min_overlap)
        self._fake_masks = masks
        self.segment_calls = 0

    def _segment(self, pixels: bytes, boxes: list[BBox]) -> list[np.ndarray]:
        self.segment_calls += 1
        return self._fake_masks


class _CountingSam2(Sam2Associator):
    def __init__(self) -> None:
        super().__init__(model="x", weights=None, device="cpu", min_overlap=0.5)
        self.load_calls = 0

    def _load_predictor(self) -> object:
        self.load_calls += 1
        return object()


def test_triple_riding_true_when_three_masks_overlap():
    moto = _row(True, True, True, True)
    masks = [
        moto,
        _row(True, True, False, False),
        _row(False, True, True, False),
        _row(True, False, False, True),
    ]
    associator = _FakeSam2(masks, min_overlap=0.5)
    persons = [_person(0.0), _person(1.0), _person(2.0)]
    assert associator.is_triple_riding(BBox(0.0, 0.0, 4.0, 2.0), persons, b"img") is True


def test_triple_riding_false_when_only_two_masks_overlap():
    moto = _row(True, True, False, False)
    masks = [
        moto,
        _row(True, True, False, False),
        _row(True, False, False, False),
        _row(False, False, True, True),
    ]
    associator = _FakeSam2(masks, min_overlap=0.5)
    persons = [_person(0.0), _person(1.0), _person(2.0)]
    assert associator.is_triple_riding(BBox(0.0, 0.0, 4.0, 2.0), persons, b"img") is False


def test_no_persons_short_circuits_without_segmenting():
    associator = _FakeSam2([], min_overlap=0.5)
    assert associator.is_triple_riding(BBox(0.0, 0.0, 4.0, 2.0), [], b"img") is False
    assert associator.segment_calls == 0


def test_predictor_loaded_once_and_cached_per_process():
    associator = _CountingSam2()
    associator._ensure_predictor()
    associator._ensure_predictor()
    assert associator.load_calls == 1
    assert associator._predictor is not None
