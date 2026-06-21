from agastya.stages.associate.rules import count_riders, is_triple_riding
from agastya.types import BBox, Detection


def _person(x1: float) -> Detection:
    return Detection(label="person", score=0.9, box=BBox(x1, 0.0, x1 + 1.0, 2.0))


def test_counts_persons_overlapping_motorcycle():
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0), _person(2.5), _person(50.0)]
    assert count_riders(moto, persons, min_overlap=0.05) == 3


def test_triple_riding_true_at_three():
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0), _person(2.5)]
    assert is_triple_riding(moto, persons, min_overlap=0.05) is True


def test_triple_riding_false_at_two():
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0)]
    assert is_triple_riding(moto, persons, min_overlap=0.05) is False
