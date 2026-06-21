from agastya.stages.associate.box import BoxOverlapAssociator
from agastya.types import BBox, Detection


def _person(x1: float) -> Detection:
    return Detection(label="person", score=0.9, box=BBox(x1, 0.0, x1 + 1.0, 2.0))


def test_box_associator_true_at_three_overlapping():
    associator = BoxOverlapAssociator(min_overlap=0.05)
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0), _person(2.5)]
    assert associator.is_triple_riding(moto, persons, b"") is True


def test_box_associator_false_at_two():
    associator = BoxOverlapAssociator(min_overlap=0.05)
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0)]
    assert associator.is_triple_riding(moto, persons, b"") is False


def test_box_associator_ignores_pixels():
    associator = BoxOverlapAssociator(min_overlap=0.05)
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0), _person(2.5)]
    assert associator.is_triple_riding(moto, persons, b"unused-bytes") is True
