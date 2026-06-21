from __future__ import annotations

from agastya.eval.scoring import accumulate, f1_of, new_counts, overall_f1
from agastya.types import BBox, Detection


def _det(label: str, score: float, x: float) -> Detection:
    return Detection(label=label, score=score, box=BBox(x, 0.0, x + 0.1, 0.1))


def test_new_counts_zeroed():
    counts = new_counts(["person", "car"])
    assert counts == {"person": [0, 0, 0], "car": [0, 0, 0]}


def test_accumulate_scores_tp_and_fn():
    names = ["person"]
    counts = new_counts(names)
    pred = _det("person", 0.9, 0.0)
    truth = _det("person", 1.0, 0.0)
    accumulate(counts, [pred], [truth], names, 0.5)
    accumulate(counts, [], [truth], names, 0.5)
    assert counts["person"] == [1, 0, 1]


def test_f1_and_overall_f1_perfect():
    names = ["person", "car"]
    counts = {"person": [4, 0, 0], "car": [6, 0, 0]}
    assert f1_of(counts, "person") == 1.0
    assert overall_f1(counts, names) == 1.0
