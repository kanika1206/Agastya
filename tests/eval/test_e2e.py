from agastya.eval.e2e import (
    ViolationCounts,
    accumulate,
    gt_violations,
    pred_violations,
)
from agastya.types import BBox, Detection, ViolationRecord


def _det(label: str, x1: float) -> Detection:
    return Detection(label=label, score=0.9, box=BBox(x1, 0.0, x1 + 1.0, 2.0))


def test_gt_violations_detects_no_helmet():
    truths = [_det("no-helmet", 0.0)]
    assert gt_violations(truths, overlap=0.05) == {"no-helmet"}


def test_gt_violations_detects_triple_riding():
    truths = [
        Detection(label="motorcycle", score=1.0, box=BBox(0.0, 0.0, 4.0, 2.0)),
        _det("person", 0.0),
        _det("person", 1.0),
        _det("person", 2.5),
    ]
    assert "triple-riding" in gt_violations(truths, overlap=0.05)


def test_gt_violations_empty_when_two_riders():
    truths = [
        Detection(label="motorcycle", score=1.0, box=BBox(0.0, 0.0, 4.0, 2.0)),
        _det("person", 0.0),
        _det("person", 1.0),
    ]
    assert gt_violations(truths, overlap=0.05) == set()


def test_pred_violations_reads_record_types():
    records = (
        ViolationRecord(violation_type="no-helmet", confidence=0.8, plate=None),
        ViolationRecord(violation_type="triple-riding", confidence=0.7, plate=None),
    )
    assert pred_violations(records) == {"no-helmet", "triple-riding"}


def test_violation_counts_tp_fp_fn_and_prf():
    counts = ViolationCounts()
    counts.update(present_pred=True, present_gt=True)
    counts.update(present_pred=True, present_gt=False)
    counts.update(present_pred=False, present_gt=True)
    counts.update(present_pred=False, present_gt=False)
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 1)
    precision, recall, f1 = counts.prf()
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_accumulate_updates_per_type_counts():
    per_type = {"no-helmet": ViolationCounts(), "triple-riding": ViolationCounts()}
    accumulate(per_type, {"no-helmet"}, {"no-helmet", "triple-riding"})
    assert per_type["no-helmet"].tp == 1
    assert per_type["triple-riding"].fn == 1
