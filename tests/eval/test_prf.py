import pytest

from agastya.eval.prf import label_predictions, match_detections, precision_recall_f1
from agastya.types import BBox, Detection


def _det(label: str, x1: float, score: float = 0.9) -> Detection:
    return Detection(label=label, score=score, box=BBox(x1, 0.0, x1 + 2.0, 2.0))


def test_precision_recall_f1_basic():
    p, r, f1 = precision_recall_f1(tp=8, fp=2, fn=2)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)


def test_prf_zero_predictions_is_zero():
    p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=5)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_match_counts_tp_fp_fn():
    preds = [_det("helmet", 0.0), _det("helmet", 10.0), _det("helmet", 50.0)]
    truths = [_det("helmet", 0.1), _det("helmet", 10.1)]
    tp, fp, fn = match_detections(preds, truths, iou_threshold=0.3)
    assert tp == 2
    assert fp == 1
    assert fn == 0


def test_match_respects_label():
    preds = [_det("helmet", 0.0)]
    truths = [_det("no-helmet", 0.0)]
    tp, fp, fn = match_detections(preds, truths, iou_threshold=0.3)
    assert (tp, fp, fn) == (0, 1, 1)


def test_label_predictions_flags_correct_match():
    preds = [_det("helmet", 0.0, score=0.9)]
    truths = [_det("helmet", 0.1)]
    assert label_predictions(preds, truths, iou_threshold=0.3) == [(preds[0], True)]


def test_label_predictions_flags_wrong_class_as_false():
    preds = [_det("helmet", 0.0, score=0.9)]
    truths = [_det("no-helmet", 0.0)]
    assert label_predictions(preds, truths, iou_threshold=0.3) == [(preds[0], False)]


def test_label_predictions_one_truth_matches_highest_score():
    high = _det("helmet", 0.0, score=0.95)
    low = _det("helmet", 0.1, score=0.60)
    result = dict(label_predictions([low, high], [_det("helmet", 0.05)], iou_threshold=0.3))
    assert result[high] is True
    assert result[low] is False
