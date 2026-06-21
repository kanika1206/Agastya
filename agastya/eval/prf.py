from __future__ import annotations

from agastya.types import Detection


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def label_predictions(
    predictions: list[Detection], truths: list[Detection], iou_threshold: float
) -> list[tuple[Detection, bool]]:
    ordered = sorted(predictions, key=lambda d: d.score, reverse=True)
    matched: set[int] = set()
    labelled: list[tuple[Detection, bool]] = []
    for pred in ordered:
        best_idx = -1
        best_iou = iou_threshold
        for idx, truth in enumerate(truths):
            if idx in matched or truth.label != pred.label:
                continue
            iou = pred.box.iou(truth.box)
            if iou >= best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            matched.add(best_idx)
            labelled.append((pred, True))
        else:
            labelled.append((pred, False))
    return labelled


def match_detections(
    predictions: list[Detection], truths: list[Detection], iou_threshold: float
) -> tuple[int, int, int]:
    ordered = sorted(predictions, key=lambda d: d.score, reverse=True)
    matched: set[int] = set()
    tp = 0
    fp = 0
    for pred in ordered:
        best_idx = -1
        best_iou = iou_threshold
        for idx, truth in enumerate(truths):
            if idx in matched or truth.label != pred.label:
                continue
            iou = pred.box.iou(truth.box)
            if iou >= best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            matched.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(truths) - len(matched)
    return tp, fp, fn
