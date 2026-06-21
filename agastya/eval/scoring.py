from __future__ import annotations

from agastya.eval.prf import match_detections, precision_recall_f1
from agastya.types import Detection


def new_counts(names: list[str]) -> dict[str, list[int]]:
    return {name: [0, 0, 0] for name in names}


def accumulate(
    counts: dict[str, list[int]],
    preds: list[Detection],
    truths: list[Detection],
    names: list[str],
    iou: float,
) -> None:
    for name in names:
        p_c = [d for d in preds if d.label == name]
        t_c = [d for d in truths if d.label == name]
        if not p_c and not t_c:
            continue
        tp, fp, fn = match_detections(p_c, t_c, iou)
        counts[name][0] += tp
        counts[name][1] += fp
        counts[name][2] += fn


def f1_of(counts: dict[str, list[int]], name: str) -> float:
    return precision_recall_f1(*counts[name])[2]


def overall_f1(counts: dict[str, list[int]], names: list[str]) -> float:
    total = [0, 0, 0]
    for name in names:
        for i in range(3):
            total[i] += counts[name][i]
    return precision_recall_f1(*total)[2]
