from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from agastya.eval.prf import precision_recall_f1
from agastya.stages.associate.rules import is_triple_riding
from agastya.types import Detection, ViolationRecord

VIOLATION_TYPES = ("no-helmet", "triple-riding")


def gt_violations(truths: Iterable[Detection], overlap: float) -> set[str]:
    truths = list(truths)
    found: set[str] = set()
    if any(t.label == "no-helmet" for t in truths):
        found.add("no-helmet")
    persons = [t for t in truths if t.label == "person"]
    for moto in truths:
        if moto.label == "motorcycle" and is_triple_riding(moto.box, persons, overlap):
            found.add("triple-riding")
            break
    return found


def pred_violations(records: Iterable[ViolationRecord]) -> set[str]:
    return {record.violation_type for record in records}


@dataclass
class ViolationCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, present_pred: bool, present_gt: bool) -> None:
        if present_pred and present_gt:
            self.tp += 1
        elif present_pred and not present_gt:
            self.fp += 1
        elif present_gt and not present_pred:
            self.fn += 1

    def prf(self) -> tuple[float, float, float]:
        return precision_recall_f1(self.tp, self.fp, self.fn)


def accumulate(
    per_type: Mapping[str, ViolationCounts],
    pred_set: set[str],
    gt_set: set[str],
) -> None:
    for violation_type, counts in per_type.items():
        counts.update(violation_type in pred_set, violation_type in gt_set)
