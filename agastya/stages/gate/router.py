from __future__ import annotations

from agastya.types import QualityScore


def score_to_decision(score: float, threshold: float) -> QualityScore:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return QualityScore(value=score, degraded=score < threshold)
