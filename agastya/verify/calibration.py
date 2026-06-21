from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

_EPS = 1e-7

CLEAN_LABEL = "clean"


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def temperature_scale(prob: float, temperature: float) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return _sigmoid(_logit(prob) / temperature)


def _nll(probs: Sequence[float], labels: Sequence[int], temperature: float) -> float:
    total = 0.0
    for prob, label in zip(probs, labels, strict=True):
        scaled = temperature_scale(prob, temperature)
        scaled = min(max(scaled, _EPS), 1.0 - _EPS)
        total -= label * math.log(scaled) + (1 - label) * math.log(1.0 - scaled)
    return total


def fit_temperature(
    probs: Sequence[float],
    labels: Sequence[int],
    grid_points: int = 200,
    max_temperature: float = 10.0,
) -> float:
    best_temperature = 1.0
    best_nll = math.inf
    for i in range(grid_points):
        temperature = max_temperature * (i + 1) / grid_points
        loss = _nll(probs, labels, temperature)
        if loss < best_nll:
            best_nll = loss
            best_temperature = temperature
    return best_temperature


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    total = len(probs)
    if total == 0:
        return 0.0
    bucket_conf = [0.0] * bins
    bucket_acc = [0.0] * bins
    bucket_n = [0] * bins
    for prob, label in zip(probs, labels, strict=True):
        idx = min(int(prob * bins), bins - 1)
        bucket_conf[idx] += prob
        bucket_acc[idx] += label
        bucket_n[idx] += 1
    ece = 0.0
    for conf_sum, acc_sum, n in zip(bucket_conf, bucket_acc, bucket_n, strict=True):
        if n == 0:
            continue
        ece += (n / total) * abs(acc_sum / n - conf_sum / n)
    return ece


def conformal_quantile(nonconformity_scores: Sequence[float], alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    n = len(nonconformity_scores)
    if n == 0:
        return 1.0
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return 1.0
    return sorted(nonconformity_scores)[rank - 1]


def conformal_set(label_probs: Mapping[str, float], qhat: float) -> set[str]:
    return {label for label, prob in label_probs.items() if (1.0 - prob) <= qhat}


def needs_human_review(prediction_set: Iterable[str]) -> bool:
    return len(set(prediction_set)) != 1


@dataclass(frozen=True)
class CalibratedConfidence:
    confidence: float
    prediction_set: frozenset[str]
    needs_review: bool


@dataclass(frozen=True)
class Calibrator:
    temperature: float
    qhat: float

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.qhat < 0.0:
            raise ValueError("qhat must be non-negative")

    @classmethod
    def from_json(cls, path: str) -> Calibrator:
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        return cls(temperature=float(data["temperature"]), qhat=float(data["qhat"]))

    def evaluate(self, raw_confidence: float, violation_type: str) -> CalibratedConfidence:
        confidence = temperature_scale(raw_confidence, self.temperature)
        probs = {violation_type: confidence, CLEAN_LABEL: 1.0 - confidence}
        prediction_set = conformal_set(probs, self.qhat)
        return CalibratedConfidence(
            confidence=confidence,
            prediction_set=frozenset(prediction_set),
            needs_review=needs_human_review(prediction_set),
        )


def fit_calibrator(
    confidences: Sequence[float], labels: Sequence[int], alpha: float
) -> Calibrator:
    temperature = fit_temperature(confidences, labels)
    scores: list[float] = []
    for conf, label in zip(confidences, labels, strict=True):
        calibrated = temperature_scale(conf, temperature)
        scores.append(1.0 - calibrated if label == 1 else calibrated)
    return Calibrator(temperature=temperature, qhat=conformal_quantile(scores, alpha))
