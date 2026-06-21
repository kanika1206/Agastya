import pytest

from agastya.stages.gate.router import score_to_decision
from agastya.types import QualityScore


def test_below_threshold_routes_to_restore():
    decision = score_to_decision(0.3, threshold=0.5)
    assert isinstance(decision, QualityScore)
    assert decision.degraded is True


def test_at_or_above_threshold_bypasses():
    assert score_to_decision(0.5, threshold=0.5).degraded is False
    assert score_to_decision(0.8, threshold=0.5).degraded is False


def test_threshold_must_be_unit_interval():
    with pytest.raises(ValueError):
        score_to_decision(0.4, threshold=1.5)


def test_score_must_be_unit_interval():
    with pytest.raises(ValueError):
        score_to_decision(1.5, threshold=0.5)
    with pytest.raises(ValueError):
        score_to_decision(-0.1, threshold=0.5)
