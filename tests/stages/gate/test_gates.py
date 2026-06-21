from __future__ import annotations

import pytest

from agastya.stages.gate.always import AlwaysDegradedGate, NeverDegradedGate
from agastya.stages.gate.arniqa import ArniqaGate
from agastya.stages.gate.errors import GateUnavailable
from agastya.stages.gate.router import score_to_decision


def test_always_gate_scores_zero_and_routes_degraded():
    gate = AlwaysDegradedGate()
    score = gate.score_image(b"anything")
    assert score == 0.0
    assert score_to_decision(score, 0.5).degraded is True


def test_never_gate_scores_one_and_bypasses():
    gate = NeverDegradedGate()
    score = gate.score_image(b"anything")
    assert score == 1.0
    assert score_to_decision(score, 0.5).degraded is False


def test_arniqa_gate_missing_weights_raises_gate_unavailable():
    gate = ArniqaGate(weights="/nonexistent/arniqa.pth", device="cpu")
    with pytest.raises(GateUnavailable):
        gate.score_image(b"\x89PNG")
