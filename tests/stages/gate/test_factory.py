from __future__ import annotations

import pytest

from agastya.config import PipelineConfig
from agastya.stages.gate.always import AlwaysDegradedGate, NeverDegradedGate
from agastya.stages.gate.arniqa import ArniqaGate
from agastya.stages.gate.factory import build_gate


def test_factory_defaults_to_never_gate():
    gate = build_gate(PipelineConfig())
    assert isinstance(gate, NeverDegradedGate)


def test_factory_builds_always_gate():
    gate = build_gate(PipelineConfig(gate_backend="always"))
    assert isinstance(gate, AlwaysDegradedGate)


def test_factory_builds_arniqa_gate():
    gate = build_gate(PipelineConfig(gate_backend="arniqa", arniqa_weights="w.pth"))
    assert isinstance(gate, ArniqaGate)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown gate_backend"):
        build_gate(PipelineConfig(gate_backend="bogus"))
