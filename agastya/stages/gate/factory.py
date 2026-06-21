from __future__ import annotations

from agastya.config import PipelineConfig
from agastya.stages.gate.always import AlwaysDegradedGate, NeverDegradedGate


def build_gate(config: PipelineConfig) -> object:
    if config.gate_backend == "never":
        return NeverDegradedGate()
    if config.gate_backend == "always":
        return AlwaysDegradedGate()
    if config.gate_backend == "arniqa":
        from agastya.stages.gate.arniqa import ArniqaGate

        return ArniqaGate(config.arniqa_weights, config.restore_device)
    raise ValueError(f"unknown gate_backend: {config.gate_backend}")
