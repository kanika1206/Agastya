from __future__ import annotations

from agastya.config import PipelineConfig
from agastya.stages.associate.box import BoxOverlapAssociator


def build_associator(config: PipelineConfig) -> object:
    if config.associate_backend == "box":
        return BoxOverlapAssociator(config.triple_riding_overlap)
    if config.associate_backend == "sam2":
        from agastya.stages.associate.sam2 import Sam2Associator

        return Sam2Associator(
            config.sam2_model,
            config.sam2_weights,
            config.restore_device,
            config.triple_riding_overlap,
        )
    raise ValueError(f"unknown associate_backend: {config.associate_backend}")
