from __future__ import annotations

from agastya.config import PipelineConfig
from agastya.stages.restore.passthrough import PassthroughRestorer


def build_restorer(config: PipelineConfig) -> object:
    if config.restore_backend == "passthrough":
        return PassthroughRestorer()
    if config.restore_backend == "nafnet":
        from agastya.stages.restore.nafnet import NafnetRestorer

        return NafnetRestorer(config.nafnet_weights, config.restore_device)
    raise ValueError(f"unknown restore_backend: {config.restore_backend}")
