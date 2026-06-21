from __future__ import annotations

from agastya.config import PipelineConfig
from agastya.stages.ocr.null import NullOcr


def build_ocr(config: PipelineConfig) -> object:
    if config.ocr_backend == "none":
        return NullOcr()
    if config.ocr_backend == "parseq":
        from agastya.stages.ocr.parseq import ParseqOcr

        return ParseqOcr(
            config.parseq_weights,
            config.restore_device,
            config.ocr_min_confidence,
        )
    raise ValueError(f"unknown ocr_backend: {config.ocr_backend}")
