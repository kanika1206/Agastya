from __future__ import annotations

from agastya.config import PipelineConfig
from agastya.stages.stubs import StubDetector


def build_detector(config: PipelineConfig) -> object:
    if config.detect_backend == "stub":
        return StubDetector()
    if config.detect_backend == "yolo":
        from agastya.stages.detect.yolo import YoloDetector

        if not config.detector_weights:
            raise ValueError("detector_weights required for yolo detect_backend")
        return YoloDetector(
            config.detector_weights,
            config.detect_imgsz,
            config.detect_conf,
            config.restore_device,
        )
    raise ValueError(f"unknown detect_backend: {config.detect_backend}")
