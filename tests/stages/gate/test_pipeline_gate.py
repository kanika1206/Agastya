from agastya.config import PipelineConfig
from agastya.pipeline import Pipeline, PipelineInput
from agastya.stages.gate.factory import build_gate
from agastya.stages.stubs import StubDetector, StubOCR


class RecordingRestorer:
    def __init__(self) -> None:
        self.called = False

    def restore(self, pixels: bytes) -> bytes:
        self.called = True
        return pixels


def _run_with_gate(gate_backend: str) -> bool:
    cfg = PipelineConfig(gate_backend=gate_backend, gate_threshold=0.5)
    restorer = RecordingRestorer()
    pipeline = Pipeline(
        config=cfg,
        gate=build_gate(cfg),
        restorer=restorer,
        detector=StubDetector(),
        ocr=StubOCR(text="MH12AB1234", confidence=0.9),
    )
    pipeline.run(PipelineInput(image_id="img1", pixels=b"raw-bytes"))
    return restorer.called


def test_never_gate_bypasses_restoration():
    assert _run_with_gate("never") is False


def test_always_gate_triggers_restoration():
    assert _run_with_gate("always") is True
