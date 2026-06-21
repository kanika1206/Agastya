from agastya.config import PipelineConfig, VRAMProfile
from agastya.pipeline import Pipeline, PipelineInput
from agastya.stages.stubs import (
    StubDetector,
    StubGate,
    StubOCR,
    StubRestorer,
)


def test_clean_image_bypasses_restore():
    restorer = StubRestorer()
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=restorer,
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-1", pixels=b"raw"))
    assert restorer.calls == 0
    assert result.records


def test_degraded_image_invokes_restore():
    restorer = StubRestorer()
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.2),
        restorer=restorer,
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    pipeline.run(PipelineInput(image_id="img-2", pixels=b"raw"))
    assert restorer.calls == 1


def test_result_records_carry_merkle_root():
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-3", pixels=b"raw"))
    assert isinstance(result.merkle_root, str)
    assert len(result.merkle_root) == 64
