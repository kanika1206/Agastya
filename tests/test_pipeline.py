import pytest

from agastya.config import PipelineConfig, VRAMProfile
from agastya.pipeline import Pipeline, PipelineInput
from agastya.stages.stubs import (
    StubDetector,
    StubGate,
    StubOCR,
    StubRestorer,
)
from agastya.types import BBox, Detection
from agastya.verify.calibration import Calibrator, temperature_scale


class _ListDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, pixels: bytes) -> list[Detection]:
        return self._detections


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


def test_triple_riding_record_emitted_when_three_riders_overlap_motorcycle():
    detector = _ListDetector(
        [
            Detection(label="motorcycle", score=0.95, box=BBox(0.0, 0.0, 6.0, 2.0)),
            Detection(label="person", score=0.90, box=BBox(0.2, 0.0, 1.0, 2.0)),
            Detection(label="person", score=0.88, box=BBox(2.2, 0.0, 3.0, 2.0)),
            Detection(label="person", score=0.86, box=BBox(4.2, 0.0, 5.0, 2.0)),
        ]
    )
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=detector,
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-tr", pixels=b"raw"))
    assert "triple-riding" in [r.violation_type for r in result.records]


def test_no_triple_riding_record_with_only_two_riders():
    detector = _ListDetector(
        [
            Detection(label="motorcycle", score=0.95, box=BBox(0.0, 0.0, 6.0, 2.0)),
            Detection(label="person", score=0.90, box=BBox(0.2, 0.0, 1.0, 2.0)),
            Detection(label="person", score=0.88, box=BBox(2.2, 0.0, 3.0, 2.0)),
        ]
    )
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=detector,
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-no-tr", pixels=b"raw"))
    assert "triple-riding" not in [r.violation_type for r in result.records]


def test_records_uncalibrated_keep_raw_confidence_and_no_review_flag():
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-uncal", pixels=b"raw"))
    record = next(r for r in result.records if r.violation_type == "no-helmet")
    assert record.confidence == pytest.approx(0.88)
    assert "human_review" not in record.metadata


def test_records_calibrated_apply_temperature_and_conformal_metadata():
    calibrator = Calibrator(temperature=2.0, qhat=0.3)
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
        calibrator=calibrator,
    )
    result = pipeline.run(PipelineInput(image_id="img-cal", pixels=b"raw"))
    record = next(r for r in result.records if r.violation_type == "no-helmet")
    assert record.confidence == pytest.approx(temperature_scale(0.88, 2.0))
    assert record.metadata["raw_confidence"] == "0.88"
    assert record.metadata["conformal_set"] == "no-helmet"
    assert record.metadata["human_review"] == "false"


def test_no_helmet_below_min_conf_not_emitted():
    detector = _ListDetector(
        [Detection(label="no-helmet", score=0.5, box=BBox(0.0, 0.0, 1.0, 1.0))]
    )
    pipeline = Pipeline(
        config=PipelineConfig(gate_threshold=0.5, no_helmet_min_conf=0.7),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=detector,
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-low", pixels=b"raw"))
    assert "no-helmet" not in [r.violation_type for r in result.records]


def test_no_helmet_at_or_above_min_conf_emitted():
    detector = _ListDetector(
        [Detection(label="no-helmet", score=0.72, box=BBox(0.0, 0.0, 1.0, 1.0))]
    )
    pipeline = Pipeline(
        config=PipelineConfig(gate_threshold=0.5, no_helmet_min_conf=0.7),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=detector,
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-high", pixels=b"raw"))
    assert "no-helmet" in [r.violation_type for r in result.records]


def test_scene_context_emits_illegal_parking():
    from agastya.stages.violations.scene import SceneContext

    detector = _ListDetector(
        [Detection(label="motorcycle", score=0.92, box=BBox(4.0, 4.0, 6.0, 6.0))]
    )
    scene = SceneContext(
        no_parking_zones=(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),)
    )
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=detector,
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
        scene=scene,
    )
    result = pipeline.run(PipelineInput(image_id="img-park", pixels=b"raw"))
    assert "illegal-parking" in [r.violation_type for r in result.records]


def test_no_scene_context_is_backcompat():
    detector = _ListDetector(
        [Detection(label="motorcycle", score=0.92, box=BBox(4.0, 4.0, 6.0, 6.0))]
    )
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=detector,
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-none", pixels=b"raw"))
    assert [r.violation_type for r in result.records] == []


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
