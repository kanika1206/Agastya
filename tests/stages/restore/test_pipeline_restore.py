from agastya.config import PipelineConfig
from agastya.pipeline import Pipeline, PipelineInput
from agastya.stages.restore.factory import build_restorer
from agastya.stages.restore.passthrough import PassthroughRestorer
from agastya.stages.stubs import StubDetector, StubGate, StubOCR


def test_factory_restorer_runs_pipeline_unchanged():
    cfg = PipelineConfig(gate_threshold=0.5)
    restorer = build_restorer(cfg)
    assert isinstance(restorer, PassthroughRestorer)
    pipeline = Pipeline(
        config=cfg,
        gate=StubGate(score=0.99),
        restorer=restorer,
        detector=StubDetector(),
        ocr=StubOCR(text="MH12AB1234", confidence=0.9),
    )
    result = pipeline.run(PipelineInput(image_id="img1", pixels=b"raw-bytes"))
    assert result.image_id == "img1"
    assert isinstance(result.merkle_root, str)
