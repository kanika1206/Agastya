import pytest

from agastya.config import PipelineConfig
from agastya.stages.ocr.factory import build_ocr
from agastya.stages.ocr.null import NullOcr


def test_factory_default_is_null_ocr():
    ocr = build_ocr(PipelineConfig())
    assert isinstance(ocr, NullOcr)


def test_null_ocr_always_abstains():
    reading = NullOcr().read(b"pixels")
    assert reading.abstained is True


def test_factory_parseq_backend_builds_adapter():
    ocr = build_ocr(PipelineConfig(ocr_backend="parseq"))
    assert ocr.__class__.__name__ == "ParseqOcr"


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_ocr(PipelineConfig(ocr_backend="bogus"))
