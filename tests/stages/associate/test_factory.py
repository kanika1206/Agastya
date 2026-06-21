import pytest

from agastya.config import PipelineConfig
from agastya.stages.associate.box import BoxOverlapAssociator
from agastya.stages.associate.factory import build_associator
from agastya.stages.associate.sam2 import Sam2Associator


def test_factory_defaults_to_box():
    assert isinstance(build_associator(PipelineConfig()), BoxOverlapAssociator)


def test_factory_box_reuses_triple_riding_overlap():
    associator = build_associator(PipelineConfig(triple_riding_overlap=0.2))
    assert isinstance(associator, BoxOverlapAssociator)
    assert associator.min_overlap == 0.2


def test_factory_builds_sam2_backend():
    associator = build_associator(PipelineConfig(associate_backend="sam2"))
    assert isinstance(associator, Sam2Associator)


def test_factory_sam2_reuses_triple_riding_overlap():
    associator = build_associator(
        PipelineConfig(associate_backend="sam2", triple_riding_overlap=0.3)
    )
    assert associator.min_overlap == 0.3


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_associator(PipelineConfig(associate_backend="bogus"))
