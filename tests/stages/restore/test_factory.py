import pytest

from agastya.config import PipelineConfig
from agastya.stages.restore.factory import build_restorer
from agastya.stages.restore.passthrough import PassthroughRestorer


def test_factory_builds_passthrough_by_default():
    restorer = build_restorer(PipelineConfig())
    assert isinstance(restorer, PassthroughRestorer)


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_restorer(PipelineConfig(restore_backend="bogus"))
