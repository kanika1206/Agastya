import os

import cv2
import numpy as np
import pytest

from agastya.config import PipelineConfig, VRAMProfile
from agastya.ingest.runner import IngestRunner
from agastya.pipeline import Pipeline
from agastya.stages.stubs import StubDetector, StubGate, StubOCR, StubRestorer
from agastya.store.sqlite_store import ViolationStore

_KEY = b"camera-signing-key"
_VERSIONS = {"detector": "stub@0.1"}


def _jpg(width=64, height=48):
    frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    return cv2.imencode(".jpg", frame)[1].tobytes()


def _pipeline():
    return Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )


@pytest.fixture
def store():
    store = ViolationStore(":memory:")
    yield store
    store.close()


def test_ingest_persists_violation(store):
    runner = IngestRunner(_pipeline(), store, signing_key=_KEY, model_versions=_VERSIONS)
    result = runner.ingest_image("img-1", _jpg())
    assert result.count == 1
    _, total = store.list()
    assert total == 1


def test_ingest_is_idempotent(store):
    runner = IngestRunner(_pipeline(), store, signing_key=_KEY, model_versions=_VERSIONS)
    pixels = _jpg()
    first = runner.ingest_image("img-1", pixels)
    second = runner.ingest_image("img-1", pixels)
    assert second.violation_ids == first.violation_ids
    _, total = store.list()
    assert total == 1


def test_ingest_writes_evidence_image(store, tmp_path):
    runner = IngestRunner(
        _pipeline(),
        store,
        signing_key=_KEY,
        model_versions=_VERSIONS,
        evidence_dir=str(tmp_path),
    )
    result = runner.ingest_image("img-1", _jpg())
    violation_id = result.violation_ids[0]
    path = os.path.join(str(tmp_path), f"{violation_id}.jpg")
    assert os.path.exists(path)
    assert store.get(violation_id)["image_path"] == path
