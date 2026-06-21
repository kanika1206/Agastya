import cv2
import numpy as np
import pytest

from agastya.ingest.annotate import annotate_violation
from agastya.types import BBox, Detection, PlateReading, ViolationRecord


def _jpg(width=64, height=48):
    frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    return cv2.imencode(".jpg", frame)[1].tobytes()


def _record():
    return ViolationRecord(
        violation_type="no-helmet",
        confidence=0.88,
        plate=PlateReading(text="KA01AB1234", confidence=0.8),
        detections=(Detection(label="no-helmet", score=0.88, box=BBox(5.0, 5.0, 30.0, 30.0)),),
    )


def test_annotate_returns_decodable_jpg():
    out = annotate_violation(_jpg(), _record())
    frame = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    assert frame.shape == (48, 64, 3)


def test_annotate_rejects_undecodable_bytes():
    with pytest.raises(ValueError):
        annotate_violation(b"not-an-image", _record())
