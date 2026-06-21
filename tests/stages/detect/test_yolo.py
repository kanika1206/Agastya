import cv2
import numpy as np

from agastya.stages.detect.yolo import YoloDetector


class _FakeBox:
    def __init__(self, xywhn: tuple[float, float, float, float], cls: int, conf: float) -> None:
        self.xywhn = [list(xywhn)]
        self.cls = cls
        self.conf = conf


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeModel:
    names = {0: "helmet", 1: "no-helmet", 2: "license-plate", 3: "motorcycle", 4: "person"}


class _FakeYolo(YoloDetector):
    def __init__(self, result: _FakeResult) -> None:
        super().__init__(weights="x", device="cpu")
        self._result = result
        self.load_calls = 0

    def _load_model(self) -> object:
        self.load_calls += 1
        return _FakeModel()

    def _predict(self, image: np.ndarray) -> _FakeResult:
        return self._result


def _png_bytes() -> bytes:
    image = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    return cv2.imencode(".png", image)[1].tobytes()


def test_detect_maps_yolo_classes_to_detection_labels():
    result = _FakeResult([_FakeBox((0.5, 0.5, 0.2, 0.2), cls=1, conf=0.9)])
    detector = _FakeYolo(result)
    detections = detector.detect(_png_bytes())
    assert len(detections) == 1
    assert detections[0].label == "no-helmet"
    assert detections[0].score == 0.9


def test_detect_returns_empty_when_no_boxes():
    detector = _FakeYolo(_FakeResult([]))
    assert detector.detect(_png_bytes()) == []


def test_model_loaded_once_and_cached():
    detector = _FakeYolo(_FakeResult([]))
    detector.detect(_png_bytes())
    detector.detect(_png_bytes())
    assert detector.load_calls == 1
