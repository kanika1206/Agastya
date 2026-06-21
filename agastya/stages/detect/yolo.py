from __future__ import annotations

import numpy as np

from agastya.eval.yolo_data import predictions_for
from agastya.stages.detect.errors import DetectUnavailable
from agastya.types import Detection


class YoloDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self._model: object | None = None

    def _load_model(self) -> object:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectUnavailable("ultralytics not installed") from exc
        try:
            return YOLO(str(self.weights))
        except Exception as exc:
            raise DetectUnavailable(f"failed to load YOLO weights: {exc}") from exc

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = self._load_model()

    def _names(self) -> list[str]:
        names = self._model.names
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        return list(names)

    def _predict(self, image: np.ndarray) -> object:
        return self._model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]

    def detect(self, pixels: bytes) -> list[Detection]:
        import cv2

        self._ensure_model()
        image = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise DetectUnavailable("could not decode image bytes")
        result = self._predict(image)
        return predictions_for(result, self._names())
