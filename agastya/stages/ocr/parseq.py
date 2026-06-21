from __future__ import annotations

import cv2
import numpy as np

from agastya.stages.ocr.errors import OcrUnavailable
from agastya.stages.ocr.guard import guard_reading
from agastya.types import PlateReading


class ParseqOcr:
    def __init__(
        self,
        weights: str | None = None,
        device: str = "cpu",
        min_confidence: float = 0.5,
    ) -> None:
        self.weights = weights
        self.device = device
        self.min_confidence = min_confidence
        self._model = None
        self._transform = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise OcrUnavailable("torch/torchvision not installed") from exc
        try:
            model = torch.hub.load("baudm/parseq", "parseq", pretrained=True)
        except Exception as exc:
            raise OcrUnavailable(f"failed to load PARSeq weights: {exc}") from exc
        self._model = model.eval().to(self.device)
        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 128), antialias=True),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def _recognize(self, pixels: bytes) -> tuple[str, float]:
        import torch

        self._ensure_model()
        image = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OcrUnavailable("could not decode image bytes")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = logits.softmax(-1)
            labels, confidences = self._model.tokenizer.decode(probs)
        text = labels[0]
        confidence = float(np.prod([c.item() for c in confidences[0]])) if len(confidences[0]) else 0.0
        return text, confidence

    def read(self, pixels: bytes) -> PlateReading:
        text, confidence = self._recognize(pixels)
        return guard_reading(text, confidence, self.min_confidence)
