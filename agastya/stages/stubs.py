from __future__ import annotations

from agastya.types import BBox, Detection, PlateReading


class StubGate:
    def __init__(self, score: float) -> None:
        self.score = score

    def score_image(self, pixels: bytes) -> float:
        return self.score


class StubRestorer:
    def __init__(self) -> None:
        self.calls = 0

    def restore(self, pixels: bytes) -> bytes:
        self.calls += 1
        return pixels


class StubDetector:
    def detect(self, pixels: bytes) -> list[Detection]:
        return [
            Detection(label="motorcycle", score=0.95, box=BBox(0.0, 0.0, 4.0, 2.0)),
            Detection(label="no-helmet", score=0.88, box=BBox(0.5, 0.0, 1.5, 1.0)),
            Detection(label="license-plate", score=0.91, box=BBox(1.0, 1.5, 2.0, 2.0)),
        ]


class StubOCR:
    def __init__(self, text: str, confidence: float) -> None:
        self.text = text
        self.confidence = confidence

    def read(self, pixels: bytes) -> PlateReading:
        return PlateReading(text=self.text, confidence=self.confidence)
