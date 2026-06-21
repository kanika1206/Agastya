from __future__ import annotations


class AlwaysDegradedGate:
    def score_image(self, pixels: bytes) -> float:
        return 0.0


class NeverDegradedGate:
    def score_image(self, pixels: bytes) -> float:
        return 1.0
