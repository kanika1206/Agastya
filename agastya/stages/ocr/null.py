from __future__ import annotations

from agastya.types import PlateReading


class NullOcr:
    def read(self, pixels: bytes) -> PlateReading:
        return PlateReading(text="", confidence=0.0, abstained=True)
