from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bbox coordinates inverted")

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def intersection(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def iou(self, other: BBox) -> float:
        inter = self.intersection(other)
        union = self.area() + other.area() - inter
        if union <= 0.0:
            return 0.0
        return inter / union


@dataclass(frozen=True)
class QualityScore:
    value: float
    degraded: bool


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: BBox


@dataclass(frozen=True)
class PlateReading:
    text: str
    confidence: float
    abstained: bool = False


@dataclass(frozen=True)
class ViolationRecord:
    violation_type: str
    confidence: float
    plate: PlateReading | None
    detections: tuple[Detection, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)
