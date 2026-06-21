from __future__ import annotations

from dataclasses import dataclass, field

from agastya.stages.violations.geometry import Point

DEFAULT_VEHICLE_LABELS = ("motorcycle", "car", "truck", "bus", "auto-rickshaw")
SEATBELT_LABELS = ("no-seatbelt", "seatbelt-absent")
SIGNAL_RED = "red"


@dataclass(frozen=True)
class StopLine:
    a: Point
    b: Point
    violation_sign: float = 1.0


@dataclass(frozen=True)
class SceneContext:
    no_parking_zones: tuple[tuple[Point, ...], ...] = ()
    stop_line: StopLine | None = None
    signal_state: str | None = None
    allowed_direction: Point | None = None
    vehicle_labels: tuple[str, ...] = DEFAULT_VEHICLE_LABELS
    headings: dict[int, Point] = field(default_factory=dict)
