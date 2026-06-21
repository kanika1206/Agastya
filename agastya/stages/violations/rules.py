from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agastya.stages.violations.geometry import (
    box_bottom_center,
    heading_is_opposite,
    point_in_polygon,
    signed_side,
)
from agastya.stages.violations.scene import (
    SEATBELT_LABELS,
    SIGNAL_RED,
    SceneContext,
)
from agastya.types import Detection


@dataclass(frozen=True)
class ViolationCandidate:
    violation_type: str
    score: float
    detections: tuple[Detection, ...]


def _vehicles(detections: Sequence[Detection], scene: SceneContext) -> list[Detection]:
    return [d for d in detections if d.label in scene.vehicle_labels]


def detect_seatbelt(
    detections: Sequence[Detection], min_conf: float
) -> list[ViolationCandidate]:
    return [
        ViolationCandidate("seatbelt", d.score, (d,))
        for d in detections
        if d.label in SEATBELT_LABELS and d.score >= min_conf
    ]


def detect_illegal_parking(
    detections: Sequence[Detection], scene: SceneContext
) -> list[ViolationCandidate]:
    if not scene.no_parking_zones:
        return []
    candidates: list[ViolationCandidate] = []
    for vehicle in _vehicles(detections, scene):
        point = box_bottom_center(vehicle.box)
        if any(point_in_polygon(point, zone) for zone in scene.no_parking_zones):
            candidates.append(ViolationCandidate("illegal-parking", vehicle.score, (vehicle,)))
    return candidates


def _past_stop_line(detections: Sequence[Detection], scene: SceneContext) -> list[Detection]:
    line = scene.stop_line
    if line is None:
        return []
    past: list[Detection] = []
    for vehicle in _vehicles(detections, scene):
        side = signed_side(box_bottom_center(vehicle.box), line.a, line.b)
        if side * line.violation_sign > 0.0:
            past.append(vehicle)
    return past


def detect_stop_line(
    detections: Sequence[Detection], scene: SceneContext
) -> list[ViolationCandidate]:
    return [
        ViolationCandidate("stop-line", v.score, (v,))
        for v in _past_stop_line(detections, scene)
    ]


def detect_red_light(
    detections: Sequence[Detection], scene: SceneContext
) -> list[ViolationCandidate]:
    if scene.signal_state != SIGNAL_RED:
        return []
    return [
        ViolationCandidate("red-light", v.score, (v,))
        for v in _past_stop_line(detections, scene)
    ]


def detect_wrong_side(
    detections: Sequence[Detection], scene: SceneContext
) -> list[ViolationCandidate]:
    if scene.allowed_direction is None or not scene.headings:
        return []
    candidates: list[ViolationCandidate] = []
    for index, vehicle in enumerate(detections):
        if vehicle.label not in scene.vehicle_labels:
            continue
        heading = scene.headings.get(index)
        if heading is not None and heading_is_opposite(heading, scene.allowed_direction):
            candidates.append(ViolationCandidate("wrong-side", vehicle.score, (vehicle,)))
    return candidates


def detect_scene_violations(
    detections: Sequence[Detection], scene: SceneContext, *, seatbelt_min_conf: float
) -> list[ViolationCandidate]:
    return [
        *detect_seatbelt(detections, seatbelt_min_conf),
        *detect_illegal_parking(detections, scene),
        *detect_stop_line(detections, scene),
        *detect_red_light(detections, scene),
        *detect_wrong_side(detections, scene),
    ]
