from __future__ import annotations

from agastya.types import BBox

Point = tuple[float, float]


def box_bottom_center(box: BBox) -> Point:
    return ((box.x1 + box.x2) / 2.0, box.y2)


def box_centroid(box: BBox) -> Point:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def signed_side(point: Point, a: Point, b: Point) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def heading_is_opposite(heading: Point, allowed: Point) -> bool:
    dot = heading[0] * allowed[0] + heading[1] * allowed[1]
    return dot < 0.0
