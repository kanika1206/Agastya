from __future__ import annotations

from agastya.types import BBox


def bbox_to_yolo(
    box: BBox, image_width: float, image_height: float
) -> tuple[float, float, float, float]:
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("image dimensions must be positive")
    cx = ((box.x1 + box.x2) / 2.0) / image_width
    cy = ((box.y1 + box.y2) / 2.0) / image_height
    w = (box.x2 - box.x1) / image_width
    h = (box.y2 - box.y1) / image_height
    return cx, cy, w, h


def yolo_to_bbox(
    cx: float, cy: float, w: float, h: float, image_width: float, image_height: float
) -> BBox:
    half_w = (w * image_width) / 2.0
    half_h = (h * image_height) / 2.0
    center_x = cx * image_width
    center_y = cy * image_height
    return BBox(
        x1=center_x - half_w,
        y1=center_y - half_h,
        x2=center_x + half_w,
        y2=center_y + half_h,
    )


def format_label_line(class_id: int, cx: float, cy: float, w: float, h: float) -> str:
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def parse_label_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}: {line!r}")
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
