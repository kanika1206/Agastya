from __future__ import annotations

import sys
from pathlib import Path

import cv2

from agastya.config import PipelineConfig
from agastya.stages.associate.factory import build_associator
from agastya.types import BBox, Detection

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/raw/triple/test/images"
LABELS = ROOT / "data/raw/triple/test/labels"

MOTO_CLASS = 0
RIDER_CLASSES = {3, 4}


def _yolo_to_bbox(parts: list[str], width: int, height: int) -> BBox:
    cx, cy, bw, bh = (float(p) for p in parts[1:5])
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    return BBox(x1, y1, x2, y2)


def _pick_sample() -> tuple[Path, BBox, list[Detection]]:
    for label_path in sorted(LABELS.glob("*.txt")):
        image_path = IMAGES / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        moto: BBox | None = None
        persons: list[Detection] = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            box = _yolo_to_bbox(parts, width, height)
            if cls == MOTO_CLASS and moto is None:
                moto = box
            elif cls in RIDER_CLASSES:
                persons.append(Detection(label="person", score=0.9, box=box))
        if moto is not None and len(persons) >= 3:
            return image_path, moto, persons
    raise SystemExit("no triple-riding sample with motorcycle + 3 rider boxes found")


def main() -> int:
    image_path, moto, persons = _pick_sample()
    pixels = image_path.read_bytes()

    config = PipelineConfig(associate_backend="sam2", triple_riding_overlap=0.1)
    associator = build_associator(config)

    boxes = [moto, *(p.box for p in persons)]
    masks = associator._segment(pixels, boxes)
    triple = associator.is_triple_riding(moto, persons, pixels)

    print(f"image: {image_path.name}")
    print(f"box prompts: {len(boxes)} (1 motorcycle + {len(persons)} riders)")
    print(f"masks returned: {len(masks)}")
    print(f"predictor cached: {associator._predictor is not None}")
    print(f"is_triple_riding: {triple}")

    if len(masks) != len(boxes):
        print("FAIL: mask count != box-prompt count")
        return 1
    print("PASS: box-prompt -> mask seam validated with real SAM weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
