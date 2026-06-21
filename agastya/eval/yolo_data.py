from __future__ import annotations

from pathlib import Path

import yaml

from agastya.types import BBox, Detection


def load_data_yaml(path: Path) -> tuple[Path, list[str]]:
    spec = yaml.safe_load(path.read_text())
    root = Path(spec["path"])
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    val_dir = root / spec["val"]
    names = spec["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return val_dir, names


def label_path_for(image_path: Path) -> Path:
    parts = [p if p != "images" else "labels" for p in image_path.parts]
    return Path(*parts).with_suffix(".txt")


def norm_box(cx: float, cy: float, w: float, h: float) -> BBox | None:
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1, y1, x2, y2)


def load_truths(label_path: Path, names: list[str]) -> list[Detection]:
    truths: list[Detection] = []
    if not label_path.exists():
        return truths
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        box = norm_box(cx, cy, w, h)
        if box is not None:
            truths.append(Detection(label=names[cls], score=1.0, box=box))
    return truths


def predictions_for(result, names: list[str]) -> list[Detection]:
    preds: list[Detection] = []
    for box in result.boxes:
        cx, cy, w, h = (float(v) for v in box.xywhn[0])
        bbox = norm_box(cx, cy, w, h)
        if bbox is None:
            continue
        preds.append(
            Detection(label=names[int(box.cls)], score=float(box.conf), box=bbox)
        )
    return preds
