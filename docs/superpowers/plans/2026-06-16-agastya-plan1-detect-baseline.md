# AGASTYA Plan 1 — Detect Baseline + Data Assembly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data-assembly and detection-baseline layer of AGASTYA — pure, tested logic for unifying IDD / AI City Track 5 / ANPR labels into one YOLO-format schema, a SAHI slice-merge with explicit NMS (the YOLO26 NMS-free caveat), and the detector/eval adapters — plus a clearly-gated operational runbook for download, YOLO26-m baseline training, and mAP evaluation that does NOT run without the user's written "go".

**Architecture:** Detection work splits into testable pure logic (label-schema mapping, YOLO label I/O, dataset manifest + deterministic split, NMS, SAHI merge, ultralytics-results adapter, precision/recall/F1 matching) and gated operations (dataset download, dataset build, training, SAHI↔NMS validation, mAP eval). Pure logic is built TDD now; operations are scripts with exact commands and expected artifacts, run only after explicit approval. Builds directly on the `agastya` package from Plan 0 (reuses `BBox`, `Detection`, `CLASSES`).

**Tech Stack:** Python 3.10, pydantic v2, pytest, ruff (from Plan 0). New runtime deps for operations only: `ultralytics` (YOLO26), `sahi`, `pyyaml`. Project rule: **zero code comments** anywhere in this repo.

---

## CRITICAL: Gating Rule

Tasks tagged **[GATED]** perform downloads, training, or large compute. Do NOT execute them during plan implementation. Implement the script/runbook file, verify it imports and `--help`/dry-run works, commit it, and STOP — surface to the user for written "go" before any real run. Tasks tagged **[TDD]** are pure logic with synthetic fixtures and are implemented and run normally.

---

### Task 1: YOLO label format I/O [TDD]

**Files:**
- Create: `agastya/data/__init__.py`
- Create: `agastya/data/yolo_format.py`
- Test: `tests/data/__init__.py`
- Test: `tests/data/test_yolo_format.py`

- [ ] **Step 1: Write the failing test**

`tests/data/__init__.py`:
```python
```

`tests/data/test_yolo_format.py`:
```python
import pytest

from agastya.data.yolo_format import (
    bbox_to_yolo,
    format_label_line,
    parse_label_line,
    yolo_to_bbox,
)
from agastya.types import BBox


def test_bbox_to_yolo_centers_and_normalizes():
    box = BBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
    cx, cy, w, h = bbox_to_yolo(box, image_width=100.0, image_height=200.0)
    assert cx == pytest.approx(0.2)
    assert cy == pytest.approx(0.2)
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.2)


def test_yolo_to_bbox_roundtrip():
    box = BBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
    cx, cy, w, h = bbox_to_yolo(box, 100.0, 200.0)
    restored = yolo_to_bbox(cx, cy, w, h, 100.0, 200.0)
    assert restored.x1 == pytest.approx(box.x1)
    assert restored.y1 == pytest.approx(box.y1)
    assert restored.x2 == pytest.approx(box.x2)
    assert restored.y2 == pytest.approx(box.y2)


def test_format_label_line_has_five_fields():
    line = format_label_line(3, 0.5, 0.5, 0.2, 0.4)
    assert line == "3 0.500000 0.500000 0.200000 0.400000"


def test_parse_label_line_roundtrip():
    class_id, cx, cy, w, h = parse_label_line("3 0.5 0.5 0.2 0.4")
    assert class_id == 3
    assert (cx, cy, w, h) == (0.5, 0.5, 0.2, 0.4)


def test_parse_rejects_malformed():
    with pytest.raises(ValueError):
        parse_label_line("3 0.5 0.5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/data/test_yolo_format.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.data'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/data/__init__.py`:
```python
```

`agastya/data/yolo_format.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/data/test_yolo_format.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/student/Flipkart && git add agastya/data/__init__.py agastya/data/yolo_format.py tests/data && git commit -m "feat: yolo label format io"
```

---

### Task 2: Source label schema mapping [TDD]

**Files:**
- Create: `agastya/data/schema_map.py`
- Test: `tests/data/test_schema_map.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_schema_map.py`:
```python
import pytest

from agastya.data.schema_map import map_source_label
from agastya.schema.classes import name_to_id


def test_aicity_motorbike_maps_to_motorcycle():
    assert map_source_label("aicity", "motorbike") == name_to_id("motorcycle")


def test_aicity_dnohelmet_maps_to_no_helmet():
    assert map_source_label("aicity", "DNoHelmet") == name_to_id("no-helmet")


def test_aicity_dhelmet_maps_to_helmet():
    assert map_source_label("aicity", "DHelmet") == name_to_id("helmet")


def test_idd_autorickshaw_maps():
    assert map_source_label("idd", "autorickshaw") == name_to_id("auto-rickshaw")


def test_unknown_source_label_returns_none():
    assert map_source_label("idd", "animal") is None


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        map_source_label("kitti", "car")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/data/test_schema_map.py -v`
Expected: FAIL — `ImportError: cannot import name 'map_source_label'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/data/schema_map.py`:
```python
from __future__ import annotations

from agastya.schema.classes import name_to_id

_IDD_MAP: dict[str, str] = {
    "motorcycle": "motorcycle",
    "rider": "rider",
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "autorickshaw": "auto-rickshaw",
}

_AICITY_MAP: dict[str, str] = {
    "motorbike": "motorcycle",
    "DHelmet": "helmet",
    "DNoHelmet": "no-helmet",
    "P1Helmet": "helmet",
    "P1NoHelmet": "no-helmet",
    "P2Helmet": "helmet",
    "P2NoHelmet": "no-helmet",
    "P0Helmet": "helmet",
    "P0NoHelmet": "no-helmet",
}

_ANPR_MAP: dict[str, str] = {
    "license-plate": "license-plate",
    "licence_plate": "license-plate",
    "plate": "license-plate",
}

_SOURCES: dict[str, dict[str, str]] = {
    "idd": _IDD_MAP,
    "aicity": _AICITY_MAP,
    "anpr": _ANPR_MAP,
}


def map_source_label(source: str, name: str) -> int | None:
    if source not in _SOURCES:
        raise ValueError(f"unknown dataset source: {source}")
    unified = _SOURCES[source].get(name)
    if unified is None:
        return None
    return name_to_id(unified)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/data/test_schema_map.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/student/Flipkart && git add agastya/data/schema_map.py tests/data/test_schema_map.py && git commit -m "feat: source-to-unified label schema mapping"
```

---

### Task 3: Dataset manifest + deterministic split [TDD]

**Files:**
- Create: `agastya/data/manifest.py`
- Test: `tests/data/test_manifest.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_manifest.py`:
```python
from agastya.data.manifest import DatasetItem, assign_split, build_manifest


def test_assign_split_is_deterministic():
    first = assign_split("img-0001", val_fraction=0.2)
    second = assign_split("img-0001", val_fraction=0.2)
    assert first == second
    assert first in {"train", "val"}


def test_assign_split_respects_fraction_roughly():
    ids = [f"img-{i:05d}" for i in range(2000)]
    val = sum(1 for i in ids if assign_split(i, val_fraction=0.2) == "val")
    assert 300 <= val <= 500


def test_build_manifest_tags_source_and_split():
    items = build_manifest(
        [("idd", "/data/idd/a.jpg"), ("aicity", "/data/aicity/b.jpg")],
        val_fraction=0.2,
    )
    assert all(isinstance(item, DatasetItem) for item in items)
    assert {item.source for item in items} == {"idd", "aicity"}
    assert all(item.split in {"train", "val"} for item in items)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/data/test_manifest.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_manifest'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/data/manifest.py`:
```python
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

_SPLIT_RESOLUTION = 10_000


@dataclass(frozen=True)
class DatasetItem:
    source: str
    image_path: str
    split: str


def assign_split(image_id: str, val_fraction: float) -> str:
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be in [0, 1]")
    digest = hashlib.sha256(image_id.encode()).hexdigest()
    bucket = int(digest, 16) % _SPLIT_RESOLUTION
    return "val" if bucket < val_fraction * _SPLIT_RESOLUTION else "train"


def build_manifest(
    entries: Iterable[tuple[str, str]], val_fraction: float
) -> tuple[DatasetItem, ...]:
    items: list[DatasetItem] = []
    for source, image_path in entries:
        split = assign_split(image_path, val_fraction)
        items.append(DatasetItem(source=source, image_path=image_path, split=split))
    return tuple(items)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/data/test_manifest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/student/Flipkart && git add agastya/data/manifest.py tests/data/test_manifest.py && git commit -m "feat: dataset manifest with deterministic hash split"
```

---

### Task 4: Ultralytics data.yaml writer [TDD]

**Files:**
- Create: `agastya/data/data_yaml.py`
- Test: `tests/data/test_data_yaml.py`

- [ ] **Step 1: Write the failing test**

`tests/data/test_data_yaml.py`:
```python
from agastya.data.data_yaml import write_data_yaml
from agastya.schema.classes import CLASSES


def test_write_data_yaml_contains_classes_and_paths(tmp_path):
    out = tmp_path / "data.yaml"
    write_data_yaml(out, root=tmp_path, train_dir="images/train", val_dir="images/val")
    text = out.read_text()
    assert "nc: 10" in text
    for name in CLASSES:
        assert name in text
    assert "images/train" in text
    assert "images/val" in text


def test_write_data_yaml_is_parseable(tmp_path):
    import yaml

    out = tmp_path / "data.yaml"
    write_data_yaml(out, root=tmp_path, train_dir="images/train", val_dir="images/val")
    parsed = yaml.safe_load(out.read_text())
    assert parsed["nc"] == len(CLASSES)
    assert parsed["names"] == list(CLASSES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/data/test_data_yaml.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.data.data_yaml'` (and possibly `yaml` missing; install with `pip install pyyaml`).

- [ ] **Step 3: Write minimal implementation**

First ensure dep: `cd /home/student/Flipkart && pip install pyyaml` and add `"pyyaml>=6.0"` to `dependencies` in `pyproject.toml`.

`agastya/data/data_yaml.py`:
```python
from __future__ import annotations

from pathlib import Path

import yaml

from agastya.schema.classes import CLASSES


def write_data_yaml(out_path: Path, root: Path, train_dir: str, val_dir: str) -> None:
    payload = {
        "path": str(root),
        "train": train_dir,
        "val": val_dir,
        "nc": len(CLASSES),
        "names": list(CLASSES),
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/data/test_data_yaml.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/student/Flipkart && git add agastya/data/data_yaml.py tests/data/test_data_yaml.py pyproject.toml && git commit -m "feat: ultralytics data.yaml writer"
```

---

### Task 5: Non-max suppression [TDD]

**Files:**
- Create: `agastya/detect/__init__.py`
- Create: `agastya/detect/nms.py`
- Test: `tests/detect/__init__.py`
- Test: `tests/detect/test_nms.py`

- [ ] **Step 1: Write the failing test**

`tests/detect/__init__.py`:
```python
```

`tests/detect/test_nms.py`:
```python
from agastya.detect.nms import non_max_suppression
from agastya.types import BBox, Detection


def _det(label: str, score: float, x1: float) -> Detection:
    return Detection(label=label, score=score, box=BBox(x1, 0.0, x1 + 2.0, 2.0))


def test_nms_suppresses_lower_score_overlap():
    high = _det("helmet", 0.9, 0.0)
    low = _det("helmet", 0.6, 0.2)
    kept = non_max_suppression([low, high], iou_threshold=0.5)
    assert kept == [high]


def test_nms_keeps_disjoint_boxes():
    a = _det("helmet", 0.9, 0.0)
    b = _det("helmet", 0.8, 50.0)
    kept = non_max_suppression([a, b], iou_threshold=0.5)
    assert set(kept) == {a, b}


def test_nms_is_per_label():
    helmet = _det("helmet", 0.9, 0.0)
    person = _det("person", 0.8, 0.1)
    kept = non_max_suppression([helmet, person], iou_threshold=0.5)
    assert set(kept) == {helmet, person}


def test_nms_empty_returns_empty():
    assert non_max_suppression([], iou_threshold=0.5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/detect/test_nms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.detect'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/detect/__init__.py`:
```python
```

`agastya/detect/nms.py`:
```python
from __future__ import annotations

from agastya.types import Detection


def non_max_suppression(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        suppressed = False
        for keeper in kept:
            if keeper.label != candidate.label:
                continue
            if keeper.box.iou(candidate.box) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/detect/test_nms.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/student/Flipkart && git add agastya/detect/__init__.py agastya/detect/nms.py tests/detect && git commit -m "feat: per-label non-max suppression"
```

---

### Task 6: SAHI slice merge with explicit NMS [TDD]

**Files:**
- Create: `agastya/detect/sahi_merge.py`
- Test: `tests/detect/test_sahi_merge.py`

This is the concrete handling of the YOLO26 NMS-free caveat from the design: slice predictions are shifted to global coordinates and merged with explicit NMS, since the one-to-many detector branch does not deduplicate across slice seams.

- [ ] **Step 1: Write the failing test**

`tests/detect/test_sahi_merge.py`:
```python
from agastya.detect.sahi_merge import SlicePrediction, merge_slice_predictions, shift_detection
from agastya.types import BBox, Detection


def test_shift_detection_offsets_box():
    det = Detection(label="helmet", score=0.9, box=BBox(0.0, 0.0, 2.0, 2.0))
    shifted = shift_detection(det, offset_x=10.0, offset_y=20.0)
    assert shifted.box == BBox(10.0, 20.0, 12.0, 22.0)
    assert shifted.label == "helmet"
    assert shifted.score == 0.9


def test_merge_dedupes_overlapping_seam_detections():
    slice_a = SlicePrediction(
        offset_x=0.0,
        offset_y=0.0,
        detections=[Detection("helmet", 0.7, BBox(98.0, 0.0, 102.0, 4.0))],
    )
    slice_b = SlicePrediction(
        offset_x=100.0,
        offset_y=0.0,
        detections=[Detection("helmet", 0.9, BBox(-2.0, 0.0, 2.0, 4.0))],
    )
    merged = merge_slice_predictions([slice_a, slice_b], iou_threshold=0.3)
    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_merge_keeps_distinct_detections():
    slice_a = SlicePrediction(
        offset_x=0.0, offset_y=0.0, detections=[Detection("helmet", 0.8, BBox(0.0, 0.0, 4.0, 4.0))]
    )
    slice_b = SlicePrediction(
        offset_x=200.0,
        offset_y=0.0,
        detections=[Detection("no-helmet", 0.8, BBox(0.0, 0.0, 4.0, 4.0))],
    )
    merged = merge_slice_predictions([slice_a, slice_b], iou_threshold=0.5)
    assert len(merged) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/detect/test_sahi_merge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.detect.sahi_merge'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/detect/sahi_merge.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from agastya.detect.nms import non_max_suppression
from agastya.types import BBox, Detection


@dataclass(frozen=True)
class SlicePrediction:
    offset_x: float
    offset_y: float
    detections: list[Detection]


def shift_detection(detection: Detection, offset_x: float, offset_y: float) -> Detection:
    box = detection.box
    return Detection(
        label=detection.label,
        score=detection.score,
        box=BBox(
            x1=box.x1 + offset_x,
            y1=box.y1 + offset_y,
            x2=box.x2 + offset_x,
            y2=box.y2 + offset_y,
        ),
    )


def merge_slice_predictions(
    slices: list[SlicePrediction], iou_threshold: float
) -> list[Detection]:
    globalized: list[Detection] = []
    for sl in slices:
        for det in sl.detections:
            globalized.append(shift_detection(det, sl.offset_x, sl.offset_y))
    return non_max_suppression(globalized, iou_threshold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/detect/test_sahi_merge.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/student/Flipkart && git add agastya/detect/sahi_merge.py tests/detect/test_sahi_merge.py && git commit -m "feat: sahi slice merge with explicit nms"
```

---

### Task 7: Ultralytics results adapter [TDD]

**Files:**
- Create: `agastya/detect/results_adapter.py`
- Test: `tests/detect/test_results_adapter.py`

Pure conversion from raw ultralytics-style arrays to `Detection` objects, so the detector wrapper (Task 8) stays a thin shell and the mapping logic is tested without importing torch/ultralytics.

- [ ] **Step 1: Write the failing test**

`tests/detect/test_results_adapter.py`:
```python
import pytest

from agastya.detect.results_adapter import boxes_to_detections
from agastya.types import BBox


def test_boxes_to_detections_maps_names():
    dets = boxes_to_detections(
        xyxy=[[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 7.0, 9.0]],
        class_ids=[2, 4],
        scores=[0.9, 0.7],
        names={2: "helmet", 4: "person"},
    )
    assert dets[0].label == "helmet"
    assert dets[0].box == BBox(0.0, 0.0, 2.0, 2.0)
    assert dets[1].label == "person"
    assert dets[1].score == 0.7


def test_boxes_to_detections_length_mismatch_raises():
    with pytest.raises(ValueError):
        boxes_to_detections(xyxy=[[0.0, 0.0, 1.0, 1.0]], class_ids=[2, 4], scores=[0.9], names={})


def test_boxes_to_detections_unknown_id_raises():
    with pytest.raises(KeyError):
        boxes_to_detections(
            xyxy=[[0.0, 0.0, 1.0, 1.0]], class_ids=[99], scores=[0.9], names={2: "helmet"}
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/detect/test_results_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.detect.results_adapter'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/detect/results_adapter.py`:
```python
from __future__ import annotations

from collections.abc import Sequence

from agastya.types import BBox, Detection


def boxes_to_detections(
    xyxy: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    scores: Sequence[float],
    names: dict[int, str],
) -> list[Detection]:
    if not len(xyxy) == len(class_ids) == len(scores):
        raise ValueError("xyxy, class_ids, and scores must have equal length")
    detections: list[Detection] = []
    for coords, class_id, score in zip(xyxy, class_ids, scores):
        label = names[class_id]
        detections.append(
            Detection(
                label=label,
                score=float(score),
                box=BBox(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
            )
        )
    return detections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/detect/test_results_adapter.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/student/Flipkart && git add agastya/detect/results_adapter.py tests/detect/test_results_adapter.py && git commit -m "feat: ultralytics results to detection adapter"
```

---

### Task 8: Precision / recall / F1 matching [TDD]

**Files:**
- Create: `agastya/eval/__init__.py`
- Create: `agastya/eval/prf.py`
- Test: `tests/eval/__init__.py`
- Test: `tests/eval/test_prf.py`

mAP itself comes from `ultralytics model.val()` in the gated eval task. This module gives a transparent, testable P/R/F1 against ground truth via greedy IoU matching — used for the per-class control table and degradation-stratified reporting.

- [ ] **Step 1: Write the failing test**

`tests/eval/__init__.py`:
```python
```

`tests/eval/test_prf.py`:
```python
import pytest

from agastya.eval.prf import match_detections, precision_recall_f1
from agastya.types import BBox, Detection


def _det(label: str, x1: float, score: float = 0.9) -> Detection:
    return Detection(label=label, score=score, box=BBox(x1, 0.0, x1 + 2.0, 2.0))


def test_precision_recall_f1_basic():
    p, r, f1 = precision_recall_f1(tp=8, fp=2, fn=2)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)


def test_prf_zero_predictions_is_zero():
    p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=5)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0


def test_match_counts_tp_fp_fn():
    preds = [_det("helmet", 0.0), _det("helmet", 10.0), _det("helmet", 50.0)]
    truths = [_det("helmet", 0.1), _det("helmet", 10.1)]
    tp, fp, fn = match_detections(preds, truths, iou_threshold=0.3)
    assert tp == 2
    assert fp == 1
    assert fn == 0


def test_match_respects_label():
    preds = [_det("helmet", 0.0)]
    truths = [_det("no-helmet", 0.0)]
    tp, fp, fn = match_detections(preds, truths, iou_threshold=0.3)
    assert (tp, fp, fn) == (0, 1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/eval/test_prf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.eval'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/eval/__init__.py`:
```python
```

`agastya/eval/prf.py`:
```python
from __future__ import annotations

from agastya.types import Detection


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def match_detections(
    predictions: list[Detection], truths: list[Detection], iou_threshold: float
) -> tuple[int, int, int]:
    ordered = sorted(predictions, key=lambda d: d.score, reverse=True)
    matched: set[int] = set()
    tp = 0
    fp = 0
    for pred in ordered:
        best_idx = -1
        best_iou = iou_threshold
        for idx, truth in enumerate(truths):
            if idx in matched or truth.label != pred.label:
                continue
            iou = pred.box.iou(truth.box)
            if iou >= best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            matched.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(truths) - len(matched)
    return tp, fp, fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python3 -m pytest tests/eval/test_prf.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/student/Flipkart && git add agastya/eval/__init__.py agastya/eval/prf.py tests/eval && git commit -m "feat: precision recall f1 with greedy iou matching"
```

---

### Task 9: Full suite green + lint

**Files:** None (verification)

- [ ] **Step 1: Run the whole suite**

Run: `cd /home/student/Flipkart && python3 -m pytest -q`
Expected: PASS — all Plan 0 + Plan 1 pure-logic tests.

- [ ] **Step 2: Lint**

Run: `cd /home/student/Flipkart && python3 -m ruff check agastya tests`
Expected: `All checks passed!`

- [ ] **Step 3: Commit any lint fixes**

```bash
cd /home/student/Flipkart && git add -A && git commit -m "chore: lint clean" || true
```

---

### Task 10: Dataset download runbook [GATED]

**Files:**
- Create: `scripts/download_datasets.sh`
- Create: `docs/runbooks/datasets.md`

Write the script and runbook only. Do NOT run any download. The script must require an explicit `--confirm` flag and print sizes/sources before fetching.

- [ ] **Step 1: Write `scripts/download_datasets.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIRM="${1:-}"

DATA_ROOT="${AGASTYA_DATA_ROOT:-data/raw}"

echo "AGASTYA dataset download plan"
echo "  target root: ${DATA_ROOT}"
echo "  IDD / IDD-Detection : register + download from https://idd.insaan.iiit.ac.in/ (manual auth)"
echo "  AI City 2024 Track5 : request access via https://www.aicitychallenge.org/ (manual auth)"
echo "  Indian ANPR set     : Kaggle indian-license-plate datasets (kaggle CLI)"

if [ "${CONFIRM}" != "--confirm" ]; then
  echo
  echo "Dry run only. Re-run with --confirm to fetch the Kaggle ANPR set."
  echo "IDD and AI City require manual registration and cannot be auto-fetched."
  exit 0
fi

mkdir -p "${DATA_ROOT}/anpr"
kaggle datasets download -d andrewmvd/car-plate-detection -p "${DATA_ROOT}/anpr" --unzip
echo "ANPR download complete: ${DATA_ROOT}/anpr"
```

- [ ] **Step 2: Write `docs/runbooks/datasets.md`**

Document, in prose: the three sources, that IDD and AI City Track 5 require manual registration/access approval (no automated download), the Kaggle CLI auth step (`~/.kaggle/kaggle.json`), the expected on-disk layout under `data/raw/{idd,aicity,anpr}/`, and the licensing/usage terms to respect for each. State clearly that the IDD motorcycle-violation derivative (arXiv:2204.08364) provides helmet/trapezium labels to reuse.

- [ ] **Step 3: Verify dry-run only (no download)**

Run: `cd /home/student/Flipkart && bash scripts/download_datasets.sh`
Expected: prints the plan and "Dry run only" then exits 0. No network fetch.

- [ ] **Step 4: Commit**

```bash
cd /home/student/Flipkart && chmod +x scripts/download_datasets.sh && git add scripts/download_datasets.sh docs/runbooks/datasets.md && git commit -m "feat: gated dataset download runbook"
```

- [ ] **Step 5: STOP — surface to user**

Report that real dataset acquisition (IDD/AI City registration, Kaggle ANPR fetch) needs explicit "go". Do not proceed.

---

### Task 11: Dataset build entrypoint [GATED]

**Files:**
- Create: `scripts/build_dataset.py`

Assembles the unified YOLO dataset using the tested `agastya.data` modules. Implement and verify it runs `--help` and a `--dry-run` over an empty/synthetic tree; do NOT run it over real data without "go".

- [ ] **Step 1: Write `scripts/build_dataset.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path

from agastya.data.data_yaml import write_data_yaml
from agastya.data.manifest import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble unified AGASTYA YOLO dataset")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_images(raw_root: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for source in ("idd", "aicity", "anpr"):
        source_dir = raw_root / source
        if not source_dir.exists():
            continue
        for image_path in source_dir.rglob("*.jpg"):
            entries.append((source, str(image_path)))
    return entries


def main() -> None:
    args = parse_args()
    entries = discover_images(args.raw_root)
    manifest = build_manifest(entries, val_fraction=args.val_fraction)
    print(f"discovered {len(manifest)} images across sources")
    if args.dry_run:
        print("dry run: no files written")
        return
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(
        args.out_root / "data.yaml",
        root=args.out_root,
        train_dir="images/train",
        val_dir="images/val",
    )
    print(f"wrote {args.out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
```

Note: this entrypoint writes `data.yaml` and the manifest; the per-image label conversion (using `bbox_to_yolo`, `map_source_label`, `format_label_line`) over real annotations is the part that runs only after "go" — extend `main()` to copy/symlink images and write label files per split at that time, since it depends on each source's annotation format on disk.

- [ ] **Step 2: Verify help + dry-run on empty tree**

Run: `cd /home/student/Flipkart && python3 scripts/build_dataset.py --raw-root /tmp/agastya_empty --out-root /tmp/agastya_out --dry-run`
Expected: prints `discovered 0 images across sources` and `dry run: no files written`, exits 0.

- [ ] **Step 3: Commit**

```bash
cd /home/student/Flipkart && git add scripts/build_dataset.py && git commit -m "feat: gated dataset build entrypoint"
```

- [ ] **Step 4: STOP — surface to user** that real label conversion needs each source's annotation parser plus "go".

---

### Task 12: YOLO26 baseline training entrypoint [GATED]

**Files:**
- Create: `scripts/train_baseline.py`
- Create: `configs/README.md`

Implement the training entrypoint and document the `yolo26-p2.yaml` usage. Do NOT train. Verify `--help` only.

- [ ] **Step 1: Write `scripts/train_baseline.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AGASTYA YOLO26 baseline")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo26-p2.yaml")
    parser.add_argument("--weights", default="yolo26m.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm:
        print("dry run: re-run with --confirm to start training")
        print(f"  data={args.data} model={args.model} weights={args.weights}")
        print(f"  imgsz={args.imgsz} epochs={args.epochs} batch={args.batch} device={args.device}")
        return
    from ultralytics import YOLO

    model = YOLO(args.model).load(args.weights)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        amp=True,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `configs/README.md`** documenting: `yolo26-p2.yaml` is instantiated from Ultralytics (P2 small-object head, no pretrained P2 weights — train from `yolo26m.pt` backbone load); ProgLoss + STAL + MuSGD are native to YOLO26; AMP on; auto-batch via `batch=-1`; low-VRAM swap to `yolo26-s` per `PipelineConfig` profile.

- [ ] **Step 3: Verify dry-run only**

Run: `cd /home/student/Flipkart && python3 scripts/train_baseline.py --data /tmp/none.yaml`
Expected: prints `dry run: re-run with --confirm to start training` and the config echo, exits 0. The `ultralytics` import sits inside the `--confirm` branch so the dry run needs no GPU/torch.

- [ ] **Step 4: Commit**

```bash
cd /home/student/Flipkart && git add scripts/train_baseline.py configs/README.md && git commit -m "feat: gated yolo26 baseline training entrypoint"
```

- [ ] **Step 5: STOP — surface to user** that training requires datasets assembled + GPU + written "go".

---

### Task 13: SAHI ↔ NMS-free validation entrypoint [GATED]

**Files:**
- Create: `scripts/validate_sahi.py`

The design's highest-risk item: confirm SAHI slicing works with YOLO26's NMS-free head by running the one-to-many branch (`end2end=False`) and merging slices with our explicit NMS (Task 6). Implement; verify `--help`; do NOT run without a trained model + "go".

- [ ] **Step 1: Write `scripts/validate_sahi.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAHI with YOLO26 NMS-free head")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--slice", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm:
        print("dry run: re-run with --confirm to execute SAHI validation")
        print(f"  weights={args.weights} image={args.image} slice={args.slice}")
        print("  will run end2end=False (one-to-many) and merge slices via explicit NMS")
        return
    from ultralytics import YOLO

    from agastya.detect.results_adapter import boxes_to_detections
    from agastya.detect.sahi_merge import SlicePrediction, merge_slice_predictions

    model = YOLO(str(args.weights))
    results = model.predict(source=str(args.image), imgsz=args.slice)
    names = results[0].names
    boxes = results[0].boxes
    dets = boxes_to_detections(
        xyxy=boxes.xyxy.tolist(),
        class_ids=[int(c) for c in boxes.cls.tolist()],
        scores=boxes.conf.tolist(),
        names=names,
    )
    merged = merge_slice_predictions(
        [SlicePrediction(offset_x=0.0, offset_y=0.0, detections=dets)], iou_threshold=args.iou
    )
    print(f"whole-image detections: {len(dets)}; merged: {len(merged)}")
    print("Next: tile the image, predict per tile with end2end=False, merge with merge_slice_predictions")
```

- [ ] **Step 2: Verify dry-run only**

Run: `cd /home/student/Flipkart && python3 scripts/validate_sahi.py --weights /tmp/none.pt --image /tmp/none.jpg`
Expected: prints `dry run: ...` and exits 0, no ultralytics import.

- [ ] **Step 3: Commit**

```bash
cd /home/student/Flipkart && git add scripts/validate_sahi.py && git commit -m "feat: gated sahi nms-free validation entrypoint"
```

- [ ] **Step 4: STOP — surface to user** that this needs a trained model + "go".

---

### Task 14: Phase 1 baseline runbook + control table template [GATED docs]

**Files:**
- Create: `docs/runbooks/phase1-baseline.md`

- [ ] **Step 1: Write `docs/runbooks/phase1-baseline.md`** documenting the end-to-end gated sequence and the metric control table to fill in after the real run:
  1. `bash scripts/download_datasets.sh --confirm` (after manual IDD/AI City access).
  2. `python3 scripts/build_dataset.py --raw-root data/raw --out-root data/processed` (after label parsers added).
  3. `python3 scripts/train_baseline.py --data data/processed/data.yaml --confirm`.
  4. `python3 scripts/validate_sahi.py --weights runs/.../best.pt --image <sample> --confirm`.
  5. `yolo val model=runs/.../best.pt data=data/processed/data.yaml` for mAP@50 / mAP@50–95.
  - Include an empty control table with columns: class, P, R, F1, AP@50, AP@50-95 — to be filled from the real run. Do NOT invent numbers. State that this baseline is the control against which every later novel block (restoration, gating, association, loss) must show a gain.

- [ ] **Step 2: Commit**

```bash
cd /home/student/Flipkart && git add docs/runbooks/phase1-baseline.md && git commit -m "docs: phase 1 baseline runbook and control table template"
```

---

## Self-Review

**Spec coverage (Plan 1 = design §7 Phase 1 + relevant §2/§5/§6):**
- Unify sources to one YOLO schema → Tasks 1 (label I/O), 2 (schema map), 3 (manifest/split), 4 (data.yaml), 11 (build entrypoint).
- Train YOLO26-m baseline (+P2, AMP, 640) → Task 12.
- Validate SAHI ↔ NMS-free → Tasks 6 (merge logic, tested) + 13 (gated end-to-end).
- Detector results → Detection → Task 7.
- Establish baseline mAP/P/R/F1 control → Task 8 (P/R/F1, tested) + Task 14 (mAP via ultralytics, gated control table).
- Dataset acquisition with licensing/access reality → Task 10.
- Deferred to later plans: restoration/gating (Plan 2), SAM2 association + OCR + conformal (Plan 3), C2PA/Grad-CAM/analytics + real Merkle proof (Plan 4), ablation detectors (RT-DETRv2/YOLOv11) and degradation-stratified sweep harness (Plan 2+).

**Placeholder scan:** No TBD/TODO. Every TDD task has complete code + expected output. GATED tasks contain complete scripts; the only intentionally-deferred logic (per-source annotation parsers, full tiling loop) is explicitly called out as needing each source's on-disk format + user "go", not hidden as a placeholder.

**Type consistency:** Reuses Plan 0 `BBox`, `Detection`, `CLASSES`, `name_to_id` unchanged. `non_max_suppression` (Task 5) consumed by `merge_slice_predictions` (Task 6). `boxes_to_detections` (Task 7) consumed by `validate_sahi.py` (Task 13). `build_manifest`/`write_data_yaml` (Tasks 3/4) consumed by `build_dataset.py` (Task 11). Signatures match across tasks.

**Gating integrity:** Every download/train/large-compute path is behind `--confirm`/`--dry-run`, imports `ultralytics` lazily inside the confirmed branch (so dry runs need no GPU/torch), and ends with an explicit STOP-and-surface step. No real data or training runs during plan implementation.
