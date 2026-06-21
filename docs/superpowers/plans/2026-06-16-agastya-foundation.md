# AGASTYA Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the AGASTYA repo skeleton — typed stage interfaces, GPU-free pure logic (quality-gate routing, triple-riding association, Merkle evidence chain, config/VRAM profiles), and a pipeline orchestrator that runs the full cascade with stub stages — all under test, with zero model/download dependencies.

**Architecture:** A four-stage Restore→Detect→Verify cascade expressed as typed `Stage` units wired by a `Pipeline` orchestrator. Foundation ships every stage as a swappable interface plus the pure-Python decision logic that does not need a GPU; heavy model stages land as stubs now and real implementations in later plans. Files are small and single-responsibility; data crossing stage boundaries uses frozen dataclasses.

**Tech Stack:** Python 3.11, pydantic v2 (configs), pytest (tests), ruff (lint). No PyTorch/CUDA in this plan. Project rule: **zero code comments** anywhere in this repo.

---

### Task 0: Initialize git (run once)

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Init repo if not already**

Run: `cd /home/student/Flipkart && git rev-parse --is-inside-work-tree 2>/dev/null || git init`
Expected: either `true`, or `Initialized empty Git repository`.

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.venv/
venv/
*.egg-info/
dist/
build/
data/raw/
data/processed/
weights/
runs/
*.pt
*.onnx
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: init repo and gitignore"
```

---

### Task 1: Project scaffold + tooling

**Files:**
- Create: `pyproject.toml`
- Create: `agastya/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
import agastya


def test_package_version_present():
    assert isinstance(agastya.__version__, str)
    assert agastya.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya'`.

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "agastya"
version = "0.1.0"
description = "Quality-adaptive, evidence-grade traffic-violation vision system"
requires-python = ">=3.11"
dependencies = ["pydantic>=2.6"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["agastya*"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Write package init files**

`agastya/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`:
```python
```

- [ ] **Step 5: Install editable + run test to verify it passes**

Run: `cd /home/student/Flipkart && pip install -e ".[dev]" && python -m pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml agastya/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "feat: project scaffold and smoke test"
```

---

### Task 2: Class schema

**Files:**
- Create: `agastya/schema/__init__.py`
- Create: `agastya/schema/classes.py`
- Test: `tests/schema/test_classes.py`

- [ ] **Step 1: Write the failing test**

`tests/schema/__init__.py`:
```python
```

`tests/schema/test_classes.py`:
```python
import pytest

from agastya.schema.classes import CLASSES, name_to_id, id_to_name, validate_class


def test_ten_classes_in_fixed_order():
    assert CLASSES == (
        "motorcycle",
        "rider",
        "helmet",
        "no-helmet",
        "person",
        "car",
        "truck",
        "bus",
        "auto-rickshaw",
        "license-plate",
    )


def test_name_id_roundtrip():
    for idx, name in enumerate(CLASSES):
        assert name_to_id(name) == idx
        assert id_to_name(idx) == name


def test_validate_rejects_unknown():
    with pytest.raises(ValueError):
        validate_class("scooter")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/schema/test_classes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.schema'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/schema/__init__.py`:
```python
```

`agastya/schema/classes.py`:
```python
CLASSES: tuple[str, ...] = (
    "motorcycle",
    "rider",
    "helmet",
    "no-helmet",
    "person",
    "car",
    "truck",
    "bus",
    "auto-rickshaw",
    "license-plate",
)

_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def validate_class(name: str) -> str:
    if name not in _NAME_TO_ID:
        raise ValueError(f"unknown class: {name}")
    return name


def name_to_id(name: str) -> int:
    return _NAME_TO_ID[validate_class(name)]


def id_to_name(class_id: int) -> str:
    if class_id < 0 or class_id >= len(CLASSES):
        raise ValueError(f"class id out of range: {class_id}")
    return CLASSES[class_id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python -m pytest tests/schema/test_classes.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add agastya/schema tests/schema
git commit -m "feat: unified ten-class schema"
```

---

### Task 3: Runtime data types

**Files:**
- Create: `agastya/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

`tests/test_types.py`:
```python
import pytest

from agastya.types import BBox, Detection


def test_bbox_area():
    box = BBox(x1=0.0, y1=0.0, x2=2.0, y2=3.0)
    assert box.area() == 6.0


def test_bbox_rejects_inverted():
    with pytest.raises(ValueError):
        BBox(x1=5.0, y1=0.0, x2=1.0, y2=1.0)


def test_bbox_iou_identical_is_one():
    box = BBox(x1=0.0, y1=0.0, x2=2.0, y2=2.0)
    assert box.iou(box) == pytest.approx(1.0)


def test_bbox_iou_disjoint_is_zero():
    a = BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    b = BBox(x1=5.0, y1=5.0, x2=6.0, y2=6.0)
    assert a.iou(b) == 0.0


def test_detection_holds_class_and_score():
    det = Detection(label="helmet", score=0.9, box=BBox(0.0, 0.0, 1.0, 1.0))
    assert det.label == "helmet"
    assert det.score == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.types'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/types.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python -m pytest tests/test_types.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add agastya/types.py tests/test_types.py
git commit -m "feat: runtime data types with bbox geometry"
```

---

### Task 4: Stage protocol

**Files:**
- Create: `agastya/stages/__init__.py`
- Create: `agastya/stages/base.py`
- Test: `tests/stages/test_base.py`

- [ ] **Step 1: Write the failing test**

`tests/stages/__init__.py`:
```python
```

`tests/stages/test_base.py`:
```python
from agastya.stages.base import Stage


class Doubler(Stage[int, int]):
    name = "doubler"

    def process(self, item: int) -> int:
        return item * 2


def test_stage_process_runs():
    assert Doubler().process(3) == 6


def test_stage_exposes_name():
    assert Doubler().name == "doubler"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.stages'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/stages/__init__.py`:
```python
```

`agastya/stages/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")


class Stage(ABC, Generic[I, O]):
    name: str = "stage"

    @abstractmethod
    def process(self, item: I) -> O:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/test_base.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add agastya/stages/__init__.py agastya/stages/base.py tests/stages
git commit -m "feat: generic stage protocol"
```

---

### Task 5: Quality-gate routing logic

**Files:**
- Create: `agastya/stages/gate/__init__.py`
- Create: `agastya/stages/gate/router.py`
- Test: `tests/stages/gate/test_router.py`

- [ ] **Step 1: Write the failing test**

`tests/stages/gate/__init__.py`:
```python
```

`tests/stages/gate/test_router.py`:
```python
import pytest

from agastya.stages.gate.router import score_to_decision
from agastya.types import QualityScore


def test_below_threshold_routes_to_restore():
    decision = score_to_decision(0.3, threshold=0.5)
    assert isinstance(decision, QualityScore)
    assert decision.degraded is True


def test_at_or_above_threshold_bypasses():
    assert score_to_decision(0.5, threshold=0.5).degraded is False
    assert score_to_decision(0.8, threshold=0.5).degraded is False


def test_threshold_must_be_unit_interval():
    with pytest.raises(ValueError):
        score_to_decision(0.4, threshold=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/gate/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.stages.gate'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/stages/gate/__init__.py`:
```python
```

`agastya/stages/gate/router.py`:
```python
from __future__ import annotations

from agastya.types import QualityScore


def score_to_decision(score: float, threshold: float) -> QualityScore:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return QualityScore(value=score, degraded=score < threshold)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/gate/test_router.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add agastya/stages/gate tests/stages/gate
git commit -m "feat: quality-gate routing decision"
```

---

### Task 6: Triple-riding association rule

**Files:**
- Create: `agastya/stages/associate/__init__.py`
- Create: `agastya/stages/associate/rules.py`
- Test: `tests/stages/associate/test_rules.py`

- [ ] **Step 1: Write the failing test**

`tests/stages/associate/__init__.py`:
```python
```

`tests/stages/associate/test_rules.py`:
```python
from agastya.stages.associate.rules import count_riders, is_triple_riding
from agastya.types import BBox, Detection


def _person(x1: float) -> Detection:
    return Detection(label="person", score=0.9, box=BBox(x1, 0.0, x1 + 1.0, 2.0))


def test_counts_persons_overlapping_motorcycle():
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0), _person(2.5), _person(50.0)]
    assert count_riders(moto, persons, min_overlap=0.05) == 3


def test_triple_riding_true_at_three():
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0), _person(2.5)]
    assert is_triple_riding(moto, persons, min_overlap=0.05) is True


def test_triple_riding_false_at_two():
    moto = BBox(0.0, 0.0, 4.0, 2.0)
    persons = [_person(0.0), _person(1.0)]
    assert is_triple_riding(moto, persons, min_overlap=0.05) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/associate/test_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.stages.associate'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/stages/associate/__init__.py`:
```python
```

`agastya/stages/associate/rules.py`:
```python
from __future__ import annotations

from collections.abc import Iterable

from agastya.types import BBox, Detection

TRIPLE_RIDING_MIN = 3


def count_riders(motorcycle: BBox, persons: Iterable[Detection], min_overlap: float) -> int:
    count = 0
    for person in persons:
        inter = motorcycle.intersection(person.box)
        person_area = person.box.area()
        if person_area <= 0.0:
            continue
        if inter / person_area >= min_overlap:
            count += 1
    return count


def is_triple_riding(motorcycle: BBox, persons: Iterable[Detection], min_overlap: float) -> bool:
    return count_riders(motorcycle, persons, min_overlap) >= TRIPLE_RIDING_MIN
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/associate/test_rules.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add agastya/stages/associate tests/stages/associate
git commit -m "feat: triple-riding association rule"
```

---

### Task 7: Merkle audit log

**Files:**
- Create: `agastya/stages/evidence/__init__.py`
- Create: `agastya/stages/evidence/merkle.py`
- Test: `tests/stages/evidence/test_merkle.py`

- [ ] **Step 1: Write the failing test**

`tests/stages/evidence/__init__.py`:
```python
```

`tests/stages/evidence/test_merkle.py`:
```python
import pytest

from agastya.stages.evidence.merkle import leaf_hash, merkle_root, verify_leaf


def test_leaf_hash_is_deterministic():
    assert leaf_hash(b"abc") == leaf_hash(b"abc")


def test_leaf_hash_changes_with_input():
    assert leaf_hash(b"abc") != leaf_hash(b"abd")


def test_root_of_single_leaf_is_leaf():
    h = leaf_hash(b"only")
    assert merkle_root([b"only"]) == h


def test_root_stable_for_two_leaves():
    root = merkle_root([b"a", b"b"])
    assert root == merkle_root([b"a", b"b"])
    assert root != merkle_root([b"b", b"a"])


def test_verify_leaf_membership():
    leaves = [b"a", b"b", b"c"]
    root = merkle_root(leaves)
    assert verify_leaf(b"b", leaves, root) is True
    assert verify_leaf(b"z", leaves, root) is False


def test_empty_leaves_rejected():
    with pytest.raises(ValueError):
        merkle_root([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/evidence/test_merkle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.stages.evidence'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/stages/evidence/__init__.py`:
```python
```

`agastya/stages/evidence/merkle.py`:
```python
from __future__ import annotations

import hashlib
from collections.abc import Sequence


def leaf_hash(data: bytes) -> str:
    return hashlib.sha256(b"\x00" + data).hexdigest()


def _pair_hash(left: str, right: str) -> str:
    return hashlib.sha256(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def merkle_root(leaves: Sequence[bytes]) -> str:
    if not leaves:
        raise ValueError("merkle_root requires at least one leaf")
    level = [leaf_hash(item) for item in leaves]
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(_pair_hash(left, right))
        level = nxt
    return level[0]


def verify_leaf(data: bytes, leaves: Sequence[bytes], root: str) -> bool:
    if leaf_hash(data) not in {leaf_hash(item) for item in leaves}:
        return False
    return merkle_root(leaves) == root
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/evidence/test_merkle.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add agastya/stages/evidence tests/stages/evidence
git commit -m "feat: merkle audit log for evidence chain"
```

---

### Task 8: Evidence metadata builder

**Files:**
- Create: `agastya/stages/evidence/manifest.py`
- Test: `tests/stages/evidence/test_manifest.py`

- [ ] **Step 1: Write the failing test**

`tests/stages/evidence/test_manifest.py`:
```python
from agastya.stages.evidence.manifest import build_manifest
from agastya.types import PlateReading, ViolationRecord


def _record() -> ViolationRecord:
    return ViolationRecord(
        violation_type="no-helmet",
        confidence=0.82,
        plate=PlateReading(text="KA01AB1234", confidence=0.77),
        metadata={"camera_id": "CAM-7", "timestamp": "2026-06-16T10:00:00Z"},
    )


def test_manifest_contains_core_fields():
    manifest = build_manifest(_record(), model_versions={"detector": "yolo26-m@0.1"})
    assert manifest["violation_type"] == "no-helmet"
    assert manifest["confidence"] == 0.82
    assert manifest["plate"] == "KA01AB1234"
    assert manifest["camera_id"] == "CAM-7"
    assert manifest["model_versions"]["detector"] == "yolo26-m@0.1"


def test_manifest_is_json_serializable_and_deterministic():
    import json

    a = build_manifest(_record(), model_versions={"detector": "yolo26-m@0.1"})
    b = build_manifest(_record(), model_versions={"detector": "yolo26-m@0.1"})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_abstained_plate_is_null():
    record = ViolationRecord(
        violation_type="no-helmet",
        confidence=0.6,
        plate=PlateReading(text="", confidence=0.1, abstained=True),
    )
    manifest = build_manifest(record, model_versions={})
    assert manifest["plate"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/evidence/test_manifest.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_manifest'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/stages/evidence/manifest.py`:
```python
from __future__ import annotations

from agastya.types import ViolationRecord


def build_manifest(record: ViolationRecord, model_versions: dict[str, str]) -> dict:
    plate_text: str | None = None
    if record.plate is not None and not record.plate.abstained:
        plate_text = record.plate.text
    manifest: dict = {
        "violation_type": record.violation_type,
        "confidence": record.confidence,
        "plate": plate_text,
        "model_versions": dict(model_versions),
    }
    manifest.update(record.metadata)
    return manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python -m pytest tests/stages/evidence/test_manifest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add agastya/stages/evidence/manifest.py tests/stages/evidence/test_manifest.py
git commit -m "feat: evidence metadata manifest builder"
```

---

### Task 9: Config models + VRAM profile selection

**Files:**
- Create: `agastya/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import pytest

from agastya.config import PipelineConfig, VRAMProfile, select_profile


def test_high_vram_selects_full_profile():
    assert select_profile(16.0) == VRAMProfile.FULL


def test_low_vram_selects_fallback_profile():
    assert select_profile(8.0) == VRAMProfile.LOW


def test_full_profile_uses_yolo26_m():
    cfg = PipelineConfig(profile=VRAMProfile.FULL)
    assert cfg.detector == "yolo26-m"
    assert cfg.use_diffusion_sr is False


def test_low_profile_swaps_to_light_models():
    cfg = PipelineConfig(profile=VRAMProfile.LOW)
    assert cfg.detector == "yolo26-s"
    assert cfg.segmenter == "mobilesam"


def test_gate_threshold_must_be_unit_interval():
    with pytest.raises(ValueError):
        PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.config'`.

- [ ] **Step 3: Write minimal implementation**

`agastya/config.py`:
```python
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

LOW_VRAM_GB = 10.0


class VRAMProfile(str, Enum):
    FULL = "full"
    LOW = "low"


def select_profile(available_gb: float) -> VRAMProfile:
    return VRAMProfile.LOW if available_gb < LOW_VRAM_GB else VRAMProfile.FULL


class PipelineConfig(BaseModel):
    profile: VRAMProfile = VRAMProfile.FULL
    gate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    triple_riding_overlap: float = Field(default=0.1, ge=0.0, le=1.0)
    detector: str = ""
    segmenter: str = ""
    use_diffusion_sr: bool = False

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> "PipelineConfig":
        if not self.detector:
            self.detector = "yolo26-s" if self.profile is VRAMProfile.LOW else "yolo26-m"
        if not self.segmenter:
            self.segmenter = "mobilesam" if self.profile is VRAMProfile.LOW else "sam2"
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/student/Flipkart && python -m pytest tests/test_config.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add agastya/config.py tests/test_config.py
git commit -m "feat: pipeline config with vram profile selection"
```

---

### Task 10: Pipeline orchestrator with stub stages

**Files:**
- Create: `agastya/pipeline.py`
- Create: `agastya/stages/stubs.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from agastya.config import PipelineConfig, VRAMProfile
from agastya.pipeline import Pipeline, PipelineInput
from agastya.stages.stubs import (
    StubDetector,
    StubGate,
    StubOCR,
    StubRestorer,
)


def test_clean_image_bypasses_restore():
    restorer = StubRestorer()
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.9),
        restorer=restorer,
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-1", pixels=b"raw"))
    assert restorer.calls == 0
    assert result.records


def test_degraded_image_invokes_restore():
    restorer = StubRestorer()
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=0.5),
        gate=StubGate(score=0.2),
        restorer=restorer,
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    pipeline.run(PipelineInput(image_id="img-2", pixels=b"raw"))
    assert restorer.calls == 1


def test_result_records_carry_merkle_root():
    pipeline = Pipeline(
        config=PipelineConfig(profile=VRAMProfile.FULL),
        gate=StubGate(score=0.9),
        restorer=StubRestorer(),
        detector=StubDetector(),
        ocr=StubOCR(text="KA01AB1234", confidence=0.8),
    )
    result = pipeline.run(PipelineInput(image_id="img-3", pixels=b"raw"))
    assert isinstance(result.merkle_root, str)
    assert len(result.merkle_root) == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/student/Flipkart && python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agastya.pipeline'`.

- [ ] **Step 3: Write stub stages**

`agastya/stages/stubs.py`:
```python
from __future__ import annotations

from agastya.types import BBox, Detection, PlateReading


class StubGate:
    def __init__(self, score: float) -> None:
        self.score = score

    def score_image(self, pixels: bytes) -> float:
        return self.score


class StubRestorer:
    def __init__(self) -> None:
        self.calls = 0

    def restore(self, pixels: bytes) -> bytes:
        self.calls += 1
        return pixels


class StubDetector:
    def detect(self, pixels: bytes) -> list[Detection]:
        return [
            Detection(label="motorcycle", score=0.95, box=BBox(0.0, 0.0, 4.0, 2.0)),
            Detection(label="no-helmet", score=0.88, box=BBox(0.5, 0.0, 1.5, 1.0)),
            Detection(label="license-plate", score=0.91, box=BBox(1.0, 1.5, 2.0, 2.0)),
        ]


class StubOCR:
    def __init__(self, text: str, confidence: float) -> None:
        self.text = text
        self.confidence = confidence

    def read(self, pixels: bytes) -> PlateReading:
        return PlateReading(text=self.text, confidence=self.confidence)
```

- [ ] **Step 4: Write the orchestrator**

`agastya/pipeline.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from agastya.config import PipelineConfig
from agastya.stages.evidence.manifest import build_manifest
from agastya.stages.evidence.merkle import merkle_root
from agastya.stages.gate.router import score_to_decision
from agastya.types import Detection, PlateReading, ViolationRecord


class GateStage(Protocol):
    def score_image(self, pixels: bytes) -> float:
        ...


class RestoreStage(Protocol):
    def restore(self, pixels: bytes) -> bytes:
        ...


class DetectStage(Protocol):
    def detect(self, pixels: bytes) -> list[Detection]:
        ...


class OCRStage(Protocol):
    def read(self, pixels: bytes) -> PlateReading:
        ...


@dataclass(frozen=True)
class PipelineInput:
    image_id: str
    pixels: bytes


@dataclass(frozen=True)
class PipelineResult:
    image_id: str
    records: tuple[ViolationRecord, ...]
    merkle_root: str
    manifests: tuple[dict, ...] = field(default_factory=tuple)


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        gate: GateStage,
        restorer: RestoreStage,
        detector: DetectStage,
        ocr: OCRStage,
    ) -> None:
        self.config = config
        self.gate = gate
        self.restorer = restorer
        self.detector = detector
        self.ocr = ocr

    def run(self, item: PipelineInput) -> PipelineResult:
        pixels = item.pixels
        decision = score_to_decision(self.gate.score_image(pixels), self.config.gate_threshold)
        if decision.degraded:
            pixels = self.restorer.restore(pixels)
        detections = self.detector.detect(pixels)
        plate = self.ocr.read(pixels)
        records = self._build_records(detections, plate)
        manifests = tuple(
            build_manifest(record, model_versions={"detector": self.config.detector})
            for record in records
        )
        leaves = [json.dumps(m, sort_keys=True).encode() for m in manifests] or [item.image_id.encode()]
        return PipelineResult(
            image_id=item.image_id,
            records=records,
            merkle_root=merkle_root(leaves),
            manifests=manifests,
        )

    def _build_records(
        self, detections: list[Detection], plate: PlateReading
    ) -> tuple[ViolationRecord, ...]:
        records: list[ViolationRecord] = []
        for det in detections:
            if det.label == "no-helmet":
                records.append(
                    ViolationRecord(
                        violation_type="no-helmet",
                        confidence=det.score,
                        plate=plate,
                        detections=(det,),
                    )
                )
        return tuple(records)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/student/Flipkart && python -m pytest tests/test_pipeline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add agastya/pipeline.py agastya/stages/stubs.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator with stub stages"
```

---

### Task 11: Full suite green + lint

**Files:**
- None (verification task)

- [ ] **Step 1: Run the whole suite**

Run: `cd /home/student/Flipkart && python -m pytest -v`
Expected: PASS — all tests across schema, types, stages, config, pipeline.

- [ ] **Step 2: Lint**

Run: `cd /home/student/Flipkart && ruff check agastya tests`
Expected: `All checks passed!`

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint clean" || true
```

---

### Task 12: Rename DRISHTI → AGASTYA in source docs

**Files:**
- Modify: `DRISHTI_master_build_prompt_v2.md` → rename to `AGASTYA_master_build_prompt_v2.md`
- Modify: `researchdone`

- [ ] **Step 1: Rename the master prompt file**

Run: `cd /home/student/Flipkart && git mv DRISHTI_master_build_prompt_v2.md AGASTYA_master_build_prompt_v2.md 2>/dev/null || mv DRISHTI_master_build_prompt_v2.md AGASTYA_master_build_prompt_v2.md`
Expected: file renamed.

- [ ] **Step 2: Replace name tokens in both docs**

Run: `cd /home/student/Flipkart && sed -i 's/DRISHTI/AGASTYA/g' AGASTYA_master_build_prompt_v2.md researchdone`
Expected: no output (in-place edit).

- [ ] **Step 3: Verify no stale tokens remain**

Run: `cd /home/student/Flipkart && grep -ric 'drishti' AGASTYA_master_build_prompt_v2.md researchdone || echo "clean"`
Expected: `0` for each file (or `clean`). Note: the backronym expansion line in the master prompt becomes a plain description — manually confirm it still reads sensibly after the swap, since AGASTYA is a pure brand with no acronym.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename DRISHTI to AGASTYA across source docs"
```

---

## Self-Review

**Spec coverage (foundation scope only):**
- Stage 0 gate routing → Task 5. Stage 2 association rule (triple-riding ≥3) → Task 6. Stage 4 Merkle + manifest → Tasks 7, 8. Class schema → Task 2. VRAM profile/fallback → Task 9. Cascade orchestration with gate-bypass → Task 10. Rename → Task 12.
- Deferred to later plans (correctly out of scope here): ARNIQA model, NAFNet/Real-ESRGAN/LCDNet, YOLO26 training, SAHI, SAM2, PARSeq, conformal prediction, C2PA signing, Grad-CAM, analytics dashboard, datasets/eval harness. Stage interfaces (`GateStage`, `RestoreStage`, `DetectStage`, `OCRStage` in Task 10) are the seams these plug into.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows expected output.

**Type consistency:** `BBox`, `Detection`, `QualityScore`, `PlateReading`, `ViolationRecord` (Task 3) reused unchanged in Tasks 5, 6, 8, 10. `score_to_decision` (Task 5), `merkle_root`/`leaf_hash` (Task 7), `build_manifest` (Task 8) called with matching signatures in Task 10. `VRAMProfile`/`PipelineConfig` (Task 9) consumed in Task 10. `Stage` protocol (Task 4) is the abstract template; runtime stages use structural `Protocol` seams in Task 10 — intentional, documented.

**Gap note:** Task 4 `Stage` ABC and Task 10 `Protocol` seams coexist by design — ABC for future concrete typed stages, Protocols for duck-typed injection/testing. No conflict.
