# Phase 4 Ablation Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an add-one-in fractional ablation that measures the marginal value of each Phase 4 novel block (restore, gate-routing, SAM2 association, calibration) against the locked control, on clean and motion-blur-degraded inputs.

**Architecture:** Pure, unit-tested logic (arm presets, delta ranking, rate calc, matrix rendering, degrade-bytes helper) lives in `agastya/eval/ablation.py`. A thin driver `scripts/eval_ablation.py` loops arms × conditions over a fixed 500-image subset, runs the same per-image stage sequence as `eval_e2e.py`, and emits a markdown matrix. No changes to pipeline or stages.

**Tech Stack:** Python 3, pytest, ruff, numpy, opencv (cv2), pydantic `PipelineConfig`, existing `agastya.eval` + `agastya.stages` modules.

## Global Constraints

- No code comments anywhere (project rule).
- No git commands — do NOT run `git add`/`git commit`. Each task's final step is a verification checkpoint (`ruff check` + `pytest`), not a commit.
- Locked control: `no_helmet_min_conf` 0.25, `triple_riding_overlap` 0.10, detector `runs/detect/train-3/weights/best.pt`, conf 0.25, device cuda.
- 5 arms (Control, +Restore blanket, +Gate routed, +SAM2, +Calibration) × {clean, degraded} = 10 cells.
- Fixed first-500 val subset (`sorted(...)[:500]`), identical every cell.
- Degraded = `motion_blur` kernel_size 15, angle_deg 30.0.
- Every cell reports absolute + Δ-vs-control for triple-riding F1, no-helmet F1, mean latency; plus `restore_invoked_count`, `restore_invoked_rate`.
- Ranking priority: (1) triple-riding ΔF1, (2) no-helmet ΔF1, (3) Δ latency.
- Run all tests with `PYTHONPATH=. pytest -q`. Lint with `ruff check agastya tests scripts`.

---

### Task 1: Arm presets and per-arm config

**Files:**
- Create: `agastya/eval/ablation.py`
- Test: `tests/eval/test_ablation.py`

**Interfaces:**
- Consumes: `agastya.config.PipelineConfig`.
- Produces:
  - `@dataclass(frozen=True) class ArmSpec: name: str; gate_backend: str; gate_threshold: float; restore_backend: str; associate_backend: str; use_calibration: bool`
  - `ARMS: tuple[ArmSpec, ...]` (5 arms; first is `name="control"`).
  - `build_arm_config(arm: ArmSpec, weights: str, nafnet_weights: str | None, device: str, conf: float = 0.25) -> PipelineConfig`

- [ ] **Step 1: Write the failing test**

```python
from agastya.eval.ablation import ARMS, ArmSpec, build_arm_config


def test_arms_has_five_with_control_first():
    assert len(ARMS) == 5
    assert ARMS[0].name == "control"


def test_control_arm_config_matches_locked_defaults():
    arm = next(a for a in ARMS if a.name == "control")
    config = build_arm_config(arm, "best.pt", None, "cuda")
    assert config.gate_backend == "never"
    assert config.restore_backend == "passthrough"
    assert config.associate_backend == "box"
    assert config.detect_conf == 0.25
    assert config.no_helmet_min_conf == 0.25
    assert config.triple_riding_overlap == 0.10


def test_gate_routed_arm_enables_arniqa_and_nafnet():
    arm = next(a for a in ARMS if a.name == "gate_routed")
    config = build_arm_config(arm, "best.pt", "models/NAFNet-GoPro-width32.pth", "cuda")
    assert config.gate_backend == "arniqa"
    assert config.gate_threshold == 0.36
    assert config.restore_backend == "nafnet"
    assert config.nafnet_weights == "models/NAFNet-GoPro-width32.pth"


def test_restore_blanket_arm_uses_always_gate():
    arm = next(a for a in ARMS if a.name == "restore_blanket")
    assert arm.gate_backend == "always"
    assert arm.restore_backend == "nafnet"


def test_sam2_arm_selects_sam2_associator():
    arm = next(a for a in ARMS if a.name == "sam2")
    config = build_arm_config(arm, "best.pt", None, "cuda")
    assert config.associate_backend == "sam2"


def test_calibration_arm_flag_set():
    arm = next(a for a in ARMS if a.name == "calibration")
    assert arm.use_calibration is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/eval/test_ablation.py -q`
Expected: FAIL with `ModuleNotFoundError: agastya.eval.ablation`.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass

from agastya.config import PipelineConfig


@dataclass(frozen=True)
class ArmSpec:
    name: str
    gate_backend: str
    gate_threshold: float
    restore_backend: str
    associate_backend: str
    use_calibration: bool


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("control", "never", 0.5, "passthrough", "box", False),
    ArmSpec("restore_blanket", "always", 0.5, "nafnet", "box", False),
    ArmSpec("gate_routed", "arniqa", 0.36, "nafnet", "box", False),
    ArmSpec("sam2", "never", 0.5, "passthrough", "sam2", False),
    ArmSpec("calibration", "never", 0.5, "passthrough", "box", True),
)


def build_arm_config(
    arm: ArmSpec,
    weights: str,
    nafnet_weights: str | None,
    device: str,
    conf: float = 0.25,
) -> PipelineConfig:
    return PipelineConfig(
        gate_backend=arm.gate_backend,
        gate_threshold=arm.gate_threshold,
        restore_backend=arm.restore_backend,
        nafnet_weights=nafnet_weights,
        restore_device=device,
        detect_backend="yolo",
        detector_weights=weights,
        detect_conf=conf,
        associate_backend=arm.associate_backend,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/eval/test_ablation.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Checkpoint**

Run: `ruff check agastya tests && PYTHONPATH=. pytest -q`
Expected: ruff clean; full suite green.

---

### Task 2: Cell result, restore rate, and delta ranking

**Files:**
- Modify: `agastya/eval/ablation.py`
- Test: `tests/eval/test_ablation.py`

**Interfaces:**
- Consumes: `ArmSpec` from Task 1.
- Produces:
  - `@dataclass(frozen=True) class CellResult: arm_name: str; condition: str; nohelmet_f1: float; triple_f1: float; mean_latency_ms: float; restore_invoked_count: int; restore_invoked_rate: float`
  - `restore_rate(count: int, n_images: int) -> float` (0.0 when `n_images == 0`)
  - `@dataclass(frozen=True) class RankedArm: arm_name: str; condition: str; d_triple_f1: float; d_nohelmet_f1: float; d_latency_ms: float`
  - `rank_arms(results: list[CellResult], condition: str) -> list[RankedArm]` — deltas vs the control cell of the same condition, sorted by (`d_triple_f1` desc, `d_nohelmet_f1` desc, `d_latency_ms` asc); control excluded from the ranked output.

- [ ] **Step 1: Write the failing test**

```python
from agastya.eval.ablation import (
    CellResult,
    RankedArm,
    rank_arms,
    restore_rate,
)


def test_restore_rate_basic():
    assert restore_rate(125, 500) == 0.25


def test_restore_rate_zero_images_guard():
    assert restore_rate(0, 0) == 0.0


def _cell(name, triple, nohelmet, latency):
    return CellResult(name, "clean", nohelmet, triple, latency, 0, 0.0)


def test_rank_excludes_control_and_orders_by_triple_then_nohelmet_then_latency():
    results = [
        _cell("control", 0.893, 0.967, 12.0),
        _cell("sam2", 0.910, 0.967, 30.0),
        _cell("calibration", 0.893, 0.967, 12.5),
        _cell("gate_routed", 0.893, 0.975, 40.0),
    ]
    ranked = rank_arms(results, "clean")
    assert [r.arm_name for r in ranked] == ["sam2", "gate_routed", "calibration"]
    assert ranked[0].d_triple_f1 == 0.910 - 0.893
    assert ranked[1].d_nohelmet_f1 == 0.975 - 0.967
    assert ranked[2].d_latency_ms == 12.5 - 12.0


def test_rank_only_uses_cells_of_requested_condition():
    results = [
        CellResult("control", "clean", 0.967, 0.893, 12.0, 0, 0.0),
        CellResult("control", "degraded", 0.800, 0.700, 13.0, 0, 0.0),
        CellResult("sam2", "degraded", 0.700, 0.750, 31.0, 0, 0.0),
    ]
    ranked = rank_arms(results, "degraded")
    assert [r.arm_name for r in ranked] == ["sam2"]
    assert ranked[0].d_triple_f1 == 0.750 - 0.700
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/eval/test_ablation.py -k "rate or rank" -q`
Expected: FAIL with `ImportError: cannot import name 'CellResult'`.

- [ ] **Step 3: Write minimal implementation**

Append to `agastya/eval/ablation.py`:

```python
@dataclass(frozen=True)
class CellResult:
    arm_name: str
    condition: str
    nohelmet_f1: float
    triple_f1: float
    mean_latency_ms: float
    restore_invoked_count: int
    restore_invoked_rate: float


@dataclass(frozen=True)
class RankedArm:
    arm_name: str
    condition: str
    d_triple_f1: float
    d_nohelmet_f1: float
    d_latency_ms: float


def restore_rate(count: int, n_images: int) -> float:
    if n_images <= 0:
        return 0.0
    return count / n_images


def rank_arms(results: list[CellResult], condition: str) -> list[RankedArm]:
    cells = [r for r in results if r.condition == condition]
    control = next(r for r in cells if r.arm_name == "control")
    ranked = [
        RankedArm(
            r.arm_name,
            condition,
            r.triple_f1 - control.triple_f1,
            r.nohelmet_f1 - control.nohelmet_f1,
            r.mean_latency_ms - control.mean_latency_ms,
        )
        for r in cells
        if r.arm_name != "control"
    ]
    ranked.sort(key=lambda r: (-r.d_triple_f1, -r.d_nohelmet_f1, r.d_latency_ms))
    return ranked
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/eval/test_ablation.py -q`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Checkpoint**

Run: `ruff check agastya tests && PYTHONPATH=. pytest -q`
Expected: ruff clean; full suite green.

---

### Task 3: Degrade-bytes helper

**Files:**
- Modify: `agastya/eval/ablation.py`
- Test: `tests/eval/test_ablation.py`

**Interfaces:**
- Consumes: `agastya.eval.degrade.motion_blur`.
- Produces: `degrade_bytes(pixels: bytes, kernel_size: int = 15, angle_deg: float = 30.0) -> bytes` — decode JPEG/PNG bytes, apply `motion_blur`, re-encode as JPEG bytes.

- [ ] **Step 1: Write the failing test**

```python
import cv2
import numpy as np

from agastya.eval.ablation import degrade_bytes


def _encode(arr):
    return cv2.imencode(".jpg", arr)[1].tobytes()


def test_degrade_bytes_returns_decodable_same_shape():
    arr = (np.random.rand(64, 96, 3) * 255).astype("uint8")
    out = degrade_bytes(_encode(arr))
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (64, 96, 3)


def test_degrade_bytes_changes_pixels():
    arr = (np.random.rand(64, 96, 3) * 255).astype("uint8")
    clean = cv2.imdecode(np.frombuffer(_encode(arr), np.uint8), cv2.IMREAD_COLOR)
    blurred = cv2.imdecode(
        np.frombuffer(degrade_bytes(_encode(arr)), np.uint8), cv2.IMREAD_COLOR
    )
    assert not np.array_equal(clean, blurred)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/eval/test_ablation.py -k degrade -q`
Expected: FAIL with `ImportError: cannot import name 'degrade_bytes'`.

- [ ] **Step 3: Write minimal implementation**

Add imports at top of `agastya/eval/ablation.py`:

```python
import cv2
import numpy as np

from agastya.eval.degrade import motion_blur
```

Append:

```python
def degrade_bytes(pixels: bytes, kernel_size: int = 15, angle_deg: float = 30.0) -> bytes:
    arr = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode image bytes")
    blurred = motion_blur(arr, kernel_size, angle_deg)
    ok, buffer = cv2.imencode(".jpg", blurred)
    if not ok:
        raise ValueError("could not encode degraded image")
    return buffer.tobytes()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/eval/test_ablation.py -k degrade -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Checkpoint**

Run: `ruff check agastya tests && PYTHONPATH=. pytest -q`
Expected: ruff clean; full suite green.

---

### Task 4: Matrix rendering (absolute + deltas)

**Files:**
- Modify: `agastya/eval/ablation.py`
- Test: `tests/eval/test_ablation.py`

**Interfaces:**
- Consumes: `CellResult` from Task 2.
- Produces: `render_matrix(results: list[CellResult]) -> str` — markdown table, one row per cell, columns: arm, condition, no-helmet F1, ΔF1, triple F1, ΔF1, latency ms, Δlatency, restore count, restore rate. Deltas are vs the control cell of the same condition (control's own deltas are 0).

- [ ] **Step 1: Write the failing test**

```python
from agastya.eval.ablation import CellResult, render_matrix


def test_render_matrix_has_header_and_delta_columns():
    results = [
        CellResult("control", "clean", 0.967, 0.893, 12.0, 0, 0.0),
        CellResult("sam2", "clean", 0.967, 0.910, 30.0, 0, 0.0),
    ]
    table = render_matrix(results)
    assert "arm" in table and "condition" in table
    assert "no-helmet F1" in table
    assert "triple F1" in table
    assert "ΔF1" in table
    assert "Δlat" in table
    assert "restore rate" in table


def test_render_matrix_control_deltas_zero_and_sam2_triple_delta():
    results = [
        CellResult("control", "clean", 0.967, 0.893, 12.0, 0, 0.0),
        CellResult("sam2", "clean", 0.967, 0.910, 30.0, 0, 0.0),
    ]
    table = render_matrix(results)
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    control_row = next(ln for ln in lines if "control" in ln)
    sam2_row = next(ln for ln in lines if "sam2" in ln)
    assert "+0.000" in control_row
    assert "+0.017" in sam2_row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/eval/test_ablation.py -k render -q`
Expected: FAIL with `ImportError: cannot import name 'render_matrix'`.

- [ ] **Step 3: Write minimal implementation**

Append to `agastya/eval/ablation.py`:

```python
def _control_for(results: list[CellResult], condition: str) -> CellResult:
    return next(r for r in results if r.condition == condition and r.arm_name == "control")


def render_matrix(results: list[CellResult]) -> str:
    header = (
        "| arm | condition | no-helmet F1 | ΔF1 | triple F1 | ΔF1 | "
        "lat ms | Δlat | restore n | restore rate |"
    )
    sep = "|" + "---|" * 10
    rows = [header, sep]
    for r in results:
        control = _control_for(results, r.condition)
        d_nohelmet = r.nohelmet_f1 - control.nohelmet_f1
        d_triple = r.triple_f1 - control.triple_f1
        d_lat = r.mean_latency_ms - control.mean_latency_ms
        rows.append(
            f"| {r.arm_name} | {r.condition} | {r.nohelmet_f1:.3f} | {d_nohelmet:+.3f} | "
            f"{r.triple_f1:.3f} | {d_triple:+.3f} | {r.mean_latency_ms:.2f} | {d_lat:+.2f} | "
            f"{r.restore_invoked_count} | {r.restore_invoked_rate:.3f} |"
        )
    return "\n".join(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/eval/test_ablation.py -k render -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Checkpoint**

Run: `ruff check agastya tests && PYTHONPATH=. pytest -q`
Expected: ruff clean; full suite green.

---

### Task 5: Driver script (stage loop + orchestration)

**Files:**
- Create: `scripts/eval_ablation.py`

**Interfaces:**
- Consumes: `ARMS`, `build_arm_config`, `CellResult`, `restore_rate`, `degrade_bytes`, `render_matrix`, `rank_arms` from `agastya.eval.ablation`; stage factories; `Calibrator.from_json`; `eval/e2e.py` helpers; `eval/yolo_data.py` loaders; `gate.router.score_to_decision`.
- Produces: an executable that prints the matrix and writes `docs/runbooks/phase4-ablation.md`. No unit test (integration glue; all logic it calls is already tested).

- [ ] **Step 1: Write the driver**

```python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from agastya.eval.ablation import (
    ARMS,
    CellResult,
    build_arm_config,
    degrade_bytes,
    rank_arms,
    render_matrix,
    restore_rate,
)
from agastya.eval.e2e import ViolationCounts, accumulate, gt_violations
from agastya.eval.yolo_data import label_path_for, load_data_yaml, load_truths
from agastya.stages.associate.factory import build_associator
from agastya.stages.detect.factory import build_detector
from agastya.stages.gate.factory import build_gate
from agastya.stages.gate.router import score_to_decision
from agastya.stages.ocr.factory import build_ocr
from agastya.stages.restore.factory import build_restorer
from agastya.verify.calibration import Calibrator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CONDITIONS = ("clean", "degraded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGASTYA Phase 4 ablation matrix")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nafnet-weights", type=Path, default=Path("models/NAFNet-GoPro-width32.pth"))
    parser.add_argument("--calibration", type=Path, default=Path("models/calibration.json"))
    parser.add_argument("--runbook", type=Path, default=Path("docs/runbooks/phase4-ablation.md"))
    return parser.parse_args()


def run_cell(arm, condition, images, args) -> CellResult:
    config = build_arm_config(
        arm, str(args.weights), str(args.nafnet_weights), args.device, args.conf
    )
    gate = build_gate(config)
    restorer = build_restorer(config)
    detector = build_detector(config)
    associator = build_associator(config)
    ocr = build_ocr(config)
    calibrator = (
        Calibrator.from_json(str(args.calibration)) if arm.use_calibration else None
    )

    per_type = {name: ViolationCounts() for name in ("no-helmet", "triple-riding")}
    restore_invoked = 0
    detect_seconds = 0.0
    val_dir, names = images

    for image_path in val_dir:
        pixels = image_path.read_bytes()
        if condition == "degraded":
            pixels = degrade_bytes(pixels)

        score = gate.score_image(pixels)
        if score_to_decision(score, config.gate_threshold).degraded:
            pixels = restorer.restore(pixels)
            restore_invoked += 1

        start = time.perf_counter()
        detections = detector.detect(pixels)
        detect_seconds += time.perf_counter() - start

        pred_set: set[str] = set()
        if any(
            d.label == "no-helmet" and d.score >= config.no_helmet_min_conf
            for d in detections
        ):
            pred_set.add("no-helmet")
        persons = [d for d in detections if d.label == "person"]
        for moto in detections:
            if moto.label == "motorcycle" and associator.is_triple_riding(
                moto.box, persons, pixels
            ):
                pred_set.add("triple-riding")
                break

        ocr.read(pixels)
        if calibrator is not None:
            for violation_type in pred_set:
                calibrator.evaluate(0.9, violation_type)

        truths = load_truths(label_path_for(image_path), names)
        accumulate(per_type, pred_set, gt_violations(truths, config.triple_riding_overlap))

    n = len(val_dir)
    _, _, nohelmet_f1 = per_type["no-helmet"].prf()
    _, _, triple_f1 = per_type["triple-riding"].prf()
    return CellResult(
        arm_name=arm.name,
        condition=condition,
        nohelmet_f1=nohelmet_f1,
        triple_f1=triple_f1,
        mean_latency_ms=1000.0 * detect_seconds / n if n else 0.0,
        restore_invoked_count=restore_invoked,
        restore_invoked_rate=restore_rate(restore_invoked, n),
    )


def main() -> int:
    args = parse_args()
    val_dir, names = load_data_yaml(args.data)
    all_images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    subset = all_images[: args.samples]
    if not subset:
        raise SystemExit(f"no val images under {val_dir}")

    results: list[CellResult] = []
    for arm in ARMS:
        for condition in CONDITIONS:
            results.append(run_cell(arm, condition, (subset, names), args))

    matrix = render_matrix(results)
    print(f"images per cell: {len(subset)}\n")
    print(matrix)
    print("\nranking (clean):")
    for r in rank_arms(results, "clean"):
        print(f"  {r.arm_name:<16} dTripleF1 {r.d_triple_f1:+.3f}  "
              f"dNoHelmetF1 {r.d_nohelmet_f1:+.3f}  dLat {r.d_latency_ms:+.2f}")
    print("\nranking (degraded):")
    for r in rank_arms(results, "degraded"):
        print(f"  {r.arm_name:<16} dTripleF1 {r.d_triple_f1:+.3f}  "
              f"dNoHelmetF1 {r.d_nohelmet_f1:+.3f}  dLat {r.d_latency_ms:+.2f}")

    args.runbook.write_text(
        f"# AGASTYA Phase 4 — Ablation Matrix\n\n"
        f"Add-one-in fractional ablation, {len(subset)} val images per cell.\n\n"
        f"{matrix}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lint and import-check the driver**

Run: `ruff check scripts/eval_ablation.py && PYTHONPATH=. python3 -c "import ast; ast.parse(open('scripts/eval_ablation.py').read())"`
Expected: ruff clean; no parse error.

- [ ] **Step 3: Smoke the driver on a tiny sample**

Run:
```bash
PYTHONPATH=. python3 scripts/eval_ablation.py \
  --weights runs/detect/train-3/weights/best.pt \
  --data data/processed/data.yaml \
  --samples 8 --device cuda
```
Expected: prints a 10-row matrix (5 arms × 2 conditions) plus clean/degraded rankings; writes `docs/runbooks/phase4-ablation.md`. SAM2 arm downloads `sam2_b.pt` on first use; NAFNet arm loads `models/NAFNet-GoPro-width32.pth`.

- [ ] **Step 4: Checkpoint**

Run: `ruff check agastya tests scripts && PYTHONPATH=. pytest -q`
Expected: ruff clean; full suite green.

---

### Task 6: 500-image run, promotion, full-val confirmation, runbook

**Files:**
- Modify: `docs/runbooks/phase4-ablation.md` (regenerated + annotated)

This task runs the experiment; no new code.

- [ ] **Step 1: Run the full 500-image matrix**

Run:
```bash
PYTHONPATH=. python3 scripts/eval_ablation.py \
  --weights runs/detect/train-3/weights/best.pt \
  --data data/processed/data.yaml \
  --samples 500 --device cuda \
  > /tmp/agastya_ablation_500.log 2>&1
```
Expected: exit 0; matrix + rankings in the log; runbook written.

- [ ] **Step 2: Read the matrix and decide promotions**

Inspect `/tmp/agastya_ablation_500.log`. Promote to full-val confirmation only arms with a meaningful F1 gain on either violation (clean or degraded), or a meaningful latency tradeoff worth recording. Record the per-arm promote/stop decision with its delta justification.

- [ ] **Step 3: Confirm promoted arms on full val**

For each promoted arm, re-run its cell(s) at `--samples 5000` (full 4566) and capture the confirmed deltas. (Control is already confirmed in `docs/runbooks/e2e-baseline.md`.)

- [ ] **Step 4: Finalize the runbook**

Edit `docs/runbooks/phase4-ablation.md` to include: the locked config, the 500-image matrix, the promotion decisions with justifications, the full-val confirmation numbers for promoted arms, and a "winners" summary naming the components that provide measurable value. Match the style of `docs/runbooks/e2e-baseline.md`.

- [ ] **Step 5: Checkpoint**

Run: `ruff check agastya tests scripts && PYTHONPATH=. pytest -q`
Expected: ruff clean; full suite green.

---

## Self-Review

**Spec coverage:**
- 5 arms / add-one-in → Task 1 (`ARMS`). ✓
- Restore blanket-vs-routed coupling → Task 1 arm definitions (`always` vs `arniqa` gate). ✓
- Clean + degraded conditions → Task 3 (`degrade_bytes`) + Task 5 (`CONDITIONS` loop). ✓
- Fixed 500 subset → Task 5 (`sorted(...)[:samples]`, default 500). ✓
- restore_invoked_count + rate → Task 2 (`restore_rate`, `CellResult`) + Task 5 (accumulation). ✓
- Absolute + delta metrics per cell → Task 4 (`render_matrix`). ✓
- Ranking priority triple→nohelmet→latency → Task 2 (`rank_arms`). ✓
- Promotion gate + full-val confirm → Task 6. ✓
- Runbook output → Task 5 (initial write) + Task 6 (finalize). ✓
- No new pipeline/stage changes → only `agastya/eval/ablation.py` + `scripts/eval_ablation.py`. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; run commands have expected output. ✓

**Type consistency:** `ArmSpec`, `CellResult`, `RankedArm` field names consistent across Tasks 1/2/4/5; `build_arm_config` signature matches its Task-5 call; `restore_rate`/`degrade_bytes`/`render_matrix`/`rank_arms` signatures match their call sites. ✓

**No-git adaptation:** Commit steps replaced with ruff+pytest checkpoints per project rule. ✓
