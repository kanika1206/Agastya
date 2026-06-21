# NAFNet Deblur Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a structured image-restoration stage with a backend toggle and a NAFNet adapter shell (no real weights yet) behind the pipeline's existing `RestoreStage` seam.

**Architecture:** New `agastya/stages/restore/` package holds a passthrough restorer, a NAFNet adapter that lazily loads torch weights, a factory selecting the backend from config, and a dedicated error type. The pipeline already calls `restorer.restore(bytes) -> bytes` on degraded frames; only the constructed backend changes.

**Tech Stack:** Python 3.10, pydantic (frozen `PipelineConfig`), pytest, ruff. torch + cv2 imported lazily inside the NAFNet backend only.

## Global Constraints

- No code comments anywhere (project rule).
- No git operations — no `git add`/`commit`. Each task ends with a verify checkpoint instead.
- `PipelineConfig` is a frozen pydantic `BaseModel`; add fields as pydantic fields, never mutate instances.
- Restorers must satisfy the existing protocol `restore(pixels: bytes) -> bytes` (`agastya/pipeline.py:20-22`).
- Missing NAFNet weights must fail loud with `RestorerUnavailable` — never silent passthrough.
- Backend default is `"passthrough"`; default pipeline behavior must stay byte-identical to today.
- torch and cv2 imported lazily inside `NafnetRestorer` methods, not at module top, so the passthrough path needs neither.

---

### Task 1: Config fields for restore backend

**Files:**
- Modify: `agastya/config.py:19-27`
- Test: `tests/test_config.py` (add cases; create if absent)

**Interfaces:**
- Produces: `PipelineConfig.restore_backend: str = "passthrough"`, `PipelineConfig.nafnet_weights: str | None = None`, `PipelineConfig.restore_device: str = "cpu"`

- [ ] **Step 1: Write the failing test**

```python
from agastya.config import PipelineConfig


def test_restore_defaults_are_passthrough_cpu():
    cfg = PipelineConfig()
    assert cfg.restore_backend == "passthrough"
    assert cfg.nafnet_weights is None
    assert cfg.restore_device == "cpu"


def test_restore_fields_are_settable():
    cfg = PipelineConfig(restore_backend="nafnet", nafnet_weights="/w/best.pt", restore_device="cuda")
    assert cfg.restore_backend == "nafnet"
    assert cfg.nafnet_weights == "/w/best.pt"
    assert cfg.restore_device == "cuda"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_config.py -k restore -v`
Expected: FAIL — `AttributeError`/validation error, fields not defined.

- [ ] **Step 3: Write minimal implementation**

In `agastya/config.py`, inside `PipelineConfig` after `use_diffusion_sr`:

```python
    restore_backend: str = "passthrough"
    nafnet_weights: str | None = None
    restore_device: str = "cpu"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_config.py -k restore -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Verify checkpoint**

Run: `ruff check agastya/config.py && PYTHONPATH=. pytest tests/test_config.py -q`
Expected: ruff clean, all config tests pass. (No git per project rule.)

---

### Task 2: Restore package, error type, passthrough restorer

**Files:**
- Create: `agastya/stages/restore/__init__.py`
- Create: `agastya/stages/restore/errors.py`
- Create: `agastya/stages/restore/passthrough.py`
- Test: `tests/stages/restore/test_passthrough.py`

**Interfaces:**
- Produces: `RestorerUnavailable(Exception)` in `errors.py`; `PassthroughRestorer` with `restore(self, pixels: bytes) -> bytes`.

- [ ] **Step 1: Write the failing test**

```python
from agastya.stages.restore.passthrough import PassthroughRestorer


def test_passthrough_returns_input_unchanged():
    restorer = PassthroughRestorer()
    data = b"\x89PNG-bytes-here"
    assert restorer.restore(data) == data


def test_passthrough_returns_same_object_identity():
    restorer = PassthroughRestorer()
    data = b"raw"
    assert restorer.restore(data) is data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/stages/restore/test_passthrough.py -v`
Expected: FAIL — `ModuleNotFoundError: agastya.stages.restore`.

- [ ] **Step 3: Write minimal implementation**

`agastya/stages/restore/__init__.py`:

```python
```

(empty file)

`agastya/stages/restore/errors.py`:

```python
from __future__ import annotations


class RestorerUnavailable(Exception):
    pass
```

`agastya/stages/restore/passthrough.py`:

```python
from __future__ import annotations


class PassthroughRestorer:
    def restore(self, pixels: bytes) -> bytes:
        return pixels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/stages/restore/test_passthrough.py -v`
Expected: PASS (2 passed). Create `tests/stages/restore/__init__.py` (empty) if test collection needs it.

- [ ] **Step 5: Verify checkpoint**

Run: `ruff check agastya/stages/restore && PYTHONPATH=. pytest tests/stages/restore -q`
Expected: ruff clean, tests pass.

---

### Task 3: Backend factory

**Files:**
- Create: `agastya/stages/restore/factory.py`
- Test: `tests/stages/restore/test_factory.py`

**Interfaces:**
- Consumes: `PipelineConfig.restore_backend` (Task 1); `PassthroughRestorer` (Task 2); `NafnetRestorer` (Task 4 — imported lazily inside the factory branch so this task is testable before Task 4 lands).
- Produces: `build_restorer(config: PipelineConfig) -> object` returning an object with `restore(bytes) -> bytes`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from agastya.config import PipelineConfig
from agastya.stages.restore.factory import build_restorer
from agastya.stages.restore.passthrough import PassthroughRestorer


def test_factory_builds_passthrough_by_default():
    restorer = build_restorer(PipelineConfig())
    assert isinstance(restorer, PassthroughRestorer)


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        build_restorer(PipelineConfig(restore_backend="bogus"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/stages/restore/test_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: ...factory`.

- [ ] **Step 3: Write minimal implementation**

`agastya/stages/restore/factory.py`:

```python
from __future__ import annotations

from agastya.config import PipelineConfig
from agastya.stages.restore.passthrough import PassthroughRestorer


def build_restorer(config: PipelineConfig) -> object:
    if config.restore_backend == "passthrough":
        return PassthroughRestorer()
    if config.restore_backend == "nafnet":
        from agastya.stages.restore.nafnet import NafnetRestorer

        return NafnetRestorer(config.nafnet_weights, config.restore_device)
    raise ValueError(f"unknown restore_backend: {config.restore_backend}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/stages/restore/test_factory.py -v`
Expected: PASS (2 passed). The `"nafnet"` branch imports lazily, so the missing `nafnet.py` does not break these two cases.

- [ ] **Step 5: Verify checkpoint**

Run: `ruff check agastya/stages/restore/factory.py && PYTHONPATH=. pytest tests/stages/restore -q`
Expected: ruff clean, tests pass.

---

### Task 4: NAFNet adapter shell (lazy load, loud failure, decode/encode)

**Files:**
- Create: `agastya/stages/restore/nafnet.py`
- Test: `tests/stages/restore/test_nafnet.py`

**Interfaces:**
- Consumes: `RestorerUnavailable` (Task 2).
- Produces: `NafnetRestorer(weights: str | None, device: str = "cpu")` with `restore(self, pixels: bytes) -> bytes`. Model loads lazily on first `restore`. `_load_model()` is the seam tests patch.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from agastya.stages.restore.errors import RestorerUnavailable
from agastya.stages.restore.nafnet import NafnetRestorer


def _png_bytes() -> bytes:
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_nafnet_without_weights_raises():
    restorer = NafnetRestorer(weights=None)
    with pytest.raises(RestorerUnavailable):
        restorer.restore(b"anything")


def test_nafnet_missing_weights_file_raises():
    restorer = NafnetRestorer(weights="/no/such/best.pt")
    with pytest.raises(RestorerUnavailable):
        restorer.restore(b"anything")


def test_nafnet_roundtrip_with_mocked_model(monkeypatch):
    pytest.importorskip("cv2")
    restorer = NafnetRestorer(weights="/fake/best.pt")
    monkeypatch.setattr(restorer, "_load_model", lambda: (lambda arr: arr))
    out = restorer.restore(_png_bytes())
    assert isinstance(out, bytes)
    assert len(out) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/stages/restore/test_nafnet.py -v`
Expected: FAIL — `ModuleNotFoundError: ...nafnet`.

- [ ] **Step 3: Write minimal implementation**

`agastya/stages/restore/nafnet.py`:

```python
from __future__ import annotations

import os
from typing import Callable

import numpy as np

from agastya.stages.restore.errors import RestorerUnavailable


class NafnetRestorer:
    def __init__(self, weights: str | None, device: str = "cpu") -> None:
        self.weights = weights
        self.device = device
        self._model: Callable[[np.ndarray], np.ndarray] | None = None

    def _load_model(self) -> Callable[[np.ndarray], np.ndarray]:
        if not self.weights or not os.path.exists(self.weights):
            raise RestorerUnavailable(f"NAFNet weights not found: {self.weights}")
        raise RestorerUnavailable("NAFNet weights loading not implemented in this slice")

    def restore(self, pixels: bytes) -> bytes:
        import cv2

        if self._model is None:
            self._model = self._load_model()
        arr = cv2.imdecode(np.frombuffer(pixels, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("failed to decode image bytes")
        out = self._model(arr)
        ok, buf = cv2.imencode(".png", out)
        if not ok:
            raise ValueError("failed to encode restored image")
        return buf.tobytes()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/stages/restore/test_nafnet.py -v`
Expected: PASS (3 passed). First two raise `RestorerUnavailable` via real `_load_model`; the third patches `_load_model` to an identity, so `restore` decodes, applies identity, re-encodes.

- [ ] **Step 5: Verify checkpoint**

Run: `ruff check agastya/stages/restore/nafnet.py && PYTHONPATH=. pytest tests/stages/restore -q`
Expected: ruff clean, all restore tests pass.

---

### Task 5: Pipeline uses the factory; passthrough stays byte-identical

**Files:**
- Test: `tests/stages/restore/test_pipeline_restore.py`
- Reference (no change required): `agastya/pipeline.py:64-69`

**Interfaces:**
- Consumes: `build_restorer` (Task 3), `PipelineConfig` (Task 1), existing `Pipeline`, `PipelineInput`, stub stages.

- [ ] **Step 1: Write the failing test**

```python
from agastya.config import PipelineConfig
from agastya.pipeline import Pipeline, PipelineInput
from agastya.stages.restore.factory import build_restorer
from agastya.stages.restore.passthrough import PassthroughRestorer
from agastya.stages.stubs import StubDetector, StubGate, StubOCR


def test_factory_restorer_runs_pipeline_unchanged():
    cfg = PipelineConfig(gate_threshold=0.5)
    restorer = build_restorer(cfg)
    assert isinstance(restorer, PassthroughRestorer)
    pipeline = Pipeline(
        config=cfg,
        gate=StubGate(score=0.99),
        restorer=restorer,
        detector=StubDetector(),
        ocr=StubOCR(text="MH12AB1234", confidence=0.9),
    )
    result = pipeline.run(PipelineInput(image_id="img1", pixels=b"raw-bytes"))
    assert result.image_id == "img1"
    assert isinstance(result.merkle_root, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/stages/restore/test_pipeline_restore.py -v`
Expected: FAIL only if an earlier task is incomplete; otherwise confirms wiring. If it errors on import, an upstream task is unfinished — fix that first.

- [ ] **Step 3: Write minimal implementation**

No production change needed — `Pipeline` already accepts any `RestoreStage`. This task proves the factory-built restorer drives the pipeline. If a caller/assembly site constructs `StubRestorer` directly, switch it to `build_restorer(cfg)` there (search: `grep -rn StubRestorer agastya scripts`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/stages/restore/test_pipeline_restore.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Verify checkpoint**

Run: `ruff check agastya tests && PYTHONPATH=. pytest -q`
Expected: ruff clean, full suite green (prior 91 tests + new restore tests). (No git per project rule.)

---

## Self-Review

- **Spec coverage:** passthrough (Task 2), nafnet shell + loud failure (Task 4), factory toggle (Task 3), config fields (Task 1), pipeline byte-identical (Task 5), errors type (Task 2). All spec sections mapped.
- **Placeholders:** none — every code step shows full code.
- **Type consistency:** `restore(pixels: bytes) -> bytes`, `build_restorer(config) -> object`, `NafnetRestorer(weights, device)`, `RestorerUnavailable` consistent across Tasks 2–5.
- **Deferred (per spec, not gaps):** real NAFNet weights, degraded-image eval vs Phase-1 control, GPU mem tuning.
