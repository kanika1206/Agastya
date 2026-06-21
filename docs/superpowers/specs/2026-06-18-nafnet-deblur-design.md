# NAFNet Deblur Block — Design Spec (Plan 2, slice 1)

**Date:** 2026-06-18
**Status:** approved, pre-implementation
**Project:** AGASTYA traffic-violation CV pipeline

## Goal

Add a structured image-restoration stage to the pipeline so degraded frames can
be deblurred before detection. This slice builds the module structure, backend
toggle, and a NAFNet adapter shell — without shipping real NAFNet weights yet.
Real weights and the eval-vs-baseline comparison are a later slice.

## Context

The pipeline already defines a `RestoreStage` protocol
(`restore(pixels: bytes) -> bytes`) and calls it in `Pipeline.run` when the gate
flags an image as degraded:

```python
decision = score_to_decision(self.gate.score_image(pixels), self.config.gate_threshold)
if decision.degraded:
    pixels = self.restorer.restore(pixels)
```

Today the only implementation is `StubRestorer` (identity passthrough) in
`agastya/stages/stubs.py`. The seam exists; this slice formalizes a real
restore module behind it and adds the NAFNet backend shell.

## Scope

### In scope

New module `agastya/stages/restore/`:

- **`passthrough.py`** — `PassthroughRestorer.restore(bytes) -> bytes`, returns
  input unchanged. Replaces ad-hoc `StubRestorer` for pipeline wiring.
- **`nafnet.py`** — `NafnetRestorer`:
  - Lazy torch model load from `config.nafnet_weights` (load on first
    `restore`, not at construction).
  - `restore`: decode bytes → tensor → NAFNet inference → encode back to bytes
    (cv2 for decode/encode).
  - If weights path is `None` or missing, raise `RestorerUnavailable` with a
    clear message. **No silent passthrough fallback** — a missing model is a
    loud failure, not a quiet no-op.
- **`factory.py`** — `build_restorer(config) -> RestoreStage`, selecting
  `"passthrough"` or `"nafnet"` by `config.restore_backend`. Unknown backend
  raises `ValueError`.
- **`errors.py`** — `RestorerUnavailable` exception.

`config.py` additions (frozen dataclass, immutable):

- `restore_backend: str = "passthrough"`
- `nafnet_weights: str | None = None`
- `restore_device: str = "cpu"`

### Out of scope (deferred to next slice)

- Real NAFNet checkpoint download / vendoring.
- Degraded-image evaluation vs the Phase-1 detection control.
- GPU memory tuning for T1000 (4 GB).
- Gate model that actually scores blur (still stubbed).

## Data flow

Unchanged. `Pipeline.run` already calls `restorer.restore` on degraded frames.
This slice only changes *which* restorer is constructed, via
`build_restorer(config)`. With `restore_backend="passthrough"` (default),
behavior is byte-identical to today.

## Components & boundaries

| Unit | Does | Used by | Depends on |
|------|------|---------|------------|
| `PassthroughRestorer` | identity restore | factory, tests | none |
| `NafnetRestorer` | deblur via NAFNet | factory, tests | torch, cv2, config |
| `build_restorer` | pick backend by config | pipeline assembly | config, the two restorers |
| `RestorerUnavailable` | signal missing model | NafnetRestorer | none |

Each restorer satisfies the existing `RestoreStage` protocol, so the pipeline
stays decoupled from the concrete backend.

## Error handling

- Missing/None weights → `RestorerUnavailable` (loud, on first `restore`).
- Unknown `restore_backend` → `ValueError` from factory.
- Decode failure (corrupt bytes) → propagate as a clear error, do not return
  empty/garbage bytes.

## Testing (TDD — RED first)

1. `PassthroughRestorer` returns input bytes unchanged.
2. `build_restorer` returns `PassthroughRestorer` for `"passthrough"`,
   `NafnetRestorer` for `"nafnet"`, raises `ValueError` for unknown.
3. `NafnetRestorer.restore` with no weights raises `RestorerUnavailable`.
4. `NafnetRestorer` decode→encode roundtrip with a mocked model (identity
   stand-in) returns valid image bytes — no real torch download.
5. Pipeline integration: `restore_backend="passthrough"` yields the same
   `PipelineResult` as the current `StubRestorer` path.

## Success criteria

- All new tests green; full suite still passes.
- `ruff` clean.
- Pipeline with `restore_backend="passthrough"` is byte-identical to current
  behavior.
- `"nafnet"` path is reachable and fails loud (`RestorerUnavailable`) without
  weights.
- No code comments (project rule). No git operations.
