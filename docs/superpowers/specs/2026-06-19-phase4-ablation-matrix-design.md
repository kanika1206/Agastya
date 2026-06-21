# Phase 4 Slice 1 — E2E Ablation Matrix (Design)

**Date:** 2026-06-19
**Status:** Approved (design)
**Phase:** 4 (Evidence + analytics + ablations) — first slice

## Goal

Identify which Phase 4 / novel-block components provide **measurable value** in
the full violation pipeline, using an add-one-in fractional ablation against the
locked control. Not an exhaustive interaction matrix — marginal value first,
interactions among winners deferred to a later slice.

## Control

The locked full-val baseline (see `docs/runbooks/e2e-baseline.md`):

- `no_helmet_min_conf` 0.25, `triple_riding_overlap` 0.10
- gate `never`, restore `passthrough`, associate `box`, calibration off
- detector `runs/detect/train-3/weights/best.pt`, conf 0.25, cuda

## Arms (add-one-in)

| arm | gate | restore | associate | calib | isolates |
|-----|------|---------|-----------|-------|----------|
| **Control** | never | passthrough | box | off | locked baseline |
| **+Restore (blanket)** | always | nafnet | box | off | always-on restore value |
| **+Gate (routed)** | arniqa @ 0.36 | nafnet | box | off | smart routing vs blanket |
| **+SAM2** | never | passthrough | sam2 | off | mask assoc vs box |
| **+Calibration** | never | passthrough | box | on | confidence / abstention |

**Restore coupling.** In the pipeline, restore fires only when the gate flags a
frame degraded (`gate=never` ⇒ restore never runs). Restore therefore cannot be
ablated as a lone toggle. Its two arms become **blanket** (`gate=always`, NAFNet
on every frame) vs **routed** (`gate=arniqa`, NAFNet only on flagged frames) —
the plan3 question re-expressed in full e2e violation terms.

5 arms.

## Conditions

- **clean** — val images as-is.
- **degraded** — motion blur via `agastya/eval/degrade.py` `motion_blur` (k15,
  the plan2 setting), applied to image bytes at the top of the eval loop before
  gate/detect, so every stage sees the degraded input.

5 arms × 2 conditions = **10 cells**.

## Sample

Fixed first-500 val images (`sorted(...)[:500]` — deterministic, identical set
every cell, comparable). Winners re-confirmed on full 4566.

## New code

Single driver `scripts/eval_ablation.py`. No changes to pipeline or stages.

- Imports existing stage builders (`build_gate`, `build_restorer`,
  `build_detector`, `build_associator`, `build_ocr`), `Calibrator.from_json`,
  `eval/e2e.py` helpers (`gt_violations`, `accumulate`, `ViolationCounts`,
  `VIOLATION_TYPES`), `eval/degrade.py` `motion_blur`, `eval/yolo_data.py`
  loaders.
- Defines the 5 arms as config presets; loops arms × {clean, degraded} over the
  fixed 500-img subset; runs the same per-image stage sequence as `eval_e2e.py`
  (gate → optional restore → detect → no-helmet floor → associate → ocr →
  optional calib), with the degraded condition applying `motion_blur` first.
- Reuses the locked floor (0.25) and overlap (0.10) from config defaults.

## Metrics per cell

- no-helmet P / R / F1, TP / FP / FN
- triple-riding P / R / F1, TP / FP / FN
- `restore_invoked` (count) **and** `restore_invoked_rate` (= restore_invoked /
  n_images) — for the routed arm this is the fraction ARNIQA sends through NAFNet
- `detect_hit` count
- mean detect latency (ms/img); total per-image latency where it differs (NAFNet
  / SAM2 arms)

## Final matrix + ranking

Emit a markdown comparison matrix (arm × condition rows, metric columns) to
stdout and to `docs/runbooks/phase4-ablation.md`. Every cell reports both
absolute metrics and deltas versus the locked control (Control row deltas are
0):

- triple-riding F1 and ΔF1
- no-helmet F1 and ΔF1
- mean latency (ms/img) and Δ latency
- `restore_invoked_count`
- `restore_invoked_rate`

Rank arms by delta vs control, in priority order:

1. triple-riding F1 delta vs control
2. no-helmet F1 delta vs control
3. latency delta vs control

## Promotion gate

After the 500-img run, promote to the full 4566-img confirmation run **only**
arms showing a meaningful F1 gain on either violation, or a meaningful latency
tradeoff worth recording. Uninformative arms (no gain, no interesting cost) stop
at 500.

## Out of scope (this slice)

- Full 16-cell interaction matrix (deferred to interactions-among-winners slice).
- 6th always-on-NAFNet-without-gate arm (not needed yet).
- C2PA / Grad-CAM / Merkle evidence block (Phase 4 block 2).
- SQLite/Parquet store + dashboard (Phase 4 block 3).
- RT-DETRv2 / YOLOv11 detector comparison, multi-frame ByteTrack, diffusion SR
  (Phase 4 optional extras).

## Testing

- Unit test arm-preset construction (each arm yields the expected
  `PipelineConfig` toggles).
- Unit test the ranking function (given synthetic per-arm metrics, returns the
  correct priority-ordered ranking and delta computation).
- Unit test `restore_invoked_rate` computation (count / n_images, 0-div guard).
- Reuse existing `degrade` / `e2e` unit coverage; no new stage logic to test.
