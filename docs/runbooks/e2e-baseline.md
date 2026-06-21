# AGASTYA End-to-End Violation Baseline Runbook

Image-level **violation** baseline for the full pipeline (gate → restore →
detect → associate → ocr → verify). Unlike [phase1-baseline.md](phase1-baseline.md)
(per-class **detection** mAP), this measures the two emitted violations —
**no-helmet** and **triple-riding** — as image-level presence (TP/FP/FN).

Numbers below are from a real run on the full validation set. Do **not** invent
metrics — re-fill only from a real `eval_e2e.py` run.

## Locked configuration

| knob | value | source |
|------|-------|--------|
| `detect_conf` | 0.25 | detector emission floor |
| `no_helmet_min_conf` | **0.25** | full-val F1 max (sweep) |
| `triple_riding_overlap` | **0.10** | full-val F1 max (sweep) |
| `gate_backend` | `never` | gate opt-in, off for baseline |
| `restore_backend` | `passthrough` | restore opt-in, off for baseline |
| `associate_backend` | `box` | default box-overlap associator |
| `ocr_backend` | `none` | OCR opt-in, off for baseline |
| calibration | `models/calibration.json` | temperature 0.75, qhat 0.1279 |

Both threshold knobs are at their full-val F1 maxima — confirmed saturated by the
sweeps in [Tuning history](#tuning-history). The two violation F1 scores are now
**detector-bound**, not threshold-bound.

## Reproduce

```bash
PYTHONPATH=. python3 scripts/eval_e2e.py \
  --weights runs/detect/train-3/weights/best.pt \
  --data data/processed/data.yaml \
  --samples 5000 --device cuda --conf 0.25 \
  --gate-backend never --associate-backend box \
  --calibration models/calibration.json
```

`--samples 5000` takes all 4566 val images (slice cap above the set size).

## Violation table (full val, 4566 images)

| violation      | P     | R     | F1    | TP   | FP | FN  |
|----------------|-------|-------|-------|------|----|-----|
| no-helmet      | 0.972 | 0.962 | **0.967** | 2174 | 63 | 87  |
| triple-riding  | 0.888 | 0.898 | **0.893** | 79   | 10 | 9   |

Per-stage mean latency: detect **12.3 ms/img** (~all pipeline cost), gate /
associate / ocr ~0 ms. OCR abstains on all 4566 (`ocr_backend=none`). Calibration
mean confidence 0.949 over 2326 emitted violations.

## Tuning history

- **no-helmet floor.** 16-sample sweep suggested a 0.70 per-class floor (claimed
  F1→1.0). Full-val sweep **inverted** this: F1 is monotonic-decreasing in the
  threshold (best 0.967 @ 0.25; 0.70 → 0.927; 0.85 → 0.775). At scale, lowering
  the floor gains recall faster than it loses precision. Floor relocked
  0.70 → **0.25**. `scripts/sweep_nohelmet_conf.py`.
- **triple-riding overlap.** Decoupled sweep (GT triple held fixed at reference
  overlap 0.10; sweep only the predictor overlap on detected boxes — avoids the
  circular confound where one shared knob moves pred and GT together). Best F1
  0.893 on a flat plateau 0.10–0.20; current default 0.10 already optimal, no
  change. `scripts/sweep_triple_overlap.py`.

> **Caveat.** `eval_e2e.py` uses the same `triple_riding_overlap` for both the
> prediction and the derived ground truth, so its triple-riding metric is
> self-referential. The decoupled `sweep_triple_overlap.py` is the clean
> measurement of the predictor threshold.

## Run record

- **Date:** 2026-06-19
- **Weights:** `runs/detect/train-3/weights/best.pt` (yolo26-p2)
- **Hardware:** local T1000 GPU (cuda)
- **Dataset:** `data/processed/data.yaml` val split, 4566 images
- **Calibration:** `models/calibration.json` (temperature 0.75, qhat 0.1279,
  alpha 0.1, n_predictions 10555)

## Next levers

Both threshold knobs are saturated, so further F1 gains require improving the
detector, not the thresholds:

- person/moto box quality (retrain) — lifts triple-riding and no-helmet box
  precision.
- SAM2 mask association — replaces box-overlap triple-riding heuristic.

Both deferred (no GPU-train in this work stream).
