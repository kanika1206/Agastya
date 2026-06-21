# AGASTYA Phase 4 — E2E Ablation Matrix

Add-one-in fractional ablation measuring the marginal value of each Phase 4 novel block
(restore, gate-routing, SAM2 association, calibration) against the locked control, on
clean and motion-blur-degraded inputs.

## Locked config (control)

- `no_helmet_min_conf` 0.25, `triple_riding_overlap` 0.10
- detector `runs/detect/train-3/weights/best.pt`, `detect_conf` 0.25, device cuda
- gate never, restore passthrough, associate box, calibration off
- Matches `docs/runbooks/e2e-baseline.md`.

## Method

- 5 arms (control, restore_blanket, gate_routed, sam2, calibration) × {clean, degraded} = 10 cells.
- Fixed first-500 val subset (`sorted(...)[:500]`), identical every cell.
- Degraded = `motion_blur` kernel_size 15, angle_deg 30.0.
- Restore runs NAFNet on CPU (8 threads); detector/SAM2 on cuda. Gate (ARNIQA) on CPU.
- Δ = arm − control of the same condition. Ranking: triple ΔF1, then no-helmet ΔF1, then Δlatency.

## Results (500-image subset)

| arm | condition | no-helmet F1 | ΔF1 | triple F1 | ΔF1 | lat ms | Δlat | restore n | rate |
|---|---|---|---|---|---|---|---|---|---|
| control | clean | 0.971 | +0.000 | 0.940 | +0.000 | 17.83 | +0.00 | 0 | 0.000 |
| control | degraded | 0.793 | +0.000 | 0.128 | +0.000 | 11.87 | +0.00 | 0 | 0.000 |
| restore_blanket | clean | 0.967 | -0.004 | 0.933 | -0.007 | 23.21 | +5.39 | 500 | 1.000 |
| restore_blanket | degraded | 0.947 | +0.154 | 0.846 | +0.718 | 22.56 | +10.69 | 500 | 1.000 |
| gate_routed | clean | 0.971 | +0.000 | 0.947 | +0.006 | 16.22 | -1.60 | 148 | 0.296 |
| gate_routed | degraded | 0.945 | +0.152 | 0.846 | +0.718 | 21.47 | +9.61 | 484 | 0.968 |
| sam2 | clean | 0.971 | +0.000 | 0.000 | -0.940 | 13.69 | -4.14 | 0 | 0.000 |
| sam2 | degraded | 0.793 | +0.000 | 0.000 | -0.128 | 12.97 | +1.10 | 0 | 0.000 |
| calibration | clean | 0.971 | +0.000 | 0.940 | +0.000 | 13.26 | -4.56 | 0 | 0.000 |
| calibration | degraded | 0.793 | +0.000 | 0.128 | +0.000 | 12.47 | +0.60 | 0 | 0.000 |

## Promotion decisions

| arm | verdict | justification |
|---|---|---|
| **gate_routed** | **WINNER** | Degraded recovery ΔtripleF1 +0.718, ΔnohelmetF1 +0.152; clean-neutral (+0.006 triple, −1.6ms); routes only 148/500 clean and 484/500 degraded. Same degraded recovery as blanket with 3.4× fewer restores and no clean-frame harm. |
| restore_blanket | dominated | Matches gate_routed's degraded recovery (+0.718 / +0.154) but *hurts* clean (−0.007 triple, −0.004 no-helmet) and restores all 500 clean frames. Strictly worse than routed. |
| sam2 | stop | triple F1 0.000 both conditions — SAM2 associator emits zero triple detections. Mask-overlap semantics untuned. Deferred (needs threshold tuning, not promotion). |
| calibration | neutral | 0.000 ΔF1 everywhere — by design: calibration adjusts confidence/conformal sets, not violation-presence, so it is invisible to this F1 metric. No detection-quality effect. |

## Winner summary

**Routed restore (ARNIQA gate → NAFNet) is the only Phase 4 block providing measurable
violation-detection value.** It recovers nearly all motion-blur damage (triple-riding F1
0.128 → 0.846, no-helmet 0.793 → 0.947) while leaving clean frames untouched and skipping
~70% of restores on clean input. This confirms the Plan-3 quality-gate thesis
(`docs/runbooks/plan3-gate-ablation.md`) end-to-end in violation terms: routed beats blanket.

Blur destroys triple-riding detection at the control (F1 0.940 → 0.128) because person/moto
box quality collapses; restore is what brings it back.

## Reproduce

```bash
PYTHONPATH=. python3 scripts/eval_ablation.py \
  --weights runs/detect/train-3/weights/best.pt \
  --data data/processed/data.yaml \
  --samples 500 --device cuda --restore-device cpu --cpu-threads 8 \
  > /tmp/agastya_ablation_500.log 2>&1
```

Run record: 500-subset matrix completed in ~30 min (10 cells), peak RSS 8.7 GB, no OOM.

## Notes / caveats

- **Full-val (4566) confirmation deferred.** The first attempt OOM'd: ARNIQA gate on cuda
  needs a ~980 MiB contiguous allocation on a large full-val frame, and the 3.8 GB T1000 has
  only ~500 MiB free with YOLO co-resident. Gate has since been moved to CPU
  (`build_gate(restore_config)`), making a rerun OOM-safe, but the 500-subset deltas
  (degraded +0.718) are already decisive, so the rerun was skipped for cost.
- **Two driver fixes** were required to run at all (`scripts/eval_ablation.py` only):
  (1) restore device is decoupled from detect/gate/sam2 via a `restore_config` copy
  (`--restore-device cpu`), so NAFNet runs on CPU while YOLO/SAM2 stay on cuda;
  (2) `torch.set_num_threads(--cpu-threads, default 8)` is pinned before each restore,
  because ultralytics YOLO resets torch to a single thread, which had made CPU NAFNet
  ~3.4× slower (3.15 → 0.9 s/img).
- The harness adds `--log-every` progress logging (cell boundaries, per-N images, wall-time,
  peak RSS) and an `--arms` filter (control always included as the Δ reference).
- The driver's triple-riding GT uses the same `triple_riding_overlap` for prediction and
  ground truth, so its triple metric is self-referential (same caveat as `e2e-baseline.md`);
  the deltas vs control remain valid since the confound is held constant across arms.

## Status: Phase 4 Slice 1 complete — remaining scope cut to future work

The locked baseline (`e2e-baseline.md`) plus this ablation are the shipped deliverable.
Every remaining lever requires GPU training or model tuning and is explicitly deferred:

- **SAM2 associator tuning** — the `sam2` arm emits 0 triples; mask-overlap thresholds in
  `agastya/stages/associate/sam2.py` need tuning (no retrain). Highest ROI when resumed.
- **person/moto detector retrain** — the real F1 ceiling is detector box quality (lifts both
  triple-riding and no-helmet). Heaviest.
- **PARSeq LoRA** — Indian-plate OCR finetune; OCR is off the critical path, lowest priority.
- **gate_routed full-val (4566) confirm** — gate is now CPU-safe so a rerun would not OOM,
  but the 500-subset deltas (degraded +0.718) are already decisive. Optional, for the record.
