# Plan 2 — NAFNet Restore Eval (degraded-image vs Phase-1 control)

Synthetic motion blur applied to clean val frames, then detection scored under three
conditions: clean control, degraded (passthrough), degraded then NAFNet deblur.

## Reproduce

```bash
PYTHONPATH=. python3 scripts/eval_restore.py \
  --weights runs/detect/train-3/weights/best.pt \
  --data data/processed/data.yaml \
  --nafnet-weights models/NAFNet-GoPro-width32.pth \
  --samples 300 --seed 0 --kernel 15 --angle 30
```

300 random val images (seed 0). NAFNet GoPro width32, CPU. conf 0.25, IoU 0.5.

## Results

Overall P/R/F1:

| condition | P | R | F1 |
|-----------|------|------|------|
| clean (control) | 0.909 | 0.933 | 0.921 |
| degraded (passthrough) | 0.793 | 0.313 | 0.449 |
| degraded -> NAFNet | 0.909 | 0.889 | 0.899 |

Per-class F1:

| class | clean | degraded | NAFNet | deblur lift | vs clean |
|-------|-------|----------|--------|-------------|----------|
| helmet | 0.915 | 0.374 | 0.882 | +0.508 | -0.033 |
| no-helmet | 0.944 | 0.435 | 0.906 | +0.471 | -0.037 |
| license-plate | 0.937 | 0.339 | 0.934 | +0.595 | -0.003 |
| motorcycle | 0.924 | 0.647 | 0.910 | +0.264 | -0.013 |
| person | 0.763 | 0.167 | 0.778 | +0.611 | +0.015 |

## Read

- Motion blur collapses recall (0.933 -> 0.313); precision holds, so the detector
  misses degraded objects rather than hallucinating.
- NAFNet restores overall F1 from 0.449 to 0.899 — within 0.022 of the clean control.
- Person (the weak class gating the rider-count rule) recovers from F1 0.167 to 0.778,
  marginally above its own clean baseline, supporting the deblur-before-detect design.
- Caveat: synthetic single-kernel blur, 300-image subset, CPU. Real-camera blur varies
  in kernel and angle; treat lifts as directional evidence, not production guarantees.

## Angle sweep (blur-direction robustness)

```bash
PYTHONPATH=. python3 scripts/eval_angle_sweep.py \
  --weights runs/detect/train-3/weights/best.pt \
  --data data/processed/data.yaml \
  --nafnet-weights models/NAFNet-GoPro-width32.pth \
  --samples 150 --seed 0 --kernel 15 --angles 0 30 60 90 120 150
```

150 val images, kernel 15, six angles.

| angle deg | degraded F1 | NAFNet F1 | lift | deg person | naf person |
|-----------|-------------|-----------|------|------------|------------|
| 0 | 0.459 | 0.900 | +0.441 | 0.286 | 0.500 |
| 30 | 0.473 | 0.910 | +0.438 | 0.000 | 0.667 |
| 60 | 0.443 | 0.911 | +0.468 | 0.286 | 0.600 |
| 90 | 0.353 | 0.895 | +0.542 | 0.471 | 0.455 |
| 120 | 0.441 | 0.908 | +0.467 | 0.250 | 0.600 |
| 150 | 0.454 | 0.912 | +0.458 | 0.000 | 0.600 |

- NAFNet F1 stays in a tight 0.895-0.912 band across all angles — recovery is
  independent of blur direction.
- Vertical blur (90 deg) hurts the raw detector most (F1 0.353) but NAFNet still
  restores to 0.895, the largest single lift (+0.542).
- Person counts are small in the 150-image subset (noisy), but NAFNet person F1
  beats degraded at every angle except 90 (within noise).
- Conclusion: deblur block is direction-robust, not overfit to the slice-3 30 deg case.
