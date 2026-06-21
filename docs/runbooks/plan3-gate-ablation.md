# Plan 3 — ARNIQA Quality Gate (Stage 0) Eval

No-reference quality gate (ARNIQA, WACV 2024, arXiv:2310.14918) routes only degraded
frames to NAFNet restoration; clean frames bypass. Backends in `agastya/stages/gate/`:
`ArniqaGate` (pyiqa), plus `AlwaysDegradedGate` / `NeverDegradedGate` ablation arms.
Opt-in: `gate_backend` defaults to `"never"`.

## Weights

ARNIQA auto-downloads via pyiqa to `~/.cache/torch/hub/checkpoints/ARNIQA.pth`
(+ resnet50 encoder + regressor_koniq10k). pyiqa must be installed.

## Threshold tuning

```bash
PYTHONPATH=. python3 scripts/tune_gate_threshold.py \
  --data data/processed/data.yaml \
  --arniqa-weights ~/.cache/torch/hub/checkpoints/ARNIQA.pth \
  --samples 40 --seed 0
```

40 val images, clean vs motion-blur (k15/30deg):

| set | mean | min | max | median |
|-----|------|-----|-----|--------|
| clean | 0.620 | 0.215 | 0.763 | 0.679 |
| blur | 0.294 | 0.183 | 0.404 | 0.292 |

Clean and blurred ARNIQA scores separate cleanly. Best split threshold **0.362**
(95% clean/blur classification); router marks `degraded = score < threshold`.
Recommended `gate_threshold = 0.36` for the ARNIQA backend.

## Quality-gate ablation

```bash
PYTHONPATH=. python3 scripts/eval_gate_ablation.py \
  --weights runs/detect/train-3/weights/best.pt \
  --data data/processed/data.yaml \
  --nafnet-weights models/NAFNet-GoPro-width32.pth \
  --arniqa-weights ~/.cache/torch/hub/checkpoints/ARNIQA.pth \
  --samples 120 --seed 0 --threshold 0.36
```

120 images, mixed 67 degraded (motion blur) + 53 clean. CPU.

| policy | overall F1 | person F1 | restores | restore_s |
|--------|-----------|-----------|----------|-----------|
| never (no restore) | 0.727 | 0.600 | 0 | 0.0 |
| always-on (restore all) | 0.820 | 0.857 | 120 | 274.8 |
| gated (ARNIQA) | 0.909 | 0.857 | 67 | 149.9 |

- Gate routed exactly the 67 degraded frames, bypassed all 53 clean.
- Gated beats always-on on overall F1 (0.909 vs 0.820): always-on also deblurs clean
  frames, and NAFNet slightly degrades already-clean inputs — the gate avoids that.
- Gated does 44% fewer restore calls and ~45% less restore time than always-on.
- Validates the Stage-0 efficiency thesis: restore only degraded crops; blanket
  restoration both wastes latency and hurts clean images.
- Caveat: synthetic single-kernel blur, 120-image subset, CPU. Real degradation is
  more varied; treat as directional evidence.
