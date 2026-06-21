# AGASTYA Training Configs

## `yolo26-p2.yaml`

The baseline detector is YOLO26-m with an added **P2 small-object head**.

- Instantiate the architecture from Ultralytics' `yolo26-p2.yaml` model
  definition. The P2 head adds a higher-resolution feature level for small
  objects (helmets, license plates, distant riders).
- There are **no pretrained P2 weights**. Train from the standard `yolo26m.pt`
  backbone via `.load(weights)` — the matching backbone layers are warm-started
  and the P2 head trains from scratch.

## YOLO26 native training behaviour

- **ProgLoss**, **STAL**, and **MuSGD** are native to YOLO26 and are active by
  default; no extra flags required.
- **AMP** (mixed precision) is enabled (`amp=True`) for VRAM and speed.
- **Auto-batch** via `batch=-1` lets Ultralytics size the batch to available
  VRAM.
- Default `imgsz=640`, `epochs=100`.

## Low-VRAM profile

When the `PipelineConfig` VRAM profile is constrained, swap the model to
`yolo26-s` (smaller variant) instead of `yolo26-m`. Keep AMP and auto-batch on.

## Running (gated)

Dry run (no GPU/torch needed — `ultralytics` is imported lazily only under
`--confirm`):

```bash
python3 scripts/train_baseline.py --data data/processed/data.yaml
```

Real training (requires assembled dataset + GPU + written "go"):

```bash
python3 scripts/train_baseline.py --data data/processed/data.yaml --confirm
```
