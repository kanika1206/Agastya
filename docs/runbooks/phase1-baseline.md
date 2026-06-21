# AGASTYA Phase 1 — Detection Baseline Runbook

End-to-end gated sequence to produce the **detection baseline control**. Every
step runs only after prerequisites are met and you give an explicit written
"go". Do **not** invent metrics — the control table is filled only from a real
run.

This baseline is the **control** against which every later novel block
(restoration, gating, evidence) must show a measurable gain.

## Prerequisites

- Roboflow **Triple Ride Detection** set downloaded to `data/raw/roboflow/`
  (see [datasets.md](datasets.md)). This is the only dataset the detection
  baseline needs.
- Roboflow class names confirmed against `schema_map.py` (`with_helmet`,
  `without_helmet`, `number_plate`, `Triple_riding`).
- A CUDA GPU for training and evaluation.

## Sequence

1. **Acquire the detection set:**
   ```bash
   export ROBOFLOW_URL="https://app.roboflow.com/ds/XXXX?key=YYYY"
   bash scripts/download_datasets.sh --confirm
   ```

2. **Build the unified YOLO dataset** (remaps Roboflow class IDs to the
   AGASTYA 4-class order, writes our `data.yaml`):
   ```bash
   PYTHONPATH=. python3 scripts/build_dataset.py \
     --raw-root data/raw --out-root data/processed
   ```

3. **Train the YOLO26-m baseline** (P2 head, AMP, imgsz 640):
   ```bash
   python3 scripts/train_baseline.py \
     --data data/processed/data.yaml --confirm
   ```

4. **Validate SAHI ↔ NMS-free head** on a sample image:
   ```bash
   python3 scripts/validate_sahi.py \
     --weights runs/detect/train/weights/best.pt \
     --image <sample.jpg> --confirm
   ```

5. **Evaluate mAP** (mAP@50 and mAP@50–95):
   ```bash
   yolo val model=runs/detect/train/weights/best.pt \
     data=data/processed/data.yaml
   ```
   For transparent per-class P/R/F1, use `agastya.eval.prf`
   (`match_detections` + `precision_recall_f1`) over the val predictions.

## Control table (fill from the real run — do NOT invent numbers)

| class         | P | R | F1 | AP@50 | AP@50-95 |
|---------------|---|---|----|-------|----------|
| helmet        |   |   |    |       |          |
| no-helmet     |   |   |    |       |          |
| license-plate |   |   |    |       |          |
| triple-riding |   |   |    |       |          |
| **overall**   |   |   |    |       |          |

Record the run date, dataset version (Roboflow export id), weights path, and
training config (epochs, imgsz, batch) alongside the filled table so the
control is reproducible.
