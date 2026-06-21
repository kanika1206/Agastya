# AGASTYA Phase 1 — Detection Baseline Runbook

End-to-end gated sequence to produce the **detection baseline control**. Every
step is gated: it runs only after the prerequisites are met and you give an
explicit written "go". Do **not** invent metrics — the control table is filled
only from a real run.

This baseline is the **control** against which every later novel block
(restoration, gating, association, loss) must demonstrate a measurable gain.

## Prerequisites

- IDD and AI City Track 5 access approved and downloaded manually (see
  [datasets.md](datasets.md)).
- Kaggle ANPR set fetched.
- Per-source annotation parsers added to `scripts/build_dataset.py` (the
  label-conversion step depends on each source's on-disk annotation format).
- A CUDA GPU for training and evaluation.

## Sequence

1. **Acquire datasets** (after manual IDD / AI City access):

   ```bash
   bash scripts/download_datasets.sh --confirm
   ```

2. **Build the unified YOLO dataset** (after label parsers added):

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

   For the transparent per-class P/R/F1 control, use `agastya.eval.prf`
   (`match_detections` + `precision_recall_f1`) over the val predictions.

## Control table (fill from the real run — do NOT invent numbers)

| class         | P | R | F1 | AP@50 | AP@50-95 |
|---------------|---|---|----|-------|----------|
| motorcycle    |   |   |    |       |          |
| rider         |   |   |    |       |          |
| person        |   |   |    |       |          |
| car           |   |   |    |       |          |
| truck         |   |   |    |       |          |
| bus           |   |   |    |       |          |
| auto-rickshaw |   |   |    |       |          |
| helmet        |   |   |    |       |          |
| no-helmet     |   |   |    |       |          |
| license-plate |   |   |    |       |          |
| **overall**   |   |   |    |       |          |

Record the run date, dataset commit/manifest, weights path, and training
config (epochs, imgsz, batch) alongside the filled table so the control is
reproducible.
