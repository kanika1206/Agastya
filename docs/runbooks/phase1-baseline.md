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

| class         | P     | R     | F1    | AP@50 | AP@50-95 |
|---------------|-------|-------|-------|-------|----------|
| helmet        | 0.950 | 0.909 | 0.929 | 0.968 | 0.703    |
| no-helmet     | 0.937 | 0.909 | 0.923 | 0.959 | 0.685    |
| license-plate | 0.957 | 0.962 | 0.959 | 0.986 | 0.724    |
| motorcycle    | 0.924 | 0.943 | 0.933 | 0.982 | 0.851    |
| person        | 0.758 | 0.744 | 0.751 | 0.732 | 0.315    |
| **overall**   | 0.905 | 0.893 | 0.899 | 0.926 | 0.656    |

P and R from `yolo val` on `best.pt` (4566 val images, 22679 instances). F1
computed as 2·P·R/(P+R). `person` is the weak class (P 0.758, R 0.744, AP@50
0.732, AP@50-95 0.315) and gates rider-count rule accuracy — prioritize it in
later improvement blocks.

triple-riding is not a detected class — it is derived post-detection by the
rider-count rule (`agastya/stages/associate/rules.py`). Evaluate it separately
on images with ground-truth rider counts, not in this per-class detection table.

Record the run date, dataset version (Roboflow export id), weights path, and
training config (epochs, imgsz, batch) alongside the filled table so the
control is reproducible.

### Run record

- **Date:** 2026-06-18
- **Weights:** `runs/detect/train-3/weights/best.pt` (yolo26-p2, 2.4M params)
- **Hardware:** Agastya cluster, Tesla V100
- **Config:** 100 epochs, imgsz 640, P2 head, AMP. Train time ~9.7h.
- **Source:** AP values from the training-run validation summary. Re-confirm
  anytime with:
  ```bash
  yolo val model=runs/detect/train-3/weights/best.pt data=data/processed/data.yaml
  ```
- **Dataset version:** record the Roboflow export id used (not captured here).
