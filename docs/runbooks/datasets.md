# AGASTYA Dataset Acquisition Runbook

Lean, instant-download dataset plan for the vertical slice — no registration
walls, no multi-day pulls. One dataset per slice task.

| # | Source | Task | Gives | Auth |
|---|--------|------|-------|------|
| 1 | Roboflow **Triple Ride Detection** | detection | `with_helmet`, `without_helmet`, `number_plate`, `Triple_riding` | Roboflow export URL |
| 2 | Kaggle **Indian vehicle** (`saisirishan/...`) | plate OCR | plate crops + (ideally) plate text | Kaggle CLI |
| 3 | Kaggle **GoPro deblur** | NAFNet restoration | paired sharp/blurry | Kaggle CLI |

Detection baseline (Plan 1) uses **dataset 1 only**. Datasets 2 (OCR, Plan 3)
and 3 (restoration, Plan 2) feed later plans.

## Dataset 1 — Roboflow Triple Ride Detection

Already in YOLO format → maps straight to our 4-class schema via
`agastya/data/schema_map.py`:

| Roboflow label | AGASTYA class |
|----------------|---------------|
| `with_helmet`    | helmet |
| `without_helmet` | no-helmet |
| `number_plate`   | license-plate |
| `Triple_riding`  | triple-riding |

Get the export URL:
1. Open the dataset on Roboflow, sign in.
2. **Download Dataset** → format **YOLOv8 / YOLO** → **show download code** →
   copy the `curl` link (contains `?key=...`).
3. Export to env and run the gated script:
   ```bash
   export ROBOFLOW_URL="https://app.roboflow.com/ds/XXXX?key=YYYY"
   bash scripts/download_datasets.sh --confirm
   ```
   Lands under `data/raw/roboflow/` with `data.yaml` + `train/valid/test`
   `images/` + `labels/`.

**On download, confirm the exact Roboflow class names** (read its `data.yaml`).
If they differ from the four above, tell me and I update `schema_map.py`.

## Dataset 2 — Kaggle Indian vehicle (plate OCR)

```bash
export OCR_SLUG="saisirishan/indian-vehicle-dataset"   # confirm exact slug
bash scripts/download_datasets.sh --confirm
```
**Confirm it includes plate-number text**, not just boxes. If boxes-only: skip
it, crop plates from dataset 1, and run PARSeq pretrained — drops us to 2
datasets.

## Dataset 3 — Kaggle GoPro deblur

```bash
export DEBLUR_SLUG="<owner>/<gopro-deblur-slug>"   # set the Kaggle mirror slug
bash scripts/download_datasets.sh --confirm
```
Standard paired sharp/blurry benchmark for NAFNet. Generic blur — nothing
India-specific needed.

## Kaggle CLI setup (datasets 2 and 3)

1. `pip install kaggle`
2. kaggle.com → avatar → **Settings** → **API** → **Create New Token** →
   downloads `kaggle.json`.
3. ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Verify: `kaggle datasets list`.

## On-disk layout

```
data/raw/
  roboflow/   # detection (YOLO format, with data.yaml)
  ocr/        # Kaggle Indian vehicle (plate OCR)
  deblur/     # Kaggle GoPro (restoration)
```

`scripts/build_dataset.py` discovers the detection images under
`data/raw/roboflow/`.

## Licensing

Check each dataset's license on its Roboflow / Kaggle page before any
redistribution. Community datasets vary; most are research/non-commercial.
