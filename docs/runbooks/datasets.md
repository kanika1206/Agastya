# AGASTYA Dataset Acquisition Runbook

Lean, instant-download dataset plan for the vertical slice — no registration
walls, no multi-day pulls. One dataset per slice task.

| # | Source | Task | Gives | Auth |
|---|--------|------|-------|------|
| 1a | Roboflow **Triple Ride Detection** (50 imgs) | detection | `with_helmet`, `without_helmet`, `number_plate`, `Triple_riding`, `motorcycle` | Roboflow export URL |
| 1b | Roboflow **Helmet + Number Plate (Motorbike Safety) v3** (20,287 imgs) | detection | `helmet`, `no-helmet`, `number-plate`, `bike` | Roboflow export URL |
| 2 | Kaggle **Indian vehicle** (`saisirishan/...`) | plate OCR | plate crops + (ideally) plate text | Kaggle CLI |
| 3 | Kaggle **GoPro deblur** | NAFNet restoration | paired sharp/blurry | Kaggle CLI |

Detection baseline (Plan 1) merges **sources 1a + 1b** into one 5-class set.
Datasets 2 (OCR, Plan 3) and 3 (restoration, Plan 2) feed later plans.

## Detection sources — 5-class schema

Both sources are YOLO format. `scripts/build_dataset.py` remaps each source's
class IDs into the unified 5-class order (`agastya/schema/classes.py`) via
`agastya/data/schema_map.py`:

| Source label | AGASTYA class (id) |
|--------------|--------------------|
| `with_helmet` / `helmet`     | helmet (0) |
| `without_helmet` / `no-helmet` | no-helmet (1) |
| `number_plate` / `number-plate` | license-plate (2) |
| `motorcycle` / `bike`        | motorcycle (3) |
| `person`                     | person (4) |

> **triple-riding is a RULE, not a detected class.** Free datasets carry almost
> no dedicated `triple-riding` boxes (triple=3, tvd2=5 instances), so learning it
> end-to-end is infeasible. Instead the detector learns `motorcycle` + `person`,
> and `agastya/stages/associate/rules.py` flags triple-riding when ≥3 person
> boxes overlap one motorcycle (`is_triple_riding`). Source `Triple_riding` /
> `Triple riding` labels are dropped during assemble.
>
> Full assemble of current raw data → **22,736 images**; class instances:
> helmet 43,655, no-helmet 20,189, license-plate 20,135, motorcycle 26,579,
> person 2,235 (person mostly from the overload source).

Some Triple Ride labels are exported as **segmentation polygons** rather than
boxes; the builder converts each polygon to its bounding box automatically.

Get the export URL:
1. Open the dataset on Roboflow, sign in.
2. **Download Dataset** → format **YOLOv8 / YOLO** → **show download code** →
   copy the `curl` link (contains `?key=...`).
3. Export to env and run the gated script:
   ```bash
   export ROBOFLOW_URL="https://app.roboflow.com/ds/XXXX?key=YYYY"
   bash scripts/download_datasets.sh --confirm
   ```
   Extract each export to its own dir: `data/raw/triple/` (source 1a) and
   `data/raw/safety/` (source 1b), each with `data.yaml` + `train/valid/test`
   `images/` + `labels/`.

**On download, confirm the exact Roboflow class names** (read each `data.yaml`).
If they differ from the mapping above, tell me and I update `schema_map.py`.

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
  triple/     # detection source 1a (YOLO, with data.yaml)
  safety/     # detection source 1b (YOLO, with data.yaml)
  tvd2/       # detection source — no-helmet (triple-riding dropped)
  overload/   # detection source — motorcycle + person (feeds rider-count rule)
  ocr/        # Kaggle Indian vehicle (plate OCR)
  deblur/     # Kaggle GoPro (restoration)
```

`scripts/build_dataset.py` discovers detection images under `data/raw/{triple,
safety,tvd2,overload}/`, remaps labels to the 5-class schema, and assembles a
unified YOLO dataset:

```bash
PYTHONPATH=. python3 scripts/build_dataset.py \
  --raw-root data/raw --out-root data/processed --val-fraction 0.2
# --dry-run prints the discovered image count without writing
```

Output layout (train/val split assigned deterministically by image-path hash,
filenames prefixed with their source to avoid collisions):

```
<out-root>/
  images/{train,val}/<source>_<name>.jpg
  labels/{train,val}/<source>_<name>.txt
  data.yaml   # path/train/val/nc=5/names
```

## Licensing

Check each dataset's license on its Roboflow / Kaggle page before any
redistribution. Community datasets vary; most are research/non-commercial.
