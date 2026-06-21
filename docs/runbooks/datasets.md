# AGASTYA Dataset Acquisition Runbook

This runbook covers acquiring the three source datasets that feed the unified
AGASTYA YOLO schema. **No dataset downloads automatically.** The download
script (`scripts/download_datasets.sh`) is a dry-run by default and only the
Kaggle ANPR set can be fetched without manual registration.

## Sources

### 1. IDD / IDD-Detection (Indian Driving Dataset)

- Source: https://idd.insaan.iiit.ac.in/
- Access: **manual registration required**. Create an account, accept the
  research-use terms, and download the IDD-Detection archive by hand. There is
  no public direct-download URL and no automated fetch.
- Motorcycle-violation derivative: the work in **arXiv:2204.08364** provides
  helmet / no-helmet and trapezium (rider-region) labels derived from IDD-style
  imagery. Reuse those annotations for the helmet and rider classes rather than
  re-labelling from scratch.
- Unified classes contributed: `motorcycle`, `rider`, `person`, `car`,
  `truck`, `bus`, `auto-rickshaw` (see `agastya/data/schema_map.py`).

### 2. AI City Challenge 2024 — Track 5 (Helmet Violation Detection)

- Source: https://www.aicitychallenge.org/
- Access: **request access / approval required**. Register for the challenge
  edition, agree to the data-use agreement, and download after approval. Cannot
  be auto-fetched.
- Unified classes contributed: `motorcycle` (from `motorbike`), `helmet`,
  `no-helmet` (from the `D*Helmet` / `P*Helmet` driver/passenger labels).

### 3. Indian ANPR / License-Plate set (Kaggle)

- Source: Kaggle, e.g. `andrewmvd/car-plate-detection` and related
  indian-license-plate datasets.
- Access: **Kaggle CLI**. Install `kaggle`, place your API token at
  `~/.kaggle/kaggle.json` (chmod 600), then the download script can fetch it
  with `--confirm`.
- Unified class contributed: `license-plate`.

## On-disk layout

After acquisition, place raw data under `AGASTYA_DATA_ROOT` (default
`data/raw`):

```
data/raw/
  idd/      # IDD-Detection images + annotations (+ arXiv:2204.08364 helmet labels)
  aicity/   # AI City Track 5 images + annotations
  anpr/     # Kaggle license-plate images + annotations
```

`scripts/build_dataset.py` discovers images under each `data/raw/{source}/`
subtree.

## Running the download script

Dry run (default, no network):

```bash
bash scripts/download_datasets.sh
```

Fetch the Kaggle ANPR set (requires `~/.kaggle/kaggle.json`):

```bash
bash scripts/download_datasets.sh --confirm
```

`AGASTYA_DATA_ROOT` overrides the target root.

## Licensing / usage terms

- **IDD**: research-use license accepted at registration. Respect the
  non-commercial / attribution terms stated on the IDD site. The
  arXiv:2204.08364 derivative carries its own terms — cite the paper.
- **AI City Track 5**: bound by the challenge data-use agreement. Typically
  research-only; do not redistribute the raw imagery.
- **Kaggle ANPR**: check the specific dataset's license field on Kaggle
  (varies by uploader). Confirm commercial vs. non-commercial before any
  downstream redistribution.
