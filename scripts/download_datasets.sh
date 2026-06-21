#!/usr/bin/env bash
set -euo pipefail

CONFIRM="${1:-}"
DATA_ROOT="${AGASTYA_DATA_ROOT:-data/raw}"

ROBOFLOW_URL="${ROBOFLOW_URL:-}"
OCR_SLUG="${OCR_SLUG:-saisirishan/indian-vehicle-dataset}"
DEBLUR_SLUG="${DEBLUR_SLUG:-}"

echo "AGASTYA dataset download plan"
echo "  target root: ${DATA_ROOT}"
echo "  1. Roboflow Triple Ride Detection (helmet/no-helmet/license-plate/triple-riding)"
echo "       set ROBOFLOW_URL to the YOLO-format export curl link from Roboflow"
echo "  2. Kaggle OCR / plate set : ${OCR_SLUG}"
echo "  3. Kaggle GoPro deblur    : ${DEBLUR_SLUG:-<set DEBLUR_SLUG>}"

if [ "${CONFIRM}" != "--confirm" ]; then
  echo
  echo "Dry run only. Re-run with --confirm to fetch."
  echo "Roboflow needs ROBOFLOW_URL; Kaggle pulls need ~/.kaggle/kaggle.json."
  exit 0
fi

if [ -z "${ROBOFLOW_URL}" ]; then
  echo "ROBOFLOW_URL not set; skipping Roboflow detection set" >&2
else
  mkdir -p "${DATA_ROOT}/roboflow"
  curl -L "${ROBOFLOW_URL}" -o "${DATA_ROOT}/roboflow/export.zip"
  unzip -o "${DATA_ROOT}/roboflow/export.zip" -d "${DATA_ROOT}/roboflow"
  echo "Roboflow set: ${DATA_ROOT}/roboflow"
fi

mkdir -p "${DATA_ROOT}/ocr"
kaggle datasets download -d "${OCR_SLUG}" -p "${DATA_ROOT}/ocr" --unzip
echo "OCR set: ${DATA_ROOT}/ocr"

if [ -n "${DEBLUR_SLUG}" ]; then
  mkdir -p "${DATA_ROOT}/deblur"
  kaggle datasets download -d "${DEBLUR_SLUG}" -p "${DATA_ROOT}/deblur" --unzip
  echo "Deblur set: ${DATA_ROOT}/deblur"
else
  echo "DEBLUR_SLUG not set; skipping GoPro deblur set" >&2
fi

echo "downloads complete under ${DATA_ROOT}"
