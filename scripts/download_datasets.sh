#!/usr/bin/env bash
set -euo pipefail

CONFIRM="${1:-}"

DATA_ROOT="${AGASTYA_DATA_ROOT:-data/raw}"

echo "AGASTYA dataset download plan"
echo "  target root: ${DATA_ROOT}"
echo "  IDD / IDD-Detection : register + download from https://idd.insaan.iiit.ac.in/ (manual auth)"
echo "  AI City 2024 Track5 : request access via https://www.aicitychallenge.org/ (manual auth)"
echo "  Indian ANPR set     : Kaggle indian-license-plate datasets (kaggle CLI)"

if [ "${CONFIRM}" != "--confirm" ]; then
  echo
  echo "Dry run only. Re-run with --confirm to fetch the Kaggle ANPR set."
  echo "IDD and AI City require manual registration and cannot be auto-fetched."
  exit 0
fi

mkdir -p "${DATA_ROOT}/anpr"
kaggle datasets download -d andrewmvd/car-plate-detection -p "${DATA_ROOT}/anpr" --unzip
echo "ANPR download complete: ${DATA_ROOT}/anpr"
