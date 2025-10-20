#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root so the script can be executed from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIOMARKER_INDICES=(14 17)
DATA_FILE="./data/data.csv"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Data file '$DATA_FILE' not found." >&2
  exit 1
fi

for roi_idx in "${BIOMARKER_INDICES[@]}"; do
  echo "Running conformal experiments for biomarker index ${roi_idx}" 

  python conformal_bootstrap.py \
    --file "$DATA_FILE" \
    --biomarker_idx "$roi_idx"

  python conformal_dkgp.py \
    --file "$DATA_FILE" \
    --biomarker_idx "$roi_idx"

  python conformal_drmc.py \
    --file "$DATA_FILE" \
    --biomarker_idx "$roi_idx"

  python conformal_quantile_regression.py \
    --file "$DATA_FILE" \
    --biomarker_idx "$roi_idx"

  python group_conditional_conformal_dkgp.py \
    --file "$DATA_FILE" \
    --roi_idx "$roi_idx"

done


echo "Conformal experiments completed."

echo "Clinical Application:"