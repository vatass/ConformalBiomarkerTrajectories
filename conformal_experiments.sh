#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root so the script can be executed from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIOMARKER_INDICES=(14 17)
ALPHAS=(0.05 0.1)
DATA_FILE="./data/data.csv"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Data file '$DATA_FILE' not found." >&2
  exit 1
fi

# Creates the 10-fold train/test splits
python create_folds.py

for roi_idx in "${BIOMARKER_INDICES[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    echo "Running conformal experiments for biomarker index ${roi_idx} with alpha=${alpha}"

    python conformal_bootstrap.py \
      --file "$DATA_FILE" \
      --biomarker_idx "$roi_idx" \
      --alpha "$alpha" \
      --calibrationset 0.2

    python conformal_dkgp.py \
      --file "$DATA_FILE" \
      --biomarker_idx "$roi_idx" \
      --alpha "$alpha" \
      --calibrationset 0.2

    python conformal_drmc.py \
      --file "$DATA_FILE" \
      --biomarker_idx "$roi_idx" \
      --alpha "$alpha" \
      --calibrationset 0.2

    python conformal_quantile_regression.py \
      --file "$DATA_FILE" \
      --biomarker_idx "$roi_idx" \
      --alpha "$alpha" \
      --calibrationset 0.2

  done
done

echo "Conformal experiments completed."

echo "Running group conditional conformal dkgp for the Diagnosis variable..."

for roi_idx in "${BIOMARKER_INDICES[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    echo "Running conformal experiments for biomarker index ${roi_idx} with alpha=${alpha}"

    python group_conditional_conformal_dkgp.py \
      --file "$DATA_FILE" \
      --biomarker_idx "$roi_idx" \
      --alpha "$alpha" \
      --calibrationset 0.2
  done
done

echo "Genarate Plots.."
python plots.py

echo "Gather Results.."
python gather_results.py

echo "Clinical Application: Identification of High Risk Subjects"
python clinical_application.py