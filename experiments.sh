#!/bin/bash

echo "Starting Conformal Prediction experiments..."
echo "Train the Volume Functions..."

biomarker='MUSE'
# List of specific ROI indices
roi_indices=(13 17 23 14 4 5 109)
# Extensive Experimentation of Calibration Set Sizes for Different Confidence Levels
# List of conformal split percentages
# conformalsplitpercentages=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)
# conformalsplitpercentages=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

conformalsplitpercentages=(0.2 0.3)
alphas=(0.1 0.01 0.05)
# Iterate over the specific ROI indices, conformal split percentages, and alphas
for roi_idx in "${roi_indices[@]}"; do
    for split_percentage in "${conformalsplitpercentages[@]}"; do
        for alpha in "${alphas[@]}"; do
            echo "Training for ROI index: $roi_idx with conformal split percentage: $split_percentage and alpha: $alpha"
            python advanced_split_conformal_dkgp.py --gpuid 0 \
                                     --file conformal_longitudinal_abniblsa.csv \
                                     --roi_idx "$roi_idx" \
                                     --conformalsplitpercentage "$split_percentage" \
                                     --alpha "$alpha" \
                                     --task "MUSE"
            
            python quantileregression.py --gpuid 0 \
                              --file conformal_longitudinal_abniblsa.csv \
                              --roi_idx "$roi_idx" \
                              --calibrationset "$split_percentage" \
                              --alpha "$alpha" \
                              --task "MUSE"
            
            python deep_regression_with_montecarlo.py --gpuid 0 \
                                           --file conformal_longitudinal_abniblsa.csv \
                                           --roi_idx "$roi_idx" \
                                           --calibrationset "$split_percentage" \
                                           --alpha  "$alpha" \
                                           --task "MUSE"

            python bootstrap.py --gpuid 0 \
                    --file conformal_longitudinal_abniblsa.csv \
                    --roi_idx "$roi_idx" \
                    --calibrationset  "$split_percentage" \
                    --alpha "$alpha" \
                    --task "MUSE"


        done
    done
done
