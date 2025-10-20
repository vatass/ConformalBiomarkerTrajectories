#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GroupKFold

# ---------- Configuration ----------
INPUT_PATH = "./data/conformal_longitudinal_adniblsa_data.csv"
COVARIATE_PATH = "./data/longitudinal_covariates_adniblsa_conformal.csv"
ANONYMIZED_DATA_PATH = "./data/data.csv"
ANONYMIZED_COVARIATES_PATH = "./data/anonymized_covariates.csv"
ID_MAPPING_PATH = "./data/ptid_to_anonid_map.pkl"
OUTPUT_DIR = "./data/folds"
ID_COLUMN = "PTID"
SEED = 42
N_SPLITS = 10
# -----------------------------------

def main():
    # --------------------------
    # 1. Anonymize main dataset
    # --------------------------
    df = pd.read_csv(INPUT_PATH)
    if ID_COLUMN not in df.columns:
        raise ValueError(f"'{ID_COLUMN}' column not found in main data.")

    # Create mapping
    unique_ptids = pd.Series(df[ID_COLUMN].unique()).sort_values().reset_index(drop=True)
    ptid_to_anon = {ptid: f"S{idx:05d}" for idx, ptid in enumerate(unique_ptids)}
    df["anon_id"] = df[ID_COLUMN].map(ptid_to_anon)

    # Save anonymized main data
    df_anonymized = df.drop(columns=[ID_COLUMN])
    df_anonymized.to_csv(ANONYMIZED_DATA_PATH, index=False)
    print(f"✅ Saved anonymized main data to: {ANONYMIZED_DATA_PATH}")

    # Save PTID → anon_id mapping
    with open(ID_MAPPING_PATH, "wb") as f:
        pickle.dump(ptid_to_anon, f)
    print(f"✅ Saved ID mapping to: {ID_MAPPING_PATH}")

    # --------------------------
    # 2. Create GroupKFold splits
    # --------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subject_df = pd.DataFrame({
        "ptid": unique_ptids,
        "anon_id": unique_ptids.map(ptid_to_anon),
        "label": 0
    })

    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(subject_df, subject_df["label"], groups=subject_df["ptid"])
    ):
        train_ids = subject_df.iloc[train_idx]["anon_id"].tolist()
        test_ids = subject_df.iloc[test_idx]["anon_id"].tolist()

        with open(os.path.join(OUTPUT_DIR, f"fold_{fold}_train.pkl"), "wb") as f:
            pickle.dump(train_ids, f)

        with open(os.path.join(OUTPUT_DIR, f"fold_{fold}_test.pkl"), "wb") as f:
            pickle.dump(test_ids, f)

        print(f"✅ Fold {fold}: saved {len(train_ids)} train IDs and {len(test_ids)} test IDs")

    # --------------------------
    # 3. Preprocess & anonymize covariates
    # --------------------------
    cov_df = pd.read_csv(COVARIATE_PATH)
    if "Time" not in cov_df.columns or "Diagnosis" not in cov_df.columns:
        raise ValueError("Covariates file must contain 'Time' and 'Diag' columns.")

    # Filter for Time = 0 and keep only Age
    baseline_df = cov_df[cov_df["Time"] == 0][[ID_COLUMN, "Diagnosis"]].copy()
    baseline_df[ID_COLUMN] = baseline_df[ID_COLUMN].astype(str)

    # Apply mapping
    baseline_df["anon_id"] = baseline_df[ID_COLUMN].map(ptid_to_anon)

    # Only keep anon_ids that exist in data.csv
    valid_anon_ids = set(df_anonymized["anon_id"].unique())
    baseline_df = baseline_df[baseline_df["anon_id"].isin(valid_anon_ids)]

    # Final columns: anon_id, Age
    final_cov_df = baseline_df[["anon_id", "Diagnosis"]]
    final_cov_df.to_csv(ANONYMIZED_COVARIATES_PATH, index=False)
    print(f"✅ Saved anonymized covariates to: {ANONYMIZED_COVARIATES_PATH}")

if __name__ == "__main__":
    main()
