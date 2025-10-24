#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GroupKFold

# ---------- Configuration ----------
ANONYMIZED_DATA_PATH = "./data/data.csv"
OUTPUT_DIR = "./data/folds"
SEED = 42
N_SPLITS = 10
# -----------------------------------

def main():
    # --------------------------
    # 1. Load dataset
    # --------------------------
    df = pd.read_csv(ANONYMIZED_DATA_PATH)
    if "anon_id" not in df.columns:
        raise ValueError("Column 'anon_id' not found in dataset.")

    # Get unique subject IDs
    unique_subjects = np.sort(df["anon_id"].unique())

    # --------------------------
    # 2. Create GroupKFold splits
    # --------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dummy labels (required by GroupKFold)
    labels = np.zeros(len(unique_subjects))

    gkf = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(unique_subjects, labels, groups=unique_subjects)
    ):
        train_subjects = unique_subjects[train_idx].tolist()
        test_subjects = unique_subjects[test_idx].tolist()

        with open(os.path.join(OUTPUT_DIR, f"fold_{fold}_train.pkl"), "wb") as f:
            pickle.dump(train_subjects, f)
        with open(os.path.join(OUTPUT_DIR, f"fold_{fold}_test.pkl"), "wb") as f:
            pickle.dump(test_subjects, f)

        print(f"✅ Fold {fold}: saved {len(train_subjects)} train IDs and {len(test_subjects)} test IDs")

if __name__ == "__main__":
    main()
