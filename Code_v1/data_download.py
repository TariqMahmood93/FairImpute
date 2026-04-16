"""
data_download.py
────────────────────────────────────────────────────────────────────────────
Reads the existing clean CSV and produces a fixed train / val / test split.

No downloading — place your files in the dataset folder before running:
    <DATASET_NAME>/
    ├── <raw_file_name>          e.g. adult_census/adult.data
    └── <DATASET_NAME>_clean.csv e.g. adult_census/adult_census_clean.csv

Files produced (once, then reused):
    <DATASET_NAME>/
    ├── <DATASET_NAME>_train.csv  ← 60 % stratified split
    ├── <DATASET_NAME>_val.csv    ← 20 % stratified split  (shared across all MCAR rates)
    └── <DATASET_NAME>_test.csv   ← 20 % stratified split  (shared across all MCAR rates)

The split is stratified on the target column and seeded with
config.RANDOM_SEED for full reproducibility. Null injection (Step 2) will
only ever touch the training split. Val and test are never modified.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import config


def download_raw_data() -> str:
    """
    Read the existing clean CSV and produce fixed train / val / test splits.
    Folder is created automatically if it does not exist.
    Returns the path to the clean CSV.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # ── Check clean CSV exists ────────────────────────────────────────────────
    if not os.path.exists(config.CLEAN_CSV_FILE):
        raise FileNotFoundError(
            f"[data] Clean CSV not found: {config.CLEAN_CSV_FILE}\n"
            f"  Place your clean CSV at that path and re-run."
        )

    size_kb = os.path.getsize(config.CLEAN_CSV_FILE) / 1024
    print(f"[data] ✓ Clean CSV found : {config.CLEAN_CSV_FILE}  ({size_kb:.0f} KB)")

    # ── Check splits already exist ────────────────────────────────────────────
    splits_exist = all(
        os.path.exists(p)
        for p in [config.TRAIN_CSV_FILE, config.VAL_CSV_FILE, config.TEST_CSV_FILE]
    )

    if splits_exist:
        for label, path in [
            ("Train CSV", config.TRAIN_CSV_FILE),
            ("Val   CSV", config.VAL_CSV_FILE),
            ("Test  CSV", config.TEST_CSV_FILE),
        ]:
            size_kb = os.path.getsize(path) / 1024
            print(f"[data] ✓ {label} exists : {path}  ({size_kb:.0f} KB)")
        return config.CLEAN_CSV_FILE

    # ── Read clean CSV and produce splits ─────────────────────────────────────
    df = pd.read_csv(config.CLEAN_CSV_FILE)
    print(f"[data]   Rows loaded     : {len(df):,}  |  cols: {list(df.columns)}")

    dist = df[config.TARGET_COL].value_counts().to_dict()
    print(f"[data]   Target dist     : {dist}")

    _save_splits(df)

    return config.CLEAN_CSV_FILE


def _save_splits(df: pd.DataFrame) -> None:
    """
    Produce a fixed stratified 60 / 20 / 20 split and save each part to CSV.

    Seeded with config.RANDOM_SEED for full reproducibility.
    Val and test are shared across all MCAR rates — never modified.
    """
    seed = config.RANDOM_SEED
    y    = df[config.TARGET_COL].values

    # ── 70 % train, 30 % temp ─────────────────────────────────────────────────
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(df)), y))

    # ── 50 / 50 split of temp → 15 % val, 15 % test ──────────────────────────
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_local, test_local = next(sss2.split(np.zeros(len(temp_idx)), y[temp_idx]))
    val_idx  = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx  ].reset_index(drop=True)
    df_test  = df.iloc[test_idx ].reset_index(drop=True)

    df_train.to_csv(config.TRAIN_CSV_FILE, index=False)
    df_val  .to_csv(config.VAL_CSV_FILE,   index=False)
    df_test .to_csv(config.TEST_CSV_FILE,  index=False)

    print(f"\n[data] ── Stratified split (seed={seed}, 60/20/20):")
    for label, part in [("Train", df_train), ("Val  ", df_val), ("Test ", df_test)]:
        dist = part[config.TARGET_COL].value_counts().to_dict()
        print(f"[data]   {label} : {len(part):,} rows  |  target dist: {dist}")

    print(f"[data] ✓ Saved: {config.TRAIN_CSV_FILE}")
    print(f"[data] ✓ Saved: {config.VAL_CSV_FILE}  ← shared across all MCAR rates")
    print(f"[data] ✓ Saved: {config.TEST_CSV_FILE}  ← shared across all MCAR rates")


if __name__ == "__main__":
    download_raw_data()
    print(f"\nClean CSV : {os.path.abspath(config.CLEAN_CSV_FILE)}")
    print(f"Train CSV : {os.path.abspath(config.TRAIN_CSV_FILE)}")
    print(f"Val   CSV : {os.path.abspath(config.VAL_CSV_FILE)}")
    print(f"Test  CSV : {os.path.abspath(config.TEST_CSV_FILE)}")
