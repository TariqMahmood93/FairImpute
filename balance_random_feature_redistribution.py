"""
balance_random_feature_redistribution.py
─────────────────────────────────────────────────────────────────────────────
Random Feature Redistribution — group-size equalization by sensitive-attribute
relabelling (no rows removed).

Strategy (applied to full dataset before splitting):
  1. Identify majority and minority groups by row count.
  2. Compute the number of samples to convert:
       NC = majority_size - (total_size / number_of_groups)
  3. Randomly select NC samples from the majority group.
  4. Reassign only their sensitive attribute value to the minority group label.
  5. No other feature values are modified; no rows are added or removed.
  6. Shuffle the resulting dataset.

After balancing: group sizes are approximately equal (differ by at most 1 due
to integer rounding). Positive rates within groups change because transferred
rows carry their original labels.

Before/after class distribution is reported.

Split: 70 / 15 / 15 stratified on sensitive_col × label.
Output folder: <dataset>_random_feature_redistribution/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import config

SENSITIVE_COLS = {
    "adult_census":  "sex",
    "german_credit": "personal_status",
}
BINARY_COL = "_label_binary"


def load_and_split() -> str:
    """Load clean CSV, apply random feature redistribution, split 70/15/15."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not os.path.exists(config.CLEAN_CSV_FILE):
        raise FileNotFoundError(
            f"[balance] Clean CSV not found: {config.CLEAN_CSV_FILE}\n"
            f"  Place your clean CSV at that path and re-run."
        )

    size_kb = os.path.getsize(config.CLEAN_CSV_FILE) / 1024
    print(f"[balance] ✓ Clean CSV found : {config.CLEAN_CSV_FILE}  ({size_kb:.0f} KB)")

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
            print(f"[balance] ✓ {label} exists : {path}  ({size_kb:.0f} KB)")
        return config.CLEAN_CSV_FILE

    df = pd.read_csv(config.CLEAN_CSV_FILE)
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    df[BINARY_COL] = (df[config.TARGET_COL] == _positive_label()).astype(int)

    sensitive_col = _get_sensitive_col()
    _print_distribution("BEFORE balancing", df, sensitive_col)

    df_balanced = _balance(df, sensitive_col)

    _print_distribution("AFTER  balancing", df_balanced, sensitive_col)
    _save_splits(df_balanced, sensitive_col)

    return config.CLEAN_CSV_FILE


# ─────────────────────────────────────────────────────────────────────────────

def _get_sensitive_col() -> str:
    col = SENSITIVE_COLS.get(config.DATASET_NAME)
    if col is None:
        raise ValueError(
            f"[balance] No sensitive column configured for '{config.DATASET_NAME}'.\n"
            f"  Add an entry to SENSITIVE_COLS in balance_random_feature_redistribution.py."
        )
    return col


def _positive_label():
    return next(k for k, v in config.TARGET_MAP.items() if v == 1)


def _balance(df: pd.DataFrame, sensitive_col: str) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)

    group_indices = {
        name: grp.index.tolist()
        for name, grp in df.groupby(sensitive_col)
    }
    n_groups   = len(group_indices)
    total      = len(df)
    target_size = total / n_groups

    majority_name = max(group_indices, key=lambda k: len(group_indices[k]))
    minority_name = min(group_indices, key=lambda k: len(group_indices[k]))

    majority_size = len(group_indices[majority_name])
    minority_size = len(group_indices[minority_name])
    NC = round(majority_size - target_size)

    print(f"\n[balance] Random Feature Redistribution by '{sensitive_col}'")
    print(f"  Total rows      : {total:,}  |  groups: {n_groups}  |  target per group: {target_size:.1f}")
    print(f"  Majority group  : '{majority_name}' ({majority_size:,} rows)")
    print(f"  Minority group  : '{minority_name}' ({minority_size:,} rows)")
    print(f"  NC (to convert) : {NC:,}")
    print(f"  Action          : reassign {NC:,} rows from '{majority_name}' → '{minority_name}'")

    convert_idx = rng.choice(group_indices[majority_name], size=NC, replace=False)

    df_balanced = df.copy()
    df_balanced.loc[convert_idx, sensitive_col] = minority_name

    new_majority = int((df_balanced[sensitive_col] == majority_name).sum())
    new_minority = int((df_balanced[sensitive_col] == minority_name).sum())
    print(f"  Post-conversion : '{majority_name}' = {new_majority:,}  |  '{minority_name}' = {new_minority:,}")

    df_balanced = (
        df_balanced
        .sample(frac=1, random_state=int(config.RANDOM_SEED))
        .reset_index(drop=True)
    )
    return df_balanced


def _save_splits(df: pd.DataFrame, sensitive_col: str) -> None:
    seed = config.RANDOM_SEED

    strat_key = (
        df[sensitive_col].astype(str) + "_" + df[BINARY_COL].astype(str)
    ).values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(df)), strat_key))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_local, test_local = next(
        sss2.split(np.zeros(len(temp_idx)), strat_key[temp_idx])
    )
    val_idx  = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx  ].reset_index(drop=True)
    df_test  = df.iloc[test_idx ].reset_index(drop=True)

    drop_cols = [BINARY_COL] if BINARY_COL in df.columns else []

    print(f"\n[balance] ── Split (seed={seed}, 70/15/15, stratified on {sensitive_col}×label):")
    for split_name, part in [("Train", df_train), ("Val  ", df_val), ("Test ", df_test)]:
        _print_split_inline(split_name, part, sensitive_col)

    df_train.drop(columns=drop_cols).to_csv(config.TRAIN_CSV_FILE, index=False)
    df_val  .drop(columns=drop_cols).to_csv(config.VAL_CSV_FILE,   index=False)
    df_test .drop(columns=drop_cols).to_csv(config.TEST_CSV_FILE,  index=False)

    print(f"\n[balance] ✓ Saved: {config.TRAIN_CSV_FILE}")
    print(f"[balance] ✓ Saved: {config.VAL_CSV_FILE}   ← shared across all MCAR rates")
    print(f"[balance] ✓ Saved: {config.TEST_CSV_FILE}   ← shared across all MCAR rates")


def _print_distribution(stage: str, df: pd.DataFrame, sensitive_col: str) -> None:
    n = len(df)
    groups = df.groupby(sensitive_col)

    print(f"\n[balance] {stage}  (total: {n:,} rows)")
    for name, grp in sorted(groups, key=lambda x: len(x[1])):
        n_g  = len(grp)
        g1   = int((grp[BINARY_COL] == 1).sum())
        g0   = int((grp[BINARY_COL] == 0).sum())
        rate = g1 / n_g if n_g > 0 else 0.0
        print(
            f"  {name:<15}: {n_g:>6,} rows  "
            f"(label=1: {g1:,}  label=0: {g0:,})  P(1|group)={rate:.4f}"
        )

    group_names = [name for name, _ in sorted(groups, key=lambda x: len(x[1]))]
    if len(group_names) == 2:
        rates = {
            name: (int((grp[BINARY_COL] == 1).sum()) / len(grp) if len(grp) > 0 else 0.0)
            for name, grp in groups
        }
        gap = abs(list(rates.values())[0] - list(rates.values())[1])
        print(f"  DP Gap : {gap:.6f}")


def _print_split_inline(split_name: str, part: pd.DataFrame, sensitive_col: str) -> None:
    n      = len(part)
    groups = {name: grp for name, grp in part.groupby(sensitive_col)}
    parts  = []
    for name, grp in sorted(groups.items(), key=lambda x: len(x[1])):
        n_g  = len(grp)
        g1   = int((grp[BINARY_COL] == 1).sum())
        rate = g1 / n_g if n_g > 0 else 0.0
        parts.append(f"{name}: {n_g:,} ({100*n_g/n:.1f}%)  P(1)={rate:.4f}")
    print(f"  {split_name} : {n:,} rows  |  " + "  |  ".join(parts))
