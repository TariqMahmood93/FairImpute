"""
balance_no_balance.py
─────────────────────────────────────────────────────────────────────────────
No balancing — original distribution preserved.

Strategy:
  Keep every row from the clean CSV unchanged.
  Stratified 70 / 15 / 15 split on <sensitive_col> × label so each partition
  mirrors the original group and label proportions.

Output folder: <dataset>_no_balance/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import config

BINARY_COL = "_label_binary"


def _sensitive_col() -> str:
    from fairness import SENSITIVE_COLS
    col = SENSITIVE_COLS.get(config.DATASET_NAME)
    if col is None:
        raise KeyError(
            f"[balance] Dataset '{config.DATASET_NAME}' not found in "
            f"fairness.SENSITIVE_COLS. Add it before running."
        )
    return col


def load_and_split() -> str:
    """Load clean CSV, skip balancing, split 70/15/15. Returns clean CSV path."""
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

    _print_distribution("BEFORE (no balancing applied)", df)
    _save_splits(df)
    _print_distribution("AFTER  (no change — original distribution)", df)

    return config.CLEAN_CSV_FILE


# ─────────────────────────────────────────────────────────────────────────────

def _positive_label():
    return next(k for k, v in config.TARGET_MAP.items() if v == 1)


def _save_splits(df: pd.DataFrame) -> None:
    seed = config.RANDOM_SEED
    sensitive_col = _sensitive_col()

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
        _print_split_inline(split_name, part)

    df_train.drop(columns=drop_cols).to_csv(config.TRAIN_CSV_FILE, index=False)
    df_val  .drop(columns=drop_cols).to_csv(config.VAL_CSV_FILE,   index=False)
    df_test .drop(columns=drop_cols).to_csv(config.TEST_CSV_FILE,  index=False)

    print(f"\n[balance] ✓ Saved: {config.TRAIN_CSV_FILE}")
    print(f"[balance] ✓ Saved: {config.VAL_CSV_FILE}   ← shared across all MCAR rates")
    print(f"[balance] ✓ Saved: {config.TEST_CSV_FILE}   ← shared across all MCAR rates")


def _print_distribution(stage: str, df: pd.DataFrame) -> None:
    sensitive_col = _sensitive_col()
    n      = len(df)
    groups = sorted(df[sensitive_col].dropna().unique())

    print(f"\n[balance] {stage}\n  Total  : {n:,}")
    rates = []
    for g in groups:
        mask_g = df[sensitive_col] == g
        n_g    = int(mask_g.sum())
        g1     = int(((df[BINARY_COL] == 1) & mask_g).sum())
        g0     = int(((df[BINARY_COL] == 0) & mask_g).sum())
        r_g    = g1 / n_g if n_g > 0 else 0.0
        rates.append(r_g)
        print(f"  {str(g):<12}: {n_g:>6,}  (label=1: {g1:,}  label=0: {g0:,})  "
              f"P(1|{g})={r_g:.4f}")
    if len(rates) >= 2:
        gap = abs(rates[0] - rates[1])
        print(f"  DP Gap : {gap:.6f}")


def _print_split_inline(split_name: str, part: pd.DataFrame) -> None:
    sensitive_col = _sensitive_col()
    n      = len(part)
    groups = sorted(part[sensitive_col].dropna().unique())
    group_strs = []
    rate_strs  = []
    rates      = []
    for g in groups:
        mask_g = part[sensitive_col] == g
        n_g    = int(mask_g.sum())
        g1     = int(((part[BINARY_COL] == 1) & mask_g).sum())
        r_g    = g1 / n_g if n_g > 0 else 0.0
        rates.append(r_g)
        group_strs.append(f"{g}: {n_g:,} ({100*n_g/n:.1f}%)")
        rate_strs.append(f"P(1|{g})={r_g:.4f}")
    gap = abs(rates[0] - rates[1]) if len(rates) >= 2 else 0.0
    print(
        f"  {split_name} : {n:,} rows  |  "
        + "  ".join(group_strs)
        + "  |  "
        + "  ".join(rate_strs)
        + f"  DP_gap={gap:.6f}"
    )
