"""
balance_twostage.py
─────────────────────────────────────────────────────────────────────────────
Two-stage male-only downsampling to achieve demographic parity.

Strategy (applied to full dataset before splitting):
  Stage 1  Downsample male label=1 so P(label=1|Male) ≈ P(label=1|Female).
           All male label=0 rows kept; all female rows kept.
  Stage 2  Further downsample male label=1 AND male label=0 so that absolute
           counts for each label exactly equal the female counts:
             male_label1_final = female_label1
             male_label0_final = female_label0

After balancing: |Male| == |Female| AND P(label=1|Male) == P(label=1|Female).
DP gap = 0 before the GNN.

Split: 70 / 15 / 15 stratified on sex × label.
Output folder: <dataset>_twostage/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import config

SENSITIVE_COL  = "sex"
GROUP_MALE     = "Male"
GROUP_FEMALE   = "Female"
BINARY_COL     = "_label_binary"


def load_and_split() -> str:
    """Load clean CSV, apply two-stage balance, split 70/15/15. Returns clean CSV path."""
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

    _print_distribution("BEFORE balancing", df)
    df_balanced = _balance(df)
    _print_distribution("AFTER  balancing", df_balanced)
    _save_splits(df_balanced)

    return config.CLEAN_CSV_FILE


# ─────────────────────────────────────────────────────────────────────────────

def _positive_label():
    return next(k for k, v in config.TARGET_MAP.items() if v == 1)


def _balance(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    df  = df.copy()

    mask_m = df[SENSITIVE_COL] == GROUP_MALE
    mask_f = df[SENSITIVE_COL] == GROUP_FEMALE

    f_idx1 = df.index[mask_f & (df[BINARY_COL] == 1)].tolist()
    f_idx0 = df.index[mask_f & (df[BINARY_COL] == 0)].tolist()
    f1, f0 = len(f_idx1), len(f_idx0)
    female_rate = f1 / (f1 + f0)

    m_idx1_all = df.index[mask_m & (df[BINARY_COL] == 1)].tolist()
    m_idx0_all = df.index[mask_m & (df[BINARY_COL] == 0)].tolist()
    m0 = len(m_idx0_all)

    # Stage 1: rate-match (keep all male label=0, downsample male label=1)
    target_m1_s1 = min(round(female_rate * m0 / (1.0 - female_rate)), len(m_idx1_all))
    m_idx1_s1 = rng.choice(m_idx1_all, size=target_m1_s1, replace=False).tolist()

    print(
        f"\n[balance] Stage 1 — rate match"
        f"\n          Female positive rate : {female_rate:.6f}"
        f"\n          Male label=1  : {len(m_idx1_all):,} → {target_m1_s1:,}"
        f"\n          Male label=0  : {m0:,} (unchanged)"
    )

    # Stage 2: exact count match (male counts == female counts)
    m_idx1_final = rng.choice(m_idx1_s1,  size=f1, replace=False).tolist()
    m_idx0_final = rng.choice(m_idx0_all, size=f0, replace=False).tolist()

    print(
        f"\n[balance] Stage 2 — exact count match"
        f"\n          Male label=1  : {target_m1_s1:,} → {f1:,}  (= Female label=1)"
        f"\n          Male label=0  : {m0:,} → {f0:,}  (= Female label=0)"
    )

    keep_idx   = m_idx1_final + m_idx0_final + f_idx1 + f_idx0
    df_balanced = df.loc[keep_idx].copy()

    # Save full balanced CSV alongside the splits
    balanced_path = os.path.join(config.DATA_DIR, f"{config.DATASET_NAME}_balanced.csv")
    df_balanced.drop(columns=[BINARY_COL]).to_csv(balanced_path, index=False)
    print(f"\n[balance] ✓ Balanced CSV saved : {balanced_path}  ({len(df_balanced):,} rows)")

    return df_balanced


def _save_splits(df: pd.DataFrame) -> None:
    seed = config.RANDOM_SEED

    strat_key = (
        df[SENSITIVE_COL].astype(str) + "_" + df[BINARY_COL].astype(str)
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

    print(f"\n[balance] ── Split (seed={seed}, 70/15/15, stratified on sex×label):")
    for split_name, part in [("Train", df_train), ("Val  ", df_val), ("Test ", df_test)]:
        _print_split_inline(split_name, part)

    df_train.drop(columns=drop_cols).to_csv(config.TRAIN_CSV_FILE, index=False)
    df_val  .drop(columns=drop_cols).to_csv(config.VAL_CSV_FILE,   index=False)
    df_test .drop(columns=drop_cols).to_csv(config.TEST_CSV_FILE,  index=False)

    print(f"\n[balance] ✓ Saved: {config.TRAIN_CSV_FILE}")
    print(f"[balance] ✓ Saved: {config.VAL_CSV_FILE}   ← shared across all MCAR rates")
    print(f"[balance] ✓ Saved: {config.TEST_CSV_FILE}   ← shared across all MCAR rates")


def _print_distribution(stage: str, df: pd.DataFrame) -> None:
    n      = len(df)
    mask_m = df[SENSITIVE_COL] == GROUP_MALE
    mask_f = df[SENSITIVE_COL] == GROUP_FEMALE
    n_m    = int(mask_m.sum())
    n_f    = int(mask_f.sum())
    m1     = int(((df[BINARY_COL] == 1) & mask_m).sum())
    m0     = int(((df[BINARY_COL] == 0) & mask_m).sum())
    f1     = int(((df[BINARY_COL] == 1) & mask_f).sum())
    f0     = int(((df[BINARY_COL] == 0) & mask_f).sum())
    r_m    = m1 / n_m if n_m > 0 else 0.0
    r_f    = f1 / n_f if n_f > 0 else 0.0
    gap    = abs(r_m - r_f)
    print(
        f"\n[balance] {stage}"
        f"\n  Total  : {n:,}"
        f"\n  Male   : {n_m:>6,}  (label=1: {m1:,}  label=0: {m0:,})  "
        f"P(1|Male)={r_m:.4f}"
        f"\n  Female : {n_f:>6,}  (label=1: {f1:,}  label=0: {f0:,})  "
        f"P(1|Female)={r_f:.4f}"
        f"\n  DP Gap : {gap:.6f}"
    )


def _print_split_inline(split_name: str, part: pd.DataFrame) -> None:
    n      = len(part)
    mask_m = part[SENSITIVE_COL] == GROUP_MALE
    mask_f = part[SENSITIVE_COL] == GROUP_FEMALE
    n_m    = int(mask_m.sum())
    n_f    = int(mask_f.sum())
    m1     = int(((part[BINARY_COL] == 1) & mask_m).sum())
    f1     = int(((part[BINARY_COL] == 1) & mask_f).sum())
    r_m    = m1 / n_m if n_m > 0 else 0.0
    r_f    = f1 / n_f if n_f > 0 else 0.0
    gap    = abs(r_m - r_f)
    print(
        f"  {split_name} : {n:,} rows  |  "
        f"Male: {n_m:,} ({100*n_m/n:.1f}%)  Female: {n_f:,} ({100*n_f/n:.1f}%)  |  "
        f"P(1|M)={r_m:.4f}  P(1|F)={r_f:.4f}  DP_gap={gap:.6f}"
    )
