"""
data_download.py
────────────────────────────────────────────────────────────────────────────
Reads the existing clean CSV, applies demographic-parity balancing, and
produces a fixed train / val / test split.

Balancing strategy (male-only downsampling — females are never touched):
  Step A │ Keep ALL female rows (label=1 and label=0 unchanged).
  Step B │ Compute the female positive rate:
          │   female_rate = female_label1 / (female_label1 + female_label0)
  Step C │ Keep ALL male label=0 rows; downsample male label=1 so that
          │   male_label1_kept / (male_label1_kept + male_label0) ≈ female_rate
          │   → solved as: x = female_rate * male_label0 / (1 - female_rate)
  Step D │ Further downsample male label=1 AND male label=0 so that both
          │   counts exactly equal the female counts (equal absolute matching).
          │   male_label1_final = female_label1
          │   male_label0_final = female_label0

After balancing, the dataset is split 70 / 15 / 15 stratified on
sex × label to preserve demographic parity in every partition.

No downloading — place your files in the dataset folder before running:
    <DATASET_NAME>/
    ├── <raw_file_name>          e.g. adult_census/adult.data
    └── <DATASET_NAME>_clean.csv e.g. adult_census/adult_census_clean.csv

Files produced (once, then reused):
    <DATASET_NAME>/
    ├── <DATASET_NAME>_balanced.csv  ← full balanced dataset (before split)
    ├── <DATASET_NAME>_train.csv     ← 70 % stratified split
    ├── <DATASET_NAME>_val.csv       ← 15 % stratified split
    └── <DATASET_NAME>_test.csv      ← 15 % stratified split
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import config

# ── Constants ─────────────────────────────────────────────────────────────────
SENSITIVE_COL   = "sex"
GROUP_MALE      = "Male"
GROUP_FEMALE    = "Female"
POSITIVE_LABEL  = ">50K"          # value in TARGET_COL that counts as label=1
BINARY_COL      = "_label_binary"  # temporary helper column (dropped before save)

# Note: BALANCE_SEED is resolved at CALL time (inside _balance) from
# config.RANDOM_SEED so that multi-seed batch runs see the current seed.


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def download_raw_data() -> str:
    """
    Read the existing clean CSV, balance it for demographic parity, and
    produce fixed train / val / test splits.  Folder is created automatically
    if it does not exist.  Returns the path to the clean CSV.
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
            ("Train CSV     ", config.TRAIN_CSV_FILE),
            ("Val   CSV     ", config.VAL_CSV_FILE),
            ("Test  CSV     ", config.TEST_CSV_FILE),
        ]:
            size_kb = os.path.getsize(path) / 1024
            print(f"[data] ✓ {label} exists : {path}  ({size_kb:.0f} KB)")
        return config.CLEAN_CSV_FILE

    # ── Read clean CSV ────────────────────────────────────────────────────────
    df = pd.read_csv(config.CLEAN_CSV_FILE)
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    _print_dataset_overview(df, "Original dataset (with balancing)")

    # ── Balance → split → save ────────────────────────────────────────────────
    df_balanced = _balance(df)
    _save_splits(df_balanced)

    return config.CLEAN_CSV_FILE


# ══════════════════════════════════════════════════════════════════════════════
# Balancing
# ══════════════════════════════════════════════════════════════════════════════

def _balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply two-stage male-only downsampling to achieve demographic parity.

    Stage 1  Downsample male label=1 so P(label=1|Male) ≈ P(label=1|Female).
             All male label=0 rows are kept; all female rows are kept.

    Stage 2  Further downsample male label=1 and male label=0 so that the
             absolute counts for each label exactly equal the female counts.
             This makes P(label=1|Male) == P(label=1|Female) exactly.

    Returns a new DataFrame with the binary helper column dropped.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)

    df = df.copy()
    df[BINARY_COL] = (df[config.TARGET_COL] == POSITIVE_LABEL).astype(int)

    mask_m = df[SENSITIVE_COL] == GROUP_MALE
    mask_f = df[SENSITIVE_COL] == GROUP_FEMALE

    # ── Female counts (never changed) ─────────────────────────────────────────
    f_idx1 = df.index[mask_f & (df[BINARY_COL] == 1)].tolist()
    f_idx0 = df.index[mask_f & (df[BINARY_COL] == 0)].tolist()
    f1, f0 = len(f_idx1), len(f_idx0)
    female_rate = f1 / (f1 + f0)

    _print_counts("BEFORE balancing", df, mask_m, mask_f)

    # ── Stage 1: rate-matching (keep all male label=0, downsample male label=1) ──
    m_idx1_all = df.index[mask_m & (df[BINARY_COL] == 1)].tolist()
    m_idx0_all = df.index[mask_m & (df[BINARY_COL] == 0)].tolist()
    m0 = len(m_idx0_all)

    target_m1_stage1 = round(female_rate * m0 / (1.0 - female_rate))
    target_m1_stage1 = min(target_m1_stage1, len(m_idx1_all))

    m_idx1_s1 = rng.choice(m_idx1_all, size=target_m1_stage1, replace=False).tolist()

    print(
        f"\n[balance] Stage 1 — rate match"
        f"\n          Female positive rate : {female_rate:.6f}"
        f"\n          Male label=1  : {len(m_idx1_all):,} → {target_m1_stage1:,}"
        f"\n          Male label=0  : {m0:,} (unchanged)"
    )

    # ── Stage 2: exact count match (male counts == female counts) ─────────────
    # Downsample from stage-1 result to exactly f1 / f0
    m_idx1_final = rng.choice(m_idx1_s1,  size=f1, replace=False).tolist()
    m_idx0_final = rng.choice(m_idx0_all, size=f0, replace=False).tolist()

    print(
        f"\n[balance] Stage 2 — exact count match"
        f"\n          Male label=1  : {target_m1_stage1:,} → {f1:,}  (= Female label=1)"
        f"\n          Male label=0  : {m0:,} → {f0:,}  (= Female label=0)"
    )

    # ── Assemble balanced dataset ──────────────────────────────────────────────
    keep_idx   = m_idx1_final + m_idx0_final + f_idx1 + f_idx0
    df_balanced = df.loc[keep_idx].copy()

    _print_counts("AFTER  balancing", df_balanced,
                  df_balanced[SENSITIVE_COL] == GROUP_MALE,
                  df_balanced[SENSITIVE_COL] == GROUP_FEMALE)

    # Save balanced CSV alongside the splits
    balanced_path = os.path.join(
        config.DATA_DIR,
        f"{config.DATASET_NAME}_balanced.csv",
    )
    df_balanced.drop(columns=[BINARY_COL]).to_csv(balanced_path, index=False)
    print(f"\n[balance] ✓ Balanced CSV saved : {balanced_path}  ({len(df_balanced):,} rows)")

    return df_balanced


# ══════════════════════════════════════════════════════════════════════════════
# Splitting
# ══════════════════════════════════════════════════════════════════════════════

def _save_splits(df: pd.DataFrame) -> None:
    """
    Produce a fixed stratified 70 / 15 / 15 split from the balanced DataFrame
    and save each part to CSV.

    Stratification key: sex × label  (4 strata: Male_0, Male_1, Female_0, Female_1)
    This preserves demographic parity in every partition.
    """
    seed = config.RANDOM_SEED

    # Stratification key: sex + binary label combined
    strat_key = (
        df[SENSITIVE_COL].astype(str) + "_" + df[BINARY_COL].astype(str)
    ).values

    # ── 70 % train, 30 % temp ─────────────────────────────────────────────────
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(df)), strat_key))

    # ── 50 / 50 split of temp → 15 % val, 15 % test ──────────────────────────
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed
    )
    val_local, test_local = next(
        sss2.split(np.zeros(len(temp_idx)), strat_key[temp_idx])
    )
    val_idx  = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx  ].reset_index(drop=True)
    df_test  = df.iloc[test_idx ].reset_index(drop=True)

    # Drop the helper binary column before saving
    drop_cols = [BINARY_COL] if BINARY_COL in df.columns else []

    df_train.drop(columns=drop_cols).to_csv(config.TRAIN_CSV_FILE, index=False)
    df_val  .drop(columns=drop_cols).to_csv(config.VAL_CSV_FILE,   index=False)
    df_test .drop(columns=drop_cols).to_csv(config.TEST_CSV_FILE,  index=False)

    print(f"\n[data] ── Stratified split (seed={seed}, 70/15/15, stratified on sex×label):")
    for label, part in [("Train", df_train), ("Val  ", df_val), ("Test ", df_test)]:
        _print_dp_inline(label, part)

    print(f"\n[data] ✓ Saved: {config.TRAIN_CSV_FILE}")
    print(f"[data] ✓ Saved: {config.VAL_CSV_FILE}   ← shared across all MCAR rates")
    print(f"[data] ✓ Saved: {config.TEST_CSV_FILE}   ← shared across all MCAR rates")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_counts(
    stage: str,
    df: pd.DataFrame,
    mask_m: pd.Series,
    mask_f: pd.Series,
) -> None:
    """Print label counts and positive rates for both groups."""
    m1 = int(((df[BINARY_COL] == 1) & mask_m).sum())
    m0 = int(((df[BINARY_COL] == 0) & mask_m).sum())
    f1 = int(((df[BINARY_COL] == 1) & mask_f).sum())
    f0 = int(((df[BINARY_COL] == 0) & mask_f).sum())
    n_m = m1 + m0
    n_f = f1 + f0
    r_m = m1 / n_m if n_m > 0 else 0.0
    r_f = f1 / n_f if n_f > 0 else 0.0
    gap = abs(r_m - r_f)

    print(
        f"\n[balance] {stage}"
        f"\n          Male   : {n_m:>6,} total  (label=1: {m1:,}  label=0: {m0:,})  P(1|Male)={r_m:.6f}"
        f"\n          Female : {n_f:>6,} total  (label=1: {f1:,}  label=0: {f0:,})  P(1|Female)={r_f:.6f}"
        f"\n          DP Gap : {gap:.6f}  |  Total rows: {len(df):,}"
    )


def _print_dataset_overview(df: pd.DataFrame, label: str = "Dataset overview") -> None:
    """Print total rows, columns, % male/female, target distribution, and
    (when BINARY_COL is present) the full 4-way group-label breakdown."""
    n     = len(df)
    ncols = len([c for c in df.columns if c != BINARY_COL])
    n_m   = int((df[SENSITIVE_COL] == GROUP_MALE).sum())
    n_f   = int((df[SENSITIVE_COL] == GROUP_FEMALE).sum())
    pct_m = 100.0 * n_m / n if n > 0 else 0.0
    pct_f = 100.0 * n_f / n if n > 0 else 0.0
    dist  = df[config.TARGET_COL].value_counts().to_dict()
    print(
        f"\n[data] ── {label}"
        f"\n[data]   Total rows    : {n:,}"
        f"\n[data]   Total columns : {ncols}"
        f"\n[data]   Male          : {n_m:,}  ({pct_m:.1f}%)"
        f"\n[data]   Female        : {n_f:,}  ({pct_f:.1f}%)"
        f"\n[data]   Target dist   : {dist}"
    )
    if BINARY_COL in df.columns:
        _print_group_label_breakdown(df, n, n_m, n_f)


def _print_group_label_breakdown(
    df: pd.DataFrame, n: int, n_m: int, n_f: int
) -> None:
    """Print the 4-way (sex × label) counts, % of total rows, and within-group rates."""
    mask_m = df[SENSITIVE_COL] == GROUP_MALE
    mask_f = df[SENSITIVE_COL] == GROUP_FEMALE
    m1 = int(((df[BINARY_COL] == 1) & mask_m).sum())
    m0 = int(((df[BINARY_COL] == 0) & mask_m).sum())
    f1 = int(((df[BINARY_COL] == 1) & mask_f).sum())
    f0 = int(((df[BINARY_COL] == 0) & mask_f).sum())
    _p  = lambda x: 100.0 * x / n   if n   > 0 else 0.0
    _rm = lambda x: x / n_m         if n_m > 0 else 0.0
    _rf = lambda x: x / n_f         if n_f > 0 else 0.0
    print(
        f"[data]   Group-label breakdown (% of total rows):"
        f"\n[data]     Male   & >50K  : {m1:>6,}  ({_p(m1):5.1f}%)"
        f"  |  P(>50K  | Male)   = {_rm(m1):.4f}"
        f"\n[data]     Male   & <=50K : {m0:>6,}  ({_p(m0):5.1f}%)"
        f"  |  P(<=50K | Male)   = {_rm(m0):.4f}"
        f"\n[data]     Female & >50K  : {f1:>6,}  ({_p(f1):5.1f}%)"
        f"  |  P(>50K  | Female) = {_rf(f1):.4f}"
        f"\n[data]     Female & <=50K : {f0:>6,}  ({_p(f0):5.1f}%)"
        f"  |  P(<=50K | Female) = {_rf(f0):.4f}"
    )


def _print_dp_inline(split_name: str, part: pd.DataFrame) -> None:
    """Print per-split row count, sex %, full 4-group breakdown, and DP gap."""
    n      = len(part)
    mask_m = part[SENSITIVE_COL] == GROUP_MALE
    mask_f = part[SENSITIVE_COL] == GROUP_FEMALE
    n_m    = int(mask_m.sum())
    n_f    = int(mask_f.sum())
    m1     = int(((part[BINARY_COL] == 1) & mask_m).sum())
    m0     = int(((part[BINARY_COL] == 0) & mask_m).sum())
    f1     = int(((part[BINARY_COL] == 1) & mask_f).sum())
    f0     = int(((part[BINARY_COL] == 0) & mask_f).sum())
    r_m    = m1 / n_m if n_m > 0 else 0.0
    r_f    = f1 / n_f if n_f > 0 else 0.0
    gap    = abs(r_m - r_f)
    pct_m  = 100.0 * n_m / n if n > 0 else 0.0
    pct_f  = 100.0 * n_f / n if n > 0 else 0.0
    dist   = part[config.TARGET_COL].value_counts().to_dict()
    _p     = lambda x: 100.0 * x / n if n > 0 else 0.0
    print(
        f"\n[data]   {split_name} : {n:,} rows"
        f"  |  Male: {n_m:,} ({pct_m:.1f}%)  Female: {n_f:,} ({pct_f:.1f}%)"
        f"  |  target dist: {dist}"
        f"\n[data]     Male   & >50K  : {m1:>6,}  ({_p(m1):5.1f}%)"
        f"  |  P(>50K  | Male)   = {r_m:.4f}"
        f"\n[data]     Male   & <=50K : {m0:>6,}  ({_p(m0):5.1f}%)"
        f"  |  P(<=50K | Male)   = {1.0 - r_m:.4f}"
        f"\n[data]     Female & >50K  : {f1:>6,}  ({_p(f1):5.1f}%)"
        f"  |  P(>50K  | Female) = {r_f:.4f}"
        f"\n[data]     Female & <=50K : {f0:>6,}  ({_p(f0):5.1f}%)"
        f"  |  P(<=50K | Female) = {1.0 - r_f:.4f}"
        f"\n[data]     DP gap (P(>50K|Male) - P(>50K|Female)) = {gap:.6f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# No-balance alternative
# ══════════════════════════════════════════════════════════════════════════════

def download_raw_data_no_balance() -> str:
    """
    Read the existing clean CSV and produce stratified 70/15/15 splits
    WITHOUT any demographic balancing.

    The original sex and label distribution is preserved proportionally in
    every partition — if the raw data is 80 % Male / 20 % Female, each split
    will also be ~80 % / ~20 %.

    Stratification key: sex × label  (4 strata: Male_0, Male_1, Female_0, Female_1)

    Saves splits to the same paths as download_raw_data() so all downstream
    cells work unchanged.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if not os.path.exists(config.CLEAN_CSV_FILE):
        raise FileNotFoundError(
            f"[data] Clean CSV not found: {config.CLEAN_CSV_FILE}\n"
            f"  Place your clean CSV at that path and re-run."
        )

    size_kb = os.path.getsize(config.CLEAN_CSV_FILE) / 1024
    print(f"[data] ✓ Clean CSV found : {config.CLEAN_CSV_FILE}  ({size_kb:.0f} KB)")

    splits_exist = all(
        os.path.exists(p)
        for p in [config.TRAIN_CSV_FILE, config.VAL_CSV_FILE, config.TEST_CSV_FILE]
    )
    if splits_exist:
        for label, path in [
            ("Train CSV     ", config.TRAIN_CSV_FILE),
            ("Val   CSV     ", config.VAL_CSV_FILE),
            ("Test  CSV     ", config.TEST_CSV_FILE),
        ]:
            size_kb = os.path.getsize(path) / 1024
            print(f"[data] ✓ {label} exists : {path}  ({size_kb:.0f} KB)")
        return config.CLEAN_CSV_FILE

    df = pd.read_csv(config.CLEAN_CSV_FILE)
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    df[BINARY_COL] = (df[config.TARGET_COL] == POSITIVE_LABEL).astype(int)

    _print_dataset_overview(df, "Original dataset (NO balancing applied)")

    _save_splits_no_balance(df)
    return config.CLEAN_CSV_FILE


def _save_splits_no_balance(df: pd.DataFrame) -> None:
    """
    Stratified 70/15/15 split that mirrors the original sex × label
    distribution into every partition — no rows added or removed.
    """
    seed = config.RANDOM_SEED

    strat_key = np.asarray(
        df[SENSITIVE_COL].astype(str) + "_" + df[BINARY_COL].astype(str)
    )

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

    print(
        f"\n[data] ── Stratified split (seed={seed}, 70/15/15, "
        f"original distribution preserved):"
    )
    for split_name, part in [("Train", df_train), ("Val  ", df_val), ("Test ", df_test)]:
        _print_dp_inline(split_name, part)

    df_train.drop(columns=drop_cols).to_csv(config.TRAIN_CSV_FILE, index=False)
    df_val  .drop(columns=drop_cols).to_csv(config.VAL_CSV_FILE,   index=False)
    df_test .drop(columns=drop_cols).to_csv(config.TEST_CSV_FILE,  index=False)

    print(f"\n[data] ✓ Saved: {config.TRAIN_CSV_FILE}")
    print(f"[data] ✓ Saved: {config.VAL_CSV_FILE}   <- shared across all MCAR rates")
    print(f"[data] ✓ Saved: {config.TEST_CSV_FILE}   <- shared across all MCAR rates")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    download_raw_data()
    print(f"\nClean CSV    : {os.path.abspath(config.CLEAN_CSV_FILE)}")
    print(f"Balanced CSV : {os.path.abspath(os.path.join(config.DATA_DIR, f'{config.DATASET_NAME}_balanced.csv'))}")
    print(f"Train CSV    : {os.path.abspath(config.TRAIN_CSV_FILE)}")
    print(f"Val   CSV    : {os.path.abspath(config.VAL_CSV_FILE)}")
    print(f"Test  CSV    : {os.path.abspath(config.TEST_CSV_FILE)}")
