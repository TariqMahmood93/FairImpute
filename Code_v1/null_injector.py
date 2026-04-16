"""
null_injector.py
────────────────────────────────────────────────────────────────────────────
Incremental MCAR null injection — TRAINING SPLIT ONLY.

Val and test splits are written through untouched so that evaluation always
happens on clean, original data (no data leakage from artificial missingness).

KEY DESIGN DECISIONS:
    1. Both categorical and numerical columns are injected (training only).
    2. Injection is CUMULATIVE — 5% nulls are a subset of 10%, which are
       a subset of 20%. Positions are tracked so no double-injection occurs.
    3. Rows that already have a null in another column are deprioritised
       to spread nulls across rows as much as possible.
    4. The target column is NEVER touched.
    5. A single seed controls all randomness for reproducibility.
    6. Val / test CSVs are copied as-is to each mcar output folder.

INCREMENTAL LOGIC:
    rates = [5, 10, 20]
    Step 1: inject 5% nulls per feature column in train  → save mcar_5
    Step 2: inject 5% MORE (keeping previous)            → total 10% → save mcar_10
    Step 3: inject 10% MORE (keeping previous)           → total 20% → save mcar_20

Files saved per rate:
    <dataset>_mcar_<rate>pct/
    ├── <dataset>_mcar_<rate>_train.csv    ← NaN cells (TRAINING ONLY)
    └── <dataset>_mcar_<rate>_train.data   ← "?" replacing NaN  (UCI format)

Val and test are NOT copied into mcar folders — they already exist in
the base dataset folder and are shared across all MCAR rates:
    <dataset>/
    ├── <dataset>_val.csv   ← clean, never modified
    └── <dataset>_test.csv  ← clean, never modified

Usage:
    from null_injector import run_null_injection
    run_null_injection(clean_train_df)
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

import config


def run_null_injection(clean_train_df: pd.DataFrame,
                       run_impute: bool = False,
                       strategies: list = None) -> None:
    """
    Inject MCAR nulls at each rate in config.MISSING_RATE into the
    TRAINING split only. Val and test CSVs are passed through unchanged.

    Parameters
    ----------
    clean_train_df : clean training DataFrame (no nulls)
    run_impute     : if True, run imputation pipeline after injection
    strategies     : imputation strategies to run (default: all non-none)
    """
    rates = config.MISSING_RATE
    _validate(rates)

    seed = config.RANDOM_SEED

    # Identify injectable columns (exclude target)
    cat_cols = [
        c for c in clean_train_df.columns
        if c != config.TARGET_COL
        and (clean_train_df[c].dtype == object
             or pd.api.types.is_categorical_dtype(clean_train_df[c]))
    ]
    num_cols = [
        c for c in clean_train_df.columns
        if c != config.TARGET_COL and c in config.NUMERICAL_COLS
    ]
    inject_cols = cat_cols + num_cols

    if not inject_cols:
        print("[null_injector] ⚠ No feature columns found — nothing to inject")
        return

    # ── Verify val / test splits exist in base folder ────────────────────────
    for label, path in [("Val CSV",  config.VAL_CSV_FILE),
                        ("Test CSV", config.TEST_CSV_FILE)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[null_injector] {label} not found: {path}\n"
                f"  Run python data_download.py first."
            )

    print("\n" + "=" * 60)
    print("  NULL INJECTION  (MCAR — TRAINING SET ONLY, CUMULATIVE)")
    print("=" * 60)
    df_val_info  = pd.read_csv(config.VAL_CSV_FILE)
    df_test_info = pd.read_csv(config.TEST_CSV_FILE)
    print(f"  Training rows     : {len(clean_train_df):,}")
    print(f"  Val rows          : {len(df_val_info):,}  ← base folder, untouched")
    print(f"  Test rows         : {len(df_test_info):,}  ← base folder, untouched")
    print(f"  Categorical cols  : {len(cat_cols)}  →  {cat_cols}")
    print(f"  Numerical cols    : {len(num_cols)}  →  {num_cols}")
    print(f"  Total inject cols : {len(inject_cols)}")
    print(f"  Rates (cumul.)    : {rates}%")
    print(f"  Random seed       : {seed}")
    print(f"  Target column     : '{config.TARGET_COL}'  ← never touched")
    if run_impute:
        if strategies is None:
            strategies = [s for s in config.IMPUTATION_STRATEGIES if s != "none"]
        print(f"  Imputation        : ENABLED  →  {strategies}")
    else:
        print(f"  Imputation        : DISABLED")
    print()

    # Track nullified positions across rates (cumulative)
    prev_nulls  = {c: set() for c in inject_cols}
    df_current  = clean_train_df.copy()

    for rate in rates:
        step_num       = rates.index(rate) + 1
        target_per_col = int(round(len(df_current) * rate / 100))

        print(f"  ── Step {step_num}: target {rate}% "
              f"({target_per_col} nulls per column in training set) ──")

        df_current, prev_nulls = _inject_to_target(
            df_current, inject_cols, target_per_col, seed, prev_nulls
        )

        # Verify
        actual_nulls = {c: df_current[c].isnull().sum() for c in inject_cols}
        total_null   = sum(actual_nulls.values())
        total_cells  = len(df_current) * len(inject_cols)
        actual_pct   = total_null / total_cells * 100 if total_cells > 0 else 0
        target_nulls = df_current[config.TARGET_COL].isnull().sum()

        null_counts_per_row = df_current[inject_cols].isnull().sum(axis=1)
        rows_with_multi     = (null_counts_per_row > 1).sum()

        print(f"    Nulls (categorical): {dict((c, actual_nulls[c]) for c in cat_cols)}")
        print(f"    Nulls (numerical)  : {dict((c, actual_nulls[c]) for c in num_cols)}")
        print(f"    Total null cells   : {total_null:,} / {total_cells:,} ({actual_pct:.2f}%)")
        print(f"    Target col nulls   : {target_nulls}  (must be 0)")
        print(f"    Rows with >1 null  : {rows_with_multi:,} / {len(df_current):,}")
        print(f"    Val  / test        : unchanged — shared from {config.VAL_CSV_FILE}")

        _save_outputs(df_current, rate)

        # ── Post-save verification ────────────────────────────────────────────
        feat_cols    = [c for c in df_current.columns if c != config.TARGET_COL]
        saved_train  = pd.read_csv(config.mcar_train_csv(rate))
        saved_val    = pd.read_csv(config.VAL_CSV_FILE)
        saved_test   = pd.read_csv(config.TEST_CSV_FILE)
        n_train_null = saved_train[feat_cols].isnull().sum().sum()
        n_val_null   = saved_val  [feat_cols].isnull().sum().sum()
        n_test_null  = saved_test [feat_cols].isnull().sum().sum()
        ok_val  = "✓" if n_val_null  == 0 else "✗ BUG"
        ok_test = "✓" if n_test_null == 0 else "✗ BUG"
        print(f"    ── Null verification after save ──────────────────────────")
        print(f"    train nulls : {n_train_null:,}  ✓")
        print(f"    val   nulls : {n_val_null}  {ok_val}  (base folder)")
        print(f"    test  nulls : {n_test_null}  {ok_test}  (base folder)")
        print(f"    Saved → {config.mcar_folder(rate)}/\n")

    print(f"  ✓ Cumulative injection complete (training only).\n")

    # ── Imputation pass ───────────────────────────────────────────────────────
    if run_impute:
        from imputer import run_imputation_pipeline
        run_imputation_pipeline(strategies=strategies, rates=rates)


def _inject_to_target(df, inject_cols, target_per_col, seed, prev_nulls):
    """
    Inject nulls into each feature column of the training DataFrame until
    target_per_col positions are null. Reuses previously injected positions.
    Prefers rows with fewer existing nulls to spread damage.
    """
    np.random.seed(seed)
    random.seed(seed)

    df_out  = df.copy()
    updated = {c: set(prev_nulls[c]) for c in inject_cols}

    for c in inject_cols:
        already = len(updated[c])
        need    = max(0, target_per_col - already)

        if need == 0:
            continue

        candidates = [
            i for i in range(len(df_out))
            if pd.notna(df_out.at[i, c]) and i not in updated[c]
        ]

        if not candidates:
            print(f"    ⚠ Column '{c}': no candidates left "
                  f"(already {already} nulls, need {need} more)")
            continue

        # Prioritise rows with fewer existing nulls
        null_count_per_row  = df_out[inject_cols].isnull().sum(axis=1)
        candidates_scored   = [(idx, null_count_per_row[idx]) for idx in candidates]
        candidates_scored.sort(key=lambda x: x[1])

        grouped = {}
        for idx, count in candidates_scored:
            grouped.setdefault(count, []).append(idx)

        pick = []
        for count in sorted(grouped.keys()):
            group = grouped[count]
            random.shuffle(group)
            remaining = need - len(pick)
            if remaining <= 0:
                break
            pick.extend(group[:remaining])

        pick = pick[:need]

        df_out.loc[pick, c] = np.nan
        updated[c].update(pick)

        print(f"    Column '{c}': {already} existing + {len(pick)} new = "
              f"{len(updated[c])} total nulls")

    return df_out, updated


def _validate(rates):
    if not rates:
        raise ValueError("[null_injector] config.MISSING_RATE must not be empty.")
    if not all(isinstance(r, int) and r > 0 for r in rates):
        raise ValueError(f"[null_injector] All values must be positive integers. Got: {rates}")
    if rates != sorted(rates):
        raise ValueError(f"[null_injector] Must be ascending order. Got: {rates}")
    if rates[-1] >= 100:
        raise ValueError(f"[null_injector] Rates must be < 100%. Got: {rates[-1]}%")


def _save_outputs(df_train, rate):
    """
    Save only the MCAR-injected training split into the rate-specific folder.
    Val and test are NOT copied — they live permanently in the base dataset
    folder (config.VAL_CSV_FILE / config.TEST_CSV_FILE) and are shared across
    all MCAR rates unchanged.
    """
    folder = config.mcar_folder(rate)
    os.makedirs(folder, exist_ok=True)

    feat_cols   = [c for c in df_train.columns if c != config.TARGET_COL]
    train_nulls = df_train[feat_cols].isnull().sum().sum()

    train_csv  = config.mcar_train_csv(rate)
    train_data = config.mcar_train_data(rate)
    df_train.to_csv(train_csv, index=False)
    df_train.fillna("?").to_csv(train_data, index=False, header=False)
    print(f"    [TRAIN] {train_csv}")
    print(f"    [TRAIN] Nulls injected : {train_nulls:,}  (rows={len(df_train):,})")
    print(f"    [TRAIN] {train_data}")
    print(f"    [VAL  ] {config.VAL_CSV_FILE}  ← shared, untouched")
    print(f"    [TEST ] {config.TEST_CSV_FILE}  ← shared, untouched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCAR Null Injection (training set only)")
    parser.add_argument("--impute",     action="store_true",
                        help="Run imputation after injection")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Imputation strategies to run (default: all)")
    args = parser.parse_args()

    if not os.path.exists(config.TRAIN_CSV_FILE):
        print(f"[null_injector] ✗ Train CSV not found: {config.TRAIN_CSV_FILE}")
        print("[null_injector]   Run python data_download.py first.")
    else:
        clean_train_df = pd.read_csv(config.TRAIN_CSV_FILE)
        print(f"[null_injector] Loaded training split: {clean_train_df.shape}"
              f"  |  rates: {config.MISSING_RATE}")
        run_null_injection(clean_train_df, run_impute=args.impute,
                           strategies=args.strategies)
