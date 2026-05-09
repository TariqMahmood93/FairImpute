"""
imputer.py
────────────────────────────────────────────────────────────────────────────
Modular imputation framework.  Dispatches to pluggable strategy handlers.

Each strategy lives in its own file:
    impute_mode.py   → Mode/Median imputation
    impute_senti.py  → SENTI transformer-based imputation
    impute_knn.py    → KNN imputation
    impute_mice.py   → MICE (chained equations) imputation

This file provides:
    1. impute(df, strategy) — the main entry point that dispatches to
       the correct strategy handler
    2. run_imputation_pipeline() — batch runner for all strategies × rates

HOW TO ADD A NEW IMPUTATION METHOD
────────────────────────────────────────────────────────────────────────────
1. Create a new file: impute_<name>.py
   - Implement: impute_<name>(df, target_col, cat_cols, num_cols) -> df
2. Add its name to config.IMPUTATION_STRATEGIES (config.py)
3. Register it in the _HANDLERS dict at the bottom of this file

That's it — main.py picks it up automatically via config.RESOLVED_IMPUTATION.

Usage:
    from imputer import impute
    df_imputed = impute(df_with_nulls, strategy="mode")
"""

import os
import numpy as np
import pandas as pd

import config

# ── Import strategy handlers from their own files ─────────────────────────
from impute_mode  import impute_mode
from impute_senti import impute_senti
from impute_knn   import impute_knn
from impute_mice  import impute_mice


# =====================================================================
# HANDLER REGISTRY — maps strategy names to their handler functions
# =====================================================================

_HANDLERS = {
    "mode"  : impute_mode,
    "senti" : impute_senti,
    "knn"   : impute_knn,
    "mice"  : impute_mice,
    # ── Add new strategies below ──────────────────────────────────────
    # "gain"        : impute_gain,         # from impute_gain import impute_gain
    # "your_method" : impute_your_method,  # from impute_your_method import ...
}


# =====================================================================
# PUBLIC API
# =====================================================================

def impute(df: pd.DataFrame, strategy: str,
           target_col: str = None, cat_cols: list = None,
           num_cols: list = None) -> pd.DataFrame:
    """
    Impute missing values in df using the specified strategy.

    Parameters
    ----------
    df         : DataFrame with NaN cells to fill
    strategy   : One of the keys in config.IMPUTATION_STRATEGIES
    target_col : Column to NEVER impute (default: config.TARGET_COL)
    cat_cols   : Categorical columns (auto-detected if None)
    num_cols   : Numerical columns (auto-detected if None)

    Returns
    -------
    DataFrame with NaNs filled (target column untouched)
    """
    if strategy == "none":
        print(f"[imputer] Strategy='none' — returning data as-is (with nulls)")
        return df.copy()

    if strategy not in _HANDLERS:
        available = list(_HANDLERS.keys())
        raise ValueError(
            f"[imputer] Unknown strategy '{strategy}'.\n"
            f"  Registered handlers: {available}\n"
            f"  To add a new one:\n"
            f"    1. Create impute_{strategy}.py with an impute_{strategy}() function\n"
            f"    2. Import it in imputer.py\n"
            f"    3. Add it to the _HANDLERS dict"
        )

    target_col = target_col or config.TARGET_COL
    feat_cols  = [c for c in df.columns if c != target_col]

    if cat_cols is None:
        cat_cols = [c for c in feat_cols
                    if df[c].dtype == object
                    or pd.api.types.is_categorical_dtype(df[c])]
    if num_cols is None:
        num_cols = [c for c in feat_cols
                    if c in config.NUMERICAL_COLS]

    before_nulls = df[feat_cols].isnull().sum().sum()
    print(f"[imputer] Strategy       : {strategy}  "
          f"({config.IMPUTATION_STRATEGIES.get(strategy, {}).get('label', '')})")
    print(f"[imputer] Handler file   : impute_{strategy}.py")
    print(f"[imputer] NaN cells before: {before_nulls:,}")

    handler = _HANDLERS[strategy]
    df_out  = handler(df.copy(), target_col, cat_cols, num_cols)

    after_nulls = df_out[feat_cols].isnull().sum().sum()
    filled      = before_nulls - after_nulls
    print(f"[imputer] NaN cells after : {after_nulls:,}  (filled {filled:,})")

    return df_out


def run_imputation_pipeline(strategies: list = None, rates: list = None,
                             run_eval: bool = True) -> None:
    """
    Run imputation for every (rate, strategy) combination.

    For each MCAR rate, reads the raw MCAR CSV and applies each strategy,
    saving the result to its own folder.

    Parameters
    ----------
    strategies : list of strategy names (default: all registered except "none")
    rates      : list of MCAR rates (default: config.MISSING_RATE)
    run_eval   : if True (default), run imputation evaluation after each
                 imputed CSV is saved and write to
                 RESULTS_DIR/imputation_evaluation.csv
    """
    if strategies is None:
        strategies = [s for s in config.IMPUTATION_STRATEGIES if s != "none"]
    if rates is None:
        rates = config.MISSING_RATE

    print(f"\n{'='*65}")
    print(f"  IMPUTATION PIPELINE")
    print(f"  Dataset    : {config.DATASET_NAME}")
    print(f"  Rates      : {rates}%")
    print(f"  Strategies : {strategies}")
    print(f"  Evaluation : {'enabled' if run_eval else 'disabled'}")
    print(f"{'='*65}\n")

    for rate in rates:
        raw_csv = config.mcar_csv(rate)
        if not os.path.exists(raw_csv):
            print(f"[imputer] ⚠ Raw MCAR CSV not found: {raw_csv}  — skipping rate {rate}%")
            continue

        df_raw = pd.read_csv(raw_csv)
        print(f"\n── Rate {rate}%  ({len(df_raw):,} rows, "
              f"{df_raw.drop(columns=[config.TARGET_COL]).isnull().sum().sum():,} NaN cells) ──")

        for strategy in strategies:
            print(f"\n  ▸ Imputing with '{strategy}' ...")
            try:
                df_imputed = impute(df_raw, strategy)

                out_folder = config.imputed_folder(rate, strategy)
                out_csv    = config.imputed_csv(rate, strategy)
                os.makedirs(out_folder, exist_ok=True)
                df_imputed.to_csv(out_csv, index=False)
                print(f"  ✓ Saved → {out_csv}")

                if run_eval:
                    try:
                        from imputation_eval import evaluate_imputation
                        evaluate_imputation(rate=rate, strategy=strategy,
                                            append_csv=True)
                    except Exception as eval_exc:
                        print(f"  ⚠ Evaluation failed: {eval_exc}")

            except Exception as exc:
                print(f"  ✗ Failed: {exc}")

    print(f"\n{'='*65}")
    print(f"  Imputation pipeline complete.")
    print(f"{'='*65}\n")


def run_imputation_pipeline_all_seeds(strategies: list = None,
                                       rates: list = None,
                                       run_eval: bool = True) -> None:
    """
    Run imputation for every (seed, rate, strategy) combination.

    For each seed in config.SEEDS:
      1. Set the global seed
      2. For each MCAR rate, read the seed-specific null-injected CSV
      3. Apply each imputation strategy
      4. Save the imputed CSV to a seed-specific folder

    This produces files like:
        adult_census_mcar_5pct_knn_seed42/adult_census_mcar_5_knn_42.csv
        adult_census_mcar_5pct_knn_seed100/adult_census_mcar_5_knn_100.csv
        ...
    """
    if strategies is None:
        strategies = [s for s in config.IMPUTATION_STRATEGIES if s != "none"]
    if rates is None:
        rates = config.MISSING_RATE

    seeds = config.SEEDS

    print(f"\n{'='*65}")
    print(f"  IMPUTATION PIPELINE — ALL SEEDS")
    print(f"  Dataset    : {config.DATASET_NAME}")
    print(f"  Seeds      : {seeds}")
    print(f"  Rates      : {rates}%")
    print(f"  Strategies : {strategies}")
    print(f"  Expected   : {len(seeds)} seeds × {len(rates)} rates × "
          f"{len(strategies)} strategies = "
          f"{len(seeds) * len(rates) * len(strategies)} imputed files")
    print(f"{'='*65}\n")

    for seed_i, seed in enumerate(seeds, 1):
        print(f"\n{'▓'*65}")
        print(f"  SEED {seed_i}/{len(seeds)} — seed={seed}")
        print(f"{'▓'*65}")

        config.set_seed(seed)
        run_imputation_pipeline(strategies=strategies, rates=rates,
                                run_eval=run_eval)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  IMPUTATION COMPLETE — ALL SEEDS")
    print(f"{'='*65}")
    for seed in seeds:
        config.set_seed(seed)
        for rate in rates:
            for strategy in strategies:
                csv_path = config.imputed_csv(rate, strategy)
                if os.path.exists(csv_path):
                    size_kb = os.path.getsize(csv_path) / 1024
                    print(f"  ✓ seed={seed}  {rate}%  {strategy}  →  "
                          f"{csv_path}  ({size_kb:.0f} KB)")
                else:
                    print(f"  ✗ seed={seed}  {rate}%  {strategy}  →  "
                          f"MISSING: {csv_path}")
    print()


# ── Standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Imputation Pipeline")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Imputation strategies to run (default: all)")
    parser.add_argument("--single-seed", action="store_true",
                        help="Only run for the current config.RANDOM_SEED "
                             "(default: run all seeds in config.SEEDS)")
    args = parser.parse_args()

    if args.single_seed:
        run_imputation_pipeline(strategies=args.strategies)
    else:
        run_imputation_pipeline_all_seeds(strategies=args.strategies)
