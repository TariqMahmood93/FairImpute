"""
impute_mice.py
────────────────────────────────────────────────────────────────────────────
MICE — Multiple Imputation by Chained Equations.

Uses scikit-learn's IterativeImputer (experimental) which models each
feature with missing values as a function of the other features in a
round-robin fashion over multiple iterations.

For categorical columns: label-encode → MICE → round → decode.
This is the standard approach and sits between simple mode/median
and heavyweight transformer-based methods in sophistication.

Pros:
    - Preserves inter-feature correlations (unlike mode/median)
    - Uses BayesianRidge by default (regularized, handles collinearity)
    - Iterative refinement converges to stable imputations
    - Deterministic with fixed random state

Cons:
    - Slower than mode/median
    - Label-encoding categoricals is approximate
    - May not converge for very high missing rates
    - Assumes linear relationships between features (BayesianRidge default)

Configuration (in config.py):
    MICE_MAX_ITER     = 10    # number of imputation rounds
    MICE_RANDOM_STATE = 42    # reproducibility
"""

import numpy as np
import pandas as pd
import config


def impute_mice(df: pd.DataFrame, target_col: str,
                cat_cols: list, num_cols: list) -> pd.DataFrame:
    """
    Impute missing values using MICE (chained equations).

    Parameters
    ----------
    df         : DataFrame with NaN cells to fill
    target_col : Column to never impute
    cat_cols   : Categorical columns (label-encoded → MICE → decoded)
    num_cols   : Numerical columns (imputed directly by MICE)

    Returns
    -------
    DataFrame with NaNs filled
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import LabelEncoder

    max_iter     = config.MICE_MAX_ITER
    random_state = config.MICE_RANDOM_STATE

    print(f"    MICE params: max_iter={max_iter}, random_state={random_state}")

    # ── Label-encode categoricals so MICE can process them ────────────────
    encoders  = {}
    df_work   = df.copy()

    for col in cat_cols:
        if df_work[col].isnull().all():
            continue
        le = LabelEncoder()
        non_null = df_work[col].dropna().astype(str)
        le.fit(non_null)
        encoders[col] = le
        df_work[col] = df_work[col].apply(
            lambda x: le.transform([str(x)])[0] if pd.notna(x) else np.nan
        )

    # ── Run IterativeImputer on all feature columns ───────────────────────
    feat_cols   = [c for c in df.columns if c != target_col]
    impute_cols = [c for c in feat_cols
                   if c in num_cols or c in encoders
                   or pd.api.types.is_numeric_dtype(df_work[c])]

    if impute_cols:
        mice = IterativeImputer(
            max_iter     = max_iter,
            random_state = random_state,
            sample_posterior = False,   # point estimates (deterministic)
        )
        df_work[impute_cols] = mice.fit_transform(df_work[impute_cols])

        # Track convergence (IterativeImputer stores n_iter_)
        actual_iter = getattr(mice, 'n_iter_', max_iter)
        print(f"    MICE converged in {actual_iter} iteration(s)")

    # ── Decode categoricals back to original labels ───────────────────────
    for col, le in encoders.items():
        rounded = df_work[col].round(0).astype(int)
        rounded = rounded.clip(0, len(le.classes_) - 1)
        df_work[col] = le.inverse_transform(rounded)

    # ── Fallback for any remaining NaNs (edge cases) ──────────────────────
    for col in feat_cols:
        if df_work[col].isnull().any():
            if df_work[col].dtype == object:
                mode_val = df_work[col].mode(dropna=True)
                if not mode_val.empty:
                    df_work[col] = df_work[col].fillna(mode_val.iloc[0])
            else:
                df_work[col] = df_work[col].fillna(df_work[col].median())

    # ── Restore integer dtypes ────────────────────────────────────────────
    for col in num_cols:
        if col in df_work.columns and pd.api.types.is_integer_dtype(df[col].dtype):
            try:
                df_work[col] = df_work[col].round(0).astype(df[col].dtype)
            except Exception:
                pass

    return df_work
