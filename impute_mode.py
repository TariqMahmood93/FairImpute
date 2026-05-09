"""
impute_mode.py
────────────────────────────────────────────────────────────────────────────
Mode (categorical) + Median (numerical) imputation.

Simple, fast, widely-used baseline. Replaces each NaN with the
column-wise mode (for categoricals) or median (for numericals).

This is the most basic imputation strategy — it ignores relationships
between columns and treats each column independently.

Pros:
    - Very fast, no dependencies beyond pandas
    - Works on any dataset without configuration
    - Good baseline to compare against more complex methods

Cons:
    - Does not preserve inter-feature correlations
    - Reduces variance (all missing values get the same fill value)
    - Can distort distributions for skewed data
"""

import pandas as pd


def impute_mode(df: pd.DataFrame, target_col: str,
                cat_cols: list, num_cols: list) -> pd.DataFrame:
    """
    Impute missing values using mode (categorical) and median (numerical).

    Parameters
    ----------
    df         : DataFrame with NaN cells to fill
    target_col : Column to never impute
    cat_cols   : Categorical columns (filled with mode)
    num_cols   : Numerical columns (filled with median)

    Returns
    -------
    DataFrame with NaNs filled
    """
    feat_cols = [c for c in df.columns if c != target_col]

    for col in feat_cols:
        if col in cat_cols or df[col].dtype == object:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
        elif col in num_cols or pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median(skipna=True)
            df[col] = df[col].fillna(median_val)
            if pd.api.types.is_integer_dtype(df[col].dtype):
                try:
                    df[col] = df[col].round(0).astype(df[col].dtype)
                except Exception:
                    pass
        else:
            # Fallback: mode for anything unclassified
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])

    return df
