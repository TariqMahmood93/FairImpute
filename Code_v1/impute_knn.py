"""
impute_knn.py
────────────────────────────────────────────────────────────────────────────
KNN imputation using scikit-learn's KNNImputer.

For numerical columns, uses distance-weighted average of k nearest
neighbors. For categorical columns, label-encodes them first, runs
KNN imputation, then rounds and decodes back to original labels.

Pros:
    - Preserves local structure (uses similar rows to fill values)
    - Distance-weighted: closer neighbors have more influence
    - No distributional assumptions

Cons:
    - Slow on large datasets (O(n²) distance computation)
    - Label-encoding categoricals is a crude approximation
    - Sensitive to the choice of k and distance metric

Configuration (in config.py):
    KNN_IMPUTE_NEIGHBORS = 5   # number of neighbors
"""

import numpy as np
import pandas as pd
import config


def impute_knn(df: pd.DataFrame, target_col: str,
               cat_cols: list, num_cols: list) -> pd.DataFrame:
    """
    Impute missing values using KNN.

    Parameters
    ----------
    df         : DataFrame with NaN cells to fill
    target_col : Column to never impute
    cat_cols   : Categorical columns (label-encoded → KNN → decoded)
    num_cols   : Numerical columns (imputed directly by KNN)

    Returns
    -------
    DataFrame with NaNs filled
    """
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import LabelEncoder

    k = config.KNN_IMPUTE_NEIGHBORS

    # ── Categorical: encode → KNN → decode ────────────────────────────────
    # KNNImputer only works on numerics, so we label-encode categoricals,
    # impute, then round and decode.
    encoders = {}
    df_encoded = df.copy()

    for col in cat_cols:
        if df_encoded[col].isnull().all():
            continue
        le = LabelEncoder()
        non_null = df_encoded[col].dropna().astype(str)
        le.fit(non_null)
        encoders[col] = le

        # Encode non-null values, keep NaN as NaN
        df_encoded[col] = df_encoded[col].apply(
            lambda x: le.transform([str(x)])[0] if pd.notna(x) else np.nan
        )

    # ── Numerical imputation ──────────────────────────────────────────────
    impute_cols = num_cols + [c for c in cat_cols if c in encoders]
    if impute_cols:
        knn = KNNImputer(n_neighbors=k, weights="distance")
        df_encoded[impute_cols] = knn.fit_transform(df_encoded[impute_cols])

    # ── Decode categoricals back ──────────────────────────────────────────
    for col, le in encoders.items():
        rounded = df_encoded[col].round(0).astype(int)
        # Clamp to valid label range
        rounded = rounded.clip(0, len(le.classes_) - 1)
        df_encoded[col] = le.inverse_transform(rounded)

    # ── Handle any remaining categoricals not in encoders ─────────────────
    feat_cols = [c for c in df.columns if c != target_col]
    for col in feat_cols:
        if df_encoded[col].isnull().any():
            if df_encoded[col].dtype == object:
                mode_val = df_encoded[col].mode(dropna=True)
                if not mode_val.empty:
                    df_encoded[col] = df_encoded[col].fillna(mode_val.iloc[0])
            else:
                df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

    return df_encoded
