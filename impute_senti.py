"""
impute_senti.py
────────────────────────────────────────────────────────────────────────────
SENTI transformer-based imputation.

Uses sentence embeddings + FAISS nearest-neighbor search to impute
missing values in a context-aware manner. Each row is represented as a
sentence (concatenation of its non-missing cell values), embedded via
a pre-trained sentence-transformer model, and missing values are filled
by looking at the nearest complete rows in embedding space.

Delegates the heavy lifting to senti_backend.impute_senti().

Pros:
    - Context-aware: uses semantic meaning of cell values
    - Handles categorical columns naturally (no label encoding needed)
    - Can capture complex relationships between columns

Cons:
    - Requires sentence-transformers and FAISS (heavier dependencies)
    - Slower than mode/median or KNN
    - Quality depends on the pre-trained transformer model

Configuration (in config.py):
    SENTI_TRANSFORMER = "all-MiniLM-L6-v2"   # sentence-transformers model
"""

import pandas as pd
import config


def impute_senti(df: pd.DataFrame, target_col: str,
                 cat_cols: list, num_cols: list) -> pd.DataFrame:
    """
    Impute missing values using SENTI transformer-based approach.

    Parameters
    ----------
    df         : DataFrame with NaN cells to fill
    target_col : Column to never impute
    cat_cols   : Categorical columns (not used directly — SENTI handles all)
    num_cols   : Numerical columns (not used directly — SENTI handles all)

    Returns
    -------
    DataFrame with NaNs filled
    """
    from senti_backend import impute_senti as _senti_impute

    all_cols = [c for c in df.columns if c != target_col]
    df_imputed, mask, stats = _senti_impute(
        df, cols=all_cols, transformer_name=config.SENTI_TRANSFORMER,
    )

    # Log FAISS stats
    if stats:
        for phase in stats.get("phases", []):
            print(f"    {phase['phase']}: n_total={phase['n_total']}, "
                  f"dim={phase['dim']}, avg_norm={phase['avg_norm']}")

    return df_imputed
