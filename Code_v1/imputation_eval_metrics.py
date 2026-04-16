from typing import Dict, Tuple
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer, util
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def record_missing_positions(source_df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean mask of positions that were missing in the incomplete dataset."""
    return source_df.isna()


def _align_frames(A: pd.DataFrame, B: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames on common columns and truncate to the same row count."""
    common = [c for c in A.columns if c in B.columns]
    A2 = A[common].copy()
    B2 = B[common].copy()
    n = min(len(A2), len(B2))
    return A2.iloc[:n].copy(), B2.iloc[:n].copy()


def exact_match_at_positions(imputed_df: pd.DataFrame,
                              ground_df: pd.DataFrame,
                              missing_mask: pd.DataFrame) -> Dict:
    """
    Exact-match rate computed only at positions that were originally missing.
    Returns dict with 'overall' stats and 'per_column' DataFrame.
    """
    impA, gtA = _align_frames(imputed_df, ground_df)
    maskA, _  = _align_frames(missing_mask, ground_df)
    mask_bool = maskA.astype(bool)

    comp = (impA == gtA) & mask_bool

    per_col_true  = comp.sum(axis=0)
    per_col_total = mask_bool.sum(axis=0)
    per_col_rate  = (per_col_true / per_col_total.replace(0, np.nan)).astype(float)

    overall_true  = int(per_col_true.sum())
    overall_total = int(per_col_total.sum())
    overall_rate  = float(overall_true / overall_total) if overall_total else float("nan")

    return {
        "overall":    {"true": overall_true, "total": overall_total, "rate": overall_rate},
        "per_column": pd.DataFrame({"true": per_col_true, "total": per_col_total, "rate": per_col_rate}),
        "match_mask": comp,
    }


def semantic_similarity_at_positions(imputed_df: pd.DataFrame,
                                      ground_df: pd.DataFrame,
                                      missing_mask: pd.DataFrame,
                                      model_name: str,
                                      batch_size: int = 64) -> Dict:
    """
    Cosine similarity between imputed and ground-truth cell strings,
    computed only at originally-missing positions.
    """
    if not _HAS_ST:
        raise RuntimeError(
            "sentence-transformers is not available. Please install it."
        )

    impA, gtA = _align_frames(imputed_df, ground_df)
    maskA, _  = _align_frames(missing_mask, ground_df)
    mask_bool = maskA.astype(bool)

    # Collect (imputed, ground_truth) string pairs at missing positions
    imp_vals, gt_vals, row_ids, col_ids = [], [], [], []
    for col in impA.columns:
        if col not in gtA.columns or col not in mask_bool.columns:
            continue
        m = mask_bool[col]
        if not m.any():
            continue
        sel = m[m].index
        imp_vals.extend(impA[col].astype(str).loc[sel].tolist())
        gt_vals.extend(gtA[col].astype(str).loc[sel].tolist())
        row_ids.extend(sel.tolist())
        col_ids.extend([col] * len(sel))

    if not imp_vals:
        return {
            "overall": {"mean": float("nan"), "count": 0},
            "per_column": pd.DataFrame(columns=["mean", "count"]),
            "cell_scores": pd.DataFrame(np.nan, index=impA.index, columns=impA.columns),
        }

    model   = SentenceTransformer(model_name)
    emb_imp = model.encode(imp_vals, batch_size=batch_size, show_progress_bar=False,
                           convert_to_tensor=True, normalize_embeddings=True)
    emb_gt  = model.encode(gt_vals,  batch_size=batch_size, show_progress_bar=False,
                           convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(emb_imp, emb_gt).diagonal().cpu().numpy()

    cell_scores = pd.DataFrame(np.nan, index=impA.index, columns=impA.columns)
    for r, c, s in zip(row_ids, col_ids, sims):
        if r in cell_scores.index and c in cell_scores.columns:
            cell_scores.at[r, c] = float(s)

    overall_mean = float(np.mean(sims)) if len(sims) else float("nan")

    per_col = {}
    for c in impA.columns:
        vals = cell_scores[c].dropna().astype(float)
        per_col[c] = {"mean": float(vals.mean()) if len(vals) else float("nan"),
                      "count": int(len(vals))}

    return {
        "overall":    {"mean": overall_mean, "count": int(len(sims))},
        "per_column": pd.DataFrame(per_col).T,
        "cell_scores": cell_scores,
    }
