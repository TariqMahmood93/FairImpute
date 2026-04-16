import pandas as pd
import numpy as np

try:
    import torch
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception:
    torch = None
    faiss = None
    SentenceTransformer = None

BATCH_SIZE    = 64
MEDIAN_K      = 25
SIM_THRESHOLD = 0.65

_model_cache = {}


def _get_device():
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(name: str):
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed, cannot run SENTI imputation."
        )
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name, device=_get_device())
    return _model_cache[name]


def _embed_tuples(df: pd.DataFrame, model) -> np.ndarray:
    """Embed each row as a sentence. Handles nullable extension dtypes (Int64, etc.)."""
    df_safe = df.copy()
    for col in df_safe.columns:
        if pd.api.types.is_extension_array_dtype(df_safe[col].dtype):
            df_safe[col] = df_safe[col].astype(object)
    texts = df_safe.fillna("").astype(str).agg(" ".join, axis=1).tolist()
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=_get_device(),
    ).astype("float32")


def _build_faiss_index(emb: np.ndarray):
    if faiss is None:
        raise ImportError("faiss is not installed, cannot run SENTI imputation.")
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx


def _build_faiss_index_with_context(emb_new: np.ndarray, emb_ctx):
    """
    Build a FAISS index containing context vectors (previously imputed rows)
    followed by new-batch vectors.
    Returns (index, n_context, n_new, n_total, dim).
    """
    if faiss is None:
        raise ImportError("faiss is not installed, cannot run SENTI imputation.")
    dim       = int(emb_new.shape[1])
    n_new     = int(emb_new.shape[0])
    n_context = int(emb_ctx.shape[0]) if (emb_ctx is not None and len(emb_ctx) > 0) else 0
    idx = faiss.IndexFlatIP(dim)
    if n_context > 0:
        idx.add(emb_ctx.astype("float32"))
    idx.add(emb_new)
    return idx, n_context, n_new, int(idx.ntotal), dim


def _phase_stats(phase, n_context, n_new, n_total, dim, avg_norm):
    delta = n_total - n_context
    return {
        "phase": phase, "n_context": n_context, "n_new": n_new,
        "n_total": n_total, "delta": delta, "dim": dim,
        "avg_norm": round(avg_norm, 4), "ok": (delta == n_new),
    }


def _local_neighbor_fallback(df, emb, cat_cols, num_cols, cols, k=MEDIAN_K):
    if not cat_cols and not num_cols:
        return df
    idx = _build_faiss_index(emb)
    out = df.copy()
    row_idxs = np.where(out[cols].isna().any(axis=1))[0]
    if len(row_idxs) == 0:
        return out
    D, I = idx.search(emb[row_idxs], k)
    for qi, ridx in enumerate(row_idxs):
        neigh = I[qi]
        row = out.iloc[ridx].tolist()
        for c in cat_cols:
            ci = cols.index(c)
            if pd.isna(row[ci]):
                vals = [out.iat[nj, ci] for nj in neigh if pd.notna(out.iat[nj, ci])]
                if vals:
                    row[ci] = pd.Series(vals).mode().iloc[0]
        for c in num_cols:
            ci = cols.index(c)
            if pd.isna(row[ci]):
                vals = [out.iat[nj, ci] for nj in neigh if pd.notna(out.iat[nj, ci])]
                if vals:
                    _med = float(np.median(vals))
                    row[ci] = int(round(_med)) if pd.api.types.is_integer_dtype(out[c].dtype) else _med
        out.iloc[ridx] = row
    return out


def impute_senti(df: pd.DataFrame, cols, transformer_name: str, context_df=None):
    """
    SENTI imputer — returns (imputed_df, mask, faiss_stats).

    context_df : optional DataFrame of previously imputed rows (incremental mode)
                 used to enrich FAISS neighbor search without being re-imputed.
    faiss_stats: dict with n_context, n_new, n_total_after, dim, phases.
    """
    if df is None or df.empty:
        return df, df, {}

    if SentenceTransformer is None or faiss is None or torch is None:
        raise ImportError(
            "SENTI requires 'torch', 'faiss' and 'sentence-transformers' to be installed."
        )

    original   = df.copy()
    phase_logs = []

    target_cols = [c for c in (cols or list(df.columns)) if c in df.columns]
    if not target_cols:
        return df, original.isna() & df.notna(), {}

    cat_cols = [c for c in target_cols
                if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c])]
    num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not cat_cols and not num_cols:
        return df, original.isna() & df.notna(), {}

    model = _get_model(transformer_name)

    # Embed context rows (previously imputed batch, if any)
    emb_ctx     = None
    n_ctx       = 0
    ctx_aligned = None
    if context_df is not None and len(context_df) > 0:
        ctx_aligned = context_df.reindex(columns=df.columns, fill_value="").copy()
        emb_ctx     = _embed_tuples(ctx_aligned, model)
        n_ctx       = len(emb_ctx)

    def _ref(cur_df):
        if ctx_aligned is not None and n_ctx > 0:
            return pd.concat([ctx_aligned, cur_df], ignore_index=True)
        return cur_df

    # ── Phase 1: Categorical ──────────────────────────────────────────────────
    emb_orig = _embed_tuples(df, model)
    idx_orig, nc1, nn1, nt1, dim1 = _build_faiss_index_with_context(emb_orig, emb_ctx)
    avg1 = float(np.linalg.norm(emb_orig, axis=1).mean()) if len(emb_orig) else 0.0
    phase_logs.append(_phase_stats("Phase 1 — categorical", nc1, nn1, nt1, dim1, avg1))

    df_cat = df.copy()
    _debug_printed = False
    if cat_cols:
        missing_cat = df_cat[cat_cols].isna().any(axis=1)
        if missing_cat.any():
            row_idxs  = np.where(missing_cat)[0]
            k_search  = min(MEDIAN_K * 2 + n_ctx, idx_orig.ntotal)
            D_cat, I_cat = idx_orig.search(emb_orig[row_idxs], k_search)
            ref1 = _ref(df)
            for qi, ridx in enumerate(row_idxs):
                sims, neigh = D_cat[qi], I_cat[qi]
                row = df_cat.iloc[ridx].tolist()

                # ── Debug: print first query tuple and its neighbors ──
                if not _debug_printed:
                    _debug_printed = True
                    print(f"\n  ┌─ DEBUG: First query tuple (row {ridx}) ─────────────")
                    print(f"  │ Query row (with NaN):")
                    for ci2, col2 in enumerate(df_cat.columns):
                        val2 = row[ci2]
                        marker = " ← MISSING" if pd.isna(val2) else ""
                        print(f"  │   {col2:<20} = {val2}{marker}")
                    print(f"  │")
                    print(f"  │ Top 10 neighbors (of {k_search} searched):")
                    print(f"  │   {'Rank':<5} {'Index':<8} {'Sim':>8}  {'Self?':<6}")
                    for ni in range(min(10, len(neigh))):
                        is_self = "YES" if neigh[ni] == ridx else ""
                        print(f"  │   {ni+1:<5} {neigh[ni]:<8} {sims[ni]:>8.4f}  {is_self}")
                    # Show neighbor values for first missing cat col
                    first_missing_cat = next((c for c in cat_cols if pd.isna(row[df_cat.columns.get_loc(c)])), None)
                    if first_missing_cat:
                        fci = df_cat.columns.get_loc(first_missing_cat)
                        print(f"  │")
                        print(f"  │ Neighbor values for '{first_missing_cat}':")
                        n_nan = 0
                        n_valid = 0
                        for ni in range(min(25, len(neigh))):
                            nj = neigh[ni]
                            if nj == ridx:
                                continue
                            if nj < len(ref1):
                                nval = ref1.iloc[nj, fci]
                                if pd.notna(nval):
                                    n_valid += 1
                                    if n_valid <= 8:
                                        print(f"  │   neighbor[{nj}] sim={sims[ni]:.4f} → {nval}")
                                else:
                                    n_nan += 1
                        print(f"  │   ... {n_valid} valid, {n_nan} also NaN (out of top 25)")
                    print(f"  └───────────────────────────────────────────────────\n")

                for c in cat_cols:
                    ci = df_cat.columns.get_loc(c)
                    if pd.isna(row[ci]):
                        # High-confidence direct match (skip self)
                        dm = next((
                            ref1.iloc[nj, ci]
                            for nj, s in zip(neigh, sims)
                            if nj != ridx and nj < len(ref1) and pd.notna(ref1.iloc[nj, ci]) and s >= 0.97
                        ), None)
                        if dm is not None:
                            row[ci] = dm
                        else:
                            votes = {}
                            for nj, s in zip(neigh, sims):
                                if nj == ridx or nj >= len(ref1):
                                    continue
                                val = ref1.iloc[nj, ci]
                                if pd.notna(val) and s >= SIM_THRESHOLD:
                                    votes[val] = votes.get(val, 0.0) + float(s)
                            if votes:
                                row[ci] = max(votes, key=votes.get)
                df_cat.iloc[ridx] = row

    # ── Phase 2: Numeric ──────────────────────────────────────────────────────
    emb_cat = _embed_tuples(df_cat, model)
    idx_cat, nc2, nn2, nt2, dim2 = _build_faiss_index_with_context(emb_cat, emb_ctx)
    avg2 = float(np.linalg.norm(emb_cat, axis=1).mean()) if len(emb_cat) else 0.0
    phase_logs.append(_phase_stats("Phase 2 — numeric", nc2, nn2, nt2, dim2, avg2))

    df_num = df_cat.copy()
    if num_cols:
        missing_num = df_num[num_cols].isna().any(axis=1)
        if missing_num.any():
            row_idxs  = np.where(missing_num)[0]
            k_search  = min(MEDIAN_K * 2 + n_ctx, idx_cat.ntotal)
            D_num, I_num = idx_cat.search(emb_cat[row_idxs], k_search)
            ref2 = _ref(df_cat)
            for qi, ridx in enumerate(row_idxs):
                sims, neigh = D_num[qi], I_num[qi]
                row = df_num.iloc[ridx].tolist()
                for c in num_cols:
                    ci = df_num.columns.get_loc(c)
                    if pd.isna(row[ci]):
                        wsum, wtot = 0.0, 0.0
                        for nj, s in zip(neigh, sims):
                            if nj == ridx or nj >= len(ref2):
                                continue
                            val = ref2.iloc[nj, ci]
                            if pd.notna(val) and s >= SIM_THRESHOLD:
                                wsum += float(val) * float(s)
                                wtot += float(s)
                        if wtot > 0:
                            _val = wsum / wtot
                            row[ci] = int(round(_val)) if pd.api.types.is_integer_dtype(df_num[c].dtype) else _val
                df_num.iloc[ridx] = row

    # ── Phase 3: Local neighbor fallback ─────────────────────────────────────
    emb_final = _embed_tuples(df_num, model)
    idx_f, nc3, nn3, nt3, dim3 = _build_faiss_index_with_context(emb_final, emb_ctx)
    avg3 = float(np.linalg.norm(emb_final, axis=1).mean()) if len(emb_final) else 0.0
    phase_logs.append(_phase_stats("Phase 3 — fallback", nc3, nn3, nt3, dim3, avg3))

    df_loc = _local_neighbor_fallback(
        df_num, emb_final, cat_cols, num_cols, list(df_num.columns), k=MEDIAN_K,
    )

    # ── Phase 4: Global column-level fallback ─────────────────────────────────
    for c in cat_cols:
        if df_loc[c].isna().any():
            mode_val = df_loc[c].mode(dropna=True)
            if not mode_val.empty:
                df_loc[c] = df_loc[c].fillna(mode_val.iloc[0])

    for c in num_cols:
        if df_loc[c].isna().any():
            df_loc[c] = df_loc[c].fillna(df_loc[c].median(skipna=True))
        s = df_loc[c]
        if pd.api.types.is_integer_dtype(s.dtype):
            try:
                df_loc[c] = s.round(0).astype(s.dtype)
            except Exception:
                df_loc[c] = s.round(0).astype("Int64")
        # floats: keep as-is, no forced rounding

    mask = original.isna() & df_loc.notna()
    faiss_stats = {
        "n_context": n_ctx, "n_new": len(df),
        "n_total_after": n_ctx + len(df), "dim": dim1, "phases": phase_logs,
    }
    return df_loc, mask, faiss_stats
