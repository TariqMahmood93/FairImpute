"""
imputation_eval.py
────────────────────────────────────────────────────────────────────────────
Evaluates imputation quality by comparing imputed values against ground truth.

For each (rate, strategy) pair:
  1. Load clean training CSV   → ground truth
  2. Load MCAR CSV             → NaN positions mark what was injected
  3. Load imputed CSV          → compare imputed values at those positions

Metrics per column
──────────────────
  n_imputed        : cells that were nulled and then imputed
  exact_matches    : cells where imputed value == original value
  exact_match_rate : exact_matches / n_imputed
  sem_sim_mean     : mean semantic similarity score in [0, 1]

Semantic similarity
───────────────────
  Categorical : cosine similarity of sentence embeddings (if sentence-
                transformers is installed), otherwise SequenceMatcher ratio.
  Numerical   : max(0,  1 – |imputed – truth| / range(col_in_truth))
  Aggregate   : weighted mean over all columns (weighted by n_imputed)

Output
──────
  RESULTS_DIR/imputation_evaluation.csv   ← one row per column + "_ALL_"
  RESULTS_DIR/<prefix>_imputation_eval.json
  RESULTS_DIR/<prefix>_imputation_eval.txt

Usage
─────
    from imputation_eval import run_eval_pipeline, evaluate_imputation

    # Evaluate all rates × strategies written so far
    run_eval_pipeline()

    # Evaluate one specific combo
    evaluate_imputation(rate=10, strategy="knn")

    # Auto-detect from current config (legacy single-source mode)
    evaluate_imputation()

    # Standalone
    python imputation_eval.py
    python imputation_eval.py --rates 10 20 --strategies knn mode
    python imputation_eval.py --no-embeddings   # skip slow embedding computation
"""

import json
import os
import argparse
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import config
from imputation_eval_metrics import (
    record_missing_positions,
    exact_match_at_positions,
    semantic_similarity_at_positions,
)

try:
    from imputation_eval_metrics import _HAS_ST
except ImportError:
    _HAS_ST = False


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_imputation(
    rate: int = None,
    strategy: str = None,
    ground_truth_csv: str = None,
    mcar_csv_path: str = None,
    imputed_csv_path: str = None,
    compute_embeddings: bool = True,
    append_csv: bool = True,
) -> dict:
    """
    Evaluate imputation quality for one (rate, strategy) pair.

    When called with no arguments, auto-detects everything from config
    (legacy single-source mode — works when config.SOURCE is already set
    to an imputed source like "mcar_10_knn").

    Parameters
    ----------
    rate               : MCAR rate (e.g. 10 for 10%). Ignored when
                         ground_truth_csv / mcar_csv_path are explicit.
    strategy           : imputation strategy (e.g. "knn").
    ground_truth_csv   : path to clean training split (default: TRAIN_CSV_FILE)
    mcar_csv_path      : path to null-injected CSV (default: mcar_train_csv(rate))
    imputed_csv_path   : path to imputed CSV (default: imputed_csv(rate, strategy))
    compute_embeddings : if True and sentence-transformers is installed,
                         compute embedding-based similarity for categorical cols.
                         Set False to skip slow embedding computation.
    append_csv         : if True, append to existing imputation_evaluation.csv.

    Returns
    -------
    dict with evaluation results, or {"available": False} on failure.
    """
    # ── Auto-detect from config when no explicit params ───────────────────
    _auto_mode = (rate is None and strategy is None
                  and ground_truth_csv is None
                  and mcar_csv_path is None)

    if _auto_mode:
        if config.RESOLVED_IMPUTATION == "none" or config.SOURCE == "clean":
            print("[imputation_eval] Skipped — no imputation to evaluate "
                  f"(SOURCE='{config.SOURCE}', "
                  f"imputation='{config.RESOLVED_IMPUTATION}')")
            return {"available": False}
        parts = config.SOURCE.split("_")
        try:
            rate     = int(parts[1])
            strategy = config.RESOLVED_IMPUTATION
        except (IndexError, ValueError):
            print(f"[imputation_eval] Cannot parse rate from SOURCE='{config.SOURCE}'")
            return {"available": False}

    # ── Resolve paths ─────────────────────────────────────────────────────
    ground_truth_csv = ground_truth_csv or config.TRAIN_CSV_FILE
    mcar_csv_path    = mcar_csv_path    or config.mcar_train_csv(rate)
    if imputed_csv_path is None:
        imputed_csv_path = (config.SOURCE_CSV if _auto_mode
                            else config.imputed_csv(rate, strategy))

    for label, path in [
        ("Ground truth CSV", ground_truth_csv),
        ("MCAR CSV",         mcar_csv_path),
        ("Imputed CSV",      imputed_csv_path),
    ]:
        if not os.path.exists(path):
            print(f"[imputation_eval] {label} not found: {path}")
            return {"available": False}

    df_clean   = pd.read_csv(ground_truth_csv)
    df_mcar    = pd.read_csv(mcar_csv_path)
    df_imputed = pd.read_csv(imputed_csv_path)

    if len(df_clean) != len(df_mcar) or len(df_clean) != len(df_imputed):
        print(
            f"[imputation_eval] Row count mismatch — "
            f"truth={len(df_clean)}, mcar={len(df_mcar)}, "
            f"imputed={len(df_imputed)}.\n"
            f"  All three CSVs must be the training split in identical row order."
        )
        return {"available": False}

    feat_cols = [c for c in df_clean.columns if c != config.TARGET_COL]
    cat_cols  = set(c for c in feat_cols if df_clean[c].dtype == object)
    num_cols  = set(config.NUMERICAL_COLS) & set(feat_cols)

    missing_mask = record_missing_positions(df_mcar)
    missing_feat = missing_mask[feat_cols]
    total_missing = int(missing_feat.sum().sum())

    imp_label = config.IMPUTATION_STRATEGIES.get(strategy, {}).get("label", strategy)

    print(f"\n[imputation_eval] rate={rate}%  strategy='{strategy}'  "
          f"({imp_label})")
    print(f"  Ground truth : {ground_truth_csv}")
    print(f"  MCAR CSV     : {mcar_csv_path}")
    print(f"  Imputed CSV  : {imputed_csv_path}")
    print(f"  Missing cells: {total_missing:,}")

    if total_missing == 0:
        print("[imputation_eval] ⚠ No missing cells found — nothing to evaluate")
        return {"available": False}

    # ── Exact match (all columns) ─────────────────────────────────────────
    exact = exact_match_at_positions(
        df_imputed[feat_cols], df_clean[feat_cols], missing_feat
    )

    # ── Semantic similarity ───────────────────────────────────────────────
    # Categorical: sentence embeddings (if available) or SequenceMatcher ratio
    # Numerical  : normalised distance-based score
    sem_scores = _compute_semantic(
        df_imputed, df_clean, missing_mask, feat_cols,
        cat_cols, num_cols, compute_embeddings,
    )

    # ── Build per-column DataFrame ────────────────────────────────────────
    per_col_records = []
    for col in feat_cols:
        if col not in missing_feat.columns:
            continue
        n_imp = int(missing_feat[col].sum())
        if n_imp == 0:
            continue

        col_type = ("categorical" if col in cat_cols
                    else "numerical" if col in num_cols else "other")

        # Exact match from metrics module
        ec_df = exact["per_column"]
        if col in ec_df.index:
            n_exact  = int(ec_df.loc[col, "true"])
            em_rate  = float(ec_df.loc[col, "rate"])
        else:
            n_exact = 0
            em_rate = float("nan")

        # Semantic similarity
        sim_mean = float(sem_scores.get(col, {}).get("mean", float("nan")))

        per_col_records.append({
            "dataset"          : config.DATASET_NAME,
            "seed"             : config.RANDOM_SEED,
            "rate"             : rate,
            "strategy"         : strategy,
            "column"           : col,
            "col_type"         : col_type,
            "n_imputed"        : n_imp,
            "exact_matches"    : n_exact,
            "exact_match_rate" : round(em_rate, 6),
            "sem_sim_mean"     : round(sim_mean, 6) if not np.isnan(sim_mean) else float("nan"),
        })

    if not per_col_records:
        print("[imputation_eval] ⚠ No per-column results produced.")
        return {"available": False}

    df_per_col = pd.DataFrame(per_col_records)

    # ── Aggregate row ─────────────────────────────────────────────────────
    total_n     = int(df_per_col["n_imputed"].sum())
    total_exact = int(df_per_col["exact_matches"].sum())
    agg_em      = total_exact / total_n if total_n > 0 else float("nan")

    weights     = df_per_col["n_imputed"].values.astype(float)
    sim_arr     = df_per_col["sem_sim_mean"].values.astype(float)
    valid_sim   = ~np.isnan(sim_arr)
    agg_sem_sim = (float(np.average(sim_arr[valid_sim], weights=weights[valid_sim]))
                   if valid_sim.any() else float("nan"))

    agg_row = {
        "dataset"          : config.DATASET_NAME,
        "seed"             : config.RANDOM_SEED,
        "rate"             : rate,
        "strategy"         : strategy,
        "column"           : "_ALL_",
        "col_type"         : "aggregate",
        "n_imputed"        : total_n,
        "exact_matches"    : total_exact,
        "exact_match_rate" : round(agg_em, 6),
        "sem_sim_mean"     : round(agg_sem_sim, 6) if not np.isnan(agg_sem_sim) else float("nan"),
    }

    df_result = pd.concat(
        [df_per_col, pd.DataFrame([agg_row])],
        ignore_index=True,
    )

    # ── Build result dict (backward-compatible with existing callers) ─────
    result = {
        "available"      : True,
        "strategy"       : strategy,
        "strategy_label" : imp_label,
        "mcar_rate"      : rate,
        "dataset"        : config.DATASET_NAME,
        "seed"           : config.RANDOM_SEED,
        "source"         : config.SOURCE,
        "total_missing"  : total_missing,
        "exact_match"    : {
            "overall_correct" : total_exact,
            "overall_total"   : total_n,
            "overall_rate"    : round(agg_em, 4),
        },
        "semantic"       : {
            "available" : True,
            "overall"   : {
                "mean"  : round(agg_sem_sim, 4) if not np.isnan(agg_sem_sim) else None,
                "count" : total_n,
            },
            "method"    : ("sentence_embeddings"
                           if (compute_embeddings and _HAS_ST)
                           else "distance_and_string"),
        },
        "per_column"     : {
            row["column"]: {
                "type"    : row["col_type"],
                "correct" : int(row["exact_matches"]),
                "total"   : int(row["n_imputed"]),
                "rate"    : float(row["exact_match_rate"]),
                "sem_sim" : float(row["sem_sim_mean"])
                            if not np.isnan(row["sem_sim_mean"]) else None,
            }
            for _, row in df_per_col.iterrows()
        },
        "_df"            : df_result,   # DataFrame for CSV writing
    }

    _print_eval(result)
    _save_eval(result)
    _append_csv(df_result, append=append_csv)

    return result


def run_eval_pipeline(
    rates: list = None,
    strategies: list = None,
    append: bool = True,
    compute_embeddings: bool = True,
) -> str:
    """
    Evaluate all (rate, strategy) combinations and append results to
    RESULTS_DIR/imputation_evaluation.csv.

    Skips combos whose imputed CSV does not yet exist.
    Returns the path to the output CSV.
    """
    if rates is None:
        rates = config.MISSING_RATE
    if strategies is None:
        strategies = [s for s in config.IMPUTATION_STRATEGIES if s != "none"]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "imputation_evaluation.csv")

    print(f"\n{'='*65}")
    print(f"  IMPUTATION EVALUATION PIPELINE")
    print(f"  Dataset    : {config.DATASET_NAME}")
    print(f"  Seed       : {config.RANDOM_SEED}")
    print(f"  Rates      : {rates}%")
    print(f"  Strategies : {strategies}")
    print(f"  Embeddings : {'yes (sentence-transformers)' if (compute_embeddings and _HAS_ST) else 'no (distance + SequenceMatcher)'}")
    print(f"{'='*65}\n")

    first_write = not os.path.exists(out_path)
    n_done = 0

    for rate in rates:
        for strategy in strategies:
            imp_csv = config.imputed_csv(rate, strategy)
            if not os.path.exists(imp_csv):
                print(f"[imputation_eval] ⚠  Skipping rate={rate}%  "
                      f"strategy='{strategy}' — imputed CSV not found")
                continue

            print(f"\n── rate={rate}%  strategy='{strategy}' ──")
            try:
                result = evaluate_imputation(
                    rate=rate, strategy=strategy,
                    compute_embeddings=compute_embeddings,
                    append_csv=(append or not first_write),
                )
                if result.get("available"):
                    first_write = False
                    n_done += 1
            except Exception as exc:
                print(f"[imputation_eval] ✗ Error: {exc}")

    print(f"\n[imputation_eval] ✓ Done — {n_done} combos evaluated")
    if os.path.exists(out_path):
        print(f"[imputation_eval]   CSV → {out_path}\n")
    return out_path


# ── Internal: semantic similarity ────────────────────────────────────────────

def _compute_semantic(
    df_imputed, df_clean, missing_mask,
    feat_cols, cat_cols, num_cols,
    compute_embeddings,
) -> dict:
    """
    Returns {col: {"mean": float, "std": float}} for every feature column.
    Categorical: sentence embeddings (if available) else SequenceMatcher.
    Numerical:   1 – |error| / range.
    """
    scores = {}   # col → list of per-cell scores

    # ── Categorical ──────────────────────────────────────────────────────
    cat_feat = [c for c in feat_cols if c in cat_cols and c in missing_mask.columns]

    if cat_feat:
        if compute_embeddings and _HAS_ST:
            try:
                sem = semantic_similarity_at_positions(
                    df_imputed[cat_feat], df_clean[cat_feat],
                    missing_mask[cat_feat],
                    model_name=config.SENTI_TRANSFORMER,
                )
                for col in cat_feat:
                    col_scores = sem["cell_scores"][col].dropna().values
                    if len(col_scores):
                        scores[col] = {"mean": float(np.mean(col_scores)),
                                       "std":  float(np.std(col_scores))}
            except Exception as exc:
                print(f"[imputation_eval] Embedding similarity failed "
                      f"({exc}) — falling back to SequenceMatcher")
                _cat_seqmatch(df_imputed, df_clean, missing_mask, cat_feat, scores)
        else:
            _cat_seqmatch(df_imputed, df_clean, missing_mask, cat_feat, scores)

    # ── Numerical ────────────────────────────────────────────────────────
    num_feat = [c for c in feat_cols if c in num_cols and c in missing_mask.columns]
    for col in num_feat:
        null_mask_col = missing_mask[col].astype(bool)
        n_null = int(null_mask_col.sum())
        if n_null == 0:
            continue
        t_vals = pd.to_numeric(df_clean.loc[null_mask_col, col], errors="coerce").values
        i_vals = pd.to_numeric(df_imputed.loc[null_mask_col, col], errors="coerce").values
        col_range = df_clean[col].max() - df_clean[col].min()
        if not np.isfinite(col_range) or col_range == 0:
            col_range = 1.0
        sim_vals = np.clip(1.0 - np.abs(t_vals - i_vals) / col_range, 0.0, 1.0)
        scores[col] = {"mean": float(np.nanmean(sim_vals)),
                       "std":  float(np.nanstd(sim_vals))}

    return scores


def _cat_seqmatch(df_imputed, df_clean, missing_mask, cat_feat, scores):
    """Fallback: SequenceMatcher string similarity for categorical columns."""
    for col in cat_feat:
        null_mask_col = missing_mask[col].astype(bool)
        if not null_mask_col.any():
            continue
        imp_vals = df_imputed.loc[null_mask_col, col].astype(str).values
        tru_vals = df_clean.loc[null_mask_col, col].astype(str).values
        sim_vals = np.array([
            SequenceMatcher(None, a.strip(), b.strip()).ratio()
            for a, b in zip(imp_vals, tru_vals)
        ])
        scores[col] = {"mean": float(np.mean(sim_vals)),
                       "std":  float(np.std(sim_vals))}


# ── Internal: CSV output ─────────────────────────────────────────────────────

def _append_csv(df_result: pd.DataFrame, append: bool = True) -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path   = os.path.join(config.RESULTS_DIR, "imputation_evaluation.csv")
    write_header = (not append) or (not os.path.exists(out_path))
    write_mode   = "w" if write_header else "a"
    df_result.to_csv(out_path, mode=write_mode, header=write_header, index=False)
    print(f"[imputation_eval] ✓ CSV appended → {out_path}  "
          f"({len(df_result)} rows)")
    return out_path


# ── Internal: print summary ──────────────────────────────────────────────────

def _print_eval(result: dict) -> None:
    em  = result["exact_match"]
    sem = result["semantic"]

    print(f"\n{'='*70}")
    print(f"  IMPUTATION QUALITY  —  "
          f"rate={result['mcar_rate']}%  strategy='{result['strategy']}'")
    print(f"  Dataset  : {result['dataset']}   "
          f"Seed: {result.get('seed', config.RANDOM_SEED)}")
    print(f"{'='*70}")

    rate_val = em["overall_rate"]
    print(f"\n  EXACT MATCH  :  "
          f"{em['overall_correct']:,} / {em['overall_total']:,}  =  "
          f"{rate_val:.4f}  ({rate_val:.2%})")
    if sem.get("available") and sem["overall"].get("mean") is not None:
        print(f"  SEM SIM MEAN :  {sem['overall']['mean']:.4f}  "
              f"(method: {sem.get('method', 'n/a')})")

    per_col = result.get("per_column", {})
    if per_col:
        cat_entries = [(c, v) for c, v in per_col.items() if v["type"] == "categorical"]
        num_entries = [(c, v) for c, v in per_col.items() if v["type"] == "numerical"]

        if cat_entries:
            print(f"\n  ── Categorical columns ──")
            print(f"  {'Column':<25} {'exact_match':>11}  {'sem_sim':>7}  {'n':>6}")
            for col, v in sorted(cat_entries, key=lambda x: x[1]["rate"]):
                sim_s = f"{v['sem_sim']:.4f}" if v["sem_sim"] is not None else "   n/a"
                print(f"  {col:<25} {v['rate']:>11.4f}  {sim_s:>7}  "
                      f"{v['total']:>6,}")

        if num_entries:
            print(f"\n  ── Numerical columns ──")
            print(f"  {'Column':<25} {'sem_sim':>7}  {'n':>6}")
            for col, v in sorted(num_entries, key=lambda x: -(x[1]["sem_sim"] or 0)):
                sim_s = f"{v['sem_sim']:.4f}" if v["sem_sim"] is not None else "   n/a"
                print(f"  {col:<25} {sim_s:>7}  {v['total']:>6,}")
    print()


# ── Internal: JSON + TXT save ────────────────────────────────────────────────

def _save_eval(result: dict) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    prefix = config._FILE_PREFIX

    # Exclude DataFrame from JSON
    json_result = {k: v for k, v in result.items() if k != "_df"}

    json_path = os.path.join(config.RESULTS_DIR,
                             f"{prefix}_imputation_eval.json")
    with open(json_path, "w") as f:
        json.dump(json_result, f, indent=2, default=str)

    txt_path = os.path.join(config.RESULTS_DIR,
                            f"{prefix}_imputation_eval.txt")
    em  = result["exact_match"]
    sem = result["semantic"]
    lines = [
        "Imputation Quality Evaluation",
        f"Dataset   : {result['dataset']}",
        f"Seed      : {result.get('seed', config.RANDOM_SEED)}",
        f"Strategy  : {result['strategy']}  ({result['strategy_label']})",
        f"MCAR rate : {result['mcar_rate']}%",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "EXACT MATCH RATE",
        "-" * 50,
        f"  {em['overall_correct']:,} / {em['overall_total']:,} = {em['overall_rate']:.4f} ({em['overall_rate']:.2%})",
    ]
    if sem.get("available") and sem["overall"].get("mean") is not None:
        lines += [
            "",
            "SEMANTIC SIMILARITY",
            "-" * 50,
            f"  Mean: {sem['overall']['mean']:.4f}  "
            f"(method: {sem.get('method', 'n/a')})",
        ]

    per_col = result.get("per_column", {})
    if per_col:
        lines += ["", "PER-COLUMN BREAKDOWN", "-" * 70,
                  f"{'Column':<22} {'Type':<13} {'Correct':>8} "
                  f"{'Total':>8} {'ExactMatch':>10} {'SemSim':>7}"]
        sorted_cols = sorted(per_col.items(),
                             key=lambda x: x[1]["rate"])
        for col, info in sorted_cols:
            if info["total"] == 0:
                continue
            sim_s = f"{info['sem_sim']:.4f}" if info["sem_sim"] is not None else "   n/a"
            lines.append(
                f"{col:<22} {info['type']:<13} {info['correct']:>8} "
                f"{info['total']:>8} {info['rate']:>10.4f} {sim_s:>7}"
            )

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[imputation_eval] ✓ JSON → {os.path.basename(json_path)}")
    print(f"[imputation_eval] ✓ TXT  → {os.path.basename(txt_path)}")


# ── Standalone execution ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imputation Evaluation")
    parser.add_argument("--rates",        nargs="+", type=int, default=None,
                        help="MCAR rates (default: config.MISSING_RATE)")
    parser.add_argument("--strategies",   nargs="+",           default=None,
                        help="Strategies (default: all non-none)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Overwrite existing CSV instead of appending")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip sentence-embedding similarity (faster)")
    args = parser.parse_args()

    run_eval_pipeline(
        rates              = args.rates,
        strategies         = args.strategies,
        append             = not args.overwrite,
        compute_embeddings = not args.no_embeddings,
    )
