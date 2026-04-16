"""
imputation_eval.py
────────────────────────────────────────────────────────────────────────────
Evaluates imputation quality by comparing imputed values to the ground
truth (clean data) at positions that were originally missing.

WHEN IS THIS CALLED?
    After imputation (Step 3) and before graph construction (Step 4).
    Only runs when SOURCE involves imputation (e.g. "mcar_10_mode").
    Skipped when SOURCE="clean" (no imputation to evaluate).

WHAT DOES IT MEASURE?
    1. Exact Match Rate — fraction of imputed cells that perfectly match
       the original clean value (best for categorical columns).
    2. Semantic Similarity — cosine similarity between sentence embeddings
       of imputed vs ground-truth cell values (useful for categorical
       columns where near-misses matter, e.g. "Bachelors" vs "Masters").
       Only computed when SENTI is the imputation strategy (since the
       sentence-transformers model is already available).

HOW TO READ THE RESULTS?
    Exact Match = 1.0 means every imputed value is identical to ground truth.
    Exact Match = 0.5 means half the imputed values are correct.
    Semantic Sim = 1.0 means imputed and ground truth are semantically identical.
    Semantic Sim > 0.8 means imputed values are very close in meaning.
    Semantic Sim < 0.5 means imputed values are semantically far off.

    Per-column breakdown shows which features the imputer handles well
    and which it struggles with — this helps you decide if a different
    imputation strategy might work better for specific columns.

Usage:
    from imputation_eval import evaluate_imputation
    eval_results = evaluate_imputation()   # auto-detects everything from config
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
from imputation_eval_metrics import (
    record_missing_positions,
    exact_match_at_positions,
    semantic_similarity_at_positions,
)


# =====================================================================
# PUBLIC API
# =====================================================================

def evaluate_imputation() -> dict:
    """
    Evaluate imputation quality by comparing imputed CSV to clean CSV.

    Automatically determines:
        - Which imputation strategy was used (from config.RESOLVED_IMPUTATION)
        - Where the missing values were (from the raw MCAR CSV)
        - What the ground truth is (from the clean CSV)

    Returns a dict with evaluation results, or {"available": False} if
    evaluation cannot be performed (e.g. SOURCE="clean").
    """
    # ── Guard: only evaluate when imputation was actually performed ────
    if config.RESOLVED_IMPUTATION == "none" or config.SOURCE == "clean":
        print("[imputation_eval] Skipped — no imputation to evaluate "
              f"(SOURCE='{config.SOURCE}', imputation='{config.RESOLVED_IMPUTATION}')")
        return {"available": False}

    # ── Parse MCAR rate from SOURCE string ────────────────────────────
    parts = config.SOURCE.split("_")
    if len(parts) < 2:
        print(f"[imputation_eval] Cannot parse MCAR rate from SOURCE='{config.SOURCE}'")
        return {"available": False}

    try:
        mcar_rate = int(parts[1])
    except ValueError:
        print(f"[imputation_eval] Cannot parse MCAR rate from SOURCE='{config.SOURCE}'")
        return {"available": False}

    # ── Load the three DataFrames we need ─────────────────────────────
    # Ground truth is the clean TRAINING split — same rows as the raw MCAR
    # training file.  Using the full clean CSV would cause a row-count mismatch
    # because null injection now operates on the training split only.
    clean_csv = config.TRAIN_CSV_FILE
    raw_csv   = config.mcar_csv(mcar_rate)   # alias → mcar_train_csv
    imp_csv   = config.SOURCE_CSV

    for label, path in [("Clean train CSV", clean_csv), ("Raw MCAR train CSV", raw_csv),
                        ("Imputed CSV", imp_csv)]:
        if not os.path.exists(path):
            print(f"[imputation_eval] {label} not found: {path}")
            return {"available": False}

    df_clean   = pd.read_csv(clean_csv)
    df_raw     = pd.read_csv(raw_csv)
    df_imputed = pd.read_csv(imp_csv)

    strategy = config.RESOLVED_IMPUTATION
    imp_label = config.IMPUTATION_STRATEGIES.get(strategy, {}).get("label", strategy)

    print(f"\n[imputation_eval] Evaluating imputation quality")
    print(f"  Strategy         : {strategy}  ({imp_label})")
    print(f"  MCAR rate        : {mcar_rate}%")
    print(f"  Clean train CSV  : {clean_csv}  (ground truth)")
    print(f"  Raw MCAR train   : {raw_csv}")
    print(f"  Imputed CSV      : {imp_csv}")

    # ── Record which positions were missing ───────────────────────────
    missing_mask = record_missing_positions(df_raw)

    # Exclude the target column from evaluation
    feat_cols = [c for c in df_clean.columns if c != config.TARGET_COL]
    missing_mask_feat = missing_mask[feat_cols]
    total_missing = int(missing_mask_feat.sum().sum())
    total_cells   = int(missing_mask_feat.size)
    print(f"  Missing cells : {total_missing:,} / {total_cells:,} "
          f"({total_missing/total_cells*100:.1f}%)")

    if total_missing == 0:
        print("[imputation_eval] No missing cells found — nothing to evaluate")
        return {"available": False}

    # ── 1. Exact Match ────────────────────────────────────────────────
    exact = exact_match_at_positions(
        df_imputed[feat_cols], df_clean[feat_cols], missing_mask_feat
    )

    # ── 2. Semantic Similarity (only for SENTI strategy) ──────────────
    semantic = {"available": False}
    if strategy == "senti":
        try:
            sem_result = semantic_similarity_at_positions(
                df_imputed[feat_cols], df_clean[feat_cols], missing_mask_feat,
                model_name=config.SENTI_TRANSFORMER,
            )
            semantic = {
                "available"  : True,
                "overall"    : sem_result["overall"],
                "per_column" : sem_result["per_column"].to_dict(orient="index"),
            }
        except Exception as exc:
            print(f"[imputation_eval] Semantic similarity failed: {exc}")
            semantic = {"available": False, "error": str(exc)}

    # ── Identify categorical vs numerical columns ─────────────────────
    cat_cols = [c for c in feat_cols
                if df_clean[c].dtype == object
                or pd.api.types.is_categorical_dtype(df_clean[c])]
    num_cols = [c for c in feat_cols if c in config.NUMERICAL_COLS]

    # ── Per-column breakdown with column type ─────────────────────────
    per_col_results = {}
    for col in feat_cols:
        col_data = exact["per_column"]
        if col in col_data.index:
            row = col_data.loc[col]
            col_type = "categorical" if col in cat_cols else "numerical" if col in num_cols else "other"
            per_col_results[col] = {
                "type"    : col_type,
                "correct" : int(row["true"]),
                "total"   : int(row["total"]),
                "rate"    : round(float(row["rate"]), 4) if not np.isnan(row["rate"]) else None,
            }

    # ── Build result dict ─────────────────────────────────────────────
    result = {
        "available"     : True,
        "strategy"      : strategy,
        "strategy_label": imp_label,
        "mcar_rate"     : mcar_rate,
        "dataset"       : config.DATASET_NAME,
        "source"        : config.SOURCE,
        "total_missing" : total_missing,
        "exact_match"   : {
            "overall_correct" : exact["overall"]["true"],
            "overall_total"   : exact["overall"]["total"],
            "overall_rate"    : round(exact["overall"]["rate"], 4),
        },
        "semantic"      : semantic,
        "per_column"    : per_col_results,
    }

    # ── Print results ─────────────────────────────────────────────────
    _print_eval(result)

    # ── Save results ──────────────────────────────────────────────────
    _save_eval(result)

    return result


# =====================================================================
# INTERNAL — Print with explanations
# =====================================================================

def _print_eval(result: dict) -> None:
    """Print imputation evaluation results with explanations."""
    strategy = result["strategy"]
    imp_label = result["strategy_label"]
    em = result["exact_match"]

    print(f"\n{'='*70}")
    print(f"  IMPUTATION QUALITY EVALUATION")
    print(f"  Dataset   : {config.DATASET_NAME}  (SOURCE='{config.SOURCE}')")
    print(f"  Strategy  : {strategy}  ({imp_label})")
    print(f"  MCAR rate : {result['mcar_rate']}%")
    print(f"{'='*70}")

    print(f"\n  PURPOSE: This evaluates how well the imputation strategy recovered")
    print(f"  the original (clean) values at positions where data was artificially")
    print(f"  removed (MCAR injection). Higher scores mean the imputer is better")
    print(f"  at guessing the true values, which means downstream models (like the")
    print(f"  GNN) receive more accurate input features.")

    # ── Overall exact match ───────────────────────────────────────────
    rate = em["overall_rate"]
    print(f"\n  {'─'*60}")
    print(f"  EXACT MATCH RATE")
    print(f"  {'─'*60}")
    print(f"  Overall : {em['overall_correct']:,} / {em['overall_total']:,} = {rate:.2%}")
    print(f"\n  What: Fraction of imputed cells that exactly match the original")
    print(f"        clean value. Works best for categorical columns (e.g. 'Male'")
    print(f"        either matches or it doesn't) and integer numericals.")

    if rate >= 0.8:
        print(f"  → Excellent — the imputer recovers ≥80% of missing values correctly.")
    elif rate >= 0.5:
        print(f"  → Moderate — the imputer gets about half of the values right.")
        print(f"    Consider trying a more sophisticated strategy (e.g. SENTI or MICE).")
    else:
        print(f"  → Low — the imputer struggles with this data/rate combination.")
        print(f"    The GNN's input features will contain significant noise from")
        print(f"    imputation errors. Consider a different strategy or lower MCAR rate.")

    # ── Semantic similarity (if available) ────────────────────────────
    if result["semantic"].get("available"):
        sem = result["semantic"]
        mean_sim = sem["overall"]["mean"]
        print(f"\n  {'─'*60}")
        print(f"  SEMANTIC SIMILARITY (sentence embeddings)")
        print(f"  {'─'*60}")
        print(f"  Overall mean cosine similarity: {mean_sim:.4f}")
        print(f"\n  What: Even when values don't exactly match, they might be close")
        print(f"        in meaning (e.g. 'Bachelors' vs 'Some-college'). This uses")
        print(f"        the {config.SENTI_TRANSFORMER} model to measure meaning-level")
        print(f"        closeness (1.0 = identical meaning, 0.0 = unrelated).")

        if mean_sim >= 0.9:
            print(f"  → Excellent — imputed values are semantically very close to ground truth.")
        elif mean_sim >= 0.7:
            print(f"  → Good — imputed values capture the right general meaning.")
        else:
            print(f"  → Weak — imputed values diverge semantically from ground truth.")

    # ── Per-column breakdown ──────────────────────────────────────────
    per_col = result["per_column"]
    if per_col:
        print(f"\n  {'─'*60}")
        print(f"  PER-COLUMN EXACT MATCH BREAKDOWN")
        print(f"  {'─'*60}")
        print(f"  {'Column':<22} {'Type':<13} {'Correct':>8} {'Total':>8} {'Rate':>8}")
        print(f"  {'-'*62}")

        # Sort: worst performing first so user sees problems immediately
        sorted_cols = sorted(per_col.items(),
                             key=lambda x: x[1]["rate"] if x[1]["rate"] is not None else 1.0)

        for col, info in sorted_cols:
            if info["total"] == 0:
                continue
            rate_str = f"{info['rate']:.2%}" if info["rate"] is not None else "N/A"
            marker = " ←" if info["rate"] is not None and info["rate"] < 0.3 else ""
            print(f"  {col:<22} {info['type']:<13} {info['correct']:>8} "
                  f"{info['total']:>8} {rate_str:>8}{marker}")

        # Call out problem columns
        problem_cols = [col for col, info in per_col.items()
                        if info["rate"] is not None and info["rate"] < 0.3 and info["total"] > 0]
        if problem_cols:
            print(f"\n  ← Columns with <30% exact match may need attention.")
            print(f"    Low-match columns: {', '.join(problem_cols)}")
            print(f"    These columns may introduce noise into the GNN features.")

    print()


# =====================================================================
# INTERNAL — Save results
# =====================================================================

def _save_eval(result: dict) -> None:
    """Save imputation evaluation results to JSON and TXT."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    prefix = config._FILE_PREFIX

    # ── JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(config.RESULTS_DIR, f"{prefix}_imputation_eval.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[imputation_eval] Saved {os.path.basename(json_path)}")

    # ── TXT report ────────────────────────────────────────────────────
    txt_path = os.path.join(config.RESULTS_DIR, f"{prefix}_imputation_eval.txt")
    lines = []

    lines.append(f"Imputation Quality Evaluation")
    lines.append(f"Dataset   : {config.DATASET_NAME}  (SOURCE='{config.SOURCE}')")
    lines.append(f"Strategy  : {result['strategy']}  ({result['strategy_label']})")
    lines.append(f"MCAR rate : {result['mcar_rate']}%")
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    lines.append(f"\nPURPOSE:")
    lines.append(f"  This evaluates how well the '{result['strategy']}' imputation strategy")
    lines.append(f"  recovered the original clean values at positions where data was")
    lines.append(f"  artificially removed ({result['mcar_rate']}% MCAR injection).")
    lines.append(f"  Higher scores = better imputation = more accurate GNN input features.")

    em = result["exact_match"]
    lines.append(f"\nEXACT MATCH RATE")
    lines.append(f"{'─'*50}")
    lines.append(f"  Overall: {em['overall_correct']:,} / {em['overall_total']:,} = {em['overall_rate']:.2%}")
    lines.append(f"  What: Fraction of imputed cells that exactly match the original clean value.")

    if result["semantic"].get("available"):
        sem = result["semantic"]
        lines.append(f"\nSEMANTIC SIMILARITY")
        lines.append(f"{'─'*50}")
        lines.append(f"  Overall mean cosine similarity: {sem['overall']['mean']:.4f}")
        lines.append(f"  What: Meaning-level closeness using sentence embeddings (1.0 = identical).")

    per_col = result.get("per_column", {})
    if per_col:
        lines.append(f"\nPER-COLUMN BREAKDOWN (sorted by rate, worst first)")
        lines.append(f"{'─'*70}")
        lines.append(f"{'Column':<22} {'Type':<13} {'Correct':>8} {'Total':>8} {'Rate':>8}")
        lines.append("-" * 62)

        sorted_cols = sorted(per_col.items(),
                             key=lambda x: x[1]["rate"] if x[1]["rate"] is not None else 1.0)
        for col, info in sorted_cols:
            if info["total"] == 0:
                continue
            rate_str = f"{info['rate']:.2%}" if info["rate"] is not None else "N/A"
            lines.append(f"{col:<22} {info['type']:<13} {info['correct']:>8} "
                         f"{info['total']:>8} {rate_str:>8}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[imputation_eval] Saved {os.path.basename(txt_path)}")
