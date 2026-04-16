"""
run_batch.py
────────────────────────────────────────────────────────────────────────────
Runs the full GNN pipeline sequentially for multiple SOURCE configurations.

Instead of editing config.py and running main.py repeatedly, this script
loops through config.BATCH_SOURCES and processes each one automatically.

USAGE:
    python run_batch.py                          # run all sources in BATCH_SOURCES
    python run_batch.py --force-graph            # force rebuild graph files
    python run_batch.py --force-train            # force retrain models
    python run_batch.py --sources clean mcar_15  # run only specific sources

CONFIGURATION:
    Edit BATCH_SOURCES in config.py to change which sources are processed:

    BATCH_SOURCES = [
        ("clean",   "none"),
        ("mcar_15", "none"),
        ("mcar_30", "none"),
        ("mcar_50", "none"),
        ("mcar_15", "mode"),     # ← with imputation
        ("mcar_15", "knn"),      # ← different strategy
    ]

WORKFLOW PER SOURCE:
    1. config.reconfigure(source, imputation)  — update all paths
    2. Load data + build splits (shared, runs once)
    3. Null injection check
    4. Imputation (if needed)
    5. Build graph files (.npy)
    6. Load graph → PyG Data
    7. Build + train model
    8. Evaluate + save results
    9. Fairness analysis (demographic parity)

All results are saved to results/ with source-specific prefixes,
so nothing collides between runs.
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import config
from data_download import download_raw_data
from graph_builder import build_graph_files, load_graph_files
from model import build_model
from trainer import run_training, evaluate
from results import save_all_results, print_results
from imputation_eval import evaluate_imputation


def _banner(text: str, char: str = "─") -> None:
    print(f"\n{char*70}\n  {text}\n{char*70}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Batch GNN Pipeline Runner")
    parser.add_argument("--force-graph", action="store_true",
                        help="Rebuild .npy graph files for every source")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain model for every source")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (clean CSV must exist)")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Run only these sources (e.g. --sources clean mcar_15)")
    return parser.parse_args()


def run_single_source(source: str, imputation: str, device: torch.device,
                      force_graph: bool = False, force_train: bool = False) -> dict:
    """
    Run the full pipeline for a single (source, imputation) pair.
    Returns a summary dict with metrics.
    All GPU/CPU memory, model weights, tensors, and graph data are
    cleaned up before returning so the next run starts fresh.
    """
    t_start = time.time()

    # Local references we'll need to clean up
    model        = None
    data         = None
    X            = None
    y            = None
    edge_index   = None
    train_mask   = None
    val_mask     = None
    test_mask    = None
    history      = None
    eval_results = None
    meta         = None

    try:
        # ── Reconfigure all config paths ──────────────────────────────────────
        config.reconfigure(source, imputation)
        config.set_global_seed()

        imp_label = config.IMPUTATION_STRATEGIES.get(
            config.RESOLVED_IMPUTATION, {}
        ).get("label", config.RESOLVED_IMPUTATION)

        print(f"\n{'='*70}")
        print(f"  GNN Pipeline — {config.DATASET_NAME}")
        print(f"  SOURCE      : '{config.SOURCE}'")
        print(f"  Imputation  : {config.RESOLVED_IMPUTATION}  ({imp_label})")
        print(f"  Source CSV  : {config.SOURCE_CSV}")
        print(f"  Output dir  : {config.GRAPH_DIR}/")
        print(f"  Device      : {device}")
        print(f"{'='*70}")

        # ── STEP 3a: Imputation (if needed) ───────────────────────────────────
        _banner("Imputation")
        if config.RESOLVED_IMPUTATION != "none":
            if not os.path.exists(config.SOURCE_CSV):
                parts = config.SOURCE.split("_")
                rate  = int(parts[1])
                raw_csv = config.mcar_csv(rate)

                if not os.path.exists(raw_csv):
                    print(f"[batch] ✗ Raw MCAR CSV not found: {raw_csv}")
                    print(f"[batch]   Run null injection first.")
                    return {"source": source, "error": "raw MCAR CSV missing"}

                print(f"[batch] Creating imputed CSV from {raw_csv} ...")
                from imputer import impute
                df_raw     = pd.read_csv(raw_csv)
                df_imputed = impute(df_raw, strategy=config.RESOLVED_IMPUTATION)
                os.makedirs(config.GRAPH_DIR, exist_ok=True)
                df_imputed.to_csv(config.SOURCE_CSV, index=False)
                print(f"[batch] ✓ Saved → {config.SOURCE_CSV}")
            else:
                print(f"[batch] Imputed CSV exists: {config.SOURCE_CSV}")
        else:
            print(f"[batch] No imputation (SOURCE='{config.SOURCE}')")

        # ── STEP 3b: Imputation eval ──────────────────────────────────────────
        _banner("Imputation Quality Evaluation")
        imputation_eval_results = evaluate_imputation()

        # ── STEP 4: Build graph ───────────────────────────────────────────────
        _banner("Graph Construction")
        if force_graph:
            for f in config.NPY_FILES:
                if os.path.exists(f):
                    os.remove(f)

        build_graph_files()

        if not config.npy_files_exist():
            print(f"[batch] ✗ Graph files missing — aborting this source")
            return {"source": source, "error": "graph build failed"}

        # ── STEP 5: Load graph ────────────────────────────────────────────────
        _banner("Load Graph")
        X, y, edge_index, train_mask, val_mask, test_mask, meta = load_graph_files()

        data = Data(
            x          = torch.tensor(X, dtype=torch.float),
            edge_index = edge_index,
            y          = torch.tensor(y, dtype=torch.long),
            train_mask = train_mask,
            val_mask   = val_mask,
            test_mask  = test_mask,
        ).to(device)

        # ── STEP 6: Build model ───────────────────────────────────────────────
        _banner("Model")
        model = build_model(
            num_features=meta["num_features"],
            num_classes=meta["num_classes"],
        ).to(device)

        # ── STEP 7a: Train ────────────────────────────────────────────────────
        _banner("Training")
        if force_train and config.model_exists():
            os.remove(config.MODEL_F)

        history = run_training(model, data, device)

        # ── STEP 7b: Evaluate ─────────────────────────────────────────────────
        _banner("Evaluation")
        eval_results = evaluate(model, data, device)

        # ── STEP 7c: Save results ─────────────────────────────────────────────
        _banner("Results")
        print_results(eval_results)
        save_all_results(history, eval_results, meta)

        # ── Fairness ──────────────────────────────────────────────────────────
        _banner("Fairness — Demographic Parity")
        try:
            from fairness import (check_demographic_parity_pre,
                                  check_demographic_parity_post,
                                  plot_demographic_parity)

            dp_pre  = check_demographic_parity_pre(test_mask)
            dp_post = check_demographic_parity_post(eval_results)
            plot_demographic_parity(dp_pre, dp_post)
        except Exception as exc:
            print(f"[batch] Fairness analysis failed: {exc}")
            dp_pre = dp_post = None

        elapsed = time.time() - t_start

        # ── Build summary ─────────────────────────────────────────────────────
        summary = {
            "source"         : source,
            "imputation"     : config.RESOLVED_IMPUTATION,
            "test_acc"       : round(eval_results["test"]["acc"], 4),
            "test_f1"        : round(eval_results["test"]["f1_macro"], 4),
            "test_auc"       : round(eval_results["test"]["roc_auc"], 4)
                               if eval_results["test"].get("roc_auc") else None,
            "dp_pre"         : dp_pre["dp_difference"] if dp_pre else None,
            "dp_post"        : dp_post["dp_difference"] if dp_post else None,
            "elapsed_sec"    : round(elapsed, 1),
        }

        print(f"\n[batch] ✓ '{source}' complete in {elapsed:.1f}s")
        print(f"[batch]   Test Acc={summary['test_acc']:.4f}  "
              f"F1={summary['test_f1']:.4f}  "
              f"AUC={summary['test_auc']}  "
              f"DP_post={summary['dp_post']}")

        return summary

    finally:
        # ── CLEANUP — runs after every source, success or failure ─────────────
        _cleanup(model, data, X, y, edge_index, train_mask, val_mask, test_mask,
                 device, source)


def _cleanup(model, data, X, y, edge_index, train_mask, val_mask, test_mask,
             device, source):
    """
    Aggressively free all memory after a pipeline run:
      1. Delete model weights from GPU/CPU
      2. Delete PyG Data object and all tensors
      3. Delete numpy arrays (features, labels)
      4. Close all matplotlib figures
      5. Run Python garbage collector
      6. Flush CUDA cache (if GPU)

    This ensures each run starts with a completely clean state —
    no leftover tensors, no accumulated GPU memory, no stale graphs.
    """
    import gc
    import matplotlib.pyplot as plt

    print(f"\n[cleanup] Cleaning up after SOURCE='{source}' ...")

    # ── 1. Delete model ───────────────────────────────────────────────────────
    if model is not None:
        model.cpu()       # move off GPU first
        del model
        print(f"[cleanup]   ✓ Model deleted")

    # ── 2. Delete PyG Data and tensors ────────────────────────────────────────
    if data is not None:
        data.cpu()
        del data
    for tensor_name, tensor_obj in [("edge_index", edge_index),
                                     ("train_mask", train_mask),
                                     ("val_mask",   val_mask),
                                     ("test_mask",  test_mask)]:
        if tensor_obj is not None:
            del tensor_obj
    print(f"[cleanup]   ✓ PyG Data and tensors deleted")

    # ── 3. Delete numpy arrays ────────────────────────────────────────────────
    if X is not None:
        del X
    if y is not None:
        del y
    print(f"[cleanup]   ✓ Numpy arrays deleted")

    # ── 4. Close all matplotlib figures ───────────────────────────────────────
    plt.close("all")
    print(f"[cleanup]   ✓ Matplotlib figures closed")

    # ── 5. Python garbage collection ──────────────────────────────────────────
    n_collected = gc.collect()
    print(f"[cleanup]   ✓ Garbage collected ({n_collected} objects freed)")

    # ── 6. CUDA cache ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        mem_resrv = torch.cuda.memory_reserved(device)  / 1024**2
        print(f"[cleanup]   ✓ CUDA cache flushed  "
              f"(allocated={mem_alloc:.1f}MB  reserved={mem_resrv:.1f}MB)")

    print(f"[cleanup] ✓ Cleanup complete — ready for next source\n")


def main():
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── STEP 1: Data download (shared, runs once) ─────────────────────────────
    if not args.skip_download:
        _banner("STEP 1 — Load Data + Build Splits (shared)", "═")
        download_raw_data()

    # ── Determine which sources to run ────────────────────────────────────────
    if args.sources:
        # Map source names to (source, imputation) pairs from BATCH_SOURCES
        batch_map = {s: (s, imp) for s, imp in config.BATCH_SOURCES}
        sources_to_run = []
        for s in args.sources:
            if s in batch_map:
                sources_to_run.append(batch_map[s])
            else:
                # Assume "none" imputation if not in BATCH_SOURCES
                sources_to_run.append((s, "none"))
    else:
        sources_to_run = config.BATCH_SOURCES

    n_total = len(sources_to_run)

    print(f"\n{'═'*70}")
    print(f"  BATCH PIPELINE — {config.DATASET_NAME}")
    print(f"  Device  : {device}")
    print(f"  Sources : {n_total}")
    for i, (src, imp) in enumerate(sources_to_run, 1):
        print(f"    {i}. SOURCE='{src}'  IMPUTATION='{imp}'")
    print(f"{'═'*70}")

    # ── STEP 2: Create ALL null-injected datasets upfront ─────────────────────
    _create_all_null_datasets(sources_to_run)

    # ── Run each source sequentially ──────────────────────────────────────────
    all_summaries = []
    t_total_start = time.time()

    for i, (source, imputation) in enumerate(sources_to_run, 1):
        print(f"\n\n{'█'*70}")
        print(f"  BATCH RUN {i} / {n_total}  —  SOURCE='{source}'  IMPUTATION='{imputation}'")
        print(f"{'█'*70}")

        try:
            summary = run_single_source(
                source, imputation, device,
                force_graph=args.force_graph,
                force_train=args.force_train,
            )
            all_summaries.append(summary)
        except Exception as exc:
            print(f"\n[batch] ✗ FAILED: SOURCE='{source}' — {exc}")
            import traceback
            traceback.print_exc()
            all_summaries.append({
                "source": source, "imputation": imputation,
                "error": str(exc),
            })

    total_elapsed = time.time() - t_total_start

    # ── Final comparison table ────────────────────────────────────────────────
    _print_comparison_table(all_summaries)

    # ── Save batch summary ────────────────────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    batch_json = os.path.join(config.RESULTS_DIR,
                              f"{config.DATASET_NAME}_batch_summary.json")
    with open(batch_json, "w") as f:
        json.dump({
            "dataset"       : config.DATASET_NAME,
            "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_elapsed" : round(total_elapsed, 1),
            "runs"          : all_summaries,
        }, f, indent=2)
    print(f"\n[batch] ✓ Batch summary saved → {batch_json}")

    print(f"\n{'═'*70}")
    print(f"  BATCH COMPLETE — {n_total} sources in {total_elapsed:.1f}s")
    print(f"{'═'*70}\n")


def _create_all_null_datasets(sources_to_run: list) -> None:
    """
    Create ALL null-injected (MCAR) datasets UPFRONT before any pipeline
    runs start.  This way the null injection happens once, and each
    subsequent pipeline run just checks if the files exist.

    For each MCAR rate found in sources_to_run:
      - If the null-injected train CSV already exists → skip with message
      - If not → run null injection for that rate

    Null injection is cumulative (15% → 30% → 50%), so we process
    all needed rates in ascending order in a single pass.
    """
    # Collect unique MCAR rates needed
    needed_rates = set()
    for source, _ in sources_to_run:
        if source == "clean":
            continue
        parts = source.split("_")
        if len(parts) >= 2 and parts[0] == "mcar":
            try:
                needed_rates.add(int(parts[1]))
            except ValueError:
                pass

    if not needed_rates:
        print(f"\n[batch] No MCAR sources requested — skipping null injection")
        return

    needed_rates = sorted(needed_rates)

    _banner("STEP 2 — Null Injection (all MCAR rates upfront)", "═")
    print(f"[batch] MCAR rates needed: {needed_rates}%")

    # Check which rates already have files
    rates_to_create = []
    for rate in needed_rates:
        csv_path = config.mcar_train_csv(rate)
        if os.path.exists(csv_path):
            size_kb = os.path.getsize(csv_path) / 1024
            print(f"[batch]   ✓ {rate}% MCAR already exists: {csv_path}  ({size_kb:.0f} KB) — skipping")
        else:
            rates_to_create.append(rate)
            print(f"[batch]   ○ {rate}% MCAR not found: {csv_path} — will create")

    if not rates_to_create:
        print(f"\n[batch] ✓ All null-injected datasets already exist — nothing to create")
        return

    # Load the clean training split
    if not os.path.exists(config.TRAIN_CSV_FILE):
        print(f"[batch] ✗ Train CSV not found: {config.TRAIN_CSV_FILE}")
        print(f"[batch]   Run data_download.py first.")
        sys.exit(1)

    clean_train_df = pd.read_csv(config.TRAIN_CSV_FILE)
    print(f"\n[batch] Loaded clean training split: {clean_train_df.shape}")
    print(f"[batch] Creating null-injected datasets for rates: {rates_to_create}%")

    # Temporarily set MISSING_RATE to only the rates we need
    # (null_injector uses config.MISSING_RATE internally)
    original_rates = config.MISSING_RATE

    # We need ALL rates up to the max needed (because injection is cumulative)
    # e.g., if we need 30% and 50%, we must also create 15% first
    all_rates_ascending = sorted(set(config.MISSING_RATE) | set(rates_to_create))
    max_needed = max(rates_to_create)
    rates_for_injection = [r for r in all_rates_ascending if r <= max_needed]

    config.MISSING_RATE = rates_for_injection

    from null_injector import run_null_injection
    run_null_injection(clean_train_df, run_impute=False)

    # Restore original rates
    config.MISSING_RATE = original_rates

    # Final verification
    print(f"\n[batch] ── Null injection verification ──")
    for rate in needed_rates:
        csv_path = config.mcar_train_csv(rate)
        if os.path.exists(csv_path):
            size_kb = os.path.getsize(csv_path) / 1024
            n_rows  = len(pd.read_csv(csv_path))
            feat_cols = [c for c in pd.read_csv(csv_path, nrows=0).columns
                         if c != config.TARGET_COL]
            n_nulls = pd.read_csv(csv_path)[feat_cols].isnull().sum().sum()
            print(f"[batch]   ✓ {rate}% MCAR: {csv_path}  "
                  f"({n_rows:,} rows, {n_nulls:,} null cells, {size_kb:.0f} KB)")
        else:
            print(f"[batch]   ✗ {rate}% MCAR: MISSING — {csv_path}")

    print(f"[batch] ✓ All null datasets ready\n")


def _print_comparison_table(summaries: list) -> None:
    """Print a side-by-side comparison of all runs."""
    print(f"\n\n{'═'*85}")
    print(f"  COMPARISON TABLE — ALL RUNS")
    print(f"{'═'*85}")
    print(f"  {'Source':<20} {'Imputation':<12} {'Test Acc':>9} {'Test F1':>9} "
          f"{'Test AUC':>9} {'DP Pre':>8} {'DP Post':>8} {'Time':>7}")
    print(f"  {'─'*80}")

    for s in summaries:
        if "error" in s:
            print(f"  {s['source']:<20} {s.get('imputation','?'):<12} "
                  f"{'FAILED':>9}  {s['error']}")
            continue

        acc  = f"{s['test_acc']:.4f}"
        f1   = f"{s['test_f1']:.4f}"
        auc  = f"{s['test_auc']:.4f}" if s['test_auc'] else "N/A"
        dp_pre  = f"{s['dp_pre']:.4f}" if s['dp_pre'] is not None else "N/A"
        dp_post = f"{s['dp_post']:.4f}" if s['dp_post'] is not None else "N/A"
        elapsed = f"{s['elapsed_sec']:.0f}s"

        print(f"  {s['source']:<20} {s['imputation']:<12} "
              f"{acc:>9} {f1:>9} {auc:>9} {dp_pre:>8} {dp_post:>8} {elapsed:>7}")

    print(f"{'═'*85}")


if __name__ == "__main__":
    main()
