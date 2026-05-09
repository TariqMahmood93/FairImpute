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
        _banner("Fairness — Demographic Parity + Extended Metrics")
        try:
            from fairness import (check_demographic_parity_pre,
                                  check_demographic_parity_post,
                                  check_fairness_metrics_post,
                                  plot_demographic_parity)

            dp_pre   = check_demographic_parity_pre(test_mask)
            dp_post  = check_demographic_parity_post(eval_results)
            fair_ext = check_fairness_metrics_post(eval_results)
            plot_demographic_parity(dp_pre, dp_post)
        except Exception as exc:
            print(f"[batch] Fairness analysis failed: {exc}")
            dp_pre = dp_post = fair_ext = None

        elapsed = time.time() - t_start

        # ── Build summary ─────────────────────────────────────────────────────
        summary = {
            "seed"                : config.RANDOM_SEED,
            "source"              : source,
            "imputation"          : config.RESOLVED_IMPUTATION,
            "test_acc"            : round(eval_results["test"]["acc"], 4),
            "test_f1"             : round(eval_results["test"]["f1_macro"], 4),
            "test_auc"            : round(eval_results["test"]["roc_auc"], 4)
                                    if eval_results["test"].get("roc_auc") else None,
            "dp_pre"              : dp_pre["dp_difference"]  if dp_pre  else None,
            "dp_post"             : dp_post["dp_difference"] if dp_post else None,
            # Per-group selection rates — {"Male": 0.114, "Female": 0.114}
            "dp_pre_group_rates"  : dp_pre["group_rates"]    if dp_pre  else {},
            "dp_post_group_rates" : dp_post["group_rates"]   if dp_post else {},
            # Extended fairness (post-GNN only)
            "tpr_parity_gap"         : fair_ext["tpr_parity_gap"]         if fair_ext else None,
            "fpr_parity_gap"         : fair_ext["fpr_parity_gap"]         if fair_ext else None,
            "equal_opportunity_diff" : fair_ext["equal_opportunity_diff"] if fair_ext else None,
            "equalized_odds_diff"    : fair_ext["equalized_odds_diff"]    if fair_ext else None,
            "tpr_by_group"           : fair_ext["tpr_by_group"]           if fair_ext else {},
            "fpr_by_group"           : fair_ext["fpr_by_group"]           if fair_ext else {},
            "elapsed_sec"         : round(elapsed, 1),
        }

        print(f"\n[batch] ✓ '{source}' complete in {elapsed:.1f}s")
        print(f"[batch]   Test Acc={summary['test_acc']:.4f}  "
              f"F1={summary['test_f1']:.4f}  "
              f"AUC={summary['test_auc']}  "
              f"DP_post={summary['dp_post']}  "
              f"EqOdds={summary['equalized_odds_diff']}  "
              f"EqOpp={summary['equal_opportunity_diff']}")

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


def _wipe_between_seeds(sources_to_run: list, stage: str = "between seeds") -> None:
    """
    Delete ALL artifacts from a prior seed's run so the next seed starts from
    a clean slate.  Called at the start of every seed iteration — this makes
    seed 1 behave identically to seeds 2..N (otherwise seed 1 could silently
    reuse stale splits/MCAR files left over from a previous run).

    `stage` is just a label for the banner, e.g. "pre-wipe (seed 1)" or
    "between seeds".

    Wiped:
      • train / val / test / balanced split CSVs  (will be re-derived from
        the clean CSV using the new seed)
      • every MCAR folder (<dataset>_mcar_*pct/)
      • every imputed folder (<dataset>_mcar_*pct_<strategy>/)
      • any per-source results files (*_summary.json, *_dp_*.json,
        *_fairness_ext.json, *_training_curves.png, *_confusion_matrix.png,
        *_classification_report.txt, *_demographic_parity.png, *.log)

    Kept:
      • the dataset's clean CSV  (<dataset>_clean.csv)
      • the cross-seed summary CSV/JSON produced by main()
    """
    import glob
    import shutil

    _banner(f"Cleanup ({stage}) — deleting splits, MCAR folders, "
            f"imputed folders, and per-source results", "═")

    # ── 1. Split CSVs + balanced CSV  (regenerated from clean CSV) ────────────
    for path in [
        config.TRAIN_CSV_FILE,
        config.VAL_CSV_FILE,
        config.TEST_CSV_FILE,
        os.path.join(config.DATA_DIR, f"{config.DATASET_NAME}_balanced.csv"),
    ]:
        if os.path.exists(path):
            os.remove(path)
            print(f"[wipe]   ✓ removed {path}")

    # ── 2. Every MCAR and imputed folder for this dataset ─────────────────────
    # Since paths now embed the seed, we need to sweep folders for ALL seeds,
    # not just the current one.
    folders_to_remove = set()
    all_seeds = list(config.SEEDS)
    for source, imputation in sources_to_run:
        if source == "clean":
            continue
        parts = source.split("_")
        if len(parts) >= 2 and parts[0] == "mcar":
            try:
                rate = int(parts[1])
            except ValueError:
                continue
            for s in all_seeds:
                folders_to_remove.add(os.path.join(config.DATA_DIR, f"{s}_mcar_{rate}"))
                if imputation and imputation != "none":
                    folders_to_remove.add(os.path.join(config.DATA_DIR, f"{s}_mcar_{rate}_{imputation}"))
                if len(parts) >= 3:
                    strategy = "_".join(parts[2:])
                    folders_to_remove.add(os.path.join(config.DATA_DIR, f"{s}_mcar_{rate}_{strategy}"))

    # Safety net: sweep anything that matches the pattern, even if not listed
    for pattern in [os.path.join(config.DATA_DIR, "*_mcar_*"),
                    os.path.join(config.DATA_DIR, "*_mcar_*_*")]:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                folders_to_remove.add(path)

    for folder in sorted(folders_to_remove):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"[wipe]   ✓ removed folder {folder}/")

    # ── 3. Per-source results files  ──────────────────────────────────────────
    # Keep the aggregated cross-seed summary files produced by main().
    if os.path.isdir(config.RESULTS_DIR):
        keep_suffixes = (
            "_batch_summary.csv",      # cross-seed aggregate
            "_batch_summary.json",     # cross-seed aggregate
        )
        for fname in os.listdir(config.RESULTS_DIR):
            if fname.endswith(keep_suffixes):
                continue
            path = os.path.join(config.RESULTS_DIR, fname)
            if os.path.isfile(path):
                os.remove(path)
                print(f"[wipe]   ✓ removed {path}")

    print(f"[wipe] ✓ Ready for next seed\n")



def main():
    args   = _parse_args()

    # ── Clear __pycache__ to prevent stale .pyc issues ────────────────────
    import shutil
    pycache = os.path.join(os.path.dirname(__file__), "__pycache__")
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)
        print(f"[batch] Cleared __pycache__ to ensure fresh imports")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Determine which sources to run (same list used for every seed) ────────
    if args.sources:
        batch_map = {s: (s, imp) for s, imp in config.BATCH_SOURCES}
        sources_to_run = []
        for s in args.sources:
            if s in batch_map:
                sources_to_run.append(batch_map[s])
            else:
                sources_to_run.append((s, "none"))
    else:
        sources_to_run = config.BATCH_SOURCES

    n_total = len(sources_to_run)
    seeds   = list(config.SEEDS)

    print(f"\n{'═'*70}")
    print(f"  MULTI-SEED BATCH PIPELINE — {config.DATASET_NAME}")
    print(f"  Device  : {device}")
    print(f"  Seeds   : {seeds}   ({len(seeds)} total)")
    print(f"  Sources : {n_total}  per seed")
    for i, (src, imp) in enumerate(sources_to_run, 1):
        print(f"    {i}. SOURCE='{src}'  IMPUTATION='{imp}'")
    print(f"  Total runs : {len(seeds) * n_total}")
    print(f"{'═'*70}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — DATA PREPARATION (all seeds upfront)
    #
    # For EVERY seed, create the seed-specific train/val/test splits and
    # null-injected MCAR training CSVs.  Because folder and file names
    # embed the seed (e.g. adult_census_mcar_10pct_seed42/), files from
    # different seeds coexist on disk without colliding.
    #
    # After this phase, every null file the pipeline will ever need is on
    # disk — the experiment phase just reads them.
    # ══════════════════════════════════════════════════════════════════════════
    _banner("PHASE 1 — DATA PREPARATION  (splits + null injection for ALL seeds)", "═")

    from null_injector import run_null_injection_all_seeds
    run_null_injection_all_seeds(run_impute=False)

    # ── Show all MCAR files on disk ───────────────────────────────────────
    import glob as _glob
    mcar_files = sorted(_glob.glob(
        os.path.join(config.DATA_DIR, "*_mcar_*", "train_*_mcar_*.csv")
    ))
    if mcar_files:
        print(f"\n[phase1] Null-injected files on disk ({len(mcar_files)} total):")
        for f in mcar_files:
            size_kb = os.path.getsize(f) / 1024
            print(f"  {f}  ({size_kb:.0f} KB)")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — EXPERIMENTS  (imputation + graph + train + eval per seed)
    #
    # For each seed, rebuild the train/val/test splits (they were overwritten
    # at the end of phase 1 by the last seed), then run each (source,
    # imputation) pair.  The imputation step reads the null-injected CSV
    # for the current seed (created in phase 1) and produces the imputed
    # CSV; graph building, training, and evaluation follow.
    # ══════════════════════════════════════════════════════════════════════════
    all_summaries = []
    t_total_start = time.time()

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n\n{'▓'*70}")
        print(f"  PHASE 2 — SEED {seed_idx} / {len(seeds)}  —  seed={seed}")
        print(f"{'▓'*70}")

        config.set_seed(seed)

        # ── Rebuild splits for this seed (quick — reads clean CSV + samples)
        for path in [config.TRAIN_CSV_FILE, config.VAL_CSV_FILE,
                     config.TEST_CSV_FILE,
                     os.path.join(config.DATA_DIR,
                                  f"{config.DATASET_NAME}_balanced.csv")]:
            if os.path.exists(path):
                os.remove(path)

        if not args.skip_download:
            download_raw_data()

        # ── Clean per-source result files from the previous seed ──────────
        _wipe_results_only()

        # ── Run each (source, imputation) pair ────────────────────────────
        for i, (source, imputation) in enumerate(sources_to_run, 1):
            print(f"\n\n{'█'*70}")
            print(f"  SEED {seed}  —  BATCH RUN {i} / {n_total}  —  "
                  f"SOURCE='{source}'  IMPUTATION='{imputation}'")
            print(f"{'█'*70}")

            try:
                summary = run_single_source(
                    source, imputation, device,
                    force_graph=args.force_graph,
                    force_train=args.force_train,
                )
                summary.setdefault("seed", seed)
                all_summaries.append(summary)
            except Exception as exc:
                print(f"\n[batch] ✗ FAILED: seed={seed}  SOURCE='{source}' — {exc}")
                import traceback
                traceback.print_exc()
                all_summaries.append({
                    "seed": seed, "source": source, "imputation": imputation,
                    "error": str(exc),
                })

    total_elapsed = time.time() - t_total_start

    # ── Final comparison table ─────────────────────────────────────────────
    _print_comparison_table(all_summaries)

    # ── Save cross-seed batch summary ─────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    batch_json = os.path.join(config.RESULTS_DIR,
                              f"{config.DATASET_NAME}_batch_summary.json")
    with open(batch_json, "w") as f:
        json.dump({
            "dataset"       : config.DATASET_NAME,
            "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seeds"         : seeds,
            "sources"       : [{"source": s, "imputation": i}
                                for s, i in sources_to_run],
            "total_elapsed" : round(total_elapsed, 1),
            "runs"          : all_summaries,
        }, f, indent=2)
    print(f"\n[batch] ✓ Cross-seed summary saved → {batch_json}")

    # ── Save CSV summary ──────────────────────────────────────────────────
    csv_rows = []
    for s in all_summaries:
        if "error" in s:
            csv_rows.append({
                "seed": s.get("seed"), "source": s.get("source"),
                "imputation": s.get("imputation", ""),
                "test_acc": None, "test_f1": None, "test_auc": None,
                "pre_male": None, "pre_female": None, "dp_pre": None,
                "post_male": None, "post_female": None, "dp_post": None,
                "tpr_male": None, "tpr_female": None,
                "fpr_male": None, "fpr_female": None,
                "tpr_parity_gap": None, "fpr_parity_gap": None,
                "equal_opportunity_diff": None, "equalized_odds_diff": None,
                "elapsed_sec": None, "error": s["error"],
            })
        else:
            pre_rates  = s.get("dp_pre_group_rates",  {})
            post_rates = s.get("dp_post_group_rates", {})
            tpr_by     = s.get("tpr_by_group", {}) or {}
            fpr_by     = s.get("fpr_by_group", {}) or {}
            csv_rows.append({
                "seed": s["seed"], "source": s["source"],
                "imputation": s["imputation"],
                "test_acc": s["test_acc"], "test_f1": s.get("test_f1"),
                "test_auc": s.get("test_auc"),
                "pre_male": pre_rates.get("Male"),
                "pre_female": pre_rates.get("Female"),
                "dp_pre": s["dp_pre"],
                "post_male": post_rates.get("Male"),
                "post_female": post_rates.get("Female"),
                "dp_post": s["dp_post"],
                "tpr_male": tpr_by.get("Male"),
                "tpr_female": tpr_by.get("Female"),
                "fpr_male": fpr_by.get("Male"),
                "fpr_female": fpr_by.get("Female"),
                "tpr_parity_gap": s.get("tpr_parity_gap"),
                "fpr_parity_gap": s.get("fpr_parity_gap"),
                "equal_opportunity_diff": s.get("equal_opportunity_diff"),
                "equalized_odds_diff": s.get("equalized_odds_diff"),
                "elapsed_sec": s["elapsed_sec"], "error": None,
            })

    batch_csv = os.path.join(
        config.RESULTS_DIR, f"{config.DATASET_NAME}_batch_summary.csv"
    )
    pd.DataFrame(csv_rows).to_csv(batch_csv, index=False)
    print(f"[batch] ✓ Cross-seed CSV saved     → {batch_csv}")

    print(f"\n{'═'*70}")
    print(f"  BATCH COMPLETE — {len(seeds)} seed(s) × {n_total} source(s) "
          f"= {len(seeds)*n_total} runs in {total_elapsed:.1f}s")
    print(f"{'═'*70}\n")


def _wipe_results_only() -> None:
    """
    Remove per-source result files (summaries, plots, reports) without
    touching null-injected MCAR folders or imputed folders.
    """
    if not os.path.isdir(config.RESULTS_DIR):
        return
    keep_suffixes = ("_batch_summary.csv", "_batch_summary.json")
    for fname in os.listdir(config.RESULTS_DIR):
        if fname.endswith(keep_suffixes):
            continue
        path = os.path.join(config.RESULTS_DIR, fname)
        if os.path.isfile(path):
            os.remove(path)

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
