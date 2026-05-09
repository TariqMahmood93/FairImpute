"""
run_layers_experiment.py
────────────────────────────────────────────────────────────────────────────
Sweep NUM_LAYERS ∈ [3, 4, 5, 6, 7] while keeping every other setting fixed.

For each (seed × source × num_layers):
  1. Reconfigure paths + set seed
  2. Download / split data (once per seed)
  3. Build graph files (once per seed × source)
  4. For each num_layers:
       a. Delete stale model checkpoint (force fresh training)
       b. Set config.NUM_LAYERS
       c. Build model → train → evaluate → fairness
       d. Collect all metrics into a row

Output: results/layers_experiment.csv
  Columns: no_layers, seed, source, imputation,
            test_acc, test_f1_macro, test_f1_weighted, test_auc,
            val_acc, val_f1_macro, val_auc,
            dp_pre, dp_post,
            pre_<group>, post_<group>  (one column per sensitive group),
            tpr_parity_gap, fpr_parity_gap,
            equal_opportunity_diff, equalized_odds_diff,
            tpr_<group>, fpr_<group>,
            best_val_epoch, elapsed_sec

Usage:
    python run_layers_experiment.py
    python run_layers_experiment.py --sources clean
    python run_layers_experiment.py --layers 3 5 7
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import gc
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
from torch_geometric.data import Data

import config
from data_download import download_raw_data
from graph_builder import build_graph_files, load_graph_files
from model import build_model
from trainer import run_training, evaluate

LAYERS_TO_TEST = [3, 4, 5, 6, 7]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="GNN layers sweep experiment")
    p.add_argument("--sources", nargs="+", default=None,
                   help="Subset of BATCH_SOURCES to run (e.g. --sources clean)")
    p.add_argument("--layers", nargs="+", type=int, default=LAYERS_TO_TEST,
                   help=f"Layer counts to test (default: {LAYERS_TO_TEST})")
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _banner(msg: str) -> None:
    print(f"\n{'─'*70}\n  {msg}\n{'─'*70}")


def _cleanup_model_only(device):
    plt.close("all")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _flatten_row(num_layers, seed, source, imputation,
                 eval_results, history, dp_pre, dp_post, fair_ext,
                 elapsed) -> dict:
    """Build a flat dict for one CSV row."""
    row = {
        "no_layers"          : num_layers,
        "seed"               : seed,
        "source"             : source,
        "imputation"         : imputation,
        # Test metrics
        "test_acc"           : round(eval_results["test"]["acc"],         4),
        "test_f1_macro"      : round(eval_results["test"]["f1_macro"],    4),
        "test_f1_weighted"   : round(eval_results["test"]["f1_weighted"], 4),
        "test_auc"           : round(eval_results["test"]["roc_auc"],     4)
                               if eval_results["test"].get("roc_auc") else None,
        # Val metrics
        "val_acc"            : round(eval_results["val"]["acc"],       4),
        "val_f1_macro"       : round(eval_results["val"]["f1_macro"],  4),
        "val_auc"            : round(eval_results["val"]["roc_auc"],   4)
                               if eval_results["val"].get("roc_auc") else None,
        # Training
        "best_val_epoch"     : history.get("best_epoch"),
        "elapsed_sec"        : round(elapsed, 1),
    }

    # Demographic parity
    row["dp_pre"]  = round(dp_pre["dp_difference"],  6) if dp_pre  else None
    row["dp_post"] = round(dp_post["dp_difference"], 6) if dp_post else None

    # Per-group selection rates — pre / post
    if dp_pre:
        for g, rate in dp_pre["group_rates"].items():
            row[f"pre_{g}"] = round(rate, 6)
    if dp_post:
        for g, rate in dp_post["group_rates"].items():
            row[f"post_{g}"] = round(rate, 6)

    # Extended fairness
    if fair_ext:
        row["tpr_parity_gap"]          = round(fair_ext["tpr_parity_gap"],         6)
        row["fpr_parity_gap"]          = round(fair_ext["fpr_parity_gap"],         6)
        row["equal_opportunity_diff"]  = round(fair_ext["equal_opportunity_diff"], 6)
        row["equalized_odds_diff"]     = round(fair_ext["equalized_odds_diff"],    6)
        for g, tpr in fair_ext.get("tpr_by_group", {}).items():
            row[f"tpr_{g}"] = round(tpr, 6)
        for g, fpr in fair_ext.get("fpr_by_group", {}).items():
            row[f"fpr_{g}"] = round(fpr, 6)
    else:
        for k in ("tpr_parity_gap", "fpr_parity_gap",
                  "equal_opportunity_diff", "equalized_odds_diff"):
            row[k] = None

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Core: one (source × seed × num_layers) run — graph already loaded
# ─────────────────────────────────────────────────────────────────────────────
def _run_one_layer(num_layers, data, meta, test_mask_tensor, device) -> dict:
    """
    Train and evaluate a fresh GCN with `num_layers` layers.
    Returns eval_results, history, dp_pre, dp_post, fair_ext.
    Deletes model checkpoint so training is never skipped.
    """
    # Force fresh training: delete checkpoint if present
    if os.path.exists(config.MODEL_F):
        os.remove(config.MODEL_F)

    config.NUM_LAYERS = num_layers

    model = build_model(
        num_features=meta["num_features"],
        num_classes=meta["num_classes"],
    ).to(device)

    history      = run_training(model, data, device)
    eval_results = evaluate(model, data, device)

    # Fairness
    dp_pre = dp_post = fair_ext = None
    try:
        from fairness import (check_demographic_parity_pre,
                              check_demographic_parity_post,
                              check_fairness_metrics_post,
                              plot_demographic_parity)
        dp_pre   = check_demographic_parity_pre(test_mask_tensor)
        dp_post  = check_demographic_parity_post(eval_results)
        fair_ext = check_fairness_metrics_post(eval_results)
        plot_demographic_parity(dp_pre, dp_post)
    except Exception as exc:
        print(f"[layers] Fairness failed for num_layers={num_layers}: {exc}")

    # Free model
    model.cpu()
    del model
    _cleanup_model_only(device)

    return eval_results, history, dp_pre, dp_post, fair_ext


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = _parse_args()
    layers = sorted(set(args.layers))

    # Clear stale bytecode
    pycache = os.path.join(os.path.dirname(__file__), "__pycache__")
    if os.path.isdir(pycache):
        shutil.rmtree(pycache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine sources to run
    if args.sources:
        src_map = {s: (s, imp) for s, imp in config.BATCH_SOURCES}
        sources_to_run = [src_map.get(s, (s, "none")) for s in args.sources]
    else:
        sources_to_run = config.BATCH_SOURCES

    seeds = list(config.SEEDS)

    print(f"\n{'═'*70}")
    print(f"  LAYERS EXPERIMENT — {config.DATASET_NAME}")
    print(f"  Layers  : {layers}")
    print(f"  Seeds   : {seeds}")
    print(f"  Sources : {[s for s, _ in sources_to_run]}")
    print(f"  Device  : {device}")
    total_runs = len(layers) * len(seeds) * len(sources_to_run)
    print(f"  Total runs : {total_runs}")
    print(f"{'═'*70}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    all_rows = []
    run_idx  = 0

    for seed in seeds:
        config.set_seed(seed)

        # Rebuild splits for this seed
        for path in [config.TRAIN_CSV_FILE, config.VAL_CSV_FILE, config.TEST_CSV_FILE]:
            if os.path.exists(path):
                os.remove(path)

        if not args.skip_download:
            _banner(f"Data download / split — seed={seed}")
            download_raw_data()

        for source, imputation in sources_to_run:
            _banner(f"Seed={seed}  Source='{source}'  Imputation='{imputation}'")
            config.reconfigure(source, imputation)
            config.set_global_seed()

            # Build graph once for this (seed, source) pair
            build_graph_files()
            if not config.npy_files_exist():
                print(f"[layers] ✗ Graph build failed — skipping source '{source}'")
                continue

            X, y, edge_index, train_mask, val_mask, test_mask, meta = load_graph_files()
            data = Data(
                x          = torch.tensor(X, dtype=torch.float),
                edge_index = edge_index,
                y          = torch.tensor(y, dtype=torch.long),
                train_mask = train_mask,
                val_mask   = val_mask,
                test_mask  = test_mask,
            ).to(device)

            # Keep a CPU copy of test_mask for fairness functions
            test_mask_tensor = test_mask

            for num_layers in layers:
                run_idx += 1
                t0 = time.time()
                print(f"\n{'█'*70}")
                print(f"  Run {run_idx}/{total_runs} — "
                      f"seed={seed}  source='{source}'  num_layers={num_layers}")
                print(f"{'█'*70}")

                try:
                    eval_results, history, dp_pre, dp_post, fair_ext = \
                        _run_one_layer(num_layers, data, meta, test_mask_tensor, device)

                    elapsed = time.time() - t0
                    row = _flatten_row(
                        num_layers, seed, source, imputation,
                        eval_results, history, dp_pre, dp_post, fair_ext,
                        elapsed,
                    )
                    all_rows.append(row)

                    print(f"\n[layers] ✓ layers={num_layers}  "
                          f"test_acc={row['test_acc']:.4f}  "
                          f"test_f1={row['test_f1_macro']:.4f}  "
                          f"dp_post={row.get('dp_post')}  "
                          f"eq_odds={row.get('equalized_odds_diff')}  "
                          f"({elapsed:.1f}s)")

                    # Save intermediate CSV after every run
                    out_csv = os.path.join(config.RESULTS_DIR, "layers_experiment.csv")
                    pd.DataFrame(all_rows).to_csv(out_csv, index=False)

                except Exception as exc:
                    import traceback
                    print(f"[layers] ✗ FAILED layers={num_layers}: {exc}")
                    traceback.print_exc()
                    all_rows.append({
                        "no_layers": num_layers, "seed": seed,
                        "source": source, "imputation": imputation,
                        "error": str(exc),
                    })

            # Free graph data for this source before next
            del data, X, y, edge_index, train_mask, val_mask, test_mask, meta
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final save
    out_csv = os.path.join(config.RESULTS_DIR, "layers_experiment.csv")
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\n{'═'*70}")
    print(f"  DONE — {len(all_rows)} rows written to {out_csv}")
    print(f"{'═'*70}\n")

    # Summary table
    if "error" not in df.columns or df["error"].isna().all():
        summary_cols = ["no_layers", "seed", "source",
                        "test_acc", "test_f1_macro", "test_auc",
                        "dp_post", "equalized_odds_diff"]
        avail = [c for c in summary_cols if c in df.columns]
        print(df[avail].to_string(index=False))


if __name__ == "__main__":
    main()
