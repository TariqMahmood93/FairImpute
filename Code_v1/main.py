"""
main.py
───────────────────────────────────────────────────────────────────────────
Entry point for the GNN pipeline with imputation.

WORKFLOW:
    1. Read existing clean CSV + produce fixed train/val/test split CSVs
    2. Null Injection (MCAR) — optional, training split ONLY
    3a. Imputation — applies strategy from config.RESOLVED_IMPUTATION
    3b. Imputation Quality Evaluation — exact match + semantic similarity
    4. Build graph files (.npy)
    5. Load graph → PyG Data
    6. Build model
    7a. Train
    7b. Evaluate
    7c. Save results

QUICK START — change 3 variables in config.py:
    DATASET_NAME      = "adult_census"
    SOURCE            = "mcar_10_mode"     ← rate + imputation in one string
    IMPUTATION_METHOD = "none"             ← ignored when SOURCE has strategy

Run:
    python main.py
    python main.py --force-graph --force-train

Flags:
    --force-graph     Rebuild .npy files even if they exist
    --force-train     Retrain model even if checkpoint exists
    --skip-download   Skip download (raw file must already exist)
"""

import argparse
import logging
import os
import sys
from datetime import datetime

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


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    """Configure root logger: console + timestamped log file in results/."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(
        config.RESULTS_DIR,
        f"{config._FILE_PREFIX}_run_{timestamp}.log"
    )

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                             datefmt="%H:%M:%S")

    file_handler    = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    logger = logging.getLogger("gnn_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Log file  : {os.path.abspath(log_file)}")
    logger.info(f"Dataset   : {config.DATASET_NAME}  SOURCE='{config.SOURCE}'")
    logger.info(f"Imputation: {config.RESOLVED_IMPUTATION}")
    return logger


def _banner(text: str) -> None:
    print(f"\n{'─'*60}\n  {text}\n{'─'*60}")


def _parse_args():
    parser = argparse.ArgumentParser(description="GNN Node Classification Pipeline")
    parser.add_argument("--force-graph",   action="store_true",
                        help="Rebuild .npy files even if they exist")
    parser.add_argument("--force-train",   action="store_true",
                        help="Retrain model even if checkpoint exists")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip split generation (clean CSV must already exist)")
    return parser.parse_args()


def main():
    args   = _parse_args()
    logger = _setup_logging()
    config.set_global_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imp_label = config.IMPUTATION_STRATEGIES.get(
        config.RESOLVED_IMPUTATION, {}
    ).get("label", config.RESOLVED_IMPUTATION)

    logger.info(f"Device      : {device}")
    logger.info(f"Source CSV  : {config.SOURCE_CSV}")
    logger.info(f"Output dir  : {config.GRAPH_DIR}/")

    print(f"\n{'='*60}")
    print(f"  GNN Node Classification — {config.DATASET_NAME}")
    print(f"  Device      : {device}")
    print(f"  SOURCE      : '{config.SOURCE}'")
    print(f"  Imputation  : {config.RESOLVED_IMPUTATION}  ({imp_label})")
    print(f"  Source CSV  : {config.SOURCE_CSV}")
    print(f"  Output dir  : {config.GRAPH_DIR}/")
    print(f"  File prefix : {config._FILE_PREFIX}")
    print(f"{'='*60}")

    # ── STEP 1: Download + save clean CSV ─────────────────────────────────────
    _banner("STEP 1 / 7 — Load Data + Build Splits")
    if args.skip_download:
        print("[main] --skip-download flag set — verifying clean CSV exists")
        if not os.path.exists(config.CLEAN_CSV_FILE):
            logger.error(f"Clean CSV not found: {config.CLEAN_CSV_FILE}")
            sys.exit(1)
    else:
        download_raw_data()
        if not os.path.exists(config.CLEAN_CSV_FILE):
            logger.error(f"Clean CSV missing: {config.CLEAN_CSV_FILE}")
            sys.exit(1)
    logger.info("STEP 1 complete — data ready")

    # ── STEP 2: Null Injection (OPTIONAL) ─────────────────────────────────────
    _banner("STEP 2 — Null Injection (MCAR, training split only)  [optional]")

    # =========================================================================
    #  To ENABLE  → uncomment BOTH lines below
    #  To DISABLE → leave them commented (default)
    #  Nulls are injected ONLY into the training split; val and test are
    #  passed through unchanged to prevent data leakage.
    #  Imputation is handled separately in Step 3a.
    # =========================================================================

    # from null_injector import run_null_injection                                          # ← UNCOMMENT
    # run_null_injection(pd.read_csv(config.TRAIN_CSV_FILE), run_impute=False)              # ← UNCOMMENT

    print(f"[main] Null injection : DISABLED")
    print(f"[main] To enable      : uncomment both lines above")
    print(f"[main] SOURCE         : '{config.SOURCE}'  →  {config.SOURCE_CSV}")

    # ── STEP 3a: Imputation (if SOURCE requires it) ─────────────────────────
    _banner("STEP 3a — Imputation")

    if config.RESOLVED_IMPUTATION != "none":
        if not os.path.exists(config.SOURCE_CSV):
            # The imputed CSV doesn't exist yet — try to create it from raw MCAR
            parts = config.SOURCE.split("_")
            rate  = int(parts[1])
            raw_csv = config.mcar_csv(rate)

            if not os.path.exists(raw_csv):
                logger.error(f"Raw MCAR CSV not found: {raw_csv}")
                logger.error(f"  Run null injection first (uncomment lines in Step 2)")
                sys.exit(1)

            print(f"[main] Imputed CSV not found — creating from raw MCAR ...")
            print(f"[main] Raw MCAR  : {raw_csv}")
            print(f"[main] Strategy  : {config.RESOLVED_IMPUTATION}")

            from imputer import impute
            df_raw     = pd.read_csv(raw_csv)
            df_imputed = impute(df_raw, strategy=config.RESOLVED_IMPUTATION)

            os.makedirs(config.GRAPH_DIR, exist_ok=True)
            df_imputed.to_csv(config.SOURCE_CSV, index=False)
            print(f"[main] ✓ Saved imputed CSV → {config.SOURCE_CSV}")
        else:
            print(f"[main] Imputed CSV already exists: {config.SOURCE_CSV}")

        print(f"[main] Imputation method : {config.RESOLVED_IMPUTATION}  ({imp_label})")
    else:
        print(f"[main] No imputation required (SOURCE='{config.SOURCE}')")

    logger.info(f"STEP 3a complete — imputation={config.RESOLVED_IMPUTATION}")

    # ── STEP 3b: Imputation Quality Evaluation ────────────────────────────
    _banner("STEP 3b — Imputation Quality Evaluation")
    imputation_eval_results = evaluate_imputation()
    if imputation_eval_results.get("available"):
        em_rate = imputation_eval_results["exact_match"]["overall_rate"]
        logger.info(f"STEP 3b complete — exact match rate={em_rate:.4f}")
    else:
        logger.info("STEP 3b complete — skipped (no imputation to evaluate)")

    # ── STEP 4: Build graph files ─────────────────────────────────────────────
    _banner("STEP 4 / 7 — Graph File Construction")

    if args.force_graph:
        for f in config.NPY_FILES:
            if os.path.exists(f):
                os.remove(f)
                print(f"[main] Deleted: {f}")

    build_graph_files()

    if not config.npy_files_exist():
        logger.error("Graph files missing after build_graph_files() — aborting")
        sys.exit(1)
    logger.info("STEP 4 complete — graph files ready")

    # ── STEP 5: Load graph ────────────────────────────────────────────────────
    _banner("STEP 5 / 7 — Load Graph into PyG Data Object")
    X, y, edge_index, train_mask, val_mask, test_mask, meta = load_graph_files()

    data = Data(
        x          = torch.tensor(X,  dtype=torch.float),   # NaN-free: nulls encoded as sentinel=-1
        edge_index = edge_index,
        y          = torch.tensor(y,  dtype=torch.long),
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    ).to(device)

    print(f"\n[main] PyG Data: {data}")
    logger.info(f"STEP 5 complete — {meta['num_nodes']:,} nodes, "
                f"{meta['num_edges']:,} edges, {meta['num_features']} features")

    # ── STEP 6: Build model ───────────────────────────────────────────────────
    _banner("STEP 6 / 7 — Model Definition")
    model = build_model(
        num_features = meta["num_features"],
        num_classes  = meta["num_classes"],
    ).to(device)
    logger.info("STEP 6 complete — model built")

    # ── STEP 7a: Train ────────────────────────────────────────────────────────
    _banner("STEP 7a / 7 — Training")
    if args.force_train and config.model_exists():
        os.remove(config.MODEL_F)
        print(f"[main] Deleted: {config.MODEL_F}")

    history = run_training(model, data, device)
    logger.info(
        f"STEP 7a complete — "
        + (f"loaded from checkpoint" if history.get("loaded")
           else f"trained {len(history['val_accs'])} epochs, "
                f"best val={max(history['val_accs']):.4f} @ epoch {history['best_epoch']}")
    )

    # ── STEP 7b: Evaluate ─────────────────────────────────────────────────────
    _banner("STEP 7b / 7 — Evaluation")
    eval_results = evaluate(model, data, device)
    logger.info(
        f"STEP 7b complete — test acc={eval_results['test']['acc']:.4f}  "
        f"f1={eval_results['test']['f1_macro']:.4f}  "
        f"auc={eval_results['test'].get('roc_auc')}"
    )

    # ── STEP 7c: Results ──────────────────────────────────────────────────────
    _banner("STEP 7c / 7 — Results")
    print_results(eval_results)
    save_all_results(history, eval_results, meta)
    logger.info("STEP 7c complete — all results saved")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  SOURCE       : '{config.SOURCE}'")
    print(f"  Imputation   : {config.RESOLVED_IMPUTATION}  ({imp_label})")
    print(f"  Source CSV   : {config.SOURCE_CSV}")
    print(f"  Graph files  : {os.path.abspath(config.GRAPH_DIR)}/")
    print(f"  Results      : {os.path.abspath(config.RESULTS_DIR)}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
