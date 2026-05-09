"""
results.py
──────────
Responsible for ONE thing only: saving, plotting, and reporting results.

Writes everything to the  results/  folder:
    results/
    ├── training_curves.png
    ├── val_confusion_matrix.png
    ├── test_confusion_matrix.png
    ├── classification_report.txt
    └── summary.json

Usage (imported):
    from results import save_all_results
    save_all_results(history, eval_results, meta)
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import config


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def save_all_results(history: dict, eval_results: dict, meta: dict) -> None:
    """
    Save all plots, reports, and a JSON summary to the results/ folder.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"[results] Saving all results to: {os.path.abspath(config.RESULTS_DIR)}/\n")

    _plot_training_curves(history)
    _plot_confusion_matrix(eval_results["val"],  "Validation", config.VAL_CM_F,  "Blues")
    _plot_confusion_matrix(eval_results["test"], "Test",       config.TEST_CM_F, "Greens")
    _save_classification_report(eval_results)
    _save_summary(history, eval_results, meta)

    print(f"\n[results] ✓ All results saved to: {config.RESULTS_DIR}/")
    _list_results_folder()


def print_results(eval_results: dict) -> None:
    """Print a clean accuracy + classification report to console with explanations."""
    class_names = config.CLASS_NAMES

    print(f"\n  HOW TO READ THE CLASSIFICATION REPORT:")
    print(f"    Precision  = Of all samples predicted as this class, how many truly are? TP/(TP+FP)")
    print(f"    Recall     = Of all samples truly in this class, how many did the model find? TP/(TP+FN)")
    print(f"    F1-score   = Harmonic mean of precision and recall (balances both)")
    print(f"    Support    = Number of true samples in this class")
    print(f"    Macro avg  = Simple average across classes (treats each class equally)")
    print(f"    Weighted   = Weighted average by support (accounts for class imbalance)")

    for split in ["val", "test"]:
        r = eval_results[split]
        label = split.upper()
        print(f"\n{'='*55}")
        print(f"{label} RESULTS")
        print(f"{'='*55}")
        print(f"Accuracy : {r['acc']:.2%}")
        auc = r.get('roc_auc')
        if auc is not None:
            print(f"ROC-AUC  : {auc:.4f}")
            if auc > 0.9:
                print(f"  → Excellent discrimination — the model separates classes very well")
            elif auc > 0.8:
                print(f"  → Good discrimination — the model reliably separates classes")
            elif auc > 0.7:
                print(f"  → Fair discrimination — some overlap between class distributions")
            else:
                print(f"  → Poor discrimination — the model struggles to separate classes")
        print()
        print(classification_report(
            r["y_true"], r["y_pred"],
            target_names=class_names,
            digits=4,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_training_curves(history: dict) -> None:
    """Plot and save loss + val-accuracy curves."""
    train_losses = history.get("train_losses", [])
    val_accs     = history.get("val_accs",     [])

    if not train_losses:
        print("[results]   Training curves skipped (model was loaded from checkpoint)")
        if os.path.exists(config.CURVES_F):
            print(f"[results]   Previous curves: {config.CURVES_F}")
        return

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(epochs, train_losses, color="steelblue", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in val_accs], color="darkorange", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3)

    imp_tag = f" [{config.RESOLVED_IMPUTATION}]" if config.RESOLVED_IMPUTATION != "none" else ""
    plt.suptitle(f"GCN — {config.DATASET_NAME}{imp_tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.CURVES_F, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[results]   ✓ Saved training_curves.png")


def _plot_confusion_matrix(
    split_result: dict, title: str, save_path: str, cmap: str,
) -> None:
    cm = confusion_matrix(split_result["y_true"], split_result["y_pred"])

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = config.CLASS_NAMES,
    ).plot(ax=ax, colorbar=True, cmap=cmap)

    acc = split_result["acc"]
    ax.set_title(f"Confusion Matrix — {title}  (acc={acc:.2%})")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    fname = os.path.basename(save_path)
    print(f"[results]   ✓ Saved {fname}")


def _save_classification_report(eval_results: dict) -> None:
    lines = []
    lines.append(f"Classification Report — {config.DATASET_NAME}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Imputation: {config.RESOLVED_IMPUTATION}")
    lines.append("=" * 60)

    for split in ["train", "val", "test"]:
        r = eval_results[split]
        auc_str = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "N/A"
        lines.append(f"\n{split.upper()} SET  "
                     f"(acc={r['acc']:.4f}  f1_macro={r['f1_macro']:.4f}  "
                     f"f1_weighted={r['f1_weighted']:.4f}  auc={auc_str})")
        lines.append("-" * 60)
        lines.append(classification_report(
            r["y_true"], r["y_pred"],
            target_names=config.CLASS_NAMES,
            digits=4,
        ))

    with open(config.REPORT_F, "w") as f:
        f.write("\n".join(lines))

    print(f"[results]   ✓ Saved classification_report.txt")


def _save_summary(history: dict, eval_results: dict, meta: dict) -> None:
    """Save a structured JSON summary including imputation metadata."""

    def _safe_auc(split):
        v = eval_results[split].get("roc_auc")
        return round(v, 4) if v is not None else None

    imp_label = config.IMPUTATION_STRATEGIES.get(
        config.RESOLVED_IMPUTATION, {}
    ).get("label", config.RESOLVED_IMPUTATION)

    summary = {
        "dataset"   : config.DATASET_NAME,
        "source"    : config.SOURCE,
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "imputation": {
            "method"  : config.RESOLVED_IMPUTATION,
            "label"   : imp_label,
        },
        "graph"     : {
            "num_nodes"    : meta["num_nodes"],
            "num_edges"    : meta["num_edges"],
            "num_features" : meta["num_features"],
            "num_classes"  : meta["num_classes"],
            "k_neighbors"  : meta["k_neighbors"],
        },
        "hyperparameters": {
            "hidden_dim"              : config.HIDDEN_DIM,
            "num_layers"              : config.NUM_LAYERS,
            "dropout"                 : config.DROPOUT,
            "lr"                      : config.LR,
            "weight_decay"            : config.WEIGHT_DECAY,
            "epochs"                  : config.EPOCHS,
        },
        "split": meta["split"],
        "accuracy": {
            "train" : round(eval_results["train"]["acc"], 4),
            "val"   : round(eval_results["val"]["acc"],   4),
            "test"  : round(eval_results["test"]["acc"],  4),
        },
        "f1_macro": {
            "train" : round(eval_results["train"].get("f1_macro", 0), 4),
            "val"   : round(eval_results["val"].get("f1_macro", 0),   4),
            "test"  : round(eval_results["test"].get("f1_macro", 0),  4),
        },
        "f1_weighted": {
            "train" : round(eval_results["train"].get("f1_weighted", 0), 4),
            "val"   : round(eval_results["val"].get("f1_weighted", 0),   4),
            "test"  : round(eval_results["test"].get("f1_weighted", 0),  4),
        },
        "roc_auc": {
            "train" : _safe_auc("train"),
            "val"   : _safe_auc("val"),
            "test"  : _safe_auc("test"),
        },
        "training": {
            "retrained"      : not history.get("loaded", True),
            "stopped_early"  : history.get("stopped_early", False),
            "best_val_acc"   : round(max(history["val_accs"]), 4) if history.get("val_accs") else None,
            "best_val_epoch" : history.get("best_epoch"),
            "total_epochs"   : len(history.get("val_accs") or []),
            "final_loss"     : round(history["train_losses"][-1], 4)
                                if history.get("train_losses") else None,
        },
    }

    with open(config.SUMMARY_F, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[results]   ✓ Saved summary.json")


def _list_results_folder() -> None:
    print(f"\n[results] Contents of {config.RESULTS_DIR}/:")
    for fname in sorted(os.listdir(config.RESULTS_DIR)):
        fpath = os.path.join(config.RESULTS_DIR, fname)
        size  = os.path.getsize(fpath)
        unit  = f"{size/1024:.1f} KB" if size < 1_000_000 else f"{size/1_048_576:.1f} MB"
        print(f"           {fname:<45} {unit}")
