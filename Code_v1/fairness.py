"""
fairness.py
─────────────────────────────────────────────────────────────────────────────
Demographic Parity check — pre-GNN (ground truth) vs post-GNN (predictions).

Demographic Parity measures whether the positive-outcome rate (selection rate)
is equal across groups defined by a sensitive attribute.

    DP Difference  = |P(Ŷ=1 | group=A) - P(Ŷ=1 | group=B)|
    Disparate Impact Ratio = min(rate_A, rate_B) / max(rate_A, rate_B)

    DP Difference ≈ 0        → perfect parity
    Disparate Impact ≥ 0.80  → within the common "four-fifths" threshold

Both checks operate on the SAME test-mask nodes so the comparison is
apples-to-apples:

    check_demographic_parity_pre(test_mask)
        DP on ground-truth labels for test nodes.

    check_demographic_parity_post(eval_results)
        DP on GNN predictions for the same test nodes.

    plot_demographic_parity(dp_pre, dp_post)
        Side-by-side figure comparing both.

The sensitive attribute is saved as a .npy file during graph building,
perfectly aligned to the node order. This avoids any CSV ↔ graph mismatch.

Usage in notebook:
    from fairness import (check_demographic_parity_pre,
                          check_demographic_parity_post,
                          plot_demographic_parity)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import config


# ─────────────────────────────────────────────────────────────────────────────
# Dataset → sensitive column mapping  (used by graph_builder at save time)
# ─────────────────────────────────────────────────────────────────────────────
SENSITIVE_COLS = {
    "adult_census":  "sex",
    "german_credit": "personal_status",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_sensitive_col() -> str:
    col = SENSITIVE_COLS.get(config.DATASET_NAME)
    if col is None:
        raise ValueError(
            f"[fairness] No sensitive column configured for "
            f"'{config.DATASET_NAME}'.\n"
            f"  Add an entry to SENSITIVE_COLS in fairness.py."
        )
    return col


def _load_sensitive_attr() -> np.ndarray:
    """Load the graph-aligned sensitive attribute array."""
    path = config.SENSITIVE_ATTR_F
    assert os.path.exists(path), (
        f"[fairness] Sensitive attribute file not found: {path}\n"
        f"  Rebuild graph files so graph_builder saves it."
    )
    return np.load(path, allow_pickle=True).astype(str)


def _compute_dp(groups_col: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute demographic parity statistics.

    Parameters
    ----------
    groups_col : array of group names (length = number of test nodes)
    labels     : array of 0/1 outcomes (same length)
    """
    unique_groups = sorted(set(groups_col))
    group_stats = {}

    for g in unique_groups:
        mask = groups_col == g
        total = int(mask.sum())
        positive = int(labels[mask].sum())
        rate = positive / total if total > 0 else 0.0
        group_stats[g] = {
            "count":          total,
            "positive":       positive,
            "selection_rate": round(rate, 6),
        }

    rates = [s["selection_rate"] for s in group_stats.values()]
    dp_diff  = round(max(rates) - min(rates), 6) if len(rates) >= 2 else 0.0
    di_ratio = round(min(rates) / max(rates), 6) if max(rates) > 0 else 0.0

    return {
        "groups":                 group_stats,
        "dp_difference":          dp_diff,
        "disparate_impact_ratio": di_ratio,
    }


def _print_dp(result: dict, stage: str) -> None:
    print()
    print("=" * 60)
    print(f"  DEMOGRAPHIC PARITY — {stage}")
    print(f"  Sensitive attr: {result['sensitive_col']}   |   "
          f"Test nodes: {result['n_samples']:,}")
    print("=" * 60)

    for g, s in result["groups"].items():
        bar_len = int(s["selection_rate"] * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {g:<25} n={s['count']:>6,}   "
              f"positive={s['positive']:>5,}   "
              f"rate={s['selection_rate']:.4f}  {bar}")

    print("-" * 60)
    dp = result["dp_difference"]
    di = result["disparate_impact_ratio"]

    if dp <= 0.05:
        verdict_dp = "✓ Low disparity"
    elif dp <= 0.10:
        verdict_dp = "~ Moderate disparity"
    else:
        verdict_dp = "✗ High disparity"

    if di >= 0.80:
        verdict_di = "✓ Passes four-fifths rule"
    else:
        verdict_di = "✗ Fails four-fifths rule"

    print(f"  DP Difference          : {dp:.4f}   ({verdict_dp})")
    print(f"  Disparate Impact Ratio : {di:.4f}   ({verdict_di})")
    print("=" * 60)
    print()


def _save_dp(result: dict, stage: str) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fname = f"{config._FILE_PREFIX}_dp_{stage}.json"
    path = os.path.join(config.RESULTS_DIR, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fairness] Saved {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def check_demographic_parity_pre(test_mask) -> dict:
    """
    Demographic parity on GROUND-TRUTH labels for the test nodes.

    Parameters
    ----------
    test_mask : bool tensor/array of shape (n_nodes,) from the graph
    """
    sensitive_col = _get_sensitive_col()
    sens_all = _load_sensitive_attr()
    y_all    = np.load(config.NODE_LABELS_F)

    # Convert tensor to numpy if needed
    if hasattr(test_mask, 'numpy'):
        test_mask = test_mask.cpu().numpy()
    test_mask = np.asarray(test_mask, dtype=bool)

    groups_test = sens_all[test_mask]
    y_test      = y_all[test_mask]

    dp = _compute_dp(groups_test, y_test)
    result = {
        "sensitive_col": sensitive_col,
        "source":        "ground_truth",
        "n_samples":     int(test_mask.sum()),
        **dp,
    }

    _print_dp(result, stage="PRE-GNN (ground-truth labels)")
    _save_dp(result, stage="pre")
    return result


def check_demographic_parity_post(eval_results: dict) -> dict:
    """
    Demographic parity on GNN PREDICTIONS for the test nodes.

    Uses eval_results["test"]["y_pred"] and the same graph-aligned
    sensitive attribute, indexed by the test mask.
    """
    sensitive_col = _get_sensitive_col()
    sens_all = _load_sensitive_attr()
    test_mask = np.load(config.TEST_MASK_F).astype(bool)

    groups_test = sens_all[test_mask]
    y_pred      = eval_results["test"]["y_pred"]

    assert len(groups_test) == len(y_pred), (
        f"[fairness] Mismatch: test mask selects {len(groups_test)} nodes "
        f"but eval_results has {len(y_pred)} predictions."
    )

    dp = _compute_dp(groups_test, y_pred)
    result = {
        "sensitive_col": sensitive_col,
        "source":        "gnn_predictions",
        "n_samples":     len(y_pred),
        **dp,
    }

    _print_dp(result, stage="POST-GNN (model predictions)")
    _save_dp(result, stage="post")
    return result


def plot_demographic_parity(dp_pre: dict, dp_post: dict) -> str:
    """
    Plot pre-GNN (ground truth) vs post-GNN (predictions) demographic
    parity in one figure.  Both use the same test-mask nodes.

    Left  : grouped bar chart of selection rates per group
    Right : DP Difference comparison with threshold lines

    Returns the path to the saved PNG.
    """
    groups_pre  = dp_pre["groups"]
    groups_post = dp_post["groups"]
    group_names = list(groups_pre.keys())

    rates_pre  = [groups_pre[g]["selection_rate"]  for g in group_names]
    rates_post = [groups_post[g]["selection_rate"] for g in group_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [3, 1.2]})

    # ── Left panel: grouped bar chart ─────────────────────────────────────────
    x = np.arange(len(group_names))
    bar_w = 0.32

    bars1 = ax1.bar(x - bar_w / 2, rates_pre,  bar_w,
                    label="Pre-GNN (test ground truth)",
                    color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars2 = ax1.bar(x + bar_w / 2, rates_post, bar_w,
                    label="Post-GNN (test predictions)",
                    color="#DD8452", edgecolor="white", linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                     f"{h:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(group_names, fontsize=10)
    ax1.set_ylabel("Selection Rate  P(Y=1 | group)", fontsize=10)
    ax1.set_title(f"Selection Rate by {dp_pre['sensitive_col']}  "
                  f"(test nodes, n={dp_pre['n_samples']:,})",
                  fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(max(rates_pre), max(rates_post)) * 1.25)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.3)

    # ── Right panel: DP Difference comparison ─────────────────────────────────
    stages     = ["Pre-GNN\n(test ground truth)", "Post-GNN\n(test predictions)"]
    dp_values  = [dp_pre["dp_difference"], dp_post["dp_difference"]]
    bar_colors = ["#4C72B0", "#DD8452"]

    bars3 = ax2.bar(stages, dp_values, width=0.5, color=bar_colors,
                    edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars3, dp_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax2.set_ylabel("DP Difference", fontsize=10)
    ax2.set_title("DP Difference", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(max(dp_values) * 1.4, 0.15))
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.3)

    fig.suptitle(f"Demographic Parity — {config.DATASET_NAME}  ({config.SOURCE})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR,
                            f"{config._FILE_PREFIX}_demographic_parity.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[fairness] Saved {os.path.basename(out_path)}")
    return out_path
