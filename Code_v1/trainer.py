"""
trainer.py
──────────
Responsible for: training the GCN, evaluating on val/test sets,
and saving / loading the model checkpoint.

Training strategy:
  - Runs ALL config.EPOCHS epochs (no early stopping)
  - Tracks best validation accuracy across all epochs
  - After training completes, restores weights from the best epoch
  - Saves only the best model checkpoint

This ensures the model sees the full training schedule and the
best-performing snapshot is always selected.

Skips training entirely if a saved checkpoint already exists
(use --force-train flag in main.py to override).

Usage (imported):
    from trainer import run_training, evaluate
    history = run_training(model, data, device)
    metrics = evaluate(model, data, device)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.data import Data

import config
from model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_training(model, data: Data, device: torch.device) -> dict:
    """
    Train the model for ALL config.EPOCHS epochs, then select the best.

    Parameters
    ----------
    model  : GCN instance (already on device)
    data   : PyG Data object (already on device)
    device : torch.device

    Returns
    -------
    dict with keys:
        "train_losses" — list of float, one per epoch
        "val_accs"     — list of float, one per epoch
        "best_epoch"   — int, epoch with highest val_acc
        "stopped_early"— bool, always False (no early stopping)
        "loaded"       — bool, True if weights were loaded (not trained)
    """
    if config.model_exists():
        print(f"[trainer] ✓ Checkpoint found: {config.MODEL_F} — loading weights, skipping training")
        state = torch.load(config.MODEL_F, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return {"train_losses": [], "val_accs": [], "best_epoch": 0,
                "stopped_early": False, "loaded": True}

    print(f"[trainer] Training for {config.EPOCHS} epochs  "
          f"(lr={config.LR}, wd={config.WEIGHT_DECAY}, dropout={config.DROPOUT})")
    print(f"[trainer] Strategy             : run ALL epochs, select best\n")

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    train_losses, val_accs = [], []
    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    print(f"  {'Epoch':>6}  {'Loss':>8}  {'Train Acc':>10}  {'Val Acc':>8}  {'Best':>6}")
    print(f"  {'-'*48}")

    for epoch in range(1, config.EPOCHS + 1):
        loss      = _train_step(model, data, optimizer)
        train_acc = _accuracy(model, data, data.train_mask)
        val_acc   = _accuracy(model, data, data.val_mask)

        train_losses.append(loss)
        val_accs.append(val_acc)

        # Track best model across ALL epochs
        is_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            is_best      = True

        if epoch % 20 == 0 or epoch == 1 or epoch == config.EPOCHS or is_best:
            marker = "  ★" if is_best else ""
            print(f"  {epoch:>6}  {loss:>8.4f}  {train_acc:>10.2%}  {val_acc:>8.2%}  "
                  f"{'★' if is_best else ' ':>4}{marker}")

    print(f"\n[trainer] All {config.EPOCHS} epochs completed")
    print(f"[trainer] Best val accuracy : {best_val_acc:.2%}  (epoch {best_epoch})")

    # Restore best weights before saving
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[trainer] Restored weights from best epoch {best_epoch}")

    torch.save(model.state_dict(), config.MODEL_F)
    print(f"[trainer] ✓ Model saved → {config.MODEL_F}")

    return {
        "train_losses" : train_losses,
        "val_accs"     : val_accs,
        "best_epoch"   : best_epoch,
        "stopped_early": False,
        "loaded"       : False,
    }


def evaluate(model, data: Data, device: torch.device) -> dict:
    """
    Compute predictions and metrics for train / val / test splits.

    Returns
    -------
    dict with keys: "train", "val", "test"
        Each value:
            "y_true"      — np.ndarray
            "y_pred"      — np.ndarray
            "acc"         — float
            "f1_macro"    — float
            "f1_weighted" — float
            "roc_auc"     — float or None (None if multiclass not supported)
    """
    splits = {
        "train": data.train_mask,
        "val"  : data.val_mask,
        "test" : data.test_mask,
    }

    results = {}
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds  = logits.argmax(dim=1)
        # Probabilities for AUC (binary: use class-1 probability)
        probs  = torch.exp(logits)          # log_softmax → softmax

    for split_name, mask in splits.items():
        y_true = data.y[mask].cpu().numpy()
        y_pred = preds[mask].cpu().numpy()
        y_prob = probs[mask].cpu().numpy()

        acc         = float((y_true == y_pred).mean())
        f1_macro    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        try:
            num_classes = y_prob.shape[1]
            if num_classes == 2:
                roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                               average="macro"))
        except Exception:
            roc_auc = None

        results[split_name] = {
            "y_true"      : y_true,
            "y_pred"      : y_pred,
            "acc"         : acc,
            "f1_macro"    : f1_macro,
            "f1_weighted" : f1_weighted,
            "roc_auc"     : roc_auc,
        }

    for split_name, r in results.items():
        auc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
        mask_sz = splits[split_name].sum().item()
        print(f"[trainer] {split_name.capitalize():<6} "
              f"Acc={r['acc']:.4f}  F1={r['f1_macro']:.4f}  AUC={auc_str}  "
              f"({mask_sz:,} nodes)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_step(model, data: Data, optimizer) -> float:
    """One forward + backward pass on train nodes."""
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def _accuracy(model, data: Data, mask: torch.Tensor) -> float:
    """Compute accuracy for a given boolean mask."""
    model.eval()
    preds   = model(data.x, data.edge_index).argmax(dim=1)
    correct = (preds[mask] == data.y[mask]).sum().item()
    return correct / mask.sum().item()


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch_geometric.data import Data
    from graph_builder import load_graph_files

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trainer] Device: {device}\n")

    X, y, edge_index, train_mask, val_mask, test_mask, meta = load_graph_files()

    data = Data(
        x          = torch.tensor(X,  dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y,  dtype=torch.long),
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    ).to(device)

    model = build_model(
        num_features = meta["num_features"],
        num_classes  = meta["num_classes"],
    ).to(device)

    history = run_training(model, data, device)
    results = evaluate(model, data, device)
