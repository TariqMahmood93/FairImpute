# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Research codebase studying how MCAR missing data and imputation affect fairness in GNN-based node classification. Two tabular datasets (adult_census, german_credit), multiple balancing strategies, multiple imputation methods, fairness measured as Demographic Parity / Equalized Odds.

## Running the Experiment

**Primary interface:** `main.ipynb` — run cells top to bottom.

**Headless batch run (CLI):**
```bash
python run_batch.py                   # all BATCH_SOURCES × all SEEDS
python run_batch.py --force-graph     # rebuild .npy graph files
python run_batch.py --force-train     # retrain models
python run_batch.py --sources clean mcar_10   # specific sources only
```

**Standalone single-source run:**
```bash
python main.py                        # uses current config.SOURCE
python null_injector.py               # inject nulls (all seeds)
python null_injector.py --single-seed # current seed only
python imputer.py --strategies knn mode
python graph_builder.py
python trainer.py
```

## config.py — The Control Center

All experiment parameters live here. **All path globals are live module attributes — they change when `reconfigure()` or `set_balance_strategy()` is called.** Never cache them at import time.

Key parameters to edit before running:

```python
DATASET_NAME     = "adult_census"   # or "german_credit"
BALANCE_STRATEGY = "no_balance"     # drives DATA_DIR name

BATCH_SOURCES = [
    ("clean",   "none"),            # uncomment mcar entries to process them
    # ("mcar_10", "none"),
    # ("mcar_10", "knn"),
]

RANDOM_SEED = 42
SEEDS       = [42]                  # multi-seed batch

MISSING_RATE = [10, 20]             # MCAR injection rates (%)

# Model hyperparameters
HIDDEN_DIM = 64
EPOCHS     = 200
NUM_LAYERS = 3
```

### Switching strategy at runtime

```python
config.set_balance_strategy("no_balance")   # updates DATA_DIR + calls reconfigure()
config.reconfigure("mcar_10", "knn")        # switches all path globals to this source
config.set_seed(100)                        # updates RANDOM_SEED + seeds all RNGs
```

Call order matters: `set_balance_strategy` → `set_seed` → `reconfigure` → graph build/train.

## Directory Layout

```
adult_census/                        ← raw + clean CSVs (input — never modified)
  adult_census_clean.csv

adult_census_no_balance/             ← DATA_DIR (one per balance strategy)
  adult_census_train.csv             ← shared splits (seed-dependent, regenerated)
  adult_census_val.csv
  adult_census_test.csv
  42_clean/                          ← graph .npy files for clean source, seed=42
  42_mcar_10/                        ← MCAR 10%, seed=42: train CSV + .npy files
    train_42_mcar_10.csv             ← null-injected training split
    train_42_mcar_10.data            ← same with "?" for NaN (UCI format)
    train_42_mcar_10_node_features.npy
    train_42_mcar_10_*.npy
  42_mcar_10_knn/                    ← imputed: KNN on 10%, seed=42
    mcar_10_knn_42.csv
    mcar_10_knn_42_*.npy
  Result_no_balance/                 ← all result files for this strategy
```

File prefix rules:
- **clean source:** `{dataset}_{strategy}_clean` (e.g. `adult_census_no_balance_clean`)
- **MCAR no imputation:** `train_{seed}_mcar_{rate}` (e.g. `train_42_mcar_10`)
- **MCAR with imputation:** `mcar_{rate}_{imputation}_{seed}` (e.g. `mcar_10_knn_42`)

## Pipeline Stages (in order)

1. **Balance strategy cell** (notebook) — calls `config.set_balance_strategy()` + `load_and_split()` from `balance_{strategy}.py`. Creates train/val/test CSVs in DATA_DIR.

2. **Null injection** (`null_injector.py`) — injects MCAR nulls into training split only. Val/test are never touched. Injection is cumulative (10% is a strict subset of 20%). Creates `{seed}_mcar_{rate}/train_{seed}_mcar_{rate}.csv` inside DATA_DIR.

3. **Imputation** (`imputer.py` → `impute_{strategy}.py`) — reads MCAR train CSV, outputs imputed CSV to `{seed}_mcar_{rate}_{strategy}/`.

4. **Graph building** (`graph_builder.py`) — encodes features (label-encoded cats, z-scored nums), replaces NaN with sentinel=-1, builds KNN graph with null-aware Euclidean distance. Stacks train|val|test into one graph with boolean masks. Saves `.npy` files to `GRAPH_DIR` (= the source's subfolder).

5. **Training** (`trainer.py`) — GCN, runs all EPOCHS, restores best-val-acc checkpoint. Skips if model file exists.

6. **Evaluation + fairness** (`results.py`, `fairness.py`) — saves metrics JSON/PNG to `RESULTS_DIR`.

## GNN Architecture

Multi-layer GCN (`model.py`): `[GCNConv→ReLU→Dropout] × NUM_LAYERS` hidden layers + 1 output GCNConv + log_softmax. `build_model()` reads `config.HIDDEN_DIM`, `config.NUM_LAYERS`, `config.DROPOUT`.

Graph is built from ALL rows (train+val+test) so message-passing flows across splits. Train/val/test are boolean masks on the same graph.

## Fairness Metrics

Computed on test-mask nodes only.

- **DP (Demographic Parity):** `|P(Ŷ=1|Male) - P(Ŷ=1|Female)|` — pre-GNN (ground truth) and post-GNN (predictions)
- **Equal Opportunity:** `|TPR_Male - TPR_Female|`
- **Equalized Odds:** `max(|ΔTPR|, |ΔFPR|)`

Sensitive column mapping lives in **two** places that must stay in sync:
- `fairness.SENSITIVE_COLS` — used by graph_builder to save `sensitive_attr.npy`
- `balance_random_feature_redistribution.SENSITIVE_COLS` — used by that balancing strategy

## Adding New Components

**New balancing strategy:**
1. Create `balance_{name}.py` with `load_and_split() -> str`
2. Add a commented option cell in `main.ipynb` that calls `config.set_balance_strategy('{name}')` before `load_and_split()`

**New imputation method:**
1. Create `impute_{name}.py` with `impute_{name}(df, target_col, cat_cols, num_cols) -> df`
2. Add to `config.IMPUTATION_STRATEGIES`
3. Import and register in `imputer._HANDLERS`

**New dataset:**
1. Add entry to `config.DATASETS`
2. Add to `fairness.SENSITIVE_COLS`
3. Add to `balance_random_feature_redistribution.SENSITIVE_COLS` if using that strategy
4. Place `{dataset}/{dataset}_clean.csv` before running

## Key Invariants

- `null_injector` touches **training split only** — val/test must always be null-free
- MCAR injection is **cumulative**: positions injected at 10% are a strict subset of positions at 20%
- Graph uses **sentinel=-1** for missing values (not NaN), tracked by `null_mask.npy`
- KNN graph distance: both-observed → `(Δ)²`; one-null → `0` (transparent); both-null → skipped
- `config.reconfigure(source, imputation)` must be called before any graph build or training when switching sources in a loop — the batch pipeline does this automatically
