"""
config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for every parameter in the pipeline.
All other modules import from here — change values once, affects everything.

HOW TO SWITCH DATASETS
─────────────────────────────────────────────────────────────────────────────
Change DATASET_NAME to the dataset you want to train on:

    DATASET_NAME = "adult_census"    → UCI Adult Income
    DATASET_NAME = "german_credit"   → UCI German Credit

BATCH MODE (recommended — run_batch.py)
─────────────────────────────────────────────────────────────────────────────
Edit BATCH_SOURCES below to list which (source, imputation) pairs to
process sequentially.  Then run:

    python run_batch.py                   # runs all sources in sequence
    python run_batch.py --force-train     # force retrain all
    python run_batch.py --sources clean mcar_15   # run specific ones

The batch runner calls config.reconfigure() for each source, so
SOURCE and IMPUTATION_METHOD below are only used as defaults for
standalone main.py runs (not batch mode).

STANDALONE MODE (main.py — single source)
─────────────────────────────────────────────────────────────────────────────
Set SOURCE and IMPUTATION_METHOD below, then run:

    python main.py

IMPUTATION OPTIONS:
    IMPUTATION_METHOD = "none"    → no imputation (sentinel-encoded nulls)
    IMPUTATION_METHOD = "mode"    → Mode (categorical) + Median (numerical)
    IMPUTATION_METHOD = "senti"   → SENTI transformer-based imputation
    IMPUTATION_METHOD = "knn"     → Scikit-learn KNN imputer
    IMPUTATION_METHOD = "mice"    → MICE (chained equations)

To add a new strategy, register it in IMPUTATION_STRATEGIES below and
implement its handler in imputer.py — no other file needs to change.
─────────────────────────────────────────────────────────────────────────────
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# DATASET — which dataset to use (applies to both batch and standalone)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_NAME = "adult_census"   # ← "adult_census" or "german_credit"

# ─────────────────────────────────────────────────────────────────────────────
# BATCH MODE — edit this list, then run:  python run_batch.py
# Each entry is (SOURCE, IMPUTATION_METHOD).
# All sources are processed sequentially with full cleanup between runs.
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SOURCES = [
    ("clean",   "none"),
    ("mcar_10", "none"),
    ("mcar_20", "none"),
    ("mcar_30", "none"),
]

# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE MODE — only used when running:  python main.py
# In batch mode these are overridden by config.reconfigure() for each source.
# ─────────────────────────────────────────────────────────────────────────────
SOURCE             = "clean"    # ← "clean", "mcar_15", "mcar_30", "mcar_50"
IMPUTATION_METHOD  = "none"     # ← "none", "mode", "senti", "knn", "mice"

# ─────────────────────────────────────────────────────────────────────────────
# Imputation strategy registry — add new methods here
# ─────────────────────────────────────────────────────────────────────────────
IMPUTATION_STRATEGIES = {
    "none"  : {"label": "No Imputation (raw nulls)",       "needs_install": []},
    "mode"  : {"label": "Mode/Median (statistical)",       "needs_install": []},
    "senti" : {"label": "SENTI (transformer-based)",       "needs_install": ["torch", "faiss-cpu", "sentence-transformers"]},
    "knn"   : {"label": "KNN Imputer (scikit-learn)",      "needs_install": []},
    "mice"  : {"label": "MICE (chained equations)",        "needs_install": []},
}

# SENTI-specific settings
SENTI_TRANSFORMER = "all-MiniLM-L6-v2"   # sentence-transformers model name

# KNN-specific settings
KNN_IMPUTE_NEIGHBORS = 5

# MICE-specific settings
MICE_MAX_ITER   = 10     # number of imputation rounds
MICE_RANDOM_STATE = 42   # reproducibility (uses config.RANDOM_SEED by default)

# ─────────────────────────────────────────────────────────────────────────────
# Each entry fully describes one dataset. Add new datasets here.
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = {

    "adult_census": {
        "raw_file_name"   : "adult.data",
        "sep"             : ",",
        "header"          : None,
        "skipinitialspace": True,
        "na_values"       : "?",
        "columns"         : [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income",
        ],
        "target_col"      : "income",
        "target_map"      : {"<=50K": 0, ">50K": 1},
        "class_names"     : ["<=50K (0)", ">50K (1)"],
        "numerical_cols"  : [
            "age", "fnlwgt", "education_num",
            "capital_gain", "capital_loss", "hours_per_week",
        ],
    },

    "german_credit": {
        "raw_file_name"   : "german.data",
        "sep"             : " ",
        "header"          : None,
        "skipinitialspace": False,
        "na_values"       : None,
        "columns"         : [
            "checking_status", "duration", "credit_history", "purpose",
            "credit_amount", "savings_status", "employment", "installment_rate",
            "personal_status", "other_debtors", "present_residence", "property",
            "age", "other_installment", "housing", "existing_credits",
            "job", "dependents", "telephone", "foreign_worker", "target",
        ],
        "target_col"      : "target",
        "target_map"      : {1: 0, 2: 1},   # 1 = good → 0,  2 = bad → 1
        "class_names"     : ["Good (0)", "Bad (1)"],
        "numerical_cols"  : [
            "duration", "credit_amount", "installment_rate",
            "present_residence", "age", "existing_credits", "dependents",
        ],
    },

}

# ── Validate DATASET_NAME ─────────────────────────────────────────────────────
if DATASET_NAME not in DATASETS:
    raise ValueError(
        f"[config] Unknown DATASET_NAME='{DATASET_NAME}'.\n"
        f"  Valid options: {list(DATASETS.keys())}\n"
        f"  To add a new dataset, insert an entry in the DATASETS dict."
    )

_DS = DATASETS[DATASET_NAME]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level constants — derived from registry, never set manually
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR          = DATASET_NAME
COLUMNS           = _DS["columns"]
TARGET_COL        = _DS["target_col"]
TARGET_MAP        = _DS["target_map"]
CLASS_NAMES       = _DS["class_names"]
NUMERICAL_COLS    = _DS["numerical_cols"]

# Input files — place these in the dataset folder before running
# <DATASET_NAME>/<raw_file_name>   e.g. adult_census/adult.data
# <DATASET_NAME>/<dataset>_clean.csv
RAW_FILE          = os.path.join(DATA_DIR, _DS["raw_file_name"])
CLEAN_CSV_FILE    = os.path.join(DATA_DIR, f"{DATASET_NAME}_clean.csv")

# Fixed train / val / test splits (produced once in data_download.py)
# Null injection operates ONLY on TRAIN_CSV_FILE.
TRAIN_CSV_FILE    = os.path.join(DATA_DIR, f"{DATASET_NAME}_train.csv")
VAL_CSV_FILE      = os.path.join(DATA_DIR, f"{DATASET_NAME}_val.csv")
TEST_CSV_FILE     = os.path.join(DATA_DIR, f"{DATASET_NAME}_test.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Source resolution — now supports imputation suffix
#
# FORMAT:  "clean"  |  "mcar_<rate>"  |  "mcar_<rate>_<imputation>"
#
# Examples:
#   "clean"            → clean data, no imputation
#   "mcar_5"           → 5% MCAR, no imputation (uses IMPUTATION_METHOD)
#   "mcar_10_mode"     → 10% MCAR, mode/median imputation
#   "mcar_20_senti"    → 20% MCAR, SENTI imputation
#   "mcar_5_knn"       → 5% MCAR, KNN imputation
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_source(source: str, imputation: str) -> tuple:
    """
    Given SOURCE string and IMPUTATION_METHOD, return
    (folder, csv_path, file_prefix, resolved_imputation).

    If SOURCE embeds an imputation suffix (e.g. "mcar_10_senti"),
    that takes priority over the standalone IMPUTATION_METHOD.
    """
    if source == "clean":
        folder = DATA_DIR
        prefix = f"{DATASET_NAME}_clean"
        csv    = CLEAN_CSV_FILE
        resolved_imp = "none"
        return folder, csv, prefix, resolved_imp

    if not source.startswith("mcar_"):
        raise ValueError(
            f"[config] Unknown SOURCE='{source}'.\n"
            f"  Valid: 'clean', 'mcar_5', 'mcar_10_mode', 'mcar_20_senti', etc."
        )

    # Parse: mcar_<rate> or mcar_<rate>_<strategy>
    parts = source.split("_")  # ["mcar", "5"] or ["mcar", "10", "mode"]
    rate  = parts[1]

    if len(parts) >= 3:
        embedded_imp = "_".join(parts[2:])   # supports multi-word strategy names
        resolved_imp = embedded_imp
    else:
        resolved_imp = imputation

    if resolved_imp == "none" or resolved_imp not in IMPUTATION_STRATEGIES:
        # Raw MCAR — no imputation applied; SOURCE_CSV is the training split
        folder = f"{DATASET_NAME}_mcar_{rate}pct"
        prefix = f"{DATASET_NAME}_mcar_{rate}"
        csv    = os.path.join(folder, f"{prefix}_train.csv")
        resolved_imp = "none"
    else:
        # Imputed MCAR
        folder = f"{DATASET_NAME}_mcar_{rate}pct_{resolved_imp}"
        prefix = f"{DATASET_NAME}_mcar_{rate}_{resolved_imp}"
        csv    = os.path.join(folder, f"{prefix}.csv")

    return folder, csv, prefix, resolved_imp

# Resolve once at import time
_SOURCE_FOLDER, SOURCE_CSV, _FILE_PREFIX, RESOLVED_IMPUTATION = _resolve_source(SOURCE, IMPUTATION_METHOD)

# ─────────────────────────────────────────────────────────────────────────────
# NULL INJECTION (MCAR) — chunk-based incremental injection
# ─────────────────────────────────────────────────────────────────────────────
MISSING_RATE = [10, 20, 30]      # cumulative percentages: 15% → +15% = 30% → +20% = 50%
INJECTION_SEEDS = [42]           # seeds for reproducibility (can run multiple)

# Chunk settings — data grows incrementally in chunks
CHUNK_INITIAL_SIZE = None        # rows in the first chunk (None = use full dataset, no chunking)
CHUNK_STEP         = None        # rows added per subsequent chunk (None = use full dataset)

def mcar_folder(rate: int) -> str:
    return f"{DATASET_NAME}_mcar_{rate}pct"

# Training split — nulls injected here
def mcar_train_csv(rate: int) -> str:
    return os.path.join(mcar_folder(rate), f"{DATASET_NAME}_mcar_{rate}_train.csv")

def mcar_train_data(rate: int) -> str:
    return os.path.join(mcar_folder(rate), f"{DATASET_NAME}_mcar_{rate}_train.data")

# Val / test splits — shared across all MCAR rates, live in the base dataset folder
# These are never modified; the same files are used regardless of MCAR rate.
def mcar_val_csv(rate: int = None) -> str:
    return VAL_CSV_FILE

def mcar_test_csv(rate: int = None) -> str:
    return TEST_CSV_FILE

# Legacy combined-file helpers — kept for imputer.py / imputation_eval.py compatibility
def mcar_csv(rate: int) -> str:
    """Legacy: points to the training split (was the single MCAR file before v6)."""
    return mcar_train_csv(rate)

def mcar_data(rate: int) -> str:
    """Legacy: points to the training split .data file."""
    return mcar_train_data(rate)

def imputed_folder(rate: int, strategy: str) -> str:
    return f"{DATASET_NAME}_mcar_{rate}pct_{strategy}"

def imputed_csv(rate: int, strategy: str) -> str:
    folder = imputed_folder(rate, strategy)
    return os.path.join(folder, f"{DATASET_NAME}_mcar_{rate}_{strategy}.csv")

# ── Graph construction ────────────────────────────────────────────────────────
K_NEIGHBORS = 5

# ── Train / Val / Test split ──────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
RANDOM_SEED = 42

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible results."""
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ── Model hyperparameters ─────────────────────────────────────────────────────
HIDDEN_DIM   = 64
DROPOUT      = 0.5
LR           = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS       = 200
NUM_LAYERS   = 2        # number of GCN conv layers (2 or 3)

# ─────────────────────────────────────────────────────────────────────────────
# File paths — all embed dataset+source prefix so files never collide
# ─────────────────────────────────────────────────────────────────────────────
GRAPH_DIR = _SOURCE_FOLDER

NODE_FEATURES_F = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_node_features.npy")
NODE_LABELS_F   = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_node_labels.npy")
EDGE_INDEX_F    = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_edge_index.npy")
TRAIN_MASK_F    = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_train_mask.npy")
VAL_MASK_F      = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_val_mask.npy")
TEST_MASK_F     = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_test_mask.npy")
METADATA_F      = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_metadata.json")
MODEL_F         = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_trained_model.pt")
SENSITIVE_ATTR_F = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_sensitive_attr.npy")

# ── Results folder ────────────────────────────────────────────────────────────
RESULTS_DIR  = "results"
CURVES_F     = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_training_curves.png")
VAL_CM_F     = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_val_confusion_matrix.png")
TEST_CM_F    = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_test_confusion_matrix.png")
REPORT_F     = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_classification_report.txt")
SUMMARY_F    = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_summary.json")

# ── Quick existence helpers ───────────────────────────────────────────────────
NPY_FILES = [
    NODE_FEATURES_F, NODE_LABELS_F, EDGE_INDEX_F,
    TRAIN_MASK_F, VAL_MASK_F, TEST_MASK_F, METADATA_F,
]

def npy_files_exist() -> bool:
    return all(os.path.exists(f) for f in NPY_FILES)

def model_exists() -> bool:
    return os.path.exists(MODEL_F)


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC RECONFIGURATION — switch SOURCE at runtime without restarting
# ─────────────────────────────────────────────────────────────────────────────
def reconfigure(source: str, imputation: str = "none") -> None:
    """
    Re-derive ALL module-level path globals for a new (source, imputation) pair.

    This lets run_batch.py loop through multiple sources in one process:
        config.reconfigure("clean",   "none")   # run pipeline
        config.reconfigure("mcar_15", "none")   # run pipeline
        config.reconfigure("mcar_30", "none")   # run pipeline

    Every module that does `import config` and reads config.SOURCE_CSV,
    config.GRAPH_DIR, config.MODEL_F, etc. will see the updated values
    after this call.
    """
    import config as _self

    _self.SOURCE             = source
    _self.IMPUTATION_METHOD  = imputation

    # Re-resolve source paths
    folder, csv, prefix, resolved_imp = _resolve_source(source, imputation)
    _self._SOURCE_FOLDER       = folder
    _self.SOURCE_CSV           = csv
    _self._FILE_PREFIX         = prefix
    _self.RESOLVED_IMPUTATION  = resolved_imp

    # Re-derive all file paths
    _self.GRAPH_DIR        = folder
    _self.NODE_FEATURES_F  = os.path.join(folder, f"{prefix}_node_features.npy")
    _self.NODE_LABELS_F    = os.path.join(folder, f"{prefix}_node_labels.npy")
    _self.EDGE_INDEX_F     = os.path.join(folder, f"{prefix}_edge_index.npy")
    _self.TRAIN_MASK_F     = os.path.join(folder, f"{prefix}_train_mask.npy")
    _self.VAL_MASK_F       = os.path.join(folder, f"{prefix}_val_mask.npy")
    _self.TEST_MASK_F      = os.path.join(folder, f"{prefix}_test_mask.npy")
    _self.METADATA_F       = os.path.join(folder, f"{prefix}_metadata.json")
    _self.MODEL_F          = os.path.join(folder, f"{prefix}_trained_model.pt")
    _self.SENSITIVE_ATTR_F = os.path.join(folder, f"{prefix}_sensitive_attr.npy")

    _self.CURVES_F   = os.path.join(RESULTS_DIR, f"{prefix}_training_curves.png")
    _self.VAL_CM_F   = os.path.join(RESULTS_DIR, f"{prefix}_val_confusion_matrix.png")
    _self.TEST_CM_F  = os.path.join(RESULTS_DIR, f"{prefix}_test_confusion_matrix.png")
    _self.REPORT_F   = os.path.join(RESULTS_DIR, f"{prefix}_classification_report.txt")
    _self.SUMMARY_F  = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")

    _self.NPY_FILES = [
        _self.NODE_FEATURES_F, _self.NODE_LABELS_F, _self.EDGE_INDEX_F,
        _self.TRAIN_MASK_F, _self.VAL_MASK_F, _self.TEST_MASK_F, _self.METADATA_F,
    ]

    imp_label = IMPUTATION_STRATEGIES.get(resolved_imp, {}).get("label", resolved_imp)
    print(f"\n[config] ✓ Reconfigured:")
    print(f"[config]   SOURCE           = '{source}'")
    print(f"[config]   IMPUTATION       = '{resolved_imp}'  ({imp_label})")
    print(f"[config]   SOURCE_CSV       = {csv}")
    print(f"[config]   GRAPH_DIR        = {folder}/")
    print(f"[config]   FILE_PREFIX      = {prefix}")
