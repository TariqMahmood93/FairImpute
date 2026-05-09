import os


# DATASET — which dataset to use (applies to both batch and standalone)
DATASET_NAME = "adult_census"   # ← "adult_census" or "german_credit"

# BALANCE STRATEGY — set by the active notebook strategy cell before data loading.
# Each strategy writes its splits into <DATASET_NAME>_<BALANCE_STRATEGY>/.
# Valid values: "no_balance", "twostage", "stratified_sampling"
BALANCE_STRATEGY = "no_balance"

# MISSINGNESS INDICATORS — if True, append binary mask (1.0=missing) to features
APPEND_MISSING_MASK = True

BATCH_SOURCES = [
    # Baseline — clean data (no nulls, no imputation)
    ("clean",   "none"),

    # No imputation — raw nulls with sentinel encoding
    ("mcar_10", "none"),
    # ("mcar_20", "none"),
    # ("mcar_30", "none"),
    # ("mcar_40", "none"),
    # ("mcar_50", "none"),


    # # # KNN imputation — all rates
    ("mcar_10", "knn"),
    # ("mcar_20", "knn"),
    # ("mcar_30", "knn"),
    # ("mcar_40", "knn"),
    # ("mcar_50", "knn"),

    # # # Mode/Median imputation — all rates
    # ("mcar_10",  "mode"),
    # ("mcar_20", "mode"),
    # ("mcar_30", "mode"),
    # ("mcar_40", "mode"),
    # ("mcar_50", "mode"),


# # SENTI imputation — all rates
    # ("mcar_10",  "senti"),
    # ("mcar_20", "senti"),
    # ("mcar_30", "senti"),
    # ("mcar_40", "senti"),
    # ("mcar_50", "senti"),

    # # MICE imputation — all rates
    # ("mcar_5",  "mice"),
    # ("mcar_10", "mice"),
    # ("mcar_20", "mice"),

]

if not BATCH_SOURCES:
    raise ValueError("[config] BATCH_SOURCES must contain at least one "
                     "(source, imputation) pair.")
SOURCE, IMPUTATION_METHOD = BATCH_SOURCES[0]

# Imputation strategy registry — add new methods here
IMPUTATION_STRATEGIES = {
    "none"  : {"label": "No Imputation (raw nulls)",       "needs_install": []},
    "knn"   : {"label": "KNN Imputer (scikit-learn)",      "needs_install": []},
    "mode"  : {"label": "Mode/Median (statistical)",       "needs_install": []},
    "senti" : {"label": "SENTI (transformer-based)",       "needs_install": ["torch", "faiss-cpu", "sentence-transformers"]},
    "mice"  : {"label": "MICE (chained equations)",        "needs_install": []},
}

# SENTI-specific settings
SENTI_TRANSFORMER = "LaBSE"   # sentence-transformers model name

# KNN-specific settings
KNN_IMPUTE_NEIGHBORS = 5

# MICE-specific settings
MICE_MAX_ITER     = 10    # number of imputation rounds
DATASETS = {

    "adult_census": {
        "raw_file_name"   : "adult.data",
        "sep"             : ",",
        "header"          : None,
        "skipinitialspace": True,
        "na_values"       : "?",
        "columns"         : [
            "age", "workclass", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income",
        ],
        "target_col"      : "income",
        "target_map"      : {"<=50K": 0, ">50K": 1},
        "class_names"     : ["<=50K (0)", ">50K (1)"],
        "numerical_cols"  : [
            "age", "education_num",
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
        "target_map"      : {1: 0, 2: 1},
        "class_names"     : ["Good (0)", "Bad (1)"],
        "numerical_cols"  : [
            "duration", "credit_amount", "installment_rate",
            "present_residence", "age", "existing_credits", "dependents",
        ],
    },

}

# Validate DATASET_NAME
if DATASET_NAME not in DATASETS:
    raise ValueError(
        f"[config] Unknown DATASET_NAME='{DATASET_NAME}'.\n"
        f"  Valid options: {list(DATASETS.keys())}\n"
        f"  To add a new dataset, insert an entry in the DATASETS dict."
    )

_DS = DATASETS[DATASET_NAME]

# Dataset-level constants — derived from registry, never set manually
COLUMNS        = _DS["columns"]
TARGET_COL     = _DS["target_col"]
TARGET_MAP     = _DS["target_map"]
CLASS_NAMES    = _DS["class_names"]
NUMERICAL_COLS = _DS["numerical_cols"]

# ── Directory layout ──────────────────────────────────────────────────────────
# Raw/clean source files always live in the base dataset folder.
# Everything else lives inside the strategy-specific DATA_DIR so different
# balancing runs never collide.
#
#   adult_census/                          ← _BASE_DIR  (raw + clean CSVs)
#   adult_census_no_balance/               ← DATA_DIR
#     train.csv  val.csv  test.csv
#     42_mcar_10/                          ← MCAR artefacts (seed_mcar_rate)
#     42_mcar_20/
#     42_mcar_10_knn/                      ← MCAR + imputation
#     Result_no_balance/                   ← results for this strategy
#   adult_census_twostage/                 ← DATA_DIR (different strategy)
#     42_mcar_10/
#     Result_twostage/

_BASE_DIR      = DATASET_NAME                                # "adult_census"
DATA_DIR       = f"{DATASET_NAME}_{BALANCE_STRATEGY}"        # "adult_census_no_balance"

# Input files — place in <_BASE_DIR>/ before running
RAW_FILE       = os.path.join(_BASE_DIR, _DS["raw_file_name"])
CLEAN_CSV_FILE = os.path.join(_BASE_DIR, f"{DATASET_NAME}_clean.csv")

# Fixed train / val / test splits produced by the active balance strategy
TRAIN_CSV_FILE = os.path.join(DATA_DIR, f"{DATASET_NAME}_train.csv")
VAL_CSV_FILE   = os.path.join(DATA_DIR, f"{DATASET_NAME}_val.csv")
TEST_CSV_FILE  = os.path.join(DATA_DIR, f"{DATASET_NAME}_test.csv")


RANDOM_SEED = 42
SEEDS = [42]
# SEEDS = [42, 100, 584, 1001, 1576, 2300, 3156, 4537, 6584, 9030]

def set_global_seed(seed: int = None) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible results."""
    import config as _self
    import random, numpy as np, torch
    if seed is None:
        seed = _self.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def set_seed(seed: int) -> None:
    """Update config.RANDOM_SEED AND seed all RNGs in one call."""
    import config as _self
    _self.RANDOM_SEED = seed
    set_global_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
def _resolve_source(source: str, imputation: str) -> tuple:
    """
    Given SOURCE string and IMPUTATION_METHOD, return
    (folder, csv_path, file_prefix, resolved_imputation).

    All folder names embed BALANCE_STRATEGY so artefacts from different
    balancing runs never collide.
    """
    bs = BALANCE_STRATEGY
    ds = DATASET_NAME

    if source == "clean":
        folder = os.path.join(DATA_DIR, f"{RANDOM_SEED}_clean")
        prefix = f"{ds}_{bs}_clean"
        csv    = CLEAN_CSV_FILE
        return folder, csv, prefix, "none"

    if not source.startswith("mcar_"):
        raise ValueError(
            f"[config] Unknown SOURCE='{source}'.\n"
            f"  Valid: 'clean', 'mcar_10', 'mcar_20_knn', etc."
        )

    # Parse: mcar_<rate> or mcar_<rate>_<strategy>
    parts = source.split("_")   # ["mcar", "10"] or ["mcar", "10", "knn"]
    rate  = parts[1]

    if len(parts) >= 3:
        embedded_imp = "_".join(parts[2:])
        resolved_imp = embedded_imp
    else:
        resolved_imp = imputation

    if resolved_imp == "none" or resolved_imp not in IMPUTATION_STRATEGIES:
        folder = os.path.join(DATA_DIR, f"{RANDOM_SEED}_mcar_{rate}")
        prefix = f"train_{RANDOM_SEED}_mcar_{rate}"
        csv    = os.path.join(folder, f"{prefix}.csv")
        resolved_imp = "none"
    else:
        folder = os.path.join(DATA_DIR, f"{RANDOM_SEED}_mcar_{rate}_{resolved_imp}")
        prefix = f"mcar_{rate}_{resolved_imp}_{RANDOM_SEED}"
        csv    = os.path.join(folder, f"{prefix}.csv")

    return folder, csv, prefix, resolved_imp

# Resolve once at import time
_SOURCE_FOLDER, SOURCE_CSV, _FILE_PREFIX, RESOLVED_IMPUTATION = _resolve_source(SOURCE, IMPUTATION_METHOD)


##################################################################################################################################

# NULL INJECTION (MCAR) — chunk-based incremental injection
MISSING_RATE = [10]

def mcar_folder(rate: int) -> str:
    return os.path.join(DATA_DIR, f"{RANDOM_SEED}_mcar_{rate}")

# Training split — nulls injected here
def mcar_train_csv(rate: int) -> str:
    return os.path.join(mcar_folder(rate), f"train_{RANDOM_SEED}_mcar_{rate}.csv")

def mcar_train_data(rate: int) -> str:
    return os.path.join(mcar_folder(rate), f"train_{RANDOM_SEED}_mcar_{rate}.data")

# Val / test splits — shared across all MCAR rates, live in the base DATA_DIR folder
def mcar_val_csv(rate: int = None) -> str:
    return VAL_CSV_FILE

def mcar_test_csv(rate: int = None) -> str:
    return TEST_CSV_FILE

# Legacy combined-file helpers — kept for imputer.py / imputation_eval.py compatibility
def mcar_csv(rate: int) -> str:
    return mcar_train_csv(rate)

def mcar_data(rate: int) -> str:
    return mcar_train_data(rate)

def imputed_folder(rate: int, strategy: str) -> str:
    return os.path.join(DATA_DIR, f"{RANDOM_SEED}_mcar_{rate}_{strategy}")

def imputed_csv(rate: int, strategy: str) -> str:
    folder = imputed_folder(rate, strategy)
    return os.path.join(folder, f"mcar_{rate}_{strategy}_{RANDOM_SEED}.csv")

# ── Graph construction ────────────────────────────────────────────────────────
K_NEIGHBORS = 5

# ── Train / Val / Test split ──────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# ── Model hyperparameters ─────────────────────────────────────────────────────
HIDDEN_DIM   = 64
DROPOUT      = 0.5
LR           = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS       = 200
NUM_LAYERS   = 3        # number of hidden GCN layers

# ─────────────────────────────────────────────────────────────────────────────
# File paths — embed dataset + balance_strategy + source prefix so files never collide
# ─────────────────────────────────────────────────────────────────────────────
GRAPH_DIR = _SOURCE_FOLDER

NODE_FEATURES_F  = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_node_features.npy")
NODE_LABELS_F    = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_node_labels.npy")
EDGE_INDEX_F     = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_edge_index.npy")
TRAIN_MASK_F     = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_train_mask.npy")
VAL_MASK_F       = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_val_mask.npy")
TEST_MASK_F      = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_test_mask.npy")
METADATA_F       = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_metadata.json")
MODEL_F          = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_trained_model.pt")
SENSITIVE_ATTR_F = os.path.join(GRAPH_DIR, f"{_FILE_PREFIX}_sensitive_attr.npy")

# ── Results folder — lives inside DATA_DIR, name embeds strategy ─────────────
RESULTS_DIR = os.path.join(DATA_DIR, f"Result_{BALANCE_STRATEGY}")
CURVES_F    = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_training_curves.png")
VAL_CM_F    = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_val_confusion_matrix.png")
TEST_CM_F   = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_test_confusion_matrix.png")
REPORT_F    = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_classification_report.txt")
SUMMARY_F   = os.path.join(RESULTS_DIR, f"{_FILE_PREFIX}_summary.json")

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
# BALANCE STRATEGY SWITCH — call this from the notebook strategy cell
# ─────────────────────────────────────────────────────────────────────────────
def set_balance_strategy(strategy: str) -> None:
    """
    Switch the active balance strategy and re-derive ALL dependent path globals.

    Call this from the notebook strategy cell BEFORE any data loading:
        import config
        config.set_balance_strategy("stratified_sampling")

    Every module that reads config.DATA_DIR, config.TRAIN_CSV_FILE,
    config.GRAPH_DIR, etc. will see the updated values after this call.
    """
    import config as _self

    _self.BALANCE_STRATEGY = strategy
    _self.DATA_DIR         = f"{_self.DATASET_NAME}_{strategy}"
    _self.TRAIN_CSV_FILE   = os.path.join(_self.DATA_DIR, f"{_self.DATASET_NAME}_train.csv")
    _self.VAL_CSV_FILE     = os.path.join(_self.DATA_DIR, f"{_self.DATASET_NAME}_val.csv")
    _self.TEST_CSV_FILE    = os.path.join(_self.DATA_DIR, f"{_self.DATASET_NAME}_test.csv")
    _self.RESULTS_DIR      = os.path.join(_self.DATA_DIR, f"Result_{strategy}")

    # Re-derive all source/graph paths for the current source
    reconfigure(_self.SOURCE, _self.IMPUTATION_METHOD)

    print(f"[config] ✓ Balance strategy : '{strategy}'")
    print(f"[config]   DATA_DIR         : {_self.DATA_DIR}/")
    print(f"[config]   RESULTS_DIR      : {_self.RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC RECONFIGURATION — switch SOURCE at runtime without restarting
# ─────────────────────────────────────────────────────────────────────────────
def reconfigure(source: str, imputation: str = "none") -> None:
    """
    Re-derive ALL module-level path globals for a new (source, imputation) pair.
    Respects the currently active BALANCE_STRATEGY.

    This lets the batch loop switch sources in one process:
        config.reconfigure("clean",   "none")
        config.reconfigure("mcar_10", "knn")
    """
    import config as _self

    _self.SOURCE            = source
    _self.IMPUTATION_METHOD = imputation

    # Re-resolve source paths (reads current BALANCE_STRATEGY from _self)
    folder, csv, prefix, resolved_imp = _resolve_source(source, imputation)
    _self._SOURCE_FOLDER      = folder
    _self.SOURCE_CSV          = csv
    _self._FILE_PREFIX        = prefix
    _self.RESOLVED_IMPUTATION = resolved_imp

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

    _self.CURVES_F  = os.path.join(_self.RESULTS_DIR, f"{prefix}_training_curves.png")
    _self.VAL_CM_F  = os.path.join(_self.RESULTS_DIR, f"{prefix}_val_confusion_matrix.png")
    _self.TEST_CM_F = os.path.join(_self.RESULTS_DIR, f"{prefix}_test_confusion_matrix.png")
    _self.REPORT_F  = os.path.join(_self.RESULTS_DIR, f"{prefix}_classification_report.txt")
    _self.SUMMARY_F = os.path.join(_self.RESULTS_DIR, f"{prefix}_summary.json")

    _self.NPY_FILES = [
        _self.NODE_FEATURES_F, _self.NODE_LABELS_F, _self.EDGE_INDEX_F,
        _self.TRAIN_MASK_F, _self.VAL_MASK_F, _self.TEST_MASK_F, _self.METADATA_F,
    ]

    imp_label = IMPUTATION_STRATEGIES.get(resolved_imp, {}).get("label", resolved_imp)
    print(f"\n[config] ✓ Reconfigured:")
    print(f"[config]   BALANCE_STRATEGY  = '{_self.BALANCE_STRATEGY}'")
    print(f"[config]   SOURCE            = '{source}'")
    print(f"[config]   IMPUTATION        = '{resolved_imp}'  ({imp_label})")
    print(f"[config]   SOURCE_CSV        = {csv}")
    print(f"[config]   GRAPH_DIR         = {folder}/")
    print(f"[config]   FILE_PREFIX       = {prefix}")
