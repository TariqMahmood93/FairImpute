"""
graph_builder.py
────────────────────────────────────────────────────────────────────────────
Converts the source CSVs into .npy graph files.

NaN handling
─────────────
NaNs are fully supported at every stage — no imputation, no silent fill.

• Encoding  : categorical NaNs become a dedicated "nan" category; numerical
              NaNs are scaled with stats computed on observed values only and
              remain np.nan in the feature matrix.
• KNN graph : built with nan-aware Euclidean distance — only observed
              dimensions are used per pair, rescaled by √(d_total/d_observed).
              This is the standard approach in missing-data GNN literature
              (GRAPE, MIWAE, etc.).
• Node feats: NaN positions in X are stored as np.nan in node_features.npy.
              The GNN model must handle or mask these (e.g. replace with 0
              before message-passing).

Graph construction
──────────────────
The graph is built from ALL rows across the three splits so that message-
passing can flow between every node at inference time.

    SOURCE = "clean"
        nodes  = all rows of <dataset>_clean.csv
        masks  = derived from TRAIN_CSV / VAL_CSV / TEST_CSV row indices

    SOURCE = "mcar_<rate>"   (IMPUTATION_METHOD = "none")
        nodes  = train rows (with NaNs) + val rows (clean) + test rows (clean)
        masks  = first len(train) rows → train_mask
                 next  len(val)  rows  → val_mask
                 last  len(test) rows  → test_mask

    SOURCE = "mcar_<rate>_<strategy>"   (imputed)
        same layout but source CSV is already NaN-free after imputation.

Usage:
    from graph_builder import build_graph_files, load_graph_files
    build_graph_files()
    X, y, edge_index, train_mask, val_mask, test_mask, meta = load_graph_files()
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def build_graph_files(force: bool = False) -> None:
    """Build and save all .npy graph files."""
    source_csv = config.SOURCE_CSV
    os.makedirs(config.GRAPH_DIR, exist_ok=True)

    if not force and config.npy_files_exist():
        print(f"[graph_builder] ✓ Graph files for SOURCE='{config.SOURCE}' already exist — skipping")
        _print_existing_files()
        return

    if not os.path.exists(source_csv):
        raise FileNotFoundError(
            f"[graph_builder] Source CSV not found: {source_csv}\n"
            f"  For SOURCE='clean'              → run data_download.py first\n"
            f"  For SOURCE='mcar_<r>'           → run null_injector.py first\n"
            f"  For SOURCE='mcar_<r>_<strategy>'→ run null_injector.py then imputer.py"
        )

    print(f"[graph_builder] Building graph files for SOURCE='{config.SOURCE}' ...")
    print(f"[graph_builder] Source CSV   : {source_csv}")
    print(f"[graph_builder] Imputation   : {config.RESOLVED_IMPUTATION}")
    print(f"[graph_builder] Output dir   : {config.GRAPH_DIR}/")
    print(f"[graph_builder] File prefix  : {config._FILE_PREFIX}\n")

    X, y, feature_cols, train_mask, val_mask, test_mask, null_mask, df_raw_features = _preprocess_with_masks(source_csv)
    edge_index = _build_knn_graph(X, null_mask, feature_cols, df_raw_features)

    # ── Option: Append Missingness Indicators ─────────────────────────────────
    if getattr(config, "APPEND_MISSING_MASK", False):
        print(f"[graph_builder] Appending missingness indicators to feature matrix ...")
        mask_float   = null_mask.astype(np.float32)
        X            = np.concatenate([X, mask_float], axis=1)
        feature_cols = feature_cols + [f"{c}_nan" for c in feature_cols]
        print(f"[graph_builder]   New feature shape : {X.shape}")

    _save_all(X, y, edge_index, train_mask, val_mask, test_mask, feature_cols, null_mask)
    print(f"\n[graph_builder] ✓ All files saved to: {config.GRAPH_DIR}/")


def load_graph_files() -> tuple:
    """Load all .npy graph files for the current config.SOURCE."""
    print(f"[graph_builder] Loading graph files for SOURCE='{config.SOURCE}' ...")

    X          = np.load(config.NODE_FEATURES_F)
    y          = np.load(config.NODE_LABELS_F)
    edge_index = torch.tensor(np.load(config.EDGE_INDEX_F), dtype=torch.long)
    train_mask = torch.tensor(np.load(config.TRAIN_MASK_F), dtype=torch.bool)
    val_mask   = torch.tensor(np.load(config.VAL_MASK_F),   dtype=torch.bool)
    test_mask  = torch.tensor(np.load(config.TEST_MASK_F),  dtype=torch.bool)

    with open(config.METADATA_F) as f:
        meta = json.load(f)

    # Load null_mask if it exists
    null_mask_path = config.NODE_FEATURES_F.replace("_node_features.npy", "_null_mask.npy")
    if os.path.exists(null_mask_path):
        null_mask = np.load(null_mask_path)
        n_sentinel = int(null_mask.sum())
        print(f"[graph_builder]   null_mask     : {null_mask.shape}  sentinel cells={n_sentinel:,}")
    else:
        null_mask = np.zeros_like(X, dtype=bool)
        print(f"[graph_builder]   null_mask     : none found (all values observed)")

    nan_in_X = int(np.isnan(X).sum())
    print(f"[graph_builder]   node_features : {X.shape}  dtype={X.dtype}  NaNs={nan_in_X:,} (should be 0)")
    print(f"[graph_builder]   node_labels   : {y.shape}  unique={np.unique(y).tolist()}")
    print(f"[graph_builder]   edge_index    : {tuple(edge_index.shape)}")
    print(f"[graph_builder]   train_mask    : {train_mask.sum().item():,} nodes")
    print(f"[graph_builder]   val_mask      : {val_mask.sum().item():,} nodes")
    print(f"[graph_builder]   test_mask     : {test_mask.sum().item():,} nodes")

    return X, y, edge_index, train_mask, val_mask, test_mask, meta


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_with_masks(source_csv: str) -> tuple:
    """
    Load and encode all three splits, stack into one graph, return boolean masks.

    SOURCE='clean':   reads the full clean CSV; masks derived from split CSVs.
    SOURCE='mcar_*':  reads train (source_csv) + val + test from base folder;
                      stacks them train | val | test; masks are contiguous blocks.
    """
    is_clean = (config.SOURCE == "clean")

    if is_clean:
        df_all = pd.read_csv(source_csv)
        print(f"[graph_builder] Rows loaded (full clean) : {len(df_all):,}")

        nan_cells = df_all.drop(columns=[config.TARGET_COL]).isnull().sum().sum()
        print(f"[graph_builder] NaN cells                : {nan_cells:,}")

        df_train_split = pd.read_csv(config.TRAIN_CSV_FILE)
        df_val_split   = pd.read_csv(config.VAL_CSV_FILE)
        df_test_split  = pd.read_csv(config.TEST_CSV_FILE)

        train_idx = _match_indices(df_all, df_train_split, _used := set())
        val_idx   = _match_indices(df_all, df_val_split,   _used)
        test_idx  = _match_indices(df_all, df_test_split,  _used)

        # Reorder as [train | val | test] — mirrors MCAR path layout so
        # n_train_rows correctly restricts scaler/encoder fit to training rows only
        ordered_idx = train_idx + val_idx + test_idx
        df_all = df_all.iloc[ordered_idx].reset_index(drop=True)

        n_train = len(train_idx)
        n_val   = len(val_idx)
        n       = len(df_all)

        train_mask = _make_mask(n, list(range(0,               n_train)))
        val_mask   = _make_mask(n, list(range(n_train,         n_train + n_val)))
        test_mask  = _make_mask(n, list(range(n_train + n_val, n)))

        X, y, feature_cols, null_mask = _encode(df_all, n_train_rows=n_train)

    else:
        df_train = pd.read_csv(source_csv)
        df_val   = pd.read_csv(config.VAL_CSV_FILE)
        df_test  = pd.read_csv(config.TEST_CSV_FILE)

        n_train = len(df_train)
        n_val   = len(df_val)
        n_test  = len(df_test)
        n_total = n_train + n_val + n_test

        train_nan = df_train.drop(columns=[config.TARGET_COL]).isnull().sum().sum()
        val_nan   = df_val  .drop(columns=[config.TARGET_COL]).isnull().sum().sum()
        test_nan  = df_test .drop(columns=[config.TARGET_COL]).isnull().sum().sum()

        print(f"[graph_builder] Rows loaded  : train={n_train:,}  val={n_val:,}  test={n_test:,}  total={n_total:,}")
        print(f"[graph_builder] NaN cells    : train={train_nan:,}  val={val_nan}  test={test_nan}")

        if val_nan > 0 or test_nan > 0:
            raise ValueError(
                f"[graph_builder] ✗ Val ({val_nan}) / test ({test_nan}) splits contain NaNs.\n"
                f"  These splits must always be null-free. Check null_injector output."
            )

        if train_nan > 0 and config.RESOLVED_IMPUTATION == "none":
            print(f"[graph_builder] Sentinel mode : {train_nan:,} NaNs will be encoded as sentinel=-1. "
                  f"KNN graph built with null-aware distance (sentinel→0 contribution).")

        # Contiguous masks: train | val | test
        train_mask = _make_mask(n_total, list(range(0,               n_train)))
        val_mask   = _make_mask(n_total, list(range(n_train,         n_train + n_val)))
        test_mask  = _make_mask(n_total, list(range(n_train + n_val, n_total)))

        # Stack all three splits; encode using train-only statistics
        df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
        X, y, feature_cols, null_mask = _encode(df_all, n_train_rows=n_train)

    _print_split_summary(y, train_mask, val_mask, test_mask)

    # Save sensitive attribute (aligned to node order) for fairness analysis
    from fairness import SENSITIVE_COLS
    sens_col = SENSITIVE_COLS.get(config.DATASET_NAME)
    if sens_col and sens_col in df_all.columns:
        sens_arr = df_all[sens_col].astype(str).values
        np.save(config.SENSITIVE_ATTR_F, sens_arr)
        print(f"[graph_builder]   ✓ {os.path.basename(config.SENSITIVE_ATTR_F)}  "
              f"{sens_arr.shape}  (groups: {sorted(set(sens_arr))})")

    # Keep a copy of the raw feature values (before encoding) for diagnostics
    df_raw_features = df_all.drop(columns=[config.TARGET_COL]).copy()

    return X, y, feature_cols, train_mask, val_mask, test_mask, null_mask, df_raw_features


def _encode(df: pd.DataFrame, n_train_rows: int = None) -> tuple:
    """
    Encode a DataFrame (possibly containing NaNs) into feature matrix X and labels y.

    NULL SENTINEL STRATEGY:
        ALL NaN values (categorical AND numerical) are replaced with a
        sentinel value of -1 AFTER encoding/scaling.  This guarantees:
          • No np.nan flows into PyTorch tensors
          • The sentinel sits outside the natural range of both
            label-encoded categoricals (0, 1, 2, …) and z-scored
            numericals (centred around 0, rarely < -3)
          • A companion null_mask (bool matrix, same shape as X) tracks
            which cells were originally missing so the distance function
            can neutralise them

    n_train_rows : if set, fit statistics on training rows only to avoid leakage.
                   If None, the full df is used (clean source).
    """
    NULL_SENTINEL = -1.0

    df = df.copy()
    df[config.TARGET_COL] = df[config.TARGET_COL].map(config.TARGET_MAP)
    y = df.pop(config.TARGET_COL).values.astype(np.int64)

    unique, counts = np.unique(y, return_counts=True)
    print(f"[graph_builder] Label distribution : "
          + "  ".join(f"{n}={c:,}" for n, c in zip(config.CLASS_NAMES, counts)))

    num_cols = [c for c in df.columns if c in config.NUMERICAL_COLS]
    cat_cols = [c for c in df.columns if c not in config.NUMERICAL_COLS]
    feature_cols = list(df.columns)

    nan_cells = df.isnull().sum().sum()
    if nan_cells > 0:
        print(f"[graph_builder] Encoding with NaNs : {nan_cells:,} cells → "
              f"ALL NaN → sentinel={NULL_SENTINEL} (categorical & numerical)")

    # ── Record which cells are NaN BEFORE any encoding ────────────────────────
    null_mask_df = df.isnull()   # True where originally missing

    # ── Reference rows for fitting statistics (train only if available) ───────
    df_ref = df.iloc[:n_train_rows] if n_train_rows else df

    # ── Categorical columns: observed → LabelEncoder, NaN stays NaN for now ──
    for col in cat_cols:
        le = LabelEncoder()
        ref_vals = df_ref[col].dropna().astype(str)
        if len(ref_vals) == 0:
            ref_vals = pd.Series(["__empty__"])
        le.fit(ref_vals)
        known = set(le.classes_)
        # Transform observed values; keep NaN as NaN (filled with sentinel later)
        def _safe_transform(x, _le=le, _known=known):
            if pd.isna(x):
                return np.nan
            s = str(x)
            if s not in _known:
                return 0.0   # fallback: first class
            return float(_le.transform([s])[0])
        df[col] = df[col].apply(_safe_transform)

    # ── Numerical columns: fit scaler on observed train values ────────────────
    for col in num_cols:
        ref_obs = df_ref[col].dropna().values.reshape(-1, 1)
        if len(ref_obs) == 0:
            print(f"[graph_builder]   ⚠ Column '{col}' has no observed values in train — skipping scale")
            continue
        scaler = StandardScaler()
        scaler.fit(ref_obs)
        # Pandas 2.x refuses to assign float64 (scaler output) into an int64
        # column — cast the column to float first so the scaled values land
        # in a float-dtype block.  This path is exercised on the "clean"
        # source where numerical columns (age, fnlwgt, ...) arrive as int64;
        # on MCAR paths the introduced NaNs have already coerced them to
        # float, so this cast is a no-op.
        if pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = df[col].astype("float64")
        obs_mask = df[col].notna()
        df.loc[obs_mask, col] = scaler.transform(
            df.loc[obs_mask, col].values.reshape(-1, 1)
        ).flatten()
        # NaN positions stay NaN — filled with sentinel below

    # ── Build feature matrix and null mask ─────────────────────────────────────
    X = df.values.astype(np.float32)
    null_mask = np.isnan(X)    # True where sentinel should go
    n_nulls = int(null_mask.sum())

    # Replace remaining NaN with sentinel.
    # If using indicators, we can use 0.0 for numerical features (the mean).
    # Otherwise, we use -1.0 for everything to stay outside valid ranges.
    use_indicators = getattr(config, "APPEND_MISSING_MASK", False)
    for col_idx, col_name in enumerate(feature_cols):
        col_nan_mask = null_mask[:, col_idx]
        if col_nan_mask.any():
            if use_indicators and col_name in num_cols:
                X[col_nan_mask, col_idx] = 0.0
            else:
                X[col_nan_mask, col_idx] = -1.0

    print(f"[graph_builder] Feature matrix     : {X.shape}  (nodes × features)")
    print(f"[graph_builder] Null sentinels     : {n_nulls:,} cells (num=0.0, cat=-1.0)")
    print(f"[graph_builder] NaN in X after fill: {int(np.isnan(X).sum())}  (should be 0)")

    # ── Print first 5 vectors that contain nulls ──────────────────────────────
    if n_nulls > 0:
        # Use a descriptive string for sentinel display in log
        _print_null_vectors(X, null_mask, feature_cols, "0.0 or -1.0", max_show=5)

    return X, y, feature_cols, null_mask


def _print_null_vectors(X, null_mask, feature_cols, sentinel, max_show=5):
    """
    Print detailed information about the first `max_show` vectors (rows)
    that contain at least one null-sentinel value.

    For each vector shows:
      • The full encoded feature vector
      • Which dimensions are real values vs sentinel (null) placeholders
      • How the null-aware distance function will treat each dimension
    """
    null_rows = np.where(null_mask.any(axis=1))[0]
    n_show = min(max_show, len(null_rows))

    print(f"\n{'─'*75}")
    print(f"  NULL-SENTINEL VECTOR INSPECTION  (first {n_show} of {len(null_rows):,} rows with nulls)")
    print(f"{'─'*75}")
    print(f"  Sentinel value = {sentinel}")
    print(f"  Distance rule  : if ONE vector has sentinel → distance contribution = 0"
          f"\n                   if BOTH observed → (Δ)², rescaled by d_total/d_used")
    print(f"                   if BOTH vectors have sentinel → dimension SKIPPED")
    print(f"{'─'*75}")

    for i in range(n_show):
        row_idx = null_rows[i]
        vec     = X[row_idx]
        mask    = null_mask[row_idx]
        n_null  = int(mask.sum())
        n_obs   = len(vec) - n_null

        print(f"\n  ┌─ Row {row_idx}  ({n_null} null / {n_obs} observed / {len(vec)} total) ─┐")
        print(f"  │ {'Dim':>4}  {'Feature':<22} {'Value':>10}  {'Status':<12} {'Distance role'}")
        print(f"  │ {'─'*70}")

        for d in range(len(vec)):
            fname = feature_cols[d] if d < len(feature_cols) else f"feat_{d}"
            val   = vec[d]
            if mask[d]:
                status = "◆ NULL"
                role   = "→ contributes 0 to distance (or skipped if partner also null)"
            else:
                status = "● observed"
                role   = "→ (Δ)² (squared diff, rescaled)"
            print(f"  │ {d:>4}  {fname:<22} {val:>10.4f}  {status:<12} {role}")

        print(f"  └{'─'*73}┘")

    # Summary
    nulls_per_row = null_mask[null_rows[:n_show]].sum(axis=1)
    print(f"\n  Summary of shown rows:")
    for i in range(n_show):
        print(f"    Row {null_rows[i]:>6} : {int(nulls_per_row[i]):>2} null dims out of {X.shape[1]}")
    print(f"{'─'*75}\n")


def _print_neighbor_distances(X, null_mask, edge_index, feature_cols=None,
                               df_raw=None, max_nodes=3, max_neighbors=3):
    """
    After the KNN graph is built, pick sample nodes that have nulls and
    print:
      1. The ACTUAL (raw) row from the CSV — so you see the real values and NaNs
      2. Each neighbour's ACTUAL (raw) row
      3. The per-dimension distance breakdown proving the three rules

    df_raw : DataFrame of raw feature values (before encoding), same row order as X.
             If None, raw rows are not printed.
    """
    n, d = X.shape
    ei_src = edge_index[0].numpy()
    ei_dst = edge_index[1].numpy()

    # Find nodes with nulls
    null_rows = np.where(null_mask.any(axis=1))[0]
    if len(null_rows) == 0:
        return

    show_nodes = null_rows[:max_nodes]
    raw_cols = list(df_raw.columns) if df_raw is not None else []

    print(f"\n{'═'*95}")
    print(f"  NEIGHBOR DISTANCE VERIFICATION  (null-aware distance proof)")
    print(f"  Showing {len(show_nodes)} sample nodes × up to {max_neighbors} neighbours each")
    print(f"{'═'*95}")
    print(f"  Rules:  both_observed → (x_i - x_j)²    one_null → 0.0    both_null → SKIP")
    print(f"{'═'*95}")

    for node_idx in show_nodes:
        # Get this node's neighbors from edge_index
        neighbor_mask_arr = ei_src == node_idx
        neighbors = ei_dst[neighbor_mask_arr]

        if len(neighbors) == 0:
            continue

        # Compute actual distances to all neighbors and sort by distance
        dists_to_neighbors = []
        for nb in neighbors:
            dist_val, _, _, _ = _compute_pair_distance(X, null_mask, node_idx, nb, d)
            dists_to_neighbors.append((nb, dist_val))
        dists_to_neighbors.sort(key=lambda x: x[1])

        # Show top max_neighbors
        show_nbs = dists_to_neighbors[:max_neighbors]

        node_nulls = int(null_mask[node_idx].sum())
        node_obs   = d - node_nulls

        print(f"\n  ┌─ Node {node_idx}  ({node_nulls} null / {node_obs} observed)  "
              f"│  {len(neighbors)} total neighbours in graph ─┐")

        # ── Print the ACTUAL raw row for this node ────────────────────────
        if df_raw is not None and node_idx < len(df_raw):
            print(f"  │")
            print(f"  │  ACTUAL ROW (raw CSV values):")
            raw_row = df_raw.iloc[node_idx]
            for col in raw_cols:
                val = raw_row[col]
                if pd.isna(val):
                    print(f"  │    {col:<22} = {'NaN':<20} ◆ MISSING")
                else:
                    print(f"  │    {col:<22} = {val}")

        for rank, (nb_idx, total_dist) in enumerate(show_nbs):
            nb_nulls = int(null_mask[nb_idx].sum())
            nb_obs   = d - nb_nulls

            dist_val, details, d_used, d_skipped = _compute_pair_distance(
                X, null_mask, node_idx, nb_idx, d
            )

            print(f"  │")
            print(f"  │  {'─'*88}")
            print(f"  │  ── Neighbour #{rank+1}: Node {nb_idx}  "
                  f"({nb_nulls} null / {nb_obs} observed)  "
                  f"distance = {dist_val:.4f} ──")

            # ── Print the ACTUAL raw row for this neighbour ───────────────
            if df_raw is not None and nb_idx < len(df_raw):
                print(f"  │")
                print(f"  │  ACTUAL ROW (raw CSV values):")
                nb_raw_row = df_raw.iloc[nb_idx]
                for col in raw_cols:
                    val = nb_raw_row[col]
                    if pd.isna(val):
                        print(f"  │    {col:<22} = {'NaN':<20} ◆ MISSING")
                    else:
                        print(f"  │    {col:<22} = {val}")

            # ── Distance breakdown ────────────────────────────────────────
            print(f"  │")
            print(f"  │  DISTANCE BREAKDOWN  (Node {node_idx} vs Node {nb_idx}):")
            print(f"  │  {'Dim':>4}  {'Feature':<20} {'Node val':>9} {'Nbr val':>9}  "
                  f"{'Rule':<18} {'Contribution':>12}")
            print(f"  │  {'─'*78}")

            sum_contrib = 0.0
            for dim_info in details:
                dim_idx = dim_info['dim']
                fname = feature_cols[dim_idx] if feature_cols and dim_idx < len(feature_cols) else f"feat_{dim_idx}"
                v_i   = dim_info['val_i']
                v_j   = dim_info['val_j']
                rule  = dim_info['rule']
                contrib = dim_info['contrib']

                if rule == 'both_null':
                    contrib_str = "SKIPPED"
                    rule_str    = "both_null→SKIP"
                elif rule == 'one_null':
                    contrib_str = "0.0000"
                    rule_str    = "one_null→0"
                else:
                    contrib_str = f"{contrib:.4f}"
                    rule_str    = "both_obs→(Δ)²"
                    sum_contrib += contrib

                # Mark which side is null
                null_marker_i = " ◆" if null_mask[node_idx, dim_idx] else ""
                null_marker_j = " ◆" if null_mask[nb_idx, dim_idx] else ""

                print(f"  │  {dim_idx:>4}  {fname:<20} {v_i:>8.4f}{null_marker_i:<2}"
                      f"{v_j:>8.4f}{null_marker_j:<2}  {rule_str:<18} {contrib_str:>12}")

            print(f"  │  {'─'*78}")
            print(f"  │  Sum of contributions             : {sum_contrib:.4f}")
            print(f"  │  Dimensions used / total         : {d_used} / {d}  "
                  f"({d_skipped} skipped = both_null)")
            print(f"  │  Rescaled distance                : "
                  f"sqrt({sum_contrib:.4f} × {d} / {d_used}) = {dist_val:.4f}")

        print(f"  └{'─'*93}┘")

    print(f"{'═'*95}\n")


def _compute_pair_distance(X, null_mask, i, j, d_total):
    """
    Compute the null-aware distance between nodes i and j,
    returning the total distance plus per-dimension details.

    Distance per dimension:
        both_observed → (x_i - x_j)²   (squared diff, rescaled by d_total/d_used)
        one_null      → 0               (transparent)
        both_null     → SKIP            (excluded)

    Returns: (distance, details_list, d_used, d_skipped)
    """
    details = []
    sum_contrib = 0.0
    d_used    = 0
    d_skipped = 0

    for dim in range(d_total):
        null_i = null_mask[i, dim]
        null_j = null_mask[j, dim]
        v_i = X[i, dim]
        v_j = X[j, dim]

        info = {'dim': dim, 'feature': '', 'val_i': float(v_i), 'val_j': float(v_j)}

        if null_i and null_j:
            info['rule']    = 'both_null'
            info['contrib'] = 0.0
            d_skipped += 1
        elif null_i or null_j:
            info['rule']    = 'one_null'
            info['contrib'] = 0.0
            d_used += 1
        else:
            sq = (v_i - v_j) ** 2
            contrib = sq
            info['rule']    = 'both_observed'
            info['contrib'] = float(contrib)
            sum_contrib += contrib
            d_used += 1

        details.append(info)

    if d_used > 0:
        dist = float(np.sqrt(sum_contrib * d_total / d_used))
    else:
        dist = float('inf')

    return dist, details, d_used, d_skipped


def _build_knn_graph(X: np.ndarray, null_mask: np.ndarray,
                     feature_cols: list = None, df_raw: pd.DataFrame = None) -> torch.Tensor:
    """
    Build a symmetric KNN graph with null-aware distance.

    Distance function (per dimension d between nodes i and j):
        • BOTH observed  → (x_i[d] - x_j[d])² + 1  (squared diff + presence weight)
        • ONE is null    → 0    (null is transparent)
        • BOTH are null  → SKIP (dimension excluded entirely)

    The rescaling factor d_total/d_used ensures distances are comparable
    across pairs with different numbers of usable dimensions.

    The total distance is rescaled by d_total / d_used to keep distances
    comparable across pairs with different numbers of usable dimensions.

    When null_mask has no True values, falls back to sklearn's fast
    kneighbors_graph (no sentinel logic needed).
    """
    k       = config.K_NEIGHBORS
    has_null = bool(null_mask.any())

    if has_null:
        print(f"[graph_builder] Building null-aware KNN graph (k={k}) ... ", end="", flush=True)
        print(f"\n[graph_builder]   Distance rule: both_observed→(Δ)², one_null→0, both_null→skip")
        edge_index = _null_aware_knn(X, null_mask, k)
    else:
        from sklearn.neighbors import kneighbors_graph
        print(f"[graph_builder] Building KNN graph (k={k}) ... ", end="", flush=True)
        A  = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                              include_self=False, n_jobs=-1)
        A  = A.maximum(A.T)
        cx = A.tocoo()
        edge_index = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)

    print(f"done  →  {edge_index.shape[1]:,} edges")

    # Diagnostics
    n = X.shape[0]
    deg = np.zeros(n, dtype=int)
    src = edge_index[0].numpy()
    np.add.at(deg, src, 1)
    print(f"[graph_builder] Graph diagnostics :")
    print(f"[graph_builder]   Avg degree : {deg.mean():.2f}  "
          f"Min : {deg.min()}  Max : {deg.max()}")

    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        rows = edge_index[0].numpy()
        cols = edge_index[1].numpy()
        A_sp = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        n_comp, _ = connected_components(A_sp, directed=False)
        if n_comp == 1:
            print(f"[graph_builder]   Graph is fully connected (1 component)")
        else:
            print(f"[graph_builder]   ⚠ Graph has {n_comp} connected components "
                  f"(consider increasing K_NEIGHBORS)")
    except ImportError:
        pass

    # ── Neighbor distance verification ────────────────────────────────────────
    if has_null:
        _print_neighbor_distances(X, null_mask, edge_index, feature_cols=feature_cols,
                                  df_raw=df_raw, max_nodes=3, max_neighbors=3)

    # ── First-5-nodes neighbor summary with similarity scores ─────────────────
    _print_first_nodes_similarity(X, null_mask, edge_index, max_nodes=5)

    return edge_index


def _cosine_similarity(u, v):
    """Cosine similarity between two 1-D arrays. Returns 0 if either is a zero vector."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def _print_first_nodes_similarity(X, null_mask, edge_index, max_nodes=5):
    """
    For the first `max_nodes` nodes, print each neighbor and the cosine similarity score.

    Similarity = cosine similarity between node feature vectors.
    Range: [-1, 1] — higher means more similar direction in feature space.
    """
    n, d    = X.shape
    ei_src  = edge_index[0].numpy()
    ei_dst  = edge_index[1].numpy()
    has_null = bool(null_mask.any())

    print(f"\n{'═'*72}")
    print(f"  NODE–NEIGHBOR SIMILARITY  (first {min(max_nodes, n)} nodes)")
    # print(f"  Similarity = cosine similarity   range: [-1, 1]  — higher = more similar")
    if has_null:
        print(f"  Distance   = null-aware Euclidean (missing dims treated as transparent)")
    else:
        pass  # print(f"  Distance   = Euclidean")
    print(f"{'═'*72}")

    for node_idx in range(min(max_nodes, n)):
        neighbor_indices = ei_dst[ei_src == node_idx]

        if len(neighbor_indices) == 0:
            print(f"\n  Node {node_idx} : no neighbors")
            continue

        # Compute distance + cosine similarity to every neighbor
        rows = []
        for nb in neighbor_indices:
            if has_null:
                dist, _, _, _ = _compute_pair_distance(X, null_mask, node_idx, int(nb), d)
            else:
                dist = float(np.sqrt(np.sum((X[node_idx] - X[nb]) ** 2)))
            sim = _cosine_similarity(X[node_idx], X[nb])
            rows.append((int(nb), dist, sim))

        rows.sort(key=lambda r: r[2], reverse=True)   # highest similarity first

        print(f"\n  Node {node_idx}  →  {len(rows)} neighbor(s)")
        print(f"  {'Neighbor':>9}  {'Distance':>10}  {'Similarity':>10}")
        print(f"  {'─'*36}")
        for nb_idx, dist, sim in rows:
            print(f"  {nb_idx:>9}  {dist:>10.4f}  {sim:>10.4f}")

    print(f"\n{'═'*72}\n")


def _null_aware_knn(X: np.ndarray, null_mask: np.ndarray, k: int) -> torch.Tensor:
    """
    Null-aware KNN using sentinel-encoded features and a boolean null_mask.

    For each pair (i, j), the distance across d dimensions is:

        For each dimension d:
          - both_observed (null_mask[i,d]=F AND null_mask[j,d]=F):
                → (X[i,d] - X[j,d])²        (squared diff)
          - one_null (exactly one of null_mask[i,d], null_mask[j,d] is True):
                → 0                          (null is transparent)
          - both_null (null_mask[i,d]=T AND null_mask[j,d]=T):
                → SKIP                       (dimension excluded entirely)

        d_used = number of dimensions that are NOT both_null
        distance = sqrt( sum_of_contributions × d_total / d_used )

    The rescaling factor d_total / d_used compensates for missing dimensions,
    inflating the distance to account for dimensions that couldn't be compared.
    This is the standard null-aware Euclidean distance used in GRAPE, MIWAE,
    and most missing-data GNN literature.
    """
    n, d = X.shape
    chunk = max(1, min(500, n))
    rows_list, cols_list = [], []

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        Xi = X[start:end]                     # (chunk, d)
        Mi = null_mask[start:end]              # (chunk, d)  True = null

        # Broadcast: (chunk, n, d)
        null_i = Mi[:, np.newaxis, :]          # (chunk, 1, d)
        null_j = null_mask[np.newaxis, :, :]   # (1, n, d)

        both_observed = (~null_i) & (~null_j)  # both have real values
        both_null     = null_i & null_j        # both are sentinel — skip
        # one_null    = ~both_observed & ~both_null  (contributes 0)

        # Squared difference where both_observed, 0 elsewhere
        diff = Xi[:, np.newaxis, :] - X[np.newaxis, :, :]
        contrib = np.where(both_observed, diff ** 2, 0.0)

        # Count usable dimensions = d_total - both_null
        d_used = d - both_null.sum(axis=2).astype(np.float32)  # (chunk, n)

        # Rescaled distance; inf where no usable dimensions
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = np.sqrt(contrib.sum(axis=2) * d / np.where(d_used > 0, d_used, np.nan))

        # Self-distance = inf
        for local_i in range(end - start):
            dist[local_i, start + local_i] = np.inf

        # k nearest per row
        actual_k = min(k, n - 1)
        nn_idx   = np.argpartition(dist, actual_k, axis=1)[:, :actual_k]

        for local_i, neighbours in enumerate(nn_idx):
            global_i = start + local_i
            valid = [j for j in neighbours if np.isfinite(dist[local_i, j])]
            for j in valid:
                rows_list.append(global_i)
                cols_list.append(j)

    if not rows_list:
        rows_list = list(range(n - 1)) + list(range(1, n))
        cols_list = list(range(1, n)) + list(range(n - 1))

    rows_arr = np.array(rows_list, dtype=np.int64)
    cols_arr = np.array(cols_list, dtype=np.int64)

    # Symmetrise
    rows_sym = np.concatenate([rows_arr, cols_arr])
    cols_sym = np.concatenate([cols_arr, rows_arr])

    # Deduplicate
    pairs      = np.unique(np.stack([rows_sym, cols_sym], axis=1), axis=0)
    edge_index = torch.tensor(pairs.T, dtype=torch.long)
    return edge_index


def _match_indices(df_all: pd.DataFrame, df_split: pd.DataFrame,
                   _used: set | None = None) -> list:
    """Return row indices in df_all that correspond to rows in df_split.

    When duplicate rows exist in df_all, the merge can match the same
    df_all row to multiple splits.  The *_used* set (shared across all
    three calls) tracks which df_all indices have already been claimed
    so that every row is assigned to exactly one split.
    """
    if _used is None:
        _used = set()
    df_all_idx = df_all.reset_index().rename(columns={"index": "_idx"})
    merged     = df_split.merge(df_all_idx, on=list(df_split.columns), how="left")
    indices: list[int] = []
    for idx in merged["_idx"].dropna().astype(int):
        if idx not in _used:
            _used.add(idx)
            indices.append(idx)
    return indices


def _make_mask(n: int, indices: list) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    if indices:
        mask[torch.tensor(indices, dtype=torch.long)] = True
    return mask


def _print_split_summary(y, train_mask, val_mask, test_mask):
    y_np = y if isinstance(y, np.ndarray) else y.numpy()
    print(f"[graph_builder] Split summary:")
    for label, mask in [("train", train_mask), ("val  ", val_mask), ("test ", test_mask)]:
        idx  = mask.numpy().nonzero()[0]
        dist = {int(c): int((y_np[idx] == c).sum()) for c in np.unique(y_np)}
        print(f"[graph_builder]   {label} : {mask.sum().item():,} nodes  dist={dist}")


def _save_all(X, y, edge_index, train_mask, val_mask, test_mask, feature_cols, null_mask):
    """Save all .npy files with dataset-name prefix."""
    np.save(config.NODE_FEATURES_F, X)
    print(f"[graph_builder]   ✓ {os.path.basename(config.NODE_FEATURES_F)}  {X.shape}")

    np.save(config.NODE_LABELS_F, y)
    print(f"[graph_builder]   ✓ {os.path.basename(config.NODE_LABELS_F)}    {y.shape}")

    np.save(config.EDGE_INDEX_F, edge_index.numpy())
    print(f"[graph_builder]   ✓ {os.path.basename(config.EDGE_INDEX_F)}     {tuple(edge_index.shape)}")

    np.save(config.TRAIN_MASK_F, train_mask.numpy())
    np.save(config.VAL_MASK_F,   val_mask.numpy())
    np.save(config.TEST_MASK_F,  test_mask.numpy())
    print(f"[graph_builder]   ✓ train/val/test mask .npy files")

    # Save null mask alongside features
    null_mask_path = config.NODE_FEATURES_F.replace("_node_features.npy", "_null_mask.npy")
    np.save(null_mask_path, null_mask)
    n_sentinel = int(null_mask.sum())
    print(f"[graph_builder]   ✓ {os.path.basename(null_mask_path)}  {null_mask.shape}  "
          f"(sentinel cells={n_sentinel:,})")

    unique, counts = np.unique(y, return_counts=True)
    nan_in_X = int(np.isnan(X).sum())
    meta = {
        "dataset"            : config.DATASET_NAME,
        "source"             : config.SOURCE,
        "imputation"         : config.RESOLVED_IMPUTATION,
        "null_sentinel"      : -1.0,
        "null_sentinel_cells": n_sentinel,
        "null_aware_graph"   : bool(n_sentinel > 0),
        "nan_in_features"    : nan_in_X,
        "file_prefix"        : config._FILE_PREFIX,
        "num_nodes"          : int(X.shape[0]),
        "num_edges"          : int(edge_index.shape[1]),
        "num_features"       : int(X.shape[1]),
        "num_classes"        : int(len(unique)),
        "class_names"        : config.CLASS_NAMES,
        "k_neighbors"        : config.K_NEIGHBORS,
        "feature_cols"       : feature_cols,
        "numerical_cols"     : config.NUMERICAL_COLS,
        "split"              : {
            "train": int(train_mask.sum()),
            "val"  : int(val_mask.sum()),
            "test" : int(test_mask.sum()),
        },
        "class_distribution" : dict(zip(config.CLASS_NAMES, counts.tolist())),
    }
    with open(config.METADATA_F, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[graph_builder]   ✓ {os.path.basename(config.METADATA_F)}")


def _print_existing_files():
    for fpath in config.NPY_FILES:
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"[graph_builder]   {os.path.basename(fpath):<45} {size:.1f} KB")


if __name__ == "__main__":
    build_graph_files()
