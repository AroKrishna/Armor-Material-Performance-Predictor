"""
data_utils.py

Single source of truth for data loading, cleaning, feature definitions,
and train/val/test splitting for the Armor V50 Prediction project.

All other scripts (train.py, test.py, main.py) should import FEATURES,
TARGET, and the helper functions from this module instead of redefining
their own copies -- this keeps preprocessing and feature order consistent
across the whole pipeline and avoids silent drift between scripts.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# ----------------------------------------------------------------------
# Feature / target definitions
# ----------------------------------------------------------------------

# Intrinsic material properties (constant for a given material/temper)
MATERIAL_FEATURES = [
    'thickness', 'density', 'modulus', 'hardness',
    'yield', 'PR', 'UTS', 'elongation',
]

# Threat/projectile properties -- these vary per test and materially affect
# V50, but were NOT used as model inputs in the original project.
PROJECTILE_FEATURES = ['calibre', 'proj_mass', 'proj_density', 'proj_hardness']

# Impact scenario properties
SCENARIO_FEATURES = ['angle']

# Full feature set used by the model
FEATURES = MATERIAL_FEATURES + PROJECTILE_FEATURES + SCENARIO_FEATURES

TARGET = 'v50'

# Column used to group rows for leakage-safe splitting (same material should
# not appear in both train and test, since properties are near-identical
# across its rows).
GROUP_COL = 'Material'

# All columns that must be numeric for the pipeline to work
NUMERIC_COLUMNS = FEATURES + [TARGET]

DEFAULT_RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Loading & cleaning
# ----------------------------------------------------------------------

def _strip_whitespace_chars(series: pd.Series) -> pd.Series:
    """
    Strip regular whitespace AND non-breaking spaces (\\xa0) from a string
    series before numeric conversion.

    The raw CSV contains ~25 values per affected column formatted like
    '10.3\\xa0' (a trailing non-breaking space). pd.to_numeric() silently
    coerces these to NaN, and a naive dropna() then silently discards the
    row. Stripping this character first recovers those rows instead of
    losing them.
    """
    return series.astype(str).str.replace('\xa0', '', regex=False).str.strip()


def load_raw_dataset(csv_path: str = 'Dataset.csv') -> pd.DataFrame:
    """Load the dataset exactly as stored on disk, no cleaning applied."""
    return pd.read_csv(csv_path)


def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Cleans the raw dataframe:
      - strips non-breaking spaces / whitespace from numeric-looking columns
      - coerces all FEATURES + TARGET columns to numeric
      - drops rows that are still unusable (missing/non-numeric) after that

    Returns a new dataframe; does not mutate the input in place.
    """
    df = df.copy()
    n_before = len(df)

    missing_cols = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Dataset is missing expected column(s): {missing_cols}. "
            f"Check that Dataset.csv matches the expected schema."
        )

    for col in NUMERIC_COLUMNS:
        # Columns may already be numeric (int64/float64), or may be stored
        # as text (numpy 'object' dtype, or pandas' newer native 'str'
        # dtype depending on pandas version) -- strip whitespace/NBSP for
        # any non-numeric dtype before attempting numeric conversion.
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = _strip_whitespace_chars(df[col])
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_bad = int(df[NUMERIC_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=NUMERIC_COLUMNS).reset_index(drop=True)

    if verbose:
        print(
            f"[data_utils] Loaded {n_before} rows -> {len(df)} rows after cleaning "
            f"({n_bad} row(s) dropped due to missing/unparseable values)."
        )

    return df


def load_clean_dataset(csv_path: str = 'Dataset.csv', verbose: bool = True) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    df = load_raw_dataset(csv_path)
    return clean_dataset(df, verbose=verbose)


# ----------------------------------------------------------------------
# Feature / target extraction
# ----------------------------------------------------------------------

def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Returns the model input matrix X, with columns in FEATURES order."""
    return df[FEATURES].values.astype(float)


def get_target_vector(df: pd.DataFrame) -> np.ndarray:
    """Returns the target vector y as a column vector, shape (n, 1)."""
    return df[[TARGET]].values.astype(float)


# ----------------------------------------------------------------------
# Leakage-safe splitting
# ----------------------------------------------------------------------

def grouped_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    group_col: str = GROUP_COL,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Splits df into (train_df, val_df, test_df) such that all rows sharing
    the same `group_col` value (i.e. the same Material) fall entirely
    within one split. This prevents the same material's near-duplicate
    rows (same properties, different thickness/angle) from leaking across
    train and test, which would otherwise inflate reported performance.

    Split is deterministic given `random_state`.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")

    groups = df[group_col].values

    # Step 1: carve off the test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(df, groups=groups))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: carve val out of the remaining train+val pool
    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    trainval_groups = trainval_df[group_col].values
    train_idx, val_idx = next(gss_val.split(trainval_df, groups=trainval_groups))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    # Sanity check: no material should ever appear in more than one split
    train_mats = set(train_df[group_col])
    val_mats = set(val_df[group_col])
    test_mats = set(test_df[group_col])
    assert not (train_mats & val_mats), "Leakage detected between train and val splits!"
    assert not (train_mats & test_mats), "Leakage detected between train and test splits!"
    assert not (val_mats & test_mats), "Leakage detected between val and test splits!"

    return train_df, val_df, test_df


def grouped_kfold_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = GROUP_COL,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Generator yielding (train_fold_df, val_fold_df) for grouped k-fold
    cross-validation, guaranteeing no Material appears in both the train
    and validation portion of a given fold.

    sklearn's GroupKFold has no built-in shuffling/random_state, so the
    dataframe is shuffled once up front (seeded) to randomize fold
    composition reproducibly.
    """
    rng = np.random.RandomState(random_state)
    shuffled_idx = rng.permutation(len(df))
    df_shuffled = df.iloc[shuffled_idx].reset_index(drop=True)
    groups_shuffled = df_shuffled[group_col].values

    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(df_shuffled, groups=groups_shuffled):
        train_fold = df_shuffled.iloc[train_idx].reset_index(drop=True)
        val_fold = df_shuffled.iloc[val_idx].reset_index(drop=True)
        yield train_fold, val_fold


# ----------------------------------------------------------------------
# Helpers for the optimization stage (main.py)
# ----------------------------------------------------------------------

def feature_ranges(df: pd.DataFrame) -> dict:
    """
    Per-feature (min, max) observed in df. Useful for flagging when a
    prediction request extrapolates beyond the training data's coverage.
    """
    return {col: (float(df[col].min()), float(df[col].max())) for col in FEATURES}


def material_property_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per Material with:
      - its intrinsic properties averaged across all recorded tests
        (density, modulus, hardness, yield, PR, UTS, elongation)
      - the observed thickness range (thickness_min, thickness_max) for
        that material

    Thickness is deliberately NOT collapsed into a single averaged value.
    Thickness is a design variable an armor designer chooses, not a fixed
    material property -- callers doing optimization (main.py) should sweep
    thickness within [thickness_min, thickness_max] (or a physically
    reasonable extension of it) rather than plug in one historical average.
    """
    intrinsic_cols = ['density', 'modulus', 'hardness', 'yield', 'PR', 'UTS', 'elongation']
    intrinsic = df.groupby(GROUP_COL)[intrinsic_cols].mean()
    thickness_range = (
        df.groupby(GROUP_COL)['thickness']
        .agg(['min', 'max'])
        .rename(columns={'min': 'thickness_min', 'max': 'thickness_max'})
    )
    table = intrinsic.join(thickness_range).reset_index()
    return table


# ----------------------------------------------------------------------
# Model/scaler metadata persistence
# ----------------------------------------------------------------------

def save_metadata(
    path: str,
    scaler_X_path: str,
    scaler_y_path: str,
    model_path: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    extra: dict | None = None,
) -> dict:
    """
    Persists the feature order and artifact file paths alongside the
    trained model. test.py and main.py should load this instead of
    hardcoding their own copy of the feature list, so the two can never
    silently drift out of sync with what the model was actually trained on.
    """
    metadata = {
        'features': FEATURES,
        'material_features': MATERIAL_FEATURES,
        'projectile_features': PROJECTILE_FEATURES,
        'scenario_features': SCENARIO_FEATURES,
        'target': TARGET,
        'group_col': GROUP_COL,
        'scaler_X_path': scaler_X_path,
        'scaler_y_path': scaler_y_path,
        'model_path': model_path,
        'random_state': random_state,
    }
    if extra:
        metadata.update(extra)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata


def load_metadata(path: str = 'metadata.json') -> dict:
    with open(path, 'r') as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Self-test when run directly: python data_utils.py
# ----------------------------------------------------------------------

if __name__ == '__main__':
    df = load_clean_dataset('Dataset.csv')
    print(f"Total cleaned rows: {len(df)}")
    print(f"Unique materials: {df[GROUP_COL].nunique()}")

    train_df, val_df, test_df = grouped_train_val_test_split(df)
    print(f"Train: {len(train_df):4d} rows / {train_df[GROUP_COL].nunique():2d} materials")
    print(f"Val:   {len(val_df):4d} rows / {val_df[GROUP_COL].nunique():2d} materials")
    print(f"Test:  {len(test_df):4d} rows / {test_df[GROUP_COL].nunique():2d} materials")

    print("\nFeature ranges (full cleaned dataset):")
    for name, (lo, hi) in feature_ranges(df).items():
        print(f"  {name:15s}: [{lo:.3f}, {hi:.3f}]")

    print("\nSample of material_property_table():")
    print(material_property_table(df).head())