"""
train.py

Trains the ArmorNet V50 regressor on Dataset.csv, using:
  - a leakage-safe, grouped (by Material) train/val/test split, with the
    test split persisted to disk so test.py evaluates on truly unseen data
  - grouped k-fold cross-validation on the train+val pool, to get a robust,
    reportable performance estimate (rather than a single noisy split)
  - early stopping on a held-out validation set for the final model
  - fixed random seeds throughout, for reproducibility
  - a Random Forest baseline trained/evaluated alongside the MLP, so the
    choice of model architecture is justified by evidence rather than
    assumed

Artifacts produced (all in the current working directory):
  armor_model.pth   - trained ArmorNet weights (final model)
  scaler_X.pkl      - StandardScaler fit on final training features
  scaler_y.pkl      - StandardScaler fit on final training target
  rf_baseline.pkl   - trained RandomForestRegressor baseline
  metadata.json     - feature list, architecture config, CV results,
                       final metrics, and the recommended model
  test_split.csv    - the held-out test rows (test.py must load ONLY this,
                       never the full Dataset.csv, or its metrics are
                       meaningless)
  training_curve.png- train/val loss curve for the final MLP
"""

import copy
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from data_utils import (
    DEFAULT_RANDOM_STATE,
    FEATURES,
    get_feature_matrix,
    get_target_vector,
    grouped_kfold_indices,
    grouped_train_val_test_split,
    load_clean_dataset,
    save_metadata,
)
from model import ArmorNet

SEED = DEFAULT_RANDOM_STATE

# MLP training hyperparameters
HIDDEN_SIZES = (128, 64)
DROPOUT = 0.15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 1000
PATIENCE = 40  # epochs of no val-loss improvement before early stopping

# Random Forest baseline hyperparameters
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None

CV_N_SPLITS = 5


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# MLP training with early stopping
# ----------------------------------------------------------------------

def train_mlp(X_train, y_train, X_val, y_val, input_size, hidden_sizes=HIDDEN_SIZES,
              dropout=DROPOUT, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
              max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED, verbose=False):
    """
    Trains an ArmorNet on (X_train, y_train), monitoring loss on
    (X_val, y_val) each epoch. Keeps the weights from the epoch with the
    lowest validation loss and restores them before returning (early
    stopping). All inputs are expected to already be scaled.
    """
    torch.manual_seed(seed)
    model = ArmorNet(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:4d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    return model, history, best_val_loss


def evaluate_mlp(model, X_scaled, y_true_raw, scaler_y) -> dict:
    """Evaluates a trained MLP on already-scaled features, comparing
    predictions (inverse-scaled back to physical units) against the
    original-scale ground truth."""
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.FloatTensor(X_scaled)).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = np.asarray(y_true_raw).flatten()
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
    }


def evaluate_rf(model, X_raw, y_true_raw) -> dict:
    """Evaluates a trained Random Forest (which needs no feature scaling)
    against original-scale ground truth."""
    y_pred = model.predict(X_raw)
    y_true = np.asarray(y_true_raw).flatten()
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------------
# Grouped cross-validation (MLP vs Random Forest)
# ----------------------------------------------------------------------

def run_cross_validation(trainval_df, n_splits=CV_N_SPLITS, seed=SEED):
    """
    Runs grouped k-fold CV over the train+val pool (never touching the
    held-out test set), training a fresh MLP and Random Forest per fold.
    Returns per-fold metric dicts for both models so they can be compared
    fairly on identical folds.
    """
    mlp_fold_metrics, rf_fold_metrics = [], []

    print(f"\n=== Grouped {n_splits}-Fold Cross-Validation (train+val pool) ===")
    for fold_i, (fold_train, fold_val) in enumerate(
        grouped_kfold_indices(trainval_df, n_splits=n_splits, random_state=seed), start=1
    ):
        X_train_raw = get_feature_matrix(fold_train)
        y_train_raw = get_target_vector(fold_train)
        X_val_raw = get_feature_matrix(fold_val)
        y_val_raw = get_target_vector(fold_val)

        scaler_X = StandardScaler().fit(X_train_raw)
        scaler_y = StandardScaler().fit(y_train_raw)
        X_train_s = scaler_X.transform(X_train_raw)
        y_train_s = scaler_y.transform(y_train_raw)
        X_val_s = scaler_X.transform(X_val_raw)

        mlp, _, _ = train_mlp(
            X_train_s, y_train_s, X_val_s, scaler_y.transform(y_val_raw),
            input_size=X_train_s.shape[1], seed=seed,
        )
        mlp_metrics = evaluate_mlp(mlp, X_val_s, y_val_raw, scaler_y)
        mlp_fold_metrics.append(mlp_metrics)

        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            random_state=seed, n_jobs=-1,
        )
        rf.fit(X_train_raw, y_train_raw.ravel())
        rf_metrics = evaluate_rf(rf, X_val_raw, y_val_raw)
        rf_fold_metrics.append(rf_metrics)

        print(f"  Fold {fold_i}/{n_splits} ({len(fold_train)} train / {len(fold_val)} val rows): "
              f"MLP R2={mlp_metrics['r2']:.3f} MAE={mlp_metrics['mae']:.1f} m/s  |  "
              f"RF R2={rf_metrics['r2']:.3f} MAE={rf_metrics['mae']:.1f} m/s")

    return mlp_fold_metrics, rf_fold_metrics


def summarize_cv(name: str, fold_metrics: list) -> dict:
    mae = np.array([m['mae'] for m in fold_metrics])
    rmse = np.array([m['rmse'] for m in fold_metrics])
    r2 = np.array([m['r2'] for m in fold_metrics])
    print(f"  {name:14s}: MAE={mae.mean():7.2f} ± {mae.std():5.2f}  |  "
          f"RMSE={rmse.mean():7.2f} ± {rmse.std():5.2f}  |  "
          f"R2={r2.mean():.3f} ± {r2.std():.3f}")
    return {
        'mae_mean': float(mae.mean()), 'mae_std': float(mae.std()),
        'rmse_mean': float(rmse.mean()), 'rmse_std': float(rmse.std()),
        'r2_mean': float(r2.mean()), 'r2_std': float(r2.std()),
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_training_curve(history: dict, out_path: str = 'training_curve.png'):
    plt.figure(figsize=(9, 5.5))
    plt.plot(history['train_loss'], label='Train loss (MSE, scaled target)')
    plt.plot(history['val_loss'], label='Val loss (MSE, scaled target)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Final Model Training Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved training curve to '{out_path}'")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    set_seed(SEED)

    print("=== Loading & Cleaning Dataset ===")
    df = load_clean_dataset('Dataset.csv')

    print("\n=== Grouped Train / Val / Test Split (by Material) ===")
    train_df, val_df, test_df = grouped_train_val_test_split(
        df, val_size=0.15, test_size=0.15, random_state=SEED,
    )
    print(f"Train: {len(train_df):4d} rows / {train_df['Material'].nunique():2d} materials")
    print(f"Val:   {len(val_df):4d} rows / {val_df['Material'].nunique():2d} materials")
    print(f"Test:  {len(test_df):4d} rows / {test_df['Material'].nunique():2d} materials")

    test_split_path = 'test_split.csv'
    test_df.to_csv(test_split_path, index=False)
    print(f"Held-out test split saved to '{test_split_path}' "
          f"(test.py must load ONLY this file, never the full dataset).")

    # ------------------------------------------------------------------
    # Cross-validation on train+val pool (test set never touched here)
    # ------------------------------------------------------------------
    trainval_df = pd_concat_reset(train_df, val_df)
    mlp_fold_metrics, rf_fold_metrics = run_cross_validation(trainval_df, n_splits=CV_N_SPLITS, seed=SEED)

    print("\n--- Cross-Validation Summary ---")
    mlp_cv_summary = summarize_cv('ArmorNet (MLP)', mlp_fold_metrics)
    rf_cv_summary = summarize_cv('Random Forest', rf_fold_metrics)

    recommended_model = 'random_forest' if rf_cv_summary['r2_mean'] > mlp_cv_summary['r2_mean'] else 'mlp'
    print(f"\nBased on mean CV R2, the recommended model is: {recommended_model.upper()}")
    print("(Both models are still trained and saved below, so main.py/test.py can use either.)")

    # ------------------------------------------------------------------
    # Final training on the fixed train split, early-stopped on val split
    # ------------------------------------------------------------------
    print("\n=== Final Model Training (train split, early-stopped on val split) ===")
    X_train_raw = get_feature_matrix(train_df)
    y_train_raw = get_target_vector(train_df)
    X_val_raw = get_feature_matrix(val_df)
    y_val_raw = get_target_vector(val_df)

    scaler_X = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)

    X_train_s = scaler_X.transform(X_train_raw)
    y_train_s = scaler_y.transform(y_train_raw)
    X_val_s = scaler_X.transform(X_val_raw)
    y_val_s = scaler_y.transform(y_val_raw)

    final_mlp, history, best_val_loss = train_mlp(
        X_train_s, y_train_s, X_val_s, y_val_s,
        input_size=X_train_s.shape[1], seed=SEED, verbose=True,
    )
    final_mlp_val_metrics = evaluate_mlp(final_mlp, X_val_s, y_val_raw, scaler_y)
    print(f"Final MLP  | val MAE={final_mlp_val_metrics['mae']:.2f} m/s  "
          f"RMSE={final_mlp_val_metrics['rmse']:.2f} m/s  R2={final_mlp_val_metrics['r2']:.3f}")

    final_rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=SEED, n_jobs=-1,
    )
    final_rf.fit(X_train_raw, y_train_raw.ravel())
    final_rf_val_metrics = evaluate_rf(final_rf, X_val_raw, y_val_raw)
    print(f"Final RF   | val MAE={final_rf_val_metrics['mae']:.2f} m/s  "
          f"RMSE={final_rf_val_metrics['rmse']:.2f} m/s  R2={final_rf_val_metrics['r2']:.3f}")

    plot_training_curve(history)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    print("\n=== Saving Artifacts ===")
    model_path = 'armor_model.pth'
    scaler_X_path = 'scaler_X.pkl'
    scaler_y_path = 'scaler_y.pkl'
    rf_path = 'rf_baseline.pkl'
    metadata_path = 'metadata.json'

    torch.save(final_mlp.state_dict(), model_path)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    joblib.dump(final_rf, rf_path)

    save_metadata(
        metadata_path,
        scaler_X_path=scaler_X_path,
        scaler_y_path=scaler_y_path,
        model_path=model_path,
        random_state=SEED,
        extra={
            'model_architecture': final_mlp.config(),
            'rf_path': rf_path,
            'rf_params': {'n_estimators': RF_N_ESTIMATORS, 'max_depth': RF_MAX_DEPTH},
            'test_split_path': test_split_path,
            'recommended_model': recommended_model,
            'cv_n_splits': CV_N_SPLITS,
            'cv_results': {'mlp': mlp_cv_summary, 'random_forest': rf_cv_summary},
            'final_val_metrics': {'mlp': final_mlp_val_metrics, 'random_forest': final_rf_val_metrics},
            'n_train_rows': len(train_df),
            'n_val_rows': len(val_df),
            'n_test_rows': len(test_df),
        },
    )

    print(f"Saved: {model_path}, {scaler_X_path}, {scaler_y_path}, {rf_path}, {metadata_path}")
    print("\nTraining complete. Run test.py to evaluate on the held-out test split.")


def pd_concat_reset(df_a, df_b):
    """Small local helper to avoid importing pandas just for one concat call
    at the top of the module (kept next to its only use for clarity)."""
    import pandas as pd
    return pd.concat([df_a, df_b], ignore_index=True)


if __name__ == '__main__':
    main()