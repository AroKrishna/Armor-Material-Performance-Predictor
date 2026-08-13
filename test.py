"""
test.py

Evaluates the trained ArmorNet (and the Random Forest baseline) on the
held-out test split produced by train.py.

IMPORTANT: this script loads ONLY 'test_split.csv' (the rows train.py
carved off and never trained or validated on), not the full Dataset.csv.
Evaluating on the full dataset -- as the original project's test.py did --
includes rows the model was trained on and produces meaningless, inflated
metrics. If 'test_split.csv' is missing, run train.py first.

Outputs:
  - printed MAE / RMSE / R2 for both the MLP and the Random Forest baseline
  - test_predictions.csv   - per-row actual vs. predicted values, both models
  - test_evaluation.png    - predicted-vs-actual and residual plots
"""

import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import FEATURES, GROUP_COL, TARGET, clean_dataset, load_metadata
from model import ArmorNet


def load_test_artifacts(metadata_path: str = 'metadata.json'):
    """Loads metadata, scalers, the trained MLP, and the RF baseline.
    Raises a clear error pointing at train.py if anything is missing."""
    try:
        metadata = load_metadata(metadata_path)
    except FileNotFoundError:
        print(f"Error: '{metadata_path}' not found. Please run train.py first.")
        sys.exit(1)

    if metadata['features'] != FEATURES:
        raise ValueError(
            "Feature list in metadata.json does not match data_utils.FEATURES. "
            "The model was trained with a different feature set than the one "
            "currently defined -- re-run train.py before testing."
        )

    try:
        scaler_X = joblib.load(metadata['scaler_X_path'])
        scaler_y = joblib.load(metadata['scaler_y_path'])
        rf_model = joblib.load(metadata['rf_path'])

        arch = metadata['model_architecture']
        mlp_model = ArmorNet(
            input_size=arch['input_size'],
            hidden_sizes=tuple(arch['hidden_sizes']),
            dropout=arch['dropout'],
        )
        mlp_model.load_state_dict(torch.load(metadata['model_path']))
        mlp_model.eval()
    except FileNotFoundError as e:
        print(f"Error: required artifact not found ({e}). Please run train.py first.")
        sys.exit(1)

    return metadata, scaler_X, scaler_y, mlp_model, rf_model


def load_held_out_test_set(metadata: dict) -> pd.DataFrame:
    """Loads ONLY the held-out test split saved by train.py."""
    test_split_path = metadata['test_split_path']
    try:
        raw = pd.read_csv(test_split_path)
    except FileNotFoundError:
        print(f"Error: '{test_split_path}' not found. Please run train.py first "
              f"(it saves the held-out test split to this file).")
        sys.exit(1)

    # Re-clean defensively (idempotent if train.py already cleaned it) so this
    # script never silently assumes the CSV on disk is well-formed.
    df = clean_dataset(raw, verbose=False)
    return df


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
    }


def predict_mlp(model, scaler_X, scaler_y, X_raw: np.ndarray) -> np.ndarray:
    X_scaled = scaler_X.transform(X_raw)
    with torch.no_grad():
        y_pred_scaled = model(torch.FloatTensor(X_scaled)).numpy()
    return scaler_y.inverse_transform(y_pred_scaled).flatten()


def predict_rf(model, X_raw: np.ndarray) -> np.ndarray:
    return model.predict(X_raw)


def print_metrics_report(mlp_metrics: dict, rf_metrics: dict, recommended_model: str, n_rows: int, n_materials: int):
    print("--- MODEL ACCURACY REPORT (held-out test set) ---")
    print(f"Test set: {n_rows} rows / {n_materials} materials never seen during training or validation\n")

    print(f"{'Model':<16}{'MAE (m/s)':>12}{'RMSE (m/s)':>13}{'R2':>10}")
    print(f"{'ArmorNet (MLP)':<16}{mlp_metrics['mae']:>12.2f}{mlp_metrics['rmse']:>13.2f}{mlp_metrics['r2']:>10.4f}")
    print(f"{'Random Forest':<16}{rf_metrics['mae']:>12.2f}{rf_metrics['rmse']:>13.2f}{rf_metrics['r2']:>10.4f}")

    print(f"\ntrain.py's cross-validation recommended: {recommended_model.upper()}")
    test_winner = 'mlp' if mlp_metrics['r2'] >= rf_metrics['r2'] else 'random_forest'
    print(f"On this held-out test set, the better performer is: {test_winner.upper()}")
    if test_winner != recommended_model:
        print("(Note: this differs from the CV recommendation -- with a dataset this size, "
              "a single test split can reasonably disagree with a 5-fold CV average. "
              "Trust the CV result more; treat this test set as one more, smaller data point.)")


def plot_evaluation(comparison_df: pd.DataFrame, out_path: str = 'test_evaluation.png'):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ax, model_name, pred_col, color in [
        (axes[0, 0], 'ArmorNet (MLP)', 'predicted_mlp', 'tab:blue'),
        (axes[0, 1], 'Random Forest', 'predicted_rf', 'tab:orange'),
    ]:
        actual = comparison_df['actual_v50']
        pred = comparison_df[pred_col]
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.scatter(actual, pred, alpha=0.6, color=color, edgecolor='k', linewidth=0.3)
        ax.plot(lims, lims, 'k--', linewidth=1, label='Perfect prediction')
        ax.set_xlabel('Actual V50 (m/s)')
        ax.set_ylabel('Predicted V50 (m/s)')
        ax.set_title(f'{model_name}: Predicted vs. Actual')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    for ax, model_name, resid_col, color in [
        (axes[1, 0], 'ArmorNet (MLP)', 'residual_mlp', 'tab:blue'),
        (axes[1, 1], 'Random Forest', 'residual_rf', 'tab:orange'),
    ]:
        ax.hist(comparison_df[resid_col], bins=20, color=color, edgecolor='k', alpha=0.75)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Residual: Actual - Predicted (m/s)')
        ax.set_ylabel('Count')
        ax.set_title(f'{model_name}: Residual Distribution')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved evaluation plots to '{out_path}'")


def main():
    metadata, scaler_X, scaler_y, mlp_model, rf_model = load_test_artifacts()
    test_df = load_held_out_test_set(metadata)

    X_test = test_df[FEATURES].values.astype(float)
    y_test = test_df[[TARGET]].values.astype(float)

    y_pred_mlp = predict_mlp(mlp_model, scaler_X, scaler_y, X_test)
    y_pred_rf = predict_rf(rf_model, X_test)

    mlp_metrics = compute_metrics(y_test, y_pred_mlp)
    rf_metrics = compute_metrics(y_test, y_pred_rf)

    n_materials = test_df[GROUP_COL].nunique() if GROUP_COL in test_df.columns else float('nan')
    print_metrics_report(
        mlp_metrics, rf_metrics,
        recommended_model=metadata.get('recommended_model', 'mlp'),
        n_rows=len(test_df), n_materials=n_materials,
    )

    # Per-row comparison table
    comparison_df = pd.DataFrame({
        'Material': test_df[GROUP_COL] if GROUP_COL in test_df.columns else np.nan,
        'actual_v50': y_test.flatten(),
        'predicted_mlp': y_pred_mlp,
        'residual_mlp': y_test.flatten() - y_pred_mlp,
        'predicted_rf': y_pred_rf,
        'residual_rf': y_test.flatten() - y_pred_rf,
    })
    comparison_df.to_csv('test_predictions.csv', index=False)
    print("\nSaved per-row predictions to 'test_predictions.csv'")

    print("\n--- SAMPLE PREDICTIONS (first 10 test rows) ---")
    print(comparison_df.head(10).to_string(index=False))

    plot_evaluation(comparison_df)


if __name__ == '__main__':
    main()