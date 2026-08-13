"""
main.py

Scenario-based dual-layer armor optimization.

Given an explicit threat scenario (a projectile + impact angle), this
script:
  1. Loads the trained model (MLP or Random Forest, per metadata.json's
     CV recommendation, or an explicit override below).
  2. For every material in Dataset.csv, sweeps a range of thicknesses
     spanning that material's OWN historically observed thickness range
     (thickness is a design variable an armor designer chooses -- it is
     NOT averaged into one fixed number, unlike the original version of
     this script).
  3. Predicts V50 for every (material, thickness) candidate against the
     specified scenario.
  4. Searches every valid strike-face + backing material pair (the harder
     material becomes the strike face) under an areal weight budget, and
     reports the ones with the highest combined V50.

Modeling assumption (please note if writing this up in a report): the two
layers are combined via V_combined = sqrt(v1^2 + v2^2), an energy-based
approximation commonly used for independently-acting sequential layers.
It ignores interaction effects between layers (bonding, delamination,
shock transfer) and should be treated as an estimate, not a guarantee.

Requires artifacts produced by train.py: armor_model.pth, scaler_X.pkl,
scaler_y.pkl, rf_baseline.pkl, metadata.json.
"""

import itertools
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_utils import (
    FEATURES,
    GROUP_COL,
    MATERIAL_FEATURES,
    PROJECTILE_FEATURES,
    SCENARIO_FEATURES,
    feature_ranges,
    load_clean_dataset,
    load_metadata,
    material_property_table,
)
from model import ArmorNet

# ======================================================================
# USER CONFIGURATION -- edit these to explore different design scenarios
# ======================================================================

# The threat the armor is being designed against. Values are in the same
# units as the corresponding columns in Dataset.csv. The defaults below
# match a common projectile in the dataset (CAL30APM2, normal incidence).
SCENARIO = {
    'calibre': 6.2484,
    'proj_mass': 5.3,
    'proj_density': 7.85,
    'proj_hardness': 570,
    'angle': 0,
}

WEIGHT_LIMIT_KG_M2 = 200      # maximum total areal weight for a dual-layer system
N_THICKNESS_STEPS = 6          # thickness samples per material, within its observed range
TOP_N_RESULTS = 10             # how many top combinations to print/save prominently
MODEL_CHOICE = 'auto'          # 'mlp', 'random_forest', or 'auto' (use train.py's CV recommendation)


# ======================================================================
# Artifact loading
# ======================================================================

def load_artifacts(metadata_path: str = 'metadata.json') -> dict:
    """Loads metadata, both scalers, the trained MLP, and the RF baseline."""
    try:
        metadata = load_metadata(metadata_path)
    except FileNotFoundError:
        print(f"Error: '{metadata_path}' not found. Please run train.py first.")
        sys.exit(1)

    if metadata['features'] != FEATURES:
        raise ValueError(
            "Feature list in metadata.json does not match data_utils.FEATURES. "
            "The model was trained with a different feature set -- re-run train.py."
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

    return {
        'metadata': metadata,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'mlp_model': mlp_model,
        'rf_model': rf_model,
    }


def select_model(metadata: dict, choice: str) -> str:
    """Resolves MODEL_CHOICE ('auto' or an explicit name) to a concrete model name."""
    resolved = metadata.get('recommended_model', 'mlp') if choice == 'auto' else choice
    if resolved not in ('mlp', 'random_forest'):
        raise ValueError(f"Unknown model choice '{resolved}'. Must be 'mlp' or 'random_forest'.")
    return resolved


# ======================================================================
# Scenario validation
# ======================================================================

def check_scenario_extrapolation(scenario: dict, ranges: dict) -> None:
    """Warns if the requested scenario falls outside the feature ranges the
    model was actually trained on -- predictions there are extrapolations
    and should be treated with reduced confidence."""
    issues = []
    for feat in PROJECTILE_FEATURES + SCENARIO_FEATURES:
        lo, hi = ranges[feat]
        val = scenario[feat]
        if val < lo or val > hi:
            issues.append(f"  - {feat}: scenario value {val} is outside the observed training range [{lo:.2f}, {hi:.2f}]")

    if issues:
        print("\nWARNING: this scenario extrapolates beyond the training data for:")
        for issue in issues:
            print(issue)
        print("Predictions below should be treated with reduced confidence.\n")
    else:
        print("Scenario is within the training data's observed range for all projectile/angle features.\n")


# ======================================================================
# Candidate generation (material x thickness sweep, for this scenario)
# ======================================================================

def build_candidate_matrix(material_table: pd.DataFrame, scenario: dict, n_steps: int):
    """
    For every material, sweeps `n_steps` thickness values spanning that
    material's own observed [thickness_min, thickness_max] range, and
    builds the corresponding model input row (material properties +
    scenario projectile/angle features) for each.

    Returns (meta_df, X):
      meta_df - one row per candidate with Material, thickness, density,
                hardness (needed later for weight calc and strike/backing
                ordering)
      X       - the matching feature matrix, columns in FEATURES order,
                ready to feed to the model
    """
    missing = [k for k in PROJECTILE_FEATURES + SCENARIO_FEATURES if k not in scenario]
    if missing:
        raise ValueError(f"SCENARIO is missing required key(s): {missing}")

    meta_rows = []
    feature_rows = []

    for _, mat in material_table.iterrows():
        t_min, t_max = mat['thickness_min'], mat['thickness_max']
        thicknesses = [t_min] if t_min == t_max else np.linspace(t_min, t_max, n_steps)

        for t in thicknesses:
            feature_row = []
            for feat in FEATURES:
                if feat == 'thickness':
                    feature_row.append(t)
                elif feat in MATERIAL_FEATURES:
                    feature_row.append(mat[feat])
                else:  # projectile / scenario feature
                    feature_row.append(scenario[feat])
            feature_rows.append(feature_row)

            meta_rows.append({
                'Material': mat['Material'],
                'thickness': t,
                'density': mat['density'],
                'hardness': mat['hardness'],
            })

    meta_df = pd.DataFrame(meta_rows)
    X = np.array(feature_rows, dtype=float)
    return meta_df, X


def predict_v50(model_choice: str, artifacts: dict, X: np.ndarray) -> np.ndarray:
    """Predicts V50 for a feature matrix X using the chosen model."""
    if model_choice == 'mlp':
        X_scaled = artifacts['scaler_X'].transform(X)
        with torch.no_grad():
            y_scaled = artifacts['mlp_model'](torch.FloatTensor(X_scaled)).numpy()
        return artifacts['scaler_y'].inverse_transform(y_scaled).flatten()
    else:  # random_forest -- trees need no feature scaling
        return artifacts['rf_model'].predict(X)


# ======================================================================
# Dual-layer combinatorial search
# ======================================================================

def search_dual_layer(single_layer_df: pd.DataFrame, weight_limit: float) -> pd.DataFrame:
    """
    Searches every pairwise combination of (material, thickness) candidates
    from DIFFERENT materials. The harder material of each pair becomes the
    strike face; the softer becomes the backing. Combines V50 via
    sqrt(v1^2 + v2^2) and computes total areal weight (kg/m^2), keeping
    only combinations within `weight_limit`.
    """
    records = single_layer_df.to_dict('records')
    results = []

    for a, b in itertools.combinations(records, 2):
        if a['Material'] == b['Material']:
            continue

        strike, backing = (a, b) if a['hardness'] >= b['hardness'] else (b, a)

        weight = strike['thickness'] * strike['density'] + backing['thickness'] * backing['density']
        if weight > weight_limit:
            continue

        v_combined = np.sqrt(strike['predicted_v50'] ** 2 + backing['predicted_v50'] ** 2)

        results.append({
            'Strike Face': strike['Material'],
            'Strike Thickness (mm)': round(strike['thickness'], 2),
            'Backing': backing['Material'],
            'Backing Thickness (mm)': round(backing['thickness'], 2),
            'Combined_V50': round(v_combined, 2),
            'Total_Weight': round(weight, 2),
            'Efficiency': round(v_combined / weight, 3),
        })

    return pd.DataFrame(results)


# ======================================================================
# Plotting
# ======================================================================

def plot_results(all_results: pd.DataFrame, top_results: pd.DataFrame, out_path: str = 'ballistic_optimization.png'):
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(
        all_results['Total_Weight'], all_results['Combined_V50'],
        alpha=0.5, c=all_results['Efficiency'], cmap='viridis', s=15,
    )
    plt.colorbar(scatter, label='Efficiency (V50 / Weight)')

    for _, row in top_results.head(3).iterrows():
        label = (f"{row['Strike Face']}({row['Strike Thickness (mm)']}mm)+"
                  f"{row['Backing']}({row['Backing Thickness (mm)']}mm)")
        plt.annotate(label, (row['Total_Weight'], row['Combined_V50']),
                     xytext=(5, 5), textcoords='offset points', fontsize=7)

    plt.title('Dual-Layer Armor Optimization: Ballistic Limit vs. Weight')
    plt.xlabel('Total Areal Weight (kg/m²)')
    plt.ylabel('Combined V50 (m/s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Graph saved as '{out_path}'")


# ======================================================================
# Main
# ======================================================================

def main():
    print("Loading trained model artifacts...")
    artifacts = load_artifacts()
    model_choice = select_model(artifacts['metadata'], MODEL_CHOICE)
    print(f"Using model: {model_choice.upper()} "
          f"(train.py's CV recommendation was: {artifacts['metadata'].get('recommended_model', 'n/a').upper()})")

    print("\nLoading and cleaning dataset...")
    try:
        df = load_clean_dataset('Dataset.csv')
    except FileNotFoundError:
        print("Error: 'Dataset.csv' not found in the current directory.")
        sys.exit(1)

    print(f"\nScenario: {SCENARIO}")
    ranges = feature_ranges(df)
    check_scenario_extrapolation(SCENARIO, ranges)

    mat_table = material_property_table(df)
    print(f"Building thickness sweep ({N_THICKNESS_STEPS} steps per material) "
          f"for {len(mat_table)} materials...")

    candidates_meta, X = build_candidate_matrix(mat_table, SCENARIO, N_THICKNESS_STEPS)
    candidates_meta['predicted_v50'] = predict_v50(model_choice, artifacts, X)
    print(f"Generated {len(candidates_meta)} single-layer candidates.")

    print(f"Searching dual-layer combinations under {WEIGHT_LIMIT_KG_M2} kg/m2...")
    results_df = search_dual_layer(candidates_meta, WEIGHT_LIMIT_KG_M2)

    if results_df.empty:
        print("\nNo combinations found within the weight limit. "
              "Try increasing WEIGHT_LIMIT_KG_M2.")
        return

    top_results = results_df.sort_values(by='Combined_V50', ascending=False)
    top_results.to_csv('optimized_armor_combinations.csv', index=False)

    print(f"\n--- TOP {TOP_N_RESULTS} COMBINATIONS ---")
    print(top_results.head(TOP_N_RESULTS).to_string(index=False))
    print(f"\nAll {len(results_df)} valid combinations saved to 'optimized_armor_combinations.csv'")

    plot_results(results_df, top_results)


if __name__ == '__main__':
    main()