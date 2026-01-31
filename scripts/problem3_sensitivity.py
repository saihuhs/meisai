# -*- coding: utf-8 -*-
"""
Problem C - Question 3 Sensitivity Analysis
Assess the robustness of the "Pro Dancer Effects" and "Celebrity Characteristics"
model by varying model specifications and data subsets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Configuration
# ============================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR = ROOT_DIR / "figures"
DATA_FILE = OUTPUT_DIR / "problem3_model_dataset.csv"

# Outputs
OUTPUT_SENS_SPECS = OUTPUT_DIR / "eval_q3_sensitivity_specs.csv"
OUTPUT_SENS_LOSO = OUTPUT_DIR / "eval_q3_loso.csv"
PLOT_LOSO = FIG_DIR / "eval_q3_loso_coef_stability.png"

# ============================
# Helper Functions
# ============================

def build_design_matrix(df: pd.DataFrame, drop_week_fe=False):
    """
    Build design matrix consistent with main Q3 script.
    """
    df = df.copy()
    
    # Ensure numeric
    if "celebrity_age_during_season" in df.columns:
        df["celebrity_age_during_season"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce").fillna(0)

    # Cat cols
    cat_cols = [
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "season",
        "ballroom_partner",
    ]
    if not drop_week_fe:
        cat_cols.append("week")

    # One-hot encoding
    # Align columns carefully if doing prediction, but here for OLS inference on fixed dataset
    # we just need consistency within the function call.
    X = pd.get_dummies(df[cat_cols], drop_first=True)
    X["age"] = df["celebrity_age_during_season"]
    X.insert(0, "intercept", 1.0)
    
    return X.fillna(0).astype(float)

def ols_fit(X: np.ndarray, y: np.ndarray):
    """Quick OLS"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def logit_transform(series: pd.Series, eps: float = 1e-6):
    s = series.astype(float).clip(eps, 1 - eps)
    return np.log(s / (1 - s))

# ============================
# Main Analysis
# ============================

def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} records.")

    # ----------------------------------------------------
    # 1. Model Specification Sensitivity
    # ----------------------------------------------------
    specs = []
    
    # Spec A: Current (Logit + SeasonFE + WeekFE)
    X_A = build_design_matrix(df, drop_week_fe=False)
    y_A_judge = logit_transform(df["judge_share"])
    y_A_fan = logit_transform(df["fan_vote_share"])
    
    beta_A_j = ols_fit(X_A.values, y_A_judge)
    beta_A_f = ols_fit(X_A.values, y_A_fan)
    
    # Extract Pro Dancer Coefs (columns starting with ballroom_partner_)
    pro_cols_A = [c for c in X_A.columns if "ballroom_partner" in c]
    
    # Store average pro effect magnitude
    specs.append({
        "model": "Logit_SeasonWeekFE",
        "outcome": "judge",
        "avg_abs_pro_coef": np.mean(np.abs(beta_A_j[[X_A.columns.get_loc(c) for c in pro_cols_A]]))
    })
    specs.append({
        "model": "Logit_SeasonWeekFE",
        "outcome": "fan",
        "avg_abs_pro_coef": np.mean(np.abs(beta_A_f[[X_A.columns.get_loc(c) for c in pro_cols_A]]))
    })

    # Spec B: Logit + SeasonFE (No Week)
    X_B = build_design_matrix(df, drop_week_fe=True) # Week removed
    beta_B_j = ols_fit(X_B.values, y_A_judge) # Y is same
    beta_B_f = ols_fit(X_B.values, y_A_fan)
    
    pro_cols_B = [c for c in X_B.columns if "ballroom_partner" in c]
    specs.append({
        "model": "Logit_SeasonOnlyFE",
        "outcome": "judge",
        "avg_abs_pro_coef": np.mean(np.abs(beta_B_j[[X_B.columns.get_loc(c) for c in pro_cols_B]]))
    })
    specs.append({
        "model": "Logit_SeasonOnlyFE",
        "outcome": "fan",
        "avg_abs_pro_coef": np.mean(np.abs(beta_B_f[[X_B.columns.get_loc(c) for c in pro_cols_B]]))
    })
    
    # Spec C: Linear Probability (Raw Share) + SeasonWeekFE
    # Y changes to raw share
    y_C_judge = df["judge_share"]
    y_C_fan = df["fan_vote_share"]
    
    beta_C_j = ols_fit(X_A.values, y_C_judge)
    beta_C_f = ols_fit(X_A.values, y_C_fan)
    
    # Scale coefficients for comparison? Linear coefs are small. 
    # Just record raw for now, or maybe standardized? 
    # Let's record raw but note they are different scale.
    specs.append({
        "model": "Linear_SeasonWeekFE",
        "outcome": "judge",
        "avg_abs_pro_coef": np.mean(np.abs(beta_C_j[[X_A.columns.get_loc(c) for c in pro_cols_A]]))
    })
    specs.append({
        "model": "Linear_SeasonWeekFE",
        "outcome": "fan",
        "avg_abs_pro_coef": np.mean(np.abs(beta_C_f[[X_A.columns.get_loc(c) for c in pro_cols_A]]))
    })

    pd.DataFrame(specs).to_csv(OUTPUT_SENS_SPECS, index=False)

    # ----------------------------------------------------
    # 2. Leave-One-Season-Out (LOSO) Stability
    # ----------------------------------------------------
    seasons = df["season"].unique()
    loso_results = []
    
    # Target: Top 5 Pros (by frequency or just track all and filter later)
    # Let's track top pros found in the full model first
    top_pro_idx = np.argsort(np.abs(beta_A_f[[X_A.columns.get_loc(c) for c in pro_cols_A]]))[-5:]
    top_pros = [pro_cols_A[i] for i in top_pro_idx]
    
    print(f"Tracking stability for: {top_pros}")

    for s in seasons:
        # Train mask: season != s
        train_mask = df["season"] != s
        if train_mask.sum() < 100: continue
        
        df_train = df[train_mask].copy()
        
        # We need to rebuild design matrix to handle dropped levels correctly?
        # Actually, simpler to just use full matrix subset, but some cols might be all zero.
        # It's safer to re-build matrix on subset or handle singular matrix.
        # But using full matrix subset with OLS is fine if we ignore zero-variance cols or use pinv.
        # OLS fit uses lstsq which handles rank deficiency.
        
        X_sub = X_A[train_mask]
        y_j_sub = y_A_judge[train_mask]
        y_f_sub = y_A_fan[train_mask]
        
        b_j = ols_fit(X_sub.values, y_j_sub)
        b_f = ols_fit(X_sub.values, y_f_sub)
        
        # Record coeffs for the tracked pros
        for pro in top_pros:
            # Column index in X_A
            col_idx = X_A.columns.get_loc(pro)
            
            loso_results.append({
                "dropped_season": s,
                "pro_dancer": pro,
                "coef_judge": b_j[col_idx],
                "coef_fan": b_f[col_idx]
            })

    loso_df = pd.DataFrame(loso_results)
    loso_df.to_csv(OUTPUT_SENS_LOSO, index=False)

    # ----------------------------------------------------
    # 3. Plotting LOSO Stability
    # ----------------------------------------------------
    if not loso_df.empty:
        plt.figure(figsize=(10, 6))
        
        # Group by pro and plot distribution of coefficients
        # Boxplot
        pros = loso_df["pro_dancer"].unique()
        data_to_plot = []
        labels = []
        for p in pros:
            vals = loso_df[loso_df["pro_dancer"] == p]["coef_fan"].values
            data_to_plot.append(vals)
            labels.append(p.replace("ballroom_partner_", ""))
        
        plt.boxplot(data_to_plot, labels=labels, vert=False)
        plt.axvline(0, color="r", linestyle="--", alpha=0.5)
        plt.title("Q3 Sensitivity: Pro Dancer Effect Stability (Fan Vote)\n(Leave-One-Season-Out CV)")
        plt.xlabel("Coefficient Value (Log Odds)")
        plt.tight_layout()
        plt.savefig(PLOT_LOSO, dpi=300)
        print(f"Generated LOSO plot: {PLOT_LOSO}")

if __name__ == "__main__":
    main()
