# -*- coding: utf-8 -*-
"""
Problem C - Question 3
Analyze effects of pro dancers and celebrity characteristics on
(1) judge scores and (2) estimated fan vote shares.

This script follows the model structure in the paper:
- Build outcome mappings J_{i,s,t} = g(x_i, p(i), s, t)
  and F_{i,s,t} = h(x_i, p(i), s, t)
- Use an interpretable linear framework (OLS) with fixed effects
  for pro dancers and categorical celebrity attributes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Global plotting configuration
# ============================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ============================
# File paths
# ============================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "2026_MCM_Problem_C_Data.csv"
FAN_VOTE_FILE = BASE_DIR / "fan_vote_estimates_q1a.csv"

OUTPUT_DATASET = BASE_DIR / "problem3_model_dataset.csv"
OUTPUT_COEF_JUDGE = BASE_DIR / "problem3_ols_coeff_judge.csv"
OUTPUT_COEF_FAN = BASE_DIR / "problem3_ols_coeff_fan.csv"
OUTPUT_FIT = BASE_DIR / "problem3_model_fit.csv"
OUTPUT_PLOT = BASE_DIR / "problem3_top_pro_dancer_effects.png"

# ============================
# Helper functions
# ============================

def parse_week_number(col_name: str) -> int:
    """Extract week number from a column name like 'week7_judge2_score'."""
    return int(col_name.split("_")[0].replace("week", ""))


def identify_week_columns(columns):
    """Return a sorted list of week numbers available in the data."""
    week_cols = [c for c in columns if c.startswith("week") and c.endswith("_score")]
    weeks = sorted({parse_week_number(c) for c in week_cols})
    return weeks


def compute_week_scores(df: pd.DataFrame, weeks):
    """Compute weekly judge average scores using missing indicators."""
    df = df.copy()
    for w in weeks:
        judge_cols = [c for c in df.columns if c.startswith(f"week{w}_") and c.endswith("_score")]
        scores = df[judge_cols].replace("N/A", np.nan).astype(float)
        m = scores.notna().sum(axis=1)
        s = scores.sum(axis=1, skipna=True)
        df[f"week{w}_count"] = m
        df[f"week{w}_sum"] = s
        df[f"week{w}_avg"] = (s / m).where((m > 0) & (s > 0), np.nan)
    return df


def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Ordinary Least Squares solution:
    beta = argmin ||y - X beta||^2
    Returns coefficients and R^2.
    """
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def build_design_matrix(df: pd.DataFrame):
    """
    Build design matrix for g(·) and h(·):
    - Numerical: age
    - Categorical: industry, homestate, homecountry/region, season, pro dancer
    Use one-hot encoding with drop-first to avoid perfect collinearity.
    """
    df = df.copy()

    # Numerical feature
    df["celebrity_age_during_season"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")

    # Categorical features
    cat_cols = [
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "season",
        "ballroom_partner",
    ]

    X = pd.get_dummies(df[cat_cols], drop_first=True)
    X["age"] = df["celebrity_age_during_season"]

    # Add intercept
    X.insert(0, "intercept", 1.0)

    # Ensure numeric matrix (replace missing with 0 and cast to float)
    X = X.fillna(0).astype(float)

    return X


# ============================
# Main process
# ============================

def main():
    # Load data
    raw = pd.read_csv(DATA_FILE)
    fan = pd.read_csv(FAN_VOTE_FILE)

    # Compute weekly judge totals
    weeks = identify_week_columns(raw.columns)
    raw = compute_week_scores(raw, weeks)

    # Build contestant-week panel for weeks with fan vote estimates
    panel_rows = []
    for (season, week), df_week in fan.groupby(["season", "week"]):
        # Extract judge totals for the same contestants in that week
        for _, row in df_week.iterrows():
            # Match by season + celebrity_name
            match = raw[(raw["season"] == season) & (raw["celebrity_name"] == row["celebrity_name"])]
            if match.empty:
                continue
            base = match.iloc[0].to_dict()
            panel_rows.append({
                "season": season,
                "week": week,
                "celebrity_name": row["celebrity_name"],
                "ballroom_partner": base["ballroom_partner"],
                "celebrity_industry": base["celebrity_industry"],
                "celebrity_homestate": base["celebrity_homestate"],
                "celebrity_homecountry/region": base["celebrity_homecountry/region"],
                "celebrity_age_during_season": base["celebrity_age_during_season"],
                "judge_total": base.get(f"week{int(week)}_avg", np.nan),
                "fan_vote_share": row["fan_vote_share"],
            })

    panel = pd.DataFrame(panel_rows).dropna(subset=["judge_total", "fan_vote_share"])

    # Save intermediate dataset
    panel.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8-sig")

    print("=== 建模数据集（前20行） ===")
    print(panel.head(20))

    # Build design matrix
    X = build_design_matrix(panel)

    # Outcome 1: Judge total score
    y_judge = panel["judge_total"].to_numpy()
    beta_judge, r2_judge = ols_fit(X.to_numpy(), y_judge)

    coef_judge = pd.DataFrame({
        "feature": X.columns,
        "coef": beta_judge
    }).sort_values("coef", ascending=False)
    coef_judge.to_csv(OUTPUT_COEF_JUDGE, index=False, encoding="utf-8-sig")

    # Outcome 2: Fan vote share
    y_fan = panel["fan_vote_share"].to_numpy()
    beta_fan, r2_fan = ols_fit(X.to_numpy(), y_fan)

    coef_fan = pd.DataFrame({
        "feature": X.columns,
        "coef": beta_fan
    }).sort_values("coef", ascending=False)
    coef_fan.to_csv(OUTPUT_COEF_FAN, index=False, encoding="utf-8-sig")

    # Fit summary
    fit_df = pd.DataFrame([
        {"model": "judge_total", "r2": r2_judge},
        {"model": "fan_vote_share", "r2": r2_fan},
    ])
    fit_df.to_csv(OUTPUT_FIT, index=False, encoding="utf-8-sig")

    print("\n=== OLS 拟合优度 ===")
    print(fit_df)

    # Visualization: top pro dancer effects for judge vs fan
    dancer_prefix = "ballroom_partner_"
    dancer_coef_j = coef_judge[coef_judge["feature"].str.startswith(dancer_prefix)].head(10)
    dancer_coef_f = coef_fan[coef_fan["feature"].str.startswith(dancer_prefix)].head(10)

    plt.figure(figsize=(10, 6))
    x = np.arange(10)
    plt.bar(x - 0.2, dancer_coef_j["coef"].values, width=0.4, label="裁判得分效应")
    plt.bar(x + 0.2, dancer_coef_f["coef"].values, width=0.4, label="观众投票效应")
    plt.xticks(x, [s.replace(dancer_prefix, "") for s in dancer_coef_j["feature"].values], rotation=45, ha="right")
    plt.title("职业舞者效应（前10）对裁判与观众的差异")
    plt.ylabel("回归系数（相对基准）")
    plt.xlabel("职业舞者")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.show()

    print(f"\n建模数据已保存：{OUTPUT_DATASET}")
    print(f"裁判得分系数已保存：{OUTPUT_COEF_JUDGE}")
    print(f"观众投票系数已保存：{OUTPUT_COEF_FAN}")
    print(f"拟合优度已保存：{OUTPUT_FIT}")
    print(f"图像已保存：{OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
