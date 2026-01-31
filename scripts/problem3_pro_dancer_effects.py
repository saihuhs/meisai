# -*- coding: utf-8 -*-
"""
Problem C - Question 3
Analyze effects of pro dancers and celebrity characteristics on
(1) judge scores and (2) estimated fan vote shares.

This script follows the model structure in the paper:
- Build outcome mappings J_{i,s,t} = g(x_i, p(i), s, t)
    and F_{i,s,t} = h(x_i, p(i), s, t)
- Use logit-transformed share outcomes and fixed effects
    (season/week dummies) with interpretable linear OLS.
"""

from pathlib import Path
import math
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
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR = ROOT_DIR / "figures"

DATA_FILE = DATA_DIR / "2026_MCM_Problem_C_Data.csv"
FAN_VOTE_FILE = OUTPUT_DIR / "fan_vote_estimates_q1a.csv"

OUTPUT_DATASET = OUTPUT_DIR / "problem3_model_dataset.csv"
OUTPUT_COEF_JUDGE = OUTPUT_DIR / "problem3_ols_coeff_judge.csv"
OUTPUT_COEF_FAN = OUTPUT_DIR / "problem3_ols_coeff_fan.csv"
OUTPUT_FIT = OUTPUT_DIR / "problem3_model_fit.csv"
OUTPUT_PLOT = FIG_DIR / "problem3_top_pro_dancer_effects.png"
OUTPUT_COEF_COMPARE = OUTPUT_DIR / "problem3_coef_comparison.csv"
OUTPUT_STANDARDIZED = OUTPUT_DIR / "problem3_standardized_coeff.csv"
OUTPUT_MARGINAL = OUTPUT_DIR / "problem3_marginal_effects.csv"
OUTPUT_VIF = OUTPUT_DIR / "problem3_vif.csv"
OUTPUT_PLACEMENT = OUTPUT_DIR / "problem3_placement_analysis.csv"
OUTPUT_CONSISTENCY = OUTPUT_DIR / "problem3_consistency_metrics.csv"
OUTPUT_FIG_INDUSTRY_SCATTER = FIG_DIR / "problem3_industry_scatter.png"
OUTPUT_FIG_TOP_JUDGE = FIG_DIR / "problem3_top_features_judge.png"
OUTPUT_FIG_TOP_FAN = FIG_DIR / "problem3_top_features_fan.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

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
    return beta, r2, y_hat


def ols_robust_se(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray):
    """HC1 robust standard errors for OLS coefficients."""
    n, k = X.shape
    resid = y - y_hat
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = X.T @ (np.diag(resid ** 2)) @ X
    cov = xtx_inv @ meat @ xtx_inv
    if n > k:
        cov = cov * (n / (n - k))
    diag = np.diag(cov)
    diag = np.where(diag < 0, np.nan, diag)
    se = np.sqrt(diag)
    return se


def normal_pvalue(z: float) -> float:
    """Two-sided p-value using normal approximation."""
    if np.isnan(z):
        return np.nan
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def compute_vif(X_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF for each feature (excluding intercept)."""
    cols = [c for c in X_df.columns if c != "intercept"]
    X = X_df.values
    vif_rows = []
    for col in cols:
        y = X_df[col].values
        X_others = X_df[[c for c in cols if c != col]].values
        X_others = np.column_stack([np.ones(X_others.shape[0]), X_others])
        beta, *_ = np.linalg.lstsq(X_others, y, rcond=None)
        y_hat = X_others @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        vif = 1 / (1 - r2) if r2 is not None and r2 < 1 else np.inf
        vif_rows.append({"feature": col, "vif": vif})
    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False)


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
        "week",
        "ballroom_partner",
    ]

    X = pd.get_dummies(df[cat_cols], drop_first=True)
    X["age"] = df["celebrity_age_during_season"]

    # Add intercept
    X.insert(0, "intercept", 1.0)

    # Ensure numeric matrix (replace missing with 0 and cast to float)
    X = X.fillna(0).astype(float)

    return X


def logit_transform(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Logit transform for proportions in (0, 1)."""
    s = series.astype(float).clip(eps, 1 - eps)
    return np.log(s / (1 - s))


def standardize_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardize columns except intercept."""
    X_std = X.copy()
    for col in X_std.columns:
        if col == "intercept":
            continue
        s = X_std[col].std()
        if s == 0 or np.isnan(s):
            X_std[col] = 0.0
        else:
            X_std[col] = (X_std[col] - X_std[col].mean()) / s
    return X_std


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

    # Compute judge share within week (proportion outcome)
    panel["judge_share"] = panel.groupby(["season", "week"])['judge_total'].transform(lambda x: x / x.sum())

    # Save intermediate dataset
    panel.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8-sig")

    print("=== 建模数据集（前20行） ===")
    print(panel.head(20))

    # Build design matrix
    X = build_design_matrix(panel)

    # Outcome 1: Judge share (logit)
    y_judge = logit_transform(panel["judge_share"]).to_numpy()
    beta_judge, r2_judge, yhat_judge = ols_fit(X.to_numpy(), y_judge)
    se_judge = ols_robust_se(X.to_numpy(), y_judge, yhat_judge)

    t_judge = np.divide(beta_judge, se_judge, out=np.full_like(beta_judge, np.nan), where=se_judge != 0)
    coef_judge = pd.DataFrame({
        "feature": X.columns,
        "coef": beta_judge,
        "se_robust": se_judge,
        "t_robust": t_judge,
    }).sort_values("coef", ascending=False)
    coef_judge.to_csv(OUTPUT_COEF_JUDGE, index=False, encoding="utf-8-sig")

    # Outcome 2: Fan vote share (logit)
    y_fan = logit_transform(panel["fan_vote_share"]).to_numpy()
    beta_fan, r2_fan, yhat_fan = ols_fit(X.to_numpy(), y_fan)
    se_fan = ols_robust_se(X.to_numpy(), y_fan, yhat_fan)

    t_fan = np.divide(beta_fan, se_fan, out=np.full_like(beta_fan, np.nan), where=se_fan != 0)
    coef_fan = pd.DataFrame({
        "feature": X.columns,
        "coef": beta_fan,
        "se_robust": se_fan,
        "t_robust": t_fan,
    }).sort_values("coef", ascending=False)
    coef_fan.to_csv(OUTPUT_COEF_FAN, index=False, encoding="utf-8-sig")

    # Fit summary
    fit_df = pd.DataFrame([
        {"model": "judge_share_logit", "r2": r2_judge},
        {"model": "fan_vote_share_logit", "r2": r2_fan},
    ])
    fit_df.to_csv(OUTPUT_FIT, index=False, encoding="utf-8-sig")
    # Standardized coefficients
    X_std = standardize_matrix(X)
    beta_j_std, r2_j_std, yhat_j_std = ols_fit(X_std.to_numpy(), y_judge)
    beta_f_std, r2_f_std, yhat_f_std = ols_fit(X_std.to_numpy(), y_fan)
    std_df = pd.DataFrame({
        "feature": X.columns,
        "coef_std_judge": beta_j_std,
        "coef_std_fan": beta_f_std,
    }).sort_values("coef_std_judge", ascending=False)
    std_df.to_csv(OUTPUT_STANDARDIZED, index=False, encoding="utf-8-sig")

    # Coefficient comparison: direction consistency + difference test
    coef_merge = coef_judge.merge(
        coef_fan, on="feature", suffixes=("_judge", "_fan"), how="inner"
    )
    coef_merge["direction_same"] = np.sign(coef_merge["coef_judge"]) == np.sign(coef_merge["coef_fan"])
    coef_merge["diff"] = coef_merge["coef_judge"] - coef_merge["coef_fan"]
    coef_merge["diff_se"] = np.sqrt(coef_merge["se_robust_judge"] ** 2 + coef_merge["se_robust_fan"] ** 2)
    coef_merge["diff_t"] = coef_merge["diff"] / coef_merge["diff_se"]
    coef_merge["diff_p"] = coef_merge["diff_t"].apply(normal_pvalue)
    coef_merge.to_csv(OUTPUT_COEF_COMPARE, index=False, encoding="utf-8-sig")

    # Consistency metrics: sign agreement & coefficient correlation
    non_intercept = coef_merge[coef_merge["feature"] != "intercept"].copy()
    sign_agree = float(non_intercept["direction_same"].mean()) if not non_intercept.empty else np.nan
    coef_corr = float(np.corrcoef(non_intercept["coef_judge"], non_intercept["coef_fan"])[0, 1]) if non_intercept.shape[0] > 1 else np.nan
    consistency_df = pd.DataFrame([
        {"metric": "sign_agree", "value": sign_agree},
        {"metric": "coef_corr", "value": coef_corr},
    ])
    consistency_df.to_csv(OUTPUT_CONSISTENCY, index=False, encoding="utf-8-sig")

    # ============================
    # Visualization: industry scatter & top features (standardized)
    # ============================
    std_non_intercept = std_df[std_df["feature"] != "intercept"].copy()

    # Industry scatter: judge vs fan
    industry = std_non_intercept[std_non_intercept["feature"].str.startswith("celebrity_industry_")].copy()
    if not industry.empty:
        industry["label"] = industry["feature"].str.replace("celebrity_industry_", "", regex=False)
        plt.figure(figsize=(7, 6))
        plt.scatter(industry["coef_std_judge"], industry["coef_std_fan"], alpha=0.8)
        plt.axhline(0, color="#999", linewidth=1)
        plt.axvline(0, color="#999", linewidth=1)
        plt.xlabel("beta_J (judges, standardized)")
        plt.ylabel("beta_V (audience, standardized)")
        plt.title("行业效应散点图：评委 vs 观众")
        for _, row in industry.iterrows():
            plt.text(row["coef_std_judge"], row["coef_std_fan"], row["label"], fontsize=8, alpha=0.8)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_INDUSTRY_SCATTER, dpi=300)
        plt.close()

    # Top features for audience (fan)
    top_fan = std_non_intercept.reindex(std_non_intercept["coef_std_fan"].abs().sort_values(ascending=False).index).head(12)
    if not top_fan.empty:
        plt.figure(figsize=(8, 5))
        plt.barh(top_fan["feature"].str.replace("celebrity_", "", regex=False), top_fan["coef_std_fan"], color="#4C78A8")
        plt.axvline(0, color="#999", linewidth=1)
        plt.xlabel("Coefficient (standardized)")
        plt.title("Top feature effects on Audience (logit vote share)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_TOP_FAN, dpi=300)
        plt.close()

    # Top features for judges
    top_judge = std_non_intercept.reindex(std_non_intercept["coef_std_judge"].abs().sort_values(ascending=False).index).head(12)
    if not top_judge.empty:
        plt.figure(figsize=(8, 5))
        plt.barh(top_judge["feature"].str.replace("celebrity_", "", regex=False), top_judge["coef_std_judge"], color="#F58518")
        plt.axvline(0, color="#999", linewidth=1)
        plt.xlabel("Coefficient (standardized)")
        plt.title("Top feature effects on Judges (logit score share)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_TOP_JUDGE, dpi=300)
        plt.close()

    # Marginal effects for categorical factors (relative to baseline)
    marginal_rows = []
    for prefix in ["ballroom_partner_", "celebrity_industry_", "celebrity_homestate_", "celebrity_homecountry/region_"]:
        subset = coef_merge[coef_merge["feature"].str.startswith(prefix)].copy()
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            marginal_rows.append({
                "feature": row["feature"],
                "group": prefix,
                "coef_judge": row["coef_judge"],
                "coef_fan": row["coef_fan"],
            })
    marginal_df = pd.DataFrame(marginal_rows)
    marginal_df.to_csv(OUTPUT_MARGINAL, index=False, encoding="utf-8-sig")

    # VIF
    vif_df = compute_vif(X)
    vif_df.to_csv(OUTPUT_VIF, index=False, encoding="utf-8-sig")

    # Placement analysis (season-level)
    if "placement" in raw.columns:
        season_panel = (
            panel.groupby(["season", "celebrity_name"])
            .agg(
                avg_judge=("judge_total", "mean"),
                avg_fan=("fan_vote_share", "mean"),
            )
            .reset_index()
        )
        placement_df = raw[["season", "celebrity_name", "placement"]].drop_duplicates()
        placement_df = season_panel.merge(placement_df, on=["season", "celebrity_name"], how="left")
        # Lower placement is better, so negative correlation is desirable
        placement_df["corr_judge_placement"] = placement_df["avg_judge"].corr(placement_df["placement"])
        placement_df["corr_fan_placement"] = placement_df["avg_fan"].corr(placement_df["placement"])
        placement_df.to_csv(OUTPUT_PLACEMENT, index=False, encoding="utf-8-sig")


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
    print(f"一致性指标已保存：{OUTPUT_CONSISTENCY}")
    print(f"图像已保存：{OUTPUT_PLOT}")
    print(f"行业散点图已保存：{OUTPUT_FIG_INDUSTRY_SCATTER}")
    print(f"观众端特征图已保存：{OUTPUT_FIG_TOP_FAN}")
    print(f"评委端特征图已保存：{OUTPUT_FIG_TOP_JUDGE}")


if __name__ == "__main__":
    main()
