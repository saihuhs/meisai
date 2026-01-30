# -*- coding: utf-8 -*-
"""
Model Evaluation & Sensitivity Analysis
Based strictly on existing model outputs for Problems 1-4.
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

BASE_DIR = Path(__file__).resolve().parent

# Input files (generated previously)
FAN_VOTES = BASE_DIR / "fan_vote_estimates_q1a.csv"
WEEK_STATUS = BASE_DIR / "fan_vote_week_status_q1a.csv"
RULE_WEEKLY = BASE_DIR / "rule_comparison_weekly_q2a.csv"
RULE_SEASON = BASE_DIR / "rule_comparison_season_q2a.csv"
CONTROVERSY_SUM = BASE_DIR / "rule_comparison_controversy_summary_q2b.csv"
MODEL_DATASET = BASE_DIR / "problem3_model_dataset.csv"
FIT_Q3 = BASE_DIR / "problem3_model_fit.csv"
FAIR_WEEKLY = BASE_DIR / "fair_rule_weekly_q4.csv"
RAW_DATA = BASE_DIR / "2026_MCM_Problem_C_Data.csv"

# Output files
OUT_Q1_SUM = BASE_DIR / "eval_q1_summary.csv"
OUT_Q1_SEASON = BASE_DIR / "eval_q1_by_season.csv"
OUT_Q1_SENS = BASE_DIR / "eval_q1_sensitivity.csv"
OUT_Q1_INTERVAL = BASE_DIR / "eval_q1_interval.csv"
OUT_Q1_INTERVAL_SUM = BASE_DIR / "eval_q1_interval_summary.csv"

OUT_Q2_SUM = BASE_DIR / "eval_q2_summary.csv"
OUT_Q2_SENS = BASE_DIR / "eval_q2_sensitivity.csv"

OUT_Q3_LOO = BASE_DIR / "eval_q3_leave_one_season.csv"
OUT_Q3_STAB = BASE_DIR / "eval_q3_coef_stability.csv"

OUT_Q4_SENS = BASE_DIR / "eval_q4_sensitivity.csv"
PLOT_Q4 = BASE_DIR / "eval_q4_alpha_sensitivity.png"


def week_consistency(df_week):
    scheme = df_week["scheme"].iloc[0]
    if scheme == "rank":
        s = df_week["judge_rank"] + df_week["fan_rank"]
        eliminated = df_week[df_week["eliminated_end_of_week"] == 1]
        if eliminated.empty:
            return None
        return s.loc[eliminated.index].iloc[0] >= s.drop(eliminated.index).max()
    else:
        s = df_week["judge_percent"] + df_week["fan_vote_share"]
        eliminated = df_week[df_week["eliminated_end_of_week"] == 1]
        if eliminated.empty:
            return None
        return s.loc[eliminated.index].iloc[0] <= s.drop(eliminated.index).min()


def build_design_matrix_full(df: pd.DataFrame):
    df = df.copy()
    df["celebrity_age_during_season"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    cat_cols = [
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "season",
        "ballroom_partner",
    ]
    X = pd.get_dummies(df[cat_cols], drop_first=True)
    X["age"] = df["celebrity_age_during_season"]
    X.insert(0, "intercept", 1.0)
    return X.fillna(0).astype(float)


def ols_fit(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def assign_fan_ranks_all(rank_judge, eliminated_idx):
    """Generate all feasible fan-rank assignments under rank-based rule."""
    n = len(rank_judge)
    contestants = list(rank_judge.index)
    rJ = rank_judge.to_dict()
    rJ_e = rJ[eliminated_idx]

    feasible_assignments = []
    for rF_e in range(1, n + 1):
        upper_bounds = {}
        feasible = True
        for i in contestants:
            if i == eliminated_idx:
                continue
            U_i = int(np.floor(rF_e + (rJ_e - rJ[i])))
            if U_i < 1:
                feasible = False
                break
            upper_bounds[i] = U_i
        if not feasible:
            continue

        available_ranks = [r for r in range(1, n + 1) if r != rF_e]
        available_ranks.sort()
        assignment = {eliminated_idx: rF_e}

        for i, U_i in sorted(upper_bounds.items(), key=lambda x: x[1]):
            if not available_ranks:
                feasible = False
                break
            if available_ranks[0] > U_i:
                feasible = False
                break
            assignment[i] = available_ranks.pop(0)

        if feasible:
            feasible_assignments.append(assignment)

    return feasible_assignments


def preference_objective(fan_share, judge_scores, prev_share, w_smooth, w_corr):
    """Preference objective: smoothness penalty minus correlation reward."""
    smooth = 0.0
    if prev_share is not None and not prev_share.empty:
        common = fan_share.index.intersection(prev_share.index)
        if len(common) > 0:
            diff = fan_share.loc[common].values - prev_share.loc[common].values
            smooth = float(np.mean(diff ** 2))

    corr = 0.0
    try:
        corr_val = np.corrcoef(
            fan_share.values,
            judge_scores.loc[fan_share.index].values,
        )[0, 1]
        corr = 0.0 if np.isnan(corr_val) else float(corr_val)
    except Exception:
        corr = 0.0

    return w_smooth * smooth - w_corr * corr


def estimate_rank_with_preferences(judge_scores, eliminated_idx, prev_share, w_smooth, w_corr):
    rank_judge = judge_scores.rank(ascending=False, method="average")
    assignments = assign_fan_ranks_all(rank_judge, eliminated_idx)
    if not assignments:
        return None
    best = None
    best_obj = np.inf
    n = len(rank_judge)
    for assignment in assignments:
        fan_rank = pd.Series(assignment)
        fan_score = (n + 1 - fan_rank).astype(float)
        fan_share = fan_score / fan_score.sum()
        obj = preference_objective(fan_share, judge_scores, prev_share, w_smooth, w_corr)
        if obj < best_obj:
            best_obj = obj
            best = fan_share
    return best


def estimate_percent_with_preferences(judge_scores, eliminated_idx, prev_share, w_smooth, w_corr, grid_step=0.001):
    contestants = list(judge_scores.index)
    n = len(contestants)
    pJ = judge_scores / judge_scores.sum()
    pJ_e = pJ.loc[eliminated_idx]

    best_solution = None
    best_obj = np.inf

    for pF_e in np.arange(0.0, 1.0 + grid_step, grid_step):
        lower_bounds = {}
        for i in contestants:
            if i == eliminated_idx:
                continue
            lb_i = pJ_e - pJ.loc[i] + pF_e
            lower_bounds[i] = max(0.0, lb_i)

        min_sum = pF_e + sum(lower_bounds.values())
        if min_sum > 1.0:
            continue

        remaining = 1.0 - min_sum
        extra = remaining / (n - 1)

        pF = {eliminated_idx: pF_e}
        for i in contestants:
            if i == eliminated_idx:
                continue
            pF[i] = lower_bounds[i] + extra

        fan_share = pd.Series(pF)
        obj = preference_objective(fan_share, judge_scores, prev_share, w_smooth, w_corr)
        if obj < best_obj:
            best_obj = obj
            best_solution = fan_share

    return best_solution


def main():
    # ============================
    # Q1: Validity, stability, sensitivity
    # ============================
    results = pd.read_csv(FAN_VOTES)
    status = pd.read_csv(WEEK_STATUS)

    processed = status[status["status"] == "processed"]

    # Consistency
    consistency = []
    for (season, week), df_week in results.groupby(["season", "week"]):
        c = week_consistency(df_week)
        if c is not None:
            consistency.append(c)
    consistency_rate = sum(consistency) / len(consistency) if consistency else np.nan

    # Fan vote share distribution
    fan_stats = results["fan_vote_share"].describe()

    # Stability: week-to-week mean absolute change within season
    results_sorted = results.sort_values(["season", "celebrity_name", "week"])
    results_sorted["fan_share_diff"] = results_sorted.groupby(["season", "celebrity_name"])["fan_vote_share"].diff().abs()
    stability_season = (
        results_sorted.groupby("season")["fan_share_diff"]
        .mean()
        .reset_index()
        .rename(columns={"fan_share_diff": "mean_abs_weekly_change"})
    )

    # Sensitivity to input variability: split weeks by judge_total dispersion
    week_disp = (
        results.groupby(["season", "week"]) ["judge_total"].agg(["std", "count"]).reset_index()
    )
    week_disp["std_group"] = pd.qcut(week_disp["std"], q=2, labels=["low_disp", "high_disp"])
    results = results.merge(week_disp[["season", "week", "std_group"]], on=["season", "week"], how="left")

    def group_consistency(df_sub):
        cons = []
        for (s, w), df_week in df_sub.groupby(["season", "week"]):
            c = week_consistency(df_week)
            if c is not None:
                cons.append(c)
        return sum(cons) / len(cons) if cons else np.nan

    sens_rows = []
    for grp in ["low_disp", "high_disp"]:
        sens_rows.append({
            "group": grp,
            "consistency_rate": group_consistency(results[results["std_group"] == grp])
        })
    sens_df = pd.DataFrame(sens_rows)

    q1_sum = pd.DataFrame([{ 
        "processed_weeks": len(processed),
        "consistency_rate": consistency_rate,
        "fan_share_mean": fan_stats["mean"],
        "fan_share_std": fan_stats["std"],
        "fan_share_min": fan_stats["min"],
        "fan_share_max": fan_stats["max"],
    }])

    q1_sum.to_csv(OUT_Q1_SUM, index=False, encoding="utf-8-sig")
    stability_season.to_csv(OUT_Q1_SEASON, index=False, encoding="utf-8-sig")
    sens_df.to_csv(OUT_Q1_SENS, index=False, encoding="utf-8-sig")

    print("=== Q1 Summary ===")
    print(q1_sum)
    print("=== Q1 Stability by Season (Head) ===")
    print(stability_season.head())
    print("=== Q1 Sensitivity ===")
    print(sens_df)

    # Interval uncertainty via preference-weight sensitivity (same model, different weights)
    weight_settings = [
        {"w_smooth": 0.5, "w_corr": 1.0},
        {"w_smooth": 1.0, "w_corr": 1.0},
        {"w_smooth": 1.5, "w_corr": 1.0},
    ]

    interval_rows = []
    prev_share_by_season = {w_idx: {} for w_idx in range(len(weight_settings))}

    for (season, week), df_week in results.groupby(["season", "week"]):
        df_week = df_week.copy()
        eliminated = df_week[df_week["eliminated_end_of_week"] == 1]
        if eliminated.empty:
            continue
        eliminated_idx = eliminated.index[0]
        judge_scores = df_week["judge_total"]
        scheme = df_week["scheme"].iloc[0]

        fan_shares = []
        for w_idx, w in enumerate(weight_settings):
            prev_share = prev_share_by_season[w_idx].get(season, None)
            if scheme == "rank":
                fan_share = estimate_rank_with_preferences(
                    judge_scores, eliminated_idx, prev_share, w["w_smooth"], w["w_corr"]
                )
            else:
                fan_share = estimate_percent_with_preferences(
                    judge_scores, eliminated_idx, prev_share, w["w_smooth"], w["w_corr"]
                )
            if fan_share is None:
                continue
            fan_shares.append(fan_share)
            prev_share_by_season[w_idx][season] = fan_share

        if not fan_shares:
            continue

        # Interval per contestant-week across weight settings
        combined = pd.concat(fan_shares, axis=1)
        for idx, row in df_week.iterrows():
            if idx not in combined.index:
                continue
            vals = combined.loc[idx].values
            interval_rows.append({
                "season": season,
                "week": week,
                "celebrity_name": row.get("celebrity_name"),
                "fan_share_min": float(np.min(vals)),
                "fan_share_max": float(np.max(vals)),
                "fan_share_width": float(np.max(vals) - np.min(vals)),
            })

    interval_df = pd.DataFrame(interval_rows)
    interval_df.to_csv(OUT_Q1_INTERVAL, index=False, encoding="utf-8-sig")

    interval_summary = interval_df["fan_share_width"].describe().to_frame().T
    interval_summary.to_csv(OUT_Q1_INTERVAL_SUM, index=False, encoding="utf-8-sig")

    print("=== Q1 Interval Uncertainty Summary ===")
    print(interval_summary)

    # ============================
    # Q2: Validity & sensitivity
    # ============================
    rule_week = pd.read_csv(RULE_WEEKLY)
    rule_season = pd.read_csv(RULE_SEASON)

    overall_weeks = rule_week.shape[0]
    overall_diff = rule_week["different"].sum()
    overall_ratio = overall_diff / overall_weeks if overall_weeks else np.nan

    q2_sum = pd.DataFrame([{
        "weeks_compared": overall_weeks,
        "differences": overall_diff,
        "difference_ratio": overall_ratio,
        "seasons_with_diff": (rule_season["differences"] > 0).sum(),
    }])
    q2_sum.to_csv(OUT_Q2_SUM, index=False, encoding="utf-8-sig")

    # Sensitivity: compare difference ratio under low/high judge dispersion
    rule_week = rule_week.merge(week_disp[["season", "week", "std_group"]], on=["season", "week"], how="left")
    sens_q2 = (
        rule_week.groupby("std_group")["different"]
        .mean()
        .reset_index()
        .rename(columns={"different": "difference_ratio"})
    )
    sens_q2.to_csv(OUT_Q2_SENS, index=False, encoding="utf-8-sig")

    print("=== Q2 Summary ===")
    print(q2_sum)
    print("=== Q2 Sensitivity ===")
    print(sens_q2)

    # ============================
    # Q3: Stability via leave-one-season-out (same linear structure)
    # ============================
    dataset = pd.read_csv(MODEL_DATASET)

    X_full = build_design_matrix_full(dataset)
    full_cols = X_full.columns

    loo_rows = []
    coef_rows = []

    for season in sorted(dataset["season"].unique()):
        subset = dataset[dataset["season"] != season].copy()

        X_sub = build_design_matrix_full(subset)
        X_sub = X_sub.reindex(columns=full_cols, fill_value=0.0)

        y_j = subset["judge_total"].to_numpy()
        y_f = subset["fan_vote_share"].to_numpy()

        beta_j, r2_j = ols_fit(X_sub.to_numpy(), y_j)
        beta_f, r2_f = ols_fit(X_sub.to_numpy(), y_f)

        loo_rows.append({"left_out_season": season, "r2_judge": r2_j, "r2_fan": r2_f})

        # store pro dancer coefficients only
        for col, bj, bf in zip(full_cols, beta_j, beta_f):
            if col.startswith("ballroom_partner_"):
                coef_rows.append({
                    "left_out_season": season,
                    "feature": col,
                    "coef_judge": bj,
                    "coef_fan": bf,
                })

    loo_df = pd.DataFrame(loo_rows)
    coef_df = pd.DataFrame(coef_rows)

    # coefficient stability: std across leave-one-season-out
    coef_stab = (
        coef_df.groupby("feature")[["coef_judge", "coef_fan"]]
        .std()
        .reset_index()
        .rename(columns={"coef_judge": "std_coef_judge", "coef_fan": "std_coef_fan"})
        .sort_values("std_coef_judge", ascending=False)
    )

    loo_df.to_csv(OUT_Q3_LOO, index=False, encoding="utf-8-sig")
    coef_stab.to_csv(OUT_Q3_STAB, index=False, encoding="utf-8-sig")

    print("=== Q3 Leave-One-Season-Out (Head) ===")
    print(loo_df.head())
    print("=== Q3 Coef Stability (Head) ===")
    print(coef_stab.head())

    # ============================
    # Q4: Sensitivity to alpha (same fair rule structure)
    # ============================
    fair = pd.read_csv(FAIR_WEEKLY)

    alpha_list = [0.3, 0.5, 0.7]
    base_alpha = 0.5

    # Precompute judge_share and fan_share from output
    sens_rows_q4 = []

    # Identify base eliminations
    base = fair.copy()
    base["fair_score"] = base_alpha * base["judge_share"] + (1 - base_alpha) * base["fan_share"]
    base_elim = (
        base.groupby(["season", "week"]) ["fair_score"]
        .idxmin()
        .reset_index()
        .rename(columns={"fair_score": "base_idx"})
    )

    for alpha in alpha_list:
        temp = fair.copy()
        temp["fair_score"] = alpha * temp["judge_share"] + (1 - alpha) * temp["fan_share"]
        elim_idx = (
            temp.groupby(["season", "week"]) ["fair_score"]
            .idxmin()
            .reset_index()
            .rename(columns={"fair_score": "idx"})
        )
        merged = elim_idx.merge(base_elim, on=["season", "week"], how="left")
        diff_rate = (merged["idx"] != merged["base_idx"]).mean()

        sens_rows_q4.append({"alpha": alpha, "difference_rate_vs_base": diff_rate})

    sens_q4 = pd.DataFrame(sens_rows_q4)
    sens_q4.to_csv(OUT_Q4_SENS, index=False, encoding="utf-8-sig")

    # Plot alpha sensitivity
    plt.figure(figsize=(6, 4))
    plt.plot(sens_q4["alpha"], sens_q4["difference_rate_vs_base"], marker="o")
    plt.title("公平合成规则对权重参数的敏感性")
    plt.xlabel("alpha（裁判权重）")
    plt.ylabel("与基准淘汰差异比例")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_Q4, dpi=300)
    plt.show()

    print("=== Q4 Sensitivity ===")
    print(sens_q4)


if __name__ == "__main__":
    main()
