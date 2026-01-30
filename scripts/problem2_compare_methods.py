# -*- coding: utf-8 -*-
"""
Problem C - Question 2
Compare rank-based vs percent-based combination rules across seasons
using estimated fan votes from Question 1a.

This script follows the model structure in the paper:
- For each processed week, compute the elimination result under BOTH rules
- Compare outputs to identify differences
- Summarize differences by season and overall
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

INPUT_VOTES = OUTPUT_DIR / "fan_vote_estimates_q1a.csv"
DATA_FILE = DATA_DIR / "2026_MCM_Problem_C_Data.csv"
OUTPUT_WEEK_DIFF = OUTPUT_DIR / "rule_comparison_weekly_q2a.csv"
OUTPUT_SEASON_SUM = OUTPUT_DIR / "rule_comparison_season_q2a.csv"
PLOT_FILE = FIG_DIR / "rule_comparison_diff_ratio_q2a.png"
OUTPUT_CONTROVERSY = OUTPUT_DIR / "rule_comparison_controversy_q2b.csv"
OUTPUT_CONTROVERSY_SUM = OUTPUT_DIR / "rule_comparison_controversy_summary_q2b.csv"
OUTPUT_BIAS = OUTPUT_DIR / "rule_bias_metrics_q2.csv"
OUTPUT_CONTROVERSY_ANALYSIS = OUTPUT_DIR / "rule_controversy_analysis_q2.csv"
OUTPUT_NONCONTROVERSY = OUTPUT_DIR / "rule_non_controversy_summary_q2.csv"
OUTPUT_RECOMMENDATION = OUTPUT_DIR / "rule_recommendation_q2.md"
OUTPUT_SEASON_BIAS = OUTPUT_DIR / "rule_bias_by_season_q2.csv"
PLOT_BIAS = FIG_DIR / "rule_bias_alignment_q2.png"
PLOT_CONFLICT = FIG_DIR / "rule_conflict_override_q2.png"
PLOT_CONTROVERSY = FIG_DIR / "rule_controversy_proxy_rank_q2.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Helper functions
# ============================

def rank_based_elimination(df_week):
    """
    Rank-based rule:
    - Judge rank + Fan rank (from fan vote share ranking)
    - Eliminate contestant with maximum sum of ranks
    """
    # Judge rank (higher score -> better rank = 1)
    judge_rank = df_week["judge_total"].rank(ascending=False, method="average")
    # Fan rank from estimated vote shares (higher share -> better rank = 1)
    fan_rank = df_week["fan_vote_share"].rank(ascending=False, method="average")
    sum_rank = judge_rank + fan_rank
    eliminated_idx = sum_rank.idxmax()
    return eliminated_idx, sum_rank


def percent_based_elimination(df_week):
    """
    Percent-based rule:
    - Judge percent + Fan percent
    - Eliminate contestant with minimum sum of percents
    """
    judge_percent = df_week["judge_total"] / df_week["judge_total"].sum()
    fan_percent = df_week["fan_vote_share"]  # already a share
    sum_percent = judge_percent + fan_percent
    eliminated_idx = sum_percent.idxmin()
    return eliminated_idx, sum_percent


def bottom_two_judges_choice(df_week, scheme="rank"):
    """
    Apply the "bottom two + judges choose" mechanism.
    - Identify bottom two under the specified scheme
    - Judges choose the one with lower judge_total
    Returns: (bottom_two_indices, eliminated_index)
    """
    if scheme == "rank":
        judge_rank = df_week["judge_total"].rank(ascending=False, method="average")
        fan_rank = df_week["fan_vote_share"].rank(ascending=False, method="average")
        sum_rank = judge_rank + fan_rank
        bottom_two = sum_rank.nlargest(2).index.tolist()
    else:
        judge_percent = df_week["judge_total"] / df_week["judge_total"].sum()
        fan_percent = df_week["fan_vote_share"]
        sum_percent = judge_percent + fan_percent
        bottom_two = sum_percent.nsmallest(2).index.tolist()

    # Judges choose lower judge_total among bottom two
    judge_total_bottom = df_week.loc[bottom_two, "judge_total"]
    eliminated_idx = judge_total_bottom.idxmin()
    return bottom_two, eliminated_idx


def chi2_pvalue_df1(chi2_stat: float) -> float:
    """Approximate p-value for chi-square with 1 df using erf."""
    if np.isnan(chi2_stat):
        return np.nan
    return 1.0 - math.erf(math.sqrt(max(chi2_stat, 0.0) / 2.0))


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """McNemar's test with continuity correction for paired binary outcomes."""
    if b + c == 0:
        return np.nan, np.nan
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return chi2, chi2_pvalue_df1(chi2)


def compute_proxy_rank(df_season: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Proxy season rank by averaging weekly composite scores."""
    df_season = df_season.copy()
    if scheme == "rank":
        judge_rank = df_season.groupby("week")["judge_total"].rank(ascending=False, method="average")
        fan_rank = df_season.groupby("week")["fan_vote_share"].rank(ascending=False, method="average")
        composite = -(judge_rank + fan_rank)  # higher is better
    else:
        judge_percent = df_season.groupby("week")["judge_total"].transform(lambda s: s / s.sum())
        fan_percent = df_season["fan_vote_share"]
        composite = judge_percent + fan_percent

    df_season["composite"] = composite
    agg = (
        df_season.groupby("celebrity_name")["composite"]
        .mean()
        .reset_index()
        .rename(columns={"composite": "proxy_score"})
    )
    agg["proxy_rank"] = agg["proxy_score"].rank(ascending=False, method="min")
    return agg


# ============================
# Main process
# ============================

def main():
    # Load estimated fan votes
    df = pd.read_csv(INPUT_VOTES)
    raw_df = pd.read_csv(DATA_FILE)

    # Ensure required columns
    required_cols = {"season", "week", "celebrity_name", "judge_total", "fan_vote_share"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    week_rows = []
    bias_rows = []
    judge_only_rows = []

    for (season, week), df_week in df.groupby(["season", "week"]):
        df_week = df_week.reset_index(drop=True)

        # Rank-based elimination
        rank_elim_idx, rank_scores = rank_based_elimination(df_week)
        rank_elim_name = df_week.loc[rank_elim_idx, "celebrity_name"]

        # Percent-based elimination
        pct_elim_idx, pct_scores = percent_based_elimination(df_week)
        pct_elim_name = df_week.loc[pct_elim_idx, "celebrity_name"]

        # Judge-only and fan-only elimination (for bias analysis)
        judge_worst_idx = df_week["judge_total"].idxmin()
        judge_worst_name = df_week.loc[judge_worst_idx, "celebrity_name"]
        fan_worst_idx = df_week["fan_vote_share"].idxmin()
        fan_worst_name = df_week.loc[fan_worst_idx, "celebrity_name"]
        conflict = int(judge_worst_name != fan_worst_name)

        # Bottom-two + judges choice (for impact analysis)
        rank_bottom_two, rank_judge_elim_idx = bottom_two_judges_choice(df_week, scheme="rank")
        pct_bottom_two, pct_judge_elim_idx = bottom_two_judges_choice(df_week, scheme="percent")
        rank_judge_elim_name = df_week.loc[rank_judge_elim_idx, "celebrity_name"]
        pct_judge_elim_name = df_week.loc[pct_judge_elim_idx, "celebrity_name"]

        # Judge-choice override of fan worst when fan worst is in bottom two
        rank_fan_in_bottom2 = int(fan_worst_name in df_week.loc[rank_bottom_two, "celebrity_name"].tolist())
        pct_fan_in_bottom2 = int(fan_worst_name in df_week.loc[pct_bottom_two, "celebrity_name"].tolist())
        rank_judge_overrides_fan = int(rank_fan_in_bottom2 and rank_judge_elim_name != fan_worst_name)
        pct_judge_overrides_fan = int(pct_fan_in_bottom2 and pct_judge_elim_name != fan_worst_name)

        week_rows.append({
            "season": season,
            "week": week,
            "rank_eliminated": rank_elim_name,
            "percent_eliminated": pct_elim_name,
            "different": int(rank_elim_name != pct_elim_name)
        })

        bias_rows.append({
            "season": season,
            "week": week,
            "conflict": conflict,
            "judge_worst": judge_worst_name,
            "fan_worst": fan_worst_name,
            "rank_eliminated": rank_elim_name,
            "percent_eliminated": pct_elim_name,
            "rank_fan_aligned": int(rank_elim_name == fan_worst_name),
            "rank_judge_aligned": int(rank_elim_name == judge_worst_name),
            "percent_fan_aligned": int(pct_elim_name == fan_worst_name),
            "percent_judge_aligned": int(pct_elim_name == judge_worst_name),
            "rank_judge_save_diff": int(rank_judge_elim_name != rank_elim_name),
            "percent_judge_save_diff": int(pct_judge_elim_name != pct_elim_name),
            "rank_fan_in_bottom2": rank_fan_in_bottom2,
            "percent_fan_in_bottom2": pct_fan_in_bottom2,
            "rank_judge_overrides_fan": rank_judge_overrides_fan,
            "percent_judge_overrides_fan": pct_judge_overrides_fan,
        })

        judge_only_rows.append({
            "season": season,
            "week": week,
            "rank_bottom_two": " / ".join(df_week.loc[rank_bottom_two, "celebrity_name"].tolist()),
            "percent_bottom_two": " / ".join(df_week.loc[pct_bottom_two, "celebrity_name"].tolist()),
            "rank_judges_eliminated": rank_judge_elim_name,
            "percent_judges_eliminated": pct_judge_elim_name,
        })

    week_df = pd.DataFrame(week_rows)
    bias_df = pd.DataFrame(bias_rows)

    # Season summary
    season_sum = (
        week_df.groupby("season")
        .agg(
            weeks_compared=("different", "count"),
            differences=("different", "sum")
        )
        .reset_index()
    )
    season_sum["difference_ratio"] = season_sum["differences"] / season_sum["weeks_compared"]

    # Bias metrics (overall and by season)
    conflict_weeks = bias_df[bias_df["conflict"] == 1]
    # McNemar for fan alignment in conflict weeks (rank vs percent)
    if not conflict_weeks.empty:
        b = int(((conflict_weeks["rank_fan_aligned"] == 1) & (conflict_weeks["percent_fan_aligned"] == 0)).sum())
        c = int(((conflict_weeks["rank_fan_aligned"] == 0) & (conflict_weeks["percent_fan_aligned"] == 1)).sum())
        chi2_fan, p_fan = mcnemar_test(b, c)

        b_j = int(((conflict_weeks["rank_judge_aligned"] == 1) & (conflict_weeks["percent_judge_aligned"] == 0)).sum())
        c_j = int(((conflict_weeks["rank_judge_aligned"] == 0) & (conflict_weeks["percent_judge_aligned"] == 1)).sum())
        chi2_judge, p_judge = mcnemar_test(b_j, c_j)
    else:
        chi2_fan = p_fan = chi2_judge = p_judge = np.nan
    overall_bias = {
        "scope": "overall",
        "weeks": len(bias_df),
        "conflict_weeks": len(conflict_weeks),
        "rank_fan_align_rate": bias_df["rank_fan_aligned"].mean(),
        "rank_judge_align_rate": bias_df["rank_judge_aligned"].mean(),
        "percent_fan_align_rate": bias_df["percent_fan_aligned"].mean(),
        "percent_judge_align_rate": bias_df["percent_judge_aligned"].mean(),
        "rank_fan_override_rate": conflict_weeks["rank_fan_aligned"].mean() if not conflict_weeks.empty else np.nan,
        "percent_fan_override_rate": conflict_weeks["percent_fan_aligned"].mean() if not conflict_weeks.empty else np.nan,
        "rank_judge_save_change_rate": bias_df["rank_judge_save_diff"].mean(),
        "percent_judge_save_change_rate": bias_df["percent_judge_save_diff"].mean(),
        "rank_judge_overrides_fan_rate": conflict_weeks["rank_judge_overrides_fan"].mean() if not conflict_weeks.empty else np.nan,
        "percent_judge_overrides_fan_rate": conflict_weeks["percent_judge_overrides_fan"].mean() if not conflict_weeks.empty else np.nan,
        "mcnemar_fan_chi2": chi2_fan,
        "mcnemar_fan_p": p_fan,
        "mcnemar_judge_chi2": chi2_judge,
        "mcnemar_judge_p": p_judge,
    }

    season_bias = (
        bias_df.groupby("season")
        .agg(
            weeks=("week", "count"),
            conflict_weeks=("conflict", "sum"),
            rank_fan_align_rate=("rank_fan_aligned", "mean"),
            rank_judge_align_rate=("rank_judge_aligned", "mean"),
            percent_fan_align_rate=("percent_fan_aligned", "mean"),
            percent_judge_align_rate=("percent_judge_aligned", "mean"),
            rank_judge_save_change_rate=("rank_judge_save_diff", "mean"),
            percent_judge_save_change_rate=("percent_judge_save_diff", "mean"),
            rank_judge_overrides_fan_rate=("rank_judge_overrides_fan", "mean"),
            percent_judge_overrides_fan_rate=("percent_judge_overrides_fan", "mean"),
        )
        .reset_index()
    )
    season_bias.insert(0, "scope", "season")
    overall_bias_df = pd.DataFrame([overall_bias])
    bias_metrics_df = pd.concat([overall_bias_df, season_bias], ignore_index=True)
    season_bias.to_csv(OUTPUT_SEASON_BIAS, index=False, encoding="utf-8-sig")

    # Save outputs
    week_df.to_csv(OUTPUT_WEEK_DIFF, index=False, encoding="utf-8-sig")
    season_sum.to_csv(OUTPUT_SEASON_SUM, index=False, encoding="utf-8-sig")
    bias_metrics_df.to_csv(OUTPUT_BIAS, index=False, encoding="utf-8-sig")

    # Display key results
    print("=== Weekly Rule Comparison (Head) ===")
    print(week_df.head(20))
    print("\n=== Season Summary (Head) ===")
    print(season_sum.head(20))

    # Visualization: difference ratio by season
    plt.figure(figsize=(10, 6))
    plt.bar(season_sum["season"].astype(str), season_sum["difference_ratio"], color="#4C78A8")
    plt.title("各赛季两种规则淘汰差异比例")
    plt.xlabel("赛季")
    plt.ylabel("差异比例")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.show()

    print(f"\n周次对比结果已保存至：{OUTPUT_WEEK_DIFF}")
    print(f"赛季汇总结果已保存至：{OUTPUT_SEASON_SUM}")
    print(f"图像已保存至：{PLOT_FILE}")
    print(f"偏向性指标已保存至：{OUTPUT_BIAS}")
    print(f"分赛季偏向性指标已保存至：{OUTPUT_SEASON_BIAS}")

    # ============================
    # Controversy cases (Q2b)
    # ============================
    controversy_cases = [
        {"season": 2, "celebrity_name": "Jerry Rice"},
        {"season": 4, "celebrity_name": "Billy Ray Cyrus"},
        {"season": 11, "celebrity_name": "Bristol Palin"},
        {"season": 27, "celebrity_name": "Bobby Bones"},
    ]

    controversy_rows = []
    summary_rows = []
    analysis_rows = []

    for case in controversy_cases:
        season = case["season"]
        name = case["celebrity_name"]

        df_case = df[(df["season"] == season) & (df["celebrity_name"] == name)].copy()

        # If exact match not found, fall back to case-insensitive contains
        if df_case.empty:
            df_case = df[(df["season"] == season) & (df["celebrity_name"].str.contains(name, case=False, na=False))]

        if df_case.empty:
            summary_rows.append({
                "season": season,
                "celebrity_name": name,
                "status": "not found in fan_vote_estimates"
            })
            continue

        weeks = sorted(df_case["week"].unique())
        actual_elim_week = None
        if "eliminated_end_of_week" in df_case.columns:
            elim_weeks = df_case[df_case["eliminated_end_of_week"] == 1]["week"].tolist()
            actual_elim_week = min(elim_weeks) if elim_weeks else None

        # Actual placement from raw data if available
        actual_place = None
        if "placement" in raw_df.columns:
            placement_rows = raw_df[(raw_df["season"] == season) & (raw_df["celebrity_name"] == name)]
            if placement_rows.empty:
                placement_rows = raw_df[(raw_df["season"] == season) & (raw_df["celebrity_name"].str.contains(name, case=False, na=False))]
            if not placement_rows.empty:
                actual_place = placement_rows["placement"].iloc[0]

        # Track predicted elimination week under each rule
        first_rank_elim = None
        first_percent_elim = None
        first_rank_judge_elim = None
        first_percent_judge_elim = None

        for week in weeks:
            df_week = df[(df["season"] == season) & (df["week"] == week)].reset_index(drop=True)

            # Rank-based
            rank_elim_idx, _ = rank_based_elimination(df_week)
            rank_elim_name = df_week.loc[rank_elim_idx, "celebrity_name"]

            # Percent-based
            pct_elim_idx, _ = percent_based_elimination(df_week)
            pct_elim_name = df_week.loc[pct_elim_idx, "celebrity_name"]

            # Bottom-two + judges choice
            rank_bottom_two, rank_judge_elim_idx = bottom_two_judges_choice(df_week, scheme="rank")
            pct_bottom_two, pct_judge_elim_idx = bottom_two_judges_choice(df_week, scheme="percent")
            rank_judge_elim_name = df_week.loc[rank_judge_elim_idx, "celebrity_name"]
            pct_judge_elim_name = df_week.loc[pct_judge_elim_idx, "celebrity_name"]

            # Record earliest predicted elimination week
            if rank_elim_name == name and first_rank_elim is None:
                first_rank_elim = week
            if pct_elim_name == name and first_percent_elim is None:
                first_percent_elim = week
            if rank_judge_elim_name == name and first_rank_judge_elim is None:
                first_rank_judge_elim = week
            if pct_judge_elim_name == name and first_percent_judge_elim is None:
                first_percent_judge_elim = week

            controversy_rows.append({
                "season": season,
                "week": week,
                "celebrity_name": name,
                "rank_eliminated": rank_elim_name,
                "percent_eliminated": pct_elim_name,
                "rank_bottom_two": " / ".join(df_week.loc[rank_bottom_two, "celebrity_name"].tolist()),
                "percent_bottom_two": " / ".join(df_week.loc[pct_bottom_two, "celebrity_name"].tolist()),
                "rank_judges_eliminated": rank_judge_elim_name,
                "percent_judges_eliminated": pct_judge_elim_name,
            })

        # Proxy ranks under both rules
        season_df = df[df["season"] == season].copy()
        rank_proxy = compute_proxy_rank(season_df, scheme="rank")
        percent_proxy = compute_proxy_rank(season_df, scheme="percent")
        rank_proxy_row = rank_proxy[rank_proxy["celebrity_name"].str.contains(name, case=False, na=False)]
        percent_proxy_row = percent_proxy[percent_proxy["celebrity_name"].str.contains(name, case=False, na=False)]

        rank_proxy_rank = int(rank_proxy_row["proxy_rank"].iloc[0]) if not rank_proxy_row.empty else None
        percent_proxy_rank = int(percent_proxy_row["proxy_rank"].iloc[0]) if not percent_proxy_row.empty else None

        # Judge-save exposure
        case_weeks_df = pd.DataFrame([r for r in controversy_rows if r["season"] == season and r["celebrity_name"] == name])
        rank_judge_save_hits = int((case_weeks_df["rank_judges_eliminated"] == name).sum()) if not case_weeks_df.empty else 0
        percent_judge_save_hits = int((case_weeks_df["percent_judges_eliminated"] == name).sum()) if not case_weeks_df.empty else 0

        summary_rows.append({
            "season": season,
            "celebrity_name": name,
            "actual_elim_week": actual_elim_week,
            "rank_elim_week": first_rank_elim,
            "percent_elim_week": first_percent_elim,
            "rank_judges_elim_week": first_rank_judge_elim,
            "percent_judges_elim_week": first_percent_judge_elim,
            "status": "processed"
        })

        analysis_rows.append({
            "season": season,
            "celebrity_name": name,
            "actual_placement": actual_place,
            "rank_proxy_rank": rank_proxy_rank,
            "percent_proxy_rank": percent_proxy_rank,
            "rank_finalist_proxy": int(rank_proxy_rank is not None and rank_proxy_rank <= 3),
            "percent_finalist_proxy": int(percent_proxy_rank is not None and percent_proxy_rank <= 3),
            "rank_judge_save_hits": rank_judge_save_hits,
            "percent_judge_save_hits": percent_judge_save_hits,
            "rank_vs_percent_same_finalist": int(
                (rank_proxy_rank is not None and percent_proxy_rank is not None) and
                ((rank_proxy_rank <= 3) == (percent_proxy_rank <= 3))
            ),
            "status": "processed",
        })

    controversy_df = pd.DataFrame(controversy_rows)
    summary_df = pd.DataFrame(summary_rows)
    analysis_df = pd.DataFrame(analysis_rows)

    controversy_df.to_csv(OUTPUT_CONTROVERSY, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_CONTROVERSY_SUM, index=False, encoding="utf-8-sig")
    analysis_df.to_csv(OUTPUT_CONTROVERSY_ANALYSIS, index=False, encoding="utf-8-sig")

    print("\n=== 争议选手周次对比（前20行） ===")
    print(controversy_df.head(20))
    print("\n=== 争议选手汇总 ===")
    print(summary_df)

    print(f"\n争议选手对比已保存：{OUTPUT_CONTROVERSY}")
    print(f"争议选手汇总已保存：{OUTPUT_CONTROVERSY_SUM}")
    print(f"争议选手分析已保存：{OUTPUT_CONTROVERSY_ANALYSIS}")

    # ============================
    # Non-controversy seasons analysis
    # ============================
    controversy_seasons = {c["season"] for c in controversy_cases}
    non_controversy_df = week_df[~week_df["season"].isin(controversy_seasons)]
    non_controversy_bias = bias_df[~bias_df["season"].isin(controversy_seasons)]
    non_summary = {
        "scope": "non_controversy_seasons",
        "weeks": int(non_controversy_df.shape[0]),
        "difference_ratio": float(non_controversy_df["different"].mean()) if not non_controversy_df.empty else np.nan,
        "rank_fan_align_rate": float(non_controversy_bias["rank_fan_aligned"].mean()) if not non_controversy_bias.empty else np.nan,
        "percent_fan_align_rate": float(non_controversy_bias["percent_fan_aligned"].mean()) if not non_controversy_bias.empty else np.nan,
        "rank_judge_save_change_rate": float(non_controversy_bias["rank_judge_save_diff"].mean()) if not non_controversy_bias.empty else np.nan,
        "percent_judge_save_change_rate": float(non_controversy_bias["percent_judge_save_diff"].mean()) if not non_controversy_bias.empty else np.nan,
    }
    non_summary_df = pd.DataFrame([non_summary])
    non_summary_df.to_csv(OUTPUT_NONCONTROVERSY, index=False, encoding="utf-8-sig")

    # ============================
    # Visualization
    # ============================
    plt.figure(figsize=(8, 5))
    labels = ["rank_fan", "percent_fan", "rank_judge", "percent_judge"]
    values = [
        overall_bias["rank_fan_align_rate"],
        overall_bias["percent_fan_align_rate"],
        overall_bias["rank_judge_align_rate"],
        overall_bias["percent_judge_align_rate"],
    ]
    plt.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"])
    plt.title("两种规则对粉丝/评委选择的一致率")
    plt.ylabel("一致率")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOT_BIAS, dpi=300)
    plt.show()

    if not conflict_weeks.empty:
        plt.figure(figsize=(6, 4))
        labels = ["rank_override", "percent_override"]
        values = [
            overall_bias["rank_judge_overrides_fan_rate"],
            overall_bias["percent_judge_overrides_fan_rate"],
        ]
        plt.bar(labels, values, color=["#A05195", "#FFA600"])
        plt.title("评委选择对粉丝淘汰的覆盖率（冲突周次）")
        plt.ylabel("覆盖率")
        plt.ylim(0, 1)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(PLOT_CONFLICT, dpi=300)
        plt.show()

    if not analysis_df.empty:
        plot_df = analysis_df.dropna(subset=["rank_proxy_rank", "percent_proxy_rank"])\
            .set_index("celebrity_name")[["rank_proxy_rank", "percent_proxy_rank"]]
        if not plot_df.empty:
            plot_df.plot(kind="bar", figsize=(8, 5))
            plt.title("争议选手代理最终名次对比（排名法 vs 百分比法）")
            plt.ylabel("代理名次（数值越小越好）")
            plt.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(PLOT_CONTROVERSY, dpi=300)
            plt.show()

    # ============================
    # Recommendation (Q2c)
    # ============================
    rank_fan = overall_bias["rank_fan_align_rate"]
    pct_fan = overall_bias["percent_fan_align_rate"]
    rank_judge = overall_bias["rank_judge_align_rate"]
    pct_judge = overall_bias["percent_judge_align_rate"]
    preferred = "percent" if pct_fan > rank_fan else "rank"
    judge_save_effect_rank = overall_bias["rank_judge_save_change_rate"]
    judge_save_effect_pct = overall_bias["percent_judge_save_change_rate"]
    recommend_judge_save = (judge_save_effect_rank > 0.05) or (judge_save_effect_pct > 0.05)

    rec_lines = [
        "# 规则比较与建议（问题二）",
        "",
        f"- 总体周次数：{overall_bias['weeks']}，冲突周次数：{overall_bias['conflict_weeks']}",
        f"- 排名法：粉丝一致率={rank_fan:.3f}，评委一致率={rank_judge:.3f}",
        f"- 百分比法：粉丝一致率={pct_fan:.3f}，评委一致率={pct_judge:.3f}",
        "",
        f"**偏向性结论**：{('百分比法' if preferred=='percent' else '排名法')} 更偏向观众投票（基于粉丝一致率比较）。",
        f"**显著性检验（冲突周次）**：粉丝一致率 McNemar p={overall_bias['mcnemar_fan_p']:.4f}，评委一致率 McNemar p={overall_bias['mcnemar_judge_p']:.4f}",
        "",
        f"- 底二评委选择改变淘汰的比例（排名法）={judge_save_effect_rank:.3f}",
        f"- 底二评委选择改变淘汰的比例（百分比法）={judge_save_effect_pct:.3f}",
        f"- 评委对粉丝淘汰的覆盖率（排名法）={overall_bias['rank_judge_overrides_fan_rate']:.3f}",
        f"- 评委对粉丝淘汰的覆盖率（百分比法）={overall_bias['percent_judge_overrides_fan_rate']:.3f}",
        "",
        f"**额外环节建议**：{'建议增加' if recommend_judge_save else '不建议强制增加'}（依据改变比例与争议样本稳定性）",
        "",
        "**争议案例结论**：见 rule_controversy_analysis_q2.csv，给出代理最终名次（proxy_rank）与评委选择的淘汰风险（judge_save_hits）。",
        "**非争议赛季结论**：见 rule_non_controversy_summary_q2.csv，用于验证常规赛季的稳定性。",
    ]
    OUTPUT_RECOMMENDATION.write_text("\n".join(rec_lines), encoding="utf-8")
    print(f"推荐结论已保存：{OUTPUT_RECOMMENDATION}")


if __name__ == "__main__":
    main()
