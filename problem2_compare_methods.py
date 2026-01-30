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
INPUT_VOTES = BASE_DIR / "fan_vote_estimates_q1a.csv"
OUTPUT_WEEK_DIFF = BASE_DIR / "rule_comparison_weekly_q2a.csv"
OUTPUT_SEASON_SUM = BASE_DIR / "rule_comparison_season_q2a.csv"
PLOT_FILE = BASE_DIR / "rule_comparison_diff_ratio_q2a.png"
OUTPUT_CONTROVERSY = BASE_DIR / "rule_comparison_controversy_q2b.csv"
OUTPUT_CONTROVERSY_SUM = BASE_DIR / "rule_comparison_controversy_summary_q2b.csv"

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


# ============================
# Main process
# ============================

def main():
    # Load estimated fan votes
    df = pd.read_csv(INPUT_VOTES)

    # Ensure required columns
    required_cols = {"season", "week", "celebrity_name", "judge_total", "fan_vote_share"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    week_rows = []

    for (season, week), df_week in df.groupby(["season", "week"]):
        df_week = df_week.reset_index(drop=True)

        # Rank-based elimination
        rank_elim_idx, rank_scores = rank_based_elimination(df_week)
        rank_elim_name = df_week.loc[rank_elim_idx, "celebrity_name"]

        # Percent-based elimination
        pct_elim_idx, pct_scores = percent_based_elimination(df_week)
        pct_elim_name = df_week.loc[pct_elim_idx, "celebrity_name"]

        week_rows.append({
            "season": season,
            "week": week,
            "rank_eliminated": rank_elim_name,
            "percent_eliminated": pct_elim_name,
            "different": int(rank_elim_name != pct_elim_name)
        })

    week_df = pd.DataFrame(week_rows)

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

    # Save outputs
    week_df.to_csv(OUTPUT_WEEK_DIFF, index=False, encoding="utf-8-sig")
    season_sum.to_csv(OUTPUT_SEASON_SUM, index=False, encoding="utf-8-sig")

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

    controversy_df = pd.DataFrame(controversy_rows)
    summary_df = pd.DataFrame(summary_rows)

    controversy_df.to_csv(OUTPUT_CONTROVERSY, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_CONTROVERSY_SUM, index=False, encoding="utf-8-sig")

    print("\n=== 争议选手周次对比（前20行） ===")
    print(controversy_df.head(20))
    print("\n=== 争议选手汇总 ===")
    print(summary_df)

    print(f"\n争议选手对比已保存：{OUTPUT_CONTROVERSY}")
    print(f"争议选手汇总已保存：{OUTPUT_CONTROVERSY_SUM}")


if __name__ == "__main__":
    main()
