# -*- coding: utf-8 -*-
"""
Problem C - Question 4
Propose an alternative "fair" weekly combination system and compute outcomes.

Model structure (from paper):
S_{i,s,t} = alpha * phi(J_{i,s,t}) + beta * psi(F_{i,s,t})
- phi: judge score share within the week
- psi: fan vote share within the week (estimated in Q1a)
- alpha, beta: weights (default equal for neutral balance)
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
# Parameters (explicit, interpretable)
# ============================
ALPHA = 0.5  # weight for judge score share
BETA = 0.5   # weight for fan vote share

# ============================
# File paths
# ============================
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR = ROOT_DIR / "figures"

INPUT_VOTES = OUTPUT_DIR / "fan_vote_estimates_q1a.csv"

OUTPUT_WEEKLY = OUTPUT_DIR / "fair_rule_weekly_q4.csv"
OUTPUT_SEASON_SUM = OUTPUT_DIR / "fair_rule_season_q4.csv"
PLOT_FILE = FIG_DIR / "fair_rule_components_q4.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Main process
# ============================

def main():
    # Load estimated fan votes (Q1a output)
    df = pd.read_csv(INPUT_VOTES)

    required_cols = {"season", "week", "celebrity_name", "judge_total", "fan_vote_share"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    weekly_rows = []

    # Process each season-week
    for (season, week), df_week in df.groupby(["season", "week"]):
        df_week = df_week.copy()

        # Compute judge share within the week (phi)
        judge_sum = df_week["judge_total"].sum()
        df_week["judge_share"] = df_week["judge_total"] / judge_sum

        # Fan vote share within the week (psi) is already a share
        df_week["fan_share"] = df_week["fan_vote_share"]

        # Combined fair score
        df_week["fair_score"] = ALPHA * df_week["judge_share"] + BETA * df_week["fan_share"]

        # Eliminate lowest fair_score
        eliminated_idx = df_week["fair_score"].idxmin()
        df_week["eliminated_fair_rule"] = (df_week.index == eliminated_idx).astype(int)

        for _, row in df_week.iterrows():
            weekly_rows.append({
                "season": season,
                "week": week,
                "celebrity_name": row["celebrity_name"],
                "judge_total": row["judge_total"],
                "judge_share": row["judge_share"],
                "fan_share": row["fan_share"],
                "fair_score": row["fair_score"],
                "eliminated_fair_rule": row["eliminated_fair_rule"],
            })

    weekly_df = pd.DataFrame(weekly_rows)

    # Season summary: count of weeks and unique eliminated names
    season_sum = (
        weekly_df.groupby("season")
        .agg(
            weeks_compared=("week", "nunique"),
            eliminations=("eliminated_fair_rule", "sum")
        )
        .reset_index()
    )

    # Save outputs
    weekly_df.to_csv(OUTPUT_WEEKLY, index=False, encoding="utf-8-sig")
    season_sum.to_csv(OUTPUT_SEASON_SUM, index=False, encoding="utf-8-sig")

    # Display key results
    print("=== 公平合成规则周次结果（前20行） ===")
    print(weekly_df.head(20))
    print("\n=== 赛季汇总（前20行） ===")
    print(season_sum.head(20))

    # Visualization: components for latest season finalists (top 3 by placement)
    latest_season = weekly_df["season"].max()
    # Use fan vote dataset to identify finalists
    finalists = (
        df[df["season"] == latest_season]
        .groupby("celebrity_name")
        .size()
        .sort_values(ascending=False)
        .head(3)
        .index
        .tolist()
    )

    plot_df = weekly_df[(weekly_df["season"] == latest_season) & (weekly_df["celebrity_name"].isin(finalists))]
    if not plot_df.empty:
        plt.figure(figsize=(10, 6))
        for name, sub in plot_df.groupby("celebrity_name"):
            sub_sorted = sub.sort_values("week")
            plt.plot(sub_sorted["week"], sub_sorted["fair_score"], marker="o", label=f"{name}-合成")
        plt.title(f"第 {latest_season} 季决赛选手公平合成得分趋势")
        plt.xlabel("周次")
        plt.ylabel("合成得分")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=300)
        plt.show()

    print(f"\n周次结果已保存：{OUTPUT_WEEKLY}")
    print(f"赛季汇总已保存：{OUTPUT_SEASON_SUM}")
    print(f"图像已保存：{PLOT_FILE}")


if __name__ == "__main__":
    main()
