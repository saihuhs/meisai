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
SENSITIVITY_WEIGHTS = [0.4, 0.5, 0.6]

# ============================
# File paths
# ============================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR = ROOT_DIR / "figures"

INPUT_VOTES = OUTPUT_DIR / "fan_vote_estimates_q1a.csv"
DATA_FILE = DATA_DIR / "2026_MCM_Problem_C_Data.csv"

OUTPUT_WEEKLY = OUTPUT_DIR / "fair_rule_weekly_q4.csv"
OUTPUT_SEASON_SUM = OUTPUT_DIR / "fair_rule_season_q4.csv"
OUTPUT_COMPARE = OUTPUT_DIR / "fair_rule_weekly_compare_q4.csv"
OUTPUT_METRICS = OUTPUT_DIR / "fair_rule_metrics_q4.csv"
OUTPUT_CONTROVERSY = OUTPUT_DIR / "fair_rule_controversy_q4.csv"
OUTPUT_PLACEMENT = OUTPUT_DIR / "fair_rule_placement_q4.csv"
OUTPUT_SENSITIVITY = OUTPUT_DIR / "fair_rule_sensitivity_q4.csv"
OUTPUT_RECOMMENDATION = OUTPUT_DIR / "fair_rule_recommendation_q4.md"
PLOT_FILE = FIG_DIR / "fair_rule_components_q4.png"
PLOT_CONTROVERSY = FIG_DIR / "fair_rule_controversy_q4.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Helper functions
# ============================

def rank_based_bottom_k(df_week: pd.DataFrame, k: int):
    judge_rank = df_week["judge_total"].rank(ascending=False, method="average")
    fan_rank = df_week["fan_vote_share"].rank(ascending=False, method="average")
    combined = judge_rank + fan_rank
    return combined.nlargest(k).index.tolist(), combined


def percent_based_bottom_k(df_week: pd.DataFrame, k: int):
    judge_percent = df_week["judge_total"] / df_week["judge_total"].sum()
    fan_percent = df_week["fan_vote_share"]
    combined = judge_percent + fan_percent
    return combined.nsmallest(k).index.tolist(), combined


def fair_rule_bottom_k(df_week: pd.DataFrame, k: int, alpha: float, beta: float):
    judge_sum = df_week["judge_total"].sum()
    judge_share = df_week["judge_total"] / judge_sum
    fan_share = df_week["fan_vote_share"]
    fair_score = alpha * judge_share + beta * fan_share
    return fair_score.nsmallest(k).index.tolist(), fair_score, judge_share, fan_share


def consistency_metrics(actual_set: set, predicted_set: set):
    match = int(actual_set == predicted_set)
    inter = len(actual_set & predicted_set)
    union = len(actual_set | predicted_set)
    overlap = float(inter / union) if union > 0 else np.nan
    recall = float(inter / len(actual_set)) if len(actual_set) > 0 else np.nan
    precision = float(inter / len(predicted_set)) if len(predicted_set) > 0 else np.nan
    deviation = len(actual_set - predicted_set)
    return match, deviation, overlap, recall, precision


def compute_rule_placement(season_df: pd.DataFrame, rule_col: str, score_col: str):
    season_df = season_df.copy().sort_values("week")
    contestants = season_df["celebrity_name"].unique().tolist()
    remaining = set(contestants)
    placements = {}

    for week, week_df in season_df.groupby("week"):
        week_df = week_df[week_df["celebrity_name"].isin(remaining)].copy()
        eliminated = week_df[week_df[rule_col] == 1].copy()
        if eliminated.empty:
            continue
        eliminated = eliminated.sort_values(score_col, ascending=True)
        for _, row in eliminated.iterrows():
            if row["celebrity_name"] in remaining:
                placements[row["celebrity_name"]] = len(remaining)
                remaining.remove(row["celebrity_name"])

    for name in remaining:
        placements[name] = 1

    return placements


def build_recommendation(metrics_df: pd.DataFrame) -> str:
    fair_row = metrics_df[metrics_df["rule"] == "fair"].iloc[0]
    rank_row = metrics_df[metrics_df["rule"] == "rank"].iloc[0]
    pct_row = metrics_df[metrics_df["rule"] == "percent"].iloc[0]

    lines = [
        "# 公平机制说明与建议（问题四）",
        "",
        "## 机制定义",
        f"- 合成得分：S = {ALPHA:.2f}·JudgeShare + {BETA:.2f}·FanShare",
        "- 公平性含义：在保持规则一致性的前提下，降低争议冲突并平衡评委与粉丝影响力。",
        "",
        "## 优越性量化对比（与两类原始机制）",
        f"- 一致性（bottom_k_match均值）：公平={fair_row['bottom_k_match']:.3f}，排名={rank_row['bottom_k_match']:.3f}，百分比={pct_row['bottom_k_match']:.3f}",
        f"- 重合度（overlap均值）：公平={fair_row['overlap']:.3f}，排名={rank_row['overlap']:.3f}，百分比={pct_row['overlap']:.3f}",
        f"- 冲突周次粉丝一致率：公平={fair_row['fan_align_conflict']:.3f}，排名={rank_row['fan_align_conflict']:.3f}，百分比={pct_row['fan_align_conflict']:.3f}",
        f"- 冲突周次评委一致率：公平={fair_row['judge_align_conflict']:.3f}，排名={rank_row['judge_align_conflict']:.3f}，百分比={pct_row['judge_align_conflict']:.3f}",
        "",
        "## 对制作方的积极意义",
        "- 兼顾专业性与热度：评委评分与粉丝投票均被保留影响力，降低极端争议。",
        "- 规则透明、易解释：加权平均机制直观，降低规则解释成本。",
        "- 可调参数：权重可按赛季策略微调，适配不同运营目标。",
        "",
        "## 实施建议",
        "- 建议保留等权作为默认基线；必要时可用敏感性结果选择更稳健权重。",
        "- 定期监控冲突周次比例与一致性指标，作为规则优化触发条件。",
    ]
    return "\n".join(lines)


# ============================
# Main process
# ============================

def main():
    # Load estimated fan votes (Q1a output)
    df = pd.read_csv(INPUT_VOTES)
    raw = pd.read_csv(DATA_FILE)

    required_cols = {"season", "week", "celebrity_name", "judge_total", "fan_vote_share"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    weekly_rows = []
    compare_rows = []

    # Process each season-week
    for (season, week), df_week in df.groupby(["season", "week"]):
        df_week = df_week.copy()

        actual_elim = df_week[df_week.get("eliminated_end_of_week", 0) == 1]
        k = int(actual_elim.shape[0]) if actual_elim.shape[0] > 0 else 0

        # Compute fair score
        if k > 0:
            fair_bottom, fair_score, judge_share, fan_share = fair_rule_bottom_k(df_week, k, ALPHA, BETA)
            rank_bottom, rank_score = rank_based_bottom_k(df_week, k)
            pct_bottom, pct_score = percent_based_bottom_k(df_week, k)
        else:
            judge_sum = df_week["judge_total"].sum()
            judge_share = df_week["judge_total"] / judge_sum
            fan_share = df_week["fan_vote_share"]
            fair_score = ALPHA * judge_share + BETA * fan_share
            rank_score = df_week["judge_total"].rank(ascending=False, method="average") + df_week["fan_vote_share"].rank(ascending=False, method="average")
            pct_score = (df_week["judge_total"] / df_week["judge_total"].sum()) + df_week["fan_vote_share"]
            fair_bottom, rank_bottom, pct_bottom = [], [], []

        df_week = df_week.copy()
        df_week["judge_share"] = judge_share
        df_week["fan_share"] = fan_share
        df_week["fair_score"] = fair_score
        df_week["rank_score"] = rank_score
        df_week["percent_score"] = pct_score
        df_week["eliminated_fair_rule"] = df_week.index.isin(fair_bottom).astype(int)
        df_week["eliminated_rank_rule"] = df_week.index.isin(rank_bottom).astype(int)
        df_week["eliminated_percent_rule"] = df_week.index.isin(pct_bottom).astype(int)

        for _, row in df_week.iterrows():
            weekly_rows.append({
                "season": season,
                "week": week,
                "celebrity_name": row["celebrity_name"],
                "judge_total": row["judge_total"],
                "judge_share": row["judge_share"],
                "fan_share": row["fan_share"],
                "fair_score": row["fair_score"],
                "rank_score": row["rank_score"],
                "percent_score": row["percent_score"],
                "eliminated_fair_rule": row["eliminated_fair_rule"],
                "eliminated_rank_rule": row["eliminated_rank_rule"],
                "eliminated_percent_rule": row["eliminated_percent_rule"],
            })

        if k > 0:
            actual_set = set(actual_elim.index.tolist())
            fair_set = set(fair_bottom)
            rank_set = set(rank_bottom)
            pct_set = set(pct_bottom)

            match_f, dev_f, ov_f, rec_f, pre_f = consistency_metrics(actual_set, fair_set)
            match_r, dev_r, ov_r, rec_r, pre_r = consistency_metrics(actual_set, rank_set)
            match_p, dev_p, ov_p, rec_p, pre_p = consistency_metrics(actual_set, pct_set)

            # Conflict metrics for balance
            judge_worst = df_week.loc[df_week["judge_total"].idxmin(), "celebrity_name"]
            fan_worst = df_week.loc[df_week["fan_vote_share"].idxmin(), "celebrity_name"]
            conflict = int(judge_worst != fan_worst)
            compare_rows.append({
                "season": season,
                "week": week,
                "conflict": conflict,
                "fair_bottom_k_match": match_f,
                "rank_bottom_k_match": match_r,
                "percent_bottom_k_match": match_p,
                "fair_overlap": ov_f,
                "rank_overlap": ov_r,
                "percent_overlap": ov_p,
                "fair_recall": rec_f,
                "rank_recall": rec_r,
                "percent_recall": rec_p,
                "fair_precision": pre_f,
                "rank_precision": pre_r,
                "percent_precision": pre_p,
                "fair_fan_aligned": int(fan_worst in fair_set),
                "rank_fan_aligned": int(fan_worst in rank_set),
                "percent_fan_aligned": int(fan_worst in pct_set),
                "fair_judge_aligned": int(judge_worst in fair_set),
                "rank_judge_aligned": int(judge_worst in rank_set),
                "percent_judge_aligned": int(judge_worst in pct_set),
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

    compare_df = pd.DataFrame(compare_rows)

    metrics_rows = []
    for rule in ["fair", "rank", "percent"]:
        match_col = f"{rule}_bottom_k_match"
        overlap_col = f"{rule}_overlap"
        recall_col = f"{rule}_recall"
        precision_col = f"{rule}_precision"
        fan_align_col = f"{rule}_fan_aligned"
        judge_align_col = f"{rule}_judge_aligned"

        conflict_df = compare_df[compare_df["conflict"] == 1]
        metrics_rows.append({
            "rule": rule,
            "weeks": int(compare_df.shape[0]),
            "bottom_k_match": float(compare_df[match_col].mean()) if not compare_df.empty else np.nan,
            "overlap": float(compare_df[overlap_col].mean()) if not compare_df.empty else np.nan,
            "recall": float(compare_df[recall_col].mean()) if not compare_df.empty else np.nan,
            "precision": float(compare_df[precision_col].mean()) if not compare_df.empty else np.nan,
            "fan_align_conflict": float(conflict_df[fan_align_col].mean()) if not conflict_df.empty else np.nan,
            "judge_align_conflict": float(conflict_df[judge_align_col].mean()) if not conflict_df.empty else np.nan,
            "balance_gap": float(abs(conflict_df[fan_align_col].mean() - conflict_df[judge_align_col].mean())) if not conflict_df.empty else np.nan,
        })

    metrics_df = pd.DataFrame(metrics_rows)

    # Controversy analysis
    controversy_cases = [
        {"season": 2, "celebrity_name": "Jerry Rice"},
        {"season": 4, "celebrity_name": "Billy Ray Cyrus"},
        {"season": 11, "celebrity_name": "Bristol Palin"},
        {"season": 27, "celebrity_name": "Bobby Bones"},
    ]

    controversy_rows = []

    for case in controversy_cases:
        season = case["season"]
        name = case["celebrity_name"]
        season_weekly = weekly_df[weekly_df["season"] == season]
        if season_weekly.empty:
            continue

        for rule, elim_col, score_col in [
            ("fair", "eliminated_fair_rule", "fair_score"),
            ("rank", "eliminated_rank_rule", "rank_score"),
            ("percent", "eliminated_percent_rule", "percent_score"),
        ]:
            sub = season_weekly[season_weekly["celebrity_name"].str.contains(name, case=False, na=False)]
            elim_weeks = sub[sub[elim_col] == 1]["week"].tolist()
            elim_week = min(elim_weeks) if elim_weeks else None

            actual_place = None
            placement_rows = raw[(raw["season"] == season) & (raw["celebrity_name"].str.contains(name, case=False, na=False))]
            if not placement_rows.empty:
                actual_place = int(placement_rows["placement"].iloc[0]) if "placement" in placement_rows.columns else None

            controversy_rows.append({
                "season": season,
                "celebrity_name": name,
                "rule": rule,
                "pred_elim_week": elim_week,
                "actual_placement": actual_place,
            })

    controversy_df = pd.DataFrame(controversy_rows)

    # Placement analysis by rule
    placement_rows = []
    for season, season_df in weekly_df.groupby("season"):
        for rule, elim_col, score_col in [
            ("fair", "eliminated_fair_rule", "fair_score"),
            ("rank", "eliminated_rank_rule", "rank_score"),
            ("percent", "eliminated_percent_rule", "percent_score"),
        ]:
            placements = compute_rule_placement(season_df, elim_col, score_col)
            for name, place in placements.items():
                placement_rows.append({
                    "season": season,
                    "celebrity_name": name,
                    "rule": rule,
                    "predicted_placement": place,
                })

    placement_df = pd.DataFrame(placement_rows)
    if "placement" in raw.columns:
        actual_place_df = raw[["season", "celebrity_name", "placement"]].drop_duplicates()
        placement_df = placement_df.merge(actual_place_df, on=["season", "celebrity_name"], how="left")

    # Sensitivity analysis
    sensitivity_rows = []
    for a in SENSITIVITY_WEIGHTS:
        b = 1.0 - a
        temp_rows = []
        for (season, week), df_week in df.groupby(["season", "week"]):
            actual_elim = df_week[df_week.get("eliminated_end_of_week", 0) == 1]
            k = int(actual_elim.shape[0]) if actual_elim.shape[0] > 0 else 0
            if k == 0:
                continue
            fair_bottom, fair_score, _, _ = fair_rule_bottom_k(df_week, k, a, b)
            actual_set = set(actual_elim.index.tolist())
            fair_set = set(fair_bottom)
            match, _, overlap, recall, precision = consistency_metrics(actual_set, fair_set)
            temp_rows.append({
                "bottom_k_match": match,
                "overlap": overlap,
                "recall": recall,
                "precision": precision,
            })
        temp_df = pd.DataFrame(temp_rows)
        sensitivity_rows.append({
            "alpha": a,
            "beta": b,
            "bottom_k_match": float(temp_df["bottom_k_match"].mean()) if not temp_df.empty else np.nan,
            "overlap": float(temp_df["overlap"].mean()) if not temp_df.empty else np.nan,
            "recall": float(temp_df["recall"].mean()) if not temp_df.empty else np.nan,
            "precision": float(temp_df["precision"].mean()) if not temp_df.empty else np.nan,
        })

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    # Save outputs
    weekly_df.to_csv(OUTPUT_WEEKLY, index=False, encoding="utf-8-sig")
    season_sum.to_csv(OUTPUT_SEASON_SUM, index=False, encoding="utf-8-sig")
    compare_df.to_csv(OUTPUT_COMPARE, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUTPUT_METRICS, index=False, encoding="utf-8-sig")
    controversy_df.to_csv(OUTPUT_CONTROVERSY, index=False, encoding="utf-8-sig")
    placement_df.to_csv(OUTPUT_PLACEMENT, index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(OUTPUT_SENSITIVITY, index=False, encoding="utf-8-sig")

    OUTPUT_RECOMMENDATION.write_text(build_recommendation(metrics_df), encoding="utf-8")

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
    print(f"对比指标已保存：{OUTPUT_COMPARE}")
    print(f"总量指标已保存：{OUTPUT_METRICS}")
    print(f"争议案例分析已保存：{OUTPUT_CONTROVERSY}")
    print(f"名次对比已保存：{OUTPUT_PLACEMENT}")
    print(f"敏感性分析已保存：{OUTPUT_SENSITIVITY}")
    print(f"建议报告已保存：{OUTPUT_RECOMMENDATION}")
    print(f"图像已保存：{PLOT_FILE}")


if __name__ == "__main__":
    main()
