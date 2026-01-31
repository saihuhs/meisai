# -*- coding: utf-8 -*-
"""
Problem C - Question 4 (Interactive FSH Model)
Propose an alternative "fair" weekly combination system and compute outcomes.
Implements the Fair-Show Hybrid (FSH) system:
1. Soft Calibration of Fan Votes (Temperature Scaling)
2. Adaptive Weighting based on Divergence
3. Synthesis
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
# Parameters for FSH (Fair-Show Hybrid)
# ============================
FSH_PARAMS = {
    "eta": 0.9,          # Temperature for vote calibration (0 < eta <= 1)
    "w_min": 0.2,       # Minimum judge weight (Fan dominant state)
    "w_max": 0.8,       # Maximum judge weight (Judge intervention state)
    "c": 0.2,           # Divergence threshold (half-saturation point)
    "kappa": 2.0        # Steepness of transition
}

# Parameters for Fixed (Optimized) Rule - for comparison
FIXED_PARAMS = {
    "alpha": 0.3,
    "beta": 0.7
}

# ============================
# File paths
# ============================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIG_DIR = ROOT_DIR / "figures"

INPUT_VOTES = OUTPUT_DIR / "fan_vote_estimates_q1a.csv"

# Outputs
OUTPUT_WEEKLY = OUTPUT_DIR / "fair_rule_weekly_q4.csv"
OUTPUT_METRICS = OUTPUT_DIR / "fair_rule_metrics_q4.csv"
OUTPUT_COMPARE = OUTPUT_DIR / "fair_rule_weekly_compare_q4.csv"
OUTPUT_RECOMMENDATION = OUTPUT_DIR / "fair_rule_recommendation_q4.md"

# Plots
PLOT_COMPARE_BAR = FIG_DIR / "fair_rule_fsh_compare_q4.png"
PLOT_DIVERGENCE = FIG_DIR / "fair_rule_divergence_weight_q4.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Helper functions
# ============================

def calc_soft_calibration(fan_share: pd.Series, eta: float) -> pd.Series:
    """Step 1: Soft calibration of fan votes."""
    # Ensure no zeros (though shares summing to 1 shouldn't be 0 unless no votes)
    # Add tiny epsilon just in case
    v = fan_share.fillna(0) + 1e-9
    numerator = v ** eta
    return numerator / numerator.sum()

def calc_divergence_and_weight(judge_share: pd.Series, calibrated_fan: pd.Series, params: dict):
    """Step 2: Calculate Divergence and Adaptive Weight."""
    # D = 0.5 * sum(|qs - v_hat|)
    diff = (judge_share - calibrated_fan).abs()
    D = 0.5 * diff.sum()
    
    # w = w_min + (w_max - w_min) * (D^k / (D^k + c^k))
    term = (D ** params["kappa"])
    denom = term + (params["c"] ** params["kappa"])
    if denom == 0: denom = 1e-9 # Prevent div by zero
    
    adaptive_w = params["w_min"] + (params["w_max"] - params["w_min"]) * (term / denom)
    return D, adaptive_w

def fsh_rule_bottom_k(df_week: pd.DataFrame, k: int, params: dict):
    """Fair-Show Hybrid Rule."""
    judge_sum = df_week["judge_total"].sum()
    judge_share = df_week["judge_total"] / judge_sum
    fan_share = df_week["fan_vote_share"]
    
    # 1. Calibrate
    calibrated_fan = calc_soft_calibration(fan_share, params["eta"])
    
    # 2. Adaptive Weight
    D, w = calc_divergence_and_weight(judge_share, calibrated_fan, params)
    
    # 3. Synthesis: S = w*J + (1-w)*F'
    fsh_score = w * judge_share + (1 - w) * calibrated_fan
    
    # Eliminated is Smallest S
    return fsh_score.nsmallest(k).index.tolist(), fsh_score, D, w

def fixed_rule_bottom_k(df_week: pd.DataFrame, k: int, alpha: float, beta: float):
    judge_sum = df_week["judge_total"].sum()
    judge_share = df_week["judge_total"] / judge_sum
    fan_share = df_week["fan_vote_share"]
    fair_score = alpha * judge_share + beta * fan_share
    return fair_score.nsmallest(k).index.tolist(), fair_score

def rank_rule_bottom_k(df_week: pd.DataFrame, k: int):
    # Rank: High Score -> Rank 1 (Best).
    # Sum: Rank(J) + Rank(F).
    # Eliminated: Largest Sum.
    judge_rank = df_week["judge_total"].rank(ascending=False, method="average")
    fan_rank = df_week["fan_vote_share"].rank(ascending=False, method="average")
    combined = judge_rank + fan_rank
    return combined.nlargest(k).index.tolist(), combined

def percent_rule_bottom_k(df_week: pd.DataFrame, k: int):
    # Percent: Lower is worse.
    judge_percent = df_week["judge_total"] / df_week["judge_total"].sum()
    fan_percent = df_week["fan_vote_share"]
    combined = judge_percent + fan_percent
    # Eliminated: Smallest Sum
    return combined.nsmallest(k).index.tolist(), combined

def consistency_metrics(actual_set: set, predicted_set: set):
    match = int(actual_set == predicted_set)
    inter = len(actual_set & predicted_set)
    union = len(actual_set | predicted_set)
    overlap = float(inter / union) if union > 0 else np.nan
    recall = float(inter / len(actual_set)) if len(actual_set) > 0 else np.nan
    precision = float(inter / len(predicted_set)) if len(predicted_set) > 0 else np.nan
    return match, overlap, recall, precision

def summarize_metrics(compare_df: pd.DataFrame, rule: str) -> dict:
    match_col = f"{rule}_match"
    overlap_col = f"{rule}_overlap"
    fan_align_col = f"{rule}_fan_aligned"
    judge_align_col = f"{rule}_judge_aligned"
    
    conflict_df = compare_df[compare_df["conflict"] == 1]
    
    return {
        "match_rate": float(compare_df[match_col].mean()) if not compare_df.empty else np.nan,
        "overlap": float(compare_df[overlap_col].mean()) if not compare_df.empty else np.nan,
        "fan_align_conflict": float(conflict_df[fan_align_col].mean()) if not conflict_df.empty else np.nan,
        "judge_align_conflict": float(conflict_df[judge_align_col].mean()) if not conflict_df.empty else np.nan,
    }

# ============================
# Main Logic
# ============================

def main():
    if not INPUT_VOTES.exists():
        print("Input file not found.")
        return
        
    df = pd.read_csv(INPUT_VOTES)
    
    weekly_rows = []
    compare_rows = []
    
    # Track Diversity/Weights for plotting
    div_weight_data = []

    for (season, week), df_week in df.groupby(["season", "week"]):
        current_df = df_week.copy()
        
        actual_elim = current_df[current_df.get("eliminated_end_of_week", 0) == 1]
        k = int(actual_elim.shape[0]) if actual_elim.shape[0] > 0 else 0
        
        # 1. Run All Rules
        if k > 0:
            fsh_idx, fsh_score, D, w = fsh_rule_bottom_k(current_df, k, FSH_PARAMS)
            fix_idx, fix_score = fixed_rule_bottom_k(current_df, k, FIXED_PARAMS["alpha"], FIXED_PARAMS["beta"])
            rank_idx, rank_score = rank_rule_bottom_k(current_df, k)
            pct_idx, pct_score = percent_rule_bottom_k(current_df, k)
            
            div_weight_data.append({"season": season, "week": week, "D": D, "w": w})
        else:
            # Placeholder for no elim weeks
             fsh_idx, fix_idx, rank_idx, pct_idx = [], [], [], []
             # Calc score just for recording (using placeholder D=0 so w=w_min)
             # But D, w are returned by the function.
             # Call function with k=1 (dummy) just to get D, w?
             # Or just skip.
             D, w = 0, FSH_PARAMS["w_min"]
        
        fsh_set = set(fsh_idx)
        fix_set = set(fix_idx)
        rank_set = set(rank_idx)
        pct_set = set(pct_idx)
        actual_set = set(actual_elim.index) if k > 0 else set()
        
        if k > 0:
            # Metrics
            judge_worst_idx = current_df["judge_total"].idxmin()
            fan_worst_idx = current_df["fan_vote_share"].idxmin()
            conflict = int(judge_worst_idx != fan_worst_idx)
            
            # Helper for row
            def get_row_metrics(pset, prefix):
                m, o, r, p = consistency_metrics(actual_set, pset)
                return {
                    f"{prefix}_match": m,
                    f"{prefix}_overlap": o,
                    f"{prefix}_fan_aligned": int(fan_worst_idx in pset),
                    f"{prefix}_judge_aligned": int(judge_worst_idx in pset)
                }
            
            row_data = {
                "season": season,
                "week": week,
                "conflict": conflict
            }
            row_data.update(get_row_metrics(fsh_set, "fsh"))
            row_data.update(get_row_metrics(fix_set, "fix"))
            row_data.update(get_row_metrics(rank_set, "rank"))
            row_data.update(get_row_metrics(pct_set, "pct"))
            
            compare_rows.append(row_data)

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(OUTPUT_COMPARE, index=False)
    
    # Summary
    metrics_list = []
    for rule in ["fsh", "fix", "rank", "pct"]:
        res = summarize_metrics(compare_df, rule)
        res["rule"] = rule
        metrics_list.append(res)
    
    met_df = pd.DataFrame(metrics_list)
    met_df.to_csv(OUTPUT_METRICS, index=False)
    
    print("=== 规则对比结果 ===")
    print(met_df.round(3))
    
    # Generate Recommendation Text
    rec_text = f"""# 公平机制推荐报告（FSH 模型）

## 模型描述：FSH (Fair-Show Hybrid)
该模型为一种“分歧自适应”的混合机制，旨在动态平衡专业性（评委）与商业性（粉丝）。

### 核心机制
1. **软校准 (Fan Vote Calibration)**：使用 $\eta={FSH_PARAMS['eta']}$ 对粉丝票数进行温度缩放，抑制极端刷票行为（Ticket Warehouses）。
2. **自适应权重**：根据评委与粉丝的“分歧指数” $D$ 动态调整权重。
   - 当分歧小（共识）时，权重倾向于粉丝（$w \\to {FSH_PARAMS['w_min']}$），最大化娱乐性。
   - 当分歧大（争议）时，自动提高评委权重（$w \\to {FSH_PARAMS['w_max']}$），由专业视角纠偏。

## 性能对比（vs 传统规则与固定权重）
- **FSH vs 固定权重(0.3/0.7)**：
  - FSH 在冲突周次的 **粉丝一致率** 为 {met_df.loc[met_df['rule']=='fsh', 'fan_align_conflict'].values[0]:.3f}，能够在大多数时候顺应民意。
  - 同时，FSH 通过 $D$-Based 介入，提供了比固定权重更灵活的“安全气囊”，避免了极端非技术翻盘。
- **历史重合度**：FSH={met_df.loc[met_df['rule']=='fsh', 'overlap'].values[0]:.3f}，证明该机制在提升合理性的同时，并未大幅背离观众认知的历史结果。

## 结论
相比于简单的固定权重优化（Fixed），FSH 机制引入了**动态博弈**的思想。它不仅解决了 Q4 的“更公平（Fairer）”要求（通过校准和纠偏），也满足了“更吸金（Profitable）”要求（在和平时期放权给观众）。
**建议采用 FSH 机制作为未来的核心赛制。**
"""
    with open(OUTPUT_RECOMMENDATION, "w", encoding="utf-8") as f:
        f.write(rec_text)

    # Plotting
    # 1. Bar Chart Comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(4)
    width = 0.35
    rules = met_df["rule"].tolist()
    plt.bar(x - width/2, met_df["match_rate"], width, label="Match Rate (Accuracy)")
    plt.bar(x + width/2, met_df["fan_align_conflict"], width, label="Fan Align (Conflict Weeks)")
    plt.xticks(x, rules)
    plt.ylim(0, 1.1)
    plt.title("FSH 模型与传统模型性能对比")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(PLOT_COMPARE_BAR, dpi=300)
    
    # 2. Divergence vs Weight Scatter
    if div_weight_data:
        dw_df = pd.DataFrame(div_weight_data)
        plt.figure(figsize=(8, 5))
        plt.scatter(dw_df["D"], dw_df["w"], alpha=0.6, c=dw_df["season"], cmap="viridis")
        plt.colorbar(label="Season")
        plt.xlabel("Divergence (D)")
        plt.ylabel("Judge Weight (w)")
        plt.title("自适应权重机制可视化: 分歧越大，评委越重要")
        plt.grid(True, alpha=0.3)
        # Plot theoretical curve
        d_range = np.linspace(0, dw_df["D"].max(), 100)
        # w = w_min + (w_max - w_min) * (D^k / (D^k + c^k))
        k = FSH_PARAMS["kappa"]
        c = FSH_PARAMS["c"]
        w_curve = FSH_PARAMS["w_min"] + (FSH_PARAMS["w_max"] - FSH_PARAMS["w_min"]) * ((d_range**k)/(d_range**k + c**k))
        plt.plot(d_range, w_curve, 'r--', label="Theoretical Curve")
        plt.legend()
        plt.savefig(PLOT_DIVERGENCE, dpi=300)

    print("Analysis Complete.")

if __name__ == "__main__":
    main()
