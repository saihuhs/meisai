# -*- coding: utf-8 -*-
"""
Problem C - Question 1a
Estimate fan votes consistent with elimination results under two voting schemes
(rank-based for seasons 1,2,28-34; percent-based for seasons 3-27).

This script follows the model structure defined in the paper:
- Define weekly judge totals J_{i,s,t}
- Define fan votes F_{i,s,t} as latent variables
- Impose elimination-consistent constraints according to the scheme
- Select a representative feasible fan-vote allocation
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
# Preference weights (reasonable smoothness & correlation)
# ============================
SMOOTHNESS_WEIGHT = 1.0
CORRELATION_WEIGHT = 1.0

# ============================
# Within-week standardization (optional)
# ============================
STANDARDIZE_WITHIN_WEEK = False
STANDARDIZE_METHOD = "zscore"  # "zscore" or "minmax"

# ============================
# File paths
# ============================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "2026_MCM_Problem_C_Data.csv"
OUTPUT_VOTES = BASE_DIR / "fan_vote_estimates_q1a.csv"
OUTPUT_STATUS = BASE_DIR / "fan_vote_week_status_q1a.csv"
PLOT_FILE = BASE_DIR / "fan_vote_trends_q1a.png"
OUTPUT_CONSISTENCY = BASE_DIR / "fan_vote_consistency_q1a.csv"
OUTPUT_UNCERTAINTY_RANK = BASE_DIR / "fan_vote_uncertainty_rank_q1a.csv"
OUTPUT_UNCERTAINTY_PERCENT = BASE_DIR / "fan_vote_uncertainty_percent_q1a.csv"
HEATMAP_FILE = BASE_DIR / "fan_vote_uncertainty_heatmap_q1a.png"
OUTPUT_CONSISTENCY_EXT = BASE_DIR / "fan_vote_consistency_extended_q1a.csv"
OUTPUT_SENSITIVITY = BASE_DIR / "fan_vote_sensitivity_weights_q1a.csv"
OUTPUT_JUDGE_UNCERTAINTY = BASE_DIR / "fan_vote_judge_variability_uncertainty_q1a.csv"
PLOT_FEASIBLE_SPACE = BASE_DIR / "fan_vote_feasible_space_q1a.png"
PLOT_PRESSURE = BASE_DIR / "fan_vote_elimination_pressure_q1a.png"

# ============================
# Absolute vote assumption
# ============================
TOTAL_VOTES_PER_WEEK = 10_000_000

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


def standardize_week_scores(series: pd.Series, method: str = "zscore"):
    """Standardize scores within week (optional)."""
    s = series.astype(float)
    if method == "minmax":
        min_v = s.min()
        max_v = s.max()
        if max_v == min_v:
            return s * 0.0
        return (s - min_v) / (max_v - min_v)

    # default: z-score
    mean_v = s.mean()
    std_v = s.std()
    if std_v == 0:
        return s * 0.0
    return (s - mean_v) / std_v


def compute_week_scores(df: pd.DataFrame, weeks):
    """
    Compute weekly judge average scores using missing indicators.
    - I_{i,s,t,j}=1 if score exists else 0
    - m_{i,s,t} = sum I
    - S_{i,s,t} = sum I * score
    - avg_{i,s,t} = S / m
    """
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


def last_active_week(row, weeks):
    """Determine last active week as the latest week with valid judge average."""
    active_weeks = [w for w in weeks if pd.notna(row.get(f"week{w}_avg", np.nan))]
    return max(active_weeks) if active_weeks else 0


def season_rule(season_number: int) -> str:
    """
    Voting scheme by season:
    - rank-based: seasons 1,2 and 28-34
    - percent-based: seasons 3-27
    """
    if season_number in [1, 2] or season_number >= 28:
        return "rank"
    return "percent"


def assign_fan_ranks(rank_judge, eliminated_idx):
    """
    Construct a feasible fan-rank assignment under rank-based rule.
    Constraint: rJ_e + rF_e >= rJ_i + rF_i for all i.

    Steps:
    1) Try each possible rF_e in {1..n}
    2) Compute upper bounds U_i = floor(rF_e + (rJ_e - rJ_i))
    3) Feasibility via earliest-deadline-first assignment
    4) Choose a representative feasible rF_e (median of feasible set)
    """
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

        # Earliest deadline first assignment
        available_ranks = [r for r in range(1, n + 1) if r != rF_e]
        available_ranks.sort()
        assignment = {eliminated_idx: rF_e}

        # Sort by upper bound (deadline)
        sorted_items = sorted(upper_bounds.items(), key=lambda x: x[1])
        for i, U_i in sorted_items:
            # Assign the smallest available rank that does not violate U_i
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


def assign_fan_ranks_multi(rank_judge, eliminated_indices):
    """
    Construct feasible fan-rank assignments under rank-based rule
    for multiple eliminations. Constraint for all e in E and i in N:
    rJ_e + rF_e >= rJ_i + rF_i.
    Approximation: assign eliminated to worst-k ranks and test feasibility.
    """
    n = len(rank_judge)
    k = len(eliminated_indices)
    contestants = list(rank_judge.index)
    rJ = rank_judge.to_dict()

    worst_ranks = list(range(n - k + 1, n + 1))
    feasible_assignments = []

    from itertools import permutations
    for perm in permutations(worst_ranks):
        assignment = {e: r for e, r in zip(eliminated_indices, perm)}

        # Upper bounds for non-eliminated from all eliminated constraints
        upper_bounds = {}
        feasible = True
        for i in contestants:
            if i in eliminated_indices:
                continue
            bounds = [assignment[e] + (rJ[e] - rJ[i]) for e in eliminated_indices]
            U_i = int(np.floor(min(bounds)))
            if U_i < 1:
                feasible = False
                break
            upper_bounds[i] = U_i
        if not feasible:
            continue

        available_ranks = [r for r in range(1, n + 1) if r not in assignment.values()]
        available_ranks.sort()

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


def preference_objective(fan_share, judge_scores, prev_share,
                         smooth_w: float = SMOOTHNESS_WEIGHT,
                         corr_w: float = CORRELATION_WEIGHT):
    """
    Preference objective to select a representative solution:
    - smoothness across weeks (penalize large jumps)
    - positive association with judge scores (allow deviations)
    Minimize: smoothness - correlation
    """
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

    return smooth_w * smooth - corr_w * corr


def spearman_corr(x, y):
    return pd.Series(x).rank().corr(pd.Series(y).rank())


def kendall_tau(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else 0.0


def entropy_from_counts(counts):
    total = np.sum(counts)
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def estimate_votes_rank_based(rank_judge, eliminated_idx, judge_scores, prev_share,
                              smooth_w: float = SMOOTHNESS_WEIGHT,
                              corr_w: float = CORRELATION_WEIGHT):
    """
    Estimate fan votes for rank-based scheme:
    - obtain fan ranks consistent with elimination
    - map ranks to vote shares via a monotone transformation
    """
    assignments = assign_fan_ranks(rank_judge, eliminated_idx)
    if not assignments:
        return None

    best = None
    best_obj = np.inf
    n = len(rank_judge)

    for assignment in assignments:
        fan_rank = pd.Series(assignment)
        fan_score = (n + 1 - fan_rank).astype(float)
        fan_share = fan_score / fan_score.sum()
        obj = preference_objective(fan_share, judge_scores, prev_share, smooth_w, corr_w)
        if obj < best_obj:
            best_obj = obj
            best = (fan_rank, fan_share)

    return best, assignments


def estimate_votes_percent_based(judge_scores, eliminated_idx, prev_share, grid_step=0.001,
                                 smooth_w: float = SMOOTHNESS_WEIGHT,
                                 corr_w: float = CORRELATION_WEIGHT):
    """
    Estimate fan vote shares under percent-based scheme:
    Constraints: pJ_e + pF_e <= pJ_i + pF_i for all i.
    We build a feasible pF by selecting a representative pF_e via grid search
    and distributing remaining mass evenly.
    """
    contestants = list(judge_scores.index)
    n = len(contestants)

    # Judge percent
    pJ = judge_scores / judge_scores.sum()
    pJ_e = pJ.loc[eliminated_idx]

    best_solution = None
    best_obj = np.inf
    feasible_pF_e = []

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

        # Distribute remaining mass evenly across non-eliminated contestants
        remaining = 1.0 - min_sum
        extra = remaining / (n - 1)

        pF = {eliminated_idx: pF_e}
        for i in contestants:
            if i == eliminated_idx:
                continue
            pF[i] = lower_bounds[i] + extra

        fan_share = pd.Series(pF)
        feasible_pF_e.append(pF_e)
        obj = preference_objective(fan_share, judge_scores, prev_share, smooth_w, corr_w)
        if obj < best_obj:
            best_obj = obj
            best_solution = pF

    if best_solution is None:
        return None

    fan_share = pd.Series(best_solution)
    return fan_share, feasible_pF_e


def estimate_votes_rank_based_multi(rank_judge, eliminated_indices, judge_scores, prev_share,
                                    smooth_w: float = SMOOTHNESS_WEIGHT,
                                    corr_w: float = CORRELATION_WEIGHT):
    assignments = assign_fan_ranks_multi(rank_judge, eliminated_indices)
    if not assignments:
        return None

    best = None
    best_obj = np.inf
    n = len(rank_judge)

    for assignment in assignments:
        fan_rank = pd.Series(assignment)
        fan_score = (n + 1 - fan_rank).astype(float)
        fan_share = fan_score / fan_score.sum()
        obj = preference_objective(fan_share, judge_scores, prev_share, smooth_w, corr_w)
        if obj < best_obj:
            best_obj = obj
            best = (fan_rank, fan_share, assignment)

    return best, assignments


def simplex_grid(k, step):
    """Generate grid points on simplex sum<=1 with given step (optimized for k<=2)."""
    if k <= 0:
        return []
    if k == 1:
        return [[v] for v in np.arange(0.0, 1.0 + 1e-9, step)]
    if k == 2:
        points = []
        for v in np.arange(0.0, 1.0 + 1e-9, step):
            remaining = 1.0 - v
            for w in np.arange(0.0, remaining + 1e-9, step):
                points.append([v, w])
        return points

    # Fallback recursion for k>2 (not recommended for performance)
    points = []

    def rec(remaining, depth, current):
        if depth == k - 1:
            for v in np.arange(0.0, remaining + 1e-9, step):
                points.append(current + [v])
            return
        for v in np.arange(0.0, remaining + 1e-9, step):
            rec(remaining - v, depth + 1, current + [v])

    rec(1.0, 0, [])
    return points


def estimate_votes_percent_based_multi(judge_scores, eliminated_indices, prev_share, grid_step=0.05,
                                       smooth_w: float = SMOOTHNESS_WEIGHT,
                                       corr_w: float = CORRELATION_WEIGHT,
                                       max_k: int = 2):
    contestants = list(judge_scores.index)
    n = len(contestants)
    pJ = judge_scores / judge_scores.sum()

    best_solution = None
    best_obj = np.inf
    solutions = []

    k = len(eliminated_indices)
    if k > max_k:
        return None, []
    grid_points = simplex_grid(k, grid_step)

    for point in grid_points:
        pF_e = {e: v for e, v in zip(eliminated_indices, point)}

        lower_bounds = {}
        feasible = True
        for i in contestants:
            if i in eliminated_indices:
                continue
            bounds = [pJ[e] - pJ.loc[i] + pF_e[e] for e in eliminated_indices]
            lb_i = max(0.0, max(bounds))
            lower_bounds[i] = lb_i

        min_sum = sum(pF_e.values()) + sum(lower_bounds.values())
        if min_sum > 1.0:
            feasible = False
        if not feasible:
            continue

        remaining = 1.0 - min_sum
        non_elims = [i for i in contestants if i not in eliminated_indices]
        extra = remaining / len(non_elims)

        pF = {i: 0.0 for i in contestants}
        for e in eliminated_indices:
            pF[e] = pF_e[e]
        for i in non_elims:
            pF[i] = lower_bounds[i] + extra

        fan_share = pd.Series(pF)
        solutions.append(fan_share)
        obj = preference_objective(fan_share, judge_scores, prev_share, smooth_w, corr_w)
        if obj < best_obj:
            best_obj = obj
            best_solution = fan_share

    return best_solution, solutions


# ============================
# Main process
# ============================

def main():
    # Load data
    df = pd.read_csv(DATA_FILE)

    # Identify weeks and compute weekly judge totals
    weeks = identify_week_columns(df.columns)
    df = compute_week_scores(df, weeks)

    # Compute last active week for each contestant
    df["last_active_week"] = df.apply(lambda r: last_active_week(r, weeks), axis=1)

    results = []
    status_rows = []
    consistency_rows = []
    consistency_ext_rows = []
    uncertainty_rank_rows = []
    uncertainty_percent_rows = []

    # Track previous week fan shares by season for smoothness preference
    prev_share_by_season = {}

    for season, season_df in df.groupby("season"):
        season_df = season_df.copy()
        max_week = season_df["last_active_week"].max()
        scheme = season_rule(int(season))

        for t in range(1, max_week + 1):
            # Active contestants for week t
            active_mask = season_df["last_active_week"] >= t
            active_df = season_df.loc[active_mask].copy()

            if active_df.empty:
                continue

            # Identify eliminated contestants at end of week t
            elimination_candidates = season_df[(season_df["last_active_week"] == t) & (t < max_week)]
            eliminated_list = elimination_candidates.index.tolist()

            # Judge scores for week t (raw average)
            judge_scores_raw = active_df.set_index(active_df.index)[f"week{t}_avg"]
            judge_scores = judge_scores_raw.copy()

            # Optional within-week standardization
            if STANDARDIZE_WITHIN_WEEK:
                judge_scores = standardize_week_scores(judge_scores, method=STANDARDIZE_METHOD)

            prev_share = prev_share_by_season.get(season, None)

            # Rank-based scheme
            if scheme == "rank":
                # Higher judge score -> better rank (1 is best)
                rank_judge = judge_scores.rank(ascending=False, method="average")
                if len(eliminated_list) == 0:
                    # no elimination: use judge rank as proxy prior (合理性：在缺乏淘汰约束时，
                    # 以评委表现作为最小偏好假设，确保可行解唯一且可解释)
                    fan_rank = rank_judge.copy()
                    n = len(rank_judge)
                    fan_score = (n + 1 - fan_rank).astype(float)
                    fan_share = fan_score / fan_score.sum()
                    status = "processed (no elimination)"
                elif len(eliminated_list) == 1:
                    eliminated_idx = eliminated_list[0]
                    estimate = estimate_votes_rank_based(rank_judge, eliminated_idx, judge_scores, prev_share)
                    if estimate is None:
                        status_rows.append({
                            "season": season,
                            "week": t,
                            "scheme": scheme,
                            "status": "no feasible fan ranks"
                        })
                        continue
                    (fan_rank, fan_share), all_assignments = estimate
                    status = "processed"

                    # Uncertainty for rank scheme (single)
                    ranks_df = pd.DataFrame(all_assignments)
                    rank_std = ranks_df.std(axis=0)
                    rank_mean = ranks_df.mean(axis=0)
                    for idx in active_df.index:
                        if idx in rank_std.index:
                            uncertainty_rank_rows.append({
                                "season": season,
                                "week": t,
                                "celebrity_name": active_df.loc[idx, "celebrity_name"],
                                "fan_rank_std": float(rank_std.loc[idx]),
                                "fan_rank_mean": float(rank_mean.loc[idx]),
                                "fan_rank_selected": float(fan_rank.loc[idx]),
                                "feasible_count": int(len(all_assignments)),
                                "selected_minus_mean": float(fan_rank.loc[idx] - rank_mean.loc[idx])
                            })
                else:
                    estimate = estimate_votes_rank_based_multi(rank_judge, eliminated_list, judge_scores, prev_share)
                    if estimate is None:
                        status_rows.append({
                            "season": season,
                            "week": t,
                            "scheme": scheme,
                            "status": "no feasible fan ranks (multi)"
                        })
                        continue
                    (fan_rank, fan_share, assignment), all_assignments = estimate
                    status = "processed (multi elimination)"

                    # Uncertainty for rank scheme (multi or single)
                    ranks_df = pd.DataFrame(all_assignments)
                    rank_std = ranks_df.std(axis=0)
                    rank_mean = ranks_df.mean(axis=0)
                    for idx in active_df.index:
                        if idx in rank_std.index:
                            uncertainty_rank_rows.append({
                                "season": season,
                                "week": t,
                                "celebrity_name": active_df.loc[idx, "celebrity_name"],
                                "fan_rank_std": float(rank_std.loc[idx]),
                                "fan_rank_mean": float(rank_mean.loc[idx]),
                                "fan_rank_selected": float(fan_rank.loc[idx]),
                                "feasible_count": int(len(all_assignments)),
                                "selected_minus_mean": float(fan_rank.loc[idx] - rank_mean.loc[idx])
                            })
                for idx, row in active_df.iterrows():
                    results.append({
                        "season": season,
                        "week": t,
                        "celebrity_name": row["celebrity_name"],
                        "judge_total": judge_scores_raw.loc[idx],
                        "judge_rank": rank_judge.loc[idx],
                        "fan_rank": fan_rank.loc[idx],
                        "fan_vote_share": fan_share.loc[idx],
                        "fan_votes_absolute": fan_share.loc[idx] * TOTAL_VOTES_PER_WEEK,
                        "total_votes_assumed": TOTAL_VOTES_PER_WEEK,
                        "scheme": scheme,
                        "eliminated_end_of_week": int(idx in eliminated_list)
                    })

                prev_share_by_season[season] = fan_share

            # Percent-based scheme
            else:
                if STANDARDIZE_WITHIN_WEEK:
                    min_v = judge_scores.min()
                    if min_v <= 0:
                        judge_scores = judge_scores - min_v + 1e-6
                if len(eliminated_list) == 0:
                    # no elimination: use judge percent as a neutral prior
                    fan_share = judge_scores / judge_scores.sum()
                    status = "processed (no elimination)"
                elif len(eliminated_list) == 1:
                    eliminated_idx = eliminated_list[0]
                    fan_share, feasible_pF_e = estimate_votes_percent_based(judge_scores, eliminated_idx, prev_share)
                    if fan_share is None:
                        status_rows.append({
                            "season": season,
                            "week": t,
                            "scheme": scheme,
                            "status": "no feasible fan shares"
                        })
                        continue
                    status = "processed"

                    # Uncertainty for percent scheme (single)
                    if feasible_pF_e:
                        pF_e_min = float(np.min(feasible_pF_e))
                        pF_e_max = float(np.max(feasible_pF_e))
                        pF_e_sel = float(fan_share.loc[eliminated_idx])

                        # Local stability: perturb eliminated share by ±0.01 and measure change
                        delta = 0.01
                        pJ = judge_scores / judge_scores.sum()
                        contestants = list(judge_scores.index)

                        def build_share(pF_e_value):
                            lower_bounds = {}
                            for i in contestants:
                                if i == eliminated_idx:
                                    continue
                                lb_i = pJ.loc[eliminated_idx] - pJ.loc[i] + pF_e_value
                                lower_bounds[i] = max(0.0, lb_i)
                            min_sum = pF_e_value + sum(lower_bounds.values())
                            if min_sum > 1.0:
                                return None
                            remaining = 1.0 - min_sum
                            extra = remaining / (len(contestants) - 1)
                            pF = {eliminated_idx: pF_e_value}
                            for i in contestants:
                                if i == eliminated_idx:
                                    continue
                                pF[i] = lower_bounds[i] + extra
                            return pd.Series(pF)

                        share_minus = build_share(max(0.0, pF_e_sel - delta))
                        share_plus = build_share(min(1.0, pF_e_sel + delta))
                        stability = np.nan
                        if share_minus is not None and share_plus is not None:
                            stability = float((share_plus - share_minus).abs().mean())

                        for idx in active_df.index:
                            uncertainty_percent_rows.append({
                                "season": season,
                                "week": t,
                                "celebrity_name": active_df.loc[idx, "celebrity_name"],
                                "fan_share_min": pF_e_min if idx == eliminated_idx else np.nan,
                                "fan_share_max": pF_e_max if idx == eliminated_idx else np.nan,
                                "fan_share_width": (pF_e_max - pF_e_min) if idx == eliminated_idx else np.nan,
                                "local_stability": stability if idx == eliminated_idx else np.nan,
                            })
                else:
                    fan_share, solutions = estimate_votes_percent_based_multi(
                        judge_scores, eliminated_list, prev_share
                    )
                    if fan_share is None:
                        status_rows.append({
                            "season": season,
                            "week": t,
                            "scheme": scheme,
                            "status": "no feasible fan shares (multi or k>2)"
                        })
                        continue
                    status = "processed (multi elimination)"

                    # Uncertainty for percent scheme (multi)
                    if solutions:
                        sol_df = pd.concat(solutions, axis=1)
                        for idx in active_df.index:
                            if idx in sol_df.index:
                                vals = sol_df.loc[idx].values
                                uncertainty_percent_rows.append({
                                    "season": season,
                                    "week": t,
                                    "celebrity_name": active_df.loc[idx, "celebrity_name"],
                                    "fan_share_min": float(np.min(vals)),
                                    "fan_share_max": float(np.max(vals)),
                                    "fan_share_width": float(np.max(vals) - np.min(vals)),
                                })

                # Judge percent for reporting consistency
                pJ = judge_scores / judge_scores.sum()

                for idx, row in active_df.iterrows():
                    results.append({
                        "season": season,
                        "week": t,
                        "celebrity_name": row["celebrity_name"],
                        "judge_total": judge_scores_raw.loc[idx],
                        "judge_percent": pJ.loc[idx],
                        "fan_vote_share": fan_share.loc[idx],
                        "fan_votes_absolute": fan_share.loc[idx] * TOTAL_VOTES_PER_WEEK,
                        "total_votes_assumed": TOTAL_VOTES_PER_WEEK,
                        "scheme": scheme,
                        "eliminated_end_of_week": int(idx in eliminated_list)
                    })

                prev_share_by_season[season] = fan_share

            status_rows.append({
                "season": season,
                "week": t,
                "scheme": scheme,
                "status": status
            })

            # Consistency metrics
            if len(eliminated_list) > 0:
                if scheme == "rank":
                    combined = rank_judge + fan_rank
                    k = len(eliminated_list)
                    bottom_k = combined.nlargest(k).index.tolist()
                    match = set(bottom_k) == set(eliminated_list)
                    deviation = len(set(eliminated_list) - set(bottom_k))
                    # Spearman/Kendall with judge scores
                    spearman = spearman_corr(fan_share.values, judge_scores.values)
                    kendall = kendall_tau(fan_share.values, judge_scores.values)
                    # elimination pressure: gap between worst and second worst
                    sorted_vals = combined.sort_values(ascending=False).values
                    pressure = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else 0.0
                else:
                    combined = (judge_scores / judge_scores.sum()) + fan_share
                    k = len(eliminated_list)
                    bottom_k = combined.nsmallest(k).index.tolist()
                    match = set(bottom_k) == set(eliminated_list)
                    deviation = len(set(eliminated_list) - set(bottom_k))
                    spearman = spearman_corr(fan_share.values, judge_scores.values)
                    kendall = kendall_tau(fan_share.values, judge_scores.values)
                    sorted_vals = combined.sort_values(ascending=True).values
                    pressure = float(sorted_vals[1] - sorted_vals[0]) if len(sorted_vals) > 1 else 0.0

                consistency_rows.append({
                    "season": season,
                    "week": t,
                    "scheme": scheme,
                    "eliminated_count": len(eliminated_list),
                    "bottom_k_match": int(match),
                    "deviation_count": deviation,
                })

                # Extended consistency metrics
                consistency_ext_rows.append({
                    "season": season,
                    "week": t,
                    "scheme": scheme,
                    "eliminated_count": len(eliminated_list),
                    "spearman_corr": float(spearman),
                    "kendall_tau": float(kendall),
                    "elimination_pressure": pressure,
                })

    # Save outputs
    results_df = pd.DataFrame(results)
    status_df = pd.DataFrame(status_rows)
    consistency_df = pd.DataFrame(consistency_rows)
    consistency_ext_df = pd.DataFrame(consistency_ext_rows)
    uncertainty_rank_df = pd.DataFrame(uncertainty_rank_rows)
    uncertainty_percent_df = pd.DataFrame(uncertainty_percent_rows)

    results_df.to_csv(OUTPUT_VOTES, index=False, encoding="utf-8-sig")
    status_df.to_csv(OUTPUT_STATUS, index=False, encoding="utf-8-sig")
    consistency_df.to_csv(OUTPUT_CONSISTENCY, index=False, encoding="utf-8-sig")
    consistency_ext_df.to_csv(OUTPUT_CONSISTENCY_EXT, index=False, encoding="utf-8-sig")
    uncertainty_rank_df.to_csv(OUTPUT_UNCERTAINTY_RANK, index=False, encoding="utf-8-sig")
    uncertainty_percent_df.to_csv(OUTPUT_UNCERTAINTY_PERCENT, index=False, encoding="utf-8-sig")

    # Display key results
    print("=== Fan Vote Estimates (Head) ===")
    print(results_df.head(20))
    print("\n=== Week Processing Status (Head) ===")
    print(status_df.head(20))
    print("\n=== Consistency Metrics (Head) ===")
    print(consistency_df.head(20))
    print("\n=== Consistency Extended (Head) ===")
    print(consistency_ext_df.head(20))

    # Example visualization: latest season finalists (placement <= 3)
    latest_season = df["season"].max()
    season_latest = df[df["season"] == latest_season].copy()
    finalists = season_latest[season_latest["placement"].isin([1, 2, 3])]["celebrity_name"].tolist()

    if finalists and not results_df.empty:
        plot_df = results_df[(results_df["season"] == latest_season) & (results_df["celebrity_name"].isin(finalists))]
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))
            for name, sub in plot_df.groupby("celebrity_name"):
                sub_sorted = sub.sort_values("week")
                plt.plot(sub_sorted["week"], sub_sorted["fan_vote_share"], marker="o", label=name)
            plt.title(f"第 {latest_season} 季决赛选手观众投票份额趋势")
            plt.xlabel("周次")
            plt.ylabel("观众投票份额")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(PLOT_FILE, dpi=300)
            plt.show()

    # Heatmap of uncertainty (fan share width) by season-week
    if not uncertainty_percent_df.empty:
        heat = uncertainty_percent_df.groupby(["season", "week"])["fan_share_width"].mean().reset_index()
        heat_pivot = heat.pivot(index="season", columns="week", values="fan_share_width").fillna(0.0)
        plt.figure(figsize=(12, 6))
        plt.imshow(heat_pivot.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar(label="投票区间宽度")
        plt.title("观众投票不确定性热力图（按赛季-周次）")
        plt.xlabel("周次")
        plt.ylabel("赛季")
        plt.tight_layout()
        plt.savefig(HEATMAP_FILE, dpi=300)
        plt.show()

    # Judge score variability vs uncertainty (relation analysis)
    variability_rows = []
    for (season, week), df_week in results_df.groupby(["season", "week"]):
        judge_scores = df_week["judge_total"].astype(float)
        if judge_scores.empty:
            continue
        q25, q75 = np.percentile(judge_scores.values, [25, 75])
        variability_rows.append({
            "season": season,
            "week": week,
            "scheme": df_week["scheme"].iloc[0],
            "judge_std": float(judge_scores.std()),
            "judge_iqr": float(q75 - q25),
        })

    variability_df = pd.DataFrame(variability_rows)
    if not variability_df.empty:
        percent_uncert = uncertainty_percent_df.groupby(["season", "week"])["fan_share_width"].mean()
        rank_uncert = uncertainty_rank_df.groupby(["season", "week"])["fan_rank_std"].mean()
        variability_df = variability_df.merge(
            percent_uncert.rename("fan_share_width_mean"),
            on=["season", "week"],
            how="left",
        )
        variability_df = variability_df.merge(
            rank_uncert.rename("fan_rank_std_mean"),
            on=["season", "week"],
            how="left",
        )
        variability_df.to_csv(OUTPUT_JUDGE_UNCERTAINTY, index=False, encoding="utf-8-sig")

        percent_df = variability_df[(variability_df["scheme"] == "percent") &
                                     (variability_df["fan_share_width_mean"].notna())]
        rank_df = variability_df[(variability_df["scheme"] == "rank") &
                                 (variability_df["fan_rank_std_mean"].notna())]
        if not percent_df.empty:
            p_corr = percent_df["judge_std"].corr(percent_df["fan_share_width_mean"])
            p_spearman = percent_df["judge_std"].rank().corr(percent_df["fan_share_width_mean"].rank())
            print(f"\n评委评分差异性-不确定性（百分比法）相关：Pearson={p_corr:.3f}, Spearman={p_spearman:.3f}")
        if not rank_df.empty:
            r_corr = rank_df["judge_std"].corr(rank_df["fan_rank_std_mean"])
            r_spearman = rank_df["judge_std"].rank().corr(rank_df["fan_rank_std_mean"].rank())
            print(f"评委评分差异性-不确定性（排名法）相关：Pearson={r_corr:.3f}, Spearman={r_spearman:.3f}")

    # Sensitivity analysis for preference weights (fixed logic)
    weight_grid = [
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 1.0),
        (1.0, 1.5),
    ]
    sensitivity_rows = []
    for w_smooth, w_corr in weight_grid:
        consistent = 0
        total = 0
        for (season, week), df_week in results_df.groupby(["season", "week"]):
            if df_week["eliminated_end_of_week"].sum() == 0:
                continue
            scheme = df_week["scheme"].iloc[0]
            judge_scores = df_week.set_index("celebrity_name")["judge_total"].astype(float)
            if scheme == "rank":
                rank_judge = judge_scores.rank(ascending=False, method="average")
                eliminated = df_week[df_week["eliminated_end_of_week"] == 1]["celebrity_name"].tolist()
                if len(eliminated) == 1:
                    est = estimate_votes_rank_based(
                        rank_judge,
                        eliminated[0],
                        judge_scores,
                        None,
                        smooth_w=w_smooth,
                        corr_w=w_corr,
                    )
                    if est is None:
                        continue
                    (fan_rank, fan_share), _ = est
                    combined = rank_judge + fan_rank
                    bottom_k = combined.nlargest(1).index.tolist()
                    consistent += int(set(bottom_k) == set(eliminated))
                    total += 1
            else:
                eliminated = df_week[df_week["eliminated_end_of_week"] == 1]["celebrity_name"].tolist()
                if len(eliminated) == 1:
                    fan_share, _ = estimate_votes_percent_based(
                        judge_scores,
                        eliminated[0],
                        None,
                        smooth_w=w_smooth,
                        corr_w=w_corr,
                    )
                    if fan_share is None:
                        continue
                    combined = (judge_scores / judge_scores.sum()) + fan_share
                    bottom_k = combined.nsmallest(1).index.tolist()
                    consistent += int(set(bottom_k) == set(eliminated))
                    total += 1

        sensitivity_rows.append({
            "smoothness_weight": w_smooth,
            "correlation_weight": w_corr,
            "consistency_rate": (consistent / total) if total > 0 else np.nan
        })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(OUTPUT_SENSITIVITY, index=False, encoding="utf-8-sig")

    # Feasible space visualization (percent scheme, single elimination)
    if not uncertainty_percent_df.empty:
        feasible_vals = uncertainty_percent_df["fan_share_width"].dropna()
        if not feasible_vals.empty:
            plt.figure(figsize=(6, 4))
            plt.hist(feasible_vals, bins=30, color="#4C78A8", alpha=0.8)
            plt.title("百分比法可行区间宽度分布")
            plt.xlabel("区间宽度")
            plt.ylabel("频数")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(PLOT_FEASIBLE_SPACE, dpi=300)
            plt.show()

    # Elimination pressure distribution
    if not consistency_ext_df.empty:
        pressure_vals = consistency_ext_df["elimination_pressure"].dropna()
        if not pressure_vals.empty:
            plt.figure(figsize=(6, 4))
            plt.hist(pressure_vals, bins=30, color="#F58518", alpha=0.8)
            plt.title("淘汰压力分布")
            plt.xlabel("压力值")
            plt.ylabel("频数")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(PLOT_PRESSURE, dpi=300)
            plt.show()

    print(f"\n结果已保存至：{OUTPUT_VOTES}")
    print(f"周次状态已保存至：{OUTPUT_STATUS}")
    print(f"图像已保存至：{PLOT_FILE}")
    print(f"一致性指标已保存至：{OUTPUT_CONSISTENCY}")
    print(f"一致性扩展指标已保存至：{OUTPUT_CONSISTENCY_EXT}")
    print(f"排名法不确定性已保存至：{OUTPUT_UNCERTAINTY_RANK}")
    print(f"百分比法不确定性已保存至：{OUTPUT_UNCERTAINTY_PERCENT}")
    print(f"评委差异性与不确定性关系已保存至：{OUTPUT_JUDGE_UNCERTAINTY}")
    print(f"不确定性热力图已保存至：{HEATMAP_FILE}")
    print(f"可行解空间图已保存至：{PLOT_FEASIBLE_SPACE}")
    print(f"淘汰压力分布图已保存至：{PLOT_PRESSURE}")


if __name__ == "__main__":
    main()
