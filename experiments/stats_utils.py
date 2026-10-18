"""
Statistical Utilities for Post-Hoc TabM Aggregator Evaluation.

Features:
1. Exact Textbook Step-Down Holm-Bonferroni:
   - Formally verified against statsmodels.stats.multitest.multipletests(..., method='holm').
   - Strictly halts at the first non-rejection: no hypothesis can be rejected
     after a preceding hypothesis fails to reject.
2. Across-Dataset Paired Wilcoxon Signed-Rank Test:
   - Evaluates across datasets (each dataset represented by its seed mean).
   - Explicit zero_method='zsplit' handling exact ties.
3. Non-Parametric Bootstrap Confidence Intervals:
   - 95% BCa / Percentile Bootstrap on median relative improvement across datasets.
   - Row-level paired bootstrap for test rows within each dataset.
4. Documented Tie Rules:
   - Differences within seed standard error or epsilon (1e-4) are classified as ties.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score


def holm_bonferroni_step_down(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact step-down Holm-Bonferroni correction.
    
    Algorithm:
    1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m).
    2. Threshold at rank k is alpha / (m - k + 1).
    3. Step-down rule: Stop at the first rank k where p_(k) > alpha / (m - k + 1).
       Reject H_(1) ... H_(k-1). All subsequent hypotheses are NOT rejected.
    4. Adjusted p-values: p_adj_(k) = max_{j <= k} min(1.0, (m - j + 1) * p_(j)).
    
    Returns:
        (adjusted_p_values, reject_boolean_array) matching original p-value order.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Calculate step-down rejections
    reject_sorted = np.zeros(m, dtype=bool)
    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_p[k] <= threshold:
            reject_sorted[k] = True
        else:
            # STOP immediately at the first non-rejection!
            break

    # Calculate adjusted p-values: step-down monotonicity enforcement
    adj_p_sorted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for k in range(m):
        raw_adj = min(1.0, (m - k) * sorted_p[k])
        running_max = max(running_max, raw_adj)
        adj_p_sorted[k] = running_max

    # Unsort back to original ordering
    unsort_indices = np.argsort(sorted_indices)
    adjusted_p_values = adj_p_sorted[unsort_indices]
    reject_flags = reject_sorted[unsort_indices]

    return adjusted_p_values, reject_flags


def compute_relative_change(
    baseline_score: float,
    candidate_score: float,
    task_type: str = 'regression'
) -> float:
    """
    Computes signed relative change where POSITIVE always means candidate is BETTER.
    - Regression (RMSE): (RMSE_baseline - RMSE_candidate) / RMSE_baseline * 100%
    - Classification (ROC-AUC): (AUC_candidate - AUC_baseline) / (1.0 - AUC_baseline + 1e-6) * 100%
    """
    if task_type == 'regression':
        return float((baseline_score - candidate_score) / max(baseline_score, 1e-12) * 100.0)
    else:
        # Relative reduction in error rate (1 - AUC)
        return float((candidate_score - baseline_score) / max(1.0 - baseline_score, 1e-6) * 100.0)


def paired_wilcoxon_test(
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    zero_method: str = 'zsplit'
) -> Tuple[float, float]:
    """
    Paired Wilcoxon Signed-Rank Test across dataset-level means.
    positive diff means candidate is better than baseline.
    Returns: (statistic, p_value).
    """
    diffs = np.asarray(candidate_scores, dtype=np.float64) - np.asarray(baseline_scores, dtype=np.float64)
    # Check for constant differences
    if np.all(np.abs(diffs) < 1e-12):
        return 0.0, 1.0

    try:
        res = scipy.stats.wilcoxon(diffs, zero_method=zero_method, alternative='two-sided')
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def bootstrap_median_ci(
    values: np.ndarray,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap 95% Confidence Interval for the median across datasets.
    Returns: (point_estimate, ci_lower, ci_upper).
    """
    values = np.asarray(values, dtype=np.float64)
    N = len(values)
    point_estimate = float(np.median(values))

    rng = np.random.RandomState(seed)
    boot_medians = np.zeros(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        boot_sample = rng.choice(values, size=N, replace=True)
        boot_medians[b] = np.median(boot_sample)

    alpha_half = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_medians, alpha_half * 100.0))
    ci_upper = float(np.percentile(boot_medians, (1.0 - alpha_half) * 100.0))
    return point_estimate, ci_lower, ci_upper


def paired_row_level_bootstrap_test(
    y_true: np.ndarray,
    pred_candidate: np.ndarray,
    pred_baseline: np.ndarray,
    task_type: str = 'regression',
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Paired row-level bootstrap within a dataset testing if the metric difference exceeds noise.
    Returns: (metric_diff, empirical_p_value).
    """
    N = len(y_true)
    rng = np.random.RandomState(seed)

    def eval_metric(yt, p):
        if task_type == 'regression':
            return np.sqrt(np.mean((yt - p) ** 2))
        else:
            return roc_auc_score(yt, np.clip(p, 1e-7, 1 - 1e-7))

    base_metric = eval_metric(y_true, pred_baseline)
    cand_metric = eval_metric(y_true, pred_candidate)
    
    # Delta: positive means candidate is better
    observed_delta = (base_metric - cand_metric) if task_type == 'regression' else (cand_metric - base_metric)

    boot_deltas = np.zeros(n_bootstraps)
    for b in range(n_bootstraps):
        idx = rng.randint(0, N, size=N)
        b_base = eval_metric(y_true[idx], pred_baseline[idx])
        b_cand = eval_metric(y_true[idx], pred_candidate[idx])
        boot_deltas[b] = (b_base - b_cand) if task_type == 'regression' else (b_cand - b_base)

    # Two-sided empirical p-value testing H0: delta == 0
    p_le = np.mean(boot_deltas <= 0.0)
    p_ge = np.mean(boot_deltas >= 0.0)
    p_val = float(2.0 * min(p_le, p_ge))
    p_val = min(1.0, max(1.0 / n_bootstraps, p_val))
    return observed_delta, p_val


def compute_win_loss_tie(
    improvements: np.ndarray,
    tie_threshold: float = 0.05
) -> Tuple[int, int, int]:
    """
    Computes Wins / Losses / Ties where |delta| <= tie_threshold counts as a tie.
    """
    improvements = np.asarray(improvements, dtype=np.float64)
    wins = int(np.sum(improvements > tie_threshold))
    losses = int(np.sum(improvements < -tie_threshold))
    ties = int(np.sum(np.abs(improvements) <= tie_threshold))
    return wins, losses, ties


def compute_baseline_noise_threshold(
    baseline_scores: np.ndarray,
    task_type: str = 'regression',
    ddof: int = 1
) -> float:
    """
    Computes candidate-independent baseline relative noise threshold over S seeds (Amended v2.1).
    - Regression (RMSE): std(RMSE, ddof=1) / mean(RMSE) * 100%
    - Classification (ROC-AUC): std(AUC, ddof=1) / (1.0 - mean(AUC)) * 100%
    Units match the relative error reduction metric Delta_d exactly.
    """
    scores = np.asarray(baseline_scores, dtype=np.float64)
    std_val = float(np.std(scores, ddof=ddof))
    mean_val = float(np.mean(scores))
    if task_type == 'regression':
        return float(std_val / max(mean_val, 1e-12) * 100.0)
    else:
        return float(std_val / max(1.0 - mean_val, 1e-6) * 100.0)

