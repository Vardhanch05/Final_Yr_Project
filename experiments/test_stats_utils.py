"""
Unit tests validating all statistical routines:
1. Exact Step-Down Holm vs statsmodels multipletests.
2. Step-down early halting logic.
3. Paired Wilcoxon signed-rank on an exact hand-calculated toy distribution.
4. Bootstrap 95% confidence intervals.
5. compute_relative_change for regression (RMSE) and classification (ROC-AUC).
6. compute_win_loss_tie with exact tie thresholds.
7. paired_row_level_bootstrap_test for both regression and classification paths.
"""

import sys
sys.path.append('experiments')
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests
from stats_utils import (
    holm_bonferroni_step_down,
    paired_wilcoxon_test,
    bootstrap_median_ci,
    paired_row_level_bootstrap_test,
    compute_win_loss_tie,
    compute_relative_change
)


def test_holm_matches_statsmodels():
    """Verify that our step-down Holm matches statsmodels multipletests exactly."""
    np.random.seed(42)
    for _ in range(20):
        m = np.random.randint(4, 15)
        raw_p = np.random.uniform(0.001, 0.5, size=m)

        sm_reject, sm_adj_p, _, _ = multipletests(raw_p, alpha=0.05, method='holm')
        our_adj_p, our_reject = holm_bonferroni_step_down(raw_p, alpha=0.05)

        np.testing.assert_allclose(our_adj_p, sm_adj_p, rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(our_reject, sm_reject)

    print("[PASS] Test 1: Step-down Holm matches statsmodels multipletests across 20 trials with 100% precision")


def test_holm_step_down_halting():
    """Verify that Holm halts at first non-rejection and does not skip."""
    # p_values:
    # index 0: 0.049
    # index 1: 0.0001
    # index 2: 0.030
    #
    # Sorted order:
    # Rank 0 -> index 1: 0.0001. Threshold: 0.05 / 3 = 0.0167 -> 0.0001 <= 0.0167 -> REJECT
    # Rank 1 -> index 2: 0.0300. Threshold: 0.05 / 2 = 0.0250 -> 0.0300 >  0.0250 -> FAIL TO REJECT (HALT!)
    # Rank 2 -> index 0: 0.0490. Threshold: 0.05 / 1 = 0.0500 -> 0.0490 <= 0.0500 -> MUST NOT REJECT (Halted!)
    p_values = np.array([0.049, 0.0001, 0.03])
    adj_p, reject = holm_bonferroni_step_down(p_values, alpha=0.05)

    assert reject[1] == True, "index 1 (p=0.0001) should be REJECTED at rank 0"
    assert reject[2] == False, "index 2 (p=0.030) should FAIL TO REJECT at rank 1"
    assert reject[0] == False, "index 0 (p=0.049) must NOT be rejected because step-down halted at rank 1"
    print("[PASS] Test 2: Step-down halts correctly at first non-rejection (no false rejections)")


def test_wilcoxon_hand_computed_toy():
    """
    Verify Wilcoxon signed-rank test against an exact by-hand analytical derivation.
    
    Pairs: x = [11, 12, 7, 14, 15], y = [10, 10, 10, 10, 10]
    Differences d = x - y = [+1, +2, -3, +4, +5]
    Magnitudes |d| = [1, 2, 3, 4, 5]
    Ranks:
      |d|=1 -> rank 1 (positive)
      |d|=2 -> rank 2 (positive)
      |d|=3 -> rank 3 (negative)
      |d|=4 -> rank 4 (positive)
      |d|=5 -> rank 5 (positive)
    
    Sum of negative ranks: W- = 3.
    Sum of positive ranks: W+ = 1 + 2 + 4 + 5 = 12.
    Test statistic W = min(W+, W-) = 3.0.
    
    Exact two-sided permutation p-value:
    Number of subset sums of {1,2,3,4,5} <= 3:
      0: empty (1)
      1: {1}   (1)
      2: {2}   (1)
      3: {3}, {1,2} (2)
      Total = 5 out of 2^5 = 32.
      Two-sided p = 2 * (5 / 32) = 10 / 32 = 0.3125.
    """
    x = np.array([11.0, 12.0, 7.0, 14.0, 15.0])
    y = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    stat, p_val = paired_wilcoxon_test(x, y, zero_method='zsplit')

    assert stat == 3.0, f"Expected hand-calculated statistic W=3.0, got {stat}"
    assert abs(p_val - 0.3125) < 1e-4, f"Expected exact analytical p=0.3125, got {p_val}"
    print(f"[PASS] Test 3: Wilcoxon signed-rank matches hand-computed analytical distribution (W=3.0, p=0.3125)")


def test_bootstrap_ci():
    """Verify non-parametric bootstrap produces valid 95% CI containing true median."""
    np.random.seed(42)
    samples = np.random.normal(loc=1.5, scale=0.5, size=20)
    point, low, high = bootstrap_median_ci(samples, n_bootstraps=1000, ci=0.95, seed=42)

    assert low < point < high, f"Invalid CI: [{low}, {point}, {high}]"
    assert low > 1.0 and high < 2.0
    print(f"[PASS] Test 4: Bootstrap 95% CI produces valid confidence bounds: [{low:.3f}, {point:.3f}, {high:.3f}]")


def test_compute_relative_change():
    """Verify signed relative change calculation for both regression and classification."""
    # Regression: positive means candidate has lower RMSE
    rel_reg_better = compute_relative_change(1.0, 0.9, task_type='regression')
    assert abs(rel_reg_better - 10.0) < 1e-6, f"Expected +10.0%, got {rel_reg_better}"

    rel_reg_worse = compute_relative_change(1.0, 1.1, task_type='regression')
    assert abs(rel_reg_worse - (-10.0)) < 1e-6, f"Expected -10.0%, got {rel_reg_worse}"

    # Classification: positive means candidate has higher ROC-AUC
    # baseline AUC = 0.80, candidate AUC = 0.85 -> relative reduction in error (1 - 0.80 = 0.20): 0.05 / 0.20 = +25.0%
    rel_cls_better = compute_relative_change(0.80, 0.85, task_type='binclass')
    assert abs(rel_cls_better - 25.0) < 1e-6, f"Expected +25.0%, got {rel_cls_better}"
    print("[PASS] Test 5: Relative change computation verified for regression and classification")


def test_compute_win_loss_tie():
    """Verify documented tie rule logic."""
    improvements = np.array([1.5, -0.8, 0.03, -0.04, 2.0, -1.2])
    wins, losses, ties = compute_win_loss_tie(improvements, tie_threshold=0.05)
    assert wins == 2, f"Expected 2 wins (> 0.05), got {wins}"
    assert losses == 2, f"Expected 2 losses (< -0.05), got {losses}"
    assert ties == 2, f"Expected 2 ties (|delta| <= 0.05), got {ties}"
    print("[PASS] Test 6: Win/Loss/Tie rule verified with exact thresholding")


def test_paired_row_level_bootstrap():
    """Verify row-level bootstrap for both regression and classification paths."""
    np.random.seed(42)
    N = 100
    y_reg = np.random.randn(N)
    pred_cand_reg = y_reg + np.random.normal(0, 0.2, N)
    pred_base_reg = y_reg + np.random.normal(0, 0.5, N)

    delta_reg, p_reg = paired_row_level_bootstrap_test(
        y_reg, pred_cand_reg, pred_base_reg, task_type='regression', n_bootstraps=200
    )
    assert delta_reg > 0, "Candidate should outperform baseline in RMSE delta"
    assert p_reg < 0.05, f"Expected significance on row level, got p={p_reg}"

    # Classification path: tests sklearn roc_auc_score import and execution
    y_cls = np.random.binomial(1, 0.5, size=N)
    pred_cand_cls = np.clip(y_cls * 0.6 + np.random.uniform(0.1, 0.3, N), 0.01, 0.99)
    pred_base_cls = np.random.uniform(0.1, 0.9, size=N)  # random guessing (~0.5 AUC)

    delta_cls, p_cls = paired_row_level_bootstrap_test(
        y_cls, pred_cand_cls, pred_base_cls, task_type='binclass', n_bootstraps=200
    )
    assert delta_cls > 0, "Candidate should outperform baseline in AUC delta"
    assert p_cls < 0.05, f"Expected significance on row level, got p={p_cls}"
    print("[PASS] Test 7: Paired row-level bootstrap test passes for both regression and classification")


if __name__ == '__main__':
    print("=" * 70)
    print("      RUNNING COMPLETE STATISTICAL RIGOR UNIT TEST SUITE      ")
    print("=" * 70)
    test_holm_matches_statsmodels()
    test_holm_step_down_halting()
    test_wilcoxon_hand_computed_toy()
    test_bootstrap_ci()
    test_compute_relative_change()
    test_compute_win_loss_tie()
    test_paired_row_level_bootstrap()
    print("=" * 70)
    print("ALL 7 STATISTICAL UNIT TESTS PASSED WITH 100% PRECISION!")
    print("=" * 70)
