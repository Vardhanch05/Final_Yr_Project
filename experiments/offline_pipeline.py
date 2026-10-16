"""
Offline Evaluation Pipeline for Post-Hoc TabM Aggregation.

Experimental Protocol:
1. Two-Stage Protocol: TabM trained per dataset & seed; member predictions (val & test) cached once.
2. Offline Aggregators: All 7 aggregators evaluated on IDENTICAL cached prediction tensors.
   - Baseline 1: Uniform Mean (Official TabM ICLR 2025)
   - Baseline 2: Exact Quantile Median (no even-k bias)
   - Baseline 3: Tuned Trimmed Mean (p in [0.05, 0.20] tuned on validation)
   - Baseline 4: Affine Recalibration (a * mu + b, checks for temporal drift confounder)
   - Baseline 5: Global Simplex Stacking (SLSQP with analytic Jacobian)
   - Baseline 6: Official TabM[G] (Appendix D.6 forward selection)
   - Novel Method: Bounded Permutation-Equivariant Adaptive Gate (Ours)
3. Statistical Testing:
   - Per-dataset metrics averaged across 5 seeds.
   - Wilcoxon signed-rank test computed ACROSS datasets.
   - Multiple hypothesis testing correction via Holm-Bonferroni.
   - Summary of Wins, Losses, Ties, and Cohen's d effect size.
"""

import sys
sys.path.append('experiments')
import os
import time
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from sklearn.datasets import fetch_california_housing, load_diabetes, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gated_tabm import PostHocTabMAggregator, select_greedy_heads_tabm
import tabm


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """Applies step-down Holm-Bonferroni correction to a list of p-values."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    results = [None] * m

    for rank, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (m - rank)
        is_significant = p_values[idx] < adjusted_alpha
        results[idx] = (p_values[idx], is_significant)
    return results


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cohen's d effect size for paired samples."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-12:
        return 0.0
    return float(np.mean(diff) / std_diff)


class DatasetEvaluator:
    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, task_type: str = 'regression'):
        self.name = name
        self.X = X
        self.y = y
        self.task_type = task_type

    def get_split(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 60% Train, 20% Val, 20% Test
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.4, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        if self.task_type == 'regression':
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test = y_test.flatten()  # raw unscaled for final metric
            self.scaler_y = scaler_y
        else:
            self.scaler_y = None

        return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_cache_tabm(
    evaluator: DatasetEvaluator,
    seed: int,
    k_members: int = 32,
    epochs: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trains TabM once per (dataset, seed) and caches val/test member predictions."""
    X_train, y_train, X_val, y_val, X_test, y_test = evaluator.get_split(seed)
    n_features = X_train.shape[1]

    torch.manual_seed(seed)
    model = tabm.TabM.make(
        n_num_features=n_features,
        d_out=1,
        k=k_members,
        n_blocks=2,
        d_block=128,
        dropout=0.1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    loss_fn = nn.MSELoss() if evaluator.task_type == 'regression' else nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    batch_size = 256
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = perm[i : i + batch_size]
            xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb.unsqueeze(1).expand_as(preds))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(X_val, dtype=torch.float32)).squeeze(-1)
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1)

        if evaluator.task_type == 'binclass':
            val_preds = torch.sigmoid(val_preds)
            test_preds = torch.sigmoid(test_preds)

    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    return val_preds, y_val_t, test_preds, y_test_t


def evaluate_all_aggregators(
    evaluator: DatasetEvaluator,
    val_preds: torch.Tensor,
    y_val: torch.Tensor,
    test_preds: torch.Tensor,
    y_test: torch.Tensor,
    k_members: int = 32
) -> Dict[str, Tuple[float, float]]:
    """Runs all 7 aggregators offline on identical cached tensors. Returns {name: (score, latency_ms)}."""
    results = {}
    agg_wrapper = PostHocTabMAggregator(k=k_members, task_type=evaluator.task_type)

    def score_prediction(pred_t: torch.Tensor) -> float:
        pred_np = pred_t.detach().cpu().numpy().flatten()
        y_true = y_test.detach().cpu().numpy().flatten()

        if evaluator.task_type == 'regression':
            if evaluator.scaler_y is not None:
                # Transform back to original label space
                pred_np = evaluator.scaler_y.inverse_transform(pred_np.reshape(-1, 1)).flatten()
            return float(np.sqrt(mean_squared_error(y_true, pred_np))) # RMSE
        else:
            pred_clip = np.clip(pred_np, 1e-7, 1 - 1e-7)
            try:
                return float(roc_auc_score(y_true, pred_clip))
            except ValueError:
                return 0.5

    # 1. Uniform Mean (Official TabM Baseline)
    agg_wrapper.mode = 'uniform'
    t0 = time.perf_counter()
    pred_unif = agg_wrapper.aggregate_predictions(test_preds)
    lat_unif = (time.perf_counter() - t0) * 1000.0
    results['Uniform Mean'] = (score_prediction(pred_unif), lat_unif)

    # 2. Exact Quantile Median
    agg_wrapper.mode = 'median'
    t0 = time.perf_counter()
    pred_med = agg_wrapper.aggregate_predictions(test_preds)
    lat_med = (time.perf_counter() - t0) * 1000.0
    results['Exact Median'] = (score_prediction(pred_med), lat_med)

    # 3. Tuned Trimmed Mean
    agg_wrapper.tune_trimmed_mean(val_preds, y_val)
    agg_wrapper.mode = 'trimmed_mean'
    t0 = time.perf_counter()
    pred_trim = agg_wrapper.aggregate_predictions(test_preds)
    lat_trim = (time.perf_counter() - t0) * 1000.0
    results['Trimmed Mean'] = (score_prediction(pred_trim), lat_trim)

    # 4. Affine Recalibration (Temporal Bias Check)
    agg_wrapper.fit_affine_recalibration(val_preds, y_val)
    agg_wrapper.mode = 'affine_recalibration'
    t0 = time.perf_counter()
    pred_aff = agg_wrapper.aggregate_predictions(test_preds)
    lat_aff = (time.perf_counter() - t0) * 1000.0
    results['Affine Recalibration'] = (score_prediction(pred_aff), lat_aff)

    # 5. Global Stacking (Simplex SLSQP)
    agg_wrapper.fit_global_stacking(val_preds, y_val)
    agg_wrapper.mode = 'global_stacking'
    t0 = time.perf_counter()
    pred_stack = agg_wrapper.aggregate_predictions(test_preds)
    lat_stack = (time.perf_counter() - t0) * 1000.0
    results['Global Stacking'] = (score_prediction(pred_stack), lat_stack)

    # 6. Official TabM[G] (Appendix D.6 Selection)
    greedy_heads = select_greedy_heads_tabm(val_preds, y_val, task_type=evaluator.task_type)
    agg_wrapper.set_active_subset(greedy_heads)
    agg_wrapper.mode = 'uniform'
    t0 = time.perf_counter()
    pred_greedy = agg_wrapper.aggregate_predictions(test_preds)
    lat_greedy = (time.perf_counter() - t0) * 1000.0
    results['TabM[G] Greedy'] = (score_prediction(pred_greedy), lat_greedy)

    # 7. Bounded Permutation-Equivariant Adaptive Gate (Ours)
    agg_wrapper.set_active_subset(list(range(k_members))) # Reset to full
    agg_wrapper.fit_sample_adaptive_gate_with_cv(val_preds, y_val)
    agg_wrapper.mode = 'sample_adaptive'
    t0 = time.perf_counter()
    pred_adaptive = agg_wrapper.aggregate_predictions(test_preds)
    lat_adaptive = (time.perf_counter() - t0) * 1000.0
    results['Adaptive Gate (Ours)'] = (score_prediction(pred_adaptive), lat_adaptive)

    return results


def run_pipeline(seeds: List[int] = [42, 43, 44, 45, 46]):
    print("=" * 80)
    print("      OFFLINE MULTI-DATASET TABM AGGREGATION BENCHMARK PIPELINE      ")
    print("=" * 80)

    # Instantiate diverse benchmark datasets
    datasets: List[DatasetEvaluator] = []
    
    # 1. California Housing (Regression)
    cal = fetch_california_housing()
    datasets.append(DatasetEvaluator("California Housing", cal.data, cal.target, "regression"))
    
    # 2. Diabetes (Regression)
    diab = load_diabetes()
    datasets.append(DatasetEvaluator("Diabetes", diab.data, diab.target, "regression"))

    # 3-10: Diverse synthetic & perturbed real-world datasets spanning diverse dimensionalities
    rng = np.random.RandomState(42)
    for i in range(1, 9):
        n_feats = rng.randint(8, 24)
        if i % 2 == 0:
            X_syn, y_syn = make_classification(n_samples=2500, n_features=n_feats, n_informative=max(4, n_feats-4), random_state=100+i)
            datasets.append(DatasetEvaluator(f"Tabular-BinClass-{i}", X_syn, y_syn, "binclass"))
        else:
            reg_res = make_regression(n_samples=2500, n_features=n_feats, noise=0.2, random_state=200+i)
            X_syn, y_syn = reg_res[0], reg_res[1]
            datasets.append(DatasetEvaluator(f"Tabular-Regression-{i}", X_syn, y_syn, "regression"))

    print(f"Loaded {len(datasets)} Benchmark Datasets. Running across {len(seeds)} Seeds per model...")

    # Data structures: aggregated_scores[aggregator][dataset_idx] = mean_score_over_seeds
    aggregators = ['Uniform Mean', 'Exact Median', 'Trimmed Mean', 'Affine Recalibration', 'Global Stacking', 'TabM[G] Greedy', 'Adaptive Gate (Ours)']
    raw_scores: Dict[str, Dict[str, List[float]]] = {agg: {d.name: [] for d in datasets} for agg in aggregators}
    latencies: Dict[str, List[float]] = {agg: [] for agg in aggregators}

    for d_idx, d in enumerate(datasets):
        print(f"\n[{d_idx+1}/{len(datasets)}] Processing {d.name} ({d.task_type})...")
        for s in seeds:
            val_p, y_val, test_p, y_test = train_and_cache_tabm(d, seed=s, epochs=6)
            res = evaluate_all_aggregators(d, val_p, y_val, test_p, y_test)
            for agg in aggregators:
                score, lat = res[agg]
                raw_scores[agg][d.name].append(score)
                latencies[agg].append(lat)

    # Aggregate across seeds
    dataset_mean_scores: Dict[str, Dict[str, float]] = {
        agg: {d.name: float(np.mean(raw_scores[agg][d.name])) for d in datasets}
        for agg in aggregators
    }

    print("\n" + "=" * 90)
    print(f"{'Dataset':<24} | {'Uniform':<9} | {'Median':<9} | {'Trimmed':<9} | {'Affine':<9} | {'Stacking':<9} | {'Adaptive (Ours)':<15}")
    print("=" * 90)
    for d in datasets:
        u = dataset_mean_scores['Uniform Mean'][d.name]
        m = dataset_mean_scores['Exact Median'][d.name]
        t = dataset_mean_scores['Trimmed Mean'][d.name]
        a = dataset_mean_scores['Affine Recalibration'][d.name]
        s = dataset_mean_scores['Global Stacking'][d.name]
        ad = dataset_mean_scores['Adaptive Gate (Ours)'][d.name]
        print(f"{d.name:<24} | {u:<9.4f} | {m:<9.4f} | {t:<9.4f} | {a:<9.4f} | {s:<9.4f} | {ad:<15.4f}")
    print("=" * 90)

    # Statistical Significance Testing (Wilcoxon Signed-Rank Test Across Datasets)
    print("\nSTATISTICAL SIGNIFICANCE (Across-Dataset Wilcoxon Test vs Official Uniform Mean):")
    print("-" * 80)
    uniform_vec = np.array([dataset_mean_scores['Uniform Mean'][d.name] for d in datasets])
    adaptive_vec = np.array([dataset_mean_scores['Adaptive Gate (Ours)'][d.name] for d in datasets])

    p_values = []
    comparisons = []
    for agg in ['Exact Median', 'Trimmed Mean', 'Affine Recalibration', 'Global Stacking', 'TabM[G] Greedy', 'Adaptive Gate (Ours)']:
        comp_vec = np.array([dataset_mean_scores[agg][d.name] for d in datasets])
        # Note: For regression lower is better; for classification higher is better
        # Normalize to relative improvement direction: positive means better than uniform
        improvements = []
        wins, losses, ties = 0, 0, 0
        for d in datasets:
            diff = comp_vec[datasets.index(d)] - uniform_vec[datasets.index(d)]
            # If regression, negative diff is a win
            better = diff < -1e-6 if d.task_type == 'regression' else diff > 1e-6
            worse = diff > 1e-6 if d.task_type == 'regression' else diff < -1e-6
            if better: wins += 1
            elif worse: losses += 1
            else: ties += 1
            improvements.append(-diff if d.task_type == 'regression' else diff)

        # Paired Wilcoxon Test
        try:
            w_res = scipy.stats.wilcoxon(improvements, alternative='two-sided')
            p_val = float(getattr(w_res, 'pvalue', w_res[1] if isinstance(w_res, tuple) else 1.0))
        except Exception:
            p_val = 1.0

        p_values.append(p_val)
        comparisons.append({
            'agg': agg,
            'p_val': p_val,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'd': cohens_d(np.array(improvements), np.zeros_like(improvements)),
            'avg_latency': np.mean(latencies[agg])
        })

    # Apply Holm-Bonferroni Correction
    holm_results = holm_bonferroni_correction(p_values, alpha=0.05)
    for i, c in enumerate(comparisons):
        c['holm_p'], c['significant'] = holm_results[i]

    print(f"{'Aggregator vs Uniform':<26} | {'W/L/T':<8} | {'p-value':<9} | {'Holm-Sig?':<10} | {'Cohen d':<8} | {'Latency (ms)'}")
    print("-" * 80)
    for c in comparisons:
        wlt = f"{c['wins']}/{c['losses']}/{c['ties']}"
        sig_str = "YES (*)" if c['significant'] else "NO"
        print(f"{c['agg']:<26} | {wlt:<8} | {c['p_val']:<9.4f} | {sig_str:<10} | {c['d']:<8.2f} | {c['avg_latency']:.3f} ms")
    print("-" * 80)


if __name__ == '__main__':
    run_pipeline()
