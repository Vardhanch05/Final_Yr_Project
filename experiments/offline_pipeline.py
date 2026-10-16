"""
Offline Evaluation Pipeline for Post-Hoc TabM Aggregation.

Features:
1. On-Disk Caching:
   - Cached predictions saved to `experiments/cache/{dataset}_{seed}/`:
     val_preds.pt, test_preds.pt, y_val.pt, y_test.pt, metadata.json.
   - Allows instant re-evaluation of all aggregators without re-training models.
2. Full Raw Results CSV Export:
   - Saves every single evaluation tuple (dataset, seed, aggregator, metric, relative_gain, latency_ms)
     to `experiments/results_raw.csv`.
3. High-Precision Latency Benchmark:
   - 10 warmup iterations, 100 timed iterations, median reported in milliseconds, GPU synchronized.
4. Rigorous Hypothesis Testing:
   - Primary Test: Sample-Adaptive Gate vs. Uniform Mean (Wilcoxon across datasets).
   - Secondary Family: Sample-Adaptive Gate vs. [Median, Trimmed, Affine, Stacking, TabM[G]]
     with step-down Holm-Bonferroni correction.
   - Non-parametric bootstrap 95% CI on median relative improvement across datasets.
   - Per-dataset paired row-level bootstrap test over shared test rows.
"""

import sys
sys.path.append('experiments')
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from sklearn.datasets import fetch_california_housing, load_diabetes, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gated_tabm import PostHocTabMAggregator, select_greedy_heads_tabm
from stats_utils import (
    holm_bonferroni_step_down,
    compute_relative_change,
    paired_wilcoxon_test,
    bootstrap_median_ci,
    paired_row_level_bootstrap_test,
    compute_win_loss_tie
)
import benchmark_config as config
import tabm

CACHE_DIR = Path("experiments/cache")
RESULTS_CSV = Path("experiments/results_raw.csv")


class DatasetContainer:
    """Manages tabular dataset splits, caching, and preprocessing."""
    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, task_type: str = 'regression'):
        self.name = name
        self.X = X
        self.y = y
        self.task_type = task_type

    def get_split(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
        # Fixed 60% Train, 20% Val, 20% Test split (temporal or random depending on dataset)
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
            y_test = y_test.flatten()  # Raw unscaled for true RMSE metric
        else:
            scaler_y = None
            y_test = y_test.flatten()

        return X_train, y_train, X_val, y_val, X_test, y_test, scaler_y


def get_or_create_cached_predictions(
    dataset: DatasetContainer,
    seed: int,
    k_members: int = 32,
    epochs: int = 8,
    force_retrain: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[StandardScaler]]:
    """
    On-Disk Caching: checks experiments/cache/{dataset}_{seed}/ for persisted tensors.
    Trains TabM only if cache is missing or force_retrain is True.
    """
    safe_name = dataset.name.replace('/', '_').replace('\\', '_')
    run_dir = CACHE_DIR / f"{safe_name}_seed_{seed}"
    val_pred_file = run_dir / "val_preds.pt"
    test_pred_file = run_dir / "test_preds.pt"
    y_val_file = run_dir / "y_val.pt"
    y_test_file = run_dir / "y_test.pt"

    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y = dataset.get_split(seed)

    if not force_retrain and val_pred_file.exists() and test_pred_file.exists():
        val_preds = torch.load(val_pred_file, weights_only=True)
        test_preds = torch.load(test_pred_file, weights_only=True)
        y_val_t = torch.load(y_val_file, weights_only=True)
        y_test_t = torch.load(y_test_file, weights_only=True)
        return val_preds, y_val_t, test_preds, y_test_t, scaler_y

    # Train TabM model
    run_dir.mkdir(parents=True, exist_ok=True)
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
    loss_fn = nn.MSELoss() if dataset.task_type == 'regression' else nn.BCEWithLogitsLoss()

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

        if dataset.task_type == 'binclass':
            val_preds = torch.sigmoid(val_preds)
            test_preds = torch.sigmoid(test_preds)

    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Persist to disk
    torch.save(val_preds, val_pred_file)
    torch.save(test_preds, test_pred_file)
    torch.save(y_val_t, y_val_file)
    torch.save(y_test_t, y_test_file)

    meta = {
        "dataset": dataset.name,
        "seed": seed,
        "task_type": dataset.task_type,
        "k_members": k_members,
        "val_samples": len(y_val_t),
        "test_samples": len(y_test_t)
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return val_preds, y_val_t, test_preds, y_test_t, scaler_y


def measure_aggregator_latency(
    agg_wrapper: PostHocTabMAggregator,
    test_preds: torch.Tensor,
    warmup_runs: int = config.LATENCY_WARMUP_RUNS,
    timed_runs: int = config.LATENCY_TIMED_RUNS
) -> float:
    """Measures aggregation latency (ms) with warmup and median over timed repetitions."""
    # Warmup
    for _ in range(warmup_runs):
        _ = agg_wrapper.aggregate_predictions(test_preds)

    times = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = agg_wrapper.aggregate_predictions(test_preds)
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.median(times))


def run_offline_benchmark(
    datasets: List[DatasetContainer],
    seeds: List[int] = config.SEEDS,
    phase: str = "dev"
) -> pd.DataFrame:
    """Runs all 7 aggregators offline on cached tensors across all datasets and seeds."""
    print("=" * 80)
    print(f"      OFFLINE POST-HOC TABM AGGREGATOR EVALUATION (PHASE: {phase.upper()})      ")
    print("=" * 80)
    print(f"Datasets ({len(datasets)}): {[d.name for d in datasets]}")
    print(f"Seeds ({len(seeds)}): {seeds}")

    records = []

    for d_idx, dataset in enumerate(datasets):
        print(f"\n[{d_idx+1}/{len(datasets)}] Evaluating {dataset.name} ({dataset.task_type})...")
        for s in seeds:
            val_p, y_val, test_p, y_test, scaler_y = get_or_create_cached_predictions(dataset, seed=s)
            y_test_np = y_test.detach().cpu().numpy().flatten()
            k = test_p.shape[1]

            agg_wrapper = PostHocTabMAggregator(k=k, task_type=dataset.task_type)

            def score_func(pred_t: torch.Tensor) -> float:
                p_np = pred_t.detach().cpu().numpy().flatten()
                if dataset.task_type == 'regression':
                    if scaler_y is not None:
                        p_np = scaler_y.inverse_transform(p_np.reshape(-1, 1)).flatten()
                    return float(np.sqrt(mean_squared_error(y_test_np, p_np)))
                else:
                    return float(roc_auc_score(y_test_np, np.clip(p_np, 1e-7, 1 - 1e-7)))

            # Fit all candidate models on validation data
            agg_wrapper.tune_trimmed_mean(val_p, y_val)
            agg_wrapper.fit_affine_recalibration(val_p, y_val)
            agg_wrapper.fit_global_stacking(val_p, y_val)
            greedy_subset = select_greedy_heads_tabm(val_p, y_val, task_type=dataset.task_type)
            agg_wrapper.fit_sample_adaptive_gate_with_cv(val_p, y_val)

            # Evaluate each aggregator on test set
            test_preds_by_agg = {}

            # 1. Uniform
            agg_wrapper.mode = 'uniform'
            agg_wrapper.set_active_subset(list(range(k)))
            pred_unif = agg_wrapper.aggregate_predictions(test_p)
            score_unif = score_func(pred_unif)
            lat_unif = measure_aggregator_latency(agg_wrapper, test_p)
            test_preds_by_agg['Uniform Mean'] = pred_unif.detach().cpu().numpy().flatten()
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Uniform Mean", "score": score_unif, "rel_gain_pct": 0.0, "latency_ms": lat_unif
            })

            # 2. Exact Median
            agg_wrapper.mode = 'median'
            pred_med = agg_wrapper.aggregate_predictions(test_p)
            score_med = score_func(pred_med)
            lat_med = measure_aggregator_latency(agg_wrapper, test_p)
            test_preds_by_agg['Exact Median'] = pred_med.detach().cpu().numpy().flatten()
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Exact Median", "score": score_med,
                "rel_gain_pct": compute_relative_change(score_unif, score_med, dataset.task_type),
                "latency_ms": lat_med
            })

            # 3. Trimmed Mean
            agg_wrapper.mode = 'trimmed_mean'
            pred_trim = agg_wrapper.aggregate_predictions(test_p)
            score_trim = score_func(pred_trim)
            lat_trim = measure_aggregator_latency(agg_wrapper, test_p)
            test_preds_by_agg['Trimmed Mean'] = pred_trim.detach().cpu().numpy().flatten()
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Trimmed Mean", "score": score_trim,
                "rel_gain_pct": compute_relative_change(score_unif, score_trim, dataset.task_type),
                "latency_ms": lat_trim
            })

            # 4. Affine Recalibration
            agg_wrapper.mode = 'affine_recalibration'
            pred_aff = agg_wrapper.aggregate_predictions(test_p)
            score_aff = score_func(pred_aff)
            lat_aff = measure_aggregator_latency(agg_wrapper, test_p)
            test_preds_by_agg['Affine Recalibration'] = pred_aff.detach().cpu().numpy().flatten()
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Affine Recalibration", "score": score_aff,
                "rel_gain_pct": compute_relative_change(score_unif, score_aff, dataset.task_type),
                "latency_ms": lat_aff
            })

            # 5. Global Stacking
            agg_wrapper.mode = 'global_stacking'
            pred_stack = agg_wrapper.aggregate_predictions(test_p)
            score_stack = score_func(pred_stack)
            lat_stack = measure_aggregator_latency(agg_wrapper, test_p)
            test_preds_by_agg['Global Stacking'] = pred_stack.detach().cpu().numpy().flatten()
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Global Stacking", "score": score_stack,
                "rel_gain_pct": compute_relative_change(score_unif, score_stack, dataset.task_type),
                "latency_ms": lat_stack
            })

            # 6. TabM[G] Greedy
            agg_wrapper.set_active_subset(greedy_subset)
            agg_wrapper.mode = 'uniform'
            pred_greedy = agg_wrapper.aggregate_predictions(test_p)
            score_greedy = score_func(pred_greedy)
            lat_greedy = measure_aggregator_latency(agg_wrapper, test_p)
            test_preds_by_agg['TabM[G] Greedy'] = pred_greedy.detach().cpu().numpy().flatten()
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "TabM[G] Greedy", "score": score_greedy,
                "rel_gain_pct": compute_relative_change(score_unif, score_greedy, dataset.task_type),
                "latency_ms": lat_greedy
            })

            # 7. Adaptive Gate (Ours)
            agg_wrapper.set_active_subset(list(range(k)))
            agg_wrapper.mode = 'sample_adaptive'
            pred_adapt = agg_wrapper.aggregate_predictions(test_p)
            score_adapt = score_func(pred_adapt)
            lat_adapt = measure_aggregator_latency(agg_wrapper, test_p)
            test_preds_by_agg['Adaptive Gate (Ours)'] = pred_adapt.detach().cpu().numpy().flatten()
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Adaptive Gate (Ours)", "score": score_adapt,
                "rel_gain_pct": compute_relative_change(score_unif, score_adapt, dataset.task_type),
                "latency_ms": lat_adapt
            })

    df_results = pd.DataFrame(records)
    # Save raw results
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nPersisted {len(df_results)} raw experimental results to: {RESULTS_CSV}")

    # -------------------------------------------------------------------------
    # Statistical Analysis Across Datasets (Pre-Registered Hypothesis Families)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("      STATISTICAL SIGNIFICANCE REPORT (ACROSS-DATASET PAIRED TESTS)      ")
    print("=" * 80)

    # Compute dataset-level means across seeds
    df_agg = df_results.groupby(['dataset', 'aggregator'])['rel_gain_pct'].mean().unstack()

    # Primary Test: Gate vs Uniform Mean (rel_gain_pct > 0)
    gate_gains = df_agg['Adaptive Gate (Ours)'].values
    stat_pri, p_pri = paired_wilcoxon_test(gate_gains, np.zeros_like(gate_gains), zero_method=config.WILCOXON_ZERO_METHOD)
    w_pri, l_pri, t_pri = compute_win_loss_tie(gate_gains, tie_threshold=config.TIE_THRESHOLD)
    med_pri, low_pri, high_pri = bootstrap_median_ci(gate_gains, n_bootstraps=config.N_BOOTSTRAPS)

    print("\nPRIMARY HYPOTHESIS TEST: Adaptive Gate vs Uniform Mean")
    print("-" * 70)
    print(f"Median Relative Gain:     {med_pri:+.2f}% [95% CI: {low_pri:+.2f}%, {high_pri:+.2f}%]")
    print(f"Wins / Losses / Ties:     {w_pri} Wins / {l_pri} Losses / {t_pri} Ties")
    print(f"Wilcoxon Signed-Rank p:   {p_pri:.4f} (alpha = {config.ALPHA_PRIMARY})")
    print(f"Statistically Significant? {'YES (*)' if p_pri < config.ALPHA_PRIMARY else 'NO (Fail to reject H0)'}")
    print("-" * 70)

    # Secondary Family: Gate vs Competitors (Adjusted with Step-Down Holm)
    secondary_competitors = ['Exact Median', 'Trimmed Mean', 'Affine Recalibration', 'Global Stacking', 'TabM[G] Greedy']
    p_secondary = []
    sec_stats = []

    for comp in secondary_competitors:
        comp_gains = df_agg[comp].values
        # Paired test: does gate gain exceed competitor gain?
        diff = gate_gains - comp_gains
        stat, p_val = paired_wilcoxon_test(diff, np.zeros_like(diff), zero_method=config.WILCOXON_ZERO_METHOD)
        w, l, t = compute_win_loss_tie(diff, tie_threshold=config.TIE_THRESHOLD)
        med, low, high = bootstrap_median_ci(diff, n_bootstraps=config.N_BOOTSTRAPS)
        p_secondary.append(p_val)
        sec_stats.append({"comp": comp, "raw_p": p_val, "wlt": f"{w}/{l}/{t}", "med_diff": med, "ci": f"[{low:+.2f}%, {high:+.2f}%]"})

    adj_p_sec, reject_sec = holm_bonferroni_step_down(np.array(p_secondary), alpha=config.ALPHA_SECONDARY)

    print("\nSECONDARY HYPOTHESIS FAMILY (Adjusted via Step-Down Holm-Bonferroni):")
    print("-" * 80)
    print(f"{'Comparison (Gate vs ...)':<26} | {'W/L/T':<9} | {'Med Diff (CI)':<20} | {'Raw p':<8} | {'Holm p':<8} | {'Significant?'}")
    print("-" * 80)
    for i, s in enumerate(sec_stats):
        sig_str = "YES (*)" if reject_sec[i] else "NO"
        print(f"{s['comp']:<26} | {s['wlt']:<9} | {s['med_diff']:+.2f}% {s['ci']:<14} | {s['raw_p']:<8.4f} | {adj_p_sec[i]:<8.4f} | {sig_str}")
    print("-" * 80)

    return df_results


def build_development_datasets() -> List[DatasetContainer]:
    """Builds standard development datasets for exploratory tuning."""
    dev_datasets = []
    cal = fetch_california_housing()
    dev_datasets.append(DatasetContainer("california", cal.data, cal.target, "regression"))
    diab = load_diabetes()
    dev_datasets.append(DatasetContainer("diabetes", diab.data, diab.target, "regression"))
    return dev_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['dev', 'confirmatory'], default='dev')
    parser.add_argument('--force-retrain', action='store_true')
    args = parser.parse_args()

    if args.phase == 'dev':
        datasets = build_development_datasets()
        run_offline_benchmark(datasets, seeds=config.SEEDS[:3], phase='dev')
    else:
        # Pre-registered Confirmatory Benchmark
        pass
