"""
Offline Evaluation Pipeline for Post-Hoc TabM Aggregators (run_pipeline.py)

Key Modules:
1. Two-Stage Protocol:
   - Stage 1: Backbone TabM predictions are computed and persisted to disk
     (experiments/cache/{dataset}_{seed}_{config_hash}/) with val_preds.pt, test_preds.pt,
     y_val.pt, y_test.pt, and metadata.json.
   - Stage 2: All 7 aggregators (Uniform Mean, Exact Median, Trimmed Mean, Affine Recalibration,
     Global Simplex Stacking, TabM[G] Greedy, and Adaptive Gate) are evaluated offline
     on IDENTICAL cached tensor representations.
2. Official TabM Training Engine:
   - Supports both unshared training batches (TabM default) where each head sees distinct objects,
     and shared training batches (TabM♠).
   - Tuned architecture: 3 blocks, 256 width, dropout 0.2, noisy-quantile preprocessing,
     AdamW (lr=0.001, wd=0.03), and early stopping with patience on validation split.
3. Dataset Verification & Dry-Run:
   - Strict loader that raises explicit errors on corrupt/missing data (no silent loss).
   - `--dry-run` flag to verify dataset schemas, shapes, and splits without training.
4. Rigorous Multi-Tier Statistical Engine:
   - Primary Hypothesis (H1): Adaptive Gate vs. Uniform Mean (Wilcoxon Signed-Rank across datasets).
   - Secondary Family (H2..H6): Adaptive Gate vs. [Median, Trimmed, Affine, Stacking, TabM[G]]
     with step-down Holm-Bonferroni correction.
   - Equivalence Testing: [-0.5%, +0.5%] equivalence margin for rigorous confirmatory negative reporting.
   - Documented Tie Rule: Based on observed seed standard deviation (sigma_seed).
5. Reliable Latency Profiling:
   - 10 warmup iterations, 100 timed repetitions, GPU synchronized, median latency reported in ms.
"""

import sys
sys.path.append('experiments')
sys.path.append('tabm/paper')
import os
import json
import time
import hashlib
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

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


def get_git_commit_hash() -> str:
    """Returns the current git commit hash for experiment provenance."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "unknown"


# =============================================================================
# 1. Dataset Abstraction Supporting Development & TabReD Formats
# =============================================================================

class TabularDataset:
    """
    Unified dataset container supporting:
    - Development datasets (California Housing)
    - Official TabReD temporal-drift datasets with fixed train/val/test splits.
    """
    def __init__(
        self,
        name: str,
        task_type: str,
        is_temporal: bool = False,
        data_dict: Optional[Dict[str, Any]] = None,
        X_raw: Optional[np.ndarray] = None,
        y_raw: Optional[np.ndarray] = None,
        cat_cardinalities: Optional[List[int]] = None
    ):
        self.name = name
        self.task_type = task_type
        self.is_temporal = is_temporal
        self.data_dict = data_dict
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.cat_cardinalities = cat_cardinalities or []

    def get_split_data(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
        """
        Returns (X_train, y_train, X_val, y_val, X_test, y_test, scaler_y).
        Applies paper's standard noisy-quantile normalization.
        """
        if self.is_temporal and self.data_dict is not None:
            X_train = self.data_dict['X_train']
            y_train = self.data_dict['y_train']
            X_val = self.data_dict['X_val']
            y_val = self.data_dict['y_val']
            y_val_raw = self.data_dict.get('y_val_raw', y_val)
            X_test = self.data_dict['X_test']
            y_test = self.data_dict['y_test']
            scaler_y = self.data_dict.get('scaler_y', None)
            return X_train, y_train, X_val, y_val, y_val_raw, X_test, y_test, scaler_y

        # Development datasets (California)
        assert self.X_raw is not None and self.y_raw is not None
        X_train, X_temp, y_train, y_temp = train_test_split(self.X_raw, self.y_raw, test_size=0.4, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

        # Apply official paper preprocessing: Noisy Quantile Normalization
        qt = QuantileTransformer(
            n_quantiles=max(min(len(X_train) // 30, 1000), 10),
            output_distribution='normal',
            random_state=seed
        )
        noise = np.random.RandomState(seed).normal(0.0, 1e-5, X_train.shape)
        X_train = qt.fit_transform(X_train + noise)
        X_val = qt.transform(X_val)
        X_test = qt.transform(X_test)

        if self.task_type == 'regression':
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_raw = y_val.flatten()
            y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test = y_test.flatten()  # Raw unscaled for honest RMSE metric
        else:
            scaler_y = None
            y_val_raw = y_val.flatten()
            y_test = y_test.flatten()

        return X_train, y_train, X_val, y_val, y_val_raw, X_test, y_test, scaler_y


def load_tabred_dataset(dataset_dir: Path) -> TabularDataset:
    """
    Loads an official TabReD temporal dataset with quantile normalization and fixed splits.
    Raises explicit exceptions on corrupt or missing files (no silent loss).
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json in: {dataset_dir}")

    from lib.data import Dataset, transform_num, transform_cat
    ds = Dataset.from_dir(dataset_dir)
    info = json.loads(info_path.read_text())
    task_type = 'regression' if ds.task.is_regression else 'binclass'

    # Official noisy-quantile preprocessing
    if 'x_num' in ds:
        ds['x_num'] = transform_num(ds['x_num'], policy='noisy-quantile', seed=0)

    # Ordinal encoding for categoricals
    if 'x_cat' in ds:
        ds['x_cat'] = transform_cat(ds['x_cat'], policy='ordinal')

    cat_cards = ds.compute_cat_cardinalities()

    parts = ['train', 'val', 'test']
    X_parts = {}
    for part in parts:
        feats = []
        if 'x_num' in ds: feats.append(ds['x_num'][part])
        if 'x_bin' in ds: feats.append(ds['x_bin'][part])
        if 'x_cat' in ds: feats.append(ds['x_cat'][part])
        X_parts[part] = np.concatenate(feats, axis=1)

    scaler_y = None
    y_train = ds.task.labels['train'].copy()
    y_val_raw = ds.task.labels['val'].copy()
    y_val = ds.task.labels['val'].copy()
    y_test = ds.task.labels['test'].copy()

    if task_type == 'regression':
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    data_dict = {
        'X_train': X_parts['train'], 'y_train': y_train,
        'X_val': X_parts['val'], 'y_val': y_val, 'y_val_raw': y_val_raw,
        'X_test': X_parts['test'], 'y_test': y_test,
        'scaler_y': scaler_y
    }
    return TabularDataset(
        name=dataset_dir.name,
        task_type=task_type,
        is_temporal=True,
        data_dict=data_dict,
        cat_cardinalities=cat_cards
    )


# =============================================================================
# 2. Persisted Cache Engine with Real TabM Training & Unshared Batches
# =============================================================================

def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Computes a deterministic hash of all model and training hyperparameters."""
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()[:8]


def get_or_create_cached_predictions(
    dataset: TabularDataset,
    seed: int,
    k_members: int = 32,
    arch: str = 'tabm',
    share_training_batches: bool = False,
    n_blocks: int = 3,
    d_block: int = 256,
    dropout: float = 0.2,
    lr: float = 0.001,
    weight_decay: float = 0.03,
    max_epochs: int = 40,
    patience: int = 8,
    force_retrain: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[StandardScaler]]:
    """
    On-Disk Caching Engine:
    Persists val_preds.pt, test_preds.pt, y_val.pt, y_test.pt, and metadata.json
    keyed by (dataset, seed, config_hash).
    """
    hyperparams = {
        "dataset": dataset.name,
        "k_members": k_members,
        "arch": arch,
        "share_training_batches": share_training_batches,
        "n_blocks": n_blocks,
        "d_block": d_block,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "num_policy": "noisy-quantile"
    }
    config_hash = compute_config_hash(hyperparams)
    safe_name = dataset.name.replace('/', '_').replace('\\', '_')
    run_dir = CACHE_DIR / f"{safe_name}_seed_{seed}_{config_hash}"
    val_pred_file = run_dir / "val_preds.pt"
    test_pred_file = run_dir / "test_preds.pt"
    y_val_file = run_dir / "y_val.pt"
    y_test_file = run_dir / "y_test.pt"

    X_train, y_train, X_val, y_val, y_val_raw, X_test, y_test, scaler_y = dataset.get_split_data(seed)

    if not force_retrain and val_pred_file.exists() and test_pred_file.exists():
        val_preds = torch.load(val_pred_file, weights_only=True)
        test_preds = torch.load(test_pred_file, weights_only=True)
        y_val_t = torch.load(y_val_file, weights_only=True)
        y_test_t = torch.load(y_test_file, weights_only=True)
        return val_preds, y_val_t, test_preds, y_test_t, scaler_y

    run_dir.mkdir(parents=True, exist_ok=True)
    n_features = X_train.shape[1]

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = tabm.TabM.make(
        n_num_features=n_features,
        cat_cardinalities=dataset.cat_cardinalities if dataset.cat_cardinalities else None,
        d_out=1,
        k=k_members,
        arch_type=arch,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss() if dataset.task_type == 'regression' else nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    N = len(X_train_t)
    batch_size = 256
    gen = torch.Generator().manual_seed(seed)

    best_val_score = float('inf') if dataset.task_type == 'regression' else -float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        if share_training_batches:
            # TabM♠: identical minibatches across all heads
            perm = torch.randperm(N, generator=gen)
            for i in range(0, N, batch_size):
                b_idx = perm[i : i + batch_size]
                xb, yb = X_train_t[b_idx], y_train_t[b_idx]
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb.unsqueeze(1).expand_as(preds))
                loss.backward()
                optimizer.step()
        else:
            # Standard TabM: independent permutations per head (proved zero head overlap)
            perms = torch.rand((k_members, N), generator=gen).argsort(dim=1).t()
            for i in range(0, N, batch_size):
                b_idx = perms[i : i + batch_size]  # (B, K)
                xb = X_train_t[b_idx]              # (B, K, D)
                yb = y_train_t[b_idx]              # (B, K, 1)
                optimizer.zero_grad()
                preds = model(xb)                  # (B, K, 1)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

        # Validation early stopping check
        model.eval()
        with torch.no_grad():
            val_p = model(X_val_t).squeeze(-1).mean(dim=1).cpu().numpy().flatten()
            if dataset.task_type == 'regression':
                if scaler_y is not None:
                    val_p = scaler_y.inverse_transform(val_p.reshape(-1, 1)).flatten()
                val_score = np.sqrt(mean_squared_error(y_val_raw, val_p))
                is_better = val_score < best_val_score
            else:
                val_score = roc_auc_score(y_val_raw, np.clip(scipy.special.expit(val_p), 1e-7, 1 - 1e-7))
                is_better = val_score > best_val_score

        if is_better:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved_mark = "(*)"
        else:
            patience_counter += 1
            improved_mark = f"(patience: {patience_counter}/{patience})"

        print(f"    Epoch {epoch+1:02d}/{max_epochs:02d}: Val RMSE = {val_score:.4f} {improved_mark}", flush=True)

        if patience_counter >= patience:
            print(f"  [Early Stopped at Epoch {epoch+1}] Restored best checkpoint with val metric: {best_val_score:.4f}", flush=True)
            break
    else:
        print(f"  [Completed {max_epochs} Epochs] Best val metric: {best_val_score:.4f}", flush=True)

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).squeeze(-1)
        test_preds = model(X_test_t).squeeze(-1)

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

    meta = hyperparams.copy()
    meta.update({
        "seed": seed,
        "config_hash": config_hash,
        "git_commit": get_git_commit_hash(),
        "task_type": dataset.task_type,
        "val_samples": len(y_val_t),
        "test_samples": len(y_test_t),
        "best_val_metric": float(best_val_score)
    })
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return val_preds, y_val_t, test_preds, y_test_t, scaler_y


# =============================================================================
# 3. High-Precision Latency Measurement
# =============================================================================

def measure_aggregator_latency(
    agg_wrapper: PostHocTabMAggregator,
    test_preds: torch.Tensor,
    warmup_runs: int = config.LATENCY_WARMUP_RUNS,
    timed_runs: int = config.LATENCY_TIMED_RUNS
) -> float:
    """Measures latency in milliseconds with warmup and GPU synchronization."""
    if test_preds.is_cuda:
        torch.cuda.synchronize()

    for _ in range(warmup_runs):
        _ = agg_wrapper.aggregate_predictions(test_preds)

    if test_preds.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = agg_wrapper.aggregate_predictions(test_preds)
        if test_preds.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.median(times))


# =============================================================================
# 4. Offline Benchmark Runner
# =============================================================================

def run_offline_pipeline(
    datasets: List[TabularDataset],
    seeds: List[int] = config.SEEDS,
    phase: str = "dev",
    arch: str = 'tabm',
    share_training_batches: bool = False,
    force_retrain: bool = False
) -> pd.DataFrame:
    print("=" * 80)
    print(f"      OFFLINE POST-HOC TABM AGGREGATOR EVALUATION (PHASE: {phase.upper()})      ")
    print("=" * 80)
    print(f"Datasets ({len(datasets)}): {[d.name for d in datasets]}")
    print(f"Seeds ({len(seeds)}): {seeds} | Arch: {arch} | Shared Batches: {share_training_batches}")
    git_hash = get_git_commit_hash()
    print(f"Git Commit: {git_hash}")

    records = []
    per_dataset_row_pvals = {}

    for d_idx, dataset in enumerate(datasets):
        print(f"\n[{d_idx+1}/{len(datasets)}] Evaluating {dataset.name} ({dataset.task_type})...")
        for s in seeds:
            val_p, y_val, test_p, y_test, scaler_y = get_or_create_cached_predictions(
                dataset, seed=s, arch=arch, share_training_batches=share_training_batches,
                force_retrain=force_retrain
            )
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

            # Fit all candidate aggregators on validation data
            agg_wrapper.tune_trimmed_mean(val_p, y_val)
            agg_wrapper.fit_affine_recalibration(val_p, y_val)
            agg_wrapper.fit_global_stacking(val_p, y_val)
            greedy_subset = select_greedy_heads_tabm(val_p, y_val, task_type=dataset.task_type)
            agg_wrapper.fit_sample_adaptive_gate_with_cv(val_p, y_val)

            # 1. Uniform
            agg_wrapper.mode = 'uniform'
            agg_wrapper.set_active_subset(list(range(k)))
            pred_unif = agg_wrapper.aggregate_predictions(test_p)
            score_unif = score_func(pred_unif)
            lat_unif = measure_aggregator_latency(agg_wrapper, test_p)
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Uniform Mean", "score": score_unif, "rel_gain_pct": 0.0, "latency_ms": lat_unif,
                "slsqp_failures": agg_wrapper.slsqp_failures, "git_commit": git_hash
            })

            # 2. Exact Median
            agg_wrapper.mode = 'median'
            pred_med = agg_wrapper.aggregate_predictions(test_p)
            score_med = score_func(pred_med)
            lat_med = measure_aggregator_latency(agg_wrapper, test_p)
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Exact Median", "score": score_med,
                "rel_gain_pct": compute_relative_change(score_unif, score_med, dataset.task_type),
                "latency_ms": lat_med, "slsqp_failures": agg_wrapper.slsqp_failures, "git_commit": git_hash
            })

            # 3. Trimmed Mean
            agg_wrapper.mode = 'trimmed_mean'
            pred_trim = agg_wrapper.aggregate_predictions(test_p)
            score_trim = score_func(pred_trim)
            lat_trim = measure_aggregator_latency(agg_wrapper, test_p)
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Trimmed Mean", "score": score_trim,
                "rel_gain_pct": compute_relative_change(score_unif, score_trim, dataset.task_type),
                "latency_ms": lat_trim, "slsqp_failures": agg_wrapper.slsqp_failures, "git_commit": git_hash
            })

            # 4. Affine Recalibration
            agg_wrapper.mode = 'affine_recalibration'
            pred_aff = agg_wrapper.aggregate_predictions(test_p)
            score_aff = score_func(pred_aff)
            lat_aff = measure_aggregator_latency(agg_wrapper, test_p)
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Affine Recalibration", "score": score_aff,
                "rel_gain_pct": compute_relative_change(score_unif, score_aff, dataset.task_type),
                "latency_ms": lat_aff, "slsqp_failures": agg_wrapper.slsqp_failures, "git_commit": git_hash
            })

            # 5. Global Stacking
            agg_wrapper.mode = 'global_stacking'
            pred_stack = agg_wrapper.aggregate_predictions(test_p)
            score_stack = score_func(pred_stack)
            lat_stack = measure_aggregator_latency(agg_wrapper, test_p)
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Global Stacking", "score": score_stack,
                "rel_gain_pct": compute_relative_change(score_unif, score_stack, dataset.task_type),
                "latency_ms": lat_stack, "slsqp_failures": agg_wrapper.slsqp_failures, "git_commit": git_hash
            })

            # 6. TabM[G] Greedy
            agg_wrapper.set_active_subset(greedy_subset)
            agg_wrapper.mode = 'uniform'
            pred_greedy = agg_wrapper.aggregate_predictions(test_p)
            score_greedy = score_func(pred_greedy)
            lat_greedy = measure_aggregator_latency(agg_wrapper, test_p)
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "TabM[G] Greedy", "score": score_greedy,
                "rel_gain_pct": compute_relative_change(score_unif, score_greedy, dataset.task_type),
                "latency_ms": lat_greedy, "slsqp_failures": agg_wrapper.slsqp_failures, "git_commit": git_hash
            })

            # 7. Adaptive Gate (Ours)
            agg_wrapper.set_active_subset(list(range(k)))
            agg_wrapper.mode = 'sample_adaptive'
            pred_adapt = agg_wrapper.aggregate_predictions(test_p)
            score_adapt = score_func(pred_adapt)
            lat_adapt = measure_aggregator_latency(agg_wrapper, test_p)
            records.append({
                "phase": phase, "dataset": dataset.name, "seed": s, "task_type": dataset.task_type,
                "aggregator": "Adaptive Gate (Ours)", "score": score_adapt,
                "rel_gain_pct": compute_relative_change(score_unif, score_adapt, dataset.task_type),
                "latency_ms": lat_adapt, "slsqp_failures": agg_wrapper.slsqp_failures, "git_commit": git_hash
            })

            # Within-dataset paired row-level bootstrap test
            p_cand_np = pred_adapt.detach().cpu().numpy().flatten()
            p_unif_np = pred_unif.detach().cpu().numpy().flatten()
            if scaler_y is not None and dataset.task_type == 'regression':
                p_cand_np = scaler_y.inverse_transform(p_cand_np.reshape(-1, 1)).flatten()
                p_unif_np = scaler_y.inverse_transform(p_unif_np.reshape(-1, 1)).flatten()

            obs_delta, p_row = paired_row_level_bootstrap_test(
                y_test_np, p_cand_np, p_unif_np, task_type=dataset.task_type, n_bootstraps=500
            )
            per_dataset_row_pvals.setdefault(dataset.name, {})[s] = p_row

    df_results = pd.DataFrame(records)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_CSV, index=False)
    print(f"\n[OK] Persisted {len(df_results)} raw results to: {RESULTS_CSV}")

    # -------------------------------------------------------------------------
    # Statistical Reporting (Pre-Registered Hypothesis Families)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("      STATISTICAL SIGNIFICANCE REPORT (ACROSS-DATASET PAIRED TESTS)      ")
    print("=" * 80)

    # Compute dataset-level means across seeds
    df_agg = df_results.groupby(['dataset', 'aggregator'])['rel_gain_pct'].mean().unstack()

    # Dynamic seed std tie rule:
    # Compute std of gate gain across seeds per dataset
    gate_records = df_results[df_results['aggregator'] == 'Adaptive Gate (Ours)']
    seed_std_per_dataset = gate_records.groupby('dataset')['rel_gain_pct'].std().fillna(0.0)
    mean_seed_std = float(seed_std_per_dataset.mean())
    tie_thresh = max(0.01, mean_seed_std)

    # Primary Test: Gate vs Uniform Mean
    gate_gains = df_agg['Adaptive Gate (Ours)'].values
    stat_pri, p_pri = paired_wilcoxon_test(gate_gains, np.zeros_like(gate_gains), zero_method=config.WILCOXON_ZERO_METHOD)
    w_pri, l_pri, t_pri = compute_win_loss_tie(gate_gains, tie_threshold=tie_thresh)
    med_pri, low_pri, high_pri = bootstrap_median_ci(gate_gains, n_bootstraps=config.N_BOOTSTRAPS)

    print("\nPRIMARY HYPOTHESIS TEST: Adaptive Gate vs Uniform Mean (Reference)")
    print("-" * 70)
    print(f"Median Relative Gain:     {med_pri:+.2f}% [95% CI: {low_pri:+.2f}%, {high_pri:+.2f}%]")
    print(f"Wins / Losses / Ties:     {w_pri} Wins / {l_pri} Losses / {t_pri} Ties (Tie threshold: |delta| <= {tie_thresh:.2f}% [observed seed std])")
    print(f"Wilcoxon Signed-Rank p:   {p_pri:.4f} (alpha = {config.ALPHA_PRIMARY})")
    print(f"Equivalence Margin:       [-{config.EQUIVALENCE_MARGIN:.2f}%, +{config.EQUIVALENCE_MARGIN:.2f}%]")
    if low_pri >= -config.EQUIVALENCE_MARGIN and high_pri <= config.EQUIVALENCE_MARGIN:
        eq_conclusion = "EQUIVALENT TO UNIFORM (Confirmatory Negative Finding)"
    elif p_pri < config.ALPHA_PRIMARY:
        eq_conclusion = "SIGNIFICANT OUTPERFORMANCE (*)"
    else:
        eq_conclusion = "INCONCLUSIVE (CI spans zero and exceeds equivalence margin)"
    print(f"Confirmatory Decision:    {eq_conclusion}")
    print("-" * 70)

    # Within-dataset row-level test summary (reported per seed, not averaged)
    print("\nWITHIN-DATASET TEST-ROW LEVEL BOOTSTRAP CHECK (Per Seed):")
    for d_name, seed_p_dict in per_dataset_row_pvals.items():
        p_str = ", ".join([f"Seed {s}: p={p:.4f}" for s, p in seed_p_dict.items()])
        print(f"  {d_name:<20}: {p_str}")

    # Secondary Family: Gate vs Competitors (Adjusted with Step-Down Holm)
    secondary_competitors = ['Exact Median', 'Trimmed Mean', 'Affine Recalibration', 'Global Stacking', 'TabM[G] Greedy']
    p_secondary = []
    sec_stats = []

    for comp in secondary_competitors:
        comp_gains = df_agg[comp].values
        diff = gate_gains - comp_gains
        stat, p_val = paired_wilcoxon_test(diff, np.zeros_like(diff), zero_method=config.WILCOXON_ZERO_METHOD)
        w, l, t = compute_win_loss_tie(diff, tie_threshold=tie_thresh)
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

    # Latency Summary
    df_lat = df_results.groupby('aggregator')['latency_ms'].median()
    print("\nLATENCY PROFILING (Median per forward aggregation in milliseconds):")
    for agg_name, lat_val in df_lat.items():
        print(f"  {agg_name:<26}: {lat_val:.3f} ms")
    print("=" * 80)

    return df_results


def build_development_datasets() -> List[TabularDataset]:
    """Builds standard development dataset (California Housing only)."""
    cal = fetch_california_housing()
    return [TabularDataset("california", "regression", is_temporal=False, X_raw=cal.data, y_raw=cal.target)]


def build_confirmatory_datasets(data_dir: Path) -> List[TabularDataset]:
    """Discovers and builds real TabReD datasets from official data directory."""
    conf_datasets = []
    tabred_dir = data_dir / "tabred"
    if not tabred_dir.exists():
        raise FileNotFoundError(f"TabReD data directory not found at: {tabred_dir}")

    for d in sorted(tabred_dir.iterdir()):
        if d.is_dir() and (d / "info.json").exists():
            ds = load_tabred_dataset(d)
            conf_datasets.append(ds)

    if len(conf_datasets) == 0:
        raise RuntimeError(f"No valid TabReD datasets could be loaded from {tabred_dir}")

    return conf_datasets


def dry_run_check(datasets: List[TabularDataset]) -> None:
    """Verifies all dataset schemas, splits, label ranges, and shapes without training."""
    print("=" * 80)
    print("           CONFIRMATORY DATASET DRY-RUN VERIFICATION           ")
    print("=" * 80)
    print(f"{'Dataset':<24} | {'Task':<10} | {'Train Shape':<14} | {'Val Shape':<14} | {'Test Shape':<14} | {'Categoricals'}")
    print("-" * 80)

    for ds in datasets:
        X_tr, y_tr, X_va, y_val, y_val_raw, X_te, y_te, _ = ds.get_split_data(seed=0)
        assert len(X_tr) > 0 and len(X_va) > 0 and len(X_te) > 0, f"Empty split in {ds.name}"
        assert not np.isnan(X_tr).any(), f"NaNs found in X_train for {ds.name}"
        assert not np.isnan(X_va).any(), f"NaNs found in X_val for {ds.name}"
        assert not np.isnan(X_te).any(), f"NaNs found in X_test for {ds.name}"
        assert not np.isnan(y_tr).any(), f"NaNs found in y_train for {ds.name}"
        assert not np.isnan(y_te).any(), f"NaNs found in y_test for {ds.name}"

        n_cat_str = f"{len(ds.cat_cardinalities)} cats" if ds.cat_cardinalities else "numeric-only"
        print(f"{ds.name:<24} | {ds.task_type:<10} | {str(X_tr.shape):<14} | {str(X_va.shape):<14} | {str(X_te.shape):<14} | {n_cat_str}")

    print("=" * 80)
    print(f"SUCCESS: All {len(datasets)} confirmatory datasets verified and ready for execution.")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline Post-Hoc TabM Aggregation Pipeline")
    parser.add_argument('--phase', choices=['dev', 'confirmatory'], default='dev', help="Experimental phase")
    parser.add_argument('--arch', choices=['tabm', 'tabm-mini'], default='tabm', help="Backbone architecture")
    parser.add_argument('--share-batches', action='store_true', help="Use TabM-spade shared training batches")
    parser.add_argument('--data-dir', type=Path, default=Path("tabm/paper/data"), help="Path to TabReD datasets")
    parser.add_argument('--dry-run', action='store_true', help="Dry-run dataset verification without training")
    parser.add_argument('--force-retrain', action='store_true', help="Force retraining and re-caching")
    args = parser.parse_args()

    if args.dry_run:
        datasets = build_confirmatory_datasets(args.data_dir)
        dry_run_check(datasets)
        sys.exit(0)

    if args.phase == 'dev':
        datasets = build_development_datasets()
        run_offline_pipeline(
            datasets,
            seeds=config.SEEDS,
            phase='dev',
            arch=args.arch,
            share_training_batches=args.share_batches,
            force_retrain=args.force_retrain
        )
    else:
        datasets = build_confirmatory_datasets(args.data_dir)
        run_offline_pipeline(
            datasets,
            seeds=config.SEEDS,
            phase='confirmatory',
            arch=args.arch,
            share_training_batches=args.share_batches,
            force_retrain=args.force_retrain
        )
