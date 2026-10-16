"""
Evaluates all post-hoc aggregators on cached head_predictions.npz for California across seeds 0..4.
Computes seed-wise test RMSE, mean +- std, and relative deltas vs uniform baseline.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.gated_tabm import (
    PostHocTabMAggregator,
    select_greedy_heads_tabm,
    select_greedy_heads_repo_variant
)

def evaluate_california_dev():
    output_base = Path("tabm/paper/exp/tabm/california/reproduce_tuned")
    seeds = [0, 1, 2, 3, 4]
    
    ready_seeds = [s for s in seeds if (output_base / str(s) / "head_predictions.npz").exists()]
    if not ready_seeds:
        print("[!] No cached predictions found yet.")
        return
        
    print(f"[*] Evaluating ready seeds: {ready_seeds}")
    
    data_dir = Path("tabm/paper/data/california")
    y_val_raw = np.load(data_dir / "Y_val.npy").astype(np.float32)
    y_test_raw = np.load(data_dir / "Y_test.npy").astype(np.float32)
    
    results = {
        'uniform': [],
        'median': [],
        'trimmed_mean': [],
        'affine': [],
        'stacking': [],
        'tabm_g_d6': [],
        'tabm_g_repo': [],
        'adaptive_gate': []
    }
    
    for s in ready_seeds:
        seed_dir = output_base / str(s)
        preds_data = np.load(seed_dir / "head_predictions.npz")
        val_preds = preds_data['val']   # (N_val, 32)
        test_preds = preds_data['test'] # (N_test, 32)
        
        report_path = seed_dir / "report.json"
        with open(report_path) as f:
            report = json.load(f)
        paper_test_score = report['metrics']['test']['score']
        paper_test_rmse = -paper_test_score
        
        u_val = val_preds.mean(axis=1)
        u_test = test_preds.mean(axis=1)
        u_test_rmse = float(np.sqrt(mean_squared_error(y_test_raw, u_test)))
        results['uniform'].append(u_test_rmse)
        
        # 1. Median
        med_test = np.median(test_preds, axis=1)
        results['median'].append(float(np.sqrt(mean_squared_error(y_test_raw, med_test))))
        
        # 2. Trimmed Mean (tuned on val)
        best_p, best_val_err = 0.0, float('inf')
        for p in [0.05, 0.10, 0.15, 0.20]:
            k = int(np.floor(p * 32))
            sorted_v = np.sort(val_preds, axis=1)
            t_val = sorted_v[:, k:32-k].mean(axis=1) if k > 0 else sorted_v.mean(axis=1)
            err = np.sqrt(mean_squared_error(y_val_raw, t_val))
            if err < best_val_err:
                best_val_err = err
                best_p = p
        k_opt = int(np.floor(best_p * 32))
        sorted_t = np.sort(test_preds, axis=1)
        t_test = sorted_t[:, k_opt:32-k_opt].mean(axis=1) if k_opt > 0 else sorted_t.mean(axis=1)
        results['trimmed_mean'].append(float(np.sqrt(mean_squared_error(y_test_raw, t_test))))
        
        # 3. Affine Recalibration (val)
        A = np.column_stack([u_val, np.ones_like(u_val)])
        params, _, _, _ = np.linalg.lstsq(A, y_val_raw, rcond=None)
        a, b = params[0], params[1]
        aff_test = a * u_test + b
        results['affine'].append(float(np.sqrt(mean_squared_error(y_test_raw, aff_test))))
        
        # 4. Global Stacking (SLSQP on simplex)
        agg = PostHocTabMAggregator(k=32, task_type='regression', mode='global_stacking')
        agg.fit_global_stacking(
            torch.tensor(val_preds, dtype=torch.float32),
            torch.tensor(y_val_raw, dtype=torch.float32)
        )
        stk_test = agg.aggregate_predictions(torch.tensor(test_preds, dtype=torch.float32)).numpy()
        results['stacking'].append(float(np.sqrt(mean_squared_error(y_test_raw, stk_test))))
        
        # 5. TabM[G] D.6 with best-prefix
        sel_d6 = select_greedy_heads_tabm(
            torch.tensor(val_preds, dtype=torch.float32),
            torch.tensor(y_val_raw, dtype=torch.float32),
            task_type='regression'
        )
        d6_test = test_preds[:, sel_d6].mean(axis=1)
        results['tabm_g_d6'].append(float(np.sqrt(mean_squared_error(y_test_raw, d6_test))))
        
        # 6. TabM[G] Repo variant
        sel_repo = select_greedy_heads_repo_variant(
            torch.tensor(val_preds, dtype=torch.float32),
            torch.tensor(y_val_raw, dtype=torch.float32),
            task_type='regression'
        )
        repo_test = test_preds[:, sel_repo].mean(axis=1)
        results['tabm_g_repo'].append(float(np.sqrt(mean_squared_error(y_test_raw, repo_test))))
        
        # 7. Adaptive Gate (with inner CV)
        gate = PostHocTabMAggregator(k=32, task_type='regression', mode='sample_adaptive', seed=s)
        gate.fit_sample_adaptive_gate_with_cv(
            val_preds=torch.tensor(val_preds, dtype=torch.float32),
            y_val=torch.tensor(y_val_raw, dtype=torch.float32),
            seed=s
        )
        gate_test = gate.aggregate_predictions(torch.tensor(test_preds, dtype=torch.float32)).detach().numpy()
        results['adaptive_gate'].append(float(np.sqrt(mean_squared_error(y_test_raw, gate_test))))
        
        print(f"Seed {s}: Uniform RMSE = {u_test_rmse:.5f} (Report = {paper_test_rmse:.5f}), Stacking = {results['stacking'][-1]:.5f}, TabM[G] = {results['tabm_g_d6'][-1]:.5f}, Gate = {results['adaptive_gate'][-1]:.5f}", flush=True)

    print("\n" + "=" * 90)
    print(f"{'Aggregator':<28} | {'Mean Test RMSE':<18} | {'Rel Gain vs Uniform (%)':<25}")
    print("-" * 90)
    u_mean = np.mean(results['uniform'])
    for name, vals in results.items():
        m = np.mean(vals)
        gain = ((u_mean - m) / u_mean) * 100.0
        print(f"{name:<28} | {m:.5f} +- {np.std(vals):.5f} | {gain:+.3f}%")
    print("=" * 90)

if __name__ == '__main__':
    evaluate_california_dev()
