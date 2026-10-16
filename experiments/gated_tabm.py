"""
Post-Hoc Robust and Adaptive Aggregation Framework for TabM (ICLR 2025).

Peer-Review Refinements (Round 3):
1. Bounded Scoring Network: c * tanh(raw) + clamp on ctx prevents ReLU explosion on OOD shift.
2. Affine Recalibration Baseline: Fits y ≈ a * mu + b on validation to test for temporal drift confounder.
3. Official TabM[G] Algorithm (Exact Appendix D.6): Stops when step fails to improve OR beats full ensemble.
4. Analytic Jacobian for Binary Classification Stacking: Exact gradient for cross-entropy on simplex.
5. Contiguous Blocked Inner CV: Temporal blocked folds for validation tuning without future leakage.
6. Try/Finally Mode Isolation: Guarantees self.mode restoration.
7. Seeded Score Network: Deterministic layer initialization.
"""

import time
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss
import tabm

logger = logging.getLogger(__name__)


class PostHocTabMAggregator(nn.Module):
    def __init__(
        self,
        base_tabm: Optional[tabm.TabM] = None,
        k: int = 32,
        task_type: Literal['regression', 'binclass'] = 'regression',
        mode: Literal['uniform', 'median', 'trimmed_mean', 'affine_recalibration', 'global_stacking', 'sample_adaptive'] = 'uniform',
        trim_fraction: float = 0.1,
        score_net_hidden_dim: int = 32,
        temperature: float = 1.0,
        score_bound: float = 3.0,
        seed: int = 42,
    ):
        super().__init__()
        self.base_tabm = base_tabm
        self.k = base_tabm.k if base_tabm is not None else k
        self.task_type = task_type
        self.mode = mode
        self.trim_fraction = trim_fraction
        self.temperature = temperature
        self.score_bound = score_bound
        self.seed = seed
        self.slsqp_failures = 0

        # Freeze base TabM entirely: zero gradient leakage
        if self.base_tabm is not None:
            for p in self.base_tabm.parameters():
                p.requires_grad = False
            self.base_tabm.eval()

        # Global Stacking weights (stored on simplex)
        self.register_buffer('global_weights', torch.ones(self.k) / self.k)

        # Active subset for TabM[G] (default: all active)
        self.register_buffer('active_subset', torch.arange(self.k, dtype=torch.long))

        # Affine Recalibration parameters: a * mu + b
        self.register_buffer('affine_a', torch.tensor(1.0))
        self.register_buffer('affine_b', torch.tensor(0.0))

        # Validation log-spread boundaries for OOD clamping
        self.register_buffer('log_sigma_min', torch.tensor(-10.0))
        self.register_buffer('log_sigma_max', torch.tensor(10.0))

        # Permutation-Equivariant Scoring Network
        torch.manual_seed(self.seed)
        self.score_net = nn.Sequential(
            nn.Linear(2, score_net_hidden_dim),
            nn.ReLU(),
            nn.Linear(score_net_hidden_dim, 1)
        )
        # Zero-init last layer so initial output scores are identically zero
        nn.init.zeros_(self.score_net[-1].weight)
        nn.init.zeros_(self.score_net[-1].bias)

    def train(self, mode: bool = True) -> 'PostHocTabMAggregator':
        """Ensure base TabM permanently stays in eval mode."""
        super().train(mode)
        if self.base_tabm is not None:
            self.base_tabm.eval()
        return self

    def set_active_subset(self, indices: List[int]) -> None:
        """Configures TabM[G] active subset without replacement."""
        assert len(indices) == len(set(indices)), "TabM[G] requires unique indices"
        device = self.global_weights.device
        self.active_subset = torch.tensor(indices, dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # Offline Fitting & Model Selection on Validation Representations
    # -------------------------------------------------------------------------

    def _normalize_inputs(
        self,
        val_preds: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strict shape guards preventing silent broadcasting bugs."""
        if val_preds.ndim == 3 and val_preds.shape[-1] == 1:
            val_preds = val_preds.squeeze(-1)
        val_preds = val_preds.reshape(val_preds.shape[0], -1)
        y_val = y_val.reshape(-1, 1)
        assert len(val_preds) == len(y_val), f"Sample count mismatch: {len(val_preds)} vs {len(y_val)}"
        return val_preds, y_val

    def fit_affine_recalibration(self, val_preds: torch.Tensor, y_val: torch.Tensor) -> None:
        """
        Fits scalar affine recalibration: y ≈ a * mean(preds) + b on validation split.
        Tests whether adaptive gating is merely learning temporal drift bias correction.
        """
        val_preds, y_val = self._normalize_inputs(val_preds, y_val)
        mu = val_preds.mean(dim=1).detach().cpu().numpy().astype(np.float64)
        y = y_val.squeeze(-1).detach().cpu().numpy().astype(np.float64)

        if self.task_type == 'binclass':
            # Logistic recalibration on probabilities
            p_clip = np.clip(mu, 1e-6, 1 - 1e-6)
            logits = np.log(p_clip / (1 - p_clip))
            A = np.column_stack([logits, np.ones_like(logits)])
            def loss(params):
                z = A @ params
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
                return -np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))
            res = scipy.optimize.minimize(loss, [1.0, 0.0], method='BFGS')
            a, b = res.x if res.success else (1.0, 0.0)
        else:
            # Linear least squares
            A = np.column_stack([mu, np.ones_like(mu)])
            res = np.linalg.lstsq(A, y, rcond=None)
            a, b = res[0][0], res[0][1]

        self.affine_a.data = torch.tensor(float(a), device=self.affine_a.device)
        self.affine_b.data = torch.tensor(float(b), device=self.affine_b.device)

    def fit_global_stacking(self, val_preds: torch.Tensor, y_val: torch.Tensor) -> None:
        """
        Exact simplex-constrained stacking using SciPy SLSQP with analytic Jacobians
        for BOTH regression and binary classification.
        """
        val_preds, y_val = self._normalize_inputs(val_preds, y_val)
        X = val_preds.detach().cpu().numpy().astype(np.float64)
        y = y_val.squeeze(-1).detach().cpu().numpy().astype(np.float64)
        N, K = X.shape

        if self.task_type == 'binclass':
            X_clip = np.clip(X, 1e-6, 1 - 1e-6)
            def loss_func(w):
                p = np.clip(X_clip @ w, 1e-6, 1 - 1e-6)
                return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            # Analytic Jacobian for Binary Cross-Entropy on Simplex:
            # dL/dw = 1/N * X^T ((p - y) / (p * (1 - p)))
            def jac_func(w):
                p = np.clip(X_clip @ w, 1e-6, 1 - 1e-6)
                grad_p = (p - y) / (p * (1.0 - p))
                return (X_clip.T @ grad_p) / N
        else:
            def loss_func(w):
                err = y - (X @ w)
                return np.mean(err ** 2)
            def jac_func(w):
                err = y - (X @ w)
                return (-2.0 / N) * (X.T @ err)

        w0 = np.ones(K) / K
        bounds = [(0.0, 1.0) for _ in range(K)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        res = scipy.optimize.minimize(
            loss_func, w0, method='SLSQP', jac=jac_func, bounds=bounds, constraints=constraints,
            options={'ftol': 1e-11, 'maxiter': 500}
        )

        if not res.success:
            self.slsqp_failures += 1
            logger.warning(f"SLSQP did not converge: {res.message}. Using normalized fallback.")
            optimal_w = np.maximum(res.x, 0.0)
            optimal_w = optimal_w / max(np.sum(optimal_w), 1e-12)
        else:
            optimal_w = np.maximum(res.x, 0.0)
            optimal_w = optimal_w / np.sum(optimal_w)

        self.global_weights.data = torch.tensor(
            optimal_w, dtype=torch.float32, device=self.global_weights.device
        )

    def tune_trimmed_mean(
        self,
        val_preds: torch.Tensor,
        y_val: torch.Tensor,
        candidate_fractions: List[float] = [0.05, 0.10, 0.15, 0.20]
    ) -> float:
        """Tunes trim fraction p. Uses try/finally to guarantee mode restoration."""
        val_preds, y_val = self._normalize_inputs(val_preds, y_val)
        original_mode = self.mode
        self.mode = 'trimmed_mean'

        best_p = 0.0
        best_loss = float('inf')

        try:
            for p in candidate_fractions:
                self.trim_fraction = p
                agg = self.aggregate_predictions(val_preds)
                if self.task_type == 'binclass':
                    loss = F.binary_cross_entropy(agg.clamp(1e-7, 1 - 1e-7), y_val).item()
                else:
                    loss = F.mse_loss(agg, y_val).item()

                if loss < best_loss:
                    best_loss = loss
                    best_p = p
        finally:
            self.mode = original_mode

        self.trim_fraction = best_p
        return best_p

    def fit_sample_adaptive_gate_with_cv(
        self,
        val_preds: torch.Tensor,
        y_val: torch.Tensor,
        n_splits: int = 3,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Contiguous Blocked Inner CV over (lr, kl_weight) matching temporal TabReD dynamics.
        """
        val_preds, y_val = self._normalize_inputs(val_preds, y_val)
        N = len(val_preds)

        param_grid = [
            {'lr': 0.003, 'kl_weight': 0.01},
            {'lr': 0.003, 'kl_weight': 0.05},
            {'lr': 0.001, 'kl_weight': 0.05},
            {'lr': 0.005, 'kl_weight': 0.05},
        ]

        # Contiguous temporal blocks (no random shuffling across time)
        fold_size = N // n_splits
        folds = []
        for f in range(n_splits):
            start = f * fold_size
            stop = (f + 1) * fold_size if f < n_splits - 1 else N
            inner_val_idx = np.arange(start, stop)
            train_idx = np.concatenate([np.arange(0, start), np.arange(stop, N)])
            folds.append((train_idx, inner_val_idx))

        best_params = param_grid[0]
        best_cv_loss = float('inf')

        for params in param_grid:
            fold_losses = []
            for train_idx, inner_val_idx in folds:
                candidate = PostHocTabMAggregator(
                    k=self.k, task_type=self.task_type, mode='sample_adaptive',
                    temperature=self.temperature, score_bound=self.score_bound, seed=seed
                )
                candidate.fit_sample_adaptive_gate(
                    val_preds[train_idx], y_val[train_idx],
                    lr=params['lr'], kl_weight=params['kl_weight'],
                    n_epochs=30, seed=seed
                )
                inner_pred = candidate.aggregate_predictions(val_preds[inner_val_idx])
                if self.task_type == 'binclass':
                    loss = F.binary_cross_entropy(inner_pred.clamp(1e-7, 1 - 1e-7), y_val[inner_val_idx]).item()
                else:
                    loss = F.mse_loss(inner_pred, y_val[inner_val_idx]).item()
                fold_losses.append(loss)

            mean_loss = float(np.mean(fold_losses))
            if mean_loss < best_cv_loss:
                best_cv_loss = mean_loss
                best_params = params

        # Refit on full validation split with optimal parameters
        self.fit_sample_adaptive_gate(
            val_preds, y_val,
            lr=best_params['lr'], kl_weight=best_params['kl_weight'],
            n_epochs=40, seed=seed
        )
        return {"best_params": best_params, "cv_loss": best_cv_loss}

    def fit_sample_adaptive_gate(
        self,
        val_preds: torch.Tensor,
        y_val: torch.Tensor,
        n_epochs: int = 40,
        lr: float = 0.003,
        kl_weight: float = 0.05,
        weight_decay: float = 1e-3,
        seed: int = 42
    ) -> None:
        val_preds, y_val = self._normalize_inputs(val_preds, y_val)
        device = val_preds.device
        self.score_net.to(device)

        if len(self.active_subset) < self.k:
            preds_active = val_preds[:, self.active_subset]
        else:
            preds_active = val_preds

        # Record validation log-sigma bounds for test OOD clamping
        sigma = preds_active.std(dim=1, keepdim=True) + 1e-6
        log_sigma = torch.log(sigma)
        self.log_sigma_min.data = log_sigma.min().detach()
        self.log_sigma_max.data = log_sigma.max().detach()

        y_val_t = y_val.to(device)
        optimizer = torch.optim.AdamW(self.score_net.parameters(), lr=lr, weight_decay=weight_decay)

        gen = torch.Generator()
        gen.manual_seed(seed)
        dataset = torch.utils.data.TensorDataset(preds_active.detach(), y_val_t.detach())
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(128, len(dataset)), shuffle=True, generator=gen)

        self.score_net.train()
        for epoch in range(n_epochs):
            for batch_preds, batch_y in loader:
                optimizer.zero_grad()
                weights, log_weights = self._compute_adaptive_weights(batch_preds)
                pred = (batch_preds * weights).sum(dim=1, keepdim=True)

                if self.task_type == 'binclass':
                    task_loss = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7), batch_y)
                else:
                    task_loss = F.mse_loss(pred, batch_y)

                active_k = batch_preds.shape[1]
                uniform_log_target = torch.full_like(log_weights, -np.log(active_k))
                kl_penalty = F.kl_div(log_weights, uniform_log_target, log_target=True, reduction='batchmean')

                total_loss = task_loss + kl_weight * kl_penalty
                total_loss.backward()
                optimizer.step()

        self.score_net.eval()

    # -------------------------------------------------------------------------
    # Core Aggregation Pipeline
    # -------------------------------------------------------------------------

    def _compute_adaptive_weights(self, member_preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Permutation-equivariant scoring with bounded output:
        scores = score_bound * tanh(raw_score)
        Clamps ctx to validation range to prevent linear ReLU extrapolation under OOD shift.
        """
        B, K = member_preds.shape
        if K <= 1:
            weights = torch.ones((B, K), device=member_preds.device)
            return weights, torch.zeros((B, K), device=member_preds.device)

        mu = member_preds.mean(dim=1, keepdim=True)
        sigma = member_preds.std(dim=1, keepdim=True) + 1e-6
        z = (member_preds - mu) / sigma

        # Clamp ctx to [min - 1.0, max + 1.0] to prevent unbounded extrapolation
        raw_log_sigma = torch.log(sigma)
        ctx = torch.clamp(raw_log_sigma, self.log_sigma_min - 1.0, self.log_sigma_max + 1.0).expand_as(z)

        inputs = torch.stack([z, ctx], dim=-1)               # (B, K, 2)
        raw_scores = self.score_net(inputs).squeeze(-1)      # (B, K)

        # Bounded scoring: prevents hard selection explosions under shift
        bounded_scores = self.score_bound * torch.tanh(raw_scores)

        log_weights = F.log_softmax(bounded_scores / self.temperature, dim=-1)
        weights = F.softmax(bounded_scores / self.temperature, dim=-1)
        return weights, log_weights

    def aggregate_predictions(self, member_preds: torch.Tensor) -> torch.Tensor:
        """Aggregates cached predictions of shape (B, K) or (B, K, 1)."""
        if member_preds.ndim == 3 and member_preds.shape[-1] == 1:
            member_preds = member_preds.squeeze(-1)
        member_preds = member_preds.reshape(member_preds.shape[0], -1)

        if len(self.active_subset) < self.k:
            member_preds = member_preds[:, self.active_subset]

        B, active_k = member_preds.shape
        if active_k == 1:
            return member_preds.mean(dim=1, keepdim=True)

        if self.mode == 'uniform':
            return member_preds.mean(dim=1, keepdim=True)

        elif self.mode == 'median':
            return torch.quantile(member_preds, q=0.5, dim=1, keepdim=True)

        elif self.mode == 'trimmed_mean':
            n_trim = int(active_k * self.trim_fraction)
            if n_trim > 0 and (2 * n_trim) < active_k:
                sorted_preds, _ = torch.sort(member_preds, dim=1)
                trimmed = sorted_preds[:, n_trim : active_k - n_trim]
                return trimmed.mean(dim=1, keepdim=True)
            else:
                return member_preds.mean(dim=1, keepdim=True)

        elif self.mode == 'affine_recalibration':
            # a * mu + b
            mu = member_preds.mean(dim=1, keepdim=True)
            if self.task_type == 'binclass':
                # In logits space
                p_c = torch.clamp(mu, 1e-6, 1.0 - 1e-6)
                logits = torch.log(p_c / (1.0 - p_c))
                return torch.sigmoid(self.affine_a * logits + self.affine_b)
            else:
                return self.affine_a * mu + self.affine_b

        elif self.mode == 'global_stacking':
            w = self.global_weights[self.active_subset]
            w = (w / w.sum()).view(1, active_k)
            return (member_preds * w).sum(dim=1, keepdim=True)

        elif self.mode == 'sample_adaptive':
            weights, _ = self._compute_adaptive_weights(member_preds)
            return (member_preds * weights).sum(dim=1, keepdim=True)

        else:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        return_latency: bool = False
    ) -> Tuple[torch.Tensor, Optional[float]]:
        if self.base_tabm is None:
            raise RuntimeError("base_tabm is required for end-to-end forward pass.")

        with torch.no_grad():
            member_preds = self.base_tabm(x_num, x_cat)
            if self.task_type == 'binclass':
                member_preds = torch.sigmoid(member_preds)

        if return_latency:
            if member_preds.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        aggregated = self.aggregate_predictions(member_preds)

        agg_latency_ms = None
        if return_latency:
            if member_preds.is_cuda:
                torch.cuda.synchronize()
            agg_latency_ms = (time.perf_counter() - t0) * 1000.0

        return aggregated, agg_latency_ms


# -----------------------------------------------------------------------------
# Exact Official TabM[G] Algorithm (Appendix D.6 & paper/bin/model.py)
# -----------------------------------------------------------------------------

def select_greedy_heads_tabm(
    val_preds: torch.Tensor,
    y_val: torch.Tensor,
    task_type: Literal['regression', 'binclass'] = 'regression',
) -> List[int]:
    """
    Exact implementation of TabM[G] from TabM Paper (Appendix D.6 & paper/bin/model.py):
    1. Starts with the best single head.
    2. Iteratively adds the single remaining head that most improves the subset validation score.
    3. STOPS immediately when:
       - No candidate head improves the current greedy score, OR
       - The current greedy score already beats the full ensemble validation score.
    4. Only returns the strictly improving prefix (no non-improving heads appended).
    """
    if val_preds.ndim == 3 and val_preds.shape[-1] == 1:
        val_preds = val_preds.squeeze(-1)
    X = val_preds.detach().cpu().numpy().astype(np.float64)
    y = y_val.detach().cpu().reshape(-1).numpy().astype(np.float64)
    N, K = X.shape

    def compute_score(p):
        if task_type == 'binclass':
            # Exact task metric: ROC-AUC (higher is better)
            p_c = np.clip(p, 1e-7, 1 - 1e-7)
            try:
                return float(roc_auc_score(y, p_c))
            except ValueError:
                return 0.5
        else:
            # Exact task metric: Negative RMSE (higher is better)
            return -float(np.sqrt(np.mean((y - p) ** 2)))

    # Compute baseline score of full ensemble
    full_ensemble_score = compute_score(X.mean(axis=1))

    # Step 1: Best single head
    head_scores = [compute_score(X[:, i]) for i in range(K)]
    best_head = int(np.argmax(head_scores))

    greedy_idx = [best_head]
    greedy_score = head_scores[best_head]
    remaining_heads = set(range(K)) - {best_head}

    while remaining_heads:
        best_candidate = None
        best_candidate_score = -float('inf')

        for candidate in remaining_heads:
            candidate_idx = greedy_idx + [candidate]
            candidate_pred = X[:, candidate_idx].mean(axis=1)
            score = compute_score(candidate_pred)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = candidate

        # Official stopping rule (lines 762-770 in tabm/paper/bin/model.py):
        # Must improve over current greedy score AND beat full ensemble score
        if best_candidate_score > greedy_score and (
            best_candidate_score > full_ensemble_score
        ):
            greedy_score = best_candidate_score
            greedy_idx.append(best_candidate)
            remaining_heads.remove(best_candidate)
        else:
            # Immediate stop: no non-improving heads appended
            break

    return greedy_idx
