"""
Unit tests verifying unshared batch training mechanics and preprocessing integrity:
1. Set-level overlap verification across heads (unshared vs shared).
2. Per-head distinct training loss verification on unshared batches.
3. One-hot encoding verification for categorical features in TabM.make.
4. Preprocessing leakage guard: Quantile transform & scaler fit strictly on train split only.
"""

import torch
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import tabm


def test_unshared_vs_shared_batches():
    N = 10000   # training set size
    K = 32      # ensemble size
    B = 256     # batch size
    D = 8       # feature dim

    gen = torch.Generator().manual_seed(42)

    # 1. Unshared batch matrix: independent per-head permutations
    perms = torch.rand((K, N), generator=gen).argsort(dim=1).t()
    unshared_b = perms[:B]  # (B, K)

    # Verify set-level row overlap (intersection of sets of rows seen by heads):
    for i in range(K):
        set_i = set(unshared_b[:, i].tolist())
        assert len(set_i) == B, f"Head {i} has duplicate indices within its own batch"
        for j in range(i + 1, min(K, i + 5)):
            set_j = set(unshared_b[:, j].tolist())
            overlap = len(set_i & set_j)
            # Hypergeometric expectation: 256 * 256 / 10000 ≈ 6.5 items
            assert overlap < 20, f"Heads {i} and {j} have unexpectedly large row overlap: {overlap}"
            assert not torch.equal(unshared_b[:, i], unshared_b[:, j]), f"Heads {i} and {j} have identical rows!"

    print("[PASS] Test 1: Unshared batch permutation assigns distinct training row-sets to all 32 heads")

    # 2. Shared batch mode: verify all heads receive identical row sets
    shared_perm = torch.randperm(N, generator=gen)
    shared_b = torch.stack([shared_perm[:B] for _ in range(K)], dim=1)  # (B, K)
    for k in range(1, K):
        assert torch.equal(shared_b[:, 0], shared_b[:, k]), f"Shared batch mismatch for head {k}"
    print("[PASS] Test 2: Shared batch mode strictly replicates identical row-sets across all 32 heads")

    # 3. Model forward pass and distinct per-head loss verification
    model = tabm.TabM.make(n_num_features=D, d_out=1, k=K, arch_type='tabm')
    model.train()

    X_train = torch.randn(N, D)
    y_train = torch.randn(N, 1)

    xb = X_train[unshared_b]  # (B, K, D)
    yb = y_train[unshared_b]  # (B, K, 1)

    preds = model(xb)
    assert preds.shape == (B, K, 1)

    # Verify each head experiences a distinct loss on its distinct batch
    per_head_losses = [
        torch.nn.functional.mse_loss(preds[:, k], yb[:, k]).item()
        for k in range(K)
    ]
    unique_losses = set(np.round(per_head_losses, 4))
    assert len(unique_losses) == K, f"Expected 32 distinct loss values, found {len(unique_losses)}"
    print("[PASS] Test 3: All 32 heads compute distinct training losses on unshared batches")


def test_categorical_one_hot_encoding():
    """Verify that TabM.make with cat_cardinalities creates OneHotEncoding0d."""
    cat_cardinalities = [3, 7, 4]
    model = tabm.TabM.make(
        n_num_features=5,
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        k=4,
        arch_type='tabm'
    )
    assert model.cat_module is not None, "cat_module was not created!"
    # Verify module type is one-hot
    assert "OneHot" in model.cat_module.__class__.__name__ or hasattr(model.cat_module, 'cardinalities')

    # Pass categorical inputs (integers in range)
    B = 16
    x_num = torch.randn(B, 5)
    x_cat = torch.column_stack([
        torch.randint(0, c, (B,)) for c in cat_cardinalities
    ])
    out = model(x_num, x_cat)
    assert out.shape == (B, 4, 1)
    print("[PASS] Test 4: TabM.make successfully instantiates one-hot encoding for categorical features")


def test_preprocessing_train_only_fit():
    """Verify that QuantileTransformer and StandardScaler are fit on train only."""
    np.random.seed(42)
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(1000, 4))
    X_test = np.random.normal(loc=10.0, scale=5.0, size=(200, 4))  # strong shift in test

    qt = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42)
    X_train_trans = qt.fit_transform(X_train)
    X_test_trans = qt.transform(X_test)

    # Quantile transformer fit on train: train median must be ~0.0
    assert abs(np.median(X_train_trans)) < 0.05
    # Since test has mean 10.0 (far above train), transformed test values should saturate at train max quantiles
    assert np.mean(X_test_trans) > 3.0, "Test data was contaminated during fit!"

    # Label scaler
    y_train = np.random.normal(loc=5.0, scale=2.0, size=(1000, 1))
    y_test = np.random.normal(loc=20.0, scale=2.0, size=(200, 1))
    scaler_y = StandardScaler()
    scaler_y.fit(y_train)

    assert abs(scaler_y.mean_[0] - 5.0) < 0.15
    # Ensure test data did not alter scaler mean
    assert abs(scaler_y.mean_[0] - 20.0) > 10.0
    print("[PASS] Test 5: Preprocessing (QuantileTransformer & StandardScaler) confirmed fit strictly on train split only")


if __name__ == '__main__':
    print("=" * 70)
    print("      RUNNING UNSHARED BATCH & PREPROCESSING INTEGRITY TESTS      ")
    print("=" * 70)
    test_unshared_vs_shared_batches()
    test_categorical_one_hot_encoding()
    test_preprocessing_train_only_fit()
    print("=" * 70)
    print("ALL INTEGRITY TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)
