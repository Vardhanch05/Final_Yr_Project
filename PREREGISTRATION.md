# Pre-Registration: Empirical Study of Post-Hoc Aggregations for TabM

**Date:** October 16, 2026  
**Status:** Pre-Registered (Prior to Confirmatory Phase Execution)  
**Target Paper Extension:** *"TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembles"* (ICLR 2025)

---

## 1. Core Research Question
Does post-hoc, sample-adaptive weighting (and robust parametric/non-parametric aggregation) of TabM's implicit submodels outperform naive uniform averaging under real-world distribution shift?

---

## 2. Hypothesis Families

### Primary Hypothesis (H1: Real Distribution Shift)
- **Claim:** Sample-adaptive gating outperforms uniform averaging on tabular benchmarks exhibiting temporal distribution shift.
- **Evaluation Set:** The 8 official **TabReD** industrial temporal benchmarks with fixed chronological splits:
  1. `weather` (regression)
  2. `sberbank-housing` (regression)
  3. `cooking-time` (regression)
  4. `delivery-eta` (regression)
  5. `ecom-offers` (binclass)
  6. `homecredit-default` (binclass)
  7. `homesite-insurance` (binclass)
  8. `maps-routing` (regression)
- **Statistical Test:** Two-sided paired Wilcoxon Signed-Rank Test across the 8 dataset-level means ($\alpha = 0.05$). Minimum exact two-sided p-value for $N=8$ is $p \approx 0.0078$.

### Secondary Hypothesis Family (H2..H6: Competitor Hierarchy)
- **Comparisons:** Adaptive Gate vs.
  1. `Exact Median` (Robust middle average of sorted predictions)
  2. `Trimmed Mean` (Symmetric trimming fraction $p \in [0.05, 0.20]$ tuned on val)
  3. `Affine Recalibration` ($y \approx a \cdot \mu + b$ fitted on validation split)
  4. `Global Stacking` (Static non-negative weights on simplex via SLSQP)
  5. `TabM[G] Greedy` (Submodel forward selection)
- **Multiplicity Correction:** Step-Down Holm-Bonferroni correction over the 5 secondary hypothesis tests ($\alpha = 0.05$).

### Secondary Control Family (No Shift)
- **Benchmarks:** 3 standard regression benchmarks evaluated under random splits (no distribution drift):
  1. `diamond`
  2. `house`
  3. `black-friday`
- **Purpose:** Descriptive control establishing whether non-uniform aggregation behaves differently in stationary vs. non-stationary regimes.

*(Note: Multiclass benchmarks such as Otto and Covtype are excluded to avoid multiclass score ambiguity and focus strictly on regression and binary classification).*

---

## 3. Decision Framework & Equivalence Testing

### Equivalence Testing Framework
- **Practical Equivalence Margin ($\Delta$):** $\pm 0.50\%$ relative metric change.
  - *Justification:* In the TabM paper, the overall architecture advantage of TabM over baseline MLP is $\approx 2.15\%$. A threshold of $\pm 0.5\%$ is $< \frac{1}{4}$ of the core ensembling gain and is within empirical seed standard deviations ($\sigma_{\text{seed}} \approx 0.1\% - 0.4\%$).
- **Equivalence Rule:** Equivalence to naive uniform averaging is declared if the **90% Confidence Interval** (corresponding to two one-sided tests at $\alpha=0.05$) of the median relative delta across datasets lies strictly within $[-\Delta, +\Delta] = [-0.50\%, +0.50\%]$.
- *Statistical Power Caveat:* With $N=8$ datasets, confidence intervals on the median may be wide, and formal equivalence may be inconclusive if confidence bounds exceed $\pm 0.5\%$. This is acknowledged a priori.

### Pre-Registered Tie Rule
- **Definition:** An observed gain $\Delta$ is classified as a "Tie" if $|\Delta| \le \sigma_{\text{seed}}$, where $\sigma_{\text{seed}} = \frac{1}{D}\sum_d \mathrm{std}_{s}(\Delta_{d,s})$ is the mean cross-seed standard deviation observed for that comparison.
- **Metric Definitions:**
  - Regression: Relative change in RMSE ($\frac{\text{RMSE}_{\text{base}} - \text{RMSE}_{\text{cand}}}{\text{RMSE}_{\text{base}}} \times 100\%$).
  - Binary Classification: Relative error-reduction in ROC-AUC ($\frac{\text{AUC}_{\text{cand}} - \text{AUC}_{\text{base}}}{1.0 - \text{AUC}_{\text{base}}} \times 100\%$).

---

## 4. Aggregator Specification & TabM[G] Variants

### TabM[G] Specifications
1. **Primary TabM[G] Baseline (Paper Appendix D.6)**:
   - Forward greedy selection stopping when **two conditions hold simultaneously**:
     1. No candidate head strictly improves the collective validation score (`best_cand_score <= cur`).
     2. The current collective score beats the full 32-head ensemble (`cur > full_ensemble_score`).
   - **Prefix Policy:** Returns the highest-scoring validation prefix encountered during the search, preventing tail degradation if non-improving steps are traversed.
2. **Secondary TabM[G] Variant (Official Repository `model.py`)**:
   - Stops immediately at the first step where no single candidate head improves the current score, without comparing against the full ensemble.

### Sample-Adaptive Gate
- **Architecture:** Permutation-equivariant scoring network:
  $s_k(x) = \text{ScoreNet}([ \hat{y}_k(x) - \mu(x),\, \ln \sigma(x) ])$.
- **Bounds:** Strictly bounded outputs via $\tanh$, centered at uniform weights at initialization ($\text{weights} = \frac{1}{K}$ when unoptimized).
- **Optimization:** Fitted on validation predictions with forward-chaining expanding window cross-validation over $(\text{lr}, \lambda_{\text{KL}})$ and early stopping. Zero training on future slices.

---

## 5. Model Architecture & Training Hyperparameters

- **Backbone Variant:** Plain BatchEnsemble (`arch_type='tabm'`) with independent per-head batch sampling (`share_training_batches=False`).
- **Development Set (California Housing Reproduction):**
  - Architecture: `n_blocks: 3, d_block: 400, dropout: 0.2077`
  - Optimizer: `AdamW(lr=0.0008725, weight_decay=0.03777)`
  - Split: Official fixed split (13,209 train / 3,303 val / 4,128 test)
  - Early stopping: `patience=16`, monitoring validation RMSE until convergence.
- **Confirmatory TabReD Datasets:**
  - Evaluated using official tuned hyperparameters and official chronological train/val/test splits.
