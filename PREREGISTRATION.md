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
  1. `cooking-time` (regression)
  2. `delivery-eta` (regression)
  3. `ecom-offers` (binary classification)
  4. `homecredit-default` (binary classification)
  5. `homesite-insurance` (binary classification)
  6. `maps-routing` (regression)
  7. `sberbank-housing` (regression)
  8. `weather` (regression)
- **Statistical Test:** Two-sided paired Wilcoxon Signed-Rank Test across the 8 dataset-level means ($\alpha = 0.05$). Minimum exact two-sided p-value for $N=8$ is $p \approx 0.0078$.

### Secondary Hypothesis Family (H2..H6: Competitor Hierarchy)
- **Comparisons:** Adaptive Gate vs.
  1. `Exact Median` (Robust middle average of sorted predictions)
  2. `Trimmed Mean` (Symmetric trimming fraction $p \in [0.05, 0.20]$ tuned on val)
  3. `Affine Recalibration` ($y \approx a \cdot \mu + b$ fitted on validation split)
  4. `Global Stacking` (Static non-negative weights on simplex via SLSQP)
  5. `TabM[G], Appendix D.6 with best-prefix` (Submodel forward selection)
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

### Practical Equivalence Testing Framework
- **Practical Equivalence Margin ($\Delta$):** $\pm 0.50\%$ relative metric change.
  - *Justification:* In the TabM paper, the overall architecture advantage of TabM over baseline MLP is $\approx 2.15\%$. A threshold of $\pm 0.50\%$ is $< \frac{1}{4}$ of the core ensembling gain and is within empirical seed standard deviations ($\sigma_{\text{seed}} \approx 0.1\% - 0.4\%$).
- **Equivalence Rule:** Equivalence to naive uniform averaging is declared if the **90% Confidence Interval** (equivalent to two one-sided tests at $\alpha=0.05$) of the median relative delta across datasets lies strictly within $[-\Delta, +\Delta] = [-0.50\%, +0.50\%]$.
- *Statistical Power Caveat:* With $N=8$ datasets, confidence intervals on the median will be wide. If the 90% CI spans zero but extends beyond $\pm 0.50\%$, the outcome is formally classified as *inconclusive* (insufficient sample size to confirm equivalence).

### Pre-Registered Data-Dependent Tie Rule (Amended v2)
- **Final Confirmatory Seed Count:** Fixed at **$S=5$ seeds** (seeds 0, 1, 2, 3, 4) across all $D=8$ TabReD datasets based on hardware throughput verification on RTX 3050 Laptop GPU (4.5 hours total runtime).
- **Candidate-Independent Baseline Noise Threshold ($\tau_d$ and $\sigma_{\text{base}, d}$):**
  To prevent volatile candidate aggregators from receiving unfairly wide tie margins compared to stable aggregators, the tie threshold is strictly candidate-independent, defined solely by the cross-seed variability of the naive uniform baseline over the $S=5$ seeds:
  - Absolute Metric Threshold per dataset $d$:
    $$\tau_d = \mathrm{std}_{s \in \{0..4\}}(\text{Metric}_{\text{base}, d, s})$$
  - Relative Percentage Threshold per dataset $d$:
    $$\sigma_{\text{base}, d} = \frac{\mathrm{std}_{s \in \{0..4\}}(\text{Metric}_{\text{base}, d, s})}{\mathrm{mean}_{s \in \{0..4\}}(\text{Metric}_{\text{base}, d, s})} \times 100\%$$
  - Benchmark-Level Mean Relative Noise:
    $$\sigma_{\text{base}} = \frac{1}{D}\sum_{d=1}^D \sigma_{\text{base}, d}$$
- **Decision Rule (Evaluated at full floating-point precision):**
  - **Win:** $\Delta_d > +\sigma_{\text{base}, d}$
  - **Loss:** $\Delta_d < -\sigma_{\text{base}, d}$
  - **Tie:** $|\Delta_d| \le \sigma_{\text{base}, d}$
- **Labelled Sensitivity Analysis:** In parallel, all outcomes are evaluated against the fixed practical equivalence margin of $\pm 0.50\%$ relative metric change.
- **Metric Definitions:**
  - Regression: Relative change in RMSE ($\frac{\text{RMSE}_{\text{base}} - \text{RMSE}_{\text{cand}}}{\text{RMSE}_{\text{base}}} \times 100\%$).
  - Binary Classification: Relative error-reduction in ROC-AUC ($\frac{\text{AUC}_{\text{cand}} - \text{AUC}_{\text{base}}}{1.0 - \text{AUC}_{\text{base}}} \times 100\%$).

### Amendment History
- **Amendment v2 (October 18, 2026, Pre-Confirmatory):**
  - *Change:* Replaced candidate-dependent paired delta standard deviation ($\sigma_{\text{cand}} = \mathrm{std}_s(\Delta_{\text{cand}, s})$) with candidate-independent baseline standard deviation ($\sigma_{\text{base}}$ over $S=5$ seeds).
  - *Motivation:* Grounded strictly and exclusively in dev results on California ($N=1, S=5$ dev data only), where TabM[G] received a wide threshold ($\sigma=0.282\%$) while trimmed mean received a narrow threshold ($\sigma=0.022\%$), creating an artificial incentive for candidate variance. Amending to baseline noise ensures an equal, invariant tie band for all competitor aggregators across all datasets.
  - *Timing:* Formalized, documented, and tagged prior to executing the single confirmatory run on TabReD. Zero confirmatory test data was evaluated prior to this amendment.

---

## 4. Aggregator Specification & TabM[G] Variants

### TabM[G] Variants
1. **Primary TabM[G] Baseline ("TabM[G], D.6 with best-prefix")**:
   - Forward greedy selection stopping when **two conditions hold simultaneously**:
     1. No candidate head strictly improves the collective validation score (`best_cand_score <= cur`).
     2. The current collective score beats the full 32-head ensemble (`cur > full_ensemble_score`).
   - **Prefix Policy:** Returns the highest-scoring validation prefix encountered during the search, preventing tail degradation if non-improving steps are traversed before beating the full ensemble.
2. **Secondary TabM[G] Variant ("TabM[G], official repository model.py variant")**:
   - Stops immediately at the first step where no single candidate head improves the current score, without comparing against the full ensemble.

### Sample-Adaptive Gate
- **Architecture:** Permutation-equivariant scoring network:
  $$s_k(x) = \text{ScoreNet}([ \hat{y}_k(x) - \mu(x),\, \ln \sigma(x) ])$$
- **Bounds:** Strictly bounded outputs via $\tanh$, centered at uniform weights at initialization ($\text{weights} = \frac{1}{K}$ when unoptimized).
- **Optimization:** Fitted on validation predictions with forward-chaining expanding window cross-validation over $(\text{lr}, \lambda_{\text{KL}})$ and early stopping. Zero training on future slices.

---

## 5. Model Architectures & Training Hyperparameters

All backbone models use plain BatchEnsemble (`arch_type='tabm'`) with independent per-head batch permutations (`share_training_batches=False`), $k=32$ heads, noisy-quantile normalization, AdamW optimizer, and early stopping with `patience=16` consecutive epochs without validation improvement (with no arbitrary epoch caps).

### Pre-Registered Tuned Hyperparameter Configurations
*(Extracted directly from official repository `exp/tabm/*/0-evaluation/0.toml` files)*

| Dataset | Family | Task Type | Blocks | Width | Dropout | Learning Rate | Weight Decay | Batch Size | Patience |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **california** | Dev (Fixed Split) | Regression | 3 | 400 | 0.2077 | 0.0008725 | 0.03777 | 256 | 16 |
| **cooking-time** | Primary (TabReD) | Regression | 2 | 176 | 0.0000 | 0.0027720 | 0.00013 | 1024 | 16 |
| **delivery-eta** | Primary (TabReD) | Regression | 4 | 192 | 0.0000 | 0.0007092 | 0.00000 | 1024 | 16 |
| **ecom-offers** | Primary (TabReD) | BinClass (AUC) | 1 | 528 | 0.0644 | 0.0010951 | 0.00000 | 1024 | 16 |
| **homecredit-default** | Primary (TabReD) | BinClass (AUC) | 5 | 560 | 0.0000 | 0.0014221 | 0.00027 | 1024 | 16 |
| **homesite-insurance** | Primary (TabReD) | BinClass (AUC) | 2 | 1024 | 0.3780 | 0.0011447 | 0.00000 | 1024 | 16 |
| **maps-routing** | Primary (TabReD) | Regression | 3 | 192 | 0.0000 | 0.0013713 | 0.00000 | 1024 | 16 |
| **sberbank-housing** | Primary (TabReD) | Regression | 3 | 240 | 0.0000 | 0.0023156 | 0.00000 | 256 | 16 |
| **weather** | Primary (TabReD) | Regression | 5 | 896 | 0.0000 | 0.0027868 | 0.00159 | 1024 | 16 |
| **diamond** | Control (No Shift) | Regression | 3 | 432 | 0.0671 | 0.0008384 | 0.09943 | 512 | 16 |
| **house** | Control (No Shift) | Regression | 5 | 784 | 0.0000 | 0.0020846 | 0.00014 | 256 | 16 |
| **black-friday** | Control (No Shift) | Regression | 2 | 1024 | 0.1631 | 0.0011783 | 0.00208 | 512 | 16 |
