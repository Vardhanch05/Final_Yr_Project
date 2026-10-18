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

### Pre-Registered Data-Dependent Tie Rule (Amended v2.1)
- **Final Confirmatory Seed Count:** Fixed at **$S=5$ seeds** (seeds 0, 1, 2, 3, 4) across all $D=8$ TabReD datasets based on measured hardware throughput on RTX 3050 Laptop GPU (realistic total compute: ~7.5 to 8.0 hours including inner gate CV).
- **Metric Definitions and Paired Gain ($\Delta_d$):**
  - **Regression (RMSE):**
    - Seed-level relative gain:
      $$\Delta_{d, s} = \frac{\text{RMSE}_{\text{base}, d, s} - \text{RMSE}_{\text{cand}, d, s}}{\text{RMSE}_{\text{base}, d, s}} \times 100\%$$
    - Dataset-level mean paired relative gain:
      $$\Delta_d = \frac{1}{S}\sum_{s=0}^{S-1} \Delta_{d, s}$$
  - **Binary Classification (ROC-AUC):**
    - Seed-level relative unachieved error reduction:
      $$\Delta_{d, s} = \frac{\text{AUC}_{\text{cand}, d, s} - \text{AUC}_{\text{base}, d, s}}{1.0 - \text{AUC}_{\text{base}, d, s}} \times 100\%$$
    - Dataset-level mean paired relative gain:
      $$\Delta_d = \frac{1}{S}\sum_{s=0}^{S-1} \Delta_{d, s}$$
- **Candidate-Independent Baseline Noise Threshold ($\tau_d$ and $\sigma_{\text{base}, d}$):**
  The tie threshold is strictly candidate-independent, defined solely by the cross-seed variability of the naive uniform baseline over the $S=5$ seeds with sample degrees of freedom (`ddof=1`):
  - **Regression Datasets** (`cooking-time`, `delivery-eta`, `maps-routing`, `sberbank-housing`, `weather`):
    $$\sigma_{\text{base}, d} = \frac{\mathrm{std}_{s \in \{0..4\}}(\text{RMSE}_{\text{base}, d, s}, \text{ddof}=1)}{\mathrm{mean}_{s \in \{0..4\}}(\text{RMSE}_{\text{base}, d, s})} \times 100\%$$
  - **Binary Classification Datasets** (`ecom-offers`, `homecredit-default`, `homesite-insurance`):
    In matching units to the unachieved error reduction metric:
    $$\sigma_{\text{base}, d} = \frac{\mathrm{std}_{s \in \{0..4\}}(\text{AUC}_{\text{base}, d, s}, \text{ddof}=1)}{1.0 - \mathrm{mean}_{s \in \{0..4\}}(\text{AUC}_{\text{base}, d, s})} \times 100\%$$
  - **Benchmark-Level Mean Relative Noise:**
    $$\sigma_{\text{base}} = \frac{1}{D}\sum_{d=1}^D \sigma_{\text{base}, d}$$
- **Decision Rule (Evaluated at full floating-point precision):**
  - **Win:** $\Delta_d > +\sigma_{\text{base}, d}$
  - **Loss:** $\Delta_d < -\sigma_{\text{base}, d}$
  - **Tie:** $|\Delta_d| \le \sigma_{\text{base}, d}$
- **Conservative Nature of the Tie Rule:** Because the baseline standard deviation is candidate-independent and un-paired across seeds, it is wider than the noise of paired differences $\Delta_{d, s}$ (since aggregators share identical underlying head representations). The decision rule therefore leans conservatively toward declaring a "tie", which aligns directly with the practical equivalence objective of this study.
- **Labelled Sensitivity Analysis:** In parallel, all outcomes are evaluated against the fixed practical equivalence margin of $\pm 0.50\%$ relative metric change.

### Erratum and Amendment History
- **Amendment v2.1 & Pre-Flight Erratum (October 18, 2026):**
  - *Erratum on Pre-Flight Outputs:* The v2 amendment text stated "zero confirmatory test data was evaluated". This is corrected: prior to the confirmatory phase, pre-flight feasibility and timing runs were performed on two TabReD datasets:
    1. `maps-routing` seed 0: head predictions were extracted and test metrics observed (uniform 0.16108, stacking 0.16099, gate 0.16102, median 0.16123, tabm_g 0.16101).
    2. `sberbank-housing` seed 0: test RMSE was observed (0.2449 vs Table 20 0.2439).
  - *Retraining Policy:* To ensure the confirmatory phase is completely uniform and uncompromised across all datasets, **all seeds (0..4) will be retrained from scratch** during the single confirmatory execution. The earlier seed-0 runs are designated strictly as pre-flight hardware and feasibility checks.
  - *AUC Threshold Correction:* Fixed the relative baseline noise threshold formula for classification datasets to divide by $(1.0 - \mathrm{mean}(\text{AUC}))$ rather than $\mathrm{mean}(\text{AUC})$, matching the units of $\Delta_d$. Specified `ddof=1` for sample standard deviations.
  - *Timing & SE Rationale Correction:* Moving from 3 to 5 seeds reduces the standard error of the mean by $1 - \sqrt{3/5} \approx 23\%$ (while 3-seed SE is 29% larger). Total confirmatory compute is re-estimated at ~7.5 to 8.0 hours based on empirical profiling.
  - *Resumability & Crash Protection:* Confirmatory execution runner is implemented with resumable on-disk checkpointing, per-seed log output, and idempotent re-entry.
- **Amendment v2 (October 18, 2026):**
  - *Change:* Replaced candidate-dependent paired delta standard deviation ($\sigma_{\text{cand}} = \mathrm{std}_s(\Delta_{\text{cand}, s})$) with candidate-independent baseline standard deviation ($\sigma_{\text{base}}$ over $S=5$ seeds).
  - *Motivation:* Grounded strictly and exclusively in dev results on California ($N=1, S=5$ dev data only), where TabM[G] received a wide threshold ($\sigma=0.282\%$) while trimmed mean received a narrow threshold ($\sigma=0.022\%$), creating an artificial incentive for candidate variance. Amending to baseline noise ensures an equal, invariant tie band for all competitor aggregators across all datasets.

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

## 5. Model Architectures & Released Hyperparameter Configurations

All backbone models use plain BatchEnsemble (`arch_type='tabm'`) with independent per-head batch permutations (`share_training_batches=False`), $k=32$ heads, noisy-quantile normalization, AdamW optimizer, and early stopping with `patience=16` consecutive epochs without validation improvement (with no arbitrary epoch caps).

### Released Benchmark Configurations and SHA256 Hashes
The confirmatory evaluation uses the released per-dataset configuration files located in `tabm/paper/exp/tabm/tabred/<dataset>/0-evaluation/0.toml`:

| Dataset | Path | SHA256 Hash | Task Type | Batch Size | Learning Rate | Weight Decay | Backbone ($N \times d$) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **cooking-time** | `tabm/paper/exp/tabm/tabred/cooking-time/0-evaluation/0.toml` | `8bd53da73578a697926b01b697e0fc2f1a3e6e87aeb46fa4bb6036da48bb14c8` | Regression | 1024 | 0.0027720 | 0.00013 | $2 \times 176$ |
| **delivery-eta** | `tabm/paper/exp/tabm/tabred/delivery-eta/0-evaluation/0.toml` | `1699a254827332ef4cb2ca0e1b6f0e4b8595aeb2160fe9ae114c2c77d0cf88c9` | Regression | 1024 | 0.0007092 | 0.00000 | $4 \times 192$ |
| **ecom-offers** | `tabm/paper/exp/tabm/tabred/ecom-offers/0-evaluation/0.toml` | `2031af829016ef83d9aa669fcf7fe5c421712a7bf8ca4bb248107936a5c2ceb9` | BinClass (AUC) | 1024 | 0.0010951 | 0.00000 | $1 \times 528$ |
| **homecredit-default** | `tabm/paper/exp/tabm/tabred/homecredit-default/0-evaluation/0.toml` | `54c5da07e78a69b3eeae4478174f884a4413f2ca97213ee73d052be17e13bbd8` | BinClass (AUC) | 1024 | 0.0014221 | 0.00027 | $5 \times 560$ |
| **homesite-insurance** | `tabm/paper/exp/tabm/tabred/homesite-insurance/0-evaluation/0.toml` | `e37118ff0b52c98696ff0cbfae8bc92b0c9533fca4ae7702f23d24220b33b91a` | BinClass (AUC) | 1024 | 0.0011447 | 0.00000 | $2 \times 1024$ |
| **maps-routing** | `tabm/paper/exp/tabm/tabred/maps-routing/0-evaluation/0.toml` | `c05460274c476e88514936d0b677a83d47d6d7e00406085a86d267fec453887c` | Regression | 1024 | 0.0013713 | 0.00000 | $3 \times 192$ |
| **sberbank-housing** | `tabm/paper/exp/tabm/tabred/sberbank-housing/0-evaluation/0.toml` | `cb4bb523573b072da6a7db18c65f978e88e898327ffae5c48b29f7cf7c0fa3e5` | Regression | 256 | 0.0023156 | 0.00000 | $3 \times 240$ |
| **weather** | `tabm/paper/exp/tabm/tabred/weather/0-evaluation/0.toml` | `7e99d0ddd0bf2cfecfa9bbd460efceb049d58a5da9092495bb3d6e5793086eb1` | Regression | 1024 | 0.0027868 | 0.00159 | $5 \times 896$ |
