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
- **Amendment v2.2 & Code Deviation Registration (October 18, 2026):**
  - *Code Deviation (Evaluation Batch Size Capped at 2048):*
    - The upstream default in `tabm/paper/bin/model.py` sets `eval_batch_size = 32768` (when `compile=False`).
    - *Hardware Failure Mode on Windows WDDM:* Under the Windows Display Driver Model (WDDM), PyTorch does NOT raise a CUDA `OutOfMemoryError` when GPU memory demand exceeds physical VRAM (4,096 MiB on NVIDIA RTX 3050 Laptop). Instead, the OS display driver silently evicts GPU tensors to system RAM (host-RAM paging). Because no CUDA OOM exception is raised, the upstream exception handler in `bin/model.py` (which catches CUDA OOM and halves `eval_batch_size`) is never invoked. The process continues executing, but throughput collapses by >50x (e.g. from 5 it/s to 5 s/it) and GPU power plunges.
    - *Architectural Choice (Code Clamp vs. Driver Sysmem Fallback):* An alternative driver-level workaround exists (configuring the NVIDIA driver sysmem fallback setting or `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` to force hard OOM crashes). We explicitly chose the **code-level modification** (`eval_batch_size = 2048`) because it is platform-agnostic, guarantees deterministic execution without host OS driver re-configuration, and keeps memory on all 8 datasets strictly below 3.1 GB physical VRAM with zero paging.
    - *Uniform Scope:* This deviation is applied uniformly across all 8 TabReD datasets and all seeds (not solely the heavy datasets).
    - *Prediction Equivalence Verification:* Evaluated on a trained `maps-routing` seed-0 checkpoint at `eval_batch_size = 16384` versus `eval_batch_size = 2048`:
      - Test RMSE at 16384: `0.1610847043324687`
      - Test RMSE at 2048:  `0.1610847390509623`
      - Absolute RMSE difference: **$3.47 \times 10^{-8}$** (identical to 7 decimal places).
      - Head-prediction coordinate max difference: $4.51 \times 10^{-5}$ (due strictly to FP16 summation ordering under AMP autocast).
  - *Maps-Routing Throughput Reconciliation:*
    - Training throughput on `maps-routing` (160,019 rows, batch 1024) is 19.5 it/s (156 steps / epoch $\approx$ 8.0 s per epoch).
    - However, `bin/model.py` evaluates both the validation split (60,000 rows) and test split (60,000 rows) across all 32 heads at the end of *every single epoch* to log progress and determine early stopping.
    - At `eval_batch_size = 2048`, evaluating 120,000 rows across 32 heads takes $\approx 20.5$ s per epoch.
    - Total wall-clock time per epoch is $\approx 8.0\text{ s (train)} + 20.5\text{ s (eval)} = 28.5\text{ s/epoch}$. Across 30 epochs plus startup and head prediction export, this totals $\approx 15\text{m } 17\text{s}$, fully reconciling the measured wall-clock time.
  - *Weather Feature Count Reconciliation:*
    - Verified on disk: `weather` contains 100 numerical features (`N.npy`) and 3 binary features (`B.npy`) $\rightarrow$ **103 total features**.
    - The mention of "33" in earlier pre-flight notes was a verbal typographical slip conflating `epoch_size = 333` (number of batches per epoch) with feature count. The underlying data tensors are 100% identical to the released TabReD benchmark.
  - *Standardized Peak Memory Measurements (`eval_batch_size = 2048`):*
    - `maps-routing`: Allocated = 1,276.4 MiB, Reserved = 1,294.0 MiB, nvidia-smi = 1,750 MiB.
    - `weather`: Allocated = 1,730.5 MiB, Reserved = 2,586.0 MiB, nvidia-smi = 1,950 MiB.
    - `homecredit-default`: Allocated $\approx 2,550$ MiB, Reserved $\approx 2,850$ MiB, nvidia-smi $\approx 3,075$ MiB (safely below the 4,096 MiB physical ceiling).
  - *Empirical Timing of Intermediate Benchmarks & Total Budget:*
    - `cooking-time` (227k rows): 56.55 it/s, $\approx 9.0$ s/epoch $\rightarrow \approx 4.5$ min/seed.
    - `delivery-eta` (279k rows): 31.20 it/s, $\approx 16.2$ s/epoch $\rightarrow \approx 8.1$ min/seed.
    - Total confirmatory budget across 40 runs ($8 \times 5$ seeds) + aggregators is realistically budgeted at **$\approx 8.0$ to 8.5 hours**.
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

All backbone models use the full, standard TabM architecture ($k=32$, non-mini, `arch_type='tabm'`) with independent per-head batch permutations (`share_training_batches=False`), noisy-quantile normalization, AdamW optimizer, and early stopping with `patience=16` consecutive epochs without validation improvement (with no arbitrary epoch caps).

### Code Modifications Relative to Pinned Upstream Repository (`0ea03a1e`)
All code modifications to the authors' codebase are restricted to:
1. `tabm/paper/bin/model.py`:
   - Line 343: Reset peak CUDA memory statistics at training start (`torch.cuda.reset_peak_memory_stats()`).
   - Line 423: Pre-registered deviation (v2.2): default `eval_batch_size = 2048` uniformly.
   - Line 648: Print ASCII progress message `>>> New best epoch!` to prevent Windows `cp1252` encoding errors.
   - Line 660: Record `peak_gpu_memory_allocated_mb` and `peak_gpu_memory_reserved_mb` into `report.json`.
   - Line 674: Export submodel head predictions (`head_predictions.npz`) at the best checkpoint for offline aggregator analysis without retraining.
2. `tabm/paper/lib/util.py`:
   - Line 446: Support PyTorch 2.6+ checkpoint deserialization (`weights_only=False` default in `load_checkpoint`).

### Released Benchmark Configurations and SHA256 Hashes
The confirmatory evaluation uses the released per-dataset configuration files located in `tabm/paper/exp/tabm/tabred/<dataset>/0-evaluation/0.toml`.
Before execution, the confirmatory runner asserts that each configuration matches the frozen SHA256 hashes:

| Dataset | Config Path | Canonical LF SHA256 | Windows CRLF SHA256 | Task Type | Batch Size | Learning Rate | Weight Decay | Backbone ($N \times d$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **cooking-time** | `tabm/paper/exp/tabm/tabred/cooking-time/0-evaluation/0.toml` | `901bbf02fcf312d864d6f41596bbdc3dfe10ae9634dfbd16c8ddf40ba87225f3` | `8bd53da73578a697438fcd26f6b4c9f20507385601aed80ba9774b2e3dd08c5c` | Regression | 1024 | 0.0027720 | 0.00013 | $2 \times 176$ |
| **delivery-eta** | `tabm/paper/exp/tabm/tabred/delivery-eta/0-evaluation/0.toml` | `fa83fb18969a192fb8f98e1fc5c1630a3efc66c41c6b8a9aade41b927c377214` | `1699a254827332ef3eec89a050f50dbfa70ebdb3967eec182f65ea9fac9734cc` | Regression | 1024 | 0.0007092 | 0.00000 | $4 \times 192$ |
| **ecom-offers** | `tabm/paper/exp/tabm/tabred/ecom-offers/0-evaluation/0.toml` | `742f82218b45212976fa93eca91e03152c9d2fa9b01fc25c8a8182e47abaf2d2` | `2031af829016ef835175b28882cfca59d9a33e8e15a16e1fe92e6d418b01eef8` | BinClass (AUC) | 1024 | 0.0010951 | 0.00000 | $1 \times 528$ |
| **homecredit-default** | `tabm/paper/exp/tabm/tabred/homecredit-default/0-evaluation/0.toml` | `f8c0cbbc07c3d474974b31fb143476ff7244f15b86943670c3658b1ad3df255a` | `54c5da07e78a69b3d1c0d129c3a4779da8ebcb3ce7c9081364b59ce21b4e6b16` | BinClass (AUC) | 1024 | 0.0014221 | 0.00027 | $5 \times 560$ |
| **homesite-insurance** | `tabm/paper/exp/tabm/tabred/homesite-insurance/0-evaluation/0.toml` | `163d767c4e5ee4e9471de1572dfbaf523d8b2a27e7c6034f2051cada8efbbe12` | `e37118ff0b52c9867b99d43b1d1f6b23e3aaf75c8b65f9580e05ffd8a2170866` | BinClass (AUC) | 1024 | 0.0011447 | 0.00000 | $2 \times 1024$ |
| **maps-routing** | `tabm/paper/exp/tabm/tabred/maps-routing/0-evaluation/0.toml` | `21c176e01af048c31d8a8726aff45b90f77288c4a012e7cd61dea733cf303e9b` | `c05460274c476e88ca44ccc73e1f0a5e498c2f434a91705ab761a387e75692e9` | Regression | 1024 | 0.0013713 | 0.00000 | $3 \times 192$ |
| **sberbank-housing** | `tabm/paper/exp/tabm/tabred/sberbank-housing/0-evaluation/0.toml` | `7323132ff67b95987cd9a3ee726cfe0444627b5d3004ea97703a0a28c0441038` | `cb4bb523573b072d0a568913eaa06b6fda109625abcb863d8a3ea98f2fb4b3db` | Regression | 256 | 0.0023156 | 0.00000 | $3 \times 240$ |
| **weather** | `tabm/paper/exp/tabm/tabred/weather/0-evaluation/0.toml` | `c59da405b8fb311eb93e02d3a82a75e88c9864e933bb166517c24d9a646fc38e` | `7e99d0ddd0bf2cfe2731a5e5979fb30ed027a2fb0bdd2ad4b3ae33a62268544a` | Regression | 1024 | 0.0027868 | 0.00159 | $5 \times 896$ |

