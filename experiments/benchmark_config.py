"""
Pre-Registered Experimental Protocol Configuration for TabM Aggregation Study.

Hypothesis Registration:
- Primary Hypothesis (H1): Sample-adaptive gating outperforms official uniform averaging
  under real distribution shift across >= 10 real-world datasets.
- Secondary Hypotheses (H2..H6): Sample-adaptive gating vs. [Median, Trimmed Mean,
  Affine Recalibration, Global Stacking, TabM[G] Greedy].
  (Adjusted via Step-Down Holm-Bonferroni correction over the 5 secondary tests).

Statistical Decision Framework:
- Equivalence Margin: [-0.5%, +0.5%]. If the 95% bootstrap CI falls entirely within [-0.5%, +0.5%],
  we conclude practical equivalence with uniform averaging (confirmatory negative result).
- Tie Rule: Based on observed seed standard deviation (sigma_seed). Any mean relative gain
  with |delta| <= sigma_seed is classified as a tie (indistinguishable from seed noise).

Dataset Partitioning:
- Development Set: California Housing (used strictly during gate development and grid selection).
- Confirmatory Set (13 Real-World Benchmarks):
  - 8 TabReD Real-world Temporal Drift Benchmarks (official fixed temporal splits):
    weather, sberbank-housing, cooking-time, delivery-eta, ecom-offers,
    homecredit-default, homesite-insurance, maps-routing.
  - 5 Standard Real-world Benchmarks from Paper:
    diamond, house, black-friday, otto, covtype2.
"""

from typing import List, Dict, Any

# Pre-registered seed set (5 seeds per dataset)
SEEDS = [0, 1, 2, 3, 4]

# Architecture Specification:
# Single fixed BatchEnsemble configuration matching the standard TabM MLP backbone:
# arch_type='tabm', n_blocks=3, d_block=256, dropout=0.2, lr=0.001, weight_decay=0.03.
PRIMARY_ARCH = 'tabm'
SHARE_TRAINING_BATCHES = False  # Real TabM with independent per-head batch permutations

# Development dataset (frozen single dataset for sanity checking)
DEVELOPMENT_DATASETS = [
    'california'
]

# PRIMARY HYPOTHESIS FAMILY:
# 8 TabReD Real-world Temporal Drift Benchmarks (official fixed temporal splits):
# All tasks are regression (5) or binary classification (3). (Multiclass otto & covtype dropped).
PRIMARY_TABRED_DATASETS = [
    'tabred/weather',
    'tabred/sberbank-housing',
    'tabred/cooking-time',
    'tabred/delivery-eta',
    'tabred/ecom-offers',
    'tabred/homecredit-default',
    'tabred/homesite-insurance',
    'tabred/maps-routing',
]

# SECONDARY CONTROL FAMILY (No-Shift Benchmarks):
# 3 Standard Real-world Paper Benchmarks evaluated under random splits (no distribution shift):
SECONDARY_CONTROL_DATASETS = [
    'diamond',
    'house',
    'black-friday'
]

# All Evaluated Benchmarks
CONFIRMATORY_DATASETS = PRIMARY_TABRED_DATASETS + SECONDARY_CONTROL_DATASETS

# Hierarchy of Evaluated Aggregators
AGGREGATORS = [
    'Uniform Mean',         # Official TabM (1/k sum)
    'Exact Median',         # Non-parametric robust median (sorted middle average)
    'Trimmed Mean',         # Symmetric trimmed mean (p tuned on val)
    'Affine Recalibration', # y ≈ a * mu + b (controls for temporal drift bias)
    'Global Stacking',      # Static non-negative weights on simplex (SLSQP analytic)
    'TabM[G] Greedy',       # Official Appendix D.6 forward selection
    'Adaptive Gate (Ours)'  # Permutation-equivariant bounded score net
]

# Statistical Parameters
ALPHA_PRIMARY = 0.05
ALPHA_SECONDARY = 0.05
WILCOXON_ZERO_METHOD = 'zsplit'

# Equivalence Margin:
# Justified against the paper's ~2.15% average TabM-MLP gain and within observed seed noise:
# Margin = +/- 0.50% relative change.
# If the 95% bootstrap CI falls strictly within [-0.50%, +0.50%], equivalence with uniform is concluded.
EQUIVALENCE_MARGIN = 0.50

N_BOOTSTRAPS = 2000

# Aggregation Timing Protocol
LATENCY_WARMUP_RUNS = 10
LATENCY_TIMED_RUNS = 100

