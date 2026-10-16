# Final Year Project: Adaptive & Robust Aggregation for Deep Ensembles

This repository contains research, codebase, and benchmarking experiments exploring **Post-Hoc Adaptive Aggregation** for deep ensemble architectures:
1. **TabM Adaptive & Robust Ensembling (ICLR 2025 Extension)**: Addressing uniform sub-model averaging in parameter-efficient tabular deep ensembles with uncertainty-weighted evidential gating and post-hoc aggregation.
2. **Multi-Agent Consensus & Minority Protection (ReConcile ACL 2024 Extension)**: Overcoming majority voting failures in collaborative multi-agent debate via dynamic minority-protective gating.

---

## 📁 Repository Structure

```
Final_Year_Project/
├── experiments/                 # Core research implementations & benchmarks
│   ├── gated_tabm.py            # Post-hoc adaptive aggregation algorithms & bounded scoring networks
│   ├── offline_pipeline.py      # Two-stage evaluation benchmark across tabular datasets
│   ├── test_gated_tabm.py       # Unit tests verifying mathematical equivalence & bounds
│   ├── minority_protection_gate.py  # Inference-time minority protection gate for LLM agent debates
│   ├── benchmark_tabm_gating.py # Comparison of uniform vs. adaptive TabM ensembles
│   └── benchmark_reconcile_gate.py # Evaluation benchmark on multi-agent consensus tasks
├── tabm/                        # TabM baseline repository (ICLR 2025)
├── ReConcile/                   # ReConcile baseline repository (ACL 2024)
├── Ideas/                       # Project ideation documents, proposals, and literature reviews
└── TabM_ICLR_2025_Paper.pdf     # Reference publication for TabM
```

---

## 🔬 Core Innovations

### 1. Post-Hoc Robust TabM Aggregator (`experiments/gated_tabm.py`)
- **Bounded Scoring Network**: $c \cdot \tanh(\text{raw}) + \text{clamp}$ preventing activation blowup on out-of-distribution (OOD) shift.
- **Affine Recalibration Baseline**: Accounts for temporal drift and calibration baselines.
- **Official TabM[G] Algorithm**: Appendix D.6 forward selection with exact stopping criteria.
- **Simplex Constrained Stacking**: Optimized convex combinations of member predictions with analytic gradients.

### 2. Multi-Agent Debate Minority Gate (`experiments/minority_protection_gate.py`)
- Guards against dominant majority hallucinations where multiple misaligned agents overpower a single accurate agent.
- Dynamically assigns confidence-weighted veto and agreement scores during consensus rounds.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

### Setup
```bash
# Clone the repository
git clone https://github.com/Vardhanch05/Final_Yr_Project.git
cd Final_Yr_Project

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install torch numpy scipy scikit-learn tabm
```

### Running Tests and Benchmarks
```bash
# Run unit tests for post-hoc aggregation
python experiments/test_gated_tabm.py

# Run offline benchmark pipeline
python experiments/offline_pipeline.py

# Run ReConcile minority protection benchmark
python experiments/benchmark_reconcile_gate.py
```
