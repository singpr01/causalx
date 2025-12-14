# causalx

**causalx** is a lightweight, modular Python toolkit for experimentation and causal inference.

It provides clear, reusable building blocks for common workflows such as:
- sample size planning and power analysis
- randomized and stratified treatment assignment
- covariate balance diagnostics
- effect estimation for continuous and binary outcomes

The project emphasizes **clarity, correctness, and reproducibility**, with APIs designed to be easy to read and reason about.

This repository is currently at **v0.1** and focuses on core experimental primitives.

---

## Scope (v0.1)

### Implemented
- **Sampling & Power**
  - Analytic sample size and power calculations
  - Simulation-based power analysis for arbitrary estimators
- **Assignment**
  - Multi-arm random assignment
  - Multi-arm stratified (blocked) assignment
  - Reproducible assignment via seeded RNG
- **Diagnostics**
  - Standardized Mean Differences (SMD) for multi-arm balance checks
- **Estimation**
  - Difference in means (continuous outcomes)
  - Difference in proportions (binary outcomes)
  - Multi-arm contrasts vs a reference arm
- **Testing & Examples**
  - Unit tests for all core modules
  - End-to-end example notebooks

### Explicitly out of scope (for now)
- Graphical causal modeling or DAG tooling
- Advanced doubly-robust or ML-based estimators
- Clustered or hierarchical designs
- Observational causal identification pipelines
- Production system integrations

---


## Project structure

```text
causalx/
├── src/causalx/
│   ├── sampling/          # power, MDE, sample size utilities
│   ├── assignment/        # random + stratified assignment, balance checks
│   ├── analysis/          # estimators and diagnostics
│   ├── data/              # lightweight schemas
│   └── utils/             # simulation and validation helpers
│
├── tests/                 # unit tests
├── notebooks/             # end-to-end examples
└── .github/workflows/     # CI (multi-version Python)

```
The package follows the **src-layout** convention and is tested against
Python **3.11, 3.12, and 3.14** in continuous integration.



## Installation (development)

`causalx` is currently intended for development and experimentation.

```bash
git clone https://github.com/<your-username>/causalx.git
cd causalx
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Example workflows

> The examples below illustrate intended usage. Function signatures may evolve
> during v0.x development.

### Assignment and Balance

```python
from causalx.assignment import assign_treatment, smd

df = assign_treatment(
    df,
    arms=("control", "A", "B"),
    probs=(0.5, 0.25, 0.25),
    seed=42,
    treatment_col="arm",
)

balance = smd(
    df,
    covariate_cols=["age", "prior_spend"],
    treatment_col="arm",
    reference="control",
)

```

---

### Estimation

```python
from causalx.analysis.estimators import diff_in_means

estimates = diff_in_means(
    df,
    outcome_col="revenue",
    treatment_col="arm",
    reference="control",
)

for e in estimates:
    print(e.contrast, e.estimate, (e.ci_low, e.ci_high))

```

---


See the `notebooks/` directory for complete, end-to-end workflows
covering simulation, assignment, estimation, and diagnostics.

The recommended way to explore the library is through the notebooks:

* **01_sample_size.ipynb** — analytic sample size and power

* **02_assignment.ipynb** — multi-arm and stratified assignment + balance

* **03_estimation.ipynb** — effect estimation for continuous and binary outcomes

* **04_simulated_power.ipynb** — simulation-based power analysis

---

## Design principles

* **Explicit assumptions**
  Estimators and diagnostics aim to surface identifying assumptions clearly.

* **Composable primitives**
  Functions are small, reusable, and easy to test or extend.

* **Minimal magic**
  Transparent logic is preferred over opaque abstractions.

* **Research-to-code translation**
  Methods are implemented as faithful translations of standard or published approaches.



---

## License

MIT License



