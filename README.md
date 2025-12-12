# causalx

**causalx** is a lightweight, modular toolkit for causal inference and experimentation.
It provides clean, reusable building blocks for common workflows such as
sample size planning, treatment assignment, estimation, and diagnostics.

The project is designed to:
- be **practitioner-oriented** (clear APIs, sensible defaults)
- scale from **simple A/B tests to observational causal analysis**
- grow from **open-source tooling into production-grade software**

This repository is currently at **v0.1** and focuses on core primitives.

> ⚠️ **API stability note**  
> `causalx` is under active development. APIs may change between minor versions
> as the project evolves and stabilizes.

---

## Scope (v0.1)

**Implemented / scaffolded**
- Sample size and power calculations
- Treatment assignment (randomized and stratified)
- Core causal estimators
- Balance and diagnostic utilities
- Synthetic data generation for testing and examples
- End-to-end example notebooks

**Out of scope (for now)**
- DAG editors or graphical identification tooling
- Advanced doubly-robust or ML-based estimators
- Production integrations (databases, metrics systems)
- Hosted or UI-based tooling

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

## Quick examples

> The examples below illustrate intended usage. Function signatures may evolve
> during v0.x development.

### Sample size planning

```python
from causalx.sampling.sample_size import sample_size

result = sample_size(
    baseline_rate=0.10,
    mde=0.02,
    alpha=0.05,
    power=0.80
)

print(result)
```

---

### Random assignment

```python
from causalx.assignment.random_assign import assign_treatment

df_assigned = assign_treatment(
    df,
    treatment_col="treatment",
    prob=0.5,
    seed=42
)
```

---

### Estimation

```python
from causalx.analysis.estimators import diff_in_means

estimate = diff_in_means(
    df_assigned,
    outcome_col="outcome",
    treatment_col="treatment"
)

print(estimate)
```

See the `notebooks/` directory for complete, end-to-end workflows
covering simulation, assignment, estimation, and diagnostics.

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

## Roadmap (high level)

* Regression adjustment and CUPED-style estimators
* Overlap and sensitivity diagnostics
* Doubly-robust estimators (AIPW / TMLE-lite)
* Heterogeneous treatment effects (meta-learners)
* Experiment and causal analysis report generation

---

## Contributing

Contributions and feedback are welcome.

This project is early-stage and evolving; interfaces may change as
core abstractions stabilize.

---

## License

MIT License

```

