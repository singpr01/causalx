import numpy as np
import pandas as pd
import pytest

from causalx.analysis.estimators import diff_in_means, diff_in_proportions
from causalx.data.schemas import CausalEstimate

# -----------------------
# Helpers
# -----------------------

def simulate_continuous_data(
    n: int = 3000,
    effects: dict = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate multi-arm continuous outcome data.

    effects: dict mapping arm -> mean shift relative to control
    """
    if effects is None:
        effects = {"control": 0.0, "A": 0.5, "B": -0.3}

    rng = np.random.default_rng(seed)

    arms = list(effects.keys())
    arm_assign = rng.choice(arms, size=n, replace=True)

    y = np.array(
        [effects[a] + rng.normal(0, 1.0) for a in arm_assign]
    )

    return pd.DataFrame(
        {
            "arm": arm_assign,
            "y": y,
        }
    )


def simulate_binary_data(
    n: int = 4000,
    probs: dict = None,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Simulate multi-arm binary outcome data.

    probs: dict mapping arm -> success probability
    """
    if probs is None:
        probs = {"control": 0.10, "A": 0.15, "B": 0.05}

    rng = np.random.default_rng(seed)

    arms = list(probs.keys())
    arm_assign = rng.choice(arms, size=n, replace=True)

    y = np.array(
        [rng.random() < probs[a] for a in arm_assign],
        dtype=int,
    )

    return pd.DataFrame(
        {
            "arm": arm_assign,
            "y": y,
        }
    )


# -----------------------
# diff_in_means tests
# -----------------------

def test_diff_in_means_returns_one_estimate_per_non_reference_arm():
    df = simulate_continuous_data()
    res = diff_in_means(df, outcome_col="y", treatment_col="arm", reference="control")

    assert isinstance(res, list)
    assert len(res) == 2
    assert all(isinstance(r, CausalEstimate) for r in res)


def test_diff_in_means_estimate_direction_matches_truth():
    effects = {"control": 0.0, "A": 0.7, "B": -0.4}
    df = simulate_continuous_data(effects=effects)

    res = diff_in_means(df, outcome_col="y", treatment_col="arm", reference="control")

    est_map = {r.contrast: r for r in res}

    assert est_map["A - control"].estimate > 0
    assert est_map["B - control"].estimate < 0


def test_diff_in_means_se_positive_and_ci_well_formed():
    df = simulate_continuous_data()

    res = diff_in_means(df, outcome_col="y", treatment_col="arm")

    for r in res:
        assert r.std_error > 0
        assert r.ci_low < r.estimate < r.ci_high


def test_diff_in_means_handles_missing_values():
    df = simulate_continuous_data()
    df.loc[:50, "y"] = np.nan

    res = diff_in_means(df, outcome_col="y", treatment_col="arm")

    assert len(res) == 2


# -----------------------
# diff_in_proportions tests
# -----------------------

def test_diff_in_proportions_returns_correct_number_of_contrasts():
    df = simulate_binary_data()
    res = diff_in_proportions(df, outcome_col="y", treatment_col="arm", reference="control")

    assert len(res) == 2
    assert all(isinstance(r, CausalEstimate) for r in res)


def test_diff_in_proportions_estimate_direction_matches_truth():
    probs = {"control": 0.10, "A": 0.18, "B": 0.04}
    df = simulate_binary_data(probs=probs)

    res = diff_in_proportions(df, outcome_col="y", treatment_col="arm", reference="control")

    est_map = {r.contrast: r for r in res}

    assert est_map["A - control"].estimate > 0
    assert est_map["B - control"].estimate < 0


def test_diff_in_proportions_se_and_ci_valid():
    df = simulate_binary_data()

    res = diff_in_proportions(df, outcome_col="y", treatment_col="arm")

    for r in res:
        assert r.std_error > 0
        assert r.ci_low < r.estimate < r.ci_high


def test_diff_in_proportions_rejects_non_binary_outcome():
    df = simulate_binary_data()
    df["y"] = 2  # invalid

    with pytest.raises(ValueError):
        diff_in_proportions(df, outcome_col="y", treatment_col="arm")

