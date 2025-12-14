import numpy as np
import pandas as pd
import pytest

from causalx.assignment import assign_treatment, assign_treatment_stratified, smd


def _toy_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x_num": rng.normal(size=n),
            "x_bin": rng.random(n) < 0.3,
            "stratum": rng.integers(0, 5, size=n),
        }
    )


def test_assign_treatment_reproducible():
    df = _toy_df(200)
    a = assign_treatment(df, arms=(0, 1, 2), probs=(0.2, 0.5, 0.3), seed=123)
    b = assign_treatment(df, arms=(0, 1, 2), probs=(0.2, 0.5, 0.3), seed=123)
    assert (a["treatment"].values == b["treatment"].values).all()


def test_assign_treatment_probability_roughly_matches():
    df = _toy_df(5000)
    out = assign_treatment(df, arms=("c", "A", "B"), probs=(0.2, 0.5, 0.3), seed=1)
    props = out["treatment"].value_counts(normalize=True).to_dict()
    assert abs(props["c"] - 0.2) < 0.03
    assert abs(props["A"] - 0.5) < 0.03
    assert abs(props["B"] - 0.3) < 0.03


def test_stratified_assignment_respects_strata_and_reproducible():
    df = _toy_df(2000)
    out1 = assign_treatment_stratified(
        df,
        strata_cols=["stratum"],
        arms=(0, 1, 2),
        probs=(0.25, 0.50, 0.25),
        seed=42,
    )
    out2 = assign_treatment_stratified(
        df,
        strata_cols=["stratum"],
        arms=(0, 1, 2),
        probs=(0.25, 0.50, 0.25),
        seed=42,
    )
    assert (out1["treatment"].values == out2["treatment"].values).all()

    # Within each stratum, proportions should be close to target
    for _s, g in out1.groupby("stratum"):
        props = g["treatment"].value_counts(normalize=True).to_dict()
        # small strata have more rounding noise; allow a bit more slack
        assert abs(props.get(0, 0.0) - 0.25) < 0.08
        assert abs(props.get(1, 0.0) - 0.50) < 0.08
        assert abs(props.get(2, 0.0) - 0.25) < 0.08


def test_smd_runs_multi_arm():
    df = _toy_df(1500)
    out = assign_treatment(df, arms=(0, 1, 2), probs=(0.3, 0.4, 0.3), seed=7)

    bal = smd(out, covariate_cols=["x_num", "x_bin"], treatment_col="treatment", reference=0)

    assert "smd_max_pairwise" in bal.columns
    assert any(c.startswith("smd_") for c in bal.columns)
    assert "mean_0" in bal.columns
    assert "mean_1" in bal.columns
    assert "mean_2" in bal.columns


def test_assign_raises_if_treatment_col_exists():
    df = _toy_df(10)
    df2 = df.copy()
    df2["treatment"] = 0
    with pytest.raises(ValueError):
        assign_treatment(df2)


def test_stratified_assignment_accepts_string_strata():
    df = _toy_df(1000)
    # add a categorical strata column
    df = df.assign(region=np.where(df["stratum"] % 2 == 0, "A", "B"))

    out = assign_treatment_stratified(
        df,
        strata_cols="region",
        arms=(0, 1, 2),
        probs=(0.25, 0.50, 0.25),
        seed=1,
    )
    assert "treatment" in out.columns
    assert set(out["treatment"].unique()) <= {0, 1, 2}
