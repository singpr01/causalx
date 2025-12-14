from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy.stats import norm

from causalx.data.schemas import CausalEstimate


def _drop_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.dropna(subset=cols)
    if out.empty:
        raise ValueError("No rows remain after dropping missing values in required columns.")
    return out


def _choose_reference(arms: list[Hashable], reference: Hashable | None) -> Hashable:
    if len(arms) < 2:
        raise ValueError("Need at least 2 treatment arms to compute contrasts.")

    if reference is not None:
        if reference not in arms:
            raise ValueError(f"reference arm '{reference}' not present in treatment column.")
        return reference

    # Stable default: try to sort, else fall back to first observed
    try:
        return sorted(arms)[0]
    except Exception:
        return arms[0]


def _z_alpha(alpha: float, two_sided: bool) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1). Got {alpha}.")
    alpha_eff = alpha / 2.0 if two_sided else alpha
    return float(norm.ppf(1.0 - alpha_eff))


def _ci_from_est_se(est: float, se: float, z: float, two_sided: bool) -> tuple[float, float]:
    if se < 0:
        raise ValueError("std_error must be non-negative.")
    if se == 0:
        return est, est
    # For one-sided, we still return a two-number interval for convenience:
    # - greater: (lower, +inf)
    # - less: (-inf, upper)
    # But to keep v0.1 simple, we return symmetric normal-approx bounds even if one-sided.
    # Users can interpret one-sided via alpha/two_sided metadata.
    return float(est - z * se), float(est + z * se)


def diff_in_means(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str = "treatment",
    reference: Hashable | None = None,
    alpha: float = 0.05,
    two_sided: bool = True,
    ddof: int = 1,
) -> list[CausalEstimate]:
    """
    Difference in means for multi-arm treatments: estimates each arm vs reference.

    Returns a list of CausalEstimate objects, one per non-reference arm.

    Assumptions (v0.1):
    - Independent units
    - Normal approximation for CI using estimated SE
    - Uses sample variance with ddof (default 1)
    - Drops rows with missing outcome or treatment
    """
    if outcome_col not in df.columns:
        raise ValueError(f"outcome_col '{outcome_col}' not found in df.")
    if treatment_col not in df.columns:
        raise ValueError(f"treatment_col '{treatment_col}' not found in df.")
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1.")

    d = _drop_missing(df, [outcome_col, treatment_col])

    # Convert outcome to numeric
    y = pd.to_numeric(d[outcome_col], errors="coerce")
    d = d.copy()
    d[outcome_col] = y
    d = _drop_missing(d, [outcome_col, treatment_col])

    arms = list(pd.unique(d[treatment_col]))
    ref = _choose_reference(arms, reference)
    z = _z_alpha(alpha, two_sided)

    # Precompute group stats
    grouped = d.groupby(treatment_col, dropna=False)[outcome_col]
    means = grouped.mean()
    vars_ = grouped.var(ddof=ddof)  # sample variance
    ns = grouped.count()

    if ns.loc[ref] < 2:
        raise ValueError(f"Reference arm '{ref}' has too few non-missing outcomes (n={int(ns.loc[ref])}).")

    results: list[CausalEstimate] = []
    ref_mean = float(means.loc[ref])
    ref_var = float(vars_.loc[ref]) if not np.isnan(vars_.loc[ref]) else 0.0
    ref_n = int(ns.loc[ref])

    for arm in arms:
        if arm == ref:
            continue

        n_t = int(ns.loc[arm])
        if n_t < 2:
            raise ValueError(f"Arm '{arm}' has too few non-missing outcomes (n={n_t}).")

        mean_t = float(means.loc[arm])
        var_t = float(vars_.loc[arm]) if not np.isnan(vars_.loc[arm]) else 0.0

        est = mean_t - ref_mean
        se = float(np.sqrt(var_t / n_t + ref_var / ref_n))
        ci_low, ci_high = _ci_from_est_se(est, se, z, two_sided)

        results.append(
            CausalEstimate(
                estimand="ATE",
                contrast=f"{arm} - {ref}",
                estimate=float(est),
                std_error=float(se),
                ci_low=ci_low,
                ci_high=ci_high,
                alpha=float(alpha),
                n=int(n_t + ref_n),
                method="diff_in_means",
                meta={
                    "outcome_col": outcome_col,
                    "treatment_col": treatment_col,
                    "reference": ref,
                    "two_sided": two_sided,
                    "ddof": ddof,
                    "n_arm": n_t,
                    "n_ref": ref_n,
                },
            )
        )

    return results


def diff_in_proportions(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    treatment_col: str = "treatment",
    reference: Hashable | None = None,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> list[CausalEstimate]:
    """
    Difference in proportions for a binary outcome across multi-arm treatments.

    The outcome is interpreted as 0/1 (or boolean). Values are coerced to numeric and must
    be in {0, 1} after dropping missing values.

    Returns a list of CausalEstimate objects, one per non-reference arm.

    Assumptions (v0.1):
    - Independent units
    - Normal approximation CI for difference in proportions
    - Drops rows with missing outcome or treatment
    """
    if outcome_col not in df.columns:
        raise ValueError(f"outcome_col '{outcome_col}' not found in df.")
    if treatment_col not in df.columns:
        raise ValueError(f"treatment_col '{treatment_col}' not found in df.")

    d = _drop_missing(df, [outcome_col, treatment_col])

    # Accept bool or numeric-like 0/1
    y = d[outcome_col]
    if pd.api.types.is_bool_dtype(y):
        y_num = y.astype(int)
    else:
        y_num = pd.to_numeric(y, errors="coerce")

    d = d.copy()
    d[outcome_col] = y_num
    d = _drop_missing(d, [outcome_col, treatment_col])

    # Validate binary values
    vals = set(pd.unique(d[outcome_col]))
    if not vals.issubset({0, 1}):
        raise ValueError(
            f"Outcome '{outcome_col}' must be binary (0/1 or bool). Found values: {sorted(vals)}"
        )

    arms = list(pd.unique(d[treatment_col]))
    ref = _choose_reference(arms, reference)
    z = _z_alpha(alpha, two_sided)

    grouped = d.groupby(treatment_col, dropna=False)[outcome_col]
    ns = grouped.count()
    sums = grouped.sum()
    props = sums / ns

    if ns.loc[ref] < 1:
        raise ValueError(f"Reference arm '{ref}' has no observations.")

    ref_p = float(props.loc[ref])
    ref_n = int(ns.loc[ref])

    results: list[CausalEstimate] = []
    for arm in arms:
        if arm == ref:
            continue

        n_t = int(ns.loc[arm])
        if n_t < 1:
            raise ValueError(f"Arm '{arm}' has no observations.")

        p_t = float(props.loc[arm])
        est = p_t - ref_p

        se = float(np.sqrt((p_t * (1 - p_t)) / n_t + (ref_p * (1 - ref_p)) / ref_n))
        ci_low, ci_high = _ci_from_est_se(est, se, z, two_sided)

        results.append(
            CausalEstimate(
                estimand="ATE",
                contrast=f"{arm} - {ref}",
                estimate=float(est),
                std_error=float(se),
                ci_low=ci_low,
                ci_high=ci_high,
                alpha=float(alpha),
                n=int(n_t + ref_n),
                method="diff_in_proportions",
                meta={
                    "outcome_col": outcome_col,
                    "treatment_col": treatment_col,
                    "reference": ref,
                    "two_sided": two_sided,
                    "n_arm": n_t,
                    "n_ref": ref_n,
                    "p_arm": p_t,
                    "p_ref": ref_p,
                },
            )
        )

    return results
