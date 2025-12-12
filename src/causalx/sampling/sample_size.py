from __future__ import annotations

from math import ceil
from typing import Literal

from scipy.stats import norm

from causalx.data.schemas import SampleSizeResult

MetricType = Literal["binary", "continuous"]


def _validate_common(alpha: float, power: float, allocation: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1). Got {alpha}.")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1). Got {power}.")
    if not (0.0 < allocation < 1.0):
        raise ValueError(f"allocation must be in (0, 1). Got {allocation}.")


def _z_values(alpha: float, power: float, two_sided: bool) -> tuple[float, float]:
    alpha_eff = alpha / 2.0 if two_sided else alpha
    z_alpha = norm.ppf(1.0 - alpha_eff)
    z_power = norm.ppf(power)
    return z_alpha, z_power


def _sample_size_continuous(
    *,
    sigma: float,
    mean_diff: float,
    alpha: float,
    power: float,
    two_sided: bool,
    allocation: float,
) -> tuple[int, int]:
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0. Got {sigma}.")
    if mean_diff == 0:
        raise ValueError("mean_diff must be non-zero.")
    z_alpha, z_power = _z_values(alpha, power, two_sided)

    # Two-sample difference in means, normal approximation
    # Var(diff) = sigma^2/n_t + sigma^2/n_c = sigma^2 * (1/n_t + 1/n_c)
    # With allocation a = n_t / (n_t + n_c), we have:
    # 1/n_t + 1/n_c = 1/(aN) + 1/((1-a)N) = (1/a + 1/(1-a))/N
    a = allocation
    scale = (1.0 / a) + (1.0 / (1.0 - a))

    n_total = (scale * (sigma**2) * (z_alpha + z_power) ** 2) / (mean_diff**2)
    n_total_int = ceil(n_total)

    n_t = ceil(a * n_total_int)
    n_c = n_total_int - n_t
    if n_c <= 0:  # just in case of extreme rounding
        n_c = 1
        n_t = n_total_int - 1
    return n_c, n_t


def _sample_size_binary(
    *,
    baseline_rate: float,
    mde: float,
    alpha: float,
    power: float,
    two_sided: bool,
    allocation: float,
) -> tuple[int, int]:
    if not (0.0 < baseline_rate < 1.0):
        raise ValueError(f"baseline_rate must be in (0, 1). Got {baseline_rate}.")
    if mde == 0:
        raise ValueError("mde must be non-zero.")
    p0 = baseline_rate
    p1 = p0 + mde
    if not (0.0 < p1 < 1.0):
        raise ValueError(
            f"baseline_rate + mde must be in (0, 1). Got baseline_rate={p0}, mde={mde}, p1={p1}."
        )

    z_alpha, z_power = _z_values(alpha, power, two_sided)
    a = allocation

    # Normal approximation for difference in proportions
    # Var(diff) = p1(1-p1)/n_t + p0(1-p0)/n_c
    # with n_t = aN, n_c = (1-a)N
    var_term = (p1 * (1 - p1)) / a + (p0 * (1 - p0)) / (1.0 - a)

    n_total = var_term * (z_alpha + z_power) ** 2 / (mde**2)
    n_total_int = ceil(n_total)

    n_t = ceil(a * n_total_int)
    n_c = n_total_int - n_t
    if n_c <= 0:
        n_c = 1
        n_t = n_total_int - 1
    return n_c, n_t


def _power_continuous(
    *,
    n_control: int,
    n_treatment: int,
    sigma: float,
    mean_diff: float,
    alpha: float,
    two_sided: bool,
) -> float:
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("n_control and n_treatment must be > 0.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if mean_diff == 0:
        return 0.0

    se = sigma * ((1.0 / n_treatment) + (1.0 / n_control)) ** 0.5
    z_alpha, _ = _z_values(alpha, power=0.5, two_sided=two_sided)  # z_alpha only
    z = abs(mean_diff) / se

    if two_sided:
        # Reject if |Z| > z_alpha, with Z ~ N(z, 1)
        # Power = P(Z > z_alpha) + P(Z < -z_alpha)
        return float(1.0 - norm.cdf(z_alpha - z) + norm.cdf(-z_alpha - z))

    # one-sided (greater): reject if Z > z_alpha
    return float(1.0 - norm.cdf(z_alpha - z))


def _power_binary(
    *,
    n_control: int,
    n_treatment: int,
    baseline_rate: float,
    mde: float,
    alpha: float,
    two_sided: bool,
) -> float:
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("n_control and n_treatment must be > 0.")
    p0 = baseline_rate
    p1 = p0 + mde
    if not (0.0 < p0 < 1.0) or not (0.0 < p1 < 1.0):
        raise ValueError("baseline_rate and baseline_rate+mde must be in (0, 1).")

    # Standard error under alternative
    se = ((p1 * (1 - p1) / n_treatment) + (p0 * (1 - p0) / n_control)) ** 0.5
    z_alpha, _ = _z_values(alpha, power=0.5, two_sided=two_sided)  # z_alpha only
    z = abs(mde) / se

    if two_sided:
        return float(1.0 - norm.cdf(z_alpha - z) + norm.cdf(-z_alpha - z))

    return float(1.0 - norm.cdf(z_alpha - z))



def power(
    *,
    metric_type: MetricType,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
    two_sided: bool = True,
    # binary:
    baseline_rate: float | None = None,
    mde: float | None = None,
    # continuous:
    sigma: float | None = None,
    mean_diff: float | None = None,
) -> float:
    """Compute achieved power for a fixed sample size under a normal approximation."""
    if metric_type == "binary":
        if baseline_rate is None or mde is None:
            raise ValueError("For metric_type='binary', baseline_rate and mde are required.")
        return _power_binary(
            n_control=n_control,
            n_treatment=n_treatment,
            baseline_rate=baseline_rate,
            mde=mde,
            alpha=alpha,
            two_sided=two_sided,
        )

    if metric_type == "continuous":
        if sigma is None or mean_diff is None:
            raise ValueError("For metric_type='continuous', sigma and mean_diff are required.")
        return _power_continuous(
            n_control=n_control,
            n_treatment=n_treatment,
            sigma=sigma,
            mean_diff=mean_diff,
            alpha=alpha,
            two_sided=two_sided,
        )

    raise ValueError(f"Unsupported metric_type '{metric_type}'. Use 'binary' or 'continuous'.")


def mde(
    *,
    metric_type: MetricType,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    # binary:
    baseline_rate: float | None = None,
    # continuous:
    sigma: float | None = None,
) -> float:
    """
    Compute minimum detectable effect (MDE) for fixed sample size.
    For binary metrics: returns absolute lift in probability.
    For continuous metrics: returns difference in means.
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("n_control and n_treatment must be > 0.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if not (0.0 < power < 1.0):
        raise ValueError("power must be in (0, 1).")

    z_alpha, z_power = _z_values(alpha, power, two_sided)

    if metric_type == "continuous":
        if sigma is None or sigma <= 0:
            raise ValueError("For metric_type='continuous', sigma must be provided and > 0.")
        se = sigma * ((1.0 / n_treatment) + (1.0 / n_control)) ** 0.5
        return float((z_alpha + z_power) * se)

    if metric_type == "binary":
        if baseline_rate is None:
            raise ValueError("For metric_type='binary', baseline_rate is required.")
        p0 = baseline_rate
        if not (0.0 < p0 < 1.0):
            raise ValueError("baseline_rate must be in (0, 1).")

        # Conservative-ish SE using p0 for both arms (common planning approximation)
        se = ((p0 * (1 - p0) / n_treatment) + (p0 * (1 - p0) / n_control)) ** 0.5
        return float((z_alpha + z_power) * se)

    raise ValueError(f"Unsupported metric_type '{metric_type}'. Use 'binary' or 'continuous'.")



def sample_size(
    *,
    metric_type: MetricType,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    allocation: float = 0.5,
    # binary:
    baseline_rate: float | None = None,
    mde: float | None = None,
    # continuous:
    sigma: float | None = None,
    mean_diff: float | None = None,
) -> SampleSizeResult:

    """
    Compute required sample size using a normal approximation.

    Supported v0.1:
      - metric_type="binary": two-sample proportions (absolute lift mde, p1 = p0 + mde)
      - metric_type="continuous": two-sample means (difference in means mean_diff, known sigma)

    Parameters
    ----------
    allocation : float
        Fraction assigned to treatment (0 < allocation < 1). Control gets (1-allocation).
    """
    _validate_common(alpha=alpha, power=power, allocation=allocation)

    if metric_type == "binary":
        if baseline_rate is None or mde is None:
            raise ValueError("For metric_type='binary', baseline_rate and mde are required.")
        n_c, n_t = _sample_size_binary(
            baseline_rate=baseline_rate,
            mde=mde,
            alpha=alpha,
            power=power,
            two_sided=two_sided,
            allocation=allocation,
        )
        assumptions = {"p0": baseline_rate, "p1": baseline_rate + mde, "mde": mde}
        return SampleSizeResult(
            metric_type="binary",
            n_control=n_c,
            n_treatment=n_t,
            n_total=n_c + n_t,
            alpha=alpha,
            power=power,
            two_sided=two_sided,
            allocation=allocation,
            baseline_rate=baseline_rate,
            mde=mde,
            method="normal_approx",
            assumptions=assumptions,
        )

    if metric_type == "continuous":
        if sigma is None or mean_diff is None:
            raise ValueError("For metric_type='continuous', sigma and mean_diff are required.")
        n_c, n_t = _sample_size_continuous(
            sigma=sigma,
            mean_diff=mean_diff,
            alpha=alpha,
            power=power,
            two_sided=two_sided,
            allocation=allocation,
        )
        assumptions = {"sigma": sigma, "mean_diff": mean_diff}
        return SampleSizeResult(
            metric_type="continuous",
            n_control=n_c,
            n_treatment=n_t,
            n_total=n_c + n_t,
            alpha=alpha,
            power=power,
            two_sided=two_sided,
            allocation=allocation,
            sigma=sigma,
            mean_diff=mean_diff,
            method="normal_approx",
            assumptions=assumptions,
        )

    raise ValueError(f"Unsupported metric_type '{metric_type}'. Use 'binary' or 'continuous'.")
