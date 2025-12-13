from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from causalx.data.schemas import EffectGridResult, SimulatedPowerResult


def _validate_direction(direction: str) -> None:
    if direction not in {"greater", "less", "two-sided"}:
        raise ValueError("direction must be one of {'greater', 'less', 'two-sided'}.")


def _empirical_pvalue(null_stats: np.ndarray, alt_stat: float, direction: str) -> float:
    """
    Empirical p-value based on null distribution of a test statistic.
    """
    if direction == "greater":
        return float(np.mean(null_stats >= alt_stat))
    if direction == "less":
        return float(np.mean(null_stats <= alt_stat))
    # two-sided: compare absolute statistics
    return float(np.mean(np.abs(null_stats) >= abs(alt_stat)))


def _summarize_stats(x: np.ndarray) -> dict[str, float]:
    qs = np.quantile(x, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q50": float(qs[2]),
        "q95": float(qs[3]),
        "q99": float(qs[4]),
    }


def simulated_power(
    *,
    n: int,
    simulate_data: Callable[[int, dict[str, Any], np.random.Generator], Any],
    null_effect: dict[str, Any],
    alt_effect: dict[str, Any],
    alpha: float = 0.05,
    npower: int = 2000,
    ndistr: int = 4000,
    direction: str = "two-sided",
    seed: int | None = None,
    test_statistic: Callable[[Any], float] | None = None,
    p_value: Callable[[Any], float] | None = None,
    return_null_stats: bool = False,
) -> SimulatedPowerResult:
    """
    Estimate power by simulation.

    Provide either:
      - test_statistic: builds an empirical null distribution (Algorithm 1 style)
      - p_value: uses direct p-values (skips ndistr)

    simulate_data(n, effect, rng) should return a dataset object consumable by
    test_statistic or p_value.
    """
    if n <= 1:
        raise ValueError("n must be > 1.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if npower <= 0:
        raise ValueError("npower must be > 0.")
    if ndistr <= 0:
        raise ValueError("ndistr must be > 0.")
    _validate_direction(direction)

    if p_value is None and test_statistic is None:
        raise ValueError("Provide either p_value or test_statistic.")
    if p_value is not None and test_statistic is not None:
        # Prefer p_value if both provided; avoids extra null simulations.
        test_statistic = None

    rng = np.random.default_rng(seed)

    # Mode B: direct p-value computation
    if p_value is not None:
        rejects = 0
        for _ in range(npower):
            data = simulate_data(n, alt_effect, rng)
            p = float(p_value(data))
            if p <= alpha:
                rejects += 1
        power_hat = rejects / npower
        mc_se = float(np.sqrt(power_hat * (1.0 - power_hat) / npower))
        return SimulatedPowerResult(
            n=n,
            alpha=alpha,
            npower=npower,
            ndistr=0,
            direction=direction,
            power=float(power_hat),
            mc_se=mc_se,
            n_reject=rejects,
            null_stat_summary=None,
            meta={"mode": "p_value"},
        )

    # Mode A: empirical null distribution of the test statistic
    null_stats = np.empty(ndistr, dtype=float)
    for i in range(ndistr):
        data0 = simulate_data(n, null_effect, rng)
        null_stats[i] = float(test_statistic(data0))  # type: ignore[arg-type]

    rejects = 0
    for _ in range(npower):
        data1 = simulate_data(n, alt_effect, rng)
        t_alt = float(test_statistic(data1))  # type: ignore[arg-type]
        p = _empirical_pvalue(null_stats, t_alt, direction)
        if p <= alpha:
            rejects += 1

    power_hat = rejects / npower
    mc_se = float(np.sqrt(power_hat * (1.0 - power_hat) / npower))

    meta: dict[str, Any] = {"mode": "empirical_null"}
    if return_null_stats:
        meta["null_stats"] = null_stats  # beware: large object

    return SimulatedPowerResult(
        n=n,
        alpha=alpha,
        npower=npower,
        ndistr=ndistr,
        direction=direction,
        power=float(power_hat),
        mc_se=mc_se,
        n_reject=rejects,
        null_stat_summary=_summarize_stats(null_stats),
        meta=meta,
    )


def simulated_power_grid(
    *,
    n: int,
    effects: list[dict[str, Any]],
    simulate_data: Callable[[int, dict[str, Any], np.random.Generator], Any],
    null_effect: dict[str, Any],
    alpha: float = 0.05,
    npower: int = 2000,
    ndistr: int = 4000,
    direction: str = "two-sided",
    seed: int | None = None,
    test_statistic: Callable[[Any], float] | None = None,
    p_value: Callable[[Any], float] | None = None,
) -> EffectGridResult:
    """
    Compute simulated power across a list of alternative effect specifications.
    """
    results: list[SimulatedPowerResult] = []
    # deterministic but different streams per effect
    base_seed = seed if seed is not None else None

    for k, alt_effect in enumerate(effects):
        eff_seed = None if base_seed is None else int(base_seed + k)
        res = simulated_power(
            n=n,
            simulate_data=simulate_data,
            null_effect=null_effect,
            alt_effect=alt_effect,
            alpha=alpha,
            npower=npower,
            ndistr=ndistr,
            direction=direction,
            seed=eff_seed,
            test_statistic=test_statistic,
            p_value=p_value,
        )
        results.append(res)

    return EffectGridResult(effects=effects, results=results)
