import numpy as np

from causalx.sampling.simulated_power import simulated_power, simulated_power_grid


def simulate_normal_means(n: int, effect: dict, rng: np.random.Generator):
    """
    Two-arm continuous outcome:
      y = mu_treatment for treated, mu_control for control, plus Normal(0, sigma)
    """
    sigma = effect["sigma"]
    mu_c = effect.get("mu_c", 0.0)
    mu_t = effect.get("mu_t", mu_c)

    n_t = n // 2
    n_c = n - n_t

    y_c = rng.normal(loc=mu_c, scale=sigma, size=n_c)
    y_t = rng.normal(loc=mu_t, scale=sigma, size=n_t)
    return {"y_c": y_c, "y_t": y_t}


def t_stat_diff_in_means(data) -> float:
    y_c = data["y_c"]
    y_t = data["y_t"]
    diff = float(np.mean(y_t) - np.mean(y_c))
    se = float(np.sqrt(np.var(y_t, ddof=1) / len(y_t) + np.var(y_c, ddof=1) / len(y_c)))
    return diff / se


def test_simulated_power_increases_with_effect():
    null = {"mu_c": 0.0, "mu_t": 0.0, "sigma": 1.0}
    alt_small = {"mu_c": 0.0, "mu_t": 0.2, "sigma": 1.0}
    alt_big = {"mu_c": 0.0, "mu_t": 0.6, "sigma": 1.0}

    res_small = simulated_power(
        n=200,
        simulate_data=simulate_normal_means,
        null_effect=null,
        alt_effect=alt_small,
        alpha=0.05,
        npower=600,
        ndistr=800,
        direction="two-sided",
        seed=123,
        test_statistic=t_stat_diff_in_means,
    )

    res_big = simulated_power(
        n=200,
        simulate_data=simulate_normal_means,
        null_effect=null,
        alt_effect=alt_big,
        alpha=0.05,
        npower=600,
        ndistr=800,
        direction="two-sided",
        seed=456,
        test_statistic=t_stat_diff_in_means,
    )

    assert 0.0 <= res_small.power <= 1.0
    assert 0.0 <= res_big.power <= 1.0
    assert res_big.power > res_small.power


def test_simulated_power_grid_runs():
    null = {"mu_c": 0.0, "mu_t": 0.0, "sigma": 1.0}
    effects = [
        {"mu_c": 0.0, "mu_t": 0.1, "sigma": 1.0},
        {"mu_c": 0.0, "mu_t": 0.3, "sigma": 1.0},
    ]

    grid = simulated_power_grid(
        n=150,
        effects=effects,
        simulate_data=simulate_normal_means,
        null_effect=null,
        alpha=0.05,
        npower=300,
        ndistr=500,
        direction="two-sided",
        seed=7,
        test_statistic=t_stat_diff_in_means,
    )

    assert len(grid.effects) == 2
    assert len(grid.results) == 2
    assert grid.results[1].power >= grid.results[0].power
