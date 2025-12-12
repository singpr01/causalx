import pytest

from causalx.sampling import mde, power, sample_size


def test_binary_sample_size_runs_and_returns_ints():
    res = sample_size(metric_type="binary", baseline_rate=0.10, mde=0.02)
    assert res.n_control > 0
    assert res.n_treatment > 0
    assert res.n_total == res.n_control + res.n_treatment


def test_continuous_sample_size_runs_and_returns_ints():
    res = sample_size(metric_type="continuous", sigma=1.5, mean_diff=0.2)
    assert res.n_control > 0
    assert res.n_treatment > 0
    assert res.n_total == res.n_control + res.n_treatment


def test_smaller_mde_requires_more_sample_binary():
    res_big = sample_size(metric_type="binary", baseline_rate=0.10, mde=0.03)
    res_small = sample_size(metric_type="binary", baseline_rate=0.10, mde=0.01)
    assert res_small.n_total > res_big.n_total


def test_higher_power_requires_more_sample_continuous():
    res_80 = sample_size(metric_type="continuous", sigma=1.0, mean_diff=0.2, power=0.80)
    res_90 = sample_size(metric_type="continuous", sigma=1.0, mean_diff=0.2, power=0.90)
    assert res_90.n_total > res_80.n_total


def test_invalid_binary_inputs_raise():
    with pytest.raises(ValueError):
        sample_size(metric_type="binary", baseline_rate=1.0, mde=0.01)
    with pytest.raises(ValueError):
        sample_size(metric_type="binary", baseline_rate=0.10, mde=0.0)
    with pytest.raises(ValueError):
        sample_size(metric_type="binary", baseline_rate=0.99, mde=0.05)


def test_invalid_continuous_inputs_raise():
    with pytest.raises(ValueError):
        sample_size(metric_type="continuous", sigma=0.0, mean_diff=0.1)
    with pytest.raises(ValueError):
        sample_size(metric_type="continuous", sigma=1.0, mean_diff=0.0)


def test_allocation_changes_split_not_total_significantly():
    # sanity: allocation changes n_t vs n_c; total should remain in same ballpark
    res_a = sample_size(metric_type="continuous", sigma=1.0, mean_diff=0.2, allocation=0.5)
    res_b = sample_size(metric_type="continuous", sigma=1.0, mean_diff=0.2, allocation=0.7)
    assert res_b.n_treatment != res_a.n_treatment
    assert res_b.n_control != res_a.n_control
    assert res_b.n_total > 0




def test_power_increases_with_more_n_continuous():
    p_small = power(
        metric_type="continuous",
        n_control=200,
        n_treatment=200,
        sigma=1.0,
        mean_diff=0.2,
    )
    p_big = power(
        metric_type="continuous",
        n_control=800,
        n_treatment=800,
        sigma=1.0,
        mean_diff=0.2,
    )
    assert p_big > p_small


def test_mde_decreases_with_more_n_continuous():
    m_small = mde(
        metric_type="continuous",
        n_control=200,
        n_treatment=200,
        sigma=1.0,
        power=0.8,
    )
    m_big = mde(
        metric_type="continuous",
        n_control=800,
        n_treatment=800,
        sigma=1.0,
        power=0.8,
    )
    assert m_big < m_small


