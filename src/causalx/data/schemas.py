from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MetricType = Literal["binary", "continuous"]


@dataclass(frozen=True)
class SampleSizeResult:
    metric_type: MetricType
    n_control: int
    n_treatment: int
    n_total: int

    alpha: float
    power: float
    two_sided: bool
    allocation: float  # fraction assigned to treatment

    # binary-specific inputs
    baseline_rate: float | None = None  # p0
    mde: float | None = None  # absolute lift, p1 = p0 + mde

    # continuous-specific inputs
    sigma: float | None = None  # assumed SD
    mean_diff: float | None = None  # difference in means (same role as mde for continuous)

    method: str = "normal_approx"
    assumptions: dict[str, Any] | None = None


@dataclass(frozen=True)
class SimulatedPowerResult:
    n: int
    alpha: float
    npower: int
    ndistr: int
    direction: str  # "greater" | "less" | "two-sided"

    power: float
    mc_se: float
    n_reject: int

    null_stat_summary: dict[str, float] | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class EffectGridResult:
    effects: list[dict[str, Any]]
    results: list[SimulatedPowerResult]


@dataclass(frozen=True)
class CausalEstimate:
    estimand: str  # e.g. "ATE"
    contrast: str  # e.g. "variant_a - control"
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    alpha: float
    n: int
    method: str  # e.g. "diff_in_means"
    meta: dict[str, Any] | None = None
