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

