from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _as_numeric_series(x: pd.Series) -> pd.Series:
    """
    Convert common binary/object encodings to numeric when possible.
    - booleans -> 0/1
    - categorical/object -> try to coerce to numeric; if fails, error
    """
    if pd.api.types.is_bool_dtype(x):
        return x.astype(int)

    if pd.api.types.is_numeric_dtype(x):
        return x.astype(float)

    # Attempt to coerce strings/categoricals that are numeric-like.
    coerced = pd.to_numeric(x, errors="coerce")
    if coerced.isna().any():
        raise ValueError(
            f"Non-numeric covariate '{x.name}' contains values that cannot be coerced to numeric. "
            "Encode categoricals before calling smd()."
        )
    return coerced.astype(float)


def _pooled_sd(s1: float, s2: float) -> float:
    return float(np.sqrt(0.5 * (s1**2 + s2**2)))


def smd(
    df: pd.DataFrame,
    *,
    covariate_cols: Sequence[str],
    treatment_col: str = "treatment",
    reference: Hashable | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Compute standardized mean differences (SMD) for covariates across multi-arm treatments.

    Outputs:
    - mean_<arm>, std_<arm>, n_<arm> for each arm
    - smd_<arm>_vs_<ref> for each non-reference arm
    - smd_max_pairwise: maximum absolute SMD across all arm pairs

    Notes:
    - Categorical covariates should be encoded before calling (e.g., one-hot).
    - For binary covariates, pass as 0/1 or bool.

    Parameters
    ----------
    reference:
        Reference arm label. If None, uses the first arm in sorted unique arms.
    """
    if treatment_col not in df.columns:
        raise ValueError(f"'{treatment_col}' column not found in df.")
    missing = [c for c in covariate_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Covariate columns not found in df: {missing}")

    arms = list(pd.unique(df[treatment_col]))
    if len(arms) < 2:
        raise ValueError("Need at least 2 treatment arms to compute SMD.")

    # Choose reference
    if reference is None:
        # stable choice: sort if possible, else keep first observed
        try:
            ref = sorted(arms)[0]
        except Exception:
            ref = arms[0]
    else:
        ref = reference
        if ref not in arms:
            raise ValueError(f"reference arm '{ref}' not present in treatment_col.")

    # Precompute per-arm masks
    arm_masks = {a: (df[treatment_col] == a) for a in arms}

    rows: list[dict[str, Any]] = []

    for col in covariate_cols:
        x = _as_numeric_series(df[col])

        stats = {}
        for a in arms:
            xa = x[arm_masks[a]].dropna()
            stats[a] = {
                "n": int(xa.shape[0]),
                "mean": float(xa.mean()) if xa.shape[0] > 0 else float("nan"),
                "std": float(xa.std(ddof=ddof)) if xa.shape[0] > 1 else 0.0,
            }

        row: dict[str, Any] = {"covariate": col, "reference": ref}

        # Add per-arm summaries
        for a in arms:
            row[f"n_{a}"] = stats[a]["n"]
            row[f"mean_{a}"] = stats[a]["mean"]
            row[f"std_{a}"] = stats[a]["std"]

        # SMD vs reference
        ref_mean = stats[ref]["mean"]
        ref_std = stats[ref]["std"]

        for a in arms:
            if a == ref:
                continue
            s = _pooled_sd(ref_std, stats[a]["std"])
            smd_val = 0.0 if s == 0 else (stats[a]["mean"] - ref_mean) / s
            row[f"smd_{a}_vs_{ref}"] = float(smd_val)

        # Max pairwise SMD
        max_abs = 0.0
        for a1, a2 in combinations(arms, 2):
            s = _pooled_sd(stats[a1]["std"], stats[a2]["std"])
            smd_pair = 0.0 if s == 0 else (stats[a1]["mean"] - stats[a2]["mean"]) / s
            max_abs = max(max_abs, abs(float(smd_pair)))
        row["smd_max_pairwise"] = float(max_abs)

        rows.append(row)

    out = pd.DataFrame(rows).set_index("covariate")
    return out
