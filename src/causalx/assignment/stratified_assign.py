from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd


def _normalize_probs(probs: Sequence[float], k: int) -> np.ndarray:
    if len(probs) != k:
        raise ValueError(f"Length of probs must match number of arms ({k}). Got {len(probs)}.")
    p = np.asarray(probs, dtype=float)
    if np.any(p < 0):
        raise ValueError("All probabilities must be non-negative.")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("At least one probability must be > 0.")
    p = p / s
    if not np.isclose(p.sum(), 1.0):
        p = p / p.sum()
    return p


def _counts_from_probs(n: int, p: np.ndarray) -> np.ndarray:
    """
    Deterministic rounding to integer counts that sum to n.
    Uses largest-remainder method (Hamilton apportionment).
    """
    expected = n * p
    base = np.floor(expected).astype(int)
    remainder = n - int(base.sum())
    if remainder > 0:
        frac = expected - base
        # tie-breaker is stable by index order
        idx = np.argsort(-frac)
        base[idx[:remainder]] += 1
    return base


def assign_treatment_stratified(
    df: pd.DataFrame,
    *,
    strata_cols: str | Sequence[str],
    arms: Sequence[Hashable] = (0, 1),
    probs: Sequence[float] | None = None,
    treatment_col: str = "treatment",
    seed: int | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Assign treatment within strata (blocking), supporting multiple arms.

    Within each stratum, we compute target arm counts based on `probs`,
    then randomly permute assignments to rows in that stratum.

    Parameters
    ----------
    strata_cols:
        Column(s) that define strata. Each unique combination is a stratum.
        Can be a single column name (e.g., "region") or a list of column names.
        Categorical columns (strings) are supported.
    arms:
        Labels for each arm.
    probs:
        Probabilities per arm. If None, uniform.
    seed:
        RNG seed for reproducibility.
    inplace:
        If True, mutate df. Otherwise return a copy.

    Returns
    -------
    DataFrame with an added `treatment_col`.
    """
    if isinstance(strata_cols, str):
        strata_cols = [strata_cols]

    if treatment_col in df.columns:
        raise ValueError(f"Column '{treatment_col}' already exists.")
    if len(strata_cols) == 0:
        raise ValueError("strata_cols must contain at least one column name.")
    missing = [c for c in strata_cols if c not in df.columns]
    if missing:
        raise ValueError(f"strata_cols not found in df: {missing}")
    if len(arms) < 2:
        raise ValueError("arms must contain at least 2 labels.")

    k = len(arms)
    p = np.ones(k) / k if probs is None else _normalize_probs(probs, k)
    rng = np.random.default_rng(seed)

    out = df if inplace else df.copy()
    out[treatment_col] = None  # fill later

    # groupby preserves row indices; sort=False avoids unexpected sorting
    grouped = out.groupby(list(strata_cols), sort=False, dropna=False)

    arms_arr = np.asarray(arms, dtype=object)

    for _, idx in grouped.indices.items():
        idx = np.asarray(idx)
        n = idx.size
        counts = _counts_from_probs(n, p)

        # build assignment vector then shuffle
        assign = np.concatenate([np.repeat(arms_arr[i], counts[i]) for i in range(k)])
        rng.shuffle(assign)

        out.loc[idx, treatment_col] = assign

    return out

