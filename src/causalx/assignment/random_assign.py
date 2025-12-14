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
    # numerical stability
    if not np.isclose(p.sum(), 1.0):
        p = p / p.sum()
    return p


def assign_treatment(
    df: pd.DataFrame,
    *,
    arms: Sequence[Hashable] = (0, 1),
    probs: Sequence[float] | None = None,
    treatment_col: str = "treatment",
    seed: int | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Randomly assign each row to one of multiple treatment arms.

    Parameters
    ----------
    arms:
        Labels for each arm, e.g. (0, 1) or ("control", "A", "B").
    probs:
        Assignment probabilities per arm. If None, uniform.
    seed:
        RNG seed for reproducibility.
    inplace:
        If True, mutate df. Otherwise return a copy.

    Returns
    -------
    DataFrame with an added `treatment_col`.
    """
    if treatment_col in df.columns:
        raise ValueError(f"Column '{treatment_col}' already exists.")
    if len(arms) < 2:
        raise ValueError("arms must contain at least 2 labels.")

    k = len(arms)
    p = np.ones(k) / k if probs is None else _normalize_probs(probs, k)

    rng = np.random.default_rng(seed)
    assignments = rng.choice(np.asarray(arms, dtype=object), size=len(df), p=p)

    out = df if inplace else df.copy()
    out[treatment_col] = assignments
    return out
