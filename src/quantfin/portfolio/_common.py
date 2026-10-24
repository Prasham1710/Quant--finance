"""Shared conventions and input validation for the portfolio package.

Conventions used everywhere in ``quantfin.portfolio``:

* Returns are simple (arithmetic) per-period returns in a ``pd.DataFrame``:
  rows are dates (ascending), columns are asset names.
* Expected returns ``mu`` and covariance matrices ``cov`` are ANNUALIZED
  (``TRADING_DAYS`` periods per year), and so is the risk-free rate ``rf``.
* Weights are a ``pd.Series`` indexed by asset name that sums to 1.
* Public functions accept numpy arrays or pandas objects and return pandas
  objects labelled by asset. Unlabelled numpy input gets labels 0..N-1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252
"""Periods per year used to annualize daily statistics."""

Bounds = tuple[float, float] | Sequence[tuple[float, float]] | None
"""Weight bounds: one (lo, hi) pair for every asset, one pair per asset, or None (unbounded)."""


class OptimizationError(RuntimeError):
    """Raised when a numerical optimizer fails to find a valid solution."""


def as_returns_frame(returns: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Return ``returns`` as a float DataFrame with no missing values.

    Raises:
        ValueError: if the data is not 2-D, is empty, or contains NaN/inf.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    frame = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(np.asarray(returns))
    if frame.ndim != 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"returns must be a non-empty 2-D table, got shape {frame.shape}")
    frame = frame.astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("returns contain NaN or infinite values; clean or align them first")
    return frame


def as_cov_frame(cov: pd.DataFrame | np.ndarray, assets: Sequence | None = None) -> pd.DataFrame:
    """Validate a covariance matrix and return it as a symmetric labelled DataFrame.

    Args:
        cov: square covariance matrix.
        assets: labels to use when ``cov`` is a numpy array (default 0..N-1).

    Raises:
        ValueError: if ``cov`` is not square, not finite, or not symmetric.
    """
    if isinstance(cov, pd.DataFrame):
        if list(cov.index) != list(cov.columns):
            raise ValueError("covariance DataFrame must have identical index and columns")
        labels = list(cov.index)
        values = cov.to_numpy(dtype=float)
    else:
        values = np.asarray(cov, dtype=float)
        labels = list(assets) if assets is not None else list(range(values.shape[0]))
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError(f"covariance must be a non-empty square matrix, got shape {values.shape}")
    if len(labels) != values.shape[0]:
        raise ValueError("number of asset labels does not match covariance size")
    if not np.isfinite(values).all():
        raise ValueError("covariance contains NaN or infinite values")
    scale = max(1.0, float(np.abs(values).max()))
    if not np.allclose(values, values.T, atol=1e-10 * scale, rtol=0.0):
        raise ValueError("covariance matrix must be symmetric")
    values = 0.5 * (values + values.T)
    return pd.DataFrame(values, index=labels, columns=labels)


def as_series(x: pd.Series | np.ndarray | Sequence[float], assets: Sequence) -> pd.Series:
    """Return ``x`` as a float Series aligned to ``assets``.

    A Series is reindexed to ``assets`` (it must contain every asset); an array
    must have one entry per asset and is labelled positionally.

    Raises:
        ValueError: on length mismatch, missing assets, or non-finite values.
    """
    assets = list(assets)
    if isinstance(x, pd.Series):
        missing = [a for a in assets if a not in x.index]
        if missing:
            raise ValueError(f"values missing for assets: {missing}")
        out = x.reindex(assets).astype(float)
    else:
        values = np.asarray(x, dtype=float).reshape(-1)
        if values.shape[0] != len(assets):
            raise ValueError(f"expected {len(assets)} values, got {values.shape[0]}")
        out = pd.Series(values, index=assets)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("values contain NaN or infinite entries")
    return out


def normalize_bounds(bounds: Bounds, n_assets: int) -> list[tuple[float, float]] | None:
    """Expand ``bounds`` to one (lo, hi) pair per asset and check feasibility.

    ``None`` means unbounded (short selling allowed) and is returned unchanged.

    Raises:
        ValueError: if a pair has lo > hi or the bounds make sum(w) == 1 infeasible.
    """
    if bounds is None:
        return None
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and all(np.isscalar(b) or b is None for b in bounds)
    ):
        pairs = [bounds] * n_assets
    else:
        pairs = list(bounds)
        if len(pairs) != n_assets:
            raise ValueError(f"expected {n_assets} bound pairs, got {len(pairs)}")
    out = []
    for lo, hi in pairs:
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        if lo > hi:
            raise ValueError(f"lower bound {lo} exceeds upper bound {hi}")
        out.append((lo, hi))
    lo_sum = sum(lo for lo, _ in out)
    hi_sum = sum(hi for _, hi in out)
    if lo_sum > 1.0 + 1e-12 or hi_sum < 1.0 - 1e-12:
        raise ValueError("bounds are infeasible: weights cannot sum to 1")
    return out
