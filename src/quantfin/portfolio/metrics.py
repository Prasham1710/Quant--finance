"""Performance statistics for simple per-period return series.

Every function takes per-period SIMPLE returns, either as a ``pd.Series`` (one
strategy), a ``pd.DataFrame`` (one column per strategy) or a numpy array (1-D
like a Series, 2-D like a DataFrame). Path-valued statistics
(:func:`cumulative_returns`, :func:`drawdowns`) return the same type they were
given; scalar statistics return a ``float`` for one strategy and a ``pd.Series``
indexed by strategy for a table. Rows are periods in chronological order.

Conventions:

* ``periods_per_year`` (default ``TRADING_DAYS`` = 252) annualizes per-period
  statistics; the risk-free rate ``rf`` is ANNUAL and is de-annualized simply,
  ``rf / periods_per_year`` per period (the same convention as the rest of the
  package, where ``mu = mean * periods_per_year``).
* Wealth starts at 1 *before* the first return, so a loss on the first period
  is already a drawdown.
* Returns must be finite and ``>= -1`` (a simple return below -100% would make
  wealth negative). Missing values are rejected rather than silently skipped:
  align or ``dropna`` the strategies first.
* Ratios whose denominator is zero (zero volatility, no downside, no drawdown)
  are ``+inf`` / ``-inf`` when the numerator is positive / negative and ``nan``
  when it is zero too. "Zero" allows for floating-point rounding (a constant
  series can have a sample std of ~1e-18 instead of 0), and no numpy
  ``RuntimeWarning`` is ever emitted. A volatility needs at least two
  observations (``ddof=1``); with one it is ``nan``.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantfin.portfolio._common import TRADING_DAYS, as_cov_frame, as_series

ReturnsLike = pd.Series | pd.DataFrame | np.ndarray

SUMMARY_COLUMNS: tuple[str, ...] = (
    "CAGR",
    "Volatility",
    "Sharpe",
    "Sortino",
    "Max drawdown",
    "Calmar",
    "VaR 95%",
    "ES 95%",
)
"""Columns of :func:`performance_summary`, in order."""

# A dispersion (std, downside deviation) at or below this fraction of the
# largest absolute return is floating-point noise: the mean of n copies of c is
# not always exactly c, so a constant series can show a std of ~1e-16 * |c|.
_ZERO_RTOL = 1e-12

# Drawdowns and CAGRs are dimensionless fractions of wealth; anything this
# small is rounding in the wealth path (e.g. +10% then -1/11 back to 1.0).
_ZERO_ATOL = 1e-12

# exp(709) is close to the largest finite double; a larger exponent overflows.
_MAX_EXP = 709.0

# T * (1 - level) within this relative distance of an integer is that integer
# (100 * (1 - 0.95) evaluates to 5.000000000000004, not 5).
_INTEGER_RTOL = 1e-9

_WEIGHT_SUM_TOL = 1e-6


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Returns:
    """Returns as a (T, K) float array plus what is needed to label the output."""

    values: np.ndarray
    kind: str  # "series", "frame", "array1d" or "array2d"
    index: pd.Index | None
    columns: pd.Index | None
    name: Hashable

    def scalar(self, out: np.ndarray) -> float | pd.Series:
        """Wrap one value per column: a float for one strategy, else a Series."""
        if self.kind in ("series", "array1d"):
            return float(out[0])
        labels = self.columns if self.columns is not None else pd.RangeIndex(out.shape[0])
        return pd.Series(out, index=labels.copy(), dtype=float)

    def path(self, out: np.ndarray) -> pd.Series | pd.DataFrame | np.ndarray:
        """Wrap a (T, K) result in the input's type."""
        if self.kind == "series":
            return pd.Series(out[:, 0], index=self.index.copy(), name=self.name)
        if self.kind == "frame":
            return pd.DataFrame(out, index=self.index.copy(), columns=self.columns.copy())
        if self.kind == "array1d":
            return out[:, 0].copy()
        return out

    def labels(self) -> list[Hashable]:
        """Row labels for a summary table (one per strategy)."""
        if self.kind == "series":
            return ["strategy" if self.name is None else self.name]
        if self.kind == "frame":
            return list(self.columns)
        if self.kind == "array1d":
            return ["strategy"]
        return list(range(self.values.shape[1]))


def _parse_returns(returns: ReturnsLike) -> _Returns:
    if isinstance(returns, pd.Series):
        values = returns.to_numpy(dtype=float).reshape(-1, 1)
        parsed = _Returns(values, "series", returns.index, None, returns.name)
    elif isinstance(returns, pd.DataFrame):
        values = returns.to_numpy(dtype=float)
        parsed = _Returns(values, "frame", returns.index, returns.columns, None)
    else:
        arr = np.asarray(returns, dtype=float)
        if arr.ndim == 1:
            parsed = _Returns(arr.reshape(-1, 1), "array1d", None, None, None)
        elif arr.ndim == 2:
            parsed = _Returns(arr, "array2d", None, None, None)
        else:
            raise ValueError(f"returns must be 1-D or 2-D, got {arr.ndim} dimensions")
    values = parsed.values
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"returns must be non-empty, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("returns contain NaN or infinite values; align or drop them first")
    if (values < -1.0).any():
        raise ValueError("simple returns cannot be below -100% (-1.0)")
    return parsed


def _check_periods_per_year(periods_per_year: float) -> float:
    ppy = float(periods_per_year)
    if not math.isfinite(ppy) or ppy <= 0:
        raise ValueError(f"periods_per_year must be a positive number, got {periods_per_year!r}")
    return ppy


def _check_rf(rf: float) -> float:
    rf = float(rf)
    if not math.isfinite(rf):
        raise ValueError(f"rf must be finite, got {rf!r}")
    return rf


def _check_level(level: float) -> float:
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be strictly between 0 and 1, got {level!r}")
    return level


# ---------------------------------------------------------------------------
# Array kernels: (T, K) array -> (K,) array. No validation, no labels.
# ---------------------------------------------------------------------------


def _safe_ratio(num: np.ndarray, den: np.ndarray, num_zero: np.ndarray) -> np.ndarray:
    """``num / den`` with den >= 0; den == 0 gives +-inf, or nan if ``num_zero``.

    ``den`` is nan where the ratio is undefined (e.g. a 1-observation std) and
    the result is then nan as well.
    """
    out = np.full(num.shape, np.nan)
    positive = den > 0.0
    np.divide(num, den, out=out, where=positive)
    zero = den == 0.0
    out[zero & ~num_zero & (num > 0.0)] = np.inf
    out[zero & ~num_zero & (num < 0.0)] = -np.inf
    return out


def _std(values: np.ndarray) -> np.ndarray:
    """Sample standard deviation (ddof=1) per column; nan if T < 2; rounding noise -> 0."""
    t = values.shape[0]
    if t < 2:
        return np.full(values.shape[1], np.nan)
    std = values.std(axis=0, ddof=1)
    scale = np.abs(values).max(axis=0)
    return np.where(std <= _ZERO_RTOL * scale, 0.0, std)


def _wiped_out(values: np.ndarray) -> np.ndarray:
    """True from the first -100% return onwards: wealth is 0 for good after it."""
    return np.logical_or.accumulate(values <= -1.0, axis=0)


def _log_wealth(values: np.ndarray) -> np.ndarray:
    """``log W_t = sum_{s<=t} log1p(r_s)``, with r = -1 treated as 0 (callers mask it).

    ``log1p`` rather than ``log(1 + r)`` keeps the precision of tiny returns,
    which ``1 + r`` would round away (1 + 1e-17 == 1).
    """
    return np.cumsum(np.log1p(np.where(values > -1.0, values, 0.0)), axis=0)


def _wealth(values: np.ndarray) -> np.ndarray:
    # Wealth above ~1.8e308 is reported as inf instead of raising an overflow
    # warning; inf * 0 (a wipe-out after that) would be nan, hence the mask.
    with np.errstate(over="ignore", invalid="ignore"):
        wealth = np.cumprod(1.0 + values, axis=0)
    return np.where(_wiped_out(values), 0.0, wealth)


def _cagr(values: np.ndarray, ppy: float) -> np.ndarray:
    wiped_out = (values <= -1.0).any(axis=0)
    # Sum of logs instead of a product: no overflow/underflow on long samples.
    log_wealth = np.log1p(np.where(values > -1.0, values, 0.0)).sum(axis=0)
    exponent = log_wealth * (ppy / values.shape[0])
    cagr = np.expm1(np.minimum(exponent, _MAX_EXP))
    cagr = np.where(exponent > _MAX_EXP, np.inf, cagr)
    return np.where(wiped_out, -1.0, cagr)


def _volatility(values: np.ndarray, ppy: float) -> np.ndarray:
    return _std(values) * math.sqrt(ppy)


def _sharpe(values: np.ndarray, rf: float, ppy: float) -> np.ndarray:
    rf_period = rf / ppy
    mean_excess = (values - rf_period).mean(axis=0)
    scale = np.maximum(np.abs(values).max(axis=0), abs(rf_period))
    num_zero = np.abs(mean_excess) <= _ZERO_RTOL * scale
    return _safe_ratio(mean_excess, _std(values), num_zero) * math.sqrt(ppy)


def _sortino(values: np.ndarray, rf: float, ppy: float) -> np.ndarray:
    rf_period = rf / ppy
    excess = values - rf_period
    mean_excess = excess.mean(axis=0)
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2, axis=0))
    # Scale by the inputs, not the excess: r == rf/P up to rounding is "no downside".
    scale = np.maximum(np.abs(values).max(axis=0), abs(rf_period))
    downside = np.where(downside <= _ZERO_RTOL * scale, 0.0, downside)
    num_zero = np.abs(mean_excess) <= _ZERO_RTOL * scale
    return _safe_ratio(mean_excess, downside, num_zero) * math.sqrt(ppy)


def _drawdowns(values: np.ndarray) -> np.ndarray:
    # In log space: W_t itself overflows to inf once sum log1p(r) > ~709 and
    # inf / inf would be nan, whereas log W_t - log peak_t is always finite.
    log_wealth = _log_wealth(values)
    # Initial wealth 1 (log 0) counts as a peak, so a first-period loss is a drawdown.
    log_peak = np.maximum(np.maximum.accumulate(log_wealth, axis=0), 0.0)
    drawdown = np.expm1(log_wealth - log_peak)
    return np.where(_wiped_out(values), -1.0, drawdown)


def _max_drawdown(values: np.ndarray) -> np.ndarray:
    return _drawdowns(values).min(axis=0)


def _calmar(values: np.ndarray, ppy: float) -> np.ndarray:
    depth = -_max_drawdown(values)
    depth = np.where(depth <= _ZERO_ATOL, 0.0, depth)
    cagr = _cagr(values, ppy)
    return _safe_ratio(cagr, depth, np.abs(cagr) <= _ZERO_ATOL)


def _tail_mass(n_obs: int, level: float) -> float:
    """``T (1 - level)``: the number of observations in the tail (may be fractional)."""
    mass = n_obs * (1.0 - level)
    nearest = round(mass)
    # Never snap down to 0: the tail must keep positive mass (level < 1).
    if nearest >= 1 and abs(mass - nearest) <= _INTEGER_RTOL * mass:
        mass = float(nearest)
    return mass


def _value_at_risk(values: np.ndarray, level: float) -> np.ndarray:
    n_obs = values.shape[0]
    k = min(math.floor(_tail_mass(n_obs, level)), n_obs - 1)
    return -np.sort(values, axis=0)[k]


def _expected_shortfall(values: np.ndarray, level: float) -> np.ndarray:
    ordered = np.sort(values, axis=0)
    mass = _tail_mass(values.shape[0], level)
    whole = math.floor(mass)
    tail_sum = ordered[:whole].sum(axis=0)
    fraction = mass - whole
    if fraction > 0.0:
        tail_sum = tail_sum + fraction * ordered[whole]
    return -tail_sum / mass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cumulative_returns(returns: ReturnsLike) -> pd.Series | pd.DataFrame | np.ndarray:
    """Compounded return since the start: ``prod_{s<=t} (1 + r_s) - 1``.

    Args:
        returns: per-period simple returns (Series, DataFrame or array).

    Returns:
        Same type and shape as ``returns``; add 1 to get the wealth path of $1.
        Wealth beyond the float range (above ~1.8e308) is ``inf``, without a
        warning; after a -100% return the value is -1 for good.

    Raises:
        ValueError: if returns are empty, non-finite or below -1.
    """
    parsed = _parse_returns(returns)
    return parsed.path(_wealth(parsed.values) - 1.0)


def annualized_return(
    returns: ReturnsLike, periods_per_year: float = TRADING_DAYS
) -> float | pd.Series:
    """Geometric average annual growth rate (CAGR).

    ``CAGR = (prod_t (1 + r_t)) ** (P / T) - 1`` with ``T`` observations and
    ``P = periods_per_year``, computed as ``expm1(P/T * sum_t log1p(r_t))``.
    A return of exactly -100% wipes out wealth and gives ``CAGR = -1``.

    Args:
        returns: per-period simple returns.
        periods_per_year: periods in one year (252 for daily data).

    Returns:
        A float for one strategy, a Series indexed by strategy for a table.

    Raises:
        ValueError: on invalid returns or a non-positive ``periods_per_year``.
    """
    ppy = _check_periods_per_year(periods_per_year)
    parsed = _parse_returns(returns)
    return parsed.scalar(_cagr(parsed.values, ppy))


def annualized_volatility(
    returns: ReturnsLike, periods_per_year: float = TRADING_DAYS
) -> float | pd.Series:
    """Annualized sample volatility ``std(r; ddof=1) * sqrt(P)``.

    The square-root-of-time scaling assumes serially uncorrelated returns.
    With a single observation the sample std is undefined and ``nan`` is
    returned.

    Args:
        returns: per-period simple returns.
        periods_per_year: periods in one year (252 for daily data).

    Returns:
        A float for one strategy, a Series indexed by strategy for a table.

    Raises:
        ValueError: on invalid returns or a non-positive ``periods_per_year``.
    """
    ppy = _check_periods_per_year(periods_per_year)
    parsed = _parse_returns(returns)
    return parsed.scalar(_volatility(parsed.values, ppy))


def sharpe_ratio(
    returns: ReturnsLike, rf: float = 0.0, periods_per_year: float = TRADING_DAYS
) -> float | pd.Series:
    """Annualized Sharpe ratio (Sharpe, 1966; 1994).

    ``SR = mean(r_t - rf/P) / std(r_t; ddof=1) * sqrt(P)``: the per-period
    excess return per unit of per-period volatility, scaled by ``sqrt(P)``
    (which assumes i.i.d. returns; Lo, 2002, "The statistics of Sharpe
    ratios", Financial Analysts Journal 58(4), shows the bias under
    autocorrelation).

    References:
        W. F. Sharpe (1966), "Mutual fund performance", Journal of Business
        39(1), 119-138. W. F. Sharpe (1994), "The Sharpe ratio", Journal of
        Portfolio Management 21(1), 49-58.

    Args:
        returns: per-period simple returns.
        rf: ANNUAL risk-free rate, de-annualized as ``rf / P``.
        periods_per_year: periods in one year (252 for daily data).

    Returns:
        A float for one strategy, a Series indexed by strategy for a table.
        Zero volatility gives ``+-inf`` (or ``nan`` if the mean excess return
        is zero too); one observation gives ``nan``.

    Raises:
        ValueError: on invalid returns, non-finite ``rf`` or bad ``periods_per_year``.
    """
    ppy = _check_periods_per_year(periods_per_year)
    rf = _check_rf(rf)
    parsed = _parse_returns(returns)
    return parsed.scalar(_sharpe(parsed.values, rf, ppy))


def sortino_ratio(
    returns: ReturnsLike, rf: float = 0.0, periods_per_year: float = TRADING_DAYS
) -> float | pd.Series:
    """Annualized Sortino ratio with the risk-free rate as target return.

    ``SoR = mean(x_t) / DD * sqrt(P)`` where ``x_t = r_t - rf/P`` and the
    downside deviation is ``DD = sqrt( (1/T) sum_t min(x_t, 0)^2 )``. The
    divisor is the total number of observations ``T`` (the lower partial
    moment of order 2), not the number of losing periods.

    References:
        F. A. Sortino and R. van der Meer (1991), "Downside risk", Journal of
        Portfolio Management 17(4), 27-31. F. A. Sortino and L. N. Price
        (1994), "Performance measurement in a downside risk framework",
        Journal of Investing 3(3), 59-64.

    Args:
        returns: per-period simple returns.
        rf: ANNUAL risk-free rate (the target), de-annualized as ``rf / P``.
        periods_per_year: periods in one year (252 for daily data).

    Returns:
        A float for one strategy, a Series indexed by strategy for a table.
        No period below the target gives ``+inf`` (``nan`` if every excess
        return is zero).

    Raises:
        ValueError: on invalid returns, non-finite ``rf`` or bad ``periods_per_year``.
    """
    ppy = _check_periods_per_year(periods_per_year)
    rf = _check_rf(rf)
    parsed = _parse_returns(returns)
    return parsed.scalar(_sortino(parsed.values, rf, ppy))


def drawdowns(returns: ReturnsLike) -> pd.Series | pd.DataFrame | np.ndarray:
    """Drawdown path: fall of wealth from its running peak, ``W_t / max_{s<=t} W_s - 1``.

    Wealth is ``W_t = prod_{s<=t} (1 + r_s)`` and the peak includes the
    initial wealth ``W_0 = 1``, so every value is ``<= 0`` and a first-period
    loss is a drawdown. It is computed in log space,
    ``expm1(log W_t - max(0, max_{s<=t} log W_s))`` with
    ``log W_t = sum_{s<=t} log1p(r_s)``, so it stays exact (no ``inf / inf``)
    on paths whose wealth would overflow a float.

    Args:
        returns: per-period simple returns.

    Returns:
        Same type and shape as ``returns``, values in ``[-1, 0]``.

    Raises:
        ValueError: if returns are empty, non-finite or below -1.
    """
    parsed = _parse_returns(returns)
    return parsed.path(_drawdowns(parsed.values))


def max_drawdown(returns: ReturnsLike) -> float | pd.Series:
    """Maximum drawdown: the most negative value of :func:`drawdowns`.

    ``MDD = min_t (W_t / max_{s<=t} W_s - 1)`` (e.g. -0.35 for a 35% peak-to-
    trough loss); 0 if wealth never fell below a previous peak. See
    M. Magdon-Ismail and A. F. Atiya (2004), "Maximum drawdown", Risk 17(10).

    Args:
        returns: per-period simple returns.

    Returns:
        A float for one strategy, a Series indexed by strategy for a table.

    Raises:
        ValueError: if returns are empty, non-finite or below -1.
    """
    parsed = _parse_returns(returns)
    return parsed.scalar(_max_drawdown(parsed.values))


def calmar_ratio(returns: ReturnsLike, periods_per_year: float = TRADING_DAYS) -> float | pd.Series:
    """Calmar ratio ``CAGR / |MDD|`` over the whole sample.

    T. W. Young (1991), "Calmar ratio: a smoother tool", Futures 20(1), used a
    trailing 36-month window; here both quantities cover the full sample.

    Args:
        returns: per-period simple returns.
        periods_per_year: periods in one year (252 for daily data).

    Returns:
        A float for one strategy, a Series indexed by strategy for a table.
        No drawdown gives ``+inf`` (``nan`` if the CAGR is zero too).

    Raises:
        ValueError: on invalid returns or a non-positive ``periods_per_year``.
    """
    ppy = _check_periods_per_year(periods_per_year)
    parsed = _parse_returns(returns)
    return parsed.scalar(_calmar(parsed.values, ppy))


def value_at_risk(returns: ReturnsLike, level: float = 0.95) -> float | pd.Series:
    """Historical Value-at-Risk at confidence ``level``, as a positive loss.

    VaR is the ``level``-quantile of the empirical LOSS distribution
    ``L = -r`` (McNeil, Frey & Embrechts, 2005, *Quantitative Risk
    Management*, Def. 2.10): ``VaR = inf{l : F_L(l) >= level}``. With the
    returns sorted ascending ``r_(1) <= ... <= r_(T)`` this is

        ``VaR = -r_(k+1)``,  ``k = floor(T (1 - level))``,

    i.e. ``np.quantile(-r, level, method="inverted_cdf")``, with
    ``T (1 - level)`` snapped to an integer when it is one up to rounding.
    Example: T = 100 at 95% gives minus the 6th-worst return (exactly 5
    observations lie strictly beyond it).

    Args:
        returns: per-period simple returns.
        level: confidence level in (0, 1), e.g. 0.95.

    Returns:
        A float for one strategy, a Series indexed by strategy for a table.
        Positive means a loss; negative if even the bad tail is a gain.

    Raises:
        ValueError: on invalid returns or ``level`` outside (0, 1).
    """
    level = _check_level(level)
    parsed = _parse_returns(returns)
    return parsed.scalar(_value_at_risk(parsed.values, level))


def expected_shortfall(returns: ReturnsLike, level: float = 0.95) -> float | pd.Series:
    """Historical expected shortfall (CVaR) at confidence ``level``, as a positive loss.

    The exact expected shortfall of the empirical distribution: the average
    of the worst ``m = T (1 - level)`` observations, the boundary observation
    entering with its fractional weight (Acerbi & Tasche, 2002):

        ``ES = -( sum_{i<=floor(m)} r_(i) + (m - floor(m)) r_(floor(m)+1) ) / m``.

    Equivalently ``ES = 1/(1-level) * integral_{level}^{1} VaR_u du``. Unlike
    "mean of returns below the VaR", it is coherent (subadditive), continuous
    in ``level``, and always ``>= VaR``.

    References:
        C. Acerbi and D. Tasche (2002), "On the coherence of expected
        shortfall", Journal of Banking & Finance 26(7), 1487-1503.

    Args:
        returns: per-period simple returns.
        level: confidence level in (0, 1), e.g. 0.95.

    Returns:
        A float for one strategy, a Series indexed by strategy for a table.

    Raises:
        ValueError: on invalid returns or ``level`` outside (0, 1).
    """
    level = _check_level(level)
    parsed = _parse_returns(returns)
    return parsed.scalar(_expected_shortfall(parsed.values, level))


def diversification_ratio(
    weights: pd.Series | np.ndarray | Sequence[float], cov: pd.DataFrame | np.ndarray
) -> float:
    """Diversification ratio ``DR = (w' sigma) / sqrt(w' Sigma w)``.

    The weighted average of the asset volatilities over the portfolio
    volatility: 1 for a single asset (or perfectly correlated assets), and
    ``sqrt(N)`` for equal weights in N uncorrelated assets of equal volatility.
    It is invariant to rescaling ``w`` or ``Sigma``, so the weights need not
    sum to 1 and ``Sigma`` may be per-period or annualized.

    References:
        Y. Choueifaty and Y. Coignard (2008), "Toward maximum
        diversification", Journal of Portfolio Management 35(1), 40-51.

    Args:
        weights: portfolio weights (Series aligned to ``cov`` by label, or an
            array in ``cov`` order).
        cov: covariance matrix.

    Returns:
        The ratio; ``+-inf`` for a zero-variance portfolio with nonzero
        ``w' sigma`` (a perfect hedge), ``nan`` if both are zero.

    Raises:
        ValueError: on mismatched/non-finite inputs (including a nonzero
            weight on an asset that is not in ``cov``), a negative variance on
            the diagonal, or a clearly negative portfolio variance (``cov`` not
            PSD).
    """
    cov_frame = as_cov_frame(cov)
    if isinstance(weights, pd.Series):
        # Aligning by label would silently drop these and measure another portfolio;
        # a zero weight on an extra asset leaves the portfolio unchanged, so it is fine.
        stray = weights[~weights.index.isin(cov_frame.index)]
        stray = stray[np.asarray(stray, dtype=float) != 0.0]  # NaN != 0 counts too
        if not stray.empty:
            raise ValueError(f"weights on assets missing from cov: {list(stray.index)}")
    w = as_series(weights, cov_frame.index).to_numpy()
    sigma_mat = cov_frame.to_numpy()
    variances = np.diag(sigma_mat)
    if (variances < 0.0).any():
        raise ValueError("covariance has a negative variance on its diagonal")
    vols = np.sqrt(variances)
    numerator = float(w @ vols)
    variance = float(w @ sigma_mat @ w)
    # Rounding can leave a hedged portfolio's variance a hair below zero.
    scale = float(np.abs(w) @ vols) ** 2
    if variance < -1e-10 * scale:
        raise ValueError(f"portfolio variance is negative ({variance:.3g}); cov is not PSD")
    if variance <= 1e-14 * scale:
        if abs(numerator) <= 1e-12 * math.sqrt(scale):
            return math.nan
        return math.copysign(math.inf, numerator)
    return numerator / math.sqrt(variance)


def effective_number_of_assets(weights: pd.Series | np.ndarray | Sequence[float]) -> float:
    """Effective number of assets ``N_eff = 1 / sum_i w_i^2``.

    The inverse of the Herfindahl-Hirschman concentration index of the
    weights: ``N`` for equal weights, 1 for a single asset. Meaningful for
    fully invested weights, so they must sum to 1 (within 1e-6).

    Args:
        weights: portfolio weights.

    Returns:
        ``1 / sum(w_i^2)``.

    Raises:
        ValueError: if weights are empty, non-finite, or do not sum to 1.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if not np.isfinite(w).all():
        raise ValueError("weights contain NaN or infinite values")
    total = float(w.sum())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(f"weights must sum to 1, got {total:.8g}")
    # sum(w) = 1 implies sum(w^2) >= 1/N > 0 (Cauchy-Schwarz), so no zero division.
    return 1.0 / float(w @ w)


def performance_summary(
    returns: pd.DataFrame | pd.Series | np.ndarray,
    rf: float = 0.0,
    periods_per_year: float = TRADING_DAYS,
) -> pd.DataFrame:
    """Table of the main performance statistics, one row per strategy.

    Columns (see the individual functions): "CAGR" (:func:`annualized_return`),
    "Volatility" (:func:`annualized_volatility`), "Sharpe" (:func:`sharpe_ratio`
    with ``rf``), "Sortino" (:func:`sortino_ratio` with ``rf``), "Max drawdown"
    (:func:`max_drawdown`, negative), "Calmar" (:func:`calmar_ratio`),
    "VaR 95%" and "ES 95%" (:func:`value_at_risk`, :func:`expected_shortfall`
    at 95%, positive per-period losses — not annualized).

    Args:
        returns: per-period simple returns; a Series is one row (named by the
            Series name, or "strategy"), a DataFrame one row per column.
        rf: ANNUAL risk-free rate.
        periods_per_year: periods in one year (252 for daily data).

    Returns:
        Float DataFrame, rows = strategies, columns = :data:`SUMMARY_COLUMNS`.
        Formatting is left to the caller.

    Raises:
        ValueError: on invalid returns, non-finite ``rf`` or bad ``periods_per_year``.
    """
    ppy = _check_periods_per_year(periods_per_year)
    rf = _check_rf(rf)
    parsed = _parse_returns(returns)
    v = parsed.values
    columns = {
        "CAGR": _cagr(v, ppy),
        "Volatility": _volatility(v, ppy),
        "Sharpe": _sharpe(v, rf, ppy),
        "Sortino": _sortino(v, rf, ppy),
        "Max drawdown": _max_drawdown(v),
        "Calmar": _calmar(v, ppy),
        "VaR 95%": _value_at_risk(v, 0.95),
        "ES 95%": _expected_shortfall(v, 0.95),
    }
    return pd.DataFrame(columns, index=parsed.labels(), columns=list(SUMMARY_COLUMNS), dtype=float)
