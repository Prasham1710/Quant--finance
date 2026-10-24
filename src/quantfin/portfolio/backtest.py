"""Walk-forward (rolling-window) backtester with weight drift, turnover and costs.

A strategy is any callable mapping a window of PAST returns to target weights
(:data:`Strategy`). :func:`walk_forward_backtest` re-estimates it on a rolling
window every ``rebalance_every`` periods and holds the result out of sample,
so every reported return is earned by weights chosen without seeing it. This is
how the package compares mean-variance, Black-Litterman, risk parity and HRP:
an in-sample "optimal" portfolio says little about how it does on new data.

The accounting follows DeMiguel, Garlappi & Uppal (2009), "Optimal versus
naive diversification: how inefficient is the 1/N portfolio strategy?",
Review of Financial Studies 22(5), 1915-1953: between rebalances the weights
drift with relative prices, turnover is measured against those drifted
weights, and proportional costs are charged on the traded amount.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantfin.portfolio import metrics
from quantfin.portfolio._common import TRADING_DAYS, as_returns_frame

Strategy = Callable[[pd.DataFrame], pd.Series]
"""Window of past returns (rows strictly before the holding period) -> target weights.

The weights' index must be a subset of the window's columns (missing assets get
weight 0) and they must sum to 1 within 1e-6.
"""

_WEIGHT_SUM_TOL = 1e-6

# Row labels of these inferred types have a natural order that must be time order.
_ORDERED_LABEL_TYPES = frozenset(
    {
        "datetime64",
        "datetime",
        "date",
        "period",
        "timedelta64",
        "timedelta",
        "integer",
        "floating",
        "mixed-integer-float",
        "decimal",
    }
)


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """Outcome of one walk-forward backtest.

    Attributes:
        name: strategy name.
        returns: NET per-period returns over the out-of-sample period
            (gross minus trading costs on rebalance dates).
        gross_returns: per-period returns before costs.
        weights: target weights (all assets, missing = 0), one row per
            rebalance date; the row's date is the first period they are held.
        turnover: per rebalance date, ``sum_i |w_target_i - w_drifted_i|``
            (the first rebalance trades out of cash, so it is ``sum_i |w_i|``).
        costs: per rebalance date, ``cost_bps / 1e4 * turnover`` as a fraction
            of wealth.
        periods_per_year: periods in one year, used to annualize turnover and
            the statistics in :func:`compare_backtests`.
    """

    name: str
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    periods_per_year: float = TRADING_DAYS

    @property
    def annual_turnover(self) -> float:
        """Total turnover divided by the length of the out-of-sample period in years.

        ``sum_k turnover_k / (n_oos / periods_per_year)``; e.g. 1.0 means the
        whole portfolio is traded once a year (counting buys and sells).
        """
        years = len(self.returns) / self.periods_per_year
        return float(self.turnover.sum()) / years


def _check_int(label: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return int(value)


def _fmt_date(label: Hashable) -> str:
    if isinstance(label, pd.Timestamp) and label == label.normalize():
        return str(label.date())
    return str(label)


def _time_axis(index: pd.Index) -> pd.Index | None:
    """The row labels as comparable points in time, or None if they have no time order.

    Dates, periods and numbers are used as they are. Strings that all parse as
    dates (e.g. a CSV read without ``parse_dates``) are compared as dates, not
    as text, so "12/31/2019" comes before "01/02/2020". Other labels (free
    text such as "day10" < "day2", mixed types, tuples) carry no reliable time
    order, and the row order itself is taken as the time order.
    """
    kind = index.inferred_type
    if kind in _ORDERED_LABEL_TYPES:
        return index
    if kind != "string":
        return None
    with warnings.catch_warnings():
        # pandas warns when it cannot infer one format and falls back to dateutil.
        warnings.simplefilter("ignore")
        try:
            # utc=True: strings with different UTC offsets compare as instants.
            return pd.DatetimeIndex(pd.to_datetime(index, utc=True))
        except (ValueError, TypeError, OverflowError):
            return None


def _check_chronological(index: pd.Index) -> None:
    """Raise unless the rows are in strictly increasing time order (where knowable)."""
    if not index.is_unique:
        raise ValueError("returns have duplicate dates in the index")
    axis = _time_axis(index)
    if axis is None:
        return
    if not axis.is_monotonic_increasing:
        raise ValueError(
            "returns must be sorted by date (ascending, oldest row first); use "
            "sort_index(), or reset_index(drop=True) if the rows are already in time order"
        )
    if not axis.is_unique:
        raise ValueError("returns have duplicate dates in the index")


def _target_weights(raw: object, assets: pd.Index, where: str) -> np.ndarray:
    """Validate a strategy's output and return it as a dense vector over ``assets``."""
    if raw is None:
        raise ValueError(f"{where}: strategy returned None instead of weights")
    if isinstance(raw, Mapping) and not isinstance(raw, pd.Series):
        raw = pd.Series(dict(raw), dtype=object)
    n_assets = len(assets)
    if isinstance(raw, pd.Series):
        if raw.empty:
            raise ValueError(f"{where}: strategy returned no weights")
        if not raw.index.is_unique:
            raise ValueError(f"{where}: weights have duplicate asset labels")
        # Fast path: most strategies return weights labelled exactly like the window.
        if raw.index.equals(assets):
            positions = np.arange(n_assets)
        else:
            positions = assets.get_indexer(raw.index)
        if (positions < 0).any():
            unknown = list(raw.index[positions < 0])
            raise ValueError(f"{where}: weights for unknown assets {unknown}")
        try:
            values = raw.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: weights are not numeric") from exc
        target = np.zeros(n_assets)
        target[positions] = values
    else:
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: weights are not numeric") from exc
        if values.ndim != 1 or values.shape[0] != n_assets:
            raise ValueError(
                f"{where}: unlabelled weights must be a 1-D array of length {n_assets}, "
                f"got shape {values.shape}"
            )
        target = values.copy()
    if not np.isfinite(target).all():
        raise ValueError(f"{where}: weights contain NaN or infinite values")
    total = float(target.sum())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(f"{where}: weights sum to {total:.8g}, expected 1")
    return target


def walk_forward_backtest(
    returns: pd.DataFrame | np.ndarray,
    strategy: Strategy,
    lookback: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
    name: str = "strategy",
    periods_per_year: float = TRADING_DAYS,
) -> BacktestResult:
    """Run ``strategy`` on a rolling window and hold its weights out of sample.

    Mechanics (positions ``0..T-1`` index the rows of ``returns``):

    * Rebalance dates are ``t = lookback, lookback + rebalance_every, ... < T``.
      At ``t`` the strategy sees ``returns.iloc[t - lookback : t]`` only (a
      copy) — never row ``t`` or later — and returns target weights ``w_t``.
    * The out-of-sample period is rows ``lookback .. T-1``. Initial holdings
      are cash (all zeros), so the first rebalance has turnover ``sum_i |w_i|``.
    * Between rebalances the weights drift with prices: after period ``s``,
      ``w <- w * (1 + r_s) / (1 + w' r_s)``. (If the targets sum to slightly
      less or more than 1, the remainder is an implicit cash position earning
      zero, for which this update is exact.)
    * Turnover at a rebalance is ``tau_t = sum_i |w_t,i - w~_i|`` where ``w~`` are
      the drifted weights held just before trading.
    * Gross return in period ``s`` is ``g_s = w_s' r_s`` with ``w_s`` the weights
      held at the start of ``s`` (post-trade on a rebalance date).
    * Net return is ``g_s - c tau_s`` on a rebalance date (``c = cost_bps / 1e4``)
      and ``g_s`` otherwise: the cost on the traded notional ``tau_s W`` is paid
      at the end of the period out of the holdings, pro rata, so it does not
      change the drifted weights. (DeMiguel et al., 2009, compound instead,
      ``(1 - c tau)(1 + g) - 1``, which differs by the second-order ``c tau g``.)

    The loop runs over periods on numpy arrays, so a 12-asset, 4000-period
    backtest costs a few milliseconds plus the strategy's own run time.

    Args:
        returns: per-period simple returns, rows in chronological order. Row
            labels that are dates, periods, numbers or date strings must be
            strictly increasing (a newest-first file would otherwise feed the
            strategy calendar-future rows); other labels are trusted to be in
            time order.
        strategy: callable ``window -> weights`` (see :data:`Strategy`). A
            ``pd.Series`` (or mapping) is matched by asset label, missing assets
            get 0; a plain array must have one weight per column, in order.
        lookback: number of past rows given to the strategy (``1 <= lookback < T``).
        rebalance_every: periods between rebalances (``>= 1``; larger than the
            out-of-sample length means buy-and-hold after the first trade).
        cost_bps: proportional cost in basis points per unit of turnover (``>= 0``).
        name: strategy name stored in the result.
        periods_per_year: periods in one year (252 for daily data); only used
            to annualize turnover and later statistics.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        ValueError: on invalid returns (non-finite, duplicate columns, unsorted
            or duplicate dates, including date strings) or parameters; if the strategy output is not
            finite, empty, labelled with unknown assets, or does not sum to 1
            (the message gives the date and reason); or if a period's gross
            return is ``<= -100%`` (leveraged blow-up) or costs wipe out wealth.
        TypeError: if ``strategy`` is not callable.
    """
    frame = as_returns_frame(returns)
    if not frame.columns.is_unique:
        raise ValueError("returns have duplicate asset columns")
    _check_chronological(frame.index)
    if not callable(strategy):
        raise TypeError(f"strategy must be callable, got {type(strategy).__name__}")
    n_obs, n_assets = frame.shape
    lookback = _check_int("lookback", lookback, minimum=1)
    if lookback >= n_obs:
        raise ValueError(
            f"lookback ({lookback}) must be smaller than the number of observations ({n_obs})"
        )
    rebalance_every = _check_int("rebalance_every", rebalance_every, minimum=1)
    cost_bps = float(cost_bps)
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError(f"cost_bps must be a finite number >= 0, got {cost_bps!r}")
    ppy = float(periods_per_year)
    if not math.isfinite(ppy) or ppy <= 0:
        raise ValueError(f"periods_per_year must be a positive number, got {periods_per_year!r}")

    assets = frame.columns
    dates = frame.index
    # Our own copy: a strategy that mutates its window cannot alter realized returns.
    realized = frame.to_numpy(dtype=float, copy=True)
    cost_rate = cost_bps / 1e4
    n_oos = n_obs - lookback
    n_rebalances = len(range(lookback, n_obs, rebalance_every))

    gross = np.empty(n_oos)
    net = np.empty(n_oos)
    targets = np.empty((n_rebalances, n_assets))
    turnover = np.empty(n_rebalances)
    costs = np.empty(n_rebalances)
    held = np.zeros(n_assets)  # start in cash
    k = 0
    for s in range(lookback, n_obs):
        cost = 0.0
        if (s - lookback) % rebalance_every == 0:
            where = f"{name!r} at {_fmt_date(dates[s])}"
            window = frame.iloc[s - lookback : s].copy()
            target = _target_weights(strategy(window), assets, where)
            traded = float(np.abs(target - held).sum())
            cost = cost_rate * traded
            targets[k], turnover[k], costs[k] = target, traded, cost
            k += 1
            held = target
        r = realized[s]
        g = float(held @ r)
        if not 1.0 + g > 0.0:
            raise ValueError(
                f"{name!r} at {_fmt_date(dates[s])}: portfolio return {g:.4g} is <= -100% "
                "(leveraged blow-up)"
            )
        if not 1.0 + g - cost > 0.0:
            raise ValueError(
                f"{name!r} at {_fmt_date(dates[s])}: trading costs {cost:.4g} wipe out wealth"
            )
        gross[s - lookback] = g
        net[s - lookback] = g - cost
        held = held * (1.0 + r) / (1.0 + g)

    oos_dates = dates[lookback:]
    rebalance_dates = dates[lookback::rebalance_every]
    return BacktestResult(
        name=name,
        returns=pd.Series(net, index=oos_dates, name=name),
        gross_returns=pd.Series(gross, index=oos_dates, name=name),
        weights=pd.DataFrame(targets, index=rebalance_dates, columns=assets),
        turnover=pd.Series(turnover, index=rebalance_dates, name="turnover"),
        costs=pd.Series(costs, index=rebalance_dates, name="costs"),
        periods_per_year=ppy,
    )


def run_backtests(
    returns: pd.DataFrame | np.ndarray, strategies: Mapping[str, Strategy], **kwargs: object
) -> dict[str, BacktestResult]:
    """Backtest several strategies on the same data and settings.

    Args:
        returns: per-period simple returns.
        strategies: name -> strategy; the name labels the result.
        **kwargs: passed to :func:`walk_forward_backtest` (``lookback``,
            ``rebalance_every``, ``cost_bps``, ``periods_per_year``).

    Returns:
        name -> :class:`BacktestResult`, in the order of ``strategies``.

    Raises:
        ValueError: if ``strategies`` is empty, plus anything
            :func:`walk_forward_backtest` raises.
        TypeError: if ``name`` is passed in ``kwargs`` (results are named by key).
    """
    if "name" in kwargs:
        raise TypeError("run_backtests names each result by its key; do not pass name=")
    if not strategies:
        raise ValueError("no strategies to backtest")
    frame = as_returns_frame(returns)
    return {
        key: walk_forward_backtest(frame, strategy, name=key, **kwargs)
        for key, strategy in strategies.items()
    }


def compare_backtests(results: Mapping[str, BacktestResult], rf: float = 0.0) -> pd.DataFrame:
    """Side-by-side out-of-sample statistics of several backtests.

    Each row is :func:`metrics.performance_summary` of a result's NET returns
    (annualized with the result's ``periods_per_year``) plus the column
    "Turnover (annual)" (:attr:`BacktestResult.annual_turnover`).

    Args:
        results: name -> result (e.g. from :func:`run_backtests`).
        rf: ANNUAL risk-free rate for the Sharpe and Sortino ratios.

    Returns:
        DataFrame indexed by the keys of ``results``, columns
        ``metrics.SUMMARY_COLUMNS + ("Turnover (annual)",)``.

    Raises:
        ValueError: if ``results`` is empty.
    """
    if not results:
        raise ValueError("no backtest results to compare")
    rows = []
    for key, result in results.items():
        row = metrics.performance_summary(
            result.returns, rf=rf, periods_per_year=result.periods_per_year
        )
        row.index = pd.Index([key])
        row["Turnover (annual)"] = result.annual_turnover
        rows.append(row)
    return pd.concat(rows, axis=0)
