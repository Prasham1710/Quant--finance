"""Portfolio construction rules for the walk-forward backtest.

Each strategy is a callable ``window -> weights``: it receives a DataFrame of PAST
daily returns (the backtester guarantees no row from the holding period is included)
and returns long-only target weights that sum to 1. The estimators turn the window
into annualized ``mu``/``cov``; the optimizers turn those into weights.

The comparison set in :func:`default_strategies` is chosen to isolate one effect at a
time: sample vs Ledoit-Wolf covariance with the same optimizer, optimizers that need
expected returns (max Sharpe, Black-Litterman) vs ones that use only risk
(min variance, risk parity, HRP), and the naive 1/N benchmark, which DeMiguel,
Garlappi & Uppal (2009) found hard to beat out of sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from quantfin.portfolio import black_litterman, estimators, hrp, markowitz, risk_parity
from quantfin.portfolio._common import Bounds, OptimizationError, normalize_bounds

CovEstimatorName = Literal["sample", "ledoit_wolf", "ewma"]
MeanEstimatorName = Literal["historical", "ewma"]

COV_ESTIMATORS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "sample": estimators.cov_sample,
    "ledoit_wolf": estimators.cov_ledoit_wolf,
    "ewma": estimators.cov_ewma,
}
"""Covariance estimators by name; each maps a returns window to an annualized matrix."""

MEAN_ESTIMATORS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "historical": estimators.mean_historical,
    "ewma": estimators.mean_ewma,
}
"""Expected-return estimators by name; each maps a returns window to annualized means."""


def _lookup(table: dict, name: str, kind: str) -> Callable:
    try:
        return table[name]
    except KeyError:
        raise ValueError(f"unknown {kind} {name!r}; choose from {sorted(table)}") from None


def _window_label(window: pd.DataFrame) -> str:
    return str(window.index[-1]) if len(window.index) else "<empty window>"


@dataclass
class _FallbackLog:
    """Counts the windows where a strategy fell back to minimum variance, and why."""

    fallbacks: int = field(default=0, init=False)
    fallback_reasons: list[tuple[str, str]] = field(default_factory=list, init=False, repr=False)

    def _record_fallback(self, window: pd.DataFrame, reason: str) -> None:
        self.fallbacks += 1
        self.fallback_reasons.append((_window_label(window), reason))


@dataclass
class EqualWeight:
    """1/N: no estimation at all, so no estimation error (DeMiguel et al., 2009)."""

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        n = window.shape[1]
        return pd.Series(np.full(n, 1.0 / n), index=window.columns)


@dataclass
class InverseVolatility:
    """Weights proportional to 1/sigma_i; ignores correlations entirely."""

    cov_estimator: str = "sample"

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        cov = _lookup(COV_ESTIMATORS, self.cov_estimator, "covariance estimator")(window)
        return risk_parity.inverse_volatility(cov)


@dataclass
class MinVariance:
    """Global minimum-variance portfolio: needs only the covariance matrix."""

    cov_estimator: str = "sample"
    bounds: Bounds = (0.0, 1.0)

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        cov = _lookup(COV_ESTIMATORS, self.cov_estimator, "covariance estimator")(window)
        return markowitz.min_variance(cov, bounds=self.bounds)


def _has_positive_excess(mu: pd.Series, rf: float, bounds: Bounds) -> bool:
    """Whether a max-Sharpe portfolio can exist under long-only style bounds.

    With non-negative lower bounds every portfolio return is a convex combination of
    the asset returns, so some asset must beat ``rf``. With short sales allowed the
    optimizer itself decides (it raises ``ValueError`` when the tangency is undefined).
    """
    pairs = normalize_bounds(bounds, len(mu))
    if pairs is not None and all(lo >= 0 for lo, _ in pairs):
        return bool((mu > rf).any())
    return True


@dataclass
class MaxSharpe(_FallbackLog):
    """Tangency portfolio from estimated means and covariances.

    When no feasible portfolio has a positive expected excess return (every asset's
    trailing mean is below ``rf``, common after a crash), the tangency portfolio does
    not exist; the strategy then holds the minimum-variance portfolio and records the
    fallback in ``fallbacks`` / ``fallback_reasons``.
    """

    mean_estimator: str = "historical"
    cov_estimator: str = "sample"
    rf: float = 0.0
    bounds: Bounds = (0.0, 1.0)

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        mu = _lookup(MEAN_ESTIMATORS, self.mean_estimator, "mean estimator")(window)
        cov = _lookup(COV_ESTIMATORS, self.cov_estimator, "covariance estimator")(window)
        return _max_sharpe_or_min_variance(self, window, mu, cov, self.rf, self.bounds)


def _max_sharpe_or_min_variance(
    log: _FallbackLog,
    window: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    bounds: Bounds,
) -> pd.Series:
    if not _has_positive_excess(mu, rf, bounds):
        log._record_fallback(window, "no asset has expected return above rf")
        return markowitz.min_variance(cov, bounds=bounds)
    try:
        return markowitz.max_sharpe(mu, cov, rf=rf, bounds=bounds)
    except (ValueError, OptimizationError) as exc:
        log._record_fallback(window, f"max_sharpe failed: {exc}")
        return markowitz.min_variance(cov, bounds=bounds)


@dataclass
class BlackLittermanMomentum(_FallbackLog):
    """Black-Litterman with an equilibrium prior and absolute momentum views.

    1. Prior: implied equilibrium excess returns ``pi = delta * cov @ w_mkt``, where
       ``w_mkt`` is ``market_weights`` (equal weight when not given — a stand-in for
       market caps, which are not available for a multi-asset ETF universe).
    2. Views: one absolute view per asset equal to its trailing annualized excess
       return over ``momentum_lookback`` days, skipping the most recent
       ``momentum_skip`` days (the standard "12-1" momentum signal, which avoids the
       one-month reversal effect).
    3. View uncertainty: Idzorek's method with the same ``view_confidence`` for every
       view (0 = ignore views and hold the prior, 1 = trust them fully).
    4. Portfolio: max Sharpe on the posterior mean (plus ``rf``) and posterior covariance.

    Low confidence pulls the posterior toward the prior and the weights toward
    ``w_mkt``. Idzorek's confidence is defined for each view in isolation, though:
    with one absolute view per asset on correlated assets the views interact, so the
    posterior can move more (or less) than ``view_confidence`` toward a given view,
    and max Sharpe on it can still be concentrated. Lower the confidence for weights
    closer to the prior.
    """

    cov_estimator: str = "ledoit_wolf"
    risk_aversion: float = 2.5
    tau: float = 0.05
    momentum_lookback: int = 252
    momentum_skip: int = 21
    view_confidence: float = 0.25
    market_weights: pd.Series | None = None
    rf: float = 0.0
    bounds: Bounds = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 < self.view_confidence <= 1.0:
            raise ValueError("view_confidence must be in (0, 1]")
        if self.momentum_skip < 0 or self.momentum_lookback <= self.momentum_skip:
            raise ValueError("need 0 <= momentum_skip < momentum_lookback")

    def prior_weights(self, assets: pd.Index) -> pd.Series:
        """Market weights aligned to ``assets`` (equal weight when none were given)."""
        if self.market_weights is None:
            return pd.Series(np.full(len(assets), 1.0 / len(assets)), index=assets)
        w = self.market_weights.reindex(assets)
        if w.isna().any() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("market_weights must be non-negative and cover every asset")
        return w / w.sum()

    def momentum_views(self, window: pd.DataFrame) -> pd.Series:
        """Annualized trailing excess return of each asset, skipping the latest days.

        Raises:
            ValueError: if the window is shorter than ``momentum_lookback`` (the views
                would silently use a shorter horizon than configured).
        """
        if len(window) < self.momentum_lookback:
            raise ValueError(
                f"window of {len(window)} rows is too short for momentum views "
                f"(lookback {self.momentum_lookback}, skip {self.momentum_skip}); use a "
                "longer backtest lookback or a shorter momentum_lookback"
            )
        end = len(window) - self.momentum_skip
        start = len(window) - self.momentum_lookback
        momentum = estimators.mean_historical(window.iloc[start:end])
        return momentum - self.rf

    def posterior(self, window: pd.DataFrame) -> black_litterman.BlackLittermanResult:
        """Black-Litterman posterior (excess returns) for this window."""
        cov = _lookup(COV_ESTIMATORS, self.cov_estimator, "covariance estimator")(window)
        prior = black_litterman.implied_equilibrium_returns(
            cov, self.prior_weights(window.columns), risk_aversion=self.risk_aversion
        )
        views = self.momentum_views(window)
        P, Q = black_litterman.build_views(window.columns, absolute=views.to_dict())
        omega = black_litterman.idzorek_omega(cov, P, [self.view_confidence] * len(Q), tau=self.tau)
        return black_litterman.black_litterman(cov, prior, P, Q, omega=omega, tau=self.tau)

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        post = self.posterior(window)
        mu_total = post.mu + self.rf
        return _max_sharpe_or_min_variance(self, window, mu_total, post.cov, self.rf, self.bounds)


@dataclass
class RiskParity:
    """Equal risk contribution (ERC): every asset adds the same share of portfolio risk."""

    cov_estimator: str = "sample"

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        cov = _lookup(COV_ESTIMATORS, self.cov_estimator, "covariance estimator")(window)
        return risk_parity.equal_risk_contribution(cov)


@dataclass
class HierarchicalRiskParity:
    """HRP (López de Prado, 2016): cluster, reorder, then split risk top-down."""

    cov_estimator: str = "sample"

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        cov = _lookup(COV_ESTIMATORS, self.cov_estimator, "covariance estimator")(window)
        return hrp.hrp_weights(cov)


def equal_weight() -> EqualWeight:
    """1/N benchmark."""
    return EqualWeight()


def inverse_volatility_strategy(cov_estimator: CovEstimatorName = "sample") -> InverseVolatility:
    """Weights proportional to inverse volatility."""
    return InverseVolatility(cov_estimator=cov_estimator)


def min_variance_strategy(
    cov_estimator: CovEstimatorName = "sample", bounds: Bounds = (0.0, 1.0)
) -> MinVariance:
    """Global minimum-variance portfolio with the chosen covariance estimator."""
    return MinVariance(cov_estimator=cov_estimator, bounds=bounds)


def max_sharpe_strategy(
    mean_estimator: MeanEstimatorName = "historical",
    cov_estimator: CovEstimatorName = "sample",
    rf: float = 0.0,
    bounds: Bounds = (0.0, 1.0),
) -> MaxSharpe:
    """Tangency portfolio; falls back to min variance when it does not exist."""
    return MaxSharpe(
        mean_estimator=mean_estimator, cov_estimator=cov_estimator, rf=rf, bounds=bounds
    )


def black_litterman_strategy(
    cov_estimator: CovEstimatorName = "ledoit_wolf",
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    momentum_lookback: int = 252,
    momentum_skip: int = 21,
    view_confidence: float = 0.25,
    market_weights: pd.Series | None = None,
    rf: float = 0.0,
    bounds: Bounds = (0.0, 1.0),
) -> BlackLittermanMomentum:
    """Black-Litterman with equilibrium prior and 12-1 momentum views."""
    return BlackLittermanMomentum(
        cov_estimator=cov_estimator,
        risk_aversion=risk_aversion,
        tau=tau,
        momentum_lookback=momentum_lookback,
        momentum_skip=momentum_skip,
        view_confidence=view_confidence,
        market_weights=market_weights,
        rf=rf,
        bounds=bounds,
    )


def risk_parity_strategy(cov_estimator: CovEstimatorName = "sample") -> RiskParity:
    """Equal risk contribution portfolio."""
    return RiskParity(cov_estimator=cov_estimator)


def hrp_strategy(cov_estimator: CovEstimatorName = "sample") -> HierarchicalRiskParity:
    """Hierarchical Risk Parity portfolio."""
    return HierarchicalRiskParity(cov_estimator=cov_estimator)


def default_strategies(rf: float = 0.0, momentum_lookback: int = 252) -> dict[str, Callable]:
    """The comparison set used by the backtest CLI.

    Args:
        rf: annual risk-free rate used by the strategies that need expected excess returns.
        momentum_lookback: days of history behind Black-Litterman's momentum views; must
            not exceed the backtest lookback window.

    Returns:
        Ordered mapping of display name to strategy.
    """
    return {
        "Equal weight": equal_weight(),
        "Inverse volatility": inverse_volatility_strategy(),
        "Min variance (sample)": min_variance_strategy("sample"),
        "Min variance (Ledoit-Wolf)": min_variance_strategy("ledoit_wolf"),
        "Max Sharpe (sample)": max_sharpe_strategy(cov_estimator="sample", rf=rf),
        "Max Sharpe (Ledoit-Wolf)": max_sharpe_strategy(cov_estimator="ledoit_wolf", rf=rf),
        "Black-Litterman": black_litterman_strategy(
            rf=rf,
            momentum_lookback=momentum_lookback,
            momentum_skip=min(21, momentum_lookback // 4),
        ),
        "Risk parity (ERC)": risk_parity_strategy(),
        "HRP": hrp_strategy(),
    }


__all__ = [
    "COV_ESTIMATORS",
    "MEAN_ESTIMATORS",
    "BlackLittermanMomentum",
    "EqualWeight",
    "HierarchicalRiskParity",
    "InverseVolatility",
    "MaxSharpe",
    "MinVariance",
    "RiskParity",
    "black_litterman_strategy",
    "default_strategies",
    "equal_weight",
    "hrp_strategy",
    "inverse_volatility_strategy",
    "max_sharpe_strategy",
    "min_variance_strategy",
    "risk_parity_strategy",
]
