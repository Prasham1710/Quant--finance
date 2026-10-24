"""Monte Carlo experiments that measure estimation error against a KNOWN truth.

In real data the true expected returns and covariances are never observed, so a
backtest cannot separate "the optimizer is wrong" from "the inputs were noisy".
These experiments fix a true ``(mu, cov)``, simulate return histories from it, run
the optimizers on the *estimated* parameters, and then score the resulting weights
with the *true* parameters. That isolates the key lesson of the project:

* Mean-variance optimization overweights assets whose returns were overestimated and
  underweights those that were underestimated, so the Sharpe ratio the optimizer
  *believes* it achieves is biased upward, and the Sharpe ratio it *actually*
  achieves is well below the true optimum (Michaud, 1989; Broadie, 1993).
* Errors in the expected returns dominate (Chopra & Ziemba, 1993): minimum variance,
  which ignores the noisy means, is far more robust, and 1/N is a strong benchmark
  (DeMiguel, Garlappi & Uppal, 2009).
* Covariance shrinkage (Ledoit-Wolf) pays off when covariance error dominates: many
  assets, short histories, unconstrained weights (:func:`shrinkage_experiment`).

All quantities are annualized; ``rf`` is the annual risk-free rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantfin.portfolio import data, estimators, markowitz
from quantfin.portfolio._common import (
    Bounds,
    OptimizationError,
    as_cov_frame,
    as_series,
    normalize_bounds,
)

_PORTFOLIOS = ("Max Sharpe", "Min variance")
_ESTIMATORS = {"sample": estimators.cov_sample, "ledoit_wolf": estimators.cov_ledoit_wolf}


def calibrate_truth(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Full-sample estimates of real data, used as the "true" parameters of a simulation.

    Treating the full-sample mean and covariance as the truth gives the simulation
    realistic magnitudes (volatilities, correlations, return dispersion) while keeping
    the truth known, so estimation error can be measured exactly.

    Returns:
        ``(mu, cov)``: annualized arithmetic mean returns and sample covariance.
    """
    return estimators.mean_historical(returns), estimators.cov_sample(returns)


def _max_sharpe_or_none(
    mu: pd.Series, cov: pd.DataFrame, rf: float, bounds: Bounds
) -> pd.Series | None:
    """Tangency weights, or None when the tangency portfolio does not exist."""
    pairs = normalize_bounds(bounds, len(mu))
    if pairs is not None and all(lo >= 0 for lo, _ in pairs) and not (mu > rf).any():
        return None
    try:
        return markowitz.max_sharpe(mu, cov, rf=rf, bounds=bounds)
    except (ValueError, OptimizationError):
        return None


def run_estimation_error_trials(
    true_mu: pd.Series | np.ndarray,
    true_cov: pd.DataFrame | np.ndarray,
    sample_sizes: Sequence[int] = (63, 126, 252, 504, 1260),
    n_trials: int = 200,
    rf: float = 0.0,
    bounds: Bounds = (0.0, 1.0),
    seed: int = 0,
    tail_df: float | None = None,
) -> pd.DataFrame:
    """Simulate, estimate, optimize, and score every trial against the truth.

    For each sample size ``n`` (days of history) and each trial, draw ``n`` i.i.d.
    daily returns from the truth, estimate ``mu`` (historical mean) and ``cov``
    (sample and Ledoit-Wolf), and build max-Sharpe and min-variance portfolios. A
    max-Sharpe portfolio that does not exist for the estimates (no asset beats ``rf``)
    falls back to min variance and is flagged, mirroring the backtest strategies.

    Args:
        true_mu: annualized true expected returns.
        true_cov: annualized true covariance.
        sample_sizes: history lengths in trading days.
        n_trials: simulations per sample size.
        rf: annual risk-free rate.
        bounds: weight bounds for every optimization.
        seed: seed for ``np.random.default_rng``; the experiment is reproducible.
        tail_df: Student-t degrees of freedom for fat-tailed draws (None = Gaussian).

    Returns:
        Long table, one row per (sample size, trial, portfolio, estimator), with columns
        ``n_obs, trial, portfolio, estimator, believed_sharpe, true_sharpe,
        true_volatility, true_return, fallback``. ``believed_sharpe`` uses the estimated
        parameters the optimizer saw; ``true_sharpe`` scores the same weights with the
        truth. Reference rows ``"Equal weight"`` and ``"True optimum"`` are included
        (``estimator = "-"``).
    """
    cov_true = as_cov_frame(true_cov)
    mu_true = as_series(true_mu, cov_true.index)
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    sizes = [int(n) for n in sample_sizes]
    if not sizes or min(sizes) < 2:
        raise ValueError("sample_sizes must be non-empty and each at least 2")
    if bounds is None and min(sizes) <= len(mu_true):
        raise ValueError(
            f"with unconstrained weights every sample size must exceed the number of assets "
            f"({len(mu_true)}): the sample covariance is singular otherwise"
        )
    rng = np.random.default_rng(seed)

    w_opt = _max_sharpe_or_none(mu_true, cov_true, rf, bounds)
    if w_opt is None:
        raise ValueError("the true parameters admit no max-Sharpe portfolio under these bounds")
    n_assets = len(mu_true)
    w_eq = pd.Series(np.full(n_assets, 1.0 / n_assets), index=mu_true.index)

    def score(w: pd.Series) -> tuple[float, float, float]:
        return (
            markowitz.portfolio_sharpe(w, mu_true, cov_true, rf),
            markowitz.portfolio_volatility(w, cov_true),
            markowitz.portfolio_return(w, mu_true),
        )

    references = {name: score(w) for name, w in (("Equal weight", w_eq), ("True optimum", w_opt))}
    rows = []
    for n_obs in sizes:
        for trial in range(n_trials):
            sim = data.simulate_returns(mu_true, cov_true, n_obs, rng, tail_df=tail_df)
            mu_hat = estimators.mean_historical(sim)
            for est_name, est in _ESTIMATORS.items():
                cov_hat = est(sim)
                w_mv = markowitz.min_variance(cov_hat, bounds=bounds)
                w_ms = _max_sharpe_or_none(mu_hat, cov_hat, rf, bounds)
                fallback = w_ms is None
                for portfolio, w in (
                    ("Max Sharpe", w_mv if fallback else w_ms),
                    ("Min variance", w_mv),
                ):
                    true_sharpe, true_vol, true_ret = score(w)
                    rows.append(
                        {
                            "n_obs": n_obs,
                            "trial": trial,
                            "portfolio": portfolio,
                            "estimator": est_name,
                            "believed_sharpe": markowitz.portfolio_sharpe(w, mu_hat, cov_hat, rf),
                            "true_sharpe": true_sharpe,
                            "true_volatility": true_vol,
                            "true_return": true_ret,
                            "fallback": fallback and portfolio == "Max Sharpe",
                        }
                    )
            for portfolio, (true_sharpe, true_vol, true_ret) in references.items():
                rows.append(
                    {
                        "n_obs": n_obs,
                        "trial": trial,
                        "portfolio": portfolio,
                        "estimator": "-",
                        "believed_sharpe": np.nan,
                        "true_sharpe": true_sharpe,
                        "true_volatility": true_vol,
                        "true_return": true_ret,
                        "fallback": False,
                    }
                )
    return pd.DataFrame(rows)


def summarize_estimation_error(trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate :func:`run_estimation_error_trials` output per configuration.

    Returns:
        One row per ``(n_obs, portfolio, estimator)`` with the mean believed and true
        Sharpe ratios, the optimism gap (believed minus true), the dispersion and 5th /
        95th percentiles of the true Sharpe, the true Sharpe as a fraction of the true
        optimum, and the max-Sharpe fallback rate.
    """
    keys = ["n_obs", "portfolio", "estimator"]
    grouped = trials.groupby(keys, sort=False)
    summary = grouped.agg(
        believed_sharpe=("believed_sharpe", "mean"),
        true_sharpe=("true_sharpe", "mean"),
        true_sharpe_std=("true_sharpe", "std"),
        true_sharpe_p05=("true_sharpe", lambda s: s.quantile(0.05)),
        true_sharpe_p95=("true_sharpe", lambda s: s.quantile(0.95)),
        true_volatility=("true_volatility", "mean"),
        fallback_rate=("fallback", "mean"),
    ).reset_index()
    summary["optimism_gap"] = summary["believed_sharpe"] - summary["true_sharpe"]
    optimum = trials[trials["portfolio"] == "True optimum"].groupby("n_obs")["true_sharpe"].first()
    summary["pct_of_optimum"] = summary["true_sharpe"] / summary["n_obs"].map(optimum)
    return summary


def estimation_error_experiment(
    true_mu: pd.Series | np.ndarray,
    true_cov: pd.DataFrame | np.ndarray,
    sample_sizes: Sequence[int] = (63, 126, 252, 504, 1260),
    n_trials: int = 200,
    rf: float = 0.0,
    bounds: Bounds = (0.0, 1.0),
    seed: int = 0,
    tail_df: float | None = None,
) -> pd.DataFrame:
    """Run the estimation-error Monte Carlo and return the per-configuration summary.

    See :func:`run_estimation_error_trials` for the procedure and
    :func:`summarize_estimation_error` for the columns.
    """
    trials = run_estimation_error_trials(
        true_mu, true_cov, sample_sizes, n_trials, rf, bounds, seed, tail_df
    )
    return summarize_estimation_error(trials)


@dataclass(frozen=True)
class FrontierExperiment:
    """Three views of the efficient frontier (Broadie, 1993).

    Attributes:
        true: the frontier computed from the true parameters (unattainable in practice).
        estimated: the frontier the optimizer *believes* in, from estimated parameters.
        realized: the estimated frontier's weights scored with the true parameters —
            what an investor actually gets. It always lies on or below ``true``.
        n_obs: days of simulated history behind the estimates.
    """

    true: pd.DataFrame
    estimated: pd.DataFrame
    realized: pd.DataFrame
    n_obs: int


def frontier_experiment(
    true_mu: pd.Series | np.ndarray,
    true_cov: pd.DataFrame | np.ndarray,
    n_obs: int = 252,
    seed: int = 0,
    n_points: int = 40,
    rf: float = 0.0,
    bounds: Bounds = (0.0, 1.0),
    cov_estimator: str = "sample",
    tail_df: float | None = None,
) -> FrontierExperiment:
    """Compare the true, estimated, and realized efficient frontiers for one sample.

    Args:
        true_mu: annualized true expected returns.
        true_cov: annualized true covariance.
        n_obs: days of simulated history used for the estimates.
        seed: random seed for the simulated history.
        n_points: points per frontier.
        rf: annual risk-free rate (for the Sharpe column).
        bounds: weight bounds.
        cov_estimator: ``"sample"`` or ``"ledoit_wolf"``.
        tail_df: Student-t degrees of freedom (None = Gaussian).

    Returns:
        A :class:`FrontierExperiment`; each frame has ``return``, ``volatility`` and
        ``sharpe`` columns followed by one weight column per asset.
    """
    if cov_estimator not in _ESTIMATORS:
        raise ValueError(f"cov_estimator must be one of {sorted(_ESTIMATORS)}")
    cov_true = as_cov_frame(true_cov)
    mu_true = as_series(true_mu, cov_true.index)
    rng = np.random.default_rng(seed)
    sim = data.simulate_returns(mu_true, cov_true, n_obs, rng, tail_df=tail_df)
    mu_hat = estimators.mean_historical(sim)
    cov_hat = _ESTIMATORS[cov_estimator](sim)

    true = markowitz.efficient_frontier(mu_true, cov_true, n_points=n_points, bounds=bounds, rf=rf)
    estimated = markowitz.efficient_frontier(
        mu_hat, cov_hat, n_points=n_points, bounds=bounds, rf=rf
    )
    assets = list(mu_true.index)
    realized = estimated.copy()
    for i, row in estimated.iterrows():
        w = row[assets].astype(float)
        realized.loc[i, "return"] = markowitz.portfolio_return(w, mu_true)
        realized.loc[i, "volatility"] = markowitz.portfolio_volatility(w, cov_true)
        realized.loc[i, "sharpe"] = markowitz.portfolio_sharpe(w, mu_true, cov_true, rf)
    return FrontierExperiment(true=true, estimated=estimated, realized=realized, n_obs=n_obs)


def shrinkage_experiment(
    true_cov: pd.DataFrame | np.ndarray,
    sample_sizes: Sequence[int] = (63, 126, 252, 504, 1260),
    n_trials: int = 200,
    bounds: Bounds = None,
    seed: int = 0,
    tail_df: float | None = None,
) -> pd.DataFrame:
    """How much covariance shrinkage helps the minimum-variance portfolio.

    Minimum variance needs no expected returns, so it isolates covariance estimation
    error. For each sample size, simulate histories from ``true_cov``, build the
    minimum-variance portfolio from the sample and from the Ledoit-Wolf covariance,
    and measure each portfolio's TRUE volatility against the true minimum.

    Shrinkage matters when the number of assets N is not small relative to the
    number of observations T, and when weights are unconstrained (``bounds=None``,
    the default here): a long-only constraint already acts like shrinkage
    (Jagannathan & Ma, 2003), which hides most of the effect.

    Args:
        true_cov: annualized true covariance.
        sample_sizes: history lengths in trading days.
        n_trials: simulations per sample size.
        bounds: weight bounds (default None: short sales allowed).
        seed: seed for ``np.random.default_rng``.
        tail_df: Student-t degrees of freedom (None = Gaussian).

    Returns:
        One row per ``(n_obs, estimator)`` with:
        ``true_volatility`` (mean over trials); ``true_min_volatility`` (the true
        minimum-variance volatility); ``excess_volatility`` (relative to the true
        minimum, e.g. 0.25 = 25% more risk than necessary); ``shrinkage`` (mean
        Ledoit-Wolf intensity, NaN for the sample estimator); and ``valid_trials``.
        With unconstrained weights and T <= N the sample covariance is singular,
        so those trials are skipped (``valid_trials`` is 0 and the metrics are NaN)
        while Ledoit-Wolf still works.
    """
    cov_true = as_cov_frame(true_cov)
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    sizes = [int(n) for n in sample_sizes]
    if not sizes or min(sizes) < 2:
        raise ValueError("sample_sizes must be non-empty and each at least 2")
    mu_zero = pd.Series(0.0, index=cov_true.index)
    vol_min = markowitz.portfolio_volatility(
        markowitz.min_variance(cov_true, bounds=bounds), cov_true
    )
    rng = np.random.default_rng(seed)
    rows = []
    for n_obs in sizes:
        vols: dict[str, list[float]] = {name: [] for name in _ESTIMATORS}
        deltas = []
        for _ in range(n_trials):
            sim = data.simulate_returns(mu_zero, cov_true, n_obs, rng, tail_df=tail_df)
            deltas.append(estimators.ledoit_wolf_shrinkage(sim)[2])
            for name, est in _ESTIMATORS.items():
                try:
                    w = markowitz.min_variance(est(sim), bounds=bounds)
                except ValueError:
                    continue  # singular sample covariance with unconstrained weights
                vols[name].append(markowitz.portfolio_volatility(w, cov_true))
        for name, values in vols.items():
            mean_vol = float(np.mean(values)) if values else np.nan
            rows.append(
                {
                    "n_obs": n_obs,
                    "estimator": name,
                    "true_volatility": mean_vol,
                    "true_min_volatility": vol_min,
                    "excess_volatility": mean_vol / vol_min - 1.0,
                    "shrinkage": float(np.mean(deltas)) if name == "ledoit_wolf" else np.nan,
                    "valid_trials": len(values),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "FrontierExperiment",
    "calibrate_truth",
    "estimation_error_experiment",
    "frontier_experiment",
    "run_estimation_error_trials",
    "shrinkage_experiment",
    "summarize_estimation_error",
]
