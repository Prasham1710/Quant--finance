"""Smoke tests for the chart builders: each returns a populated figure that saves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from quantfin.portfolio import backtest, estimators, experiments, markowitz, plots, strategies
from quantfin.portfolio.data import synthetic_market


@pytest.fixture(scope="module")
def market():
    return synthetic_market(n_assets=4, n_years=2.0, seed=1)


@pytest.fixture(scope="module")
def results(market):
    chosen = {
        k: v for k, v in strategies.default_strategies().items() if k in ("Equal weight", "HRP")
    }
    return backtest.run_backtests(market.returns, chosen, lookback=126, rebalance_every=21)


def _assert_drawn(fig: Figure, tmp_path, name: str) -> None:
    assert isinstance(fig, Figure)
    assert fig.axes and any(ax.has_data() for ax in fig.axes)
    path = plots.save_figure(fig, tmp_path / "nested" / f"{name}.png")
    assert path.exists() and path.stat().st_size > 1000


def test_frontier_plot(market, tmp_path):
    mu = estimators.mean_historical(market.returns)
    cov = estimators.cov_sample(market.returns)
    frontier = markowitz.efficient_frontier(mu, cov, n_points=15)
    portfolios = {
        "Min variance": markowitz.min_variance(cov),
        "Max Sharpe": markowitz.max_sharpe(mu, cov),
    }
    fig = plots.plot_efficient_frontier(
        frontier, mu, cov, portfolios, rf=0.0, extra_frontiers={"again": frontier}
    )
    _assert_drawn(fig, tmp_path, "frontier")


def test_weight_and_risk_plots(market, tmp_path):
    cov = estimators.cov_sample(market.returns)
    weights = {
        "Min variance": markowitz.min_variance(cov),
        "Equal": pd.Series(0.25, index=cov.index),
    }
    _assert_drawn(plots.plot_weights(weights), tmp_path, "weights")
    _assert_drawn(plots.plot_risk_contributions(weights, cov), tmp_path, "risk")


def test_backtest_plots(results, tmp_path):
    _assert_drawn(plots.plot_cumulative_returns(results), tmp_path, "cumulative")
    _assert_drawn(plots.plot_drawdowns(results), tmp_path, "drawdowns")
    _assert_drawn(plots.plot_weights_over_time(results["HRP"]), tmp_path, "weights_time")


def test_experiment_plots(market, tmp_path):
    summary = experiments.estimation_error_experiment(
        market.true_mu, market.true_cov, sample_sizes=(63, 126), n_trials=3, seed=0
    )
    _assert_drawn(plots.plot_estimation_error(summary), tmp_path, "estimation_error")
    fe = experiments.frontier_experiment(market.true_mu, market.true_cov, n_obs=126, n_points=10)
    _assert_drawn(plots.plot_frontier_experiment(fe), tmp_path, "frontiers")
    assert np.isfinite(fe.realized["volatility"]).all()


def test_shrinkage_plot(market, tmp_path):
    table = experiments.shrinkage_experiment(market.true_cov, sample_sizes=(3, 63), n_trials=3)
    _assert_drawn(plots.plot_shrinkage(table), tmp_path, "shrinkage")
