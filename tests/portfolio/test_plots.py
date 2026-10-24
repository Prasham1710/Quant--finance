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


def test_frontier_plot_draws_the_right_data(market):
    mu = estimators.mean_historical(market.returns)
    cov = estimators.cov_sample(market.returns)
    frontier = markowitz.efficient_frontier(mu, cov, n_points=15, rf=0.0)
    fig = plots.plot_efficient_frontier(frontier, rf=0.01)
    line, cml = fig.axes[0].get_lines()[:2]
    np.testing.assert_allclose(line.get_xdata(), frontier["volatility"])
    np.testing.assert_allclose(line.get_ydata(), frontier["return"])
    # The capital market line starts at rf and touches (never crosses) the frontier.
    x, y = cml.get_xdata(), cml.get_ydata()
    slope = (y[1] - y[0]) / (x[1] - x[0])
    assert y[0] == pytest.approx(0.01)
    cml_at_frontier = 0.01 + slope * frontier["volatility"]
    assert (frontier["return"] <= cml_at_frontier + 1e-12).all()
    assert (cml_at_frontier - frontier["return"]).min() == pytest.approx(0.0, abs=1e-12)


def test_estimation_error_plot_puts_believed_and_true_in_their_panels(market):
    summary = experiments.estimation_error_experiment(
        market.true_mu, market.true_cov, sample_sizes=(126, 63), n_trials=3, seed=0
    )
    fig = plots.plot_estimation_error(summary)
    believed_ax, true_ax = fig.axes[:2]
    rows = summary[(summary["portfolio"] == "Max Sharpe") & (summary["estimator"] == "sample")]
    rows = rows.sort_values("n_obs")
    first_believed = believed_ax.get_lines()[0]
    first_true = true_ax.get_lines()[0]
    np.testing.assert_allclose(first_believed.get_xdata(), rows["n_obs"])
    np.testing.assert_allclose(first_believed.get_ydata(), rows["believed_sharpe"])
    np.testing.assert_allclose(first_true.get_ydata(), rows["true_sharpe"])


def test_weights_over_time_covers_the_final_rebalance(results):
    fig = plots.plot_weights_over_time(results["HRP"])
    collection = fig.axes[0].collections[0]
    x_max = collection.get_paths()[0].vertices[:, 0].max()
    end = results["HRP"].returns.index[-1]
    assert x_max >= fig.axes[0].convert_xunits(end)
