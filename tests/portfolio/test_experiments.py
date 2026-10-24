"""Tests for the known-truth estimation-error experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfin.portfolio import estimators, experiments, markowitz


@pytest.fixture(scope="module")
def truth() -> tuple[pd.Series, pd.DataFrame]:
    vols = np.array([0.15, 0.20, 0.07, 0.25, 0.10])
    corr = np.array(
        [
            [1.0, 0.7, -0.1, 0.5, 0.2],
            [0.7, 1.0, -0.2, 0.6, 0.1],
            [-0.1, -0.2, 1.0, 0.0, 0.3],
            [0.5, 0.6, 0.0, 1.0, 0.2],
            [0.2, 0.1, 0.3, 0.2, 1.0],
        ]
    )
    names = ["Stocks", "Tech", "Bonds", "EM", "Credit"]
    cov = pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)
    mu = pd.Series([0.07, 0.09, 0.03, 0.08, 0.045], index=names)
    return mu, cov


@pytest.fixture(scope="module")
def trials(truth) -> pd.DataFrame:
    mu, cov = truth
    return experiments.run_estimation_error_trials(
        mu, cov, sample_sizes=(63, 2520), n_trials=40, seed=3
    )


def test_trials_table_shape_and_columns(trials):
    # per (size, trial): 2 estimators x 2 portfolios + 2 references
    assert len(trials) == 2 * 40 * 6
    assert set(trials.columns) == {
        "n_obs",
        "trial",
        "portfolio",
        "estimator",
        "believed_sharpe",
        "true_sharpe",
        "true_volatility",
        "true_return",
        "fallback",
    }
    assert set(trials["portfolio"]) == {
        "Max Sharpe",
        "Min variance",
        "Equal weight",
        "True optimum",
    }


def test_no_portfolio_beats_the_true_optimum(trials):
    # The true tangency portfolio maximizes the true Sharpe ratio over all long-only
    # portfolios, so every estimated portfolio must score at or below it.
    optimum = trials.loc[trials["portfolio"] == "True optimum", "true_sharpe"].iloc[0]
    assert (trials["true_sharpe"] <= optimum + 1e-9).all()


def test_optimizer_is_overconfident_with_little_data(trials):
    summary = experiments.summarize_estimation_error(trials)
    row = summary[
        (summary["n_obs"] == 63)
        & (summary["portfolio"] == "Max Sharpe")
        & (summary["estimator"] == "sample")
    ].iloc[0]
    assert row["believed_sharpe"] > row["true_sharpe"] + 0.5
    assert row["optimism_gap"] == pytest.approx(row["believed_sharpe"] - row["true_sharpe"])


def test_more_data_shrinks_the_optimism_gap_and_improves_true_sharpe(trials):
    summary = experiments.summarize_estimation_error(trials).set_index(
        ["n_obs", "portfolio", "estimator"]
    )
    short = summary.loc[(63, "Max Sharpe", "sample")]
    long = summary.loc[(2520, "Max Sharpe", "sample")]
    assert long["optimism_gap"] < short["optimism_gap"]
    assert long["true_sharpe"] > short["true_sharpe"]
    assert long["pct_of_optimum"] <= 1.0 + 1e-12


def test_reference_rows_are_constant_and_correct(truth, trials):
    mu, cov = truth
    w_opt = markowitz.max_sharpe(mu, cov)
    eq = trials[trials["portfolio"] == "Equal weight"]
    opt = trials[trials["portfolio"] == "True optimum"]
    assert eq["true_sharpe"].nunique() == 1
    assert opt["true_sharpe"].iloc[0] == pytest.approx(markowitz.portfolio_sharpe(w_opt, mu, cov))
    w_eq = pd.Series(0.2, index=mu.index)
    assert eq["true_sharpe"].iloc[0] == pytest.approx(markowitz.portfolio_sharpe(w_eq, mu, cov))
    assert eq["believed_sharpe"].isna().all()


def test_summary_percent_of_optimum(trials):
    summary = experiments.summarize_estimation_error(trials)
    opt = summary[summary["portfolio"] == "True optimum"]
    assert np.allclose(opt["pct_of_optimum"], 1.0)
    assert (summary["fallback_rate"].between(0.0, 1.0)).all()


def test_experiment_is_reproducible(truth):
    mu, cov = truth
    a = experiments.estimation_error_experiment(mu, cov, sample_sizes=(63,), n_trials=5, seed=11)
    b = experiments.estimation_error_experiment(mu, cov, sample_sizes=(63,), n_trials=5, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_fallback_flag_when_estimates_have_no_positive_excess_return(truth):
    mu, cov = truth
    # With rf far above every true mean, short samples rarely produce any asset
    # beating rf, so max Sharpe falls back to min variance; the truth itself still
    # needs a valid tangency, so lift rf only slightly below the best true mean.
    trials = experiments.run_estimation_error_trials(
        mu, cov, sample_sizes=(63,), n_trials=30, rf=0.088, seed=5
    )
    ms = trials[(trials["portfolio"] == "Max Sharpe")]
    mv = trials[(trials["portfolio"] == "Min variance")]
    assert ms["fallback"].any()
    fell_back = ms["fallback"].to_numpy()
    assert np.allclose(
        ms["true_sharpe"].to_numpy()[fell_back], mv["true_sharpe"].to_numpy()[fell_back]
    )


def test_truth_without_tangency_is_rejected(truth):
    mu, cov = truth
    with pytest.raises(ValueError, match="no max-Sharpe"):
        experiments.run_estimation_error_trials(mu, cov, sample_sizes=(63,), n_trials=1, rf=0.5)


@pytest.mark.parametrize("kwargs", [{"n_trials": 0}, {"sample_sizes": ()}, {"sample_sizes": (1,)}])
def test_invalid_experiment_settings(truth, kwargs):
    mu, cov = truth
    with pytest.raises(ValueError):
        experiments.run_estimation_error_trials(mu, cov, **kwargs)


def test_calibrate_truth_uses_full_sample_estimates():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, size=(500, 3)), columns=list("abc"))
    mu, cov = experiments.calibrate_truth(returns)
    pd.testing.assert_series_equal(mu, estimators.mean_historical(returns))
    pd.testing.assert_frame_equal(cov, estimators.cov_sample(returns))


def test_frontier_experiment_realized_frontier_lies_below_the_true_frontier(truth):
    mu, cov = truth
    fe = experiments.frontier_experiment(mu, cov, n_obs=126, seed=2, n_points=25)
    assets = list(mu.index)
    pd.testing.assert_frame_equal(fe.realized[assets], fe.estimated[assets])
    true_best = fe.true["sharpe"].max()
    assert (fe.realized["sharpe"] <= true_best + 1e-9).all()
    min_vol = fe.true["volatility"].min()
    assert (fe.realized["volatility"] >= min_vol - 1e-9).all()
    # Every realized point is dominated: at its volatility, the best true portfolio
    # with no more risk earns at least as much expected return.
    for _, row in fe.realized.iterrows():
        best = markowitz.efficient_risk(mu, cov, target_volatility=row["volatility"] + 1e-9)
        assert row["return"] <= markowitz.portfolio_return(best, mu) + 1e-7
    assert fe.n_obs == 126


def test_frontier_experiment_rejects_unknown_estimator(truth):
    mu, cov = truth
    with pytest.raises(ValueError, match="cov_estimator"):
        experiments.frontier_experiment(mu, cov, cov_estimator="ewma")


@pytest.fixture(scope="module")
def shrinkage_table() -> pd.DataFrame:
    from quantfin.portfolio.data import synthetic_market

    cov = synthetic_market(n_assets=30, n_years=1, seed=0).true_cov
    return experiments.shrinkage_experiment(cov, sample_sizes=(20, 63, 1260), n_trials=12, seed=4)


def test_shrinkage_experiment_table(shrinkage_table):
    assert len(shrinkage_table) == 3 * 2
    assert set(shrinkage_table["estimator"]) == {"sample", "ledoit_wolf"}
    valid = shrinkage_table[shrinkage_table["valid_trials"] > 0]
    # Nothing beats the true minimum-variance portfolio on true risk.
    assert (valid["excess_volatility"] >= -1e-12).all()
    np.testing.assert_allclose(
        valid["true_volatility"], valid["true_min_volatility"] * (1 + valid["excess_volatility"])
    )


def test_sample_covariance_is_unusable_when_t_is_below_n(shrinkage_table):
    short = shrinkage_table[shrinkage_table["n_obs"] == 20].set_index("estimator")
    assert short.loc["sample", "valid_trials"] == 0
    assert np.isnan(short.loc["sample", "excess_volatility"])
    assert short.loc["ledoit_wolf", "valid_trials"] == 12


def test_shrinkage_helps_most_with_little_data(shrinkage_table):
    t = shrinkage_table.pivot(index="n_obs", columns="estimator", values="excess_volatility")
    assert t.loc[63, "ledoit_wolf"] < 0.5 * t.loc[63, "sample"]
    assert t.loc[1260, "sample"] < t.loc[63, "sample"]
    delta = shrinkage_table.dropna(subset=["shrinkage"]).set_index("n_obs")["shrinkage"]
    assert delta.loc[20] > delta.loc[63] > delta.loc[1260] > 0
    assert shrinkage_table.loc[shrinkage_table["estimator"] == "sample", "shrinkage"].isna().all()


def test_shrinkage_experiment_validation(truth):
    _, cov = truth
    with pytest.raises(ValueError):
        experiments.shrinkage_experiment(cov, n_trials=0)
    with pytest.raises(ValueError):
        experiments.shrinkage_experiment(cov, sample_sizes=(1,))
