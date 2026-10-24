"""Tests for the backtest strategy wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfin.portfolio import black_litterman, estimators, markowitz, strategies
from quantfin.portfolio.data import synthetic_market


@pytest.fixture(scope="module")
def window() -> pd.DataFrame:
    return synthetic_market(n_assets=6, n_years=2.0, seed=7).returns.iloc[:504]


def _check_long_only(w: pd.Series, window: pd.DataFrame) -> None:
    assert list(w.index) == list(window.columns)
    assert np.isfinite(w.to_numpy()).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w >= -1e-10).all()


def test_every_default_strategy_returns_valid_long_only_weights(window):
    for name, strategy in strategies.default_strategies().items():
        w = strategy(window)
        _check_long_only(w, window)
        assert w.name is None or isinstance(w.name, str), name


def test_equal_weight_is_one_over_n(window):
    w = strategies.equal_weight()(window)
    assert np.allclose(w.to_numpy(), 1.0 / window.shape[1])


def test_min_variance_wrapper_matches_direct_call(window):
    for name in ("sample", "ledoit_wolf"):
        w = strategies.min_variance_strategy(name)(window)
        expected = markowitz.min_variance(strategies.COV_ESTIMATORS[name](window))
        pd.testing.assert_series_equal(w, expected)


def test_max_sharpe_wrapper_matches_direct_call(window):
    mu = estimators.mean_historical(window)
    rf = 0.5 * float(mu.max())  # below the best asset, so the tangency portfolio exists
    assert rf > 0
    w = strategies.max_sharpe_strategy(cov_estimator="ledoit_wolf", rf=rf)(window)
    expected = markowitz.max_sharpe(mu, estimators.cov_ledoit_wolf(window), rf=rf)
    pd.testing.assert_series_equal(w, expected)


def test_max_sharpe_falls_back_to_min_variance_when_no_asset_beats_rf(window):
    strategy = strategies.max_sharpe_strategy(rf=0.0)
    crashed = window - 0.01  # every asset now loses ~250% a year on average
    w = strategy(crashed)
    assert strategy.fallbacks == 1
    assert strategy.fallback_reasons[0][0] == str(crashed.index[-1])
    pd.testing.assert_series_equal(w, markowitz.min_variance(estimators.cov_sample(crashed)))
    strategy(window)
    assert strategy.fallbacks == 1  # a normal window does not fall back


def test_risk_based_strategies_match_their_optimizers(window):
    from quantfin.portfolio import hrp, risk_parity

    cov = estimators.cov_sample(window)
    pd.testing.assert_series_equal(
        strategies.risk_parity_strategy()(window), risk_parity.equal_risk_contribution(cov)
    )
    pd.testing.assert_series_equal(strategies.hrp_strategy()(window), hrp.hrp_weights(cov))
    pd.testing.assert_series_equal(
        strategies.inverse_volatility_strategy()(window), risk_parity.inverse_volatility(cov)
    )


def test_black_litterman_momentum_views_are_trailing_12_1_excess_returns(window):
    strategy = strategies.black_litterman_strategy(momentum_lookback=252, momentum_skip=21, rf=0.02)
    views = strategy.momentum_views(window)
    expected = estimators.mean_historical(window.iloc[len(window) - 252 : len(window) - 21]) - 0.02
    pd.testing.assert_series_equal(views, expected)


def test_black_litterman_with_negligible_confidence_holds_the_prior(window):
    # With views ignored the posterior mean is pi = delta * cov @ w_mkt, whose tangency
    # portfolio is w_mkt itself (reverse optimization), so the strategy holds w_mkt.
    w_mkt = pd.Series(np.linspace(1.0, 2.0, window.shape[1]), index=window.columns)
    w_mkt /= w_mkt.sum()
    strategy = strategies.black_litterman_strategy(view_confidence=1e-9, market_weights=w_mkt)
    w = strategy(window)
    assert np.allclose(w.to_numpy(), w_mkt.to_numpy(), atol=1e-5)


def test_black_litterman_full_confidence_posterior_matches_views(window):
    strategy = strategies.black_litterman_strategy(view_confidence=1.0)
    post = strategy.posterior(window)
    views = strategy.momentum_views(window)
    assert np.allclose(post.mu.to_numpy(), views.to_numpy(), atol=1e-8)


def test_black_litterman_posterior_uses_equal_weight_prior_by_default(window):
    strategy = strategies.black_litterman_strategy()
    post = strategy.posterior(window)
    cov = estimators.cov_ledoit_wolf(window)
    equal = pd.Series(1.0 / window.shape[1], index=window.columns)
    expected = black_litterman.implied_equilibrium_returns(cov, equal, risk_aversion=2.5)
    pd.testing.assert_series_equal(post.prior, expected, check_names=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"view_confidence": 0.0},
        {"view_confidence": 1.5},
        {"momentum_lookback": 21, "momentum_skip": 21},
        {"momentum_skip": -1},
    ],
)
def test_black_litterman_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        strategies.black_litterman_strategy(**kwargs)


def test_black_litterman_rejects_bad_market_weights(window):
    bad = pd.Series(1.0, index=window.columns[:-1])
    with pytest.raises(ValueError, match="market_weights"):
        strategies.black_litterman_strategy(market_weights=bad)(window)


def test_black_litterman_rejects_too_short_window(window):
    strategy = strategies.black_litterman_strategy(momentum_lookback=252, momentum_skip=21)
    with pytest.raises(ValueError, match="too short"):
        strategy.momentum_views(window.iloc[:22])


def test_unknown_estimator_names_raise(window):
    with pytest.raises(ValueError, match="unknown covariance estimator"):
        strategies.min_variance_strategy("shrunk")(window)
    with pytest.raises(ValueError, match="unknown mean estimator"):
        strategies.max_sharpe_strategy(mean_estimator="median")(window)


def test_default_strategy_names():
    assert list(strategies.default_strategies()) == [
        "Equal weight",
        "Inverse volatility",
        "Min variance (sample)",
        "Min variance (Ledoit-Wolf)",
        "Max Sharpe (sample)",
        "Max Sharpe (Ledoit-Wolf)",
        "Black-Litterman",
        "Risk parity (ERC)",
        "HRP",
    ]


def test_black_litterman_handles_rf_as_excess_returns(window):
    # The posterior is in excess returns; the strategy adds rf back and lets max
    # Sharpe subtract it again, so the result equals max Sharpe on the excess mean.
    rf = 0.01
    # Low confidence keeps the posterior near the (positive) equilibrium prior.
    strategy = strategies.black_litterman_strategy(rf=rf, view_confidence=0.05)
    post = strategy.posterior(window)
    assert post.mu.max() > 0  # a tangency portfolio exists
    w = strategy(window)
    assert strategy.fallbacks == 0
    pd.testing.assert_series_equal(w, markowitz.max_sharpe(post.mu + rf, post.cov, rf=rf))
    np.testing.assert_allclose(w, markowitz.max_sharpe(post.mu, post.cov, rf=0.0), atol=1e-7)


def test_max_sharpe_falls_back_when_every_mean_is_positive_but_below_rf(window):
    mu = estimators.mean_historical(window)
    shifted = window + (0.01 - float(mu.min())) / 252  # every annual mean now >= 1%
    mu_shifted = estimators.mean_historical(shifted)
    assert (mu_shifted > 0).all()
    strategy = strategies.max_sharpe_strategy(rf=float(mu_shifted.max()) + 0.01)
    w = strategy(shifted)
    assert strategy.fallbacks == 1
    assert strategy.fallback_reasons[0][1] == "no asset has expected return above rf"
    pd.testing.assert_series_equal(w, markowitz.min_variance(estimators.cov_sample(shifted)))


def test_black_litterman_requires_a_full_momentum_window(window):
    strategy = strategies.black_litterman_strategy(momentum_lookback=252)
    with pytest.raises(ValueError, match="too short"):
        strategy(window.iloc[:200])
