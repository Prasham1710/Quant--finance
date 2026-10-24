"""Tests for quantfin.portfolio.metrics.

Every statistic is checked against an independent answer: values computed by
hand on tiny series, the ``statistics`` standard library (ddof=1), plain Python
loops, an O(T^2) brute-force maximum drawdown, numpy's ``inverted_cdf``
quantile for VaR, and a numerical integral of the quantile function for the
expected shortfall. Several tests are designed to fail if a sign, the ddof, the
Sortino divisor, or an annualization factor were wrong.
"""

from __future__ import annotations

import math
import statistics
import warnings

import numpy as np
import pandas as pd
import pytest

from quantfin.portfolio import metrics
from quantfin.portfolio.metrics import (
    SUMMARY_COLUMNS,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cumulative_returns,
    diversification_ratio,
    drawdowns,
    effective_number_of_assets,
    expected_shortfall,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


@pytest.fixture
def daily() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    values = rng.normal(0.0004, 0.011, size=(750, 3))
    values[:, 2] = -0.5 * values[:, 0] + rng.normal(0.0, 0.004, 750)
    index = pd.bdate_range("2015-01-01", periods=750)
    return pd.DataFrame(values, index=index, columns=["alpha", "beta", "gamma"])


SCALAR_METRICS = [
    lambda r: annualized_return(r),
    lambda r: annualized_return(r, periods_per_year=12),
    lambda r: annualized_volatility(r),
    lambda r: sharpe_ratio(r, rf=0.03),
    lambda r: sortino_ratio(r, rf=0.03),
    lambda r: max_drawdown(r),
    lambda r: calmar_ratio(r),
    lambda r: value_at_risk(r, 0.95),
    lambda r: expected_shortfall(r, 0.975),
]


# ---------------------------------------------------------------------------
# Cumulative returns, CAGR, volatility
# ---------------------------------------------------------------------------


def test_cumulative_returns_hand_values_and_types():
    r = pd.Series([0.10, -0.10, 0.05], index=pd.date_range("2020-01-01", periods=3), name="s")
    out = cumulative_returns(r)
    # 1.1 - 1; 1.1 * 0.9 - 1; 1.1 * 0.9 * 1.05 - 1
    np.testing.assert_allclose(out.to_numpy(), [0.10, -0.01, 0.0395], rtol=0, atol=1e-15)
    assert isinstance(out, pd.Series)
    assert out.name == "s"
    assert out.index.equals(r.index)

    arr = cumulative_returns(r.to_numpy())
    assert isinstance(arr, np.ndarray) and arr.shape == (3,)
    np.testing.assert_allclose(arr, out.to_numpy(), rtol=0, atol=0)

    frame = cumulative_returns(pd.DataFrame({"a": r, "b": -r}))
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["a", "b"]
    np.testing.assert_allclose(frame["b"], [-0.10, -0.01, -0.0595], rtol=0, atol=1e-15)


def test_cumulative_returns_matches_python_loop(daily):
    out = cumulative_returns(daily)
    for col in daily.columns:
        wealth, expected = 1.0, []
        for r in daily[col]:
            wealth *= 1.0 + r
            expected.append(wealth - 1.0)
        np.testing.assert_allclose(out[col].to_numpy(), expected, rtol=1e-12, atol=1e-14)


def test_annualized_return_hand_values():
    # Wealth 1.1 * 0.9 = 0.99 after 2 periods.
    assert annualized_return([0.10, -0.10], periods_per_year=2) == pytest.approx(-0.01, abs=1e-15)
    # One year of data per period: the 2-period growth 0.99 spread over 2 years.
    assert annualized_return([0.10, -0.10], periods_per_year=1) == pytest.approx(
        math.sqrt(0.99) - 1.0, rel=1e-13
    )
    # A constant daily return c compounds to (1 + c)^252 - 1 per year, whatever T is.
    c = 0.0004
    for n in (10, 252, 1000):
        assert annualized_return(np.full(n, c)) == pytest.approx((1 + c) ** 252 - 1, rel=1e-12)


def test_annualized_return_keeps_precision_of_tiny_returns():
    # Cash-like returns: 1 + 1e-17 rounds to 1, so CAGR must be built from log1p(r).
    # Binomial expansion (1 + x)^252 - 1 = 252 x + C(252, 2) x^2 + O(x^3).
    for x in (1e-17, 1e-12, 1e-9):
        expected = 252 * x + 252 * 251 / 2 * x**2
        # abs=0: pytest.approx's default abs=1e-12 would swamp these values.
        got = annualized_return(np.full(252, x))
        assert got == pytest.approx(expected, rel=1e-12, abs=0)
        got = annualized_return(np.full(504, -x))
        assert got == pytest.approx(-252 * x + 252 * 251 / 2 * x**2, rel=1e-12, abs=0)


def test_annualized_return_matches_product_formula(daily):
    for ppy in (252, 52, 12):
        got = annualized_return(daily, periods_per_year=ppy)
        expected = (1.0 + daily).prod() ** (ppy / len(daily)) - 1.0
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_annualized_volatility_hand_value_uses_ddof_1():
    r = [0.01, -0.01, 0.02, 0.0]
    # mean 0.005; squared deviations sum to 5e-4; divided by T - 1 = 3.
    per_period = math.sqrt(5e-4 / 3)
    assert per_period == pytest.approx(statistics.stdev(r), rel=1e-14)
    assert annualized_volatility(r) == pytest.approx(per_period * math.sqrt(252), rel=1e-13)
    assert annualized_volatility(r, periods_per_year=12) == pytest.approx(
        per_period * math.sqrt(12), rel=1e-13
    )
    # ddof = 0 would give sqrt(5e-4 / 4) — make sure that is NOT what we compute.
    assert annualized_volatility(r) != pytest.approx(math.sqrt(5e-4 / 4) * math.sqrt(252))


def test_volatility_matches_statistics_stdev(daily):
    got = annualized_volatility(daily)
    for col in daily.columns:
        expected = statistics.stdev(daily[col].tolist()) * math.sqrt(252)
        assert got[col] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Sharpe and Sortino
# ---------------------------------------------------------------------------


def test_sharpe_hand_value_with_risk_free_rate():
    r = [0.01, -0.005, 0.02, 0.0]
    rf = 0.0252  # 1 bp per day with P = 252
    expected = (statistics.mean(r) - 0.0001) / statistics.stdev(r) * math.sqrt(252)
    assert sharpe_ratio(r, rf=rf) == pytest.approx(expected, rel=1e-12)
    # (0.00625 - 0.0001) / 0.0110868 * sqrt(252), written out as a number as well.
    assert sharpe_ratio(r, rf=rf) == pytest.approx(8.8058, abs=1e-4)


def test_sharpe_identities(daily):
    r = daily["alpha"]
    base = sharpe_ratio(r)
    # Subtracting rf/P from every return is the same as passing rf.
    assert sharpe_ratio(r, rf=0.05) == pytest.approx(sharpe_ratio(r - 0.05 / 252), rel=1e-12)
    # A larger rf lowers the Sharpe ratio (sign of the rf term).
    assert sharpe_ratio(r, rf=0.05) < base
    # Leverage invariance (rf = 0) and sign flip.
    assert sharpe_ratio(3.0 * r) == pytest.approx(base, rel=1e-12)
    assert sharpe_ratio(-r) == pytest.approx(-base, rel=1e-12)
    # Annualization: sqrt(P) scaling.
    assert sharpe_ratio(r, periods_per_year=12) == pytest.approx(
        base * math.sqrt(12 / 252), rel=1e-12
    )


def test_sortino_hand_value_divides_by_all_observations():
    r = [0.02, -0.01, 0.03, -0.02]
    # mean 0.005; downside squares 1e-4 + 4e-4 averaged over T = 4 (not over the 2 losses).
    downside = math.sqrt(5e-4 / 4)
    expected = 0.005 / downside * math.sqrt(252)
    assert sortino_ratio(r) == pytest.approx(expected, rel=1e-13)
    assert sortino_ratio(r) == pytest.approx(7.0993, abs=1e-4)
    assert sortino_ratio(r) != pytest.approx(0.005 / math.sqrt(5e-4 / 2) * math.sqrt(252))


def test_sortino_target_is_the_risk_free_rate():
    r = [0.0003, 0.0001, -0.0002, 0.0005]
    rf = 0.0252
    target = 0.0001
    excess = [x - target for x in r]
    downside = math.sqrt(sum(min(x, 0.0) ** 2 for x in excess) / len(r))
    expected = statistics.mean(excess) / downside * math.sqrt(252)
    assert sortino_ratio(r, rf=rf) == pytest.approx(expected, rel=1e-10)
    # The day at exactly the target contributes no downside: only -0.0003 counts.
    assert downside == pytest.approx(0.0003 / 2, rel=1e-12)


def test_sortino_exceeds_sharpe_for_symmetric_returns(daily):
    # For a roughly symmetric series with positive mean, DD (about half the squared
    # deviations, divisor T) < std, so Sortino > Sharpe > 0.
    r = daily["alpha"] - daily["alpha"].mean() + 0.0005
    assert sortino_ratio(r) > sharpe_ratio(r) > 0
    assert sortino_ratio(-r) < sharpe_ratio(-r) < 0


# ---------------------------------------------------------------------------
# Drawdowns and Calmar
# ---------------------------------------------------------------------------


def test_drawdowns_hand_values():
    r = pd.Series([0.10, -0.20, 0.05, 0.30])
    # wealth 1.1, 0.88, 0.924, 1.2012; running peak 1.1, 1.1, 1.1, 1.2012
    np.testing.assert_allclose(drawdowns(r), [0.0, -0.20, -0.16, 0.0], rtol=0, atol=1e-15)
    assert max_drawdown(r) == pytest.approx(-0.20, abs=1e-15)


def test_first_period_loss_is_a_drawdown_from_initial_wealth():
    r = [-0.10, 0.05]
    # The peak includes W_0 = 1: wealth 0.9, 0.945 -> drawdowns -10%, -5.5%.
    np.testing.assert_allclose(drawdowns(r), [-0.10, -0.055], rtol=0, atol=1e-15)
    assert max_drawdown(r) == pytest.approx(-0.10, abs=1e-15)


def test_max_drawdown_matches_brute_force(daily):
    got = max_drawdown(daily)
    for col in daily.columns:
        wealth = np.concatenate([[1.0], np.cumprod(1.0 + daily[col].to_numpy())])
        # max over i <= j of 1 - W_j / W_i, all pairs at once.
        ratio = wealth[None, :] / wealth[:, None]
        upper = np.triu(np.ones_like(ratio, dtype=bool))
        expected = (ratio[upper] - 1.0).min()
        assert got[col] == pytest.approx(expected, rel=1e-12)
        assert got[col] < 0


@pytest.mark.usefixtures("no_warnings")
def test_drawdowns_survive_wealth_overflow():
    # 1100 doublings: wealth 2^1100 overflows a float (sum log1p(r) = 762 > 709),
    # but a path that only ever rises has no drawdown at all.
    up = np.full(1100, 1.0)
    assert max_drawdown(up) == 0.0
    assert calmar_ratio(up) == math.inf
    np.testing.assert_array_equal(drawdowns(up), np.zeros(1100))
    assert cumulative_returns(up)[-1] == math.inf
    table = performance_summary(pd.DataFrame({"up": up}))
    assert table.loc["up", "Max drawdown"] == 0.0
    assert table.loc["up", "Calmar"] == math.inf
    # Falling from an overflowed peak: -50%, then +20% of that -> -40% from the peak.
    fall = np.r_[up, -0.5, 0.2]
    np.testing.assert_allclose(drawdowns(fall)[-3:], [0.0, -0.5, -0.4], rtol=1e-12, atol=0)
    assert max_drawdown(fall) == pytest.approx(-0.5, rel=1e-12)
    # A wipe-out after the overflow: wealth is 0 for good, not inf * 0 = nan.
    wiped = np.r_[up, -1.0, 0.3]
    np.testing.assert_array_equal(cumulative_returns(wiped)[-2:], [-1.0, -1.0])
    np.testing.assert_array_equal(drawdowns(wiped)[-2:], [-1.0, -1.0])
    assert max_drawdown(wiped) == -1.0


def test_calmar_hand_value():
    r = [0.10, -0.20, 0.05, 0.30]
    cagr = 1.2012 ** (252 / 4) - 1
    assert calmar_ratio(r) == pytest.approx(cagr / 0.20, rel=1e-12)
    cagr12 = 1.2012 ** (12 / 4) - 1
    assert calmar_ratio(r, periods_per_year=12) == pytest.approx(cagr12 / 0.20, rel=1e-12)


# ---------------------------------------------------------------------------
# Value-at-Risk and expected shortfall
# ---------------------------------------------------------------------------


@pytest.fixture
def ladder() -> np.ndarray:
    # -10%, -9%, ..., +9%, shuffled so sorting is actually exercised.
    values = np.arange(-10, 10) / 100.0
    return np.random.default_rng(3).permutation(values)


def test_var_es_hand_values(ladder):
    # T = 20. 95%: tail mass m = 1 -> VaR = -r_(2), ES = -r_(1).
    assert value_at_risk(ladder, 0.95) == pytest.approx(0.09, abs=1e-15)
    assert expected_shortfall(ladder, 0.95) == pytest.approx(0.10, abs=1e-15)
    # 90%: m = 20 * 0.1 = 2 (1.9999999999999996 in floating point!) -> VaR = -r_(3).
    assert value_at_risk(ladder, 0.90) == pytest.approx(0.08, abs=1e-15)
    assert expected_shortfall(ladder, 0.90) == pytest.approx(0.095, abs=1e-15)
    # 87.5%: m = 2.5 -> VaR = -r_(3), ES = (0.10 + 0.09 + 0.5 * 0.08) / 2.5.
    assert value_at_risk(ladder, 0.875) == pytest.approx(0.08, abs=1e-15)
    assert expected_shortfall(ladder, 0.875) == pytest.approx(0.092, abs=1e-15)
    # 99%: m = 0.2 < 1 -> both are the worst loss.
    assert value_at_risk(ladder, 0.99) == pytest.approx(0.10, abs=1e-15)
    assert expected_shortfall(ladder, 0.99) == pytest.approx(0.10, abs=1e-15)


def test_var_es_hundred_observations_at_95():
    r = -np.arange(1, 101) / 1000.0  # losses 0.1%, ..., 10%
    # Exactly 5 observations lie beyond the 6th-worst return: VaR = 9.5%.
    assert value_at_risk(r, 0.95) == pytest.approx(0.095, abs=1e-15)
    assert expected_shortfall(r, 0.95) == pytest.approx(np.mean([0.1, 0.099, 0.098, 0.097, 0.096]))


@pytest.mark.parametrize("level", [0.95, 0.99, 0.975, 0.9])
def test_var_matches_numpy_inverted_cdf_of_losses(level):
    r = np.random.default_rng(11).standard_t(4, size=250) * 0.01
    expected = np.quantile(-r, level, method="inverted_cdf")
    assert value_at_risk(r, level) == pytest.approx(expected, abs=0)


@pytest.mark.parametrize(("n_obs", "level"), [(250, 0.95), (97, 0.9), (20, 0.875), (13, 0.99)])
def test_es_is_the_average_var_beyond_the_level(n_obs, level):
    # ES_a = 1/(1-a) * integral_a^1 VaR_u du, with VaR_u the inverted-cdf u-quantile of losses.
    losses = np.sort(-np.random.default_rng(n_obs).normal(0.0, 0.01, n_obs))
    n_grid = 400_000
    u = level + (np.arange(n_grid) + 0.5) * (1.0 - level) / n_grid
    var_u = losses[np.ceil(n_obs * u).astype(int) - 1]
    expected = var_u.mean()
    assert expected_shortfall(-losses, level) == pytest.approx(expected, rel=2e-5)


def test_es_dominates_var_and_both_increase_with_level(daily):
    levels = [0.8, 0.9, 0.95, 0.975, 0.99]
    for col in daily.columns:
        var = [value_at_risk(daily[col], a) for a in levels]
        es = [expected_shortfall(daily[col], a) for a in levels]
        assert all(e >= v - 1e-15 for e, v in zip(es, var, strict=True))
        assert np.all(np.diff(var) >= 0) and np.all(np.diff(es) >= 0)
        assert var[2] > 0  # the 5% left tail of these returns is a loss


def test_var_sign_for_gains_only():
    # A strategy whose worst day is +1% has a negative "loss" VaR.
    assert value_at_risk([0.01, 0.02, 0.03], 0.95) == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# Portfolio-level statistics
# ---------------------------------------------------------------------------


def _cov(vols, corr):
    vols = np.asarray(vols, dtype=float)
    return np.asarray(corr) * np.outer(vols, vols)


def test_diversification_ratio_hand_value():
    cov = _cov([0.1, 0.2], [[1.0, 0.5], [0.5, 1.0]])
    # w'sigma = 0.15; w'Cov w = 0.0025 + 0.01 + 2 * 0.25 * 0.01 = 0.0175
    expected = 0.15 / math.sqrt(0.0175)
    assert diversification_ratio([0.5, 0.5], cov) == pytest.approx(expected, rel=1e-14)
    assert expected == pytest.approx(1.1338934, abs=1e-7)


def test_diversification_ratio_closed_forms():
    n = 5
    # Uncorrelated, equal vols, equal weights: sqrt(N).
    cov = 0.04 * np.eye(n)
    assert diversification_ratio(np.full(n, 1 / n), cov) == pytest.approx(math.sqrt(n))
    # Perfectly correlated: no diversification, DR = 1 for any long weights.
    vols = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    perfectly = _cov(vols, np.ones((n, n)))
    w = np.array([0.1, 0.3, 0.2, 0.25, 0.15])
    assert diversification_ratio(w, perfectly) == pytest.approx(1.0, rel=1e-12)
    # A single asset: DR = 1.
    assert diversification_ratio(np.eye(n)[2], _cov(vols, np.eye(n))) == pytest.approx(1.0)


def test_diversification_ratio_invariances_and_labels():
    labels = ["a", "b", "c"]
    corr = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    cov = pd.DataFrame(_cov([0.1, 0.2, 0.3], corr), index=labels, columns=labels)
    w = pd.Series([0.5, 0.3, 0.2], index=labels)
    base = diversification_ratio(w, cov)
    assert base > 1.0
    assert diversification_ratio(2.0 * w, cov) == pytest.approx(base, rel=1e-13)
    assert diversification_ratio(w, cov / 252) == pytest.approx(base, rel=1e-13)
    # Series weights are matched by label, not position.
    assert diversification_ratio(w[["c", "a", "b"]], cov) == pytest.approx(base, rel=1e-15)


def test_diversification_ratio_rejects_weights_on_assets_not_in_cov():
    cov = pd.DataFrame(np.diag([0.04, 0.09]), index=["a", "b"], columns=["a", "b"])
    # 20% in "c", which cov knows nothing about: dropping it would measure another portfolio.
    with pytest.raises(ValueError, match=r"missing from cov: \['c'\]"):
        diversification_ratio(pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"]), cov)
    with pytest.raises(ValueError, match="missing from cov"):
        diversification_ratio(pd.Series([0.5, 0.5, np.nan], index=["a", "b", "c"]), cov)
    # A zero weight on an extra asset is the same portfolio, so it is accepted.
    padded = pd.Series([0.6, 0.4, 0.0], index=["a", "b", "c"])
    # w'sigma = 0.6 * 0.2 + 0.4 * 0.3 = 0.24; w'Cov w = 0.36 * 0.04 + 0.16 * 0.09 = 0.0288
    expected = 0.24 / math.sqrt(0.0288)
    assert diversification_ratio(padded, cov) == pytest.approx(expected, rel=1e-14)


def test_diversification_ratio_zero_variance_without_warnings():
    cov = _cov([0.1, 0.1], np.ones((2, 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 50/50 in perfectly NEGATIVELY correlated equal-vol assets: a perfect hedge,
        # zero variance but w'sigma = 0.1 > 0.
        hedge = _cov([0.1, 0.1], [[1.0, -1.0], [-1.0, 1.0]])
        assert diversification_ratio([0.5, 0.5], hedge) == math.inf
        # Long 1, short 1 of perfectly correlated assets: zero variance, zero w'sigma.
        assert math.isnan(diversification_ratio([1.0, -1.0], cov))
        # Long 2, short 1 of equal-vol perfectly correlated assets: variance 0.01, fine.
        assert diversification_ratio([2.0, -1.0], cov) == pytest.approx(1.0)
        zero = np.zeros((2, 2))
        assert math.isnan(diversification_ratio([0.5, 0.5], zero))


def test_effective_number_of_assets():
    assert effective_number_of_assets(np.full(8, 1 / 8)) == pytest.approx(8.0)
    assert effective_number_of_assets([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert effective_number_of_assets(pd.Series([0.5, 0.25, 0.25])) == pytest.approx(8 / 3)
    # Long-short weights summing to 1 are allowed: 1 / (1.5^2 + 0.5^2) = 0.4.
    assert effective_number_of_assets([1.5, -0.5]) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Summary table and type consistency
# ---------------------------------------------------------------------------


def test_performance_summary_matches_individual_functions(daily):
    table = performance_summary(daily, rf=0.02)
    assert list(table.columns) == list(SUMMARY_COLUMNS)
    assert list(table.index) == list(daily.columns)
    assert all(dtype == np.float64 for dtype in table.dtypes)
    expected = pd.DataFrame(
        {
            "CAGR": annualized_return(daily),
            "Volatility": annualized_volatility(daily),
            "Sharpe": sharpe_ratio(daily, rf=0.02),
            "Sortino": sortino_ratio(daily, rf=0.02),
            "Max drawdown": max_drawdown(daily),
            "Calmar": calmar_ratio(daily),
            "VaR 95%": value_at_risk(daily, 0.95),
            "ES 95%": expected_shortfall(daily, 0.95),
        }
    )
    pd.testing.assert_frame_equal(table, expected, check_exact=False, rtol=1e-14)
    # rf only enters Sharpe and Sortino.
    no_rf = performance_summary(daily)
    unchanged = ["CAGR", "Volatility", "Max drawdown", "Calmar", "VaR 95%", "ES 95%"]
    pd.testing.assert_frame_equal(no_rf[unchanged], table[unchanged])
    assert (no_rf["Sharpe"] > table["Sharpe"]).all()


def test_performance_summary_row_labels(daily):
    named = performance_summary(daily["beta"])
    assert list(named.index) == ["beta"]
    unnamed = performance_summary(daily["beta"].rename(None))
    assert list(unnamed.index) == ["strategy"]
    pd.testing.assert_frame_equal(named.reset_index(drop=True), unnamed.reset_index(drop=True))
    arr = performance_summary(daily.to_numpy())
    assert list(arr.index) == [0, 1, 2]


@pytest.mark.parametrize("func", SCALAR_METRICS)
def test_series_frame_and_array_inputs_agree(daily, func):
    as_frame = func(daily)
    assert isinstance(as_frame, pd.Series)
    assert list(as_frame.index) == list(daily.columns)
    as_array = func(daily.to_numpy())
    np.testing.assert_allclose(as_array.to_numpy(), as_frame.to_numpy(), rtol=0, atol=0)
    for col in daily.columns:
        one = func(daily[col])
        assert isinstance(one, float)
        assert one == pytest.approx(as_frame[col], rel=1e-14)
        assert func(daily[col].to_numpy()) == one


def test_drawdowns_frame_type(daily):
    dd = drawdowns(daily)
    assert isinstance(dd, pd.DataFrame)
    assert dd.index.equals(daily.index) and list(dd.columns) == list(daily.columns)
    assert (dd.to_numpy() <= 0).all()
    pd.testing.assert_series_equal(dd["gamma"], drawdowns(daily["gamma"]))


# ---------------------------------------------------------------------------
# Degenerate cases: nan / inf, deterministically, with no RuntimeWarning
# ---------------------------------------------------------------------------


@pytest.fixture
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.mark.usefixtures("no_warnings")
def test_constant_positive_returns_give_infinite_ratios():
    # The sample std of ten copies of 0.0123 is ~1.8e-18 in floating point, not 0.
    r = np.full(10, 0.0123)
    assert np.std(r, ddof=1) > 0  # the rounding noise the guard must absorb
    assert annualized_volatility(r) == 0.0
    assert sharpe_ratio(r) == math.inf
    assert sortino_ratio(r) == math.inf
    assert max_drawdown(r) == 0.0
    assert calmar_ratio(r) == math.inf
    assert sharpe_ratio(-r) == -math.inf
    # Constant losses: downside deviation is the loss itself.
    assert sortino_ratio(-r) == pytest.approx(-math.sqrt(252))


@pytest.mark.usefixtures("no_warnings")
def test_zero_returns_give_nan_ratios():
    r = np.zeros(30)
    assert annualized_return(r) == 0.0
    assert annualized_volatility(r) == 0.0
    assert math.isnan(sharpe_ratio(r))
    assert math.isnan(sortino_ratio(r))
    assert max_drawdown(r) == 0.0
    assert math.isnan(calmar_ratio(r))
    # Returns equal to the risk-free rate (up to rounding of rf / 252): zero excess.
    flat = np.full(5, 0.0001)
    assert math.isnan(sharpe_ratio(flat, rf=0.0252))
    assert math.isnan(sortino_ratio(flat, rf=0.0252))


@pytest.mark.usefixtures("no_warnings")
def test_single_observation():
    assert math.isnan(annualized_volatility([0.01]))
    assert math.isnan(sharpe_ratio([0.01]))
    assert sortino_ratio([0.01]) == math.inf
    assert annualized_return([0.01]) == pytest.approx(1.01**252 - 1)
    assert value_at_risk([0.01]) == pytest.approx(-0.01)
    table = performance_summary(pd.Series([-0.02], name="x"))
    assert math.isnan(table.loc["x", "Sharpe"])
    assert table.loc["x", "Max drawdown"] == pytest.approx(-0.02)


@pytest.mark.usefixtures("no_warnings")
def test_total_loss_is_handled():
    r = [0.10, -1.0, 0.20]
    assert annualized_return(r) == -1.0
    assert max_drawdown(r) == -1.0
    assert calmar_ratio(r) == -1.0
    np.testing.assert_allclose(cumulative_returns(r), [0.10, -1.0, -1.0])
    np.testing.assert_allclose(drawdowns(r), [0.0, -1.0, -1.0])


@pytest.mark.usefixtures("no_warnings")
def test_cagr_overflow_is_inf_without_warning():
    assert annualized_return([100.0]) == math.inf


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        [],
        np.empty((0, 3)),
        [0.01, np.nan],
        [0.01, np.inf],
        [0.01, -1.5],
        np.zeros((2, 2, 2)),
        pd.DataFrame({"a": [0.01, None]}),
    ],
)
def test_invalid_returns_raise(bad):
    with pytest.raises(ValueError):
        sharpe_ratio(bad)
    with pytest.raises(ValueError):
        cumulative_returns(bad)
    with pytest.raises(ValueError):
        performance_summary(bad)


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5, -0.1, float("nan")])
def test_invalid_level_raises(level):
    with pytest.raises(ValueError, match="level"):
        value_at_risk([0.01, -0.02], level)
    with pytest.raises(ValueError, match="level"):
        expected_shortfall([0.01, -0.02], level)


@pytest.mark.parametrize("ppy", [0, -12, float("nan"), float("inf")])
def test_invalid_periods_per_year_raises(ppy):
    for func in (annualized_return, annualized_volatility, sharpe_ratio, calmar_ratio):
        with pytest.raises(ValueError, match="periods_per_year"):
            func([0.01, -0.02], periods_per_year=ppy)


def test_invalid_rf_raises():
    with pytest.raises(ValueError, match="rf"):
        sharpe_ratio([0.01, -0.02], rf=float("nan"))
    with pytest.raises(ValueError, match="rf"):
        performance_summary(pd.Series([0.01, -0.02]), rf=float("inf"))


def test_invalid_weights_raise():
    with pytest.raises(ValueError, match="sum to 1"):
        effective_number_of_assets([0.5, 0.4])
    with pytest.raises(ValueError):
        effective_number_of_assets([])
    with pytest.raises(ValueError):
        effective_number_of_assets([0.5, np.nan, 0.5])
    cov = np.eye(2) * 0.04
    with pytest.raises(ValueError):
        diversification_ratio([1.0], cov)
    with pytest.raises(ValueError):
        diversification_ratio(pd.Series([0.5, 0.5], index=["x", "y"]), cov)
    with pytest.raises(ValueError, match="negative variance"):
        diversification_ratio([0.5, 0.5], np.array([[-0.01, 0.0], [0.0, 0.04]]))
    with pytest.raises(ValueError, match="PSD"):
        # Correlation -3: 50/50 variance is 0.005 - 0.015 < 0.
        diversification_ratio([0.5, 0.5], np.array([[0.01, -0.03], [-0.03, 0.01]]))


def test_module_exports_summary_columns():
    assert metrics.SUMMARY_COLUMNS == (
        "CAGR",
        "Volatility",
        "Sharpe",
        "Sortino",
        "Max drawdown",
        "Calmar",
        "VaR 95%",
        "ES 95%",
    )
