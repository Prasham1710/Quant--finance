"""Tests for quantfin.portfolio.backtest.

No look-ahead is proven two ways: a spy strategy records every window it is
shown, and an invariance test perturbs the future and checks that no earlier
decision or return changes. The accounting (drift, turnover, costs) is checked
against a tiny example computed by hand, closed forms (1/N with daily
rebalancing, buy-and-hold as value-weighted growth), and an independent
reference simulator that tracks DOLLAR holdings rather than weights.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantfin.portfolio import backtest, metrics
from quantfin.portfolio.backtest import (
    BacktestResult,
    compare_backtests,
    run_backtests,
    walk_forward_backtest,
)


@pytest.fixture
def market() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    n_obs, n_assets = 160, 4
    vols = np.array([0.006, 0.010, 0.014, 0.020])
    values = 0.0003 + rng.standard_normal((n_obs, n_assets)) * vols
    index = pd.bdate_range("2021-01-04", periods=n_obs)
    return pd.DataFrame(values, index=index, columns=["bond", "gold", "stock", "crypto"])


def equal_weight(window: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / window.shape[1], index=window.columns)


def inverse_variance(window: pd.DataFrame) -> pd.Series:
    inv = 1.0 / window.var()
    return inv / inv.sum()


def fixed(weights) -> backtest.Strategy:
    def strategy(window: pd.DataFrame) -> pd.Series:
        return pd.Series(weights, index=window.columns, dtype=float)

    return strategy


def reference_backtest(returns, strategy, lookback, rebalance_every, cost_bps):
    """Independent simulator: dollar holdings per asset plus a cash account.

    Wealth W and holdings h are tracked in dollars; a rebalance buys/sells to
    W * w_target (cash takes the rest), the traded notional is sum |W w - h|,
    and its cost is paid at the end of the period, pro rata from all positions.
    """
    values = returns.to_numpy()
    n_obs, n_assets = values.shape
    holdings = np.zeros(n_assets)
    cash = 1.0
    gross, net, turnover, targets = [], [], [], []
    for s in range(lookback, n_obs):
        wealth = holdings.sum() + cash
        cost = 0.0
        if (s - lookback) % rebalance_every == 0:
            w = strategy(returns.iloc[s - lookback : s]).reindex(returns.columns).fillna(0.0)
            new_holdings = wealth * w.to_numpy()
            traded = np.abs(new_holdings - holdings).sum()
            turnover.append(traded / wealth)
            targets.append(w.to_numpy())
            cost = cost_bps / 1e4 * traded
            cash = wealth - new_holdings.sum()
            holdings = new_holdings
        holdings = holdings * (1.0 + values[s])
        end_wealth = holdings.sum() + cash
        gross.append(end_wealth / wealth - 1.0)
        after_cost = end_wealth - cost
        holdings *= after_cost / end_wealth
        cash *= after_cost / end_wealth
        net.append(after_cost / wealth - 1.0)
    return np.array(gross), np.array(net), np.array(turnover), np.array(targets)


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("lookback", "rebalance_every"), [(20, 7), (1, 1), (60, 200)])
def test_spy_strategy_only_sees_the_past(market, lookback, rebalance_every):
    seen = []

    def spy(window: pd.DataFrame) -> pd.Series:
        seen.append(window.copy())
        return equal_weight(window)

    result = walk_forward_backtest(
        market, spy, lookback=lookback, rebalance_every=rebalance_every, cost_bps=0.0
    )
    positions = list(range(lookback, len(market), rebalance_every))
    assert len(seen) == len(positions) == len(result.weights)
    for window, t, held_from in zip(seen, positions, result.weights.index, strict=True):
        assert held_from == market.index[t]
        # The window ends strictly before the first period the weights are held.
        assert window.index[-1] < held_from
        assert len(window) == lookback
        pd.testing.assert_frame_equal(window, market.iloc[t - lookback : t])


def test_future_data_cannot_change_past_decisions(market):
    cut = 97  # rows >= cut are replaced by something completely different
    future = market.copy()
    future.iloc[cut:] = np.random.default_rng(0).normal(0.0, 0.03, future.iloc[cut:].shape)
    kwargs = {"lookback": 30, "rebalance_every": 9, "cost_bps": 15.0}
    base = walk_forward_backtest(market, inverse_variance, **kwargs)
    alt = walk_forward_backtest(future, inverse_variance, **kwargs)

    decided_before = base.weights.index <= market.index[cut]
    pd.testing.assert_frame_equal(base.weights[decided_before], alt.weights[decided_before])
    earned_before = base.returns.index < market.index[cut]
    pd.testing.assert_series_equal(base.returns[earned_before], alt.returns[earned_before])
    pd.testing.assert_series_equal(base.turnover[decided_before], alt.turnover[decided_before])
    # Sanity: the perturbation does matter afterwards.
    assert not np.allclose(base.weights[~decided_before], alt.weights[~decided_before])
    assert not np.allclose(base.returns[~earned_before], alt.returns[~earned_before])


def test_window_mutation_cannot_leak_into_results(market):
    def vandal(window: pd.DataFrame) -> pd.Series:
        weights = inverse_variance(window)
        window.iloc[:, :] = 0.5  # try to corrupt the data in place
        return weights

    original = market.copy()
    clean = walk_forward_backtest(market, inverse_variance, lookback=25, rebalance_every=5)
    dirty = walk_forward_backtest(market, vandal, lookback=25, rebalance_every=5)
    pd.testing.assert_series_equal(clean.returns, dirty.returns)
    pd.testing.assert_frame_equal(clean.weights, dirty.weights)
    pd.testing.assert_frame_equal(market, original)


# ---------------------------------------------------------------------------
# Accounting: hand-computed example, closed forms, reference simulator
# ---------------------------------------------------------------------------


def test_two_asset_example_by_hand():
    returns = pd.DataFrame(
        [[0.00, 0.00], [0.10, -0.05], [0.02, 0.04], [-0.03, 0.01]],
        columns=["a", "b"],
        index=pd.date_range("2024-01-01", periods=4),
    )
    result = walk_forward_backtest(
        returns, fixed([0.5, 0.5]), lookback=1, rebalance_every=2, cost_bps=10.0
    )
    # Day 1: buy 50/50 from cash (turnover 1, cost 10 bp); gross = (0.10 - 0.05) / 2.
    # Drift: holdings 0.55, 0.475 of wealth 1.025.
    # Day 2: gross = (0.55 * 0.02 + 0.475 * 0.04) / 1.025 = 0.03 / 1.025.
    #        holdings 0.561, 0.494 of wealth 1.055.
    # Day 3: rebalance from (0.561, 0.494) / 1.055 to 50/50: turnover 2 * 0.0335 / 1.055.
    #        gross = (-0.03 + 0.01) / 2.
    turnover_3 = 0.067 / 1.055
    np.testing.assert_allclose(result.gross_returns, [0.025, 0.03 / 1.025, -0.01], atol=1e-15)
    np.testing.assert_allclose(result.turnover, [1.0, turnover_3], atol=1e-15)
    np.testing.assert_allclose(result.costs, [0.001, 0.001 * turnover_3], atol=1e-15)
    np.testing.assert_allclose(
        result.returns, [0.024, 0.03 / 1.025, -0.01 - 0.001 * turnover_3], atol=1e-15
    )
    assert list(result.weights.index) == list(returns.index[[1, 3]])
    assert list(result.returns.index) == list(returns.index[1:])


# Target weights that change at every rebalance, include a short position, and
# leave 1e-7 in cash, so that drift, turnover and costs are all exercised.
ROTATION = np.array(
    [[0.4, 0.3, 0.2, 0.1 - 1e-7], [0.1, 0.1, 0.5, 0.3 - 1e-7], [0.7, -0.2, 0.25, 0.25 - 1e-7]]
)


def rotating(window: pd.DataFrame) -> pd.Series:
    """Deterministic in the window: picks a row of ROTATION from its last date."""
    return pd.Series(ROTATION[window.index[-1].day_of_year % 3], index=window.columns)


@pytest.mark.parametrize(("lookback", "rebalance_every", "cost_bps"), [(10, 3, 25.0), (40, 13, 7)])
def test_matches_dollar_holdings_reference(market, lookback, rebalance_every, cost_bps):
    result = walk_forward_backtest(
        market, rotating, lookback=lookback, rebalance_every=rebalance_every, cost_bps=cost_bps
    )
    gross, net, turnover, targets = reference_backtest(
        market, rotating, lookback, rebalance_every, cost_bps
    )
    assert len(np.unique(targets, axis=0)) == 3  # the rotation really happens
    np.testing.assert_allclose(result.gross_returns, gross, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.returns, net, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.turnover, turnover, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.weights, targets, rtol=0, atol=0)
    np.testing.assert_allclose(result.costs, cost_bps / 1e4 * turnover, rtol=0, atol=1e-16)


def test_equal_weight_daily_rebalance_is_cross_sectional_mean(market):
    result = walk_forward_backtest(
        market, equal_weight, lookback=5, rebalance_every=1, cost_bps=0.0
    )
    expected = market.iloc[5:].mean(axis=1)
    np.testing.assert_allclose(result.returns, expected, rtol=0, atol=1e-16)
    np.testing.assert_allclose(result.gross_returns, expected, rtol=0, atol=1e-16)
    assert len(result.weights) == len(market) - 5


def test_buy_and_hold_is_value_weighted_growth(market):
    w0 = np.array([0.1, 0.2, 0.3, 0.4])
    lookback = 50
    result = walk_forward_backtest(
        market, fixed(w0), lookback=lookback, rebalance_every=len(market), cost_bps=0.0
    )
    growth = (1.0 + market.iloc[lookback:]).cumprod().to_numpy()  # per-asset growth of $1
    value = growth @ w0
    expected = value / np.concatenate([[1.0], value[:-1]]) - 1.0
    np.testing.assert_allclose(result.gross_returns, expected, rtol=0, atol=1e-14)
    assert len(result.weights) == 1
    np.testing.assert_allclose(result.turnover, [1.0])
    # Final wealth equals the value of the untouched initial holdings.
    assert (1.0 + result.returns).prod() == pytest.approx(value[-1], rel=1e-13)


def test_turnover_is_measured_against_drifted_weights(market):
    lookback, every = 30, 10
    result = walk_forward_backtest(
        market, fixed([0.25] * 4), lookback=lookback, rebalance_every=every, cost_bps=0.0
    )
    # Before each rebalance the weights are w_i G_i / sum_j w_j G_j, with G the growth
    # of each asset since the previous rebalance.
    expected = [1.0]
    for t in range(lookback + every, len(market), every):
        growth = (1.0 + market.iloc[t - every : t]).prod().to_numpy()
        drifted = 0.25 * growth / (0.25 * growth).sum()
        expected.append(np.abs(0.25 - drifted).sum())
    np.testing.assert_allclose(result.turnover, expected, rtol=1e-12, atol=1e-16)
    assert (result.turnover.iloc[1:] > 0).all()


def test_turnover_from_cash_counts_short_positions():
    returns = pd.DataFrame(np.full((5, 2), 0.001), columns=["x", "y"])
    result = walk_forward_backtest(returns, fixed([1.5, -0.5]), lookback=2, rebalance_every=10)
    assert result.turnover.iloc[0] == pytest.approx(2.0)
    assert result.costs.iloc[0] == pytest.approx(10.0 / 1e4 * 2.0)


def test_costs_are_charged_exactly_on_rebalance_dates(market):
    kwargs = {"lookback": 20, "rebalance_every": 6}
    free = walk_forward_backtest(market, inverse_variance, cost_bps=0.0, **kwargs)
    costly = walk_forward_backtest(market, inverse_variance, cost_bps=25.0, **kwargs)
    pd.testing.assert_series_equal(free.gross_returns, costly.gross_returns)
    pd.testing.assert_series_equal(free.turnover, costly.turnover)
    pd.testing.assert_frame_equal(free.weights, costly.weights)
    np.testing.assert_allclose(costly.costs, 25e-4 * costly.turnover, rtol=0, atol=1e-18)
    charged = (costly.gross_returns - costly.returns).to_numpy()
    expected = np.zeros(len(charged))
    expected[:: kwargs["rebalance_every"]] = costly.costs.to_numpy()
    np.testing.assert_allclose(charged, expected, rtol=0, atol=1e-16)
    np.testing.assert_allclose(free.costs, 0.0)
    pd.testing.assert_series_equal(free.returns, free.gross_returns)


def test_annual_turnover(market):
    result = walk_forward_backtest(market, inverse_variance, lookback=20, rebalance_every=5)
    years = (len(market) - 20) / 252
    assert result.annual_turnover == pytest.approx(result.turnover.sum() / years, rel=1e-14)
    monthly = walk_forward_backtest(
        market, inverse_variance, lookback=20, rebalance_every=5, periods_per_year=12
    )
    assert monthly.annual_turnover == pytest.approx(result.annual_turnover * 12 / 252, rel=1e-14)


# ---------------------------------------------------------------------------
# Output shape and accepted strategy outputs
# ---------------------------------------------------------------------------


def test_result_structure_and_missing_assets(market):
    def two_assets(window: pd.DataFrame) -> pd.Series:
        return pd.Series({"stock": 0.6, "bond": 0.4})

    result = walk_forward_backtest(
        market, two_assets, lookback=21, rebalance_every=21, name="60/40"
    )
    assert isinstance(result, BacktestResult)
    assert result.name == "60/40"
    assert result.returns.name == "60/40"
    assert result.returns.index.equals(market.index[21:])
    assert result.gross_returns.index.equals(market.index[21:])
    assert result.weights.index.equals(market.index[21::21])
    assert result.turnover.index.equals(result.weights.index)
    assert result.costs.index.equals(result.weights.index)
    assert list(result.weights.columns) == list(market.columns)
    assert (result.weights[["gold", "crypto"]] == 0.0).all().all()
    np.testing.assert_allclose(result.weights["stock"], 0.6)
    # Missing assets really are not held: the return only involves stock and bond.
    first = market.iloc[21]
    assert result.gross_returns.iloc[0] == pytest.approx(0.6 * first["stock"] + 0.4 * first["bond"])


def test_numpy_returns_and_array_or_mapping_weights():
    values = np.random.default_rng(5).normal(0.0, 0.01, size=(40, 3))
    by_array = walk_forward_backtest(
        values, lambda window: np.array([0.2, 0.3, 0.5]), lookback=10, rebalance_every=4
    )
    by_mapping = walk_forward_backtest(
        values, lambda window: {0: 0.2, 1: 0.3, 2: 0.5}, lookback=10, rebalance_every=4
    )
    pd.testing.assert_series_equal(by_array.returns, by_mapping.returns)
    assert isinstance(by_array.returns.index, pd.RangeIndex)
    assert list(by_array.weights.index) == list(range(10, 40, 4))


def test_large_backtest_matches_block_formula():
    # The intended scale: 12 assets x 4000 days. Within each holding block the
    # portfolio value is w' cumprod(1 + r), so gross returns are value ratios.
    rng = np.random.default_rng(9)
    n_obs, n_assets, lookback, every = 4000, 12, 252, 21
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (n_obs, n_assets)))
    result = walk_forward_backtest(
        returns, inverse_variance, lookback=lookback, rebalance_every=every
    )
    expected = []
    for k, t in enumerate(range(lookback, n_obs, every)):
        w = result.weights.iloc[k].to_numpy()
        block = returns.iloc[t : t + every].to_numpy()
        value = np.cumprod(1.0 + block, axis=0) @ w
        expected.extend(value / np.concatenate([[1.0], value[:-1]]) - 1.0)
    np.testing.assert_allclose(result.gross_returns, expected, rtol=0, atol=1e-13)
    assert len(result.returns) == n_obs - lookback


# ---------------------------------------------------------------------------
# Several strategies
# ---------------------------------------------------------------------------


def test_run_and_compare_backtests(market):
    strategies = {"1/N": equal_weight, "IVP": inverse_variance}
    results = run_backtests(market, strategies, lookback=30, rebalance_every=10, cost_bps=5.0)
    assert list(results) == ["1/N", "IVP"]
    assert all(results[k].name == k for k in results)
    single = walk_forward_backtest(
        market, inverse_variance, lookback=30, rebalance_every=10, cost_bps=5.0, name="IVP"
    )
    pd.testing.assert_series_equal(results["IVP"].returns, single.returns)

    table = compare_backtests(results, rf=0.02)
    assert list(table.index) == ["1/N", "IVP"]
    assert list(table.columns) == [*metrics.SUMMARY_COLUMNS, "Turnover (annual)"]
    for key, result in results.items():
        expected = metrics.performance_summary(result.returns, rf=0.02).iloc[0]
        np.testing.assert_allclose(
            table.loc[key, list(metrics.SUMMARY_COLUMNS)].to_numpy(dtype=float),
            expected.to_numpy(),
            rtol=1e-14,
        )
        assert table.loc[key, "Turnover (annual)"] == pytest.approx(result.annual_turnover)
    # Statistics are of NET returns.
    assert table.loc["IVP", "CAGR"] == pytest.approx(metrics.annualized_return(single.returns))


def test_compare_uses_each_results_periods_per_year(market):
    monthly = walk_forward_backtest(
        market, equal_weight, lookback=12, rebalance_every=3, periods_per_year=12
    )
    table = compare_backtests({"m": monthly})
    assert table.loc["m", "Volatility"] == pytest.approx(
        metrics.annualized_volatility(monthly.returns, periods_per_year=12)
    )


def test_run_and_compare_validation(market):
    with pytest.raises(ValueError, match="no strategies"):
        run_backtests(market, {})
    with pytest.raises(TypeError, match="name"):
        run_backtests(market, {"a": equal_weight}, name="x")
    with pytest.raises(ValueError, match="no backtest results"):
        compare_backtests({})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lookback": 0}, "lookback"),
        ({"lookback": 160}, "lookback"),
        ({"lookback": 500}, "lookback"),
        ({"lookback": 2.5}, "lookback"),
        ({"lookback": True}, "lookback"),
        ({"rebalance_every": 0}, "rebalance_every"),
        ({"rebalance_every": 1.0}, "rebalance_every"),
        ({"cost_bps": -1.0}, "cost_bps"),
        ({"cost_bps": math.nan}, "cost_bps"),
        ({"periods_per_year": 0}, "periods_per_year"),
    ],
)
def test_invalid_parameters(market, kwargs, match):
    with pytest.raises(ValueError, match=match):
        walk_forward_backtest(market, equal_weight, **{"lookback": 10, **kwargs})


def test_invalid_returns(market):
    bad = market.copy()
    bad.iloc[50, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        walk_forward_backtest(bad, equal_weight, lookback=10)
    dup_cols = market.copy()
    dup_cols.columns = ["a", "a", "b", "c"]
    with pytest.raises(ValueError, match="duplicate asset"):
        walk_forward_backtest(dup_cols, equal_weight, lookback=10)
    with pytest.raises(ValueError, match="sorted"):
        walk_forward_backtest(market.iloc[::-1], equal_weight, lookback=10)
    with pytest.raises(ValueError, match="duplicate dates"):
        walk_forward_backtest(pd.concat([market, market]), equal_weight, lookback=10)
    with pytest.raises(TypeError, match="callable"):
        walk_forward_backtest(market, "equal weight", lookback=10)


def _never_called(window: pd.DataFrame) -> pd.Series:
    raise AssertionError("the strategy must not run on unsorted data")


_DAYS = pd.bdate_range("2019-12-24", periods=12)  # crosses a year boundary


@pytest.mark.parametrize(
    "index",
    [
        pytest.param([str(d.date()) for d in _DAYS][::-1], id="iso-strings-newest-first"),
        pytest.param([d.strftime("%b %d %Y") for d in _DAYS][::-1], id="text-dates-newest-first"),
        pytest.param([d.date() for d in _DAYS][::-1], id="date-objects-newest-first"),
        pytest.param(pd.period_range("2019-01", periods=12, freq="M")[::-1], id="periods"),
        pytest.param(list(range(2030, 2018, -1)), id="descending-integers"),
        pytest.param([str(d.date()) for d in _DAYS[[0, 2, 1, *range(3, 12)]]], id="one-swap"),
    ],
)
def test_rows_out_of_time_order_are_rejected_for_any_date_like_index(index):
    # A newest-first file read without parse_dates would hand the strategy
    # calendar-future rows as its "past" window.
    returns = pd.DataFrame(np.full((12, 2), 0.001), index=index, columns=["a", "b"])
    with pytest.raises(ValueError, match="sorted by date"):
        walk_forward_backtest(returns, _never_called, lookback=3)


def test_date_strings_are_compared_as_dates_not_text():
    values = np.random.default_rng(3).normal(0.0, 0.01, (12, 2))
    by_dates = walk_forward_backtest(
        pd.DataFrame(values, index=_DAYS), equal_weight, lookback=3, rebalance_every=2
    )
    # US-style strings in time order are not in text order ("12/31/2019" > "01/02/2020").
    us_style = [d.strftime("%m/%d/%Y") for d in _DAYS]
    assert sorted(us_style) != us_style
    by_strings = walk_forward_backtest(
        pd.DataFrame(values, index=us_style), equal_weight, lookback=3, rebalance_every=2
    )
    np.testing.assert_array_equal(by_strings.returns.to_numpy(), by_dates.returns.to_numpy())
    assert list(by_strings.returns.index) == us_style[3:]
    # Labels that are not dates carry no time order of their own: rows are trusted.
    labels = [f"day{i}" for i in range(1, 13)]  # "day10" < "day2" as text
    by_labels = walk_forward_backtest(
        pd.DataFrame(values, index=labels), equal_weight, lookback=3, rebalance_every=2
    )
    np.testing.assert_array_equal(by_labels.returns.to_numpy(), by_dates.returns.to_numpy())
    # Distinct strings for the same instant are duplicate dates.
    same_instant = ["2020-01-01T00:00+01:00", "2019-12-31T23:00+00:00", "2020-01-02T00:00+00:00"]
    with pytest.raises(ValueError, match="duplicate dates"):
        walk_forward_backtest(
            pd.DataFrame(np.zeros((3, 2)), index=same_instant), _never_called, lookback=1
        )


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (pd.Series([0.5, np.nan, 0.5, 0.0], index=["bond", "gold", "stock", "crypto"]), "NaN"),
        (pd.Series([], dtype=float), "no weights"),
        (pd.Series([0.5, 0.4], index=["bond", "gold"]), "sum to 0.9"),
        (pd.Series([0.5, 0.5], index=["bond", "oil"]), "unknown assets"),
        (pd.Series([0.5, 0.5], index=["bond", "bond"]), "duplicate"),
        (pd.Series(["x", "y"], index=["bond", "gold"]), "not numeric"),
        (None, "None"),
        (np.array([0.5, 0.5]), "length 4"),
        (np.full((2, 2), 0.25), "1-D"),
    ],
)
def test_invalid_strategy_output_reports_date_and_reason(market, output, match):
    with pytest.raises(ValueError, match=match) as info:
        walk_forward_backtest(market, lambda window: output, lookback=10, name="bad")
    assert "2021-01-18" in str(info.value)  # market.index[10], the first rebalance date
    assert "'bad'" in str(info.value)


def test_weight_sum_tolerance(market):
    ok = pd.Series([0.25, 0.25, 0.25, 0.25 + 5e-7], index=market.columns)
    walk_forward_backtest(market, lambda window: ok, lookback=10)
    off = pd.Series([0.25, 0.25, 0.25, 0.25 + 5e-6], index=market.columns)
    with pytest.raises(ValueError, match="sum to"):
        walk_forward_backtest(market, lambda window: off, lookback=10)


def test_leveraged_blow_up_raises():
    returns = pd.DataFrame(
        [[0.01, 0.0], [0.0, 0.01], [-0.40, 0.30], [0.0, 0.0]],
        columns=["long", "short"],
        index=pd.date_range("2024-03-01", periods=4),
    )
    # 3x long / 2x short: -1.2 - 0.6 = -180% on 2024-03-03.
    with pytest.raises(ValueError, match=r"2024-03-03.*blow-up"):
        walk_forward_backtest(returns, fixed([3.0, -2.0]), lookback=1, rebalance_every=1)


def test_costs_wiping_out_wealth_raise():
    returns = pd.DataFrame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], columns=["a", "b"])
    # Turnover 41 from cash at 250 bp -> cost 1.025 > 100% of wealth.
    with pytest.raises(ValueError, match="wipe out"):
        walk_forward_backtest(returns, fixed([21.0, -20.0]), lookback=1, cost_bps=250.0)
