"""Tests for ``quantfin.portfolio.data``.

The network is never touched: every Yahoo test runs against a fake ``yfinance`` module
injected into ``sys.modules`` (an autouse fixture installs one that fails loudly if a
test forgets to configure it). The synthetic-market and simulation tests check the
known-truth moments with statistical tolerances derived from the sampling distribution
(z-scores), plus exact identities (compounding, telescoping logs, factor-model rank,
scaling invariance) that would break under a wrong sign, transpose or annualization.
"""

from __future__ import annotations

import math
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantfin.portfolio._common import TRADING_DAYS
from quantfin.portfolio.data import (
    DEFAULT_START,
    DEFAULT_UNIVERSE,
    SyntheticMarket,
    load_prices_csv,
    load_prices_yahoo,
    prices_to_returns,
    simulate_returns,
    synthetic_market,
)

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _corr_scale_error(sample: np.ndarray, truth: np.ndarray) -> float:
    """Max |S_ij - C_ij| / sqrt(C_ii C_jj): covariance error on the correlation scale."""
    sd = np.sqrt(np.diag(truth))
    return float(np.abs((sample - truth) / np.outer(sd, sd)).max())


def _generic_cov() -> pd.DataFrame:
    """A 4-asset annual covariance with distinct vols and mixed-sign correlations."""
    vols = np.array([0.10, 0.18, 0.25, 0.32])
    corr = np.array(
        [
            [1.0, 0.6, -0.3, 0.2],
            [0.6, 1.0, 0.1, 0.5],
            [-0.3, 0.1, 1.0, -0.4],
            [0.2, 0.5, -0.4, 1.0],
        ]
    )
    assets = ["W", "X", "Y", "Z"]
    return pd.DataFrame(np.outer(vols, vols) * corr, index=assets, columns=assets)


GENERIC_MU = pd.Series([0.03, 0.06, -0.02, 0.10], index=["W", "X", "Y", "Z"])


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------


def test_default_universe_matches_spec():
    assert list(DEFAULT_UNIVERSE) == [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ",
    ]  # fmt: skip
    assert all(isinstance(d, str) and d for d in DEFAULT_UNIVERSE.values())
    assert DEFAULT_START == "2008-01-01"


# --------------------------------------------------------------------------------------
# prices_to_returns
# --------------------------------------------------------------------------------------


def test_simple_returns_hand_computed():
    idx = _dates(3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 55.0]}, index=idx)
    out = prices_to_returns(prices)
    expected = pd.DataFrame({"A": [0.10, -0.10], "B": [0.0, 0.10]}, index=idx[1:])
    pd.testing.assert_frame_equal(out, expected, rtol=1e-14, atol=1e-15)


def test_log_returns_hand_computed_and_identities():
    rng = np.random.default_rng(7)
    idx = _dates(60)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (60, 3)), axis=0)),
        index=idx,
        columns=list("abc"),
    )
    simple = prices_to_returns(prices, kind="simple")
    log = prices_to_returns(prices, kind="log")
    # log return = ln(1 + simple return), exactly the same rows/columns
    np.testing.assert_allclose(log.to_numpy(), np.log1p(simple.to_numpy()), rtol=1e-12, atol=1e-15)
    # telescoping: sum of log returns = ln(P_T / P_0); compounding: prod(1+R) = P_T / P_0
    growth = (prices.iloc[-1] / prices.iloc[0]).to_numpy()
    np.testing.assert_allclose(log.sum().to_numpy(), np.log(growth), rtol=1e-12)
    np.testing.assert_allclose((1 + simple).prod().to_numpy(), growth, rtol=1e-12)
    # reconstruct the full price path from returns
    rebuilt = prices.iloc[0].to_numpy() * np.exp(log.cumsum().to_numpy())
    np.testing.assert_allclose(rebuilt, prices.iloc[1:].to_numpy(), rtol=1e-12)


def test_unsorted_input_is_sorted_by_date():
    rng = np.random.default_rng(1)
    idx = _dates(20)
    prices = pd.DataFrame(rng.uniform(50, 150, (20, 2)), index=idx, columns=["A", "B"])
    shuffled = prices.iloc[rng.permutation(20)]
    pd.testing.assert_frame_equal(
        prices_to_returns(shuffled), prices_to_returns(prices), check_freq=False
    )


def test_common_start_drops_rows_before_every_asset_trades():
    idx = _dates(6)
    prices = pd.DataFrame(
        {
            "old": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "new": [np.nan, np.nan, 20.0, 21.0, 22.0, 24.0],
        },
        index=idx,
    )
    out = prices_to_returns(prices)
    assert list(out.index) == list(idx[3:])
    np.testing.assert_allclose(out["old"], [13 / 12 - 1, 14 / 13 - 1, 15 / 14 - 1])
    np.testing.assert_allclose(out["new"], [21 / 20 - 1, 22 / 21 - 1, 24 / 22 - 1])


def test_forward_fill_up_to_five_inside_sample():
    idx = _dates(9)
    a = [100.0, 101.0, np.nan, np.nan, np.nan, np.nan, np.nan, 110.0, 111.0]
    b = [50.0 + i for i in range(9)]
    out = prices_to_returns(pd.DataFrame({"A": a, "B": b}, index=idx))
    assert len(out) == 8
    np.testing.assert_allclose(out["A"].iloc[1:6], 0.0)  # filled days: flat price
    assert out["A"].iloc[6] == pytest.approx(110 / 101 - 1, rel=1e-14)  # jump over the gap
    assert out["A"].iloc[7] == pytest.approx(111 / 110 - 1, rel=1e-14)


def test_gap_longer_than_five_raises():
    idx = _dates(10)
    a = [100.0, 101.0, *([np.nan] * 6), 110.0, 111.0]
    prices = pd.DataFrame({"A": a, "B": np.linspace(50, 60, 10)}, index=idx)
    with pytest.raises(ValueError, match=r"'A'.*more than 5 consecutive"):
        prices_to_returns(prices)


def test_trailing_gap_up_to_five_is_filled_but_six_raises():
    idx = _dates(10)
    b = np.linspace(50, 60, 10)
    ok = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0, *[np.nan] * 5], "B": b}, idx)
    out = prices_to_returns(ok)
    np.testing.assert_allclose(out["A"].iloc[4:], 0.0)
    bad = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, *[np.nan] * 6], "B": b}, idx)
    with pytest.raises(ValueError, match="consecutive missing"):
        prices_to_returns(bad)


def test_holiday_on_common_start_is_filled_from_own_history():
    # "B" starts trading on day 2, the same day "A" has a holiday; A's day-2 price must
    # come from A's day-1 price (before the common start), not be left missing.
    idx = _dates(5)
    prices = pd.DataFrame(
        {"A": [10.0, 11.0, np.nan, 12.0, 13.0], "B": [np.nan, np.nan, 5.0, 6.0, 7.0]}, index=idx
    )
    out = prices_to_returns(prices)
    assert list(out.index) == list(idx[3:])
    np.testing.assert_allclose(out["A"], [12 / 11 - 1, 13 / 12 - 1])
    np.testing.assert_allclose(out["B"], [6 / 5 - 1, 7 / 6 - 1])


def test_rows_with_no_prices_are_dropped():
    idx = _dates(4)
    prices = pd.DataFrame({"A": [10.0, np.nan, 12.0, 13.0], "B": [5.0, np.nan, 6.0, 7.0]}, idx)
    out = prices_to_returns(prices)
    assert list(out.index) == [idx[2], idx[3]]  # no fake zero-return day for idx[1]
    np.testing.assert_allclose(out.iloc[0], [12 / 10 - 1, 6 / 5 - 1])


def test_series_input_and_index_name_preserved():
    idx = _dates(3).rename("Date")
    out = prices_to_returns(pd.Series([1.0, 2.0, 3.0], index=idx, name="A"))
    assert list(out.columns) == ["A"]
    assert out.index.name == "Date"
    np.testing.assert_allclose(out["A"], [1.0, 0.5])


@pytest.mark.parametrize(
    ("column", "match"),
    [
        ([1.0, 0.0, 2.0], "strictly positive"),
        ([1.0, -1.0, 2.0], "strictly positive"),
        ([1.0, np.inf, 2.0], "infinite"),
        ([np.nan, np.nan, np.nan], "no prices"),
    ],
)
def test_bad_prices_raise(column, match):
    prices = pd.DataFrame({"A": column, "B": [1.0, 2.0, 3.0]}, index=_dates(3))
    with pytest.raises(ValueError, match=match):
        prices_to_returns(prices)


def test_prices_to_returns_validation():
    good = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=_dates(3))
    with pytest.raises(ValueError, match="kind"):
        prices_to_returns(good, kind="pct")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="DataFrame"):
        prices_to_returns(np.array([[1.0], [2.0]]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        prices_to_returns(pd.DataFrame())
    with pytest.raises(ValueError, match="duplicate"):
        prices_to_returns(pd.DataFrame({"A": [1.0, 2.0]}, index=[_dates(1)[0]] * 2))
    with pytest.raises(ValueError, match="at least two"):
        prices_to_returns(good.iloc[:1])
    with pytest.raises(ValueError, match="numeric"):
        prices_to_returns(pd.DataFrame({"A": ["x", "y"]}, index=_dates(2)))


# --------------------------------------------------------------------------------------
# load_prices_csv
# --------------------------------------------------------------------------------------


def test_csv_round_trip_is_bit_exact(tmp_path: Path):
    rng = np.random.default_rng(3)
    idx = _dates(30).rename("Date")
    frame = pd.DataFrame(rng.lognormal(4, 0.3, (30, 3)), index=idx, columns=["SPY", "TLT", "GLD"])
    frame.iloc[4, 1] = np.nan  # missing cells survive as NaN
    path = tmp_path / "prices.csv"
    frame.to_csv(path)
    for p in (path, str(path)):
        out = load_prices_csv(p)
        assert list(out.columns) == ["SPY", "TLT", "GLD"]
        assert list(out.index) == list(idx)
        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.name == "Date"
        np.testing.assert_array_equal(out.to_numpy(), frame.to_numpy())  # NaN-aware, exact


def test_csv_sorted_and_feeds_prices_to_returns(tmp_path: Path):
    path = tmp_path / "p.csv"
    path.write_text("date,A,B\n2020-01-03,12,6\n2020-01-01,10,5\n2020-01-02,11,\n")
    out = load_prices_csv(path)
    assert list(out.index) == list(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    rets = prices_to_returns(out)
    np.testing.assert_allclose(rets.to_numpy(), [[0.1, 0.0], [12 / 11 - 1, 0.2]])


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("date,A\n1,10\n2,11\n", "dates"),
        ("date,A\nfoo,10\nbar,11\n", "dates"),
        ("date,A\n2020-01-01,10\n2020-01-01,11\n", "duplicate"),
        ("date,A\n2020-01-01,10\n2020-01-02,abc\n", "numeric"),
        ("date\n2020-01-01\n", "price columns"),
        ("date,A\n", "no rows"),
    ],
)
def test_csv_validation(tmp_path: Path, content: str, match: str):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_prices_csv(path)


def test_csv_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_prices_csv(tmp_path / "nope.csv")


# --------------------------------------------------------------------------------------
# load_prices_yahoo (fake yfinance; never the network)
# --------------------------------------------------------------------------------------

FIELDS = ["Close", "High", "Low", "Open", "Volume"]


def _close_values(ticker: str, n: int) -> np.ndarray:
    base = 10.0 + sum(ord(c) for c in ticker) % 97
    return base * (1.0 + 0.001 * np.arange(n)) + 1.0 / 3.0


def _yahoo_frame(
    tickers: list[str], n: int = 8, layout: str = "column", tz: str | None = None
) -> pd.DataFrame:
    """Mimic yfinance.download output (tickers come back alphabetically sorted)."""
    idx = pd.bdate_range("2021-03-01", periods=n, tz=tz, name="Date")
    order = sorted(tickers)
    data = {}
    for t in order:
        close = _close_values(t, n)
        fields = {"Close": close, "High": close * 1.01, "Low": close * 0.98, "Open": close * 0.99}
        fields["Volume"] = np.full(n, 1e6)
        for f in FIELDS:
            data[(t, f) if layout == "ticker" else (f, t)] = fields[f]
    frame = pd.DataFrame(data, index=idx)
    if layout == "flat":  # yfinance < 0.2.48 with a single ticker: Open/High/Low/Close/...
        assert len(tickers) == 1
        frame.columns = pd.Index([f for f, _ in frame.columns])
    else:
        names = ["Price", "Ticker"] if layout == "column" else ["Ticker", "Price"]
        frame.columns = pd.MultiIndex.from_tuples(list(frame.columns), names=names)
    return frame


class FakeYF:
    """Records ``download`` calls; returns whatever ``responder`` builds."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[tuple[list[str], dict]] = []

    def download(self, tickers, **kwargs):
        self.calls.append((list(tickers), kwargs))
        return self.responder(list(tickers), **kwargs)


@pytest.fixture(autouse=True)
def _block_real_yfinance(monkeypatch: pytest.MonkeyPatch):
    """Guarantee no test reaches the real yfinance / network."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("test tried to use the real yfinance")

    module = types.ModuleType("yfinance")
    module.download = _forbidden  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", module)


@pytest.fixture
def install_yf(monkeypatch: pytest.MonkeyPatch):
    def _install(responder=None) -> FakeYF:
        fake = FakeYF(responder or (lambda tickers, **kw: _yahoo_frame(tickers)))
        module = types.ModuleType("yfinance")
        module.download = fake.download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "yfinance", module)
        return fake

    return _install


def test_yahoo_close_field_in_requested_order(install_yf, tmp_path: Path):
    fake = install_yf()
    tickers = ["SPY", "GLD", "AGG"]
    out = load_prices_yahoo(tickers, start="2021-01-01", end="2021-06-01", cache_dir=tmp_path)
    assert list(out.columns) == tickers  # not the alphabetical order the fake returns
    for t in tickers:
        np.testing.assert_array_equal(out[t].to_numpy(), _close_values(t, 8))  # Close, not Open
    assert out.index.name == "Date"
    assert out.index.dtype == np.dtype("datetime64[ns]")
    assert out.index.is_monotonic_increasing
    (called, kwargs) = fake.calls[0]
    assert called == tickers
    assert kwargs["auto_adjust"] is True
    assert kwargs["start"] == "2021-01-01"
    assert kwargs["end"] == "2021-06-01"


def test_yahoo_cache_hit_skips_download_and_round_trips_exactly(
    install_yf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = install_yf()
    cache = tmp_path / "nested" / "cache"
    first = load_prices_yahoo(["QQQ", "TLT"], start="2015-01-01", cache_dir=cache)
    files = list(cache.glob("*.csv"))
    assert len(files) == 1 and len(fake.calls) == 1
    second = load_prices_yahoo(["QQQ", "TLT"], start="2015-01-01", cache_dir=cache)
    assert len(fake.calls) == 1  # served from the cache
    pd.testing.assert_frame_equal(first, second, check_exact=True, check_freq=False)
    # With yfinance uninstalled the cache still works.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    third = load_prices_yahoo(["QQQ", "TLT"], start="2015-01-01", cache_dir=cache)
    pd.testing.assert_frame_equal(first, third, check_exact=True, check_freq=False)


def test_yahoo_cache_is_keyed_by_tickers_start_end(install_yf, tmp_path: Path):
    fake = install_yf()
    requests = [
        (["SPY", "TLT"], "2010-01-01", None),
        (["TLT", "SPY"], "2010-01-01", None),
        (["SPY", "TLT"], "2011-01-01", None),
        (["SPY", "TLT"], "2010-01-01", "2020-01-01"),
    ]
    for tickers, start, end in requests:
        out = load_prices_yahoo(tickers, start=start, end=end, cache_dir=tmp_path)
        assert list(out.columns) == tickers
    assert len(fake.calls) == 4
    assert len(list(tmp_path.glob("*.csv"))) == 4


def test_yahoo_cache_filename_sanitized_and_collision_free(install_yf, tmp_path: Path):
    install_yf()
    load_prices_yahoo(["^GSPC", "BRK-B"], start="2010-01-01", cache_dir=tmp_path)
    load_prices_yahoo(["_GSPC", "BRK-B"], start="2010-01-01", cache_dir=tmp_path)
    many = [f"T{i:03d}" for i in range(200)]
    load_prices_yahoo(many, start="2010-01-01", cache_dir=tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 3  # ^GSPC and _GSPC do not collide
    for name in names:
        assert re.fullmatch(r"[A-Za-z0-9._-]+\.csv", name)
        assert len(name) < 200


def test_yahoo_without_cache_writes_nothing(
    install_yf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = install_yf()
    monkeypatch.chdir(tmp_path)
    load_prices_yahoo(["SPY"], cache_dir=None)
    load_prices_yahoo(["SPY"], cache_dir=None)
    assert len(fake.calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_yahoo_default_arguments(install_yf, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = install_yf()
    monkeypatch.chdir(tmp_path)
    out = load_prices_yahoo()
    assert list(out.columns) == list(DEFAULT_UNIVERSE)
    assert fake.calls[0][1]["start"] == DEFAULT_START
    assert fake.calls[0][1]["end"] is None
    assert len(list((tmp_path / "data" / "cache").glob("*.csv"))) == 1


@pytest.mark.parametrize("layout", ["column", "ticker"])
def test_yahoo_multiindex_layouts(install_yf, tmp_path: Path, layout: str):
    install_yf(lambda tickers, **kw: _yahoo_frame(tickers, layout=layout))
    out = load_prices_yahoo(["IEF", "EEM"], cache_dir=None)
    assert list(out.columns) == ["IEF", "EEM"]
    np.testing.assert_array_equal(out["EEM"].to_numpy(), _close_values("EEM", 8))


def test_yahoo_flat_single_ticker_and_string_ticker(install_yf):
    fake = install_yf(lambda tickers, **kw: _yahoo_frame(tickers, layout="flat"))
    out = load_prices_yahoo("GLD", cache_dir=None)  # a bare string is one ticker, not 3
    assert fake.calls[0][0] == ["GLD"]
    assert list(out.columns) == ["GLD"]
    np.testing.assert_array_equal(out["GLD"].to_numpy(), _close_values("GLD", 8))


def test_yahoo_tz_aware_index_becomes_naive_dates(install_yf):
    install_yf(lambda tickers, **kw: _yahoo_frame(tickers, tz="America/New_York"))
    out = load_prices_yahoo(["SPY"], cache_dir=None)
    assert out.index.tz is None
    assert list(out.index) == list(pd.bdate_range("2021-03-01", periods=8))


def test_yahoo_drops_all_nan_rows_keeps_partial(install_yf):
    def responder(tickers, **kw):
        frame = _yahoo_frame(tickers)
        frame.iloc[2, :] = np.nan
        frame.loc[frame.index[5], ("Close", "SPY")] = np.nan
        return frame

    install_yf(responder)
    out = load_prices_yahoo(["SPY", "TLT"], cache_dir=None)
    assert len(out) == 7
    assert out["SPY"].isna().sum() == 1


@pytest.mark.parametrize("empty", [None, pd.DataFrame()])
def test_yahoo_no_data_raises(install_yf, tmp_path: Path, empty):
    install_yf(lambda tickers, **kw: empty)
    with pytest.raises(RuntimeError, match="no data"):
        load_prices_yahoo(["SPY"], cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_yahoo_all_nan_ticker_raises_and_is_not_cached(install_yf, tmp_path: Path):
    def responder(tickers, **kw):
        frame = _yahoo_frame(tickers)
        frame[("Close", "BAD")] = np.nan
        return frame

    install_yf(responder)
    with pytest.raises(RuntimeError, match=r"missing prices.*BAD"):
        load_prices_yahoo(["SPY", "BAD"], cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_yahoo_ticker_absent_from_result_raises(install_yf):
    install_yf(lambda tickers, **kw: _yahoo_frame([t for t in tickers if t != "ZZZ"]))
    with pytest.raises(RuntimeError, match="ZZZ"):
        load_prices_yahoo(["SPY", "ZZZ"], cache_dir=None)


def test_yahoo_duplicate_price_columns_raise(install_yf):
    def responder(tickers, **kw):
        frame = _yahoo_frame(tickers)
        return pd.concat([frame, frame[[("Close", "SPY")]]], axis=1)

    install_yf(responder)
    with pytest.raises(RuntimeError, match="duplicate"):
        load_prices_yahoo(["SPY", "TLT"], cache_dir=None)


def test_yahoo_corrupt_cache_is_redownloaded(install_yf, tmp_path: Path):
    fake = install_yf()
    load_prices_yahoo(["SPY", "TLT"], cache_dir=tmp_path)
    (path,) = tmp_path.glob("*.csv")
    pd.read_csv(path, index_col=0)[["SPY"]].to_csv(path)  # drop a column
    out = load_prices_yahoo(["SPY", "TLT"], cache_dir=tmp_path)
    assert len(fake.calls) == 2
    assert list(out.columns) == ["SPY", "TLT"]


def test_yahoo_missing_dependency_gives_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setitem(sys.modules, "yfinance", None)  # makes `import yfinance` fail
    with pytest.raises(ImportError, match=re.escape('pip install -e ".[data]"')):
        load_prices_yahoo(["SPY"], cache_dir=tmp_path)


@pytest.mark.parametrize("tickers", [[], ["SPY", "SPY"], ["SPY", ""], ["SPY", 3]])
def test_yahoo_ticker_validation(install_yf, tickers):
    fake = install_yf()
    with pytest.raises(ValueError, match="tickers"):
        load_prices_yahoo(tickers, cache_dir=None)
    assert fake.calls == []


# --------------------------------------------------------------------------------------
# synthetic_market
# --------------------------------------------------------------------------------------

LONG_YEARS = 200  # 50,400 daily rows (stays inside the datetime64[ns] range on pandas 2)


@pytest.fixture(scope="module")
def default_market() -> SyntheticMarket:
    return synthetic_market()


@pytest.fixture(scope="module")
def long_t_market() -> SyntheticMarket:
    return synthetic_market(n_years=LONG_YEARS, seed=1, tail_df=5.0)


@pytest.fixture(scope="module")
def long_gauss_market() -> SyntheticMarket:
    return synthetic_market(n_years=LONG_YEARS, seed=2, tail_df=None)


def test_synthetic_shapes_labels_and_dates(default_market: SyntheticMarket):
    m = default_market
    n = round(15.0 * TRADING_DAYS)
    assets = [f"A{i:02d}" for i in range(1, 11)]
    assert m.returns.shape == (n, 10) and m.prices.shape == (n + 1, 10)
    assert list(m.returns.columns) == assets == list(m.prices.columns)
    assert list(m.true_mu.index) == assets
    assert list(m.true_cov.index) == assets == list(m.true_cov.columns)
    pd.testing.assert_index_equal(m.returns.index, pd.bdate_range("2005-01-03", periods=n))
    assert m.returns.index.freqstr == "B"
    assert m.prices.index[0] == pd.Timestamp("2004-12-31")  # business day before start
    np.testing.assert_array_equal(m.prices.iloc[0].to_numpy(), 100.0)
    assert np.isfinite(m.returns.to_numpy()).all()
    with pytest.raises(AttributeError):
        m.returns = m.returns  # type: ignore[misc]  # frozen dataclass


def test_synthetic_prices_compound_returns(default_market: SyntheticMarket):
    m = default_market
    p = m.prices.to_numpy()
    r = m.returns.to_numpy()
    np.testing.assert_allclose(p[1:] / p[:-1] - 1.0, r, rtol=0, atol=1e-12)
    np.testing.assert_allclose(p[-1], 100.0 * np.prod(1.0 + r, axis=0), rtol=1e-10)
    # and the package's own converter recovers the returns exactly, with the same dates
    back = prices_to_returns(m.prices)
    pd.testing.assert_index_equal(back.index, m.returns.index, check_exact=True)
    np.testing.assert_allclose(back.to_numpy(), r, rtol=0, atol=1e-12)


def test_synthetic_weekend_start_rolls_forward():
    m = synthetic_market(n_assets=3, n_years=0.5, start="2005-01-01")  # a Saturday
    n = round(0.5 * TRADING_DAYS)
    pd.testing.assert_index_equal(m.returns.index, pd.bdate_range("2005-01-01", periods=n))
    assert m.returns.index[0] == pd.Timestamp("2005-01-03")
    assert m.prices.index[0] == pd.Timestamp("2004-12-31")


def test_true_cov_symmetric_positive_definite(default_market: SyntheticMarket):
    cov = default_market.true_cov.to_numpy()
    np.testing.assert_array_equal(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    np.linalg.cholesky(cov)


@pytest.mark.parametrize("n_factors", [1, 2, 3, 4])
def test_true_cov_has_exact_factor_structure(n_factors: int):
    # Cov = B diag(sf^2) B' + D with D diagonal, so any off-diagonal block Cov[I, J]
    # (I, J disjoint) equals B_I diag(sf^2) B_J' and has rank exactly K = n_factors.
    m = synthetic_market(n_assets=12, n_years=0.1, n_factors=n_factors, seed=5)
    block = m.true_cov.to_numpy()[:6, 6:]
    sv = np.linalg.svd(block, compute_uv=False)
    assert sv[n_factors - 1] / sv[0] > 1e-6
    if n_factors < 6:
        assert sv[n_factors] / sv[0] < 1e-12


def test_one_factor_tetrad_identity():
    # One-factor model: c_ij c_kq = c_iq c_kj for distinct i, j, k, q (Spearman's tetrads).
    c = synthetic_market(n_assets=6, n_years=0.1, n_factors=1, seed=9).true_cov.to_numpy()
    for i, j, k, q in [(0, 1, 2, 3), (1, 4, 3, 5), (5, 0, 2, 4)]:
        assert c[i, j] * c[k, q] == pytest.approx(c[i, q] * c[k, j], rel=1e-12)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("n_factors", [1, 3, 5])
def test_synthetic_calibration_ranges(seed: int, n_factors: int):
    m = synthetic_market(n_years=0.1, seed=seed, n_factors=n_factors)
    cov = m.true_cov.to_numpy()
    vols = np.sqrt(np.diag(cov))
    assert vols.min() >= 0.08 and vols.max() <= 0.30
    assert m.true_mu.min() >= 0.02 and m.true_mu.max() <= 0.12
    corr = cov / np.outer(vols, vols)
    pairs = corr[np.triu_indices(10, 1)]
    assert pairs.min() > 0.0  # positive market betas dominate
    assert np.mean((pairs >= 0.1) & (pairs <= 0.8)) >= 0.9
    # riskier assets earn more on average (mu built as 1% + 0.3 vol + noise)
    assert np.corrcoef(vols, m.true_mu.to_numpy())[0, 1] > 0.5


@pytest.mark.parametrize("fixture_name", ["long_t_market", "long_gauss_market"])
def test_sample_mean_converges_to_true_mu(fixture_name: str, request: pytest.FixtureRequest):
    # Annualized sample mean = 252 * daily mean; its standard error is
    # sqrt(252^2 * (Sigma_ii / 252) / n) = sqrt(Sigma_ii / n_years).
    m: SyntheticMarket = request.getfixturevalue(fixture_name)
    mu_hat = m.returns.mean().to_numpy() * TRADING_DAYS
    se = np.sqrt(np.diag(m.true_cov.to_numpy()) / LONG_YEARS)
    z = (mu_hat - m.true_mu.to_numpy()) / se
    assert np.abs(z).max() < 4.5


def test_sample_cov_converges_gaussian_z_scores(long_gauss_market: SyntheticMarket):
    # Gaussian: Var(S_ij) ~= (Sigma_ij^2 + Sigma_ii Sigma_jj) / n (Anderson, 2003, Thm 3.4.4).
    m = long_gauss_market
    n = len(m.returns)
    truth = m.true_cov.to_numpy()
    sample = np.cov(m.returns.to_numpy(), rowvar=False, ddof=1) * TRADING_DAYS
    se = np.sqrt((truth**2 + np.outer(np.diag(truth), np.diag(truth))) / n)
    assert np.abs((sample - truth) / se).max() < 4.5
    # a wrong annualization (sqrt(252) instead of 252) would be off by a factor ~16
    assert _corr_scale_error(sample, truth) < 0.05


def test_sample_cov_converges_student_t(long_t_market: SyntheticMarket):
    m = long_t_market
    sample = np.cov(m.returns.to_numpy(), rowvar=False, ddof=1) * TRADING_DAYS
    assert _corr_scale_error(sample, m.true_cov.to_numpy()) < 0.06


def test_fat_tails_only_with_tail_df(long_t_market, long_gauss_market):
    kurt_t = stats.kurtosis(long_t_market.returns.to_numpy(), axis=0)  # excess kurtosis
    kurt_g = stats.kurtosis(long_gauss_market.returns.to_numpy(), axis=0)
    assert kurt_t.min() > 0.5
    # Gaussian: SE of sample excess kurtosis ~ sqrt(24 / n) ~= 0.022
    assert np.abs(kurt_g).max() < 0.12


def test_synthetic_reproducible_by_seed():
    a = synthetic_market(n_assets=5, n_years=2, seed=11)
    b = synthetic_market(n_assets=5, n_years=2, seed=11)
    c = synthetic_market(n_assets=5, n_years=2, seed=12)
    pd.testing.assert_frame_equal(a.prices, b.prices, check_exact=True)
    pd.testing.assert_frame_equal(a.returns, b.returns, check_exact=True)
    pd.testing.assert_series_equal(a.true_mu, b.true_mu, check_exact=True)
    pd.testing.assert_frame_equal(a.true_cov, b.true_cov, check_exact=True)
    assert not np.allclose(a.returns.to_numpy(), c.returns.to_numpy())
    assert not np.allclose(a.true_mu.to_numpy(), c.true_mu.to_numpy())


def test_truth_does_not_depend_on_sample_length_or_tails():
    a = synthetic_market(seed=4, n_years=1, tail_df=5.0)
    b = synthetic_market(seed=4, n_years=20, tail_df=None)
    pd.testing.assert_series_equal(a.true_mu, b.true_mu, check_exact=True)
    pd.testing.assert_frame_equal(a.true_cov, b.true_cov, check_exact=True)


def test_tail_df_inf_equals_gaussian():
    a = synthetic_market(n_assets=4, n_years=1, seed=3, tail_df=math.inf)
    b = synthetic_market(n_assets=4, n_years=1, seed=3, tail_df=None)
    pd.testing.assert_frame_equal(a.returns, b.returns, check_exact=True)


def test_synthetic_small_and_large_universes():
    one = synthetic_market(n_assets=1, n_factors=1, n_years=1)
    assert list(one.returns.columns) == ["A01"] and one.true_cov.shape == (1, 1)
    many = synthetic_market(n_assets=12, n_factors=8, n_years=1.5)
    assert many.returns.shape == (378, 12)
    assert list(many.returns.columns)[-1] == "A12"


def test_extreme_tails_that_would_bankrupt_an_asset_raise():
    # tail_df just above 2 has enormous shocks; seed 1 produces a daily return <= -100%.
    with pytest.raises(ValueError, match="-100%"):
        synthetic_market(seed=1, tail_df=2.05, n_years=15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_assets": 0},
        {"n_assets": 2.5},
        {"n_assets": True},
        {"n_factors": 0},
        {"n_years": 0.0},
        {"n_years": -1.0},
        {"n_years": float("nan")},
        {"n_years": float("inf")},
        {"n_years": 1 / TRADING_DAYS},  # a single return
        {"tail_df": 2.0},
        {"tail_df": 1.0},
        {"tail_df": float("nan")},
    ],
)
def test_synthetic_validation(kwargs):
    with pytest.raises(ValueError):
        synthetic_market(**kwargs)


# --------------------------------------------------------------------------------------
# simulate_returns
# --------------------------------------------------------------------------------------


def test_simulate_gaussian_moments_match_daily_truth():
    cov = _generic_cov()
    n = 200_000
    out = simulate_returns(GENERIC_MU, cov, n, np.random.default_rng(0))
    assert isinstance(out.index, pd.RangeIndex) and len(out) == n
    assert list(out.columns) == ["W", "X", "Y", "Z"]
    daily_cov = cov.to_numpy() / TRADING_DAYS
    # mean: daily mu / 252 with SE sqrt(Sigma_ii / 252 / n)
    z_mean = (out.mean().to_numpy() - GENERIC_MU.to_numpy() / TRADING_DAYS) / np.sqrt(
        np.diag(daily_cov) / n
    )
    assert np.abs(z_mean).max() < 4.5
    # covariance z-scores (Gaussian sampling variance of S_ij)
    sample = np.cov(out.to_numpy(), rowvar=False, ddof=1)
    se = np.sqrt((daily_cov**2 + np.outer(np.diag(daily_cov), np.diag(daily_cov))) / n)
    assert np.abs((sample - daily_cov) / se).max() < 4.5


def test_simulate_would_detect_a_transposed_factor():
    # Sanity check that the moment test above is sensitive to A vs A': for this
    # covariance, L'L differs from LL' = Sigma by far more than the sampling error.
    chol = np.linalg.cholesky(_generic_cov().to_numpy())
    assert _corr_scale_error(chol.T @ chol, chol @ chol.T) > 0.2
    out = simulate_returns(GENERIC_MU, _generic_cov(), 100_000, np.random.default_rng(1))
    sample = np.cov(out.to_numpy(), rowvar=False) * TRADING_DAYS
    assert _corr_scale_error(sample, _generic_cov().to_numpy()) < 0.03


def test_simulate_student_t_covariance_and_kurtosis():
    cov = _generic_cov()
    out = simulate_returns(GENERIC_MU, cov, 400_000, np.random.default_rng(2), tail_df=5.0)
    sample = np.cov(out.to_numpy(), rowvar=False, ddof=1) * TRADING_DAYS
    assert _corr_scale_error(sample, cov.to_numpy()) < 0.04
    assert stats.kurtosis(out.to_numpy(), axis=0).min() > 2.0  # theory: 6 / (5 - 4) = 6


def test_simulate_student_t_closed_form_kurtosis_and_tail_dependence():
    # Multivariate t (nu = 10) with identity scale: marginal excess kurtosis 6/(nu-4) = 1,
    # and the shared chi^2 mixing makes squared returns correlated:
    # corr(X1^2, X2^2) = (2/(nu-4)) / ((2nu-2)/(nu-4)) = 1/(nu-1) = 1/9, versus 0 for
    # independent t marginals or a Gaussian.
    nu, n = 10.0, 400_000
    eye = pd.DataFrame(np.eye(2) * TRADING_DAYS, index=["p", "q"], columns=["p", "q"])
    zero = pd.Series([0.0, 0.0], index=["p", "q"])
    x = simulate_returns(zero, eye, n, np.random.default_rng(3), tail_df=nu).to_numpy()
    np.testing.assert_allclose(x.var(axis=0, ddof=1), 1.0, atol=0.02)
    np.testing.assert_allclose(stats.kurtosis(x, axis=0), 6 / (nu - 4), atol=0.25)
    rho_sq = np.corrcoef(x[:, 0] ** 2, x[:, 1] ** 2)[0, 1]
    assert rho_sq == pytest.approx(1 / (nu - 1), abs=0.03)
    g = simulate_returns(zero, eye, n, np.random.default_rng(3)).to_numpy()
    assert abs(np.corrcoef(g[:, 0] ** 2, g[:, 1] ** 2)[0, 1]) < 0.015
    assert np.abs(stats.kurtosis(g, axis=0)).max() < 0.05


def test_simulate_zero_covariance_gives_exact_daily_mean_aligned_by_label():
    # Degenerate (PSD) zero covariance: every row equals mu / 252 exactly, which pins
    # down both the annualization factor and label alignment of a reordered mu.
    cov = pd.DataFrame(np.zeros((3, 3)), index=list("abc"), columns=list("abc"))
    mu = pd.Series({"c": 0.252, "a": -0.504, "b": 0.0})
    for tail_df in (None, 4.0):
        out = simulate_returns(mu, cov, 5, np.random.default_rng(0), tail_df=tail_df)
        assert list(out.columns) == ["a", "b", "c"]
        np.testing.assert_allclose(out.to_numpy(), np.tile([-0.002, 0.0, 0.001], (5, 1)))


def test_simulate_singular_cov_perfectly_correlated_assets():
    vol = 0.2
    cov = np.array([[vol**2, 2 * vol**2], [2 * vol**2, 4 * vol**2]])  # asset 2 = 2 x asset 1
    out = simulate_returns(np.array([0.05, 0.10]), cov, 1000, np.random.default_rng(4))
    assert list(out.columns) == [0, 1]
    dev = out - np.array([0.05, 0.10]) / TRADING_DAYS
    np.testing.assert_allclose(dev[1].to_numpy(), 2 * dev[0].to_numpy(), rtol=1e-9, atol=1e-15)
    assert dev[0].std() == pytest.approx(vol / math.sqrt(TRADING_DAYS), rel=0.1)


@pytest.mark.parametrize("tail_df", [None, 6.0])
def test_simulate_scale_equivariance(tail_df):
    # Same random draws: (c mu, c^2 Sigma) must give exactly c times the returns.
    c = 3.0
    base = simulate_returns(GENERIC_MU, _generic_cov(), 50, np.random.default_rng(5), tail_df)
    scaled = simulate_returns(
        c * GENERIC_MU, c**2 * _generic_cov(), 50, np.random.default_rng(5), tail_df
    )
    np.testing.assert_allclose(scaled.to_numpy(), c * base.to_numpy(), rtol=1e-12, atol=1e-16)


def test_simulate_reproducible_and_consumes_rng():
    rng = np.random.default_rng(9)
    first = simulate_returns(GENERIC_MU, _generic_cov(), 10, rng)
    second = simulate_returns(GENERIC_MU, _generic_cov(), 10, rng)
    again = simulate_returns(GENERIC_MU, _generic_cov(), 10, np.random.default_rng(9))
    pd.testing.assert_frame_equal(first, again, check_exact=True)
    assert not np.allclose(first.to_numpy(), second.to_numpy())


def test_simulate_tail_df_inf_is_gaussian_and_single_row():
    a = simulate_returns(GENERIC_MU, _generic_cov(), 1, np.random.default_rng(0), math.inf)
    b = simulate_returns(GENERIC_MU, _generic_cov(), 1, np.random.default_rng(0), None)
    assert a.shape == (1, 4)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_simulate_validation():
    rng = np.random.default_rng(0)
    cov = _generic_cov()
    not_psd = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    with pytest.raises(ValueError, match="semi-definite"):
        simulate_returns(pd.Series([0.0, 0.0], index=["a", "b"]), not_psd, 10, rng)
    with pytest.raises(ValueError, match="symmetric"):
        simulate_returns(np.zeros(2), np.array([[1.0, 0.5], [0.1, 1.0]]), 10, rng)
    with pytest.raises(ValueError, match="missing"):
        simulate_returns(pd.Series([0.1], index=["W"]), cov, 10, rng)
    with pytest.raises(ValueError, match="expected 4"):
        simulate_returns(np.zeros(3), cov, 10, rng)
    with pytest.raises(ValueError, match="NaN"):
        simulate_returns(GENERIC_MU, cov.replace(cov.iloc[0, 0], np.nan), 10, rng)
    for bad_n in (0, -5, 2.5, True):
        with pytest.raises(ValueError, match="n_obs"):
            simulate_returns(GENERIC_MU, cov, bad_n, rng)  # type: ignore[arg-type]
    for bad_df in (2.0, 0.5, float("nan")):
        with pytest.raises(ValueError, match="tail_df"):
            simulate_returns(GENERIC_MU, cov, 10, rng, tail_df=bad_df)
    for bad_rng in (0, np.random.RandomState(0)):
        with pytest.raises(TypeError, match="Generator"):
            simulate_returns(GENERIC_MU, cov, 10, bad_rng)  # type: ignore[arg-type]
