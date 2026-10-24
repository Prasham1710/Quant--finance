"""Price data: Yahoo Finance / CSV loaders and a synthetic market with known truth.

Two kinds of data feed the portfolio experiments:

* **Real prices** (:func:`load_prices_yahoo`, :func:`load_prices_csv`), turned into
  clean, aligned simple returns by :func:`prices_to_returns`.
* **A synthetic market** (:func:`synthetic_market`) whose true expected returns and
  true covariance are known exactly. Because the truth is known, we can measure how
  far an optimizer's *estimated* optimum is from the *true* optimum — the central
  lesson of the project (estimation error is magnified by mean-variance optimization).
  :func:`simulate_returns` draws i.i.d. samples from any given ``(mu, cov)`` for
  Monte Carlo experiments.

All annualized quantities use ``TRADING_DAYS = 252`` (see ``_common``).
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import re
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from quantfin.portfolio._common import TRADING_DAYS, as_cov_frame, as_series

DEFAULT_UNIVERSE: dict[str, str] = {
    "SPY": "US large cap equities (S&P 500)",
    "QQQ": "US tech / growth equities (Nasdaq-100)",
    "IWM": "US small cap equities (Russell 2000)",
    "EFA": "Developed markets ex-US equities (MSCI EAFE)",
    "EEM": "Emerging markets equities (MSCI EM)",
    "TLT": "US Treasuries, 20+ years",
    "IEF": "US Treasuries, 7-10 years",
    "LQD": "US investment-grade corporate bonds",
    "HYG": "US high-yield corporate bonds",
    "GLD": "Gold",
    "DBC": "Broad commodities",
    "VNQ": "US real estate investment trusts (REITs)",
}
"""Ticker -> description for 12 liquid ETFs spanning the major asset classes."""

DEFAULT_START = "2008-01-01"
"""Default first date for downloads (covers the 2008 crisis; all default ETFs trade by then)."""

_MAX_FILL_GAP = 5
"""Longest run of missing prices (inside an asset's history) that is forward-filled."""

_INSTALL_HINT = 'pip install -e ".[data]"'

_UTC_OFFSET = re.compile(r"(\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*(?:Z|[+-]\d{2}:?\d{2})$")
"""A trailing UTC offset (``Z``, ``-05:00``, ``+0900``) after a clock time; group 1 = the time.

Requiring the clock time keeps date-only text such as ``01-02-2020`` intact.
"""

# pandas 2 indexes dates at nanosecond resolution only; whole days inside that range.
_FIRST_NS_DAY = np.datetime64(pd.Timestamp.min.ceil("D").date(), "D")  # 1677-09-22
_LAST_NS_DAY = np.datetime64(pd.Timestamp.max.floor("D").date(), "D")  # 2262-04-11


# --------------------------------------------------------------------------------------
# Loading real prices
# --------------------------------------------------------------------------------------


def _validate_tickers(tickers: Sequence[str]) -> list[str]:
    """Canonical Yahoo symbols (whitespace stripped, upper-case), in the order given.

    Yahoo symbols are case-insensitive and ``yfinance.download`` upper-cases whatever it
    is given (``set(t.upper() for t in tickers)``), so its result has a ``"SPY"`` column
    even when asked for ``"spy"``. Using the canonical spelling to request, match, label
    and cache keeps the result independent of how the symbols were typed, and makes
    ``["SPY", "spy"]`` the duplicate it really is.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    if not tickers:
        raise ValueError("tickers must contain at least one symbol")
    if not all(isinstance(t, str) and t.strip() for t in tickers):
        raise ValueError(f"tickers must be non-empty strings, got {tickers!r}")
    symbols = [t.strip().upper() for t in tickers]
    if len(set(symbols)) != len(symbols):
        raise ValueError(
            f"tickers contain duplicates (Yahoo symbols are case-insensitive): {tickers!r}"
        )
    return symbols


def _cache_path(cache_dir: str | Path, tickers: list[str], start: str, end: str | None) -> Path:
    """File name keyed by tickers + start + end.

    The readable part is sanitized to ``[A-Za-z0-9._-]`` and truncated; a short hash of
    the exact key keeps distinct requests apart even when sanitization or truncation
    would make their readable parts collide (e.g. ``^GSPC`` vs ``_GSPC``).
    """
    key = "|".join([",".join(tickers), str(start), "latest" if end is None else str(end)])
    readable = "_".join(["-".join(tickers), str(start), "latest" if end is None else str(end)])
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", readable)[:120]
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return Path(cache_dir) / f"prices_{readable}_{digest}.csv"


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the (adjusted) ``Close`` field out of a ``yfinance.download`` result.

    yfinance has returned several layouts over its versions: MultiIndex columns
    ``(field, ticker)`` (``group_by="column"``, the default), ``(ticker, field)``
    (``group_by="ticker"``), or flat ``Open/High/Low/Close/...`` columns for a single
    ticker (versions before 0.2.48).
    """
    columns = raw.columns
    if isinstance(columns, pd.MultiIndex):
        for level in range(columns.nlevels):
            if "Close" in columns.get_level_values(level):
                close = raw.xs("Close", axis=1, level=level)
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=tickers[0])
                return close
        raise RuntimeError("yfinance result has no 'Close' field; cannot extract prices")
    if "Close" in columns and len(tickers) == 1:
        return raw[["Close"]].set_axis([tickers[0]], axis=1)
    if all(t in columns for t in tickers):
        return raw[tickers]
    raise RuntimeError(
        f"could not find close prices for {tickers} in yfinance columns {list(columns)}"
    )


def _normalize_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Tz-naive nanosecond DatetimeIndex, so fresh downloads and cached CSVs compare equal."""
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        # Keep the exchange-local calendar date rather than converting to UTC.
        dates = dates.tz_localize(None)
    return dates.as_unit("ns")


def _parse_dates(labels: pd.Index) -> pd.DatetimeIndex:
    """Parse date labels (text, ``date`` or ``Timestamp`` objects) to a tz-naive index.

    A trailing UTC offset is removed *before* parsing, keeping the wall-clock time, i.e.
    the exchange-local calendar date (as :func:`_normalize_dates` does for tz-aware
    indexes). Parsing the offsets instead fails whenever they change at a DST switch
    (``-05:00`` -> ``-04:00`` in a tz-aware frame written by ``DataFrame.to_csv``), and
    converting to UTC would move every date of a UTC+ exchange back by one day.

    Returns:
        DatetimeIndex named like ``labels``; labels pandas reads as missing become NaT.

    Raises:
        ValueError: if the labels cannot be parsed as dates (same message on pandas 2
            and 3).
    """
    text = pd.Index([str(label) for label in labels], dtype=object)
    text = text.str.strip().str.replace(_UTC_OFFSET, r"\1", regex=True)
    try:
        with warnings.catch_warnings():
            # Format-inference (UserWarning) and pandas-2 mixed-offset (FutureWarning)
            # notices: the result is validated by the callers (NaT, duplicates).
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            dates = _normalize_dates(pd.to_datetime(text))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"could not parse the labels as dates ({exc})") from exc
    return dates.rename(labels.name)


def _read_cache(path: Path, tickers: list[str]) -> pd.DataFrame | None:
    """Cached prices for ``tickers``, or ``None`` if the file is absent or unusable.

    An unusable file (empty, unparseable, duplicate dates, a missing or all-NaN ticker)
    can only come from an outside edit or an older non-atomic write; it is treated as a
    cache miss so a fresh download replaces it instead of failing on every call.
    """
    if not path.is_file():
        return None
    try:
        cached = load_prices_csv(path)
    except (ValueError, OSError):  # ValueError covers EmptyDataError, ParserError, decoding
        return None
    if any(t not in cached.columns for t in tickers):
        return None
    cached = cached[tickers]
    if cached.empty or cached.isna().all(axis=0).any():
        return None
    return cached


def _write_cache(prices: pd.DataFrame, path: Path) -> None:
    """Write the CSV cache atomically: a temporary file in the same directory, renamed.

    ``os.replace`` is atomic on POSIX and Windows (same file system), so an interrupted
    write leaves either no cache or a complete one, never a truncated file that a later
    call would serve as if it were the full history.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # newline="" as pandas recommends for handles (no "\r\r\n" on Windows).
        with open(tmp, "x", encoding="utf-8", newline="") as handle:
            prices.to_csv(handle, index_label="Date")
            handle.flush()
            os.fsync(handle.fileno())  # data on disk before the rename publishes it
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _collapse_repeated_dates(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop exact repeats of a dated row; raise if one date carries different prices.

    Yahoo occasionally sends the same day twice. Identical rows are harmless and are
    collapsed; differing rows leave no way to tell which price is right.
    """
    if not prices.index.has_duplicates:
        return prices
    # Integer column labels so a ticker can never clash with the date column; NaN
    # equals NaN for ``duplicated``, so a repeated row with a gap is still a repeat.
    rows = pd.concat(
        [pd.Series(prices.index), pd.DataFrame(prices.to_numpy())], axis=1, ignore_index=True
    )
    prices = prices[~rows.duplicated().to_numpy()]
    if prices.index.has_duplicates:
        dates = prices.index[prices.index.duplicated()].unique()
        raise RuntimeError(
            "Yahoo Finance returned conflicting prices for the same date(s) "
            f"{[f'{d:%Y-%m-%d}' for d in dates[:5]]}; retry later or choose another `end`"
        )
    return prices


def load_prices_yahoo(
    tickers: Sequence[str] = tuple(DEFAULT_UNIVERSE),
    start: str = DEFAULT_START,
    end: str | None = None,
    cache_dir: str | Path | None = "data/cache",
) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance (via ``yfinance``).

    Prices are split- and dividend-adjusted (``yfinance.download(..., auto_adjust=True)``),
    so ``P_t / P_{t-1} - 1`` is the total return including distributions.

    Symbols are case-insensitive and surrounding whitespace is ignored, as on Yahoo
    (``yfinance`` upper-cases them): ``"spy"`` is requested, labelled and cached as
    ``"SPY"``.

    If ``cache_dir`` is set, the result is stored as a CSV whose name is keyed by
    ``tickers``, ``start`` and ``end``; later calls with the same key read the CSV and
    never touch the network (nor need ``yfinance`` installed). The file is written
    atomically, and an unreadable or incomplete cache file is replaced by a fresh
    download. With ``end=None`` the cache is keyed as ``"latest"`` and is *not*
    refreshed automatically — delete the file to pull newer data.

    Args:
        tickers: Yahoo symbols; the output columns are their upper-case forms, in this
            order.
        start: first date (inclusive), ``"YYYY-MM-DD"``.
        end: last date (exclusive, as in yfinance) or ``None`` for today.
        cache_dir: directory for the CSV cache, or ``None`` to disable caching.

    Returns:
        Wide DataFrame of prices: tz-naive ``DatetimeIndex`` named ``"Date"`` (ascending,
        unique), one float column per ticker. Holiday mismatches may leave NaNs; clean
        them with :func:`prices_to_returns`.

    Raises:
        ValueError: if ``tickers`` is empty, contains non-strings or duplicates (ignoring
            case).
        ImportError: if ``yfinance`` is needed (cache miss) but not installed.
        RuntimeError: if the download returns no data, any ticker has only NaNs, or one
            date comes with two different rows of prices.
    """
    tickers = _validate_tickers(tickers)
    path = _cache_path(cache_dir, tickers, start, end) if cache_dir is not None else None
    if path is not None:
        cached = _read_cache(path, tickers)
        if cached is not None:
            return cached

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            f"load_prices_yahoo needs the optional dependency 'yfinance'; install it with "
            f"`{_INSTALL_HINT}` (or `pip install yfinance`)"
        ) from exc

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no data for {tickers} from {start} to {end or 'today'}; "
            "check the symbols, the date range and your network connection"
        )
    close = _extract_close(raw, tickers)
    if close.columns.has_duplicates:
        raise RuntimeError(f"yfinance result has duplicate price columns: {list(close.columns)}")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no prices for tickers: {missing}")
    prices = close[tickers].astype(float)
    prices.index = _normalize_dates(prices.index)
    prices.index.name = "Date"
    prices.columns = pd.Index(tickers)
    prices = prices.sort_index().dropna(how="all")
    all_nan = [t for t in tickers if prices[t].isna().all()]
    if prices.empty or all_nan:
        raise RuntimeError(
            f"Yahoo Finance returned only missing prices for tickers: {all_nan or tickers}"
        )
    # Before caching: a repeated date would otherwise be returned now and then make
    # every later read of the cache fail (load_prices_csv rejects duplicate dates).
    prices = _collapse_repeated_dates(prices)

    if path is not None:
        _write_cache(prices, path)
    return prices


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Read a wide price CSV: first column dates, one column of prices per asset.

    Missing cells are kept as NaN (see :func:`prices_to_returns` for alignment). Floats
    are parsed with ``float_precision="round_trip"`` so a frame written by
    ``DataFrame.to_csv`` reads back bit-for-bit. Dates written with a UTC offset (a
    tz-aware frame, whose offset changes at DST switches) keep their local calendar
    date and time; the offset is dropped, not converted to UTC.

    Args:
        path: CSV file path.

    Returns:
        Float DataFrame with a tz-naive ``DatetimeIndex`` sorted ascending.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if there are no price columns or rows, the first column is not
            parseable as dates, dates repeat, or a price cell is not numeric.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"price file not found: {path}")
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
    if frame.shape[1] == 0:
        raise ValueError(f"{path}: expected a date column followed by price columns")
    if frame.shape[0] == 0:
        raise ValueError(f"{path}: contains no rows")
    # Integers would otherwise be read as nanoseconds since 1970 — silent garbage.
    if pd.api.types.is_numeric_dtype(frame.index):
        raise ValueError(f"{path}: first column must contain dates, got numbers")
    try:
        dates = _parse_dates(frame.index)
    except ValueError as exc:
        raise ValueError(f"{path}: could not parse the first column as dates") from exc
    if dates.isna().any():
        raise ValueError(f"{path}: first column has missing or unparseable dates")
    if dates.has_duplicates:
        raise ValueError(f"{path}: duplicate dates {list(dates[dates.duplicated()][:5])}")
    try:
        frame = frame.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: price columns must be numeric") from exc
    frame.index = dates
    return frame.sort_index()


def prices_to_returns(
    prices: pd.DataFrame, kind: Literal["simple", "log"] = "simple"
) -> pd.DataFrame:
    """Align a price panel and convert it to per-period returns.

    Steps (Campbell, Lo & MacKinlay, 1997, *The Econometrics of Financial Markets*, ch. 1):

    1. sort by date and drop rows where no asset has a price (non-trading days);
       a text index (e.g. ``pd.read_csv(path, index_col=0)`` without ``parse_dates``)
       is parsed to dates first, because text order is not date order
       (``"2020-1-10" < "2020-1-9"``);
    2. forward-fill at most 5 consecutive missing prices inside each asset's history
       (holiday-calendar mismatches across exchanges); leading NaNs are never filled;
    3. drop all rows before the common start (the date by which every asset trades);
    4. simple returns ``R_t = P_t / P_{t-1} - 1`` or log returns
       ``r_t = ln(P_t / P_{t-1}) = ln(1 + R_t)``; the first row (no predecessor) is dropped.

    Args:
        prices: wide table, rows = dates, columns = assets, strictly positive prices.
            The index is a ``DatetimeIndex`` (or Period/Timedelta), numeric periods,
            or date labels that parse as dates.
        kind: ``"simple"`` (the package convention) or ``"log"``.

    Returns:
        Returns DataFrame indexed by the dates of the *later* price of each pair (a
        parsed tz-naive ``DatetimeIndex`` when the input index held date labels).

    Raises:
        ValueError: on an unknown ``kind``, empty or non-numeric input, an index that
            is neither dates nor numbers, missing (NaT/NaN) index labels on any index
            type, duplicate dates, non-positive or infinite
            prices, an asset with no prices, a gap longer than 5 prices (NaN would
            remain), or fewer than two aligned rows.
    """
    if kind not in ("simple", "log"):
        raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    if not isinstance(prices, pd.DataFrame):
        raise ValueError(f"prices must be a pandas DataFrame, got {type(prices).__name__}")
    if prices.shape[0] == 0 or prices.shape[1] == 0:
        raise ValueError(f"prices must be a non-empty 2-D table, got shape {prices.shape}")
    index = prices.index
    if not (
        isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex))
        or pd.api.types.is_numeric_dtype(index)
    ):
        # Sorting labels as text would pair non-adjacent days (silently wrong returns).
        try:
            dates = _parse_dates(index)
        except ValueError as exc:
            raise ValueError(
                "prices index must hold dates (a DatetimeIndex or date labels) or numeric "
                "periods; could not parse it as dates"
            ) from exc
        prices = prices.set_axis(dates, axis=0)
    # On every index type (the parsed text above, Datetime/Period/Timedelta, numeric):
    # sort_index puts a NaT/NaN label last, so it would be paired with the last real
    # date and yield a fabricated return at an unknown date. A typical source is
    # pd.to_datetime(..., errors="coerce") on a messy date column. Checked before
    # duplicates, since two NaT labels are missing dates rather than a repeated one.
    if prices.index.hasnans:
        raise ValueError("prices index has missing or unparseable dates (NaT/NaN labels)")
    if prices.index.has_duplicates:
        raise ValueError("prices index has duplicate dates")
    try:
        frame = prices.astype(float).sort_index()
    except (ValueError, TypeError) as exc:
        raise ValueError("prices must be numeric") from exc

    values = frame.to_numpy()
    if np.isinf(values).any():
        raise ValueError("prices contain infinite values")
    observed = ~np.isnan(values)
    if (values[observed] <= 0).any():
        raise ValueError("prices must be strictly positive")
    empty = [
        c for c, has_data in zip(frame.columns, observed.any(axis=0), strict=True) if not has_data
    ]
    if empty:
        raise ValueError(f"assets with no prices at all: {empty}")

    frame = frame.dropna(how="all")
    # Fill before trimming so an asset missing a price on the common start date (a
    # holiday) is filled from its own earlier history instead of being left NaN.
    filled = frame.ffill(limit=_MAX_FILL_GAP)
    first_valid = (~np.isnan(frame.to_numpy())).argmax(axis=0)
    filled = filled.iloc[int(first_valid.max()) :]

    gaps = filled.isna()
    if gaps.to_numpy().any():
        asset = gaps.any(axis=0).idxmax()
        date = gaps[asset].idxmax()
        raise ValueError(
            f"asset {asset!r} has more than {_MAX_FILL_GAP} consecutive missing prices "
            f"(still NaN at {date} after forward-filling); clean or drop it"
        )
    if filled.shape[0] < 2:
        raise ValueError("need at least two aligned price rows to compute returns")

    p = filled.to_numpy()
    ratio = p[1:] / p[:-1]
    out = ratio - 1.0 if kind == "simple" else np.log(ratio)
    return pd.DataFrame(out, index=filled.index[1:], columns=filled.columns)


# --------------------------------------------------------------------------------------
# Synthetic market with known truth
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticMarket:
    """A simulated market whose true moments are known exactly.

    Attributes:
        prices: price paths starting at 100 on the business day *before* the first
            return date, so ``prices_to_returns(prices)`` reproduces ``returns``.
        returns: daily simple returns, ``DatetimeIndex = pd.bdate_range(start, periods=n)``.
        true_mu: ANNUALIZED true expected simple returns (daily mean * 252).
        true_cov: ANNUALIZED true covariance (daily covariance * 252).
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    true_mu: pd.Series
    true_cov: pd.DataFrame


def _check_tail_df(tail_df: float | None) -> float | None:
    """Return ``None`` for Gaussian shocks, else a validated finite ``df > 2``."""
    if tail_df is None:
        return None
    tail_df = float(tail_df)
    if math.isnan(tail_df) or tail_df <= 2.0:
        raise ValueError(f"tail_df must exceed 2 for a finite variance, got {tail_df}")
    return None if math.isinf(tail_df) else tail_df


def _unit_variance_shocks(
    rng: np.random.Generator, size: tuple[int, int], tail_df: float | None
) -> np.ndarray:
    """I.i.d. shocks with mean 0 and variance 1: N(0, 1), or t_nu * sqrt((nu - 2) / nu).

    ``Var(t_nu) = nu / (nu - 2)``, so the rescaling makes the variance exactly one while
    keeping the fat tails (excess kurtosis ``6 / (nu - 4)`` for ``nu > 4``).
    """
    if tail_df is None:
        return rng.standard_normal(size)
    return rng.standard_t(tail_df, size) * math.sqrt((tail_df - 2.0) / tail_df)


def _check_positive_int(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _check_dates_fit(start: str, n_obs: int) -> None:
    """Raise unless the ``n_obs + 1`` business-day price dates fit pandas' ns range.

    pandas 2 indexes dates at nanosecond resolution only (1677-09-22 .. 2262-04-11) and
    fails inside ``bdate_range`` with an internal ``OutOfBoundsDatetime`` beyond it,
    while pandas 3 goes further. Checking up front with numpy's business-day arithmetic
    (Mon-Fri, as ``bdate_range``; no range limit) gives the same behaviour and a clear
    message on both.
    """
    first = np.busday_offset(np.datetime64(pd.Timestamp(start).date(), "D"), 0, roll="forward")
    before = np.busday_offset(first, -1)  # the price row preceding the first return
    try:
        last = np.busday_offset(first, n_obs - 1)
    except (OverflowError, ValueError):
        last = None
    if before < _FIRST_NS_DAY or last is None or last > _LAST_NS_DAY:
        raise ValueError(
            f"{n_obs} daily returns starting {first} need price dates from {before} to "
            f"{'an unrepresentable date' if last is None else last}, outside {_FIRST_NS_DAY} "
            f"to {_LAST_NS_DAY} (what pandas can index at nanosecond resolution); use "
            "another start or fewer years, or simulate_returns (no dates) for longer "
            "Monte Carlo samples"
        )


def synthetic_market(
    n_assets: int = 10,
    n_years: float = 15.0,
    n_factors: int = 3,
    seed: int = 0,
    tail_df: float | None = 5.0,
    start: str = "2005-01-03",
) -> SyntheticMarket:
    """Simulate daily returns from a linear factor model with known true moments.

    Model (Sharpe, 1963, "A simplified model for portfolio analysis"; Ross, 1976, "The
    arbitrage theory of capital asset pricing"), for each business day ``t``::

        r_t = m + B f_t + e_t,   f_t = diag(s_f) z_t,   e_t = diag(s_e) u_t,

    with ``z_t``, ``u_t`` i.i.d. unit-variance shocks — Student-t with ``tail_df``
    degrees of freedom rescaled by ``sqrt((nu - 2) / nu)`` (fat tails, Praetz, 1972,
    "The distribution of share price changes"), or Gaussian if ``tail_df`` is None. The
    shocks are mutually independent, so the true moments are exact::

        E[r_t]   = m = true_mu / 252
        Cov(r_t) = B diag(s_f^2) B' + diag(s_e^2) = true_cov / 252

    Calibration (equity/bond-like magnitudes; drawn from ``np.random.default_rng(seed)``
    *before* any shock, so the truth does not depend on ``n_years`` or ``tail_df``):

    * factor 1 is the market: annual vol 16%, loadings strictly positive;
    * factors 2..K are "sector" factors (annual vol 6%-12%): asset ``i`` loads mainly on
      sector ``i mod (K-1)`` (interleaved, so clusters are not contiguous columns), with
      small loadings on the other sectors;
    * each asset's total annual vol is drawn from U(8%, 30%) and its systematic share
      R^2 from U(0.30, 0.75); loadings are rescaled to hit both exactly. Pairwise
      correlations then equal ``R_i R_j cos(angle_ij)`` and fall mostly in 0.1-0.8;
    * annual expected returns ``mu_i = 1% + 0.3 vol_i + N(0, 1%)``, clipped to [2%, 12%]
      (riskier assets earn more on average, Sharpe ratios ~0.2-0.6).

    Args:
        n_assets: number of assets, named ``"A01"``, ``"A02"``, ...
        n_years: sample length; ``n = round(n_years * 252)`` daily returns. The dates
            must stay within 1677-09-22 .. 2262-04-11 (pandas' nanosecond range), i.e.
            at most about 257 years from the default start; use
            :func:`simulate_returns` for longer undated samples.
        n_factors: number of factors ``K >= 1`` (the first is the market).
        seed: seed for ``np.random.default_rng``; equal seeds give identical markets.
        tail_df: Student-t degrees of freedom (> 2) or ``None`` for Gaussian shocks.
            Below about 4 some seeds raise (see Raises); the default 5 has not.
        start: first return date (rolled forward to a business day if needed).

    Returns:
        :class:`SyntheticMarket` with prices, returns, ``true_mu`` and ``true_cov``.

    Raises:
        ValueError: on invalid arguments, if the price dates would leave pandas'
            nanosecond date range (the same on pandas 2 and 3), or if a simulated daily
            return is <= -100%, since prices would then turn non-positive. That happens
            with heavy tails: for the default 15-year, 10-asset, 3-factor market,
            about 4% of seeds raise at ``tail_df=2.5``, 1% at ``tail_df=3`` (2% for 50
            years: longer samples see more extreme shocks) and 0.2% at ``tail_df=3.5``;
            none of 1,000 seeds raised at ``tail_df`` 4 or 5.
    """
    n_assets = _check_positive_int(n_assets, "n_assets")
    n_factors = _check_positive_int(n_factors, "n_factors")
    if not (isinstance(n_years, (int, float, np.number)) and math.isfinite(n_years)):
        raise ValueError(f"n_years must be a finite number, got {n_years!r}")
    n_obs = round(float(n_years) * TRADING_DAYS)
    if n_obs < 2:
        raise ValueError(f"n_years={n_years} gives {n_obs} daily returns; need at least 2")
    tail_df = _check_tail_df(tail_df)
    _check_dates_fit(start, n_obs)
    # One bdate_range call (it is slow for long samples): prices start one business day
    # before the first return date, so price_dates[1:] == pd.bdate_range(start, periods=n).
    first_day = pd.offsets.BDay().rollforward(pd.Timestamp(start).normalize())
    price_dates = pd.bdate_range(first_day - pd.offsets.BDay(1), periods=n_obs + 1)
    rng = np.random.default_rng(seed)

    # ---- true parameters (annualized) --------------------------------------------
    k_sectors = n_factors - 1
    factor_vol = np.concatenate(([0.16], rng.uniform(0.06, 0.12, k_sectors)))
    loadings = np.empty((n_assets, n_factors))
    loadings[:, 0] = rng.uniform(0.5, 1.5, n_assets)
    if k_sectors:
        loadings[:, 1:] = rng.normal(0.0, 0.15, (n_assets, k_sectors))
        sector = np.arange(n_assets) % k_sectors
        loadings[np.arange(n_assets), 1 + sector] = rng.uniform(0.6, 1.2, n_assets)
    total_vol = rng.uniform(0.08, 0.30, n_assets)
    r_squared = rng.uniform(0.30, 0.75, n_assets)
    raw_systematic = (loadings**2) @ (factor_vol**2)
    # Rescale each asset's loadings so its systematic variance is exactly R^2 vol^2.
    betas = loadings * np.sqrt(r_squared * total_vol**2 / raw_systematic)[:, None]
    idio_var = (1.0 - r_squared) * total_vol**2
    mu = np.clip(0.01 + 0.3 * total_vol + rng.normal(0.0, 0.01, n_assets), 0.02, 0.12)

    cov = (betas * factor_vol**2) @ betas.T + np.diag(idio_var)
    cov = 0.5 * (cov + cov.T)

    # ---- simulate daily returns ------------------------------------------------------
    sqrt_days = math.sqrt(TRADING_DAYS)
    factors = _unit_variance_shocks(rng, (n_obs, n_factors), tail_df) * (factor_vol / sqrt_days)
    idio = _unit_variance_shocks(rng, (n_obs, n_assets), tail_df) * (np.sqrt(idio_var) / sqrt_days)
    daily = mu / TRADING_DAYS + factors @ betas.T + idio
    if (daily <= -1.0).any():
        raise ValueError(
            "a simulated daily return is <= -100%, which would make prices non-positive; "
            "use a larger tail_df (or None) or a different seed"
        )

    assets = [f"A{i + 1:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(daily, index=price_dates[1:], columns=assets)
    growth = np.vstack([np.ones((1, n_assets)), np.cumprod(1.0 + daily, axis=0)])
    prices = pd.DataFrame(100.0 * growth, index=price_dates, columns=assets)
    return SyntheticMarket(
        prices=prices,
        returns=returns,
        true_mu=pd.Series(mu, index=assets),
        true_cov=pd.DataFrame(cov, index=assets, columns=assets),
    )


def _psd_factor(cov: np.ndarray) -> np.ndarray:
    """Matrix ``A`` with ``A A' = cov``: Cholesky, or a symmetric eigen-factor if singular.

    Raises:
        ValueError: if ``cov`` has a materially negative eigenvalue (not PSD).
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        pass
    eigval, eigvec = np.linalg.eigh(cov)
    # Purely relative tolerance: semi-definiteness does not depend on units, and daily
    # covariances are ~1e-4 or smaller, so an absolute floor would wave through clearly
    # indefinite small-scale matrices. (A zero matrix gives tol = 0 and min eig = 0.)
    tol = 1e-10 * float(np.abs(eigval).max())
    if eigval.min() < -tol:
        raise ValueError(
            f"covariance must be positive semi-definite (min eigenvalue {eigval.min():.3g})"
        )
    return eigvec * np.sqrt(np.clip(eigval, 0.0, None))


def simulate_returns(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_obs: int,
    rng: np.random.Generator,
    tail_df: float | None = None,
) -> pd.DataFrame:
    """Draw i.i.d. daily returns with given ANNUALIZED mean and covariance.

    Gaussian: ``r_t = mu/252 + A z_t`` with ``z_t ~ N(0, I)`` and ``A A' = cov/252``
    (Cholesky; a symmetric eigen-factor when ``cov`` is only semi-definite).

    Student-t (``tail_df = nu > 2``): the multivariate t stochastic representation
    (Kotz & Nadarajah, 2004, *Multivariate t Distributions and Their Applications*)::

        r_t = mu/252 + sqrt((nu - 2) / nu) * A z_t / sqrt(W_t / nu),   W_t ~ chi^2_nu,

    with one ``W_t`` per row (joint crashes / tail dependence). Since
    ``E[nu / W_t] = nu / (nu - 2)``, the covariance is exactly ``cov / 252``; marginals
    have excess kurtosis ``6 / (nu - 4)`` for ``nu > 4``.

    Args:
        mu: annualized expected simple returns (Series aligned to ``cov`` by label, or
            array in ``cov`` order).
        cov: annualized covariance, symmetric positive semi-definite.
        n_obs: number of daily observations (>= 1).
        rng: numpy random Generator (consumed; pass a seeded one for reproducibility).
        tail_df: Student-t degrees of freedom (> 2) or ``None`` for Gaussian.

    Returns:
        DataFrame ``(n_obs, N)`` with a RangeIndex and the asset labels of ``cov``.

    Raises:
        ValueError: on invalid shapes/values, ``n_obs < 1``, ``tail_df <= 2``, or a
            covariance that is not positive semi-definite.
        TypeError: if ``rng`` is not a ``numpy.random.Generator``.
    """
    cov_frame = as_cov_frame(cov)
    assets = list(cov_frame.index)
    mu_series = as_series(mu, assets)
    n_obs = _check_positive_int(n_obs, "n_obs")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    tail_df = _check_tail_df(tail_df)

    factor = _psd_factor(cov_frame.to_numpy() / TRADING_DAYS)
    shocks = rng.standard_normal((n_obs, len(assets))) @ factor.T
    if tail_df is not None:
        mixing = rng.chisquare(tail_df, n_obs) / tail_df
        shocks *= math.sqrt((tail_df - 2.0) / tail_df) / np.sqrt(mixing)[:, None]
    daily = mu_series.to_numpy() / TRADING_DAYS + shocks
    return pd.DataFrame(daily, columns=assets)
