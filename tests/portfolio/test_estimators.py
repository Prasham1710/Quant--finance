"""Tests for quantfin.portfolio.estimators.

The math is checked against independent answers: hand-computed values,
numpy/pandas reference computations, scikit-learn's Ledoit-Wolf, a pure-Python
loop transcription of the Ledoit-Wolf (2004, "Honey") formulas, and the
closed-form population shrinkage constant for Gaussian data.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import ledoit_wolf as sk_ledoit_wolf

from quantfin.portfolio._common import TRADING_DAYS
from quantfin.portfolio.estimators import (
    cov_ewma,
    cov_ledoit_wolf,
    cov_sample,
    cov_to_corr,
    is_positive_definite,
    ledoit_wolf_shrinkage,
    mean_ewma,
    mean_historical,
)

# A daily "true" covariance with heterogeneous volatilities and NON-constant
# correlations, so that neither shrinkage target is correct.
TRUE_CORR = np.array(
    [
        [1.0, 0.7, 0.2, 0.1, 0.0],
        [0.7, 1.0, 0.3, 0.2, 0.1],
        [0.2, 0.3, 1.0, 0.5, 0.4],
        [0.1, 0.2, 0.5, 1.0, 0.6],
        [0.0, 0.1, 0.4, 0.6, 1.0],
    ]
)
TRUE_VOL = np.array([0.010, 0.020, 0.015, 0.008, 0.030])
TRUE_COV = TRUE_CORR * np.outer(TRUE_VOL, TRUE_VOL)
ASSETS = ["SPY", "TLT", "GLD", "EEM", "HYG"]

# Hand-computable sample: column means are both 0.01.
#   A deviations (0, .02, -.02): var = .0008/2 = .0004
#   B deviations (.01, -.02, .01): var = .0006/2 = .0003
#   cov = (0*.01 + .02*-.02 + -.02*.01)/2 = -.0003
TINY = pd.DataFrame(
    {"A": [0.01, 0.03, -0.01], "B": [0.02, -0.01, 0.02]},
    index=pd.date_range("2024-01-01", periods=3, freq="B"),
)


def gaussian_returns(
    n_obs: int, cov: np.ndarray = TRUE_COV, seed: int = 0, mean: float = 0.0005
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(cov)
    values = mean + rng.standard_normal((n_obs, cov.shape[0])) @ chol.T
    labels = ASSETS if cov.shape[0] == len(ASSETS) else [f"X{i}" for i in range(cov.shape[0])]
    return pd.DataFrame(values, columns=labels, index=pd.RangeIndex(n_obs))


def random_returns(n_obs: int, n_assets: int, seed: int, heavy_tails: bool = False) -> np.ndarray:
    """Correlated returns with unequal volatilities (one common factor)."""
    rng = np.random.default_rng(seed)
    if heavy_tails:
        shocks = rng.standard_t(3, size=(n_obs, n_assets + 1))
    else:
        shocks = rng.standard_normal((n_obs, n_assets + 1))
    betas = rng.uniform(0.2, 1.2, size=n_assets)
    vols = rng.uniform(0.005, 0.03, size=n_assets)
    common = shocks[:, :1] * betas
    return 0.0003 + vols * (common + shocks[:, 1:])


def mle_cov(values: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(np.asarray(values), rowvar=False, ddof=0))


# ---------------------------------------------------------------------------
# Independent reference: Ledoit & Wolf (2004, JPM "Honey") Appendix B formulas,
# written as explicit sums over i, j and t in pure Python. It deliberately uses
# the paper's symmetric form of rho (both theta_ii,ij and theta_jj,ij terms)
# instead of covCor.m's vectorized algebra.
# ---------------------------------------------------------------------------


def honey_reference(data: np.ndarray) -> tuple[np.ndarray, float]:
    """Constant-correlation target and clipped intensity delta = max(0, min(1, kappa / T))."""
    f, kappa, n_obs = honey_reference_kappa(data)
    return f, max(0.0, min(1.0, kappa / n_obs))


def honey_reference_kappa(data: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Constant-correlation target, UNCLIPPED kappa = (pi - rho) / gamma, and T."""
    rows = np.asarray(data, dtype=float).tolist()
    T, N = len(rows), len(rows[0])
    mean = [sum(rows[t][i] for t in range(T)) / T for i in range(N)]
    x = [[rows[t][i] - mean[i] for i in range(N)] for t in range(T)]
    s = [[sum(x[t][i] * x[t][j] for t in range(T)) / T for j in range(N)] for i in range(N)]
    sd = [math.sqrt(s[i][i]) for i in range(N)]
    rbar = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                rbar += s[i][j] / (sd[i] * sd[j])
    rbar /= N * (N - 1)
    f = [[s[i][i] if i == j else rbar * sd[i] * sd[j] for j in range(N)] for i in range(N)]

    def pi_hat(i: int, j: int) -> float:
        return sum((x[t][i] * x[t][j] - s[i][j]) ** 2 for t in range(T)) / T

    def theta_hat(k: int, i: int, j: int) -> float:
        # asymptotic covariance of sqrt(T) s_kk and sqrt(T) s_ij
        return sum((x[t][k] ** 2 - s[k][k]) * (x[t][i] * x[t][j] - s[i][j]) for t in range(T)) / T

    pi = sum(pi_hat(i, j) for i in range(N) for j in range(N))
    rho = sum(pi_hat(i, i) for i in range(N))
    for i in range(N):
        for j in range(N):
            if i != j:
                rho += (rbar / 2.0) * (
                    math.sqrt(s[j][j] / s[i][i]) * theta_hat(i, i, j)
                    + math.sqrt(s[i][i] / s[j][j]) * theta_hat(j, i, j)
                )
    gamma = sum((f[i][j] - s[i][j]) ** 2 for i in range(N) for j in range(N))
    return np.array(f), (pi - rho) / gamma, T


def common_factor_heavy_tail_returns(seed: int) -> np.ndarray:
    """T <= N returns with a strong common factor and t(3) noise (3 <= T <= N < 40).

    Strong correlation with heavy tails is the regime in which the finite-sample
    rho-hat of the constant-correlation estimator can exceed pi-hat (kappa < 0).
    """
    rng = np.random.default_rng(seed)
    n_assets = int(rng.integers(3, 40))
    n_obs = int(rng.integers(3, n_assets + 1))
    factor = rng.standard_normal((n_obs, 1)) * rng.uniform(0.5, 1.5, n_assets) * 3
    noise = rng.standard_t(3, (n_obs, n_assets))
    return (factor + noise) * rng.uniform(0.001, 0.05, n_assets)


# Seeds of common_factor_heavy_tail_returns whose estimated kappa is negative.
NEGATIVE_KAPPA_SEEDS = (313, 1672)


def gaussian_population_kappa(cov: np.ndarray, target: str) -> float:
    """Limit of T * delta for i.i.d. Gaussian data (Isserlis' theorem).

    For Gaussian x: AsyVar(sqrt(T) s_ij) = s_ii s_jj + s_ij^2 and
    AsyCov(sqrt(T) s_ii, sqrt(T) s_ij) = 2 s_ii s_ij.
    """
    n = cov.shape[0]
    pi = np.trace(cov) ** 2 + np.sum(cov**2)
    if target == "identity":
        m = np.trace(cov) / n
        return float(pi / np.sum((cov - m * np.eye(n)) ** 2))
    vol = np.sqrt(np.diag(cov))
    off = ~np.eye(n, dtype=bool)
    rbar = (cov / np.outer(vol, vol))[off].mean()
    phi = rbar * np.outer(vol, vol)
    np.fill_diagonal(phi, vol**2)
    # rho = sum_i pi_ii
    #       + sum_{i != j} (rbar/2)(sqrt(s_jj/s_ii) 2 s_ii s_ij + sqrt(s_ii/s_jj) 2 s_jj s_ij)
    rho = 2.0 * np.sum(np.diag(cov) ** 2) + 2.0 * rbar * np.sum((cov * np.outer(vol, vol))[off])
    return float((pi - rho) / np.sum((phi - cov) ** 2))


# ---------------------------------------------------------------------------
# mean_historical
# ---------------------------------------------------------------------------


class TestMeanHistorical:
    def test_hand_computed_arithmetic_and_compounding(self):
        mu = mean_historical(TINY)
        assert list(mu.index) == ["A", "B"]
        np.testing.assert_allclose(mu.to_numpy(), [0.01 * 252, 0.01 * 252], rtol=1e-12)
        geo = mean_historical(TINY, compounding=True)
        expected_a = (1.01 * 1.03 * 0.99) ** (252 / 3) - 1
        expected_b = (1.02 * 0.99 * 1.02) ** (252 / 3) - 1
        np.testing.assert_allclose(geo.to_numpy(), [expected_a, expected_b], rtol=1e-12)

    def test_annualization_is_linear_in_periods_per_year(self):
        data = random_returns(100, 4, seed=1)
        monthly = mean_historical(data, periods_per_year=12)
        np.testing.assert_allclose(monthly.to_numpy(), 12 * data.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(
            mean_historical(data).to_numpy(), TRADING_DAYS * data.mean(axis=0), rtol=1e-12
        )

    def test_compounding_with_ppy_equal_to_sample_length_is_total_return(self):
        data = random_returns(37, 3, seed=2)
        total = np.prod(1 + data, axis=0) - 1
        geo = mean_historical(data, periods_per_year=37, compounding=True)
        np.testing.assert_allclose(geo.to_numpy(), total, rtol=1e-12)

    def test_geometric_mean_never_exceeds_arithmetic(self):
        # AM-GM: (prod (1+r))^(1/T) <= 1 + mean(r), so the per-period geometric rate is lower.
        data = random_returns(500, 6, seed=3, heavy_tails=True)
        geo = mean_historical(data, compounding=True).to_numpy()
        per_period_geo = (1 + geo) ** (1 / TRADING_DAYS) - 1
        assert np.all(per_period_geo <= data.mean(axis=0) + 1e-15)

    def test_wiped_out_asset_compounds_to_minus_one(self):
        data = np.array([[0.1, 0.02], [-1.0, 0.01], [0.5, 0.03]])
        geo = mean_historical(data, compounding=True)
        assert geo.iloc[0] == -1.0
        assert list(geo.index) == [0, 1]

    def test_numpy_and_series_inputs(self):
        series = pd.Series([0.01, 0.02, 0.03], name="X")
        mu = mean_historical(series, periods_per_year=1)
        assert list(mu.index) == ["X"]
        assert mu.iloc[0] == pytest.approx(0.02)

    @pytest.mark.parametrize(
        "kwargs",
        [{"periods_per_year": 0}, {"periods_per_year": -12}, {"periods_per_year": float("nan")}],
    )
    def test_invalid_periods_per_year(self, kwargs):
        with pytest.raises(ValueError, match="periods_per_year"):
            mean_historical(TINY, **kwargs)

    def test_invalid_returns(self):
        with pytest.raises(ValueError, match="NaN"):
            mean_historical(np.array([[0.01, np.nan], [0.02, 0.01]]))
        with pytest.raises(ValueError, match="-100%"):
            mean_historical(np.array([[0.01], [-1.5]]), compounding=True)
        with pytest.raises(ValueError):
            mean_historical(np.empty((0, 3)))


# ---------------------------------------------------------------------------
# mean_ewma
# ---------------------------------------------------------------------------


class TestMeanEwma:
    def test_matches_explicit_weights(self):
        data = random_returns(40, 3, seed=4)
        span = 10
        alpha = 2.0 / (span + 1)
        ages = np.arange(39, -1, -1)
        w = (1 - alpha) ** ages
        expected = 252 * (w @ data) / w.sum()
        np.testing.assert_allclose(mean_ewma(data, span=span).to_numpy(), expected, rtol=1e-12)

    def test_span_one_is_last_observation(self):
        out = mean_ewma(TINY, span=1, periods_per_year=12)
        np.testing.assert_allclose(out.to_numpy(), 12 * TINY.iloc[-1].to_numpy(), rtol=1e-12)
        assert list(out.index) == ["A", "B"]

    def test_infinite_span_is_arithmetic_mean(self):
        data = random_returns(60, 4, seed=5)
        np.testing.assert_allclose(
            mean_ewma(data, span=10**12).to_numpy(),
            mean_historical(data).to_numpy(),
            rtol=1e-8,
        )

    def test_recent_observations_dominate(self):
        # +1%/day for 100 days then -1%/day for 20 days: the EWMA must see the recent regime.
        data = np.r_[np.full(100, 0.01), np.full(20, -0.01)][:, None]
        assert mean_ewma(data, span=5).iloc[0] < -0.009 * 252
        assert mean_historical(data).iloc[0] > 0

    @pytest.mark.parametrize("span", [0, 0.5, -3, float("nan")])
    def test_invalid_span(self, span):
        with pytest.raises(ValueError, match="span"):
            mean_ewma(TINY, span=span)


# ---------------------------------------------------------------------------
# Row order for the order-dependent (exponentially weighted) estimators
# ---------------------------------------------------------------------------


def regime_switch_returns(index: pd.Index) -> pd.DataFrame:
    """Old high-vol regime with positive drift, then a recent low-vol regime with negative drift."""
    rng = np.random.default_rng(9)
    n_old = len(index) * 2 // 3
    n_new = len(index) - n_old
    values = np.r_[
        0.002 + 0.03 * rng.standard_normal(n_old), -0.003 + 0.005 * rng.standard_normal(n_new)
    ]
    return pd.DataFrame({"A": values, "B": 0.5 * values + 0.001}, index=index)


EWMA_ESTIMATORS = [
    pytest.param(lambda r: mean_ewma(r, span=5), id="mean_ewma"),
    pytest.param(lambda r: cov_ewma(r, halflife=10), id="cov_ewma"),
]
DATE_INDEXES = [
    pytest.param(pd.bdate_range("2020-01-01", periods=600), id="datetime"),
    pytest.param(pd.period_range("2020-01-01", periods=600, freq="D"), id="period"),
]


class TestEwmaRowOrder:
    @pytest.mark.parametrize("estimator", EWMA_ESTIMATORS)
    @pytest.mark.parametrize("index", DATE_INDEXES)
    def test_newest_first_dates_are_rejected(self, estimator, index):
        # Vendor data often arrives newest-first. Weighting it as-is would give the oldest,
        # high-vol regime the largest weight (an 8x wrong daily vol) without any error.
        ascending = regime_switch_returns(index)
        with pytest.raises(ValueError, match="ascending date order"):
            estimator(ascending.iloc[::-1])
        shuffled = ascending.sample(frac=1.0, random_state=0)
        with pytest.raises(ValueError, match="ascending date order"):
            estimator(shuffled)

    @pytest.mark.parametrize("estimator", EWMA_ESTIMATORS)
    @pytest.mark.parametrize("index", DATE_INDEXES)
    def test_sorting_restores_the_ascending_estimate(self, estimator, index):
        ascending = regime_switch_returns(index)
        expected = estimator(ascending)
        restored = estimator(ascending.iloc[::-1].sort_index())
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(restored, expected, rtol=1e-14)
        else:
            pd.testing.assert_series_equal(restored, expected, rtol=1e-14)
        # The date index only labels rows; the estimate equals the positional numpy one.
        np.testing.assert_allclose(
            np.asarray(estimator(ascending.to_numpy())), np.asarray(expected), rtol=1e-14
        )

    def test_ascending_dates_see_the_recent_regime(self):
        data = regime_switch_returns(pd.bdate_range("2020-01-01", periods=600))
        daily_vol = math.sqrt(cov_ewma(data, halflife=10).loc["A", "A"] / 252)
        assert 0.003 < daily_vol < 0.008
        assert mean_ewma(data, span=5)["A"] < 0

    @pytest.mark.parametrize("estimator", EWMA_ESTIMATORS)
    def test_non_date_index_is_taken_positionally(self, estimator):
        # Integer labels carry no time information, so rows are used in the given order
        # (last row = newest), exactly as for numpy input.
        reversed_rows = regime_switch_returns(pd.RangeIndex(600)).iloc[::-1]
        assert not reversed_rows.index.is_monotonic_increasing
        np.testing.assert_allclose(
            np.asarray(estimator(reversed_rows)),
            np.asarray(estimator(reversed_rows.to_numpy())),
            rtol=1e-14,
        )


# ---------------------------------------------------------------------------
# cov_sample
# ---------------------------------------------------------------------------


class TestCovSample:
    def test_hand_computed_ddof_one(self):
        cov = cov_sample(TINY)
        expected = 252 * np.array([[0.0004, -0.0003], [-0.0003, 0.0003]])
        np.testing.assert_allclose(cov.to_numpy(), expected, rtol=1e-12)
        assert list(cov.index) == ["A", "B"] and list(cov.columns) == ["A", "B"]

    def test_matches_numpy_and_pandas(self):
        data = gaussian_returns(30)
        cov = cov_sample(data, periods_per_year=12)
        np.testing.assert_allclose(cov.to_numpy(), 12 * np.cov(data.to_numpy().T), rtol=1e-12)
        np.testing.assert_allclose(cov.to_numpy(), 12 * data.cov().to_numpy(), rtol=1e-12)
        np.testing.assert_array_equal(cov.to_numpy(), cov.to_numpy().T)

    def test_shift_and_permutation_invariance(self):
        data = gaussian_returns(50)
        base = cov_sample(data)
        shifted = cov_sample(data + np.arange(5) * 0.01)
        np.testing.assert_allclose(shifted.to_numpy(), base.to_numpy(), rtol=1e-9)
        order = ["HYG", "SPY", "EEM", "TLT", "GLD"]
        permuted = cov_sample(data[order])
        pd.testing.assert_frame_equal(permuted, base.loc[order, order], rtol=1e-12)

    def test_numpy_input_gets_positional_labels(self):
        cov = cov_sample(np.asarray(TINY))
        assert list(cov.index) == [0, 1]

    def test_validation(self):
        with pytest.raises(ValueError, match="at least 2"):
            cov_sample(TINY.iloc[:1])
        with pytest.raises(ValueError, match="periods_per_year"):
            cov_sample(TINY, periods_per_year=0)
        with pytest.raises(ValueError, match="NaN"):
            cov_sample(np.array([[0.01, np.inf], [0.02, 0.01]]))


# ---------------------------------------------------------------------------
# cov_ewma
# ---------------------------------------------------------------------------


class TestCovEwma:
    def test_matches_pandas_ewm_cov(self):
        data = gaussian_returns(120, seed=6)
        data.index = pd.bdate_range("2020-01-01", periods=120)
        for halflife in (1.0, 7.5, 63.0):
            ours = cov_ewma(data, halflife=halflife)
            ref = data.ewm(halflife=halflife).cov().loc[data.index[-1]].loc[ASSETS, ASSETS]
            np.testing.assert_allclose(ours.to_numpy(), 252 * ref.to_numpy(), rtol=1e-10)

    def test_matches_explicit_weighted_formula(self):
        data = random_returns(25, 3, seed=7)
        halflife = 4.0
        v = 0.5 ** ((24 - np.arange(25)) / halflife)
        w = v / v.sum()
        centered = data - w @ data
        expected = 12 * (centered.T * w) @ centered / (1 - np.sum(w**2))
        ours = cov_ewma(data, halflife=halflife, periods_per_year=12)
        np.testing.assert_allclose(ours.to_numpy(), expected, rtol=1e-12)

    def test_infinite_halflife_is_sample_covariance(self):
        data = gaussian_returns(40, seed=8)
        np.testing.assert_allclose(
            cov_ewma(data, halflife=1e12).to_numpy(), cov_sample(data).to_numpy(), rtol=1e-8
        )

    def test_recent_regime_dominates(self):
        # High vol (3%) long ago, low vol (0.5%) recently. Reversed weights would see 3%.
        rng = np.random.default_rng(9)
        data = np.r_[0.03 * rng.standard_normal(400), 0.005 * rng.standard_normal(200)][:, None]
        vol = math.sqrt(cov_ewma(data, halflife=10).iloc[0, 0] / 252)
        assert 0.003 < vol < 0.008
        assert math.sqrt(cov_sample(data).iloc[0, 0] / 252) > 0.02

    def test_symmetric_psd_even_with_more_assets_than_observations(self):
        data = random_returns(8, 15, seed=10)
        cov = cov_ewma(data, halflife=3).to_numpy()
        np.testing.assert_array_equal(cov, cov.T)
        eig = np.linalg.eigvalsh(cov)
        assert eig.min() > -1e-12 * eig.max()

    @pytest.mark.parametrize("halflife", [0, -1, float("inf"), float("nan")])
    def test_invalid_halflife(self, halflife):
        with pytest.raises(ValueError, match="halflife"):
            cov_ewma(TINY, halflife=halflife)

    def test_halflife_too_short_for_sample(self):
        with pytest.raises(ValueError, match="too short"):
            cov_ewma(random_returns(10, 2, seed=11), halflife=1e-3)

    def test_needs_two_observations(self):
        with pytest.raises(ValueError, match="at least 2"):
            cov_ewma(np.ones((1, 3)))


# ---------------------------------------------------------------------------
# Ledoit-Wolf, identity target
# ---------------------------------------------------------------------------


class TestLedoitWolfIdentity:
    @pytest.mark.parametrize(
        ("n_obs", "n_assets", "seed"),
        [(250, 5, 0), (60, 12, 1), (10, 20, 2), (3, 2, 3), (500, 1, 4), (40, 40, 5)],
    )
    def test_matches_sklearn(self, n_obs, n_assets, seed):
        data = random_returns(n_obs, n_assets, seed=seed)
        shrunk, target, delta = ledoit_wolf_shrinkage(data, target="identity")
        sk_cov, sk_delta = sk_ledoit_wolf(data)
        assert delta == pytest.approx(sk_delta, abs=1e-10)
        scale = np.abs(sk_cov).max()
        np.testing.assert_allclose(shrunk, sk_cov, rtol=1e-10, atol=1e-12 * scale)
        np.testing.assert_allclose(target, np.trace(mle_cov(data)) / n_assets * np.eye(n_assets))

    def test_sklearn_cases_exercise_interior_shrinkage(self):
        # Guard against the comparison above passing only on clipped values.
        deltas = [
            ledoit_wolf_shrinkage(random_returns(t, n, seed=s), target="identity")[2]
            for t, n, s in [(250, 5, 0), (60, 12, 1), (10, 20, 2)]
        ]
        assert all(0.0 < d < 1.0 for d in deltas)

    def test_population_limit_for_gaussian_data(self):
        data = gaussian_returns(200_000, seed=12)
        _, _, delta = ledoit_wolf_shrinkage(data, target="identity")
        kappa = gaussian_population_kappa(TRUE_COV, "identity")
        assert 200_000 * delta == pytest.approx(kappa, rel=0.03)

    def test_spherical_truth_is_shrunk_fully(self):
        # If the truth is a multiple of I the target is correct, and b^2 >= d^2 asymptotically.
        rng = np.random.default_rng(13)
        _, _, delta = ledoit_wolf_shrinkage(0.01 * rng.standard_normal((500, 8)), "identity")
        assert delta == 1.0

    def test_two_observations_give_zero_intensity(self):
        # Demeaned rows satisfy x_2 = -x_1, so x_t x_t' = S for both t and b_bar^2 = 0:
        # nothing is shrunk and the rank-1 sample covariance comes back unchanged.
        data = random_returns(2, 5, seed=30)
        shrunk, _, delta = ledoit_wolf_shrinkage(data, target="identity")
        sk_cov, sk_delta = sk_ledoit_wolf(data)
        assert delta == pytest.approx(0.0, abs=1e-12)
        assert sk_delta == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(shrunk, mle_cov(data), rtol=1e-10, atol=1e-16)
        np.testing.assert_allclose(shrunk, sk_cov, rtol=1e-10, atol=1e-16)
        assert not is_positive_definite(shrunk)


# ---------------------------------------------------------------------------
# Ledoit-Wolf, constant-correlation target (covCor.m)
# ---------------------------------------------------------------------------


class TestLedoitWolfConstantCorrelation:
    @pytest.mark.parametrize(("n_obs", "seed"), [(40, 0), (60, 1), (200, 2)])
    def test_matches_loop_reference(self, n_obs, seed):
        data = gaussian_returns(n_obs, seed=seed).to_numpy()
        shrunk, target, delta = ledoit_wolf_shrinkage(data)
        ref_target, ref_delta = honey_reference(data)
        assert 0.0 < ref_delta < 1.0  # interior: the formula, not the clip, is being compared
        assert delta == pytest.approx(ref_delta, rel=1e-10, abs=1e-14)
        np.testing.assert_allclose(target, ref_target, rtol=1e-10, atol=1e-18)
        sample = mle_cov(data)
        np.testing.assert_allclose(
            shrunk, ref_delta * ref_target + (1 - ref_delta) * sample, rtol=1e-10, atol=1e-18
        )

    def test_matches_loop_reference_heavy_tails_unequal_scales(self):
        data = random_returns(50, 6, seed=21, heavy_tails=True)
        _, target, delta = ledoit_wolf_shrinkage(data)
        ref_target, ref_delta = honey_reference(data)
        assert delta == pytest.approx(ref_delta, rel=1e-10, abs=1e-14)
        np.testing.assert_allclose(target, ref_target, rtol=1e-10, atol=1e-18)

    def test_matches_loop_reference_when_clipped_at_one(self):
        data = random_returns(30, 4, seed=14)
        _, target, delta = ledoit_wolf_shrinkage(data)
        ref_target, ref_delta = honey_reference(data)
        assert delta == ref_delta == 1.0
        np.testing.assert_allclose(target, ref_target, rtol=1e-10, atol=1e-18)

    def test_population_limit_for_gaussian_data(self):
        # T * delta -> kappa = (pi - rho) / gamma, computable in closed form for Gaussians.
        # A transposed theta (s_jj instead of s_ii) or a sign flip in rho moves kappa a lot.
        data = gaussian_returns(200_000, seed=15)
        _, _, delta = ledoit_wolf_shrinkage(data)
        kappa = gaussian_population_kappa(TRUE_COV, "constant_correlation")
        assert 200_000 * delta == pytest.approx(kappa, rel=0.03)

    def test_delta_in_unit_interval_on_many_shapes(self):
        deltas = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            n_assets = int(rng.integers(2, 15))
            n_obs = int(rng.integers(3, 300))
            data = random_returns(n_obs, n_assets, seed=seed, heavy_tails=bool(seed % 2))
            _, _, delta = ledoit_wolf_shrinkage(data)
            assert 0.0 <= delta <= 1.0
            deltas.append(delta)
        assert any(0.0 < d < 1.0 for d in deltas)

    def test_delta_decreases_to_zero_as_sample_grows(self):
        sizes = [50, 200, 1000, 5000, 25000]
        mean_deltas = [
            np.mean([ledoit_wolf_shrinkage(gaussian_returns(t, seed=s))[2] for s in range(5)])
            for t in sizes
        ]
        assert all(a > b for a, b in itertools.pairwise(mean_deltas))
        assert mean_deltas[-1] < 0.005

    def test_positive_definite_when_fewer_observations_than_assets(self):
        # Shrinkage repairs a singular S only when delta > 0 (see the negative-kappa test).
        data = random_returns(12, 30, seed=16)
        shrunk, target, delta = ledoit_wolf_shrinkage(data)
        sample = mle_cov(data)
        assert delta > 0
        assert not is_positive_definite(sample)
        assert np.linalg.matrix_rank(sample) <= 11
        assert is_positive_definite(target)
        assert is_positive_definite(shrunk)
        assert np.linalg.eigvalsh(shrunk).min() > 0

    def test_positive_definite_whenever_intensity_is_positive_with_t_le_n(self):
        # Weyl: lambda_min(delta F + (1 - delta) S) >= delta lambda_min(F) because S is PSD,
        # so the result is PD exactly when delta > 0 (F is PD). delta = 0 returns S itself.
        positive, zero = 0, 0
        for seed in [*range(200), *NEGATIVE_KAPPA_SEEDS]:
            data = common_factor_heavy_tail_returns(seed)
            shrunk, target, delta = ledoit_wolf_shrinkage(data)
            assert is_positive_definite(target)
            if delta > 0:
                positive += 1
                eig_min = np.linalg.eigvalsh(shrunk).min()
                floor = delta * np.linalg.eigvalsh(target).min()
                assert eig_min >= floor - 1e-12 * np.abs(shrunk).max()
                assert is_positive_definite(shrunk)
            else:
                zero += 1
                np.testing.assert_allclose(shrunk, mle_cov(data), rtol=1e-10, atol=1e-18)
                assert not is_positive_definite(shrunk)
        assert positive >= 190 and zero >= len(NEGATIVE_KAPPA_SEEDS)

    @pytest.mark.parametrize("seed", NEGATIVE_KAPPA_SEEDS)
    def test_negative_kappa_gives_zero_intensity_and_singular_result_when_t_le_n(self, seed):
        # Documented edge case of the faithful covCor.m transcription: the finite-sample
        # rho-hat exceeds pi-hat, kappa < 0 and delta is clipped to 0, so with T <= N the
        # "shrunk" matrix is the singular sample covariance. Callers must check PD.
        data = common_factor_heavy_tail_returns(seed)
        n_obs, n_assets = data.shape
        assert n_obs <= n_assets
        _, ref_kappa, _ = honey_reference_kappa(data)
        assert ref_kappa < 0
        shrunk, target, delta = ledoit_wolf_shrinkage(data)
        assert delta == 0.0
        np.testing.assert_allclose(shrunk, mle_cov(data), rtol=1e-10, atol=1e-18)
        assert is_positive_definite(target)  # the target itself is fine; it just gets no weight
        cov = cov_ledoit_wolf(data)
        assert not is_positive_definite(cov)
        assert np.linalg.matrix_rank(cov.to_numpy()) <= n_obs - 1

    def test_correct_target_is_shrunk_toward_heavily(self):
        # Constant-correlation analogue of the spherical identity test. The truth HAS constant
        # correlation (0.4) with unequal vols, so F is unbiased and far less noisy than S, and
        # delta should be near 1. ||S - F||^2 is then only ~1e-4 of ||S||^2. That is exactly
        # where an over-eager "S already equals F" tolerance would wrongly force delta = 0.
        n_assets = 8
        vol = np.linspace(0.005, 0.03, n_assets)
        corr = np.full((n_assets, n_assets), 0.4)
        np.fill_diagonal(corr, 1.0)
        truth = corr * np.outer(vol, vol)
        deltas, gap_ratios, err_shrunk, err_sample = [], [], [], []
        for seed in range(10):
            data = gaussian_returns(5000, cov=truth, seed=300 + seed).to_numpy()
            shrunk, target, delta = ledoit_wolf_shrinkage(data)
            sample = mle_cov(data)
            deltas.append(delta)
            gap_ratios.append(np.sum((sample - target) ** 2) / np.sum(sample**2))
            err_shrunk.append(np.sum((shrunk - truth) ** 2))
            err_sample.append(np.sum((sample - truth) ** 2))
        assert max(gap_ratios) < 1e-3  # the regime the test is about
        assert np.mean(deltas) > 0.9
        # F keeps the noisy sample variances, so only the off-diagonal error is removed;
        # across seed sets the total Frobenius error drops to ~0.65 of the sample's.
        assert np.mean(err_shrunk) < 0.8 * np.mean(err_sample)

    def test_target_has_unit_diagonal_and_constant_average_correlation(self):
        data = gaussian_returns(300, seed=17).to_numpy()
        _, target, _ = ledoit_wolf_shrinkage(data)
        sample_corr = np.corrcoef(data, rowvar=False)  # independent rbar (ddof cancels)
        off = ~np.eye(5, dtype=bool)
        rbar = sample_corr[off].mean()
        vol = np.sqrt(np.diag(target))
        target_corr = target / np.outer(vol, vol)
        np.testing.assert_allclose(np.diag(target_corr), 1.0, rtol=1e-14)
        np.testing.assert_allclose(target_corr[off], rbar, rtol=1e-12)
        # The target keeps the ddof=0 sample variances; with ddof=1 this would be off by T/(T-1).
        np.testing.assert_allclose(np.diag(target), data.var(axis=0, ddof=0), rtol=1e-12)

    def test_shrunk_is_convex_combination_with_mle_sample(self):
        data = gaussian_returns(80, seed=18).to_numpy()
        shrunk, target, delta = ledoit_wolf_shrinkage(data)
        sample = mle_cov(data)
        np.testing.assert_allclose(shrunk, delta * target + (1 - delta) * sample, rtol=1e-12)
        np.testing.assert_allclose(np.diag(shrunk), data.var(axis=0, ddof=0), rtol=1e-12)
        np.testing.assert_array_equal(shrunk, shrunk.T)

    def test_invariances(self):
        data = gaussian_returns(150, seed=19).to_numpy()
        shrunk, target, delta = ledoit_wolf_shrinkage(data)
        # common rescaling: matrices scale by c^2, intensity unchanged
        s2, t2, d2 = ledoit_wolf_shrinkage(3.0 * data)
        assert d2 == pytest.approx(delta, rel=1e-10)
        np.testing.assert_allclose(s2, 9.0 * shrunk, rtol=1e-10)
        np.testing.assert_allclose(t2, 9.0 * target, rtol=1e-10)
        # shifting each asset's returns by a constant changes nothing (data are demeaned)
        s3, _, d3 = ledoit_wolf_shrinkage(data + np.array([0.01, -0.02, 0.0, 0.05, 0.001]))
        assert d3 == pytest.approx(delta, rel=1e-8)
        np.testing.assert_allclose(s3, shrunk, rtol=1e-8)
        # relabelling assets permutes the estimate
        perm = [3, 0, 4, 1, 2]
        s4, t4, d4 = ledoit_wolf_shrinkage(data[:, perm])
        assert d4 == pytest.approx(delta, rel=1e-12)
        np.testing.assert_allclose(s4, shrunk[np.ix_(perm, perm)], rtol=1e-12)
        np.testing.assert_allclose(t4, target[np.ix_(perm, perm)], rtol=1e-12)

    def test_two_assets_target_equals_sample(self):
        # With N = 2 the average correlation IS the correlation, so F == S and delta = 0.
        data = random_returns(30, 2, seed=20)
        shrunk, target, delta = ledoit_wolf_shrinkage(data)
        assert delta == 0.0
        np.testing.assert_allclose(target, mle_cov(data), rtol=1e-12)
        np.testing.assert_allclose(shrunk, mle_cov(data), rtol=1e-12)

    def test_single_asset(self):
        data = np.array([[0.01], [0.03], [-0.02], [0.00]])
        shrunk, target, delta = ledoit_wolf_shrinkage(data)
        assert delta == 0.0
        np.testing.assert_allclose(shrunk, [[data.var(ddof=0)]], rtol=1e-12)
        np.testing.assert_allclose(target, shrunk, rtol=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError, match="target"):
            ledoit_wolf_shrinkage(TINY, target="diagonal")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least 2"):
            ledoit_wolf_shrinkage(TINY.iloc[:1])
        with pytest.raises(ValueError, match="NaN"):
            ledoit_wolf_shrinkage(np.array([[0.01, np.nan], [0.02, 0.01]]))
        constant = np.c_[random_returns(20, 2, seed=22), np.full(20, 0.001)]
        with pytest.raises(ValueError, match="constant returns"):
            ledoit_wolf_shrinkage(constant)
        # the identity target does not need correlations, so a cash-like column is fine
        shrunk, _, _ = ledoit_wolf_shrinkage(constant, target="identity")
        np.testing.assert_allclose(shrunk, sk_ledoit_wolf(constant)[0], rtol=1e-10, atol=1e-16)


# ---------------------------------------------------------------------------
# cov_ledoit_wolf
# ---------------------------------------------------------------------------


class TestCovLedoitWolf:
    def test_is_annualized_labelled_shrinkage(self):
        data = gaussian_returns(100, seed=23)
        for target in ("constant_correlation", "identity"):
            daily, _, _ = ledoit_wolf_shrinkage(data, target=target)
            cov = cov_ledoit_wolf(data, target=target)
            assert list(cov.index) == ASSETS and list(cov.columns) == ASSETS
            np.testing.assert_allclose(cov.to_numpy(), 252 * daily, rtol=1e-12)
            monthly = cov_ledoit_wolf(data, target=target, periods_per_year=12)
            np.testing.assert_allclose(monthly.to_numpy(), 12 * daily, rtol=1e-12)

    def test_identity_matches_sklearn_annualized(self):
        data = gaussian_returns(70, seed=24)
        sk_cov, _ = sk_ledoit_wolf(data.to_numpy())
        np.testing.assert_allclose(
            cov_ledoit_wolf(data, target="identity").to_numpy(), 252 * sk_cov, rtol=1e-10
        )

    def test_closer_to_truth_than_sample_in_small_samples(self):
        # The whole point of shrinkage: lower Frobenius error than S when T is small.
        errors_sample, errors_lw = [], []
        annual_truth = 252 * TRUE_COV
        for seed in range(30):
            data = gaussian_returns(30, seed=100 + seed)
            errors_sample.append(np.sum((cov_sample(data).to_numpy() - annual_truth) ** 2))
            errors_lw.append(np.sum((cov_ledoit_wolf(data).to_numpy() - annual_truth) ** 2))
        assert np.mean(errors_lw) < np.mean(errors_sample)

    def test_validation(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            cov_ledoit_wolf(TINY, periods_per_year=-1)
        with pytest.raises(ValueError, match="target"):
            cov_ledoit_wolf(TINY, target="nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# cov_to_corr
# ---------------------------------------------------------------------------


class TestCovToCorr:
    def test_hand_computed(self):
        cov = np.array([[0.04, 0.006], [0.006, 0.09]])
        corr, vols = cov_to_corr(cov)
        np.testing.assert_allclose(vols.to_numpy(), [0.2, 0.3], rtol=1e-14)
        np.testing.assert_allclose(corr.to_numpy(), [[1.0, 0.1], [0.1, 1.0]], rtol=1e-14)
        assert list(corr.index) == [0, 1] and list(vols.index) == [0, 1]

    def test_round_trip_and_agreement_with_corrcoef(self):
        data = gaussian_returns(90, seed=25)
        cov = cov_sample(data)
        corr, vols = cov_to_corr(cov)
        assert list(corr.columns) == ASSETS and list(vols.index) == ASSETS
        np.testing.assert_allclose(corr.to_numpy(), np.corrcoef(data.to_numpy().T), rtol=1e-12)
        # annualized vols scale with sqrt(ppy), not ppy
        np.testing.assert_allclose(
            vols.to_numpy(), data.std(ddof=1).to_numpy() * math.sqrt(252), rtol=1e-12
        )
        rebuilt = np.outer(vols, vols) * corr.to_numpy()
        np.testing.assert_allclose(rebuilt, cov.to_numpy(), rtol=1e-12)
        np.testing.assert_array_equal(np.diag(corr.to_numpy()), 1.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="positive"):
            cov_to_corr(np.array([[0.0, 0.0], [0.0, 1.0]]))
        with pytest.raises(ValueError, match="positive"):
            cov_to_corr(np.array([[-1.0, 0.0], [0.0, 1.0]]))
        with pytest.raises(ValueError, match="correlation"):
            cov_to_corr(np.array([[1.0, 2.0], [2.0, 1.0]]))
        with pytest.raises(ValueError, match="symmetric"):
            cov_to_corr(np.array([[1.0, 0.5], [0.1, 1.0]]))
        with pytest.raises(ValueError, match="square"):
            cov_to_corr(np.ones((2, 3)))

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_correlation_tolerance_covers_rounding_only(self, sign):
        vols = np.array([0.2, 0.3])

        def cov_with_corr(rho: float) -> np.ndarray:
            return np.array([[1.0, rho], [rho, 1.0]]) * np.outer(vols, vols)

        # A correlation 1e-6 beyond +-1 is a broken matrix, not rounding.
        with pytest.raises(ValueError, match="correlation"):
            cov_to_corr(cov_with_corr(sign * (1.0 + 1e-6)))
        # 1e-12 beyond +-1 is rounding dust: accepted and clipped onto the boundary.
        corr, out_vols = cov_to_corr(cov_with_corr(sign * (1.0 + 1e-12)))
        np.testing.assert_array_equal(corr.to_numpy(), [[1.0, sign], [sign, 1.0]])
        np.testing.assert_allclose(out_vols.to_numpy(), vols, rtol=1e-14)


# ---------------------------------------------------------------------------
# is_positive_definite
# ---------------------------------------------------------------------------


class TestIsPositiveDefinite:
    @pytest.mark.parametrize(
        ("matrix", "expected"),
        [
            (np.eye(4), True),
            (np.array([[2.0, 0.5], [0.5, 1.0]]), True),
            (1e-200 * np.eye(3), True),
            (np.diag([1e-8, 1e8]), True),  # badly scaled but PD
            (np.array([[1.0, 2.0], [2.0, 1.0]]), False),  # eigenvalues 3, -1
            (np.array([[1.0, 1.0], [1.0, 1.0]]), False),  # singular PSD
            (np.zeros((3, 3)), False),
            (-np.eye(2), False),
        ],
    )
    def test_known_matrices(self, matrix, expected):
        assert is_positive_definite(matrix) is expected

    def test_dataframe_input_and_true_covariances(self):
        data = gaussian_returns(100, seed=26)
        assert is_positive_definite(cov_sample(data))
        assert is_positive_definite(252 * TRUE_COV)

    def test_ill_conditioned_but_definite(self):
        rng = np.random.default_rng(27)
        q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
        matrix = (q * np.logspace(0, -8, 10)) @ q.T
        assert is_positive_definite(0.5 * (matrix + matrix.T))

    def test_rank_deficient_sample_covariances_are_rejected(self):
        # LAPACK's Cholesky sometimes "succeeds" on singular matrices through rounding;
        # the pivot tolerance must reject them all.
        lucky_factorizations = 0
        for seed in range(300):
            rng = np.random.default_rng(seed)
            n_assets = int(rng.integers(3, 30))
            n_obs = int(rng.integers(2, n_assets + 1))
            sample = np.cov(random_returns(n_obs, n_assets, seed=seed), rowvar=False)
            try:
                np.linalg.cholesky(sample)
                lucky_factorizations += 1
            except np.linalg.LinAlgError:
                pass
            assert not is_positive_definite(sample)
        assert lucky_factorizations > 0

    def test_validation(self):
        with pytest.raises(ValueError, match="square"):
            is_positive_definite(np.ones((2, 3)))
        with pytest.raises(ValueError, match="NaN"):
            is_positive_definite(np.array([[1.0, np.nan], [np.nan, 1.0]]))
        with pytest.raises(ValueError, match="symmetric"):
            is_positive_definite(np.array([[1.0, 0.0], [0.5, 1.0]]))
