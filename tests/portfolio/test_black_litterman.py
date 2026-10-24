"""Tests for quantfin.portfolio.black_litterman.

The math is checked against independent answers rather than against itself:

* hand-computed numbers (risk aversion, implied returns, He-Litterman omega);
* reverse-optimization closed forms written inline (unconstrained and
  budget-constrained utility maximization, the tangency portfolio);
* the precision ("information") form of the posterior;
* Theil-Goldberger mixed estimation, solved as a whitened least-squares problem;
* the KKT system of the certain-view problem (min distance to the prior subject
  to P mu = Q);
* Idzorek's original one-dimensional search for omega (scipy brentq), against
  Walters' closed form;
* Gaussian identities and invariances: two identical views equal one
  precision-weighted view, posterior mean free of tau when Omega is
  proportional to tau, view rescaling / sign flips / reordering, asset
  permutations, and block independence.

Sign, transpose, ddof and annualization mistakes are each caught by an exact
test (see the hand-computed risk aversion, the midpoint and beta-revision tests,
and the block-independence test, which uses K != N).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag, solve_triangular
from scipy.optimize import brentq

from quantfin.portfolio._common import TRADING_DAYS
from quantfin.portfolio.black_litterman import (
    BlackLittermanResult,
    black_litterman,
    build_views,
    he_litterman_omega,
    idzorek_omega,
    implied_equilibrium_returns,
    market_implied_risk_aversion,
)

ASSETS = ["US", "EU", "EM", "BOND", "GOLD"]
VOLS = np.array([0.16, 0.18, 0.24, 0.06, 0.15])
# GOLD is negatively correlated with BOND so that a bullish GOLD view must LOWER
# the posterior BOND return (a sign test), and BOND/EM are uncorrelated.
CORR = np.array(
    [
        [1.00, 0.80, 0.70, 0.10, 0.05],
        [0.80, 1.00, 0.75, 0.05, 0.10],
        [0.70, 0.75, 1.00, 0.00, 0.15],
        [0.10, 0.05, 0.00, 1.00, -0.20],
        [0.05, 0.10, 0.15, -0.20, 1.00],
    ]
)
SIGMA = CORR * np.outer(VOLS, VOLS)
COV = pd.DataFrame(SIGMA, index=ASSETS, columns=ASSETS)
W_MKT = pd.Series([0.35, 0.25, 0.10, 0.25, 0.05], index=ASSETS)
DELTA = 2.5
TAU = 0.05
PI = pd.Series(DELTA * SIGMA @ W_MKT.to_numpy(), index=ASSETS)


def _random_cov(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n + 4))
    gram = a @ a.T
    d = np.sqrt(np.diag(gram))
    vols = rng.uniform(0.05, 0.30, n)
    return gram / np.outer(d, d) * np.outer(vols, vols)


def _random_problem(seed: int, n: int = 7, k: int = 3, full_omega: bool = False):
    rng = np.random.default_rng(seed)
    sigma = _random_cov(rng, n)
    pi = rng.normal(0.04, 0.02, n)
    p = rng.normal(size=(k, n))
    q = rng.normal(0.03, 0.05, k)
    if full_omega:
        b = rng.normal(size=(k, k))
        omega = 0.01 * (b @ b.T / k + 0.1 * np.eye(k))
    else:
        omega = np.diag(rng.uniform(0.001, 0.02, k))
    return sigma, pi, p, q, omega


def _exact_moments_series(mean: float, var: float, n: int = 500, seed: int = 1) -> np.ndarray:
    """A sample whose mean and ddof=1 variance are exactly ``mean`` and ``var``."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + np.sqrt(var) * z


def _min_eig(matrix: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[0])


def test_fixture_covariance_is_positive_definite():
    assert _min_eig(SIGMA) > 1e-4


# --------------------------------------------------------------------------
# market_implied_risk_aversion
# --------------------------------------------------------------------------


def test_risk_aversion_hand_computed_uses_ddof1_and_annualizes_rf_correctly():
    r = [0.01, -0.02, 0.015, 0.005]
    # mean = 0.0025; squared deviations sum to 7.25e-4; ddof=1 variance = 7.25e-4 / 3.
    # delta = (252 * 0.0025 - 0.02) / (252 * 7.25e-4 / 3) = 0.61 / 0.0609.
    assert market_implied_risk_aversion(r, rf=0.02) == pytest.approx(0.61 / 0.0609, rel=1e-12)
    # The wrong ddof (=0) or a de-annualized rf would give clearly different numbers.
    assert abs(market_implied_risk_aversion(r, rf=0.02) - 0.61 / (252 * 7.25e-4 / 4)) > 1.0
    assert abs(market_implied_risk_aversion(r, rf=0.02) - (0.0025 - 0.02) / (7.25e-4 / 3)) > 1.0


@pytest.mark.parametrize("ppy", [TRADING_DAYS, 52, 12])
@pytest.mark.parametrize("rf", [0.0, 0.03])
def test_risk_aversion_matches_formula_on_exact_moments(ppy, rf):
    mean, var = 0.0006, 1.1e-4
    r = _exact_moments_series(mean, var)
    expected = (ppy * mean - rf) / (ppy * var)
    assert market_implied_risk_aversion(r, rf=rf, periods_per_year=ppy) == pytest.approx(
        expected, rel=1e-10
    )


def test_risk_aversion_annualization_cancels_only_when_rf_is_zero():
    r = _exact_moments_series(0.0004, 1e-4)
    assert market_implied_risk_aversion(r, periods_per_year=252) == pytest.approx(
        market_implied_risk_aversion(r, periods_per_year=12), rel=1e-12
    )
    assert market_implied_risk_aversion(r, rf=0.02, periods_per_year=252) != pytest.approx(
        market_implied_risk_aversion(r, rf=0.02, periods_per_year=12), rel=1e-3
    )


def test_risk_aversion_round_trips_with_reverse_optimization():
    # If pi = delta Sigma w, the market's annual excess return is w'pi = delta w'Sigma w,
    # so a market return series with those (annualized) moments must give back delta.
    rf = 0.015
    w = W_MKT.to_numpy()
    excess = float(w @ PI.to_numpy())
    var = float(w @ SIGMA @ w)
    r = _exact_moments_series((excess + rf) / TRADING_DAYS, var / TRADING_DAYS, n=1000)
    assert market_implied_risk_aversion(r, rf=rf) == pytest.approx(DELTA, rel=1e-10)


def test_risk_aversion_accepts_array_series_and_single_column_inputs():
    r = _exact_moments_series(0.0005, 2e-4, n=50)
    expected = market_implied_risk_aversion(r, rf=0.01)
    idx = pd.bdate_range("2024-01-01", periods=50)
    for variant in (
        list(r),
        pd.Series(r, index=idx),
        pd.DataFrame({"mkt": r}, index=idx),
        r.reshape(-1, 1),
    ):
        assert market_implied_risk_aversion(variant, rf=0.01) == pytest.approx(expected, rel=1e-14)


def test_risk_aversion_is_negative_when_market_underperforms_cash():
    r = _exact_moments_series(0.0, 1e-4)
    delta = market_implied_risk_aversion(r, rf=0.05)
    assert delta < 0
    with pytest.raises(ValueError, match="risk_aversion"):
        implied_equilibrium_returns(COV, W_MKT, risk_aversion=delta)


@pytest.mark.parametrize(
    ("returns", "kwargs", "match"),
    [
        ([0.01], {}, "two observations"),
        ([], {}, "two observations"),
        ([0.01, np.nan, 0.02], {}, "NaN"),
        ([0.01, np.inf, 0.02], {}, "NaN"),
        ([0.01] * 5, {}, "zero variance"),
        # Constants whose np.mean is not exact: the computed variance is ~1e-35, not 0,
        # and without a rounding-aware test delta came out ~1e34.
        ([0.001] * 10, {}, "zero variance"),
        ([0.1] * 3, {}, "zero variance"),
        ([0.0004] * TRADING_DAYS, {}, "zero variance"),
        # A deterministic constant-growth price path: returns equal up to rounding.
        (pd.Series(100 * 1.0004 ** np.arange(253)).pct_change().dropna(), {}, "zero variance"),
        (np.ones((4, 2)), {}, "1-D"),
        (pd.DataFrame({"a": [0.01, 0.02], "b": [0.0, 0.01]}), {}, "single"),
        ([0.01, 0.02, -0.01], {"periods_per_year": 0}, "periods_per_year"),
        ([0.01, 0.02, -0.01], {"periods_per_year": -12}, "periods_per_year"),
        ([0.01, 0.02, -0.01], {"rf": np.nan}, "rf"),
    ],
)
def test_risk_aversion_validation(returns, kwargs, match):
    with pytest.raises(ValueError, match=match):
        market_implied_risk_aversion(returns, **kwargs)


@pytest.mark.parametrize("vol", [1e-4, 1e-6, 1e-9])
def test_risk_aversion_accepts_genuinely_low_volatility_series(vol):
    # A money-market-like series is tiny but not constant; it must not be mistaken for
    # rounding noise by the constant-series guard.
    mean = 1e-4
    r = _exact_moments_series(mean, vol**2, n=TRADING_DAYS)
    assert market_implied_risk_aversion(r) == pytest.approx(mean / vol**2, rel=1e-6)


# --------------------------------------------------------------------------
# implied_equilibrium_returns
# --------------------------------------------------------------------------


def test_implied_returns_hand_computed():
    cov = pd.DataFrame([[0.04, 0.006], [0.006, 0.01]], index=["A", "B"], columns=["A", "B"])
    # Sigma w = (0.04*0.6 + 0.006*0.4, 0.006*0.6 + 0.01*0.4) = (0.0264, 0.0076); times 3.
    pi = implied_equilibrium_returns(cov, [0.6, 0.4], risk_aversion=3.0)
    assert list(pi.index) == ["A", "B"]
    np.testing.assert_allclose(pi.to_numpy(), [0.0792, 0.0228], rtol=1e-13)


@pytest.mark.parametrize("delta", [1.0, 2.5, 7.0])
@pytest.mark.parametrize("rf", [0.0, 0.02])
def test_reverse_optimization_round_trip(delta, rf):
    pi = implied_equilibrium_returns(COV, W_MKT, risk_aversion=delta).to_numpy()
    w = W_MKT.to_numpy()
    ones = np.ones(len(w))
    sigma_inv = np.linalg.inv(SIGMA)

    # 1) Unconstrained max  w'pi - delta/2 w'Sigma w  ->  w = (delta Sigma)^-1 pi.
    np.testing.assert_allclose(np.linalg.solve(delta * SIGMA, pi), w, atol=1e-12)

    # 2) Fully invested max  w'mu - delta/2 w'Sigma w  s.t. 1'w = 1 with TOTAL returns
    #    mu = pi + rf:  w = Sigma^-1 (mu - lam 1) / delta,
    #    lam = (1'Sigma^-1 mu - delta) / (1'Sigma^-1 1).  The multiplier must equal rf.
    mu = pi + rf
    lam = (ones @ sigma_inv @ mu - delta) / (ones @ sigma_inv @ ones)
    np.testing.assert_allclose(sigma_inv @ (mu - lam * ones) / delta, w, atol=1e-12)
    assert lam == pytest.approx(rf, abs=1e-12)

    # 3) Tangency portfolio  Sigma^-1 (mu - rf) / 1'Sigma^-1 (mu - rf)  is the market.
    tangency = sigma_inv @ (mu - rf)
    np.testing.assert_allclose(tangency / tangency.sum(), w, atol=1e-12)

    # 4) The market attains the maximal Sharpe ratio sqrt(pi' Sigma^-1 pi), and
    #    delta = market excess return / market variance.
    market_sharpe = (w @ pi) / np.sqrt(w @ SIGMA @ w)
    assert market_sharpe == pytest.approx(np.sqrt(pi @ sigma_inv @ pi), rel=1e-12)
    assert (w @ pi) / (w @ SIGMA @ w) == pytest.approx(delta, rel=1e-12)


def test_implied_returns_aligns_labels_and_labels_numpy_input():
    shuffled = W_MKT.iloc[[4, 2, 0, 3, 1]]
    pd.testing.assert_series_equal(
        implied_equilibrium_returns(COV, shuffled, risk_aversion=DELTA), PI
    )
    pi_np = implied_equilibrium_returns(SIGMA, W_MKT.to_numpy(), risk_aversion=DELTA)
    assert list(pi_np.index) == list(range(len(ASSETS)))
    np.testing.assert_allclose(pi_np.to_numpy(), PI.to_numpy(), rtol=1e-14)


def test_implied_returns_with_annualized_cov_are_annual():
    # pi is linear in Sigma, so daily cov * 252 gives 252 times the daily implied returns.
    daily = implied_equilibrium_returns(SIGMA / TRADING_DAYS, W_MKT.to_numpy())
    annual = implied_equilibrium_returns(SIGMA, W_MKT.to_numpy())
    np.testing.assert_allclose(annual.to_numpy(), TRADING_DAYS * daily.to_numpy(), rtol=1e-13)


@pytest.mark.parametrize(
    ("cov", "weights", "kwargs", "match"),
    [
        (COV, W_MKT * 2, {}, "sum to 1"),
        (COV, W_MKT, {"risk_aversion": 0.0}, "risk_aversion"),
        (COV, W_MKT, {"risk_aversion": -1.0}, "risk_aversion"),
        (COV, W_MKT, {"risk_aversion": np.nan}, "risk_aversion"),
        (COV, W_MKT.iloc[:-1], {}, "missing"),
        (COV, W_MKT.to_numpy()[:-1], {}, "expected 5"),
        (COV, W_MKT.where(W_MKT.index != "EM", np.nan), {}, "NaN"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), [0.5, 0.5], {}, "positive semidefinite"),
        (np.array([[1.0, 0.1], [0.2, 1.0]]), [0.5, 0.5], {}, "symmetric"),
    ],
)
def test_implied_returns_validation(cov, weights, kwargs, match):
    with pytest.raises(ValueError, match=match):
        implied_equilibrium_returns(cov, weights, **kwargs)


# --------------------------------------------------------------------------
# build_views
# --------------------------------------------------------------------------


def test_build_views_rows_values_and_labels():
    P, Q = build_views(
        ASSETS,
        absolute={"EM": 0.07, "BOND": 0.01},
        relative=[("EU", "US", 0.02), ("GOLD", "EM", -0.01)],
    )
    expected = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [-1, 1, 0, 0, 0],
            [0, 0, -1, 0, 1],
        ],
        dtype=float,
    )
    assert list(P.columns) == ASSETS
    assert list(P.index) == ["EM", "BOND", "EU - US", "GOLD - EM"]
    np.testing.assert_array_equal(P.to_numpy(), expected)
    assert list(Q.index) == list(P.index)
    np.testing.assert_array_equal(Q.to_numpy(), [0.07, 0.01, 0.02, -0.01])
    # Sign semantics: the view portfolio of ("EU", "US", x) earns mu_EU - mu_US.
    mu = pd.Series([0.05, 0.08, 0.11, 0.02, 0.04], index=ASSETS)
    view_returns = P @ mu
    assert view_returns["EU - US"] == pytest.approx(0.08 - 0.05)
    assert view_returns["GOLD - EM"] == pytest.approx(0.04 - 0.11)


def test_build_views_keeps_index_and_handles_integer_labels():
    idx = pd.Index(ASSETS, name="ticker")
    P, _ = build_views(idx, absolute=pd.Series({"US": 0.05}))
    assert P.columns.equals(idx)
    P_int, Q_int = build_views(range(3), relative=[(2, 0, 0.01)])
    np.testing.assert_array_equal(P_int.to_numpy(), [[-1.0, 0.0, 1.0]])
    assert list(P_int.index) == ["2 - 0"]
    assert Q_int.iloc[0] == 0.01


@pytest.mark.parametrize(
    ("assets", "kwargs", "match"),
    [
        (ASSETS, {"absolute": {"JP": 0.05}}, "unknown asset"),
        (ASSETS, {"relative": [("US", "JP", 0.01)]}, "unknown asset"),
        (ASSETS, {"relative": [("US", "US", 0.01)]}, "itself"),
        (ASSETS, {}, "no views"),
        (ASSETS, {"absolute": {}, "relative": []}, "no views"),
        (ASSETS, {"absolute": {"US": np.nan}}, "finite"),
        (ASSETS, {"relative": [("US", "EU", np.inf)]}, "finite"),
        (ASSETS, {"absolute": {"US": "high"}}, "real number"),
        (ASSETS, {"relative": [("US", "EU")]}, "tuple"),
        (ASSETS, {"relative": ("US", "EU", 0.01)}, "tuple"),
        (ASSETS, {"absolute": [("US", 0.05)]}, "mapping"),
        (["US", "US"], {"absolute": {"US": 0.05}}, "unique"),
        ([], {"absolute": {"US": 0.05}}, "empty"),
        ("USEU", {"absolute": {"U": 0.05}}, "single string"),
    ],
)
def test_build_views_validation(assets, kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_views(assets, **kwargs)


# --------------------------------------------------------------------------
# he_litterman_omega / idzorek_omega
# --------------------------------------------------------------------------


def test_he_litterman_omega_hand_computed():
    P, _ = build_views(ASSETS, absolute={"EM": 0.07}, relative=[("EU", "US", 0.02)])
    omega = he_litterman_omega(COV, P, tau=TAU)
    # Var(EM) = 0.24^2 = 0.0576;  Var(EU - US) = 0.18^2 + 0.16^2 - 2*0.8*0.16*0.18 = 0.01192.
    np.testing.assert_allclose(omega, np.diag([0.05 * 0.0576, 0.05 * 0.01192]), rtol=1e-12)
    assert omega[0, 1] == 0.0 and omega[1, 0] == 0.0


def test_he_litterman_omega_accepts_vector_and_reorders_dataframe_columns():
    p = np.array([0.0, 1.0, 0.0, 0.0, -1.0])
    omega = he_litterman_omega(SIGMA, p, tau=0.1)
    assert omega.shape == (1, 1)
    assert omega[0, 0] == pytest.approx(0.1 * p @ SIGMA @ p, rel=1e-13)
    P, _ = build_views(ASSETS, relative=[("EU", "GOLD", 0.0), ("EM", "US", 0.0)])
    shuffled = P[["GOLD", "BOND", "US", "EM", "EU"]]
    np.testing.assert_allclose(he_litterman_omega(COV, shuffled), he_litterman_omega(COV, P))


def test_idzorek_omega_formula_and_special_cases():
    P, _ = build_views(ASSETS, absolute={"EM": 0.07, "US": 0.03, "GOLD": 0.02})
    s = TAU * np.diag(P.to_numpy() @ SIGMA @ P.to_numpy().T)
    omega = idzorek_omega(COV, P, [1.0, 0.5, 0.25], tau=TAU)
    np.testing.assert_allclose(np.diag(omega), [0.0, s[1], 3.0 * s[2]], rtol=1e-13)
    assert np.count_nonzero(omega - np.diag(np.diag(omega))) == 0
    # c = 1/2 is exactly He-Litterman.
    np.testing.assert_allclose(
        idzorek_omega(COV, P, [0.5, 0.5, 0.5], tau=TAU), he_litterman_omega(COV, P, tau=TAU)
    )


def _idzorek_search(
    sigma: np.ndarray, pi: np.ndarray, p: np.ndarray, q: float, c: float, tau: float, delta: float
) -> float:
    """Idzorek's (2005) ORIGINAL procedure for one view, written independently.

    Find omega such that the tilt of the unconstrained optimal portfolio
    (delta Sigma)^-1 mu_BL away from w_mkt is the fraction c of the tilt at
    100% confidence (omega = 0). Single-view posterior written out by hand.
    """
    s = tau * p @ sigma @ p
    w_mkt = np.linalg.solve(delta * sigma, pi)

    def weights(omega_k: float) -> np.ndarray:
        mu = pi + tau * sigma @ p * (q - p @ pi) / (s + omega_k)
        return np.linalg.solve(delta * sigma, mu)

    full_tilt = weights(0.0) - w_mkt

    def gap(omega_k: float) -> float:
        return (weights(omega_k) - w_mkt) @ full_tilt / (full_tilt @ full_tilt) - c

    return brentq(gap, 0.0, 1e6 * s, xtol=1e-20, rtol=1e-14, maxiter=500)


@pytest.mark.parametrize("c", [0.1, 0.35, 0.5, 0.8, 0.95])
@pytest.mark.parametrize(
    "view", [{"absolute": {"EM": 0.12}}, {"relative": [("GOLD", "BOND", 0.04)]}]
)
def test_idzorek_closed_form_matches_iterative_search(c, view):
    P, Q = build_views(ASSETS, **view)
    p = P.to_numpy()[0]
    searched = _idzorek_search(SIGMA, PI.to_numpy(), p, Q.iloc[0], c, TAU, DELTA)
    closed = idzorek_omega(COV, P, [c], tau=TAU)[0, 0]
    assert closed == pytest.approx(searched, rel=1e-8)


@pytest.mark.parametrize("c", [0.2, 0.6, 1.0])
def test_idzorek_confidence_is_the_fraction_of_tilt_and_of_view_reached(c):
    P, Q = build_views(ASSETS, relative=[("EM", "US", 0.06)])
    p, q = P.to_numpy()[0], Q.iloc[0]
    w_mkt = W_MKT.to_numpy()

    def unconstrained_weights(mu: pd.Series) -> np.ndarray:
        return np.linalg.solve(DELTA * SIGMA, mu.to_numpy())

    full = black_litterman(COV, PI, P, Q, omega=idzorek_omega(COV, P, [1.0]), tau=TAU)
    part = black_litterman(COV, PI, P, Q, omega=idzorek_omega(COV, P, [c]), tau=TAU)
    np.testing.assert_allclose(
        unconstrained_weights(part.mu) - w_mkt,
        c * (unconstrained_weights(full.mu) - w_mkt),
        atol=1e-12,
    )
    prior_view = p @ PI.to_numpy()
    assert p @ part.mu.to_numpy() == pytest.approx(prior_view + c * (q - prior_view), abs=1e-13)


def test_idzorek_low_confidence_approaches_prior():
    P, Q = build_views(ASSETS, absolute={"EM": 0.15}, relative=[("US", "EU", 0.05)])
    gaps = []
    for c in (1e-2, 1e-4, 1e-6, 1e-8):
        res = black_litterman(COV, PI, P, Q, omega=idzorek_omega(COV, P, [c, c]), tau=TAU)
        gap = np.abs(res.mu - PI).max()
        # The revision shrinks linearly in c (it is at most c times the view surprise).
        assert gap <= c * np.abs(Q.to_numpy() - P.to_numpy() @ PI.to_numpy()).max() * 2
        gaps.append(gap)
        np.testing.assert_allclose(res.cov.to_numpy(), (1 + TAU) * SIGMA, atol=10 * c * TAU)
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-9


@pytest.mark.parametrize(
    ("confidences", "match"),
    [
        ([0.0], r"\(0, 1\]"),
        ([1.5], r"\(0, 1\]"),
        ([-0.1], r"\(0, 1\]"),
        ([np.nan], r"\(0, 1\]"),
        ([0.5, 0.5], "one per view"),
        ([5e-324], "too close to 0"),
    ],
)
def test_idzorek_validation(confidences, match):
    P, _ = build_views(ASSETS, absolute={"EM": 0.07})
    with pytest.raises(ValueError, match=match):
        idzorek_omega(COV, P, confidences)


def test_idzorek_labelled_confidences_are_aligned_to_view_labels():
    # Q is aligned to P's rows by label; labelled confidences must be too, or a
    # reordered Series silently swaps which view gets which confidence.
    P, _ = build_views(
        ASSETS, absolute={"EM": 0.09}, relative=[("GOLD", "BOND", 0.02), ("EU", "US", 0.01)]
    )
    by_position = [0.2, 0.9, 0.6]  # EM, GOLD - BOND, EU - US
    expected = idzorek_omega(COV, P, by_position)
    s = TAU * np.diag(P.to_numpy() @ SIGMA @ P.to_numpy().T)
    np.testing.assert_allclose(np.diag(expected), [4.0 * s[0], s[1] / 9.0, s[2] * 0.4 / 0.6])

    labelled = pd.Series(by_position, index=P.index).iloc[[2, 0, 1]]
    np.testing.assert_array_equal(idzorek_omega(COV, P, labelled), expected)
    as_dict = {"GOLD - BOND": 0.9, "EU - US": 0.6, "EM": 0.2}
    np.testing.assert_array_equal(idzorek_omega(COV, P, as_dict), expected)
    # Unlabelled P: a Series (like Q) and a plain sequence are taken by position.
    np.testing.assert_array_equal(
        idzorek_omega(SIGMA, P.to_numpy(), pd.Series(by_position, index=["x", "y", "z"])),
        expected,
    )


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda P: pd.Series([0.5, 0.5], index=["EM", "GOLD"]), "row labels of P"),
        (lambda P: pd.Series([0.5, 0.5, 0.5], index=[*P.index, "x"]), "row labels of P"),
        (lambda P: {"EM": 0.5}, "row labels of P"),
        (lambda P: {"EM": 0.5, "GOLD - BOND": "high"}, r"numbers in \(0, 1\]"),
        (lambda P: ["high", 0.5], r"numbers in \(0, 1\]"),
        (lambda P: [{"EM": 0.5}, 0.5], r"numbers in \(0, 1\]"),
    ],
)
def test_idzorek_labelled_confidence_validation(make, match):
    P, _ = _views()
    with pytest.raises(ValueError, match=match):
        idzorek_omega(COV, P, make(P))


def test_idzorek_mapping_confidences_need_a_labelled_P():
    P, _ = _views()
    with pytest.raises(ValueError, match="labelled P"):
        idzorek_omega(COV, P.to_numpy(), {"EM": 0.5, "GOLD - BOND": 0.5})


@pytest.mark.parametrize("func", ["he", "idzorek"])
def test_omega_functions_validate_inputs(func):
    def call(cov, P, tau=TAU):
        if func == "he":
            return he_litterman_omega(cov, P, tau=tau)
        return idzorek_omega(cov, P, [0.5] * np.atleast_2d(np.asarray(P)).shape[0], tau=tau)

    P, _ = build_views(ASSETS, absolute={"EM": 0.07})
    with pytest.raises(ValueError, match="tau"):
        call(COV, P, tau=0.0)
    with pytest.raises(ValueError, match="shape"):
        call(SIGMA, np.ones((1, 4)))
    with pytest.raises(ValueError, match="all-zero"):
        call(SIGMA, np.zeros((1, 5)))
    with pytest.raises(ValueError, match="exactly the covariance assets"):
        call(COV, P.rename(columns={"EM": "JP"}))
    with pytest.raises(ValueError, match="positive semidefinite"):
        call(-COV, P)


# --------------------------------------------------------------------------
# black_litterman: closed-form special cases
# --------------------------------------------------------------------------


def test_no_views_returns_the_prior():
    res = black_litterman(COV, PI, tau=TAU)
    pd.testing.assert_series_equal(res.mu, PI)
    pd.testing.assert_series_equal(res.prior, PI)
    pd.testing.assert_frame_equal(res.cov, (1 + TAU) * COV)
    assert res.omega.shape == (0, 0)
    assert res.tau == TAU


@pytest.mark.parametrize("tau", [0.01, 0.05, 1.0])
def test_single_view_he_litterman_gives_midpoint_and_beta_revisions(tau):
    q = PI["GOLD"] + 0.05  # bullish surprise on GOLD
    P, Q = build_views(ASSETS, absolute={"GOLD": q})
    res = black_litterman(COV, PI, P, Q, tau=tau)
    # Omega = tau p Sigma p' equals the prior variance of the view, so the posterior
    # view return is the midpoint of prior and view, whatever tau is.
    assert res.mu["GOLD"] == pytest.approx(0.5 * (PI["GOLD"] + q), abs=1e-14)
    # Every asset is revised by its regression beta on GOLD times GOLD's revision.
    beta = SIGMA[:, 4] / SIGMA[4, 4]
    np.testing.assert_allclose(
        (res.mu - PI).to_numpy(), beta * (res.mu["GOLD"] - PI["GOLD"]), atol=1e-14
    )
    # Direction: positively correlated assets go up, the negatively correlated BOND down.
    assert res.mu["US"] > PI["US"] and res.mu["EM"] > PI["EM"]
    assert res.mu["BOND"] < PI["BOND"]


def test_relative_view_moves_spread_halfway_to_view():
    P, Q = build_views(ASSETS, relative=[("EU", "US", 0.03)])
    res = black_litterman(COV, PI, P, Q, tau=TAU)
    prior_spread = PI["EU"] - PI["US"]
    assert res.mu["EU"] - res.mu["US"] == pytest.approx(0.5 * (prior_spread + 0.03), abs=1e-14)
    assert res.mu["EU"] - res.mu["US"] > prior_spread  # 0.03 is above the prior spread


def test_certain_views_hold_exactly_and_match_kkt_projection():
    P, Q = build_views(ASSETS, absolute={"EM": 0.10}, relative=[("GOLD", "BOND", 0.03)])
    p, q = P.to_numpy(), Q.to_numpy()
    res = black_litterman(COV, PI, P, Q, omega=np.zeros((2, 2)), tau=TAU)
    mu = res.mu.to_numpy()
    np.testing.assert_allclose(p @ mu, q, atol=1e-14)

    # Independent route: min (mu - pi)'(tau Sigma)^-1 (mu - pi) s.t. P mu = Q via its KKT system.
    n, k = len(ASSETS), len(q)
    prec = np.linalg.inv(TAU * SIGMA)
    kkt = np.block([[prec, p.T], [p, np.zeros((k, k))]])
    rhs = np.concatenate([prec @ PI.to_numpy(), q])
    np.testing.assert_allclose(mu, np.linalg.solve(kkt, rhs)[:n], atol=1e-13)

    # No uncertainty is left about the view portfolios: P M = 0.
    m = res.cov.to_numpy() - SIGMA
    np.testing.assert_allclose(p @ m, 0.0, atol=1e-15)
    assert _min_eig(m) > -1e-15


def test_full_confidence_views_on_every_asset_return_the_views():
    views = pd.Series([0.03, 0.05, 0.08, 0.01, 0.02], index=ASSETS)
    P, Q = build_views(ASSETS, absolute=views.to_dict())
    omega = idzorek_omega(COV, P, [1.0] * len(ASSETS), tau=TAU)
    res = black_litterman(COV, PI, P, Q, omega=omega, tau=TAU)
    np.testing.assert_allclose(res.mu.to_numpy(), views.to_numpy(), atol=1e-14)
    np.testing.assert_allclose(res.cov.to_numpy(), SIGMA, atol=1e-15)


def test_infinitely_uncertain_views_return_the_prior():
    P, Q = build_views(ASSETS, absolute={"EM": 0.5}, relative=[("US", "EU", -0.3)])
    res = black_litterman(COV, PI, P, Q, omega=1e12 * np.eye(2), tau=TAU)
    np.testing.assert_allclose(res.mu.to_numpy(), PI.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(res.cov.to_numpy(), (1 + TAU) * SIGMA, atol=1e-14)


def test_two_identical_views_equal_one_precision_weighted_view():
    # Two independent noisy readings of the same portfolio combine like Gaussian measurements:
    # precision adds, and the view value is the precision-weighted average.
    p = np.array([[0.0, 0.5, 0.5, -1.0, 0.0]])
    q1, q2, w1, w2 = 0.04, 0.01, 0.002, 0.006
    two = black_litterman(
        SIGMA, PI.to_numpy(), np.vstack([p, p]), [q1, q2], omega=np.diag([w1, w2]), tau=TAU
    )
    combined_omega = 1.0 / (1.0 / w1 + 1.0 / w2)
    combined_q = combined_omega * (q1 / w1 + q2 / w2)
    one = black_litterman(SIGMA, PI.to_numpy(), p, [combined_q], omega=[[combined_omega]], tau=TAU)
    np.testing.assert_allclose(two.mu.to_numpy(), one.mu.to_numpy(), atol=1e-14)
    np.testing.assert_allclose(two.cov.to_numpy(), one.cov.to_numpy(), atol=1e-15)


# --------------------------------------------------------------------------
# black_litterman: independent reference implementations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("full_omega", [False, True])
@pytest.mark.parametrize("tau", [0.025, 0.5])
def test_matches_precision_form(seed, full_omega, tau):
    sigma, pi, p, q, omega = _random_problem(seed, full_omega=full_omega)
    res = black_litterman(sigma, pi, p, q, omega=omega, tau=tau)
    prior_prec = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(omega)
    post_prec = prior_prec + p.T @ omega_inv @ p
    mu = np.linalg.solve(post_prec, prior_prec @ pi + p.T @ omega_inv @ q)
    np.testing.assert_allclose(res.mu.to_numpy(), mu, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(res.cov.to_numpy(), sigma + np.linalg.inv(post_prec), atol=1e-13)


@pytest.mark.parametrize("seed", range(4))
def test_matches_theil_goldberger_mixed_estimation(seed):
    # GLS on the stacked system [I; P] mu = [pi; Q] + e, Var(e) = blockdiag(tau Sigma, Omega),
    # solved by whitening with a Cholesky factor and least squares.
    sigma, pi, p, q, omega = _random_problem(seed + 10, n=6, k=4, full_omega=True)
    n = len(pi)
    chol = np.linalg.cholesky(block_diag(TAU * sigma, omega))
    x = solve_triangular(chol, np.vstack([np.eye(n), p]), lower=True)
    y = solve_triangular(chol, np.concatenate([pi, q]), lower=True)
    mu_gls, *_ = np.linalg.lstsq(x, y, rcond=None)
    cov_gls = np.linalg.inv(x.T @ x)
    res = black_litterman(sigma, pi, p, q, omega=omega, tau=TAU)
    np.testing.assert_allclose(res.mu.to_numpy(), mu_gls, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.cov.to_numpy() - sigma, cov_gls, atol=1e-13)


# --------------------------------------------------------------------------
# black_litterman: invariances and structure
# --------------------------------------------------------------------------


def test_posterior_mean_is_free_of_tau_when_omega_scales_with_tau():
    P, Q = build_views(ASSETS, absolute={"EM": 0.1, "BOND": 0.0}, relative=[("GOLD", "US", 0.02)])
    ref = black_litterman(COV, PI, P, Q, tau=1.0)
    m_ref = ref.cov.to_numpy() - SIGMA
    for tau in (0.01, 0.05, 0.3):
        he = black_litterman(COV, PI, P, Q, tau=tau)  # default He-Litterman Omega
        np.testing.assert_allclose(he.mu.to_numpy(), ref.mu.to_numpy(), atol=1e-14)
        # M = tau [Sigma - Sigma P'(P Sigma P' + D)^-1 P Sigma] is linear in tau.
        np.testing.assert_allclose(he.cov.to_numpy() - SIGMA, tau * m_ref, atol=1e-15)
        omega = idzorek_omega(COV, P, [0.3, 0.6, 0.9], tau=tau)
        idz = black_litterman(COV, PI, P, Q, omega=omega, tau=tau)
        idz_ref = black_litterman(
            COV, PI, P, Q, omega=idzorek_omega(COV, P, [0.3, 0.6, 0.9], tau=1.0), tau=1.0
        )
        np.testing.assert_allclose(idz.mu.to_numpy(), idz_ref.mu.to_numpy(), atol=1e-14)


@pytest.mark.parametrize("scale", [3.0, -1.0, -0.25])
def test_view_rescaling_and_sign_flip_invariance(scale):
    sigma, pi, p, q, omega = _random_problem(3, full_omega=True)
    base = black_litterman(sigma, pi, p, q, omega=omega, tau=TAU)
    # "View k: p_k mu = q_k" is the same statement as "(c p_k) mu = c q_k" with
    # uncertainty scaled by c^2 (and its covariance with other views by c).
    d = np.ones(len(q))
    d[1] = scale
    scaled = black_litterman(
        sigma, pi, d[:, None] * p, d * q, omega=omega * np.outer(d, d), tau=TAU
    )
    np.testing.assert_allclose(scaled.mu.to_numpy(), base.mu.to_numpy(), atol=1e-14)
    np.testing.assert_allclose(scaled.cov.to_numpy(), base.cov.to_numpy(), atol=1e-15)
    # The default He-Litterman Omega rescales automatically.
    np.testing.assert_allclose(
        black_litterman(sigma, pi, d[:, None] * p, d * q, tau=TAU).mu.to_numpy(),
        black_litterman(sigma, pi, p, q, tau=TAU).mu.to_numpy(),
        atol=1e-14,
    )


def test_view_order_and_asset_order_do_not_matter():
    P, Q = build_views(
        ASSETS, absolute={"EM": 0.09}, relative=[("GOLD", "BOND", 0.02), ("EU", "US", 0.01)]
    )
    omega = idzorek_omega(COV, P, [0.7, 0.4, 0.2])
    base = black_litterman(COV, PI, P, Q, omega=omega, tau=TAU)

    order = [2, 0, 1]
    permuted_views = black_litterman(
        COV, PI, P.iloc[order], Q.iloc[order], omega=omega[np.ix_(order, order)], tau=TAU
    )
    pd.testing.assert_series_equal(permuted_views.mu, base.mu, atol=1e-15, rtol=0)

    perm = ["BOND", "GOLD", "US", "EM", "EU"]
    shuffled_cov = COV.loc[perm, perm]
    permuted_assets = black_litterman(shuffled_cov, PI, P, Q, omega=omega, tau=TAU)
    assert list(permuted_assets.mu.index) == perm
    pd.testing.assert_series_equal(permuted_assets.mu.loc[ASSETS], base.mu, atol=1e-15, rtol=0)
    pd.testing.assert_frame_equal(
        permuted_assets.cov.loc[ASSETS, ASSETS], base.cov, atol=1e-16, rtol=0
    )


def test_assets_uncorrelated_with_the_views_are_untouched():
    # Block-diagonal Sigma with 3 views (K != N) all inside the first block: the second
    # block's posterior mean and covariance must equal the prior's exactly.
    rng = np.random.default_rng(5)
    s1, s2 = _random_cov(rng, 4), _random_cov(rng, 3)
    sigma = block_diag(s1, s2)
    pi = rng.normal(0.04, 0.02, 7)
    p = np.hstack([rng.normal(size=(3, 4)), np.zeros((3, 3))])
    q = np.array([0.1, -0.05, 0.02])
    res = black_litterman(sigma, pi, p, q, tau=TAU)
    mu, post = res.mu.to_numpy(), res.cov.to_numpy()
    np.testing.assert_array_equal(mu[4:], pi[4:])
    np.testing.assert_allclose(post[4:, 4:], (1 + TAU) * s2, rtol=1e-14)
    np.testing.assert_array_equal(post[:4, 4:], 0.0)
    assert np.abs(mu[:4] - pi[:4]).max() > 1e-3  # the viewed block did move


@pytest.mark.parametrize("seed", range(3))
def test_posterior_covariance_is_symmetric_psd_and_below_prior(seed):
    sigma, pi, p, q, _ = _random_problem(seed + 20, n=8, k=5)
    res = black_litterman(sigma, pi, p, q, tau=TAU)
    post = res.cov.to_numpy()
    np.testing.assert_array_equal(post, post.T)
    m = post - sigma
    # 0 <= M <= tau Sigma in the Loewner order: views can only remove uncertainty.
    assert _min_eig(m) > -1e-15
    assert _min_eig(TAU * sigma - m) > -1e-15
    assert _min_eig(post) > 0


# 1e308 and the float maximum used to overflow in 0.5 * (A + A'), warn, and then be
# reported as a view "held with certainty" -- the opposite of the truth.
@pytest.mark.parametrize("huge", [1e200, 1e308, np.finfo(float).max])
def test_extreme_omega_scale_does_not_degrade_other_views(huge):
    P, Q = build_views(ASSETS, absolute={"US": 0.06}, relative=[("EU", "EM", 0.4)])
    omega = np.diag([0.0, huge])
    both = black_litterman(COV, PI, P, Q, omega=omega, tau=TAU)
    only_certain = black_litterman(COV, PI, P.iloc[:1], Q.iloc[:1], omega=[[0.0]], tau=TAU)
    assert both.mu["US"] == pytest.approx(0.06, abs=1e-15)
    np.testing.assert_allclose(both.mu.to_numpy(), only_certain.mu.to_numpy(), atol=1e-15)
    np.testing.assert_allclose(both.cov.to_numpy(), only_certain.cov.to_numpy(), atol=1e-16)
    np.testing.assert_array_equal(both.omega, omega)


def test_float_max_omega_alone_returns_the_prior():
    P, Q = build_views(ASSETS, absolute={"EM": 0.5}, relative=[("US", "EU", -0.3)])
    # [[2, 1], [1, 3]] / 4 of the float maximum: PSD, and 0.5 * (A + A') would overflow.
    omega = (np.finfo(float).max / 4) * np.array([[2.0, 1.0], [1.0, 3.0]])
    res = black_litterman(COV, PI, P, Q, omega=omega, tau=TAU)
    np.testing.assert_allclose(res.mu.to_numpy(), PI.to_numpy(), atol=1e-15)
    np.testing.assert_allclose(res.cov.to_numpy(), (1 + TAU) * SIGMA, atol=1e-16)


def test_view_system_that_overflows_is_reported_as_such():
    # p tau Sigma p' ~ 1e297 plus an omega at the float maximum is not representable.
    P, Q = build_views(ASSETS, absolute={"US": 0.06}, relative=[("EU", "EM", 0.4)])
    big_p = P.to_numpy() * np.array([[1.0], [1e150]])
    q = Q.to_numpy() * np.array([1.0, 1e150])
    omega = np.diag([0.0, np.finfo(float).max])
    with pytest.raises(ValueError, match="overflows"):
        black_litterman(COV, PI, big_p, q, omega=omega, tau=TAU)


def test_asymmetric_omega_near_float_max_is_rejected_without_overflow():
    big = np.finfo(float).max
    P, Q = _views()
    with pytest.raises(ValueError, match="symmetric"):
        black_litterman(COV, PI, P, Q, omega=np.array([[big, big], [-big, big]]), tau=TAU)


# The PSD tolerance on omega used to be relative to its LARGEST eigenvalue, so one
# 1e12 entry (the usual way to all but ignore a view) let every other view's block have
# eigenvalues down to -100 and still pass. A negative omega_kk is "more certain than
# certain": EM then overshot its own view by 21% and p'Mp went negative.
_HUGE_NEXT_TO_INVALID = {
    "negative variance": (["US", "EM"], np.diag([1e12, -0.0005])),
    # Larger in size than EM's prior variance tau * 0.24^2 = 0.00288; this one used to be
    # misreported as "zero prior variance and zero uncertainty".
    "negative variance above the prior variance": (["US", "EM"], np.diag([1e12, -0.01])),
    "indefinite off-diagonal block": (
        ["US", "EM", "GOLD"],
        np.array([[1e12, 0.0, 0.0], [0.0, 0.001, 0.0012], [0.0, 0.0012, 0.001]]),
    ),
    "certain view with nonzero covariance": (
        ["US", "EM", "GOLD"],
        np.array([[1e12, 0.0, 0.0], [0.0, 1e-3, 1e-4], [0.0, 1e-4, 0.0]]),
    ),
    "certain view coupled to the ignored one": (["US", "EM"], np.array([[1e12, 1e3], [1e3, 0]])),
}


@pytest.mark.parametrize("case", list(_HUGE_NEXT_TO_INVALID))
def test_huge_omega_for_one_view_cannot_hide_an_invalid_omega_for_another(case):
    assets, omega = _HUGE_NEXT_TO_INVALID[case]
    # Genuinely indefinite (not rounding: views' prior variances are ~1e-3), just tiny
    # next to 1e12.
    assert _min_eig(omega) < -1e-7
    P, Q = build_views(ASSETS, absolute={a: 0.05 for a in assets})
    with pytest.raises(ValueError, match="positive semidefinite"):
        black_litterman(COV, PI, P, Q, omega=omega, tau=TAU)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_omega_verdict_does_not_depend_on_how_another_view_is_scaled(scale):
    # Stating view 0 as `scale` units of the same portfolio (its row of P, its Q and its
    # row/column of omega scaled together) is the same opinion. It must not change
    # whether view 1's omega is valid, nor the posterior when it is.
    P, Q = build_views(ASSETS, absolute={"US": 0.06, "EM": 0.12})
    d = np.array([scale, 1.0])
    scaled_p, scaled_q = d[:, None] * P.to_numpy(), d * Q.to_numpy()

    invalid = np.diag([0.001, -0.0005])  # rejected at scale 1 by any tolerance
    with pytest.raises(ValueError, match="negative variance"):
        black_litterman(SIGMA, PI.to_numpy(), scaled_p, scaled_q, omega=invalid * np.outer(d, d))

    # A rounding-level negative (1e-18 vs EM's prior variance 0.00288) is accepted at
    # every scale and acts as the certain view it stands for.
    rounding = np.diag([0.001, -1e-18])
    res = black_litterman(SIGMA, PI.to_numpy(), scaled_p, scaled_q, omega=rounding * np.outer(d, d))
    certain = black_litterman(SIGMA, PI.to_numpy(), P.to_numpy(), Q.to_numpy(), omega=[0.001, 0])
    np.testing.assert_allclose(res.mu.to_numpy(), certain.mu.to_numpy(), rtol=0, atol=1e-14)
    np.testing.assert_allclose(res.cov.to_numpy(), certain.cov.to_numpy(), rtol=0, atol=1e-15)
    assert res.mu[2] == pytest.approx(0.12, abs=1e-14)


def test_valid_omega_with_wildly_different_view_scales_is_accepted():
    # PSD matrices whose views differ in scale by up to 1e14 -- with real correlations,
    # and one singular -- are valid, and the posterior keeps 0 <= M <= tau Sigma.
    P, Q = build_views(ASSETS, absolute={"US": 0.06, "EM": 0.12, "GOLD": 0.02})
    corr = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    b = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 1.0]])
    for d, core in (([1e6, 1e-2, 1e-9], corr), ([1e8, 1.0, 1e-6], b @ b.T)):
        omega = np.outer(d, d) * core
        res = black_litterman(COV, PI, P, Q, omega=omega, tau=TAU)
        m = res.cov.to_numpy() - SIGMA
        assert _min_eig(m) > -1e-15
        assert _min_eig(TAU * SIGMA - m) > -1e-15


def test_omega_row_of_a_riskless_certain_view_must_be_zero():
    # View 0 is on an asset with zero prior variance and has omega_00 = 0, so it has no
    # variance at all; a nonzero covariance with view 1 makes omega indefinite even
    # though it is invisible next to omega_11 = 1e12.
    sigma = np.diag([0.04, 0.0, 0.01])
    omega = np.array([[0.0, 1e-9], [1e-9, 1e12]])
    with pytest.raises(ValueError, match="nonzero covariance"):
        black_litterman(sigma, [0.05, 0.01, 0.02], [[0, 1, 0], [1, 0, 0]], [0.02, 0.1], omega=omega)


def test_mixed_certain_and_uncertain_views():
    P, Q = build_views(ASSETS, absolute={"EM": 0.11, "GOLD": 0.07})
    omega = idzorek_omega(COV, P, [1.0, 0.3])
    res = black_litterman(COV, PI, P, Q, omega=omega, tau=TAU)
    assert res.mu["EM"] == pytest.approx(0.11, abs=1e-14)
    assert PI["GOLD"] < res.mu["GOLD"] < 0.07 or 0.07 < res.mu["GOLD"] < PI["GOLD"]


# --------------------------------------------------------------------------
# black_litterman: input handling
# --------------------------------------------------------------------------


def test_dataframe_inputs_are_aligned_by_label():
    P, Q = build_views(
        ASSETS, absolute={"EM": 0.09}, relative=[("GOLD", "BOND", 0.02), ("EU", "US", -0.01)]
    )
    # Distinct entries everywhere, so any misalignment of Q or of omega's rows/columns shows.
    omega = np.array([[0.004, 0.001, -0.0005], [0.001, 0.002, 0.0003], [-0.0005, 0.0003, 0.001]])
    base = black_litterman(COV, PI, P, Q, omega=omega, tau=TAU)

    # [2, 0, 1] is a 3-cycle: unlike a 2-element reversal it is NOT its own inverse,
    # so reordering by the inverse permutation by mistake cannot pass.
    cycle = [2, 0, 1]
    assert cycle != list(np.argsort(cycle))
    shuffled_cols = P[["GOLD", "EU", "BOND", "US", "EM"]]
    shuffled_q = Q.iloc[cycle]  # aligned back to P's rows by label
    omega_df = pd.DataFrame(omega, index=P.index, columns=P.index).iloc[cycle, cycle]
    res = black_litterman(COV, PI.iloc[::-1], shuffled_cols, shuffled_q, omega=omega_df, tau=TAU)
    pd.testing.assert_series_equal(res.mu, base.mu, atol=1e-16, rtol=0)
    np.testing.assert_array_equal(res.omega, omega)
    # Each input aligned on its own (the other passed in P's order) gives the same answer.
    only_q = black_litterman(COV, PI, P, shuffled_q, omega=omega, tau=TAU)
    pd.testing.assert_series_equal(only_q.mu, base.mu, atol=1e-16, rtol=0)
    only_omega = black_litterman(COV, PI, P, Q, omega=omega_df, tau=TAU)
    pd.testing.assert_series_equal(only_omega.mu, base.mu, atol=1e-16, rtol=0)
    # And the posterior really depends on the alignment: a positional mix-up moves it.
    wrong = black_litterman(COV, PI, P, Q.iloc[cycle].to_numpy(), omega=omega, tau=TAU)
    assert np.abs(wrong.mu - base.mu).max() > 1e-3


def test_omega_series_of_view_variances_is_aligned_by_label():
    # A Series of view variances is aligned to P's rows like Q, an omega DataFrame and
    # Idzorek confidences are; it used to be applied by position, silently giving each
    # view another view's uncertainty.
    P, Q = build_views(
        ASSETS, absolute={"EM": 0.12}, relative=[("GOLD", "BOND", 0.04), ("EU", "US", -0.01)]
    )
    variances = pd.Series([0.0001, 0.05, 0.002], index=P.index)
    base = black_litterman(COV, PI, P, Q, omega=np.diag(variances.to_numpy()), tau=TAU)
    cycle = [2, 0, 1]  # a 3-cycle is not its own inverse (see the DataFrame test above)
    res = black_litterman(COV, PI, P, Q, omega=variances.iloc[cycle], tau=TAU)
    np.testing.assert_array_equal(res.omega, np.diag(variances.to_numpy()))
    pd.testing.assert_series_equal(res.mu, base.mu, atol=1e-16, rtol=0)
    pd.testing.assert_frame_equal(res.cov, base.cov, atol=1e-16, rtol=0)
    # Same order as P: unchanged. Unlabelled P: nothing to align to, so by position.
    same = black_litterman(COV, PI, P, Q, omega=variances, tau=TAU)
    pd.testing.assert_series_equal(same.mu, base.mu, atol=1e-16, rtol=0)
    positional = black_litterman(
        SIGMA, PI.to_numpy(), P.to_numpy(), Q.to_numpy(), omega=variances.iloc[cycle], tau=TAU
    )
    np.testing.assert_array_equal(np.diag(positional.omega), variances.iloc[cycle].to_numpy())
    assert np.abs(positional.mu.to_numpy() - base.mu.to_numpy()).max() > 1e-3
    with pytest.raises(ValueError, match="row labels of P"):
        black_litterman(COV, PI, P, Q, omega=variances.set_axis(["a", "b", "c"]), tau=TAU)


def test_numpy_inputs_vector_view_and_diagonal_omega_vector():
    p = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    res_vec = black_litterman(SIGMA, PI.to_numpy(), p, [0.12], omega=[0.003], tau=TAU)
    res_mat = black_litterman(SIGMA, PI.to_numpy(), p[None, :], [0.12], omega=[[0.003]], tau=TAU)
    assert list(res_vec.mu.index) == list(range(5))
    pd.testing.assert_series_equal(res_vec.mu, res_mat.mu)
    np.testing.assert_array_equal(res_vec.omega, [[0.003]])
    # Single-view closed form: mu = pi + tau Sigma p (q - p'pi) / (p tau Sigma p + omega).
    s = TAU * p @ SIGMA @ p
    expected = PI.to_numpy() + TAU * SIGMA @ p * (0.12 - p @ PI.to_numpy()) / (s + 0.003)
    np.testing.assert_allclose(res_vec.mu.to_numpy(), expected, atol=1e-15)


def test_single_view_as_labelled_series_is_aligned_by_asset():
    p = pd.Series({"GOLD": 1.0, "EM": -1.0, "US": 0.0, "EU": 0.0, "BOND": 0.0})
    from_series = black_litterman(COV, PI, p, [0.01], tau=TAU)
    P, Q = build_views(ASSETS, relative=[("GOLD", "EM", 0.01)])
    from_frame = black_litterman(COV, PI, P, Q, tau=TAU)
    pd.testing.assert_series_equal(from_series.mu, from_frame.mu, atol=1e-16, rtol=0)


def test_default_omega_is_he_litterman_and_result_is_frozen():
    P, Q = build_views(ASSETS, absolute={"EM": 0.09}, relative=[("GOLD", "BOND", 0.02)])
    res = black_litterman(COV, PI, P, Q, tau=0.1)
    assert isinstance(res, BlackLittermanResult)
    np.testing.assert_allclose(res.omega, he_litterman_omega(COV, P, tau=0.1), rtol=1e-15)
    assert isinstance(res.mu, pd.Series) and isinstance(res.cov, pd.DataFrame)
    assert list(res.cov.index) == ASSETS and list(res.cov.columns) == ASSETS
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.tau = 0.2  # type: ignore[misc]


@pytest.mark.parametrize("with_views", [True, False])
def test_result_does_not_alias_the_callers_arrays(with_views):
    # pandas 2.x wraps a float64 ndarray in a Series without copying, so storing the
    # validated prior as-is let a later write to the caller's array change res.prior.
    sigma = SIGMA.copy()
    pi = PI.to_numpy().copy()
    p = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    q = np.array([0.1])
    omega = np.array([[0.003]])
    if with_views:
        res = black_litterman(sigma, pi, p, q, omega=omega, tau=TAU)
    else:
        res = black_litterman(sigma, pi, tau=TAU)
    snapshot = (res.mu.copy(), res.cov.copy(), res.prior.copy(), res.omega.copy())
    for array in (sigma, pi, p, q, omega):
        array[...] = 999.0
    pd.testing.assert_series_equal(res.prior, pd.Series(PI.to_numpy(), index=range(5)))
    pd.testing.assert_series_equal(res.prior, snapshot[2])
    pd.testing.assert_series_equal(res.mu, snapshot[0])
    pd.testing.assert_frame_equal(res.cov, snapshot[1])
    np.testing.assert_array_equal(res.omega, snapshot[3])
    for attr in (res.mu, res.prior):
        assert not np.shares_memory(attr.to_numpy(), pi)


def test_end_to_end_with_estimated_risk_aversion():
    # A typical pipeline: estimate delta from market returns, reverse-optimize, add views.
    rng = np.random.default_rng(11)
    daily = rng.multivariate_normal(PI.to_numpy() / 252, SIGMA / 252, size=2000)
    market = daily @ W_MKT.to_numpy()
    delta = market_implied_risk_aversion(market)
    assert delta > 0
    prior = implied_equilibrium_returns(COV, W_MKT, risk_aversion=delta)
    P, Q = build_views(ASSETS, relative=[("EM", "EU", 0.03)])
    res = black_litterman(COV, prior, P, Q, omega=idzorek_omega(COV, P, [0.5]))
    spread = res.mu["EM"] - res.mu["EU"]
    prior_spread = prior["EM"] - prior["EU"]
    assert spread == pytest.approx(0.5 * (prior_spread + 0.03), abs=1e-14)


def _views():
    P, Q = build_views(ASSETS, absolute={"EM": 0.09}, relative=[("GOLD", "BOND", 0.02)])
    return P, Q


@pytest.mark.parametrize(
    ("make_kwargs", "match"),
    [
        (lambda P, Q: {"P": P}, "together"),
        (lambda P, Q: {"Q": Q}, "together"),
        (lambda P, Q: {"omega": np.eye(2)}, "without views"),
        (lambda P, Q: {"P": P, "Q": Q.to_numpy()[:1]}, "one entry per view"),
        (lambda P, Q: {"P": P, "Q": [0.09, np.nan]}, "NaN"),
        (lambda P, Q: {"P": P.to_numpy()[:, :4], "Q": Q.to_numpy()}, "shape"),
        (lambda P, Q: {"P": P.rename(columns={"EM": "JP"}), "Q": Q}, "unknown"),
        (lambda P, Q: {"P": P.drop(columns="US"), "Q": Q}, "missing"),
        (lambda P, Q: {"P": np.zeros((2, 5)), "Q": [0.1, 0.2]}, "all-zero"),
        (lambda P, Q: {"P": np.full((1, 5), np.nan), "Q": [0.1]}, "NaN"),
        (lambda P, Q: {"P": np.empty((0, 5)), "Q": []}, "no views"),
        (lambda P, Q: {"P": P, "Q": Q.set_axis(["a", "b"])}, "row labels of P"),
        (lambda P, Q: {"P": P, "Q": Q, "omega": np.eye(3)}, "shape"),
        (lambda P, Q: {"P": P, "Q": Q, "omega": np.array([[1, 0.5], [0, 1]])}, "symmetric"),
        (lambda P, Q: {"P": P, "Q": Q, "omega": np.diag([0.01, -0.01])}, "semidefinite"),
        (lambda P, Q: {"P": P, "Q": Q, "omega": np.diag([1e12, -0.0005])}, "semidefinite"),
        # Rows relabelled but not columns: not a labelled K x K matrix, so it cannot be
        # aligned to P (reordering it with np.ix_ would mix up rows and columns).
        (
            lambda P, Q: {
                "P": P,
                "Q": Q,
                "omega": pd.DataFrame(np.eye(2), index=P.index[::-1], columns=P.index),
            },
            "identical index and columns",
        ),
        (lambda P, Q: {"P": P, "Q": Q, "omega": np.diag([0.01, np.nan])}, "NaN"),
        (lambda P, Q: {"P": P, "Q": Q, "omega": np.diag([0.01, np.inf])}, "NaN"),
        (lambda P, Q: {"P": P, "Q": Q, "tau": 0.0}, "tau"),
        (lambda P, Q: {"P": P, "Q": Q, "tau": -0.05}, "tau"),
        (lambda P, Q: {"P": P, "Q": Q, "tau": np.nan}, "tau"),
        (lambda P, Q: {"P": P, "Q": Q, "tau": np.inf}, "tau"),
        (lambda P, Q: {"prior": PI.to_numpy()[:4]}, "expected 5"),
        (lambda P, Q: {"prior": PI.drop("EM")}, "missing"),
        (lambda P, Q: {"cov": COV - 0.05 * np.eye(5)}, "positive semidefinite"),
    ],
)
def test_black_litterman_validation(make_kwargs, match):
    P, Q = _views()
    kwargs = {"cov": COV, "prior": PI, **make_kwargs(P, Q)}
    with pytest.raises(ValueError, match=match):
        black_litterman(**kwargs)


def test_duplicate_column_labels_in_P_are_rejected():
    P, Q = _views()
    dup = P.copy()
    dup.columns = ["US", "EU", "EM", "BOND", "US"]
    with pytest.raises(ValueError, match="duplicate"):
        black_litterman(COV, PI, dup, Q)


def test_linearly_dependent_certain_views_are_rejected_but_uncertain_ones_are_fine():
    P, Q = build_views(ASSETS, absolute={"EM": 0.09}, relative=[("EU", "US", 0.01)])
    dependent = pd.concat([P, 2.0 * P.iloc[[1]]])
    q = np.concatenate([Q.to_numpy(), [0.02]])
    with pytest.raises(ValueError, match="singular"):
        black_litterman(COV, PI, dependent.to_numpy(), q, omega=np.zeros((3, 3)), tau=TAU)
    # The same redundancy with uncertain views is a legitimate (overlapping) opinion.
    res = black_litterman(COV, PI, dependent.to_numpy(), q, tau=TAU)
    assert np.isfinite(res.mu.to_numpy()).all()


def test_certain_view_on_a_zero_variance_portfolio_is_rejected():
    sigma = np.diag([0.04, 0.0, 0.01])  # asset 1 is riskless under the prior
    with pytest.raises(ValueError, match="zero prior variance"):
        black_litterman(sigma, [0.05, 0.01, 0.02], [[0.0, 1.0, 0.0]], [0.02], omega=[[0.0]])


def _twin_returns(seed: int, tracking_error: float = 0.0) -> pd.DataFrame:
    """B is A minus a constant fee (plus optional noise): Var(A - B) is 0 in exact arithmetic."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0005, 0.01, 500)
    b = a - 0.0001 + rng.normal(0.0, tracking_error, 500) if tracking_error else a - 0.0001
    return pd.DataFrame({"A": a, "B": b, "C": rng.normal(0.0003, 0.012, 500)})


_TWIN_PI = pd.Series([0.05, 0.05, 0.04], index=list("ABC"))


@pytest.mark.parametrize("seed", [0, 1, 78])
@pytest.mark.parametrize(
    "omega_kind", ["he_litterman", "zero", "idzorek_certain", "rounding_negative"]
)
def test_view_on_a_spread_whose_variance_is_rounding_noise_is_rejected(seed, omega_kind):
    # Var(A - B) of the sample covariance rounds to exactly 0 for some seeds and to
    # ~1e-18 for others (seeds 0 and 78). It used to raise in the first case but silently
    # return a posterior that was a ratio of two rounding errors in the second: which leg
    # moved, and how the unrelated asset C moved, was decided by the noise.
    # A rounding-level negative omega (-1e-30) must get the same diagnosis whatever the
    # noise: the omega PSD check measures it against the view's noise floor, not against
    # a prior variance that may be exactly 0 or 1e-18.
    cov = _twin_returns(seed).cov() * TRADING_DAYS
    P, Q = build_views(list("ABC"), relative=[("A", "B", 0.0252)])
    certain = -1e-30 if omega_kind == "rounding_negative" else 0.0
    omega = {
        "he_litterman": None,
        "zero": np.zeros((1, 1)),
        "idzorek_certain": idzorek_omega(cov, P, [1.0]),
        "rounding_negative": np.array([[certain]]),
    }[omega_kind]
    with pytest.raises(ValueError, match="zero prior variance"):
        black_litterman(cov, _TWIN_PI, P, Q, omega=omega, tau=TAU)
    # A legitimate view in the same call does not rescue it.
    P2, Q2 = build_views(list("ABC"), absolute={"C": 0.06}, relative=[("A", "B", 0.0252)])
    omega2 = None if omega is None else np.diag([0.001, certain])
    with pytest.raises(ValueError, match=r"positions \[1\]"):
        black_litterman(cov, _TWIN_PI, P2, Q2, omega=omega2, tau=TAU)


@pytest.mark.parametrize("perturbed", [0, 1])
def test_noise_level_variance_is_rejected_whichever_entry_carries_it(perturbed):
    # Two covariance matrices that differ only in the last bit of one entry gave opposite
    # answers: one ulp on A's variance moved A by the full surprise, one ulp on B's moved B.
    # Neither is information, so both must be rejected.
    v, c, w = 0.0252, 0.004, 0.0361
    sigma = np.array([[v, v, c], [v, v, c], [c, c, w]])
    sigma[perturbed, perturbed] = np.nextafter(v, 1.0)
    assert np.array([1.0, -1.0, 0.0]) @ sigma @ np.array([1.0, -1.0, 0.0]) > 0.0
    with pytest.raises(ValueError, match="zero prior variance"):
        black_litterman(sigma, _TWIN_PI.to_numpy(), [[1.0, -1.0, 0.0]], [0.0252], omega=[[0.0]])


def test_uncertain_view_on_a_zero_variance_spread_leaves_the_prior():
    # With a positive omega the system is well posed, and since the prior already pins the
    # spread down exactly, the view cannot move anything (up to rounding).
    cov = _twin_returns(0).cov() * TRADING_DAYS
    P, Q = build_views(list("ABC"), relative=[("A", "B", 0.0252)])
    res = black_litterman(cov, _TWIN_PI, P, Q, omega=[[0.001]], tau=TAU)
    np.testing.assert_allclose(res.mu.to_numpy(), _TWIN_PI.to_numpy(), atol=1e-12)


@pytest.mark.parametrize("tracking_error", [1e-6, 1e-7])
def test_near_twins_with_real_tracking_error_are_not_mistaken_for_noise(tracking_error):
    # A daily tracking error of 1e-7 on 1% daily vol is tiny but real: the certain view
    # must be accepted and must hold, and the He-Litterman view must give the midpoint.
    cov = _twin_returns(0, tracking_error).cov() * TRADING_DAYS
    P, Q = build_views(list("ABC"), relative=[("A", "B", 0.0252)])
    certain = black_litterman(cov, _TWIN_PI, P, Q, omega=[[0.0]], tau=TAU)
    assert certain.mu["A"] - certain.mu["B"] == pytest.approx(0.0252, abs=1e-12)
    he = black_litterman(cov, _TWIN_PI, P, Q, tau=TAU)
    # Omega and A each compute p Sigma p' (~1e-10, from entries ~0.03) with cancellation,
    # so the midpoint is only accurate to about eps * 0.03 / 1e-10 relative.
    assert he.mu["A"] - he.mu["B"] == pytest.approx(0.5 * 0.0252, rel=1e-6)
