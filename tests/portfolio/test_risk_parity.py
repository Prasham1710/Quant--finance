"""Tests for quantfin.portfolio.risk_parity.

The math is checked against independent answers:

* risk contributions: the explicit double sum, a finite-difference Euler
  gradient of σ(w), a hand-computed example, and the sample volatility of the
  portfolio's own return series (daily vs annualized, ddof = 1);
* inverse volatility / variance: hand-computed fractions and the closed-form
  minimum-variance portfolio of a diagonal covariance;
* risk budgeting: closed forms (2 assets for any correlation and any budgets,
  equal correlations, diagonal covariance, a 3-asset block example), two
  independent reference solvers (cyclical coordinate descent, and SciPy's
  trust-region + Powell-hybrid solvers in log coordinates), Spinu's
  first-order optimality condition, Maillard et al.'s volatility ordering,
  and invariances (scale, per-asset units, permutations);
* the damped Newton line search: an independent backtracking implementation
  that evaluates the objective directly, and a 60-digit ``decimal`` evaluation
  of the objective change where float64 subtraction cannot resolve it (tiny
  risk budgets), plus a semi-closed form for constant correlations with
  budgets down to 1e-16.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from quantfin.portfolio._common import TRADING_DAYS, OptimizationError
from quantfin.portfolio.risk_parity import (
    _armijo_holds,
    _line_search,
    equal_risk_contribution,
    inverse_variance,
    inverse_volatility,
    risk_contributions,
)

# ---------------------------------------------------------------------------
# Fixtures and independent reference implementations
# ---------------------------------------------------------------------------


def random_cov(
    n: int, rng: np.random.Generator, vol_low: float = 0.05, vol_high: float = 0.5
) -> np.ndarray:
    """Random positive-definite covariance with mixed-sign correlations."""
    a = rng.standard_normal((n, n + 5))
    s = a @ a.T
    d = np.sqrt(np.diag(s))
    corr = s / np.outer(d, d)
    vols = rng.uniform(vol_low, vol_high, n)
    cov = corr * np.outer(vols, vols)
    return 0.5 * (cov + cov.T)


def equicorrelated_cov(vols: np.ndarray, rho: float) -> np.ndarray:
    n = len(vols)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(vols, vols)


def highly_correlated_cov(n: int, rng: np.random.Generator, rho: float = 0.99) -> np.ndarray:
    """Correlations around ``rho`` (not all equal) and vols spanning 4 orders of magnitude."""
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    loadings = 0.05 * rng.standard_normal((n, 3))
    corr = corr + loadings @ loadings.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    vols = np.exp(rng.uniform(np.log(1e-3), np.log(10.0), n))
    cov = corr * np.outer(vols, vols)
    return 0.5 * (cov + cov.T)


def variance_shares(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Independent relative risk contributions w_i (Σw)_i / w'Σw (explicit loops)."""
    n = len(w)
    contrib = np.array([sum(w[i] * w[j] * cov[i, j] for j in range(n)) for i in range(n)])
    return contrib / contrib.sum()


def ccd_risk_budgeting(cov: np.ndarray, b: np.ndarray, sweeps: int = 50_000) -> np.ndarray:
    """Cyclical coordinate descent on Spinu's objective (Griveau-Billion, Richard & Roncalli 2013).

    Each coordinate solves ∂f/∂y_i = 0 exactly: Σ_ii y_i² + c_i y_i - b_i = 0
    with c_i = Σ_{j≠i} Σ_ij y_j, whose positive root is taken.
    """
    b = np.asarray(b, dtype=float) / np.sum(b)
    y = b / np.sqrt(np.diag(cov))
    for _ in range(sweeps):
        for i in range(len(b)):
            c = cov[i] @ y - cov[i, i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4.0 * cov[i, i] * b[i])) / (2.0 * cov[i, i])
        if np.max(np.abs(y * (cov @ y) - b)) < 1e-15:
            break
    return y / y.sum()


def scipy_risk_budgeting(cov: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference via SciPy only, in log coordinates y = exp(z) (no correlation scaling).

    g(z) = ½ y'Σy - b'z has gradient y∘(Σy) - b and Hessian diag(y∘Σy) + YΣY;
    its only stationary point is the risk-budgeting solution. A global
    trust-region solve (Moré-Sorensen "trust-exact") gets close, then MINPACK's
    Powell hybrid root finder polishes ∇g = 0 to machine precision. (Root
    finding alone from a far start stalls at least-squares local minima.)
    """
    b = np.asarray(b, dtype=float) / np.sum(b)

    def g(z: np.ndarray) -> float:
        y = np.exp(z)
        return 0.5 * y @ cov @ y - b @ z

    def grad(z: np.ndarray) -> np.ndarray:
        y = np.exp(z)
        return y * (cov @ y) - b

    def hess(z: np.ndarray) -> np.ndarray:
        y = np.exp(z)
        return np.diag(y * (cov @ y)) + y[:, None] * cov * y[None, :]

    coarse = optimize.minimize(
        g, np.zeros(len(b)), jac=grad, hess=hess, method="trust-exact", options={"gtol": 1e-8}
    )
    polished = optimize.root(grad, coarse.x, jac=hess, method="hybr", options={"xtol": 1e-14})
    assert np.max(np.abs(grad(polished.x))) < 1e-13
    y = np.exp(polished.x)
    return y / y.sum()


def two_asset_budget_weights(s1: float, s2: float, rho: float, b1: float, b2: float) -> np.ndarray:
    """Closed form for 2 assets: with u_i = w_i σ_i and t = u1/u2, the condition
    (u1² + ρ u1 u2) / (u2² + ρ u1 u2) = b1/b2 =: k is t² + ρ(1 - k) t - k = 0."""
    k = b1 / b2
    t = (-rho * (1.0 - k) + np.sqrt(rho**2 * (1.0 - k) ** 2 + 4.0 * k)) / 2.0
    w = np.array([t / s1, 1.0 / s2])
    return w / w.sum()


def equicorrelated_budget_weights(vols: np.ndarray, rho: float, b: np.ndarray) -> np.ndarray:
    """Semi-closed form for a constant correlation ρ and any budgets.

    In correlation units x_i = y_i σ_i the conditions x_i (Cx)_i = b_i read
    (1 - ρ) x_i² + ρ S x_i - b_i = 0 with S = Σ_j x_j, so x_i(S) is the
    positive root of a quadratic and S solves the scalar equation
    Σ_i x_i(S) = S (Brent). Each root is written in the form without
    cancellation for the sign of ρ, so budgets of 1e-16 lose no accuracy.
    """
    b = np.asarray(b, dtype=float) / np.sum(b)

    def x_of(total: float) -> np.ndarray:
        disc = np.sqrt((rho * total) ** 2 + 4.0 * (1.0 - rho) * b)
        if rho >= 0.0:
            return 2.0 * b / (rho * total + disc)
        return (-rho * total + disc) / (2.0 * (1.0 - rho))

    total = optimize.brentq(lambda t: x_of(t).sum() - t, 1e-8, 1e4, xtol=1e-15, rtol=1e-15)
    y = x_of(total) / vols
    return y / y.sum()


def spinu_objective(corr: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """f(x) = ½ x'Cx - Σ b_i ln x_i, evaluated directly."""
    return float(0.5 * x @ corr @ x - b @ np.log(x))


# A 3-asset problem on which the solver's first full Newton step INCREASES
# Spinu's objective (a strong hedge between assets 1 and 2), so the line
# search has to backtrack.
BACKTRACK_CORR = np.array([[1.0, -0.79, 0.03], [-0.79, 1.0, 0.57], [0.03, 0.57, 1.0]])
BACKTRACK_BUDGETS = np.array([0.48, 0.37, 0.15])


# ---------------------------------------------------------------------------
# risk_contributions
# ---------------------------------------------------------------------------


def test_risk_contributions_hand_computed_two_assets():
    # σ = (10%, 20%), ρ = 0.5 → Σ = [[0.01, 0.01], [0.01, 0.04]]; w = (0.6, 0.4)
    # Σw = (0.010, 0.022); w∘Σw = (0.006, 0.0088); w'Σw = 0.0148.
    cov = pd.DataFrame([[0.01, 0.01], [0.01, 0.04]], index=["A", "B"], columns=["A", "B"])
    w = pd.Series({"A": 0.6, "B": 0.4})
    rc = risk_contributions(w, cov)
    sigma_p = np.sqrt(0.0148)
    np.testing.assert_allclose(rc.to_numpy(), [0.006 / sigma_p, 0.0088 / sigma_p], rtol=1e-14)
    assert list(rc.index) == ["A", "B"]
    rel = risk_contributions(w, cov, relative=True)
    np.testing.assert_allclose(rel.to_numpy(), [15.0 / 37.0, 22.0 / 37.0], rtol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 7, 30])
def test_risk_contributions_sum_to_volatility_and_match_double_sum(n):
    rng = np.random.default_rng(10 + n)
    cov = random_cov(n, rng)
    w = rng.normal(size=n)  # includes short positions; need not sum to 1
    rc = risk_contributions(w, cov).to_numpy()
    sigma_p = np.sqrt(w @ cov @ w)
    assert rc.sum() == pytest.approx(sigma_p, rel=1e-12)
    np.testing.assert_allclose(rc / sigma_p, variance_shares(w, cov), atol=1e-12)
    rel = risk_contributions(w, cov, relative=True).to_numpy()
    assert rel.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rel, rc / sigma_p, rtol=1e-12, atol=1e-15)


def test_risk_contributions_are_euler_gradient_terms():
    """RC_i = w_i ∂σ/∂w_i with the gradient taken by central finite differences."""
    rng = np.random.default_rng(1)
    n = 6
    cov = random_cov(n, rng)
    w = rng.dirichlet(np.ones(n)) - 0.1

    def vol(v: np.ndarray) -> float:
        return float(np.sqrt(v @ cov @ v))

    h = 1e-6
    grad = np.array([(vol(w + h * e) - vol(w - h * e)) / (2 * h) for e in np.eye(n)])
    np.testing.assert_allclose(risk_contributions(w, cov).to_numpy(), w * grad, atol=1e-9)


def test_risk_contributions_hedge_is_negative():
    # σ1 = σ2 = 1, ρ = 0.9, w = (1, -0.5): Σw = (0.55, 0.4), w∘Σw = (0.55, -0.2).
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    rc = risk_contributions([1.0, -0.5], cov).to_numpy()
    np.testing.assert_allclose(rc, np.array([0.55, -0.2]) / np.sqrt(0.35), rtol=1e-14)
    assert rc[1] < 0


def test_risk_contributions_from_returns_daily_vs_annualized_ddof1():
    """Contributions sum to the portfolio's sample volatility (ddof=1), and an
    annualized covariance (×252) scales them by exactly sqrt(252)."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.01, size=(400, 4)) + rng.normal(0, 0.006, size=(400, 1))
    w = np.array([0.4, 0.3, 0.2, 0.1])
    cov_daily = np.cov(returns, rowvar=False, ddof=1)
    port = returns @ w
    rc_daily = risk_contributions(w, cov_daily)
    assert rc_daily.sum() == pytest.approx(np.std(port, ddof=1), rel=1e-12)
    rc_annual = risk_contributions(w, cov_daily * TRADING_DAYS)
    assert rc_annual.sum() == pytest.approx(np.std(port, ddof=1) * np.sqrt(TRADING_DAYS), rel=1e-12)
    np.testing.assert_allclose(rc_annual, rc_daily * np.sqrt(TRADING_DAYS), rtol=1e-12)
    np.testing.assert_allclose(
        risk_contributions(w, cov_daily * TRADING_DAYS, relative=True),
        risk_contributions(w, cov_daily, relative=True),
        rtol=1e-12,
    )


def test_risk_contributions_homogeneity():
    """σ is homogeneous of degree 1, so RC(k w) = |k| RC(w); shares are invariant."""
    rng = np.random.default_rng(3)
    cov = random_cov(5, rng)
    w = rng.dirichlet(np.ones(5))
    base = risk_contributions(w, cov).to_numpy()
    for k in (3.0, 0.01, -2.0):
        np.testing.assert_allclose(risk_contributions(k * w, cov).to_numpy(), abs(k) * base)
        np.testing.assert_allclose(
            risk_contributions(k * w, cov, relative=True).to_numpy(), base / base.sum()
        )


def test_risk_contributions_aligns_series_by_label():
    rng = np.random.default_rng(4)
    assets = ["SPY", "TLT", "GLD", "EEM"]
    cov = pd.DataFrame(random_cov(4, rng), index=assets, columns=assets)
    w = pd.Series([0.1, 0.2, 0.3, 0.4], index=assets)
    shuffled = w[["GLD", "EEM", "SPY", "TLT"]]
    expected = risk_contributions(w.to_numpy(), cov.to_numpy()).to_numpy()
    rc = risk_contributions(shuffled, cov)
    assert list(rc.index) == assets
    np.testing.assert_allclose(rc.to_numpy(), expected, rtol=1e-15)


def test_risk_contributions_numpy_labels():
    rc = risk_contributions([0.5, 0.5], np.diag([0.04, 0.01]))
    assert list(rc.index) == [0, 1]
    # Diagonal Σ: RC_i = w_i² σ_i² / σ_p.
    sigma_p = np.sqrt(0.25 * 0.04 + 0.25 * 0.01)
    np.testing.assert_allclose(rc.to_numpy(), [0.01 / sigma_p, 0.0025 / sigma_p], rtol=1e-14)


@pytest.mark.parametrize(
    ("weights", "cov", "match"),
    [
        ([0.0, 0.0], np.eye(2), "not strictly positive"),
        # ρ = 1 and equal vols: the long-short portfolio has exactly zero risk.
        ([1.0, -1.0], np.ones((2, 2)), "not strictly positive"),
        # Not a covariance (ρ = 2). Rejected whatever the weights: with long-only
        # weights w'Σw > 0, so only a PSD check catches it.
        ([1.0, -1.0], np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semi-definite"),
        ([1.0, 1.0], np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semi-definite"),
        # A negative variance would give a long position a negative variance share.
        ([0.5, 0.5], np.diag([-1.0, 4.0]), "non-negative"),
        # Every pairwise correlation is valid, but jointly they are not (eigenvalue < 0).
        (
            [0.4, 0.3, 0.3],
            np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]),
            "positive semi-definite",
        ),
        # Zero variance but a non-zero covariance: the 2x2 minor is negative.
        ([0.5, 0.5], np.array([[0.0, 0.1], [0.1, 1.0]]), "positive semi-definite"),
        # |ρ| ~ 1e320 overflows the correlation; must be a clean ValueError, not a warning.
        ([0.5, 0.5], np.array([[1e-320, 1.0], [1.0, 1e-320]]), "positive semi-definite"),
        ([0.5, 0.5, 0.0], np.eye(2), "expected 2 values"),
        ([0.5, np.nan], np.eye(2), "NaN"),
        (pd.Series({0: 0.5, 1: 0.3, 9: 0.2}), np.eye(2), "not in the covariance"),
        (pd.Series({0: 1.0}), np.eye(2), "missing"),
        ([0.5, 0.5], np.array([[1.0, 0.2], [0.1, 1.0]]), "symmetric"),
        ([0.5, 0.5], np.ones((2, 3)), "square"),
    ],
)
def test_risk_contributions_validation(weights, cov, match):
    for relative in (False, True):
        with pytest.raises(ValueError, match=match):
            risk_contributions(weights, cov, relative=relative)


def test_risk_contributions_accepts_valid_singular_covariances():
    """PSD-but-singular matrices are valid covariances and must not be rejected."""
    # Cash: a zero-variance asset contributes nothing; the rest is the 2-asset problem.
    vols = np.array([0.1, 0.2])
    risky = np.array([[1.0, 0.5], [0.5, 1.0]]) * np.outer(vols, vols)
    cov = np.zeros((3, 3))
    cov[:2, :2] = risky
    w = np.array([0.5, 0.3, 0.2])
    rc = risk_contributions(w, cov).to_numpy()
    np.testing.assert_allclose(rc[:2], risk_contributions(w[:2], risky).to_numpy(), rtol=1e-14)
    assert rc[2] == 0.0
    # Fewer observations than assets: rank-deficient sample covariance whose
    # smallest eigenvalues are rounding noise of either sign.
    rng = np.random.default_rng(15)
    sample = np.cov(rng.standard_normal((8, 20)), rowvar=False)
    assert np.linalg.eigvalsh(sample)[0] < 1e-12
    w = rng.dirichlet(np.ones(20))
    rel = risk_contributions(w, sample, relative=True).to_numpy()
    np.testing.assert_allclose(rel, variance_shares(w, sample), atol=1e-12)


# ---------------------------------------------------------------------------
# inverse_volatility / inverse_variance
# ---------------------------------------------------------------------------


def test_inverse_volatility_and_variance_hand_computed():
    # σ = (0.1, 0.2, 0.4): 1/σ = (10, 5, 2.5) → (4, 2, 1)/7; 1/σ² = (100, 25, 6.25) → (16, 4, 1)/21.
    vols = np.array([0.1, 0.2, 0.4])
    corr = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.5], [-0.2, 0.5, 1.0]])
    labels = ["x", "y", "z"]
    cov = pd.DataFrame(corr * np.outer(vols, vols), index=labels, columns=labels)
    iv = inverse_volatility(cov)
    ivar = inverse_variance(cov)
    np.testing.assert_allclose(iv.to_numpy(), np.array([4.0, 2.0, 1.0]) / 7.0, rtol=1e-14)
    np.testing.assert_allclose(ivar.to_numpy(), np.array([16.0, 4.0, 1.0]) / 21.0, rtol=1e-14)
    assert list(iv.index) == labels and list(ivar.index) == labels


def test_inverse_weights_ignore_correlations_and_scale():
    rng = np.random.default_rng(5)
    cov = random_cov(6, rng)
    diag = np.diag(np.diag(cov))
    for fn in (inverse_volatility, inverse_variance):
        base = fn(cov).to_numpy()
        np.testing.assert_allclose(fn(diag).to_numpy(), base, rtol=1e-14)
        # Daily variances and absurdly small/large scales give the same weights.
        for scale in (1.0 / TRADING_DAYS, float(TRADING_DAYS), 1e-200, 1e200):
            np.testing.assert_allclose(fn(cov * scale).to_numpy(), base, rtol=1e-13)


def test_inverse_variance_is_min_variance_for_diagonal_cov():
    rng = np.random.default_rng(6)
    var = rng.uniform(0.01, 0.2, 8)
    cov = np.diag(var)
    ones = np.ones(8)
    gmv = np.linalg.solve(cov, ones)
    gmv /= ones @ gmv
    np.testing.assert_allclose(inverse_variance(cov).to_numpy(), gmv, rtol=1e-13)


def test_inverse_volatility_equalizes_standalone_risk():
    rng = np.random.default_rng(8)
    cov = random_cov(5, rng)
    w = inverse_volatility(cov).to_numpy()
    standalone = w * np.sqrt(np.diag(cov))
    np.testing.assert_allclose(standalone, standalone.mean(), rtol=1e-13)
    # Inverse variance = squared inverse volatility, renormalized.
    np.testing.assert_allclose(inverse_variance(cov).to_numpy(), w**2 / np.sum(w**2), rtol=1e-13)


@pytest.mark.parametrize("fn", [inverse_volatility, inverse_variance])
@pytest.mark.parametrize(
    ("cov", "match"),
    [
        (np.diag([0.04, 0.0]), "strictly positive variance"),
        (np.diag([0.04, -0.01]), "strictly positive variance"),
        (np.array([[0.04, np.nan], [np.nan, 0.01]]), "NaN"),
        (np.zeros((0, 0)), "non-empty"),
    ],
)
def test_inverse_weights_validation(fn, cov, match):
    with pytest.raises(ValueError, match=match):
        fn(cov)


# ---------------------------------------------------------------------------
# equal_risk_contribution: closed forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [-0.99999, -0.99, -0.5, 0.0, 0.3, 0.99, 1.0])
def test_erc_two_assets_is_inverse_volatility_for_any_correlation(rho):
    """w1(Σw)_1 - w2(Σw)_2 = (w1σ1)² - (w2σ2)², so ERC ⇔ w1σ1 = w2σ2."""
    s1, s2 = 0.08, 0.3
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w = equal_risk_contribution(cov).to_numpy()
    np.testing.assert_allclose(w, np.array([1 / s1, 1 / s2]) / (1 / s1 + 1 / s2), rtol=1e-12)


@pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.6, 0.95])
@pytest.mark.parametrize("budgets", [(0.2, 0.8), (0.9, 0.1), (0.5, 0.5)])
def test_risk_budgeting_two_assets_closed_form(rho, budgets):
    s1, s2 = 0.25, 0.1
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w = equal_risk_contribution(cov, budgets).to_numpy()
    np.testing.assert_allclose(w, two_asset_budget_weights(s1, s2, rho, *budgets), rtol=1e-11)
    shares = variance_shares(w, cov)
    np.testing.assert_allclose(shares, budgets, atol=1e-12)


@pytest.mark.parametrize(("n", "rho"), [(5, -0.2), (5, 0.0), (10, 0.5), (50, 0.99), (50, 0.999)])
def test_erc_equal_correlations_is_inverse_volatility(n, rho):
    """Maillard et al. (2010): with a constant correlation, ERC = inverse volatility,
    because (Σw)_i = σ_i[(1-ρ) w_iσ_i + ρ Σ_j w_jσ_j] is equal-share when w_iσ_i is constant."""
    rng = np.random.default_rng(n)
    vols = np.exp(rng.uniform(np.log(0.01), np.log(1.0), n))
    cov = equicorrelated_cov(vols, rho)
    w = equal_risk_contribution(cov)
    np.testing.assert_allclose(w.to_numpy(), inverse_volatility(cov).to_numpy(), rtol=1e-11)


def test_erc_diagonal_cov_closed_forms():
    """Zero correlations: w_i²σ_i² ∝ b_i, so w ∝ sqrt(b)/σ (inverse volatility for b = 1/N)."""
    vols = np.array([0.05, 0.1, 0.2, 0.4])
    cov = np.diag(vols**2)
    np.testing.assert_allclose(
        equal_risk_contribution(cov).to_numpy(), inverse_volatility(cov).to_numpy(), rtol=1e-12
    )
    b = np.array([0.1, 0.2, 0.3, 0.4])
    expected = np.sqrt(b) / vols
    np.testing.assert_allclose(
        equal_risk_contribution(cov, b).to_numpy(), expected / expected.sum(), rtol=1e-12
    )


@pytest.mark.parametrize("rho", [-0.5, 0.5, 0.9])
def test_erc_three_asset_block_closed_form(rho):
    """Assets 1-2 correlated ρ, asset 3 independent. In risk units u_i = w_iσ_i:
    u1(u1 + ρu2) = u2(u2 + ρu1) ⇒ u1 = u2, and u3² = u1²(1 + ρ)."""
    vols = np.array([0.1, 0.2, 0.3])
    corr = np.array([[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov = corr * np.outer(vols, vols)
    u = np.array([1.0, 1.0, np.sqrt(1.0 + rho)])
    expected = u / vols
    np.testing.assert_allclose(
        equal_risk_contribution(cov).to_numpy(), expected / expected.sum(), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# equal_risk_contribution: optimality, references, hard cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5, 10, 25, 50])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_erc_random_covariances_equal_contributions(n, seed):
    rng = np.random.default_rng(100 * n + seed)
    cov = random_cov(n, rng)
    w = equal_risk_contribution(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert (w > 0).all()
    shares = risk_contributions(w, cov, relative=True).to_numpy()
    np.testing.assert_allclose(shares, 1.0 / n, atol=1e-11)


@pytest.mark.parametrize("n", [5, 20, 50])
@pytest.mark.parametrize("seed", [0, 1])
def test_risk_budgeting_random_budgets(n, seed):
    rng = np.random.default_rng(1000 * n + seed)
    cov = random_cov(n, rng)
    b = rng.dirichlet(np.full(n, 0.5))
    b[0] = 1e-6  # a tiny budget forces a long way from the start point
    b /= b.sum()
    w = equal_risk_contribution(cov, b)
    assert (w > 0).all()
    np.testing.assert_allclose(risk_contributions(w, cov, relative=True).to_numpy(), b, atol=1e-11)


@pytest.mark.parametrize("n", [5, 20, 50])
@pytest.mark.parametrize("budgets", ["equal", "random"])
def test_erc_highly_correlated_and_heterogeneous_vols(n, budgets):
    """ρ ≈ 0.99 (condition number ~1e3-1e4) and vols from 0.1% to 1000%."""
    rng = np.random.default_rng(n)
    cov = highly_correlated_cov(n, rng)
    b = np.full(n, 1.0 / n) if budgets == "equal" else rng.dirichlet(np.ones(n))
    w = equal_risk_contribution(cov, b, max_iter=50)
    np.testing.assert_allclose(variance_shares(w.to_numpy(), cov), b, atol=1e-11)


def test_erc_matches_independent_reference_solvers():
    rng = np.random.default_rng(42)
    n = 8
    cov = random_cov(n, rng)
    b = rng.dirichlet(np.ones(n))
    w = equal_risk_contribution(cov, b).to_numpy()
    np.testing.assert_allclose(w, ccd_risk_budgeting(cov, b), rtol=1e-10)
    np.testing.assert_allclose(w, scipy_risk_budgeting(cov, b), rtol=1e-9)
    w_eq = equal_risk_contribution(cov).to_numpy()
    np.testing.assert_allclose(w_eq, ccd_risk_budgeting(cov, np.ones(n)), rtol=1e-10)


def test_erc_satisfies_spinu_optimality_condition():
    """y = w/σ_p is the minimizer of ½y'Σy - Σ b ln y: gradient Σy - b/y = 0 there,
    and (strict convexity) every feasible perturbation increases the objective."""
    rng = np.random.default_rng(9)
    n = 12
    cov = random_cov(n, rng)
    b = rng.dirichlet(np.ones(n))
    w = equal_risk_contribution(cov, b).to_numpy()
    y = w / np.sqrt(w @ cov @ w)
    grad = cov @ y - b / y
    np.testing.assert_allclose(grad * y, 0.0, atol=1e-12)

    def f(v: np.ndarray) -> float:
        return 0.5 * v @ cov @ v - b @ np.log(v)

    for _ in range(20):
        v = y * np.exp(0.01 * rng.standard_normal(n))
        assert f(v) > f(y)


@pytest.mark.parametrize("seed", range(5))
def test_erc_volatility_between_min_variance_and_equal_weight(seed):
    """Maillard et al. (2010): σ_MV <= σ_ERC <= σ_1/N (here σ_MV of the unconstrained
    minimum-variance portfolio, which is a lower bound of the long-only one)."""
    rng = np.random.default_rng(seed)
    n = 10
    cov = random_cov(n, rng)
    ones = np.ones(n)
    gmv = np.linalg.solve(cov, ones)
    gmv /= ones @ gmv
    w = equal_risk_contribution(cov).to_numpy()
    ew = ones / n

    def vol(v: np.ndarray) -> float:
        return float(np.sqrt(v @ cov @ v))

    assert vol(gmv) <= vol(w) <= vol(ew)


# ---------------------------------------------------------------------------
# equal_risk_contribution: damped Newton line search and tiny budgets
# ---------------------------------------------------------------------------


def _newton_step(corr: np.ndarray, b: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Newton step of Spinu's objective, computed independently."""
    grad = corr @ x - b / x
    hess = corr + np.diag(b / x**2)
    return grad, np.linalg.solve(hess, grad)


def _reference_backtracking(
    corr: np.ndarray, b: np.ndarray, x: np.ndarray, step: np.ndarray, c: float = 1e-4
) -> tuple[float, float]:
    """Boyd & Vandenberghe Alg. 9.2 with f evaluated directly (fine when f changes a lot).

    Returns (α after the positivity halvings only, final Armijo α).
    """
    slope = float((corr @ x - b / x) @ step)
    alpha = 1.0
    while not (x - alpha * step > 0).all():
        alpha *= 0.5
    feasible = alpha
    f_x = spinu_objective(corr, b, x)
    while spinu_objective(corr, b, x - alpha * step) > f_x - c * alpha * slope:
        alpha *= 0.5
    return feasible, alpha


@pytest.mark.parametrize(
    ("start_scale", "direction", "stretch"),
    [
        (1.0, "newton", 1.0),  # the solver's own first step: 1 Armijo halving
        (0.1, "newton", 300.0),  # overlong step away from the boundary: 4 halvings
        (1.0, "gradient", 10.0),  # positivity halvings first, then Armijo
    ],
)
def test_line_search_matches_armijo_backtracking(start_scale, direction, stretch):
    corr, b = BACKTRACK_CORR, BACKTRACK_BUDGETS
    x = start_scale * b / np.sqrt(b @ corr @ b)
    grad, newton = _newton_step(corr, b, x)
    step = stretch * (newton if direction == "newton" else grad)
    feasible, expected = _reference_backtracking(corr, b, x, step)
    assert expected < feasible  # the Armijo condition really has to backtrack here
    alpha = _line_search(corr, b, x, step, float(grad @ step))
    assert alpha == expected
    assert spinu_objective(corr, b, x - alpha * step) < spinu_objective(corr, b, x)


def test_damped_newton_backtracks_and_converges_fast():
    """From the solver's start x ∝ b the full Newton step increases f. With the
    Armijo line search the solver needs 5 Newton steps; undamped Newton needs 8.
    The result matches both independent reference solvers."""
    corr, b = BACKTRACK_CORR, BACKTRACK_BUDGETS
    x0 = b / np.sqrt(b @ corr @ b)
    _, newton = _newton_step(corr, b, x0)
    assert (x0 - newton > 0).all()
    assert spinu_objective(corr, b, x0 - newton) > spinu_objective(corr, b, x0)
    vols = np.array([0.2, 0.15, 0.3])
    cov = corr * np.outer(vols, vols)
    w = equal_risk_contribution(cov, b, max_iter=6).to_numpy()
    np.testing.assert_allclose(w, ccd_risk_budgeting(cov, b), rtol=1e-12)
    np.testing.assert_allclose(w, scipy_risk_budgeting(cov, b), rtol=1e-12)
    np.testing.assert_allclose(variance_shares(w, cov), b, atol=1e-14)


def test_armijo_test_is_exact_where_float_subtraction_fails():
    """With a budget of 1e-17 the objective changes by at most a couple of ulps
    of |f| when that asset moves, so f(x - αΔ) - f(x) in float64 is pure
    rounding (a line search comparing two float64 objective values stalls on
    such problems). The module's
    cancellation-free Armijo test must agree with the condition evaluated
    exactly (60 significant digits), for accepted and rejected steps alike."""
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.5], [0.2, 0.5, 1.0]])
    b = np.array([1e-17, 0.6, 0.4])
    # Assets 2-3 at their 2-asset solution (x_i (Cx)_i = b_i, the tiny x_1
    # changes it by ~1e-17); asset 1 then has x_1 (Cx)_1 = b_1.
    k, rho = b[1] / b[2], corr[1, 2]
    t = (-rho * (1.0 - k) + np.sqrt(rho**2 * (1.0 - k) ** 2 + 4.0 * k)) / 2.0
    x3 = np.sqrt(b[1] / (t * t + rho * t))
    x = np.array([0.0, t * x3, x3])
    x[0] = b[0] / (corr[0, 1:] @ x[1:])

    def exact_armijo(x: np.ndarray, step: np.ndarray, alpha: float, slope: float) -> bool:
        with localcontext() as ctx:
            ctx.prec = 60
            n = len(x)
            cd = [[Decimal(float(v)) for v in row] for row in corr]
            bd = [Decimal(float(v)) for v in b]

            def f(v: list[Decimal]) -> Decimal:
                quad = sum(v[i] * cd[i][j] * v[j] for i in range(n) for j in range(n))
                return quad / 2 - sum(bd[i] * v[i].ln() for i in range(n))

            xd = [Decimal(float(v)) for v in x]
            new = [xd[i] - Decimal(alpha) * Decimal(float(step[i])) for i in range(n)]
            change = f(new) - f(xd)
            # Within 2 ulps of f: a float64 difference of two objective values
            # carries no information at the scale of c α g'Δ (~1e-21 here).
            assert abs(change) < 2 * Decimal(np.finfo(float).eps) * abs(f(xd))
            return change <= -Decimal("1e-4") * Decimal(alpha) * Decimal(slope)

    outcomes = set()
    for factor in (0.3, 3.0):  # asset 1 too small / too large
        x_trial = x.copy()
        x_trial[0] *= factor
        grad, step = _newton_step(corr, b, x_trial)
        slope = float(grad @ step)
        curvature = float(step @ corr @ step)
        for alpha in (1.0, 0.5, 0.25, 0.125):
            if not (x_trial - alpha * step > 0).all():
                continue
            expected = exact_armijo(x_trial, step, alpha, slope)
            assert _armijo_holds(b, x_trial, step, alpha, curvature, slope) == expected
            outcomes.add(expected)
    assert outcomes == {True, False}


@pytest.mark.parametrize(
    ("n", "rho", "n_tiny", "tiny"),
    [(4, -0.3, 1, 1e-16), (10, -0.1, 4, 1e-15), (8, 0.6, 3, 1e-14)],
)
def test_risk_budgeting_tiny_budgets_constant_correlation(n, rho, n_tiny, tiny):
    """Budgets of 1e-16..1e-14 against the constant-correlation semi-closed form.
    (A line search that subtracts float64 objective values stalls on the first two.)"""
    vols = np.geomspace(0.05, 0.5, n)
    cov = equicorrelated_cov(vols, rho)
    b = np.arange(1.0, n + 1.0)
    b[:n_tiny] = tiny
    b /= b.sum()
    w = equal_risk_contribution(cov, b).to_numpy()
    expected = equicorrelated_budget_weights(vols, rho, b)
    # The residual tolerance is absolute, so an asset whose budget is far below
    # it is pinned down only in absolute terms when its weight is tiny too
    # (ρ >= 0); with ρ < 0 it is a hedge with a sizeable weight.
    np.testing.assert_allclose(w, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(variance_shares(w, cov), b, atol=1e-12)


def test_risk_budgeting_budgets_near_1e_14_regression():
    """A two-factor covariance with budgets floored at 1e-14 on which a line
    search that subtracts float64 objective values stalls at a risk-budget
    residual of 1.4e-9 (OptimizationError after 500 iterations). The
    assets with budgets 1e-14 and 1.4e-8 are hedges that end up with weights
    of 7% and 63%."""
    rng = np.random.default_rng(1877)
    n = int(rng.choice([3, 5, 10, 20]))
    loadings = 3.0 * rng.standard_normal((n, 2))
    cov = loadings @ loadings.T + np.diag(rng.uniform(1e-4, 1e-1, n))
    cov = 0.5 * (cov + cov.T)
    b = rng.dirichlet(np.full(n, float(rng.choice([0.05, 0.3, 1.0]))))
    b = np.maximum(b, 1e-14)
    b /= b.sum()
    assert np.sum(b < 1e-13) == 2
    w = equal_risk_contribution(cov, b).to_numpy()
    assert (w > 0).all()
    np.testing.assert_allclose(w, scipy_risk_budgeting(cov, b), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(variance_shares(w, cov), b, atol=1e-12)


# ---------------------------------------------------------------------------
# equal_risk_contribution: invariances
# ---------------------------------------------------------------------------


def test_erc_scale_invariance_daily_vs_annualized():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0003, 0.01, size=(500, 6)) @ np.triu(rng.uniform(0.2, 1, (6, 6)))
    cov_daily = np.cov(returns, rowvar=False, ddof=1)
    w = equal_risk_contribution(cov_daily).to_numpy()
    for scale in (float(TRADING_DAYS), 1.0 / TRADING_DAYS, (500 - 1) / 500, 1e-8, 1e6):
        np.testing.assert_allclose(equal_risk_contribution(cov_daily * scale), w, rtol=1e-12)
    # Scaling by powers of 4 is exact in floating point (sqrt is exact too), and the
    # solver works in correlation units, so the weights must be bit-identical.
    for scale in (4.0**5, 4.0**-7):
        np.testing.assert_array_equal(equal_risk_contribution(cov_daily * scale).to_numpy(), w)


def test_erc_per_asset_units_invariance():
    """Rescaling asset i's returns by d_i (Σ → DΣD) must rescale its weight by 1/d_i:
    w_i(Σw)_i is unchanged under (DΣD, D⁻¹w). This fails if y = D x instead of D⁻¹x."""
    rng = np.random.default_rng(12)
    n = 7
    cov = random_cov(n, rng)
    b = rng.dirichlet(np.ones(n))
    d = np.exp(rng.uniform(-3, 3, n))
    w = equal_risk_contribution(cov, b).to_numpy()
    w_scaled = equal_risk_contribution(cov * np.outer(d, d), b).to_numpy()
    expected = (w / d) / np.sum(w / d)
    np.testing.assert_allclose(w_scaled, expected, rtol=1e-11)


def test_erc_permutation_equivariance_and_labels():
    rng = np.random.default_rng(13)
    assets = [f"A{i:02d}" for i in range(9)]
    cov = pd.DataFrame(random_cov(9, rng), index=assets, columns=assets)
    b = pd.Series(rng.dirichlet(np.ones(9)), index=assets)
    w = equal_risk_contribution(cov, b)
    assert list(w.index) == assets
    perm = list(rng.permutation(assets))
    w_perm = equal_risk_contribution(cov.loc[perm, perm], b)  # budgets aligned by label
    assert list(w_perm.index) == perm
    np.testing.assert_allclose(w_perm[assets].to_numpy(), w.to_numpy(), rtol=1e-12)


def test_risk_budgets_are_normalized_and_default_is_equal():
    rng = np.random.default_rng(14)
    cov = random_cov(4, rng)
    np.testing.assert_allclose(
        equal_risk_contribution(cov, [1.0, 2.0, 3.0, 4.0]).to_numpy(),
        equal_risk_contribution(cov, [0.1, 0.2, 0.3, 0.4]).to_numpy(),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        equal_risk_contribution(cov, [7.0] * 4).to_numpy(),
        equal_risk_contribution(cov).to_numpy(),
        rtol=1e-13,
    )


# ---------------------------------------------------------------------------
# equal_risk_contribution: singular covariances, edge cases, validation
# ---------------------------------------------------------------------------


def test_erc_single_asset():
    w = equal_risk_contribution(pd.DataFrame([[0.04]], index=["only"], columns=["only"]))
    assert list(w.index) == ["only"]
    assert w.iloc[0] == 1.0


def test_erc_singular_but_solvable_covariances():
    # Perfect positive correlation: rank 1, still equicorrelated → inverse volatility.
    vols = np.array([0.1, 0.2, 0.3])
    cov = np.outer(vols, vols)
    np.testing.assert_allclose(
        equal_risk_contribution(cov).to_numpy(), inverse_volatility(cov).to_numpy(), rtol=1e-12
    )
    # Fewer observations than assets: rank-deficient sample covariance, but no
    # long-only portfolio is riskless, so the risk-budgeting portfolio exists.
    rng = np.random.default_rng(2)
    returns = 0.6 * rng.standard_normal((10, 1)) + rng.standard_normal((10, 20))
    sample = np.cov(returns, rowvar=False)
    assert np.linalg.matrix_rank(sample) < 20
    w = equal_risk_contribution(sample).to_numpy()
    np.testing.assert_allclose(variance_shares(w, sample), 1.0 / 20, atol=1e-11)


@pytest.mark.parametrize(
    ("cov", "budgets"),
    [
        # ρ = -1: the equal-risk long-only mix is riskless, so f(c·1) → -∞.
        (np.array([[1.0, -1.0], [-1.0, 1.0]]), None),
        (np.array([[1.0, -1.0], [-1.0, 1.0]]), [0.3, 0.7]),
        # Pairwise ρ = -0.5 for 3 assets: 1'C1 = 0, the equal-weight mix is riskless.
        (np.array([[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]]), [0.2, 0.3, 0.5]),
    ],
)
def test_erc_no_solution_raises_optimization_error(cov, budgets):
    with pytest.raises(OptimizationError):
        equal_risk_contribution(cov, budgets)


def test_erc_not_converged_raises_optimization_error():
    cov = highly_correlated_cov(10, np.random.default_rng(0))
    with pytest.raises(OptimizationError, match="did not converge"):
        equal_risk_contribution(cov, max_iter=1)


def test_erc_loose_tolerance_still_meets_budget_check():
    """A loose ``tol`` is capped so that the spec's 1e-8 budget check always holds."""
    cov = highly_correlated_cov(10, np.random.default_rng(1))
    for tol in (1e-2, 1e-4, 1e-12):
        w = equal_risk_contribution(cov, tol=tol).to_numpy()
        assert np.max(np.abs(variance_shares(w, cov) - 0.1)) < 1e-8


def test_erc_accepts_numpy_integer_max_iter():
    w = equal_risk_contribution(np.diag([0.04, 0.09]), max_iter=np.int64(20))
    np.testing.assert_allclose(w.to_numpy(), [0.6, 0.4], rtol=1e-12)


@pytest.mark.parametrize(
    ("cov", "kwargs", "match"),
    [
        (np.array([[1.0, 2.0], [2.0, 1.0]]), {}, "positive semi-definite"),
        (
            np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]),
            {},
            "positive semi-definite",
        ),
        (np.diag([0.04, 0.0]), {}, "strictly positive variance"),
        (np.array([[0.04, 0.01], [0.0, 0.09]]), {}, "symmetric"),
        (np.array([[0.04, np.inf], [np.inf, 0.09]]), {}, "NaN or infinite"),
        (np.eye(3), {"risk_budgets": [0.5, 0.5]}, "expected 3 values"),
        (np.eye(3), {"risk_budgets": [0.5, 0.5, 0.0]}, "strictly positive"),
        (np.eye(3), {"risk_budgets": [0.5, 0.7, -0.2]}, "strictly positive"),
        (np.eye(3), {"risk_budgets": [0.5, np.nan, 0.5]}, "NaN"),
        (np.eye(2), {"risk_budgets": pd.Series({0: 0.5, 1: 0.3, "z": 0.2})}, "not in the cov"),
        (np.eye(2), {"risk_budgets": pd.Series({0: 1.0})}, "missing"),
        (np.eye(2), {"tol": 0.0}, "tol"),
        (np.eye(2), {"tol": -1e-9}, "tol"),
        (np.eye(2), {"tol": float("nan")}, "tol"),
        (np.eye(2), {"max_iter": 0}, "max_iter"),
        (np.eye(2), {"max_iter": 2.5}, "max_iter"),
        (np.eye(2), {"max_iter": True}, "max_iter"),
        (np.eye(2), {"max_iter": None}, "max_iter"),
        (np.eye(2), {"max_iter": float("nan")}, "max_iter"),
        (np.eye(2), {"tol": None}, "tol"),
        (np.eye(2), {"tol": float("inf")}, "tol"),
    ],
)
def test_erc_validation(cov, kwargs, match):
    with pytest.raises(ValueError, match=match):
        equal_risk_contribution(cov, **kwargs)
