"""Tests for quantfin.portfolio.markowitz.

The math is checked against independent answers:

* hand-computed two-asset numbers and textbook two-asset formulas;
* Merton's (1972) A, B, C, D closed forms written here with an explicit
  matrix inverse (the module uses a centred Cholesky form instead);
* a brute-force reference for the bounded QPs that enumerates every
  lower/free/upper pattern of the bounds (3^N equality-constrained solves) and
  keeps the best feasible point;
* first-order (KKT) optimality conditions, including the signs of the
  multipliers of active bounds, on N = 30 problems with well- and
  ill-conditioned (condition number 1e6) covariances;
* identities and invariances: two-fund separation, the tangency return
  (C - B rf)/(B - A rf), max Sharpe = sqrt((μ-rf)'Σ⁻¹(μ-rf)), scale and
  permutation invariance, shifting μ and rf together.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from quantfin.portfolio import markowitz
from quantfin.portfolio._common import TRADING_DAYS, OptimizationError
from quantfin.portfolio.markowitz import (
    efficient_frontier,
    efficient_return,
    efficient_risk,
    max_sharpe,
    mean_variance_utility,
    min_variance,
    portfolio_return,
    portfolio_sharpe,
    portfolio_volatility,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def factor_cov(n: int, seed: int) -> np.ndarray:
    """Annualized covariance with equity/bond-like vols (8%-30%) and mostly positive correlation."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.08, 0.30, n)
    loadings = rng.normal(size=(n, 3)) * np.array([0.7, 0.35, 0.25])
    loadings[:, 0] = np.abs(loadings[:, 0]) + 0.2
    raw = loadings @ loadings.T + np.diag(rng.uniform(0.1, 0.5, n))
    d = np.sqrt(np.diag(raw))
    corr = raw / np.outer(d, d)
    cov = corr * np.outer(vols, vols)
    return 0.5 * (cov + cov.T)


def ill_conditioned_cov(n: int, cond: float, seed: int) -> np.ndarray:
    """Q diag(λ) Q' with λ log-spaced from 0.04 down to 0.04 / cond."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = 0.04 * np.logspace(0.0, -np.log10(cond), n)
    cov = (q * lam) @ q.T
    return 0.5 * (cov + cov.T)


def random_mu(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed + 1000).uniform(0.02, 0.12, n)


COV_CASES = {
    "well": (factor_cov(30, 1), random_mu(30, 1)),
    "ill": (ill_conditioned_cov(30, 1e6, 2), random_mu(30, 2)),
}

# Small problems for the brute-force reference.
SMALL_COV = factor_cov(5, 7)
SMALL_MU = np.array([0.05, 0.09, 0.07, 0.11, 0.04])


def merton_abcd(mu: np.ndarray, cov: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    """Merton (1972) constants computed with an explicit inverse."""
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    a = ones @ inv @ ones
    b = ones @ inv @ mu
    c = mu @ inv @ mu
    return a, b, c, a * c - b * b, inv


def brute_force_qp(
    H: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Global minimizer of ½x'Hx - c'x s.t. Ax = b, lo <= x <= hi (H positive definite).

    The optimum of a convex QP solves the equality-constrained problem for its
    own active set, so enumerating every (lower, free, upper) pattern, solving
    each with lstsq and keeping the best feasible point finds it exactly.
    """
    n, m = H.shape[0], A.shape[0]
    best_x, best_f = None, np.inf
    for pattern in itertools.product((-1, 0, 1), repeat=n):
        state = np.array(pattern)
        x = np.zeros(n)
        x[state == -1] = lo[state == -1]
        x[state == 1] = hi[state == 1]
        free = state == 0
        if free.any():
            fixed = ~free
            k = int(free.sum())
            kkt = np.block([[H[np.ix_(free, free)], A[:, free].T], [A[:, free], np.zeros((m, m))]])
            rhs = np.concatenate(
                [c[free] - H[np.ix_(free, fixed)] @ x[fixed], b - A[:, fixed] @ x[fixed]]
            )
            x[free] = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:k]
        if np.abs(A @ x - b).max() > 1e-10 or np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12):
            continue
        f = 0.5 * x @ H @ x - c @ x
        if f < best_f:
            best_x, best_f = x, f
    assert best_x is not None, "brute force found no feasible point"
    return best_x


def bounds_arrays(lo: float | np.ndarray, hi: float | np.ndarray, n: int):
    return np.broadcast_to(np.asarray(lo, float), (n,)), np.broadcast_to(
        np.asarray(hi, float), (n,)
    )


def kkt_violation(
    grad: np.ndarray, w: np.ndarray, lo: np.ndarray, hi: np.ndarray, eq_rows: np.ndarray
) -> tuple[float, float]:
    """KKT residuals of  min f(w) s.t. eq_rows @ w = const, lo <= w <= hi,  with ∇f(w) = grad.

    Stationarity: grad = eq_rows' ν + μ_lo - μ_hi with μ_lo, μ_hi >= 0 supported
    on the assets at their lower / upper bound. ν is fitted on the free assets;
    the remaining residual on bound assets is the bound multiplier, whose sign
    is checked. Both residuals are relative to max |grad|.

    At a vertex (too few free assets to pin down ν) the multipliers are not
    unique; then a bounded least-squares search looks for ANY valid
    (ν, μ_lo >= 0, μ_hi >= 0) and its residual is reported as stationarity.
    """
    at_lo = w <= lo + 1e-10
    at_hi = w >= hi - 1e-10
    free = ~(at_lo | at_hi)
    E = np.atleast_2d(eq_rows).T
    scale = float(np.abs(grad).max())
    if free.sum() < E.shape[1] or np.linalg.matrix_rank(E[free]) < E.shape[1]:
        n_eq, eye = E.shape[1], np.eye(len(w))
        M = np.hstack([E, eye[:, at_lo], -eye[:, at_hi]])
        lower = np.r_[np.full(n_eq, -np.inf), np.zeros(int(at_lo.sum() + at_hi.sum()))]
        fit = optimize.lsq_linear(M, grad, bounds=(lower, np.inf), method="bvls")
        return float(np.abs(M @ fit.x - grad).max()) / scale, 0.0
    nu = np.linalg.lstsq(E[free], grad[free], rcond=None)[0]
    mult = grad - E @ nu
    stationarity = float(np.abs(mult[free]).max(initial=0.0)) / scale
    wrong_sign = max(float(-mult[at_lo].min(initial=0.0)), float(mult[at_hi].max(initial=0.0)), 0.0)
    return stationarity, wrong_sign / scale


def sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> float:
    return float((w @ mu - rf) / np.sqrt(w @ cov @ w))


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------


def test_portfolio_statistics_hand_computed_long_short():
    # σ = (20%, 10%), ρ = -0.3  =>  cov12 = -0.006
    # w = (1.5, -0.5): σ² = 2.25·0.04 + 0.25·0.01 + 2·1.5·(-0.5)·(-0.006) = 0.1015
    cov = np.array([[0.04, -0.006], [-0.006, 0.01]])
    w = np.array([1.5, -0.5])
    mu = np.array([0.08, 0.03])
    assert portfolio_return(w, mu) == pytest.approx(0.105, abs=1e-15)
    assert portfolio_volatility(w, cov) == pytest.approx(math.sqrt(0.1015), abs=1e-15)
    expected = (0.105 - 0.02) / math.sqrt(0.1015)
    assert portfolio_sharpe(w, mu, cov, rf=0.02) == pytest.approx(expected, abs=1e-14)


def test_portfolio_statistics_align_series_by_label():
    cov = pd.DataFrame(SMALL_COV, index=list("ABCDE"), columns=list("ABCDE"))
    mu = pd.Series(SMALL_MU, index=list("ABCDE"))
    w = pd.Series([0.1, 0.2, 0.3, 0.4, 0.0], index=list("ABCDE"))
    shuffled_w, shuffled_mu = w[list("EDCBA")], mu[list("CAEBD")]
    assert portfolio_return(shuffled_w, shuffled_mu) == pytest.approx(w @ mu, abs=1e-15)
    assert portfolio_volatility(shuffled_w, cov) == pytest.approx(
        math.sqrt(w.to_numpy() @ SMALL_COV @ w.to_numpy()), abs=1e-15
    )
    assert portfolio_sharpe(shuffled_w, shuffled_mu, cov, 0.01) == pytest.approx(
        sharpe(w.to_numpy(), SMALL_MU, SMALL_COV, 0.01), abs=1e-13
    )
    # An unlabelled covariance borrows the labels of a Series argument.
    assert portfolio_volatility(w, SMALL_COV) == pytest.approx(
        portfolio_volatility(w.to_numpy(), SMALL_COV), abs=1e-15
    )


def test_portfolio_statistics_reject_misaligned_inputs():
    cov = pd.DataFrame(SMALL_COV, index=list("ABCDE"), columns=list("ABCDE"))
    with pytest.raises(ValueError, match="missing"):
        portfolio_volatility(pd.Series([0.5, 0.5], index=["A", "B"]), cov)
    with pytest.raises(ValueError, match="not in the asset universe"):
        portfolio_volatility(pd.Series(0.2, index=list("ABCDEF")), cov)
    with pytest.raises(ValueError, match="expected 5 values"):
        portfolio_volatility(np.ones(4) / 4, cov)
    with pytest.raises(ValueError, match="NaN"):
        portfolio_return([0.5, np.nan], [0.1, 0.2])
    with pytest.raises(ValueError, match="rf"):
        portfolio_sharpe([1.0], [0.1], [[0.04]], rf=float("nan"))


def test_portfolio_sharpe_riskless_portfolio_has_no_warnings():
    cov = [[0.0, 0.0], [0.0, 0.04]]
    assert portfolio_sharpe([1.0, 0.0], [0.05, 0.1], cov, rf=0.01) == math.inf
    assert portfolio_sharpe([1.0, 0.0], [0.00, 0.1], cov, rf=0.01) == -math.inf
    assert math.isnan(portfolio_sharpe([1.0, 0.0], [0.01, 0.1], cov, rf=0.01))


def test_portfolio_volatility_rejects_indefinite_covariance():
    indefinite = np.array([[0.04, 0.05], [0.05, 0.04]])  # eigenvalues 0.09 and -0.01
    with pytest.raises(ValueError, match="positive semi-definite"):
        portfolio_volatility([1.0, -1.0], indefinite)


# ---------------------------------------------------------------------------
# Closed forms (bounds=None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["well", "ill"])
def test_min_variance_closed_form_matches_inverse_formula(case):
    cov, _ = COV_CASES[case]
    w = min_variance(cov, bounds=None).to_numpy()
    inv = np.linalg.inv(cov)
    expected = inv @ np.ones(30) / (np.ones(30) @ inv @ np.ones(30))
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-9 * np.abs(expected).max())
    # Every asset has the same covariance with the MVP, equal to its variance:
    # Σ w_mvp = σ²_mvp 1 (first-order condition of the budget-only problem).
    np.testing.assert_allclose(cov @ w, (w @ cov @ w) * np.ones(30), rtol=1e-8)


def test_min_variance_two_asset_textbook_formula():
    s1, s2, rho = 0.2, 0.1, 0.35
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    np.testing.assert_allclose(min_variance(cov, bounds=None), [w1, 1 - w1], atol=1e-15)
    # The interior solution is long-only, so the long-only optimizer agrees.
    np.testing.assert_allclose(min_variance(cov), [w1, 1 - w1], atol=1e-12)


@pytest.mark.parametrize("rf", [0.0, 0.03])
@pytest.mark.parametrize("case", ["well", "ill"])
def test_tangency_closed_form_and_max_sharpe_identity(case, rf):
    cov, mu = COV_CASES[case]
    w = max_sharpe(mu, cov, rf=rf, bounds=None).to_numpy()
    inv = np.linalg.inv(cov)
    z = inv @ (mu - rf)
    np.testing.assert_allclose(w, z / z.sum(), rtol=0, atol=1e-8 * np.abs(w).max())
    # The tangency Sharpe ratio is the slope of the capital market line:
    # sqrt((μ - rf)'Σ⁻¹(μ - rf)).
    assert sharpe(w, mu, cov, rf) == pytest.approx(math.sqrt((mu - rf) @ z), rel=1e-9)


def test_tangency_return_and_variance_formulas():
    # Tangency return (C - B rf)/(B - A rf) and variance (C - 2B rf + A rf²)/(B - A rf)²
    # (e.g. Campbell, Lo & MacKinlay 1997, ch. 5); a sign error on rf breaks both.
    cov, mu, rf = SMALL_COV, SMALL_MU, 0.02
    a, b, c, _, _ = merton_abcd(mu, cov)
    w = max_sharpe(mu, cov, rf=rf, bounds=None).to_numpy()
    assert w @ mu == pytest.approx((c - b * rf) / (b - a * rf), rel=1e-12)
    assert w @ cov @ w == pytest.approx((c - 2 * b * rf + a * rf**2) / (b - a * rf) ** 2, rel=1e-12)
    w_zero = max_sharpe(mu, cov, rf=0.0, bounds=None).to_numpy()
    assert np.abs(w - w_zero).max() > 1e-3  # rf actually enters the answer


def test_tangency_two_asset_textbook_formula():
    s1, s2, rho, rf = 0.25, 0.12, 0.2, 0.01
    e1, e2 = 0.10 - rf, 0.05 - rf
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    num = e1 * s2**2 - e2 * rho * s1 * s2
    den = e1 * s2**2 + e2 * s1**2 - (e1 + e2) * rho * s1 * s2
    np.testing.assert_allclose(
        max_sharpe([0.10, 0.05], cov, rf=rf, bounds=None), [num / den, 1 - num / den], atol=1e-14
    )


def test_tangency_undefined_when_rf_above_mvp_return():
    cov, mu = SMALL_COV, SMALL_MU
    r_mvp = float(min_variance(cov, bounds=None) @ mu)
    with pytest.raises(ValueError, match="tangency portfolio is undefined"):
        max_sharpe(mu, cov, rf=r_mvp + 0.01, bounds=None)
    with pytest.raises(ValueError, match="tangency portfolio is undefined"):
        max_sharpe(mu, cov, rf=r_mvp, bounds=None)


@pytest.mark.parametrize("case", ["well", "ill"])
def test_efficient_return_matches_merton_abcd(case):
    cov, mu = COV_CASES[case]
    a, b, c, d, inv = merton_abcd(mu, cov)
    for target in (0.03, b / a, 0.15):
        w = efficient_return(mu, cov, target, bounds=None).to_numpy()
        expected = ((c - b * target) * inv @ np.ones(30) + (a * target - b) * inv @ mu) / d
        np.testing.assert_allclose(w, expected, rtol=0, atol=1e-7 * np.abs(expected).max())
        assert w @ mu == pytest.approx(target, abs=1e-12)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        # The frontier hyperbola σ² = (A r² - 2B r + C)/D.
        assert w @ cov @ w == pytest.approx((a * target**2 - 2 * b * target + c) / d, rel=1e-7)


def test_two_fund_separation():
    cov, mu = COV_CASES["well"]
    w1 = efficient_return(mu, cov, 0.06, bounds=None).to_numpy()
    w2 = efficient_return(mu, cov, 0.14, bounds=None).to_numpy()
    for target in (0.02, 0.09, 0.25):
        lam = (target - 0.14) / (0.06 - 0.14)
        np.testing.assert_allclose(
            efficient_return(mu, cov, target, bounds=None).to_numpy(),
            lam * w1 + (1 - lam) * w2,
            atol=1e-12,
        )


def test_efficient_return_beats_feasible_perturbations():
    cov, mu = SMALL_COV, SMALL_MU
    w = efficient_return(mu, cov, 0.08, bounds=None).to_numpy()
    # Directions that keep both the budget and the return: null space of [1; μ].
    _, _, vt = np.linalg.svd(np.vstack([np.ones(5), mu]))
    null = vt[2:]
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = rng.normal(size=3) @ null
        assert (w + 0.05 * d) @ cov @ (w + 0.05 * d) > w @ cov @ w


def test_mean_variance_utility_closed_form_vs_lagrangian_system():
    cov, mu = COV_CASES["well"]
    for delta in (0.5, 3.0, 25.0):
        # Solve the first-order conditions [δΣ 1; 1' 0][w; λ] = [μ; 1] directly.
        kkt = np.block([[delta * cov, np.ones((30, 1))], [np.ones((1, 30)), np.zeros((1, 1))]])
        expected = np.linalg.solve(kkt, np.concatenate([mu, [1.0]]))[:30]
        w = mean_variance_utility(mu, cov, delta, bounds=None).to_numpy()
        np.testing.assert_allclose(w, expected, atol=1e-11)
        # Expected return B/A + (D/A)/δ.
        a, b, _, d, _ = merton_abcd(mu, cov)
        assert w @ mu == pytest.approx(b / a + d / a / delta, rel=1e-10)


def test_utility_weights_are_affine_in_mu_with_inverse_covariance_sensitivity():
    # The key lesson in one identity: dw/dμ = (Σ⁻¹ - Σ⁻¹11'Σ⁻¹/A)/δ, whose norm
    # is about 1/(δ λ_min). With an ill-conditioned Σ, a tiny error in μ along
    # a low-variance direction moves the "optimal" weights enormously.
    cov, mu = COV_CASES["ill"]
    delta = 5.0
    inv = np.linalg.inv(cov)
    one = np.ones(30)
    sens = (inv - np.outer(inv @ one, inv @ one) / (one @ inv @ one)) / delta
    _, eigvec = np.linalg.eigh(cov)
    bump = 1e-4 * eigvec[:, 0]  # a 1bp-sized error along the lowest-variance eigenvector
    w0 = mean_variance_utility(mu, cov, delta, bounds=None).to_numpy()
    w1 = mean_variance_utility(mu + bump, cov, delta, bounds=None).to_numpy()
    # Differences of O(100) weights from a cond-1e6 system: compare relative to the move.
    expected = sens @ bump
    np.testing.assert_allclose(w1 - w0, expected, rtol=0, atol=1e-7 * np.abs(expected).max())
    assert np.abs(w1 - w0).sum() > 100.0  # a 0.01% error becomes > 10000% of turnover


def test_utility_large_risk_aversion_tends_to_min_variance():
    cov, mu = SMALL_COV, SMALL_MU
    np.testing.assert_allclose(
        mean_variance_utility(mu, cov, 1e9, bounds=None), min_variance(cov, bounds=None), atol=1e-9
    )
    np.testing.assert_allclose(mean_variance_utility(mu, cov, 1e9), min_variance(cov), atol=1e-9)


@pytest.mark.parametrize("case", ["well", "ill"])
def test_efficient_risk_unbounded_inverts_the_frontier(case):
    cov, mu = COV_CASES[case]
    a, b, _, d, _ = merton_abcd(mu, cov)
    sigma_mvp = math.sqrt(1 / a)
    for target in (sigma_mvp * 1.5, 0.3):
        w = efficient_risk(mu, cov, target, bounds=None).to_numpy()
        assert math.sqrt(w @ cov @ w) == pytest.approx(target, rel=1e-9)
        # Upper branch: r = B/A + sqrt(D (A σ² - 1)) / A.
        expected_r = b / a + math.sqrt(d * (a * target**2 - 1)) / a
        assert w @ mu == pytest.approx(expected_r, rel=1e-9)
        np.testing.assert_allclose(
            w,
            efficient_return(mu, cov, expected_r, bounds=None),
            atol=1e-8 * np.abs(w).max(),
        )
    np.testing.assert_allclose(
        efficient_risk(mu, cov, sigma_mvp, bounds=None),
        min_variance(cov, bounds=None),
        atol=1e-7 * np.abs(min_variance(cov, bounds=None)).max(),
    )


# ---------------------------------------------------------------------------
# Bounded solvers: agree with closed forms when the bounds are inactive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["well", "ill"])
@pytest.mark.parametrize(
    "solve",
    [
        pytest.param(lambda mu, cov, b: min_variance(cov, bounds=b), id="min_variance"),
        pytest.param(lambda mu, cov, b: max_sharpe(mu, cov, 0.01, bounds=b), id="max_sharpe"),
        pytest.param(lambda mu, cov, b: efficient_return(mu, cov, 0.1, bounds=b), id="eff_ret"),
        pytest.param(lambda mu, cov, b: mean_variance_utility(mu, cov, 4.0, bounds=b), id="util"),
    ],
)
def test_bounded_solver_equals_closed_form_when_bounds_inactive(case, solve):
    cov, mu = COV_CASES[case]
    exact = solve(mu, cov, None).to_numpy()
    loose = (float(exact.min()) - 1.0, float(exact.max()) + 1.0)
    bounded = solve(mu, cov, loose).to_numpy()
    np.testing.assert_allclose(bounded, exact, rtol=0, atol=1e-8 * max(1.0, np.abs(exact).max()))


@pytest.mark.parametrize("case", ["well", "ill"])
def test_bounded_efficient_risk_equals_closed_form_when_bounds_inactive(case):
    cov, mu = COV_CASES[case]
    target = 1.5 * portfolio_volatility(min_variance(cov, bounds=None), cov)
    exact = efficient_risk(mu, cov, target, bounds=None).to_numpy()
    loose = (float(exact.min()) - 1.0, float(exact.max()) + 1.0)
    bounded = efficient_risk(mu, cov, target, bounds=loose).to_numpy()
    np.testing.assert_allclose(bounded, exact, rtol=0, atol=1e-7 * max(1.0, np.abs(exact).max()))


def test_all_infinite_bounds_are_the_unbounded_problem():
    cov, mu = SMALL_COV, SMALL_MU
    unbounded = max_sharpe(mu, cov, bounds=None)
    pd.testing.assert_series_equal(max_sharpe(mu, cov, bounds=(None, None)), unbounded)
    pd.testing.assert_series_equal(max_sharpe(mu, cov, bounds=[(None, None)] * 5), unbounded)


def test_open_upper_bound_equals_long_only():
    # With w >= 0 and 1'w = 1, an upper bound of 1 (or none) is redundant.
    cov, mu = COV_CASES["well"]
    np.testing.assert_allclose(min_variance(cov, (0, None)), min_variance(cov), atol=1e-12)
    np.testing.assert_allclose(max_sharpe(mu, cov, 0, (0, None)), max_sharpe(mu, cov), atol=1e-10)
    np.testing.assert_allclose(
        efficient_return(mu, cov, 0.09, (0, None)), efficient_return(mu, cov, 0.09), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Bounded solvers: KKT conditions (stationarity + multiplier signs)
# ---------------------------------------------------------------------------

BOUND_CASES = [(0.0, 1.0), (-0.1, 0.25)]


@pytest.mark.parametrize("bounds", BOUND_CASES, ids=["long_only", "box"])
@pytest.mark.parametrize("case", ["well", "ill"])
def test_min_variance_kkt(case, bounds):
    cov, _ = COV_CASES[case]
    w = min_variance(cov, bounds).to_numpy()
    lo, hi = bounds_arrays(*bounds, 30)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(w >= lo) and np.all(w <= hi)
    active = (w <= lo + 1e-10) | (w >= hi - 1e-10)
    assert active.any(), "test needs active bounds"
    stationarity, wrong_sign = kkt_violation(cov @ w, w, lo, hi, np.ones(30))
    assert stationarity < 1e-9
    assert wrong_sign < 1e-9
    # Direct check of optimality: random small feasible moves, including moving
    # assets off their active bounds into the interior, never lower the variance.
    rng = np.random.default_rng(5)
    at_lo, at_hi = w <= lo + 1e-10, w >= hi - 1e-10
    free = ~(at_lo | at_hi)
    checked = 0
    for _ in range(200):
        d = rng.normal(size=30)
        d[at_lo], d[at_hi] = np.abs(d[at_lo]), -np.abs(d[at_hi])
        d[free] -= d.sum() / free.sum()  # keep the budget
        cand = w + 1e-4 * d / np.abs(d).max()
        if np.all(cand >= lo) and np.all(cand <= hi):
            checked += 1
            assert cand @ cov @ cand >= w @ cov @ w
    assert checked > 50


@pytest.mark.parametrize("bounds", BOUND_CASES, ids=["long_only", "box"])
@pytest.mark.parametrize("case", ["well", "ill"])
def test_efficient_return_kkt(case, bounds):
    cov, mu = COV_CASES[case]
    lo, hi = bounds_arrays(*bounds, 30)
    w_mvp = min_variance(cov, bounds).to_numpy()
    top = efficient_frontier(mu, cov, 2, bounds)["return"].iloc[-1]
    for frac in (0.2, 0.6, 0.9):
        target = float(w_mvp @ mu + frac * (top - w_mvp @ mu))
        w = efficient_return(mu, cov, target, bounds).to_numpy()
        assert w @ mu == pytest.approx(target, abs=1e-12)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(w >= lo) and np.all(w <= hi)
        stationarity, wrong_sign = kkt_violation(cov @ w, w, lo, hi, np.vstack([np.ones(30), mu]))
        assert stationarity < 1e-9
        assert wrong_sign < 1e-9


@pytest.mark.parametrize("bounds", BOUND_CASES, ids=["long_only", "box"])
@pytest.mark.parametrize("case", ["well", "ill"])
def test_mean_variance_utility_kkt(case, bounds):
    cov, mu = COV_CASES[case]
    lo, hi = bounds_arrays(*bounds, 30)
    for delta in (1.0, 8.0):
        w = mean_variance_utility(mu, cov, delta, bounds).to_numpy()
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        stationarity, wrong_sign = kkt_violation(delta * cov @ w - mu, w, lo, hi, np.ones(30))
        assert stationarity < 1e-9
        assert wrong_sign < 1e-9


@pytest.mark.parametrize("bounds", BOUND_CASES, ids=["long_only", "box"])
@pytest.mark.parametrize("case", ["well", "ill"])
def test_max_sharpe_kkt(case, bounds):
    # For f = -Sharpe, ∇f ∝ Σw - (σ²/(e'w)) e with a positive factor
    # (e = μ - rf), so KKT can be checked on that vector.
    cov, mu = COV_CASES[case]
    rf = 0.02
    lo, hi = bounds_arrays(*bounds, 30)
    w = max_sharpe(mu, cov, rf, bounds).to_numpy()
    e = mu - rf
    grad = cov @ w - (w @ cov @ w) / (e @ w) * e
    stationarity, wrong_sign = kkt_violation(grad, w, lo, hi, np.ones(30))
    assert stationarity < 1e-9
    assert wrong_sign < 1e-9


# ---------------------------------------------------------------------------
# Bounded solvers vs brute-force enumeration of active sets (N = 5)
# ---------------------------------------------------------------------------

SMALL_BOUNDS = [
    (0.0, 1.0),
    (-0.3, 0.5),
    [(0.0, 0.4), (0.1, 0.6), (-0.2, 1.0), (0.0, 0.3), (0.05, 1.0)],
]


def small_bound_arrays(bounds) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bounds, tuple):
        return bounds_arrays(*bounds, 5)
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


@pytest.mark.parametrize("bounds", SMALL_BOUNDS, ids=["long_only", "box", "per_asset"])
def test_min_variance_matches_brute_force(bounds):
    lo, hi = small_bound_arrays(bounds)
    ref = brute_force_qp(SMALL_COV, np.zeros(5), np.ones((1, 5)), np.ones(1), lo, hi)
    np.testing.assert_allclose(min_variance(SMALL_COV, bounds), ref, atol=1e-12)


@pytest.mark.parametrize("bounds", SMALL_BOUNDS, ids=["long_only", "box", "per_asset"])
def test_efficient_return_matches_brute_force(bounds):
    lo, hi = small_bound_arrays(bounds)
    A = np.vstack([np.ones(5), SMALL_MU])
    for target in (0.05, 0.075, 0.09):
        ref = brute_force_qp(SMALL_COV, np.zeros(5), A, np.array([1.0, target]), lo, hi)
        np.testing.assert_allclose(
            efficient_return(SMALL_MU, SMALL_COV, target, bounds), ref, atol=1e-12
        )


@pytest.mark.parametrize("bounds", SMALL_BOUNDS, ids=["long_only", "box", "per_asset"])
def test_mean_variance_utility_matches_brute_force(bounds):
    lo, hi = small_bound_arrays(bounds)
    for delta in (0.5, 3.0, 20.0):
        ref = brute_force_qp(delta * SMALL_COV, SMALL_MU, np.ones((1, 5)), np.ones(1), lo, hi)
        np.testing.assert_allclose(
            mean_variance_utility(SMALL_MU, SMALL_COV, delta, bounds), ref, atol=1e-12
        )


def brute_force_frontier_vol(r: float, lo: np.ndarray, hi: np.ndarray) -> tuple[float, np.ndarray]:
    A = np.vstack([np.ones(5), SMALL_MU])
    w = brute_force_qp(SMALL_COV, np.zeros(5), A, np.array([1.0, r]), lo, hi)
    return math.sqrt(w @ SMALL_COV @ w), w


@pytest.mark.parametrize("bounds", SMALL_BOUNDS[:2], ids=["long_only", "box"])
def test_max_sharpe_matches_scalar_search_on_brute_force_frontier(bounds):
    # Independent route: maximize (r - rf)/σ(r) over the exact bounded frontier
    # (σ from the brute-force QP) with a scalar Brent search. The Sharpe ratio
    # is quasi-concave along the efficient branch, so the search is global.
    rf = 0.01
    lo, hi = small_bound_arrays(bounds)
    w_mvp = brute_force_qp(SMALL_COV, np.zeros(5), np.ones((1, 5)), np.ones(1), lo, hi)
    r_top = efficient_frontier(SMALL_MU, SMALL_COV, 2, bounds)["return"].iloc[-1]
    res = optimize.minimize_scalar(
        lambda r: -(r - rf) / brute_force_frontier_vol(r, lo, hi)[0],
        bounds=(float(w_mvp @ SMALL_MU), float(r_top)),
        method="bounded",
        options={"xatol": 1e-12},
    )
    _, w_ref = brute_force_frontier_vol(res.x, lo, hi)
    w = max_sharpe(SMALL_MU, SMALL_COV, rf, bounds).to_numpy()
    assert sharpe(w, SMALL_MU, SMALL_COV, rf) >= -res.fun - 1e-12
    assert sharpe(w, SMALL_MU, SMALL_COV, rf) == pytest.approx(-res.fun, rel=1e-10)
    np.testing.assert_allclose(w, w_ref, atol=1e-6)


@pytest.mark.parametrize("bounds", SMALL_BOUNDS[:2], ids=["long_only", "box"])
def test_efficient_risk_matches_brute_force(bounds):
    lo, hi = small_bound_arrays(bounds)
    w_mvp = min_variance(SMALL_COV, bounds).to_numpy()
    r_top = float(efficient_frontier(SMALL_MU, SMALL_COV, 2, bounds)["return"].iloc[-1])
    target = 0.5 * (
        math.sqrt(w_mvp @ SMALL_COV @ w_mvp) + brute_force_frontier_vol(r_top, lo, hi)[0]
    )
    r_ref = optimize.brentq(
        lambda r: brute_force_frontier_vol(r, lo, hi)[0] - target,
        float(w_mvp @ SMALL_MU),
        r_top,
        xtol=1e-14,
    )
    w = efficient_risk(SMALL_MU, SMALL_COV, target, bounds).to_numpy()
    assert w @ SMALL_MU == pytest.approx(r_ref, abs=1e-11)
    assert math.sqrt(w @ SMALL_COV @ w) == pytest.approx(target, rel=1e-10)
    np.testing.assert_allclose(w, brute_force_frontier_vol(r_ref, lo, hi)[1], atol=1e-8)


# ---------------------------------------------------------------------------
# Maximum Sharpe ratio (bounded) properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["well", "ill"])
def test_bounded_tangency_is_utility_fixed_point_and_on_frontier(case):
    # The KKT conditions of max Sharpe are those of max w'μ - (δ/2)w'Σw with
    # δ* = (μ - rf)'w / w'Σw, so the tangency must reproduce itself; it is also
    # the minimum-variance portfolio for its own return.
    cov, mu = COV_CASES[case]
    rf = 0.015
    w = max_sharpe(mu, cov, rf).to_numpy()
    delta = (w @ mu - rf) / (w @ cov @ w)
    np.testing.assert_allclose(mean_variance_utility(mu, cov, delta), w, atol=1e-8)
    np.testing.assert_allclose(efficient_return(mu, cov, float(w @ mu)), w, atol=1e-8)


def test_bounded_tangency_beats_random_and_frontier_portfolios():
    cov, mu = COV_CASES["well"]
    rf = 0.02
    w = max_sharpe(mu, cov, rf).to_numpy()
    best = sharpe(w, mu, cov, rf)
    rng = np.random.default_rng(11)
    for alpha in (0.05, 1.0):
        samples = rng.dirichlet(np.full(30, alpha), size=5000)
        sharpes = (samples @ mu - rf) / np.sqrt(np.einsum("ij,jk,ik->i", samples, cov, samples))
        assert sharpes.max() < best
    frontier = efficient_frontier(mu, cov, 40, rf=rf)
    assert frontier["sharpe"].max() <= best + 1e-12
    assert frontier["sharpe"].max() > best - 0.02  # a dense frontier gets close


def test_max_sharpe_rf_shift_invariance():
    # Only μ - rf matters: shifting every expected return and rf together is a no-op.
    cov, mu = COV_CASES["well"]
    for bounds in [(0.0, 1.0), (-0.2, 0.4), None]:
        np.testing.assert_allclose(
            max_sharpe(mu + 0.05, cov, rf=0.06, bounds=bounds),
            max_sharpe(mu, cov, rf=0.01, bounds=bounds),
            atol=1e-9,
        )


def test_max_sharpe_requires_a_portfolio_beating_rf():
    cov, mu = SMALL_COV, SMALL_MU
    with pytest.raises(ValueError, match="above rf"):
        max_sharpe(mu, cov, rf=0.2)
    with pytest.raises(ValueError, match="above rf"):
        max_sharpe(mu, cov, rf=float(mu.max()))  # equal to rf is not above it
    # With shorting, a portfolio can beat rf even though no single asset does:
    # w = (-1, 2) on two assets earns 2·0.10 - 0.06 = 0.14 > rf = 0.11.
    two = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = max_sharpe([0.06, 0.10], two, rf=0.11, bounds=(-1.0, 2.0))
    assert w @ np.array([0.06, 0.10]) > 0.11
    np.testing.assert_allclose(w, [-1.0, 2.0], atol=1e-12)


def test_max_sharpe_with_single_positive_asset_can_still_diversify():
    cov = np.array([[0.04, -0.01], [-0.01, 0.01]])
    mu = np.array([0.10, 0.00])  # the second asset only hedges
    w = max_sharpe(mu, cov, rf=0.0).to_numpy()
    assert 0.0 < w[1] < 1.0
    assert sharpe(w, mu, cov) > sharpe(np.array([1.0, 0.0]), mu, cov)


def test_max_sharpe_singular_covariance_with_arbitrage_raises():
    # A zero-variance asset that beats rf: the Sharpe ratio is unbounded.
    with pytest.raises(ValueError, match="riskless portfolio"):
        max_sharpe([0.05, 0.10], [[0.0, 0.0], [0.0, 0.04]], rf=0.01)
    # T < N sample covariance: a riskless long-only combination exists.
    rng = np.random.default_rng(3)
    sample = np.cov(rng.normal(size=(5, 8)) * 0.01, rowvar=False) * TRADING_DAYS
    with pytest.raises(ValueError, match="riskless portfolio"):
        max_sharpe(rng.uniform(0.02, 0.1, 8), sample)


def pair_bound_arrays(bounds, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) arrays for one (lo, hi) pair or a list of pairs; None means unbounded."""
    pairs = [bounds] * n if isinstance(bounds, tuple) else list(bounds)
    lo = np.array([-np.inf if a is None else a for a, _ in pairs], dtype=float)
    hi = np.array([np.inf if b is None else b for _, b in pairs], dtype=float)
    return lo, hi


HURDLE_CASES = {
    "well": COV_CASES["well"],
    # The seed on which the old start silently returned a suboptimal portfolio.
    "ill_seed12": (ill_conditioned_cov(30, 1e6, 12), random_mu(30, 12)),
}


@pytest.mark.parametrize(
    "bounds",
    [(0.0, 1.0), (-0.1, 0.25), [(None, None)] + [(0.0, None)] * 29],
    ids=["long_only", "box", "one_short_unbounded"],
)
@pytest.mark.parametrize("gap", [0.0, 1e-11])
@pytest.mark.parametrize("case", list(HURDLE_CASES))
def test_max_sharpe_with_rf_at_the_equal_weight_return(case, gap, bounds):
    # Regression: for all three bound sets the feasible centre is the
    # equal-weight portfolio, so with rf = mean(μ) its excess return is
    # rounding noise. It used to be the QP start anyway, which put
    # |y0| ~ 1e15 in the Sharpe QP: SLSQP failed ("Inequality constraints
    # incompatible") and the refinement either failed or certified a
    # suboptimal point. The answer must satisfy the max-Sharpe KKT conditions.
    cov, mu = HURDLE_CASES[case]
    rf = float(mu.mean()) - gap
    lo, hi = pair_bound_arrays(bounds, 30)
    w = max_sharpe(mu, cov, rf, bounds).to_numpy()
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(w >= lo) and np.all(w <= hi)
    e = mu - rf
    grad = cov @ w - (w @ cov @ w) / (e @ w) * e
    stationarity, wrong_sign = kkt_violation(grad, w, lo, hi, np.ones(30))
    assert stationarity < 1e-9
    assert wrong_sign < 1e-9
    # It also beats the solution for a slightly lower hurdle, evaluated at rf.
    w_alt = max_sharpe(mu, cov, rf - 1e-6, bounds).to_numpy()
    assert sharpe(w, mu, cov, rf) >= sharpe(w_alt, mu, cov, rf) - 1e-12


def test_max_sharpe_start_has_a_comfortable_excess_return():
    # Regression for the start point itself (the refinement's tolerances are
    # tested separately): on this daily-scale box problem a start whose excess
    # return is rounding noise made both SLSQP and the refinement fail, even
    # with correctly scaled refinement tolerances.
    cov, mu = factor_cov(30, 4), random_mu(30, 4)
    bounds, day = (-0.1, 0.25), 1.0 / TRADING_DAYS
    w_eq = np.full(30, 1.0 / 30)  # the feasible centre of these bounds
    rf = float(mu @ w_eq)
    # Deterministic data: the centre's daily excess return is +1e-20, pure rounding.
    assert 0.0 < (mu * day - rf * day) @ w_eq < 1e-18
    daily = max_sharpe(mu * day, cov * day, rf * day, bounds).to_numpy()
    lo, hi = bounds_arrays(*bounds, 30)
    e = mu - rf
    grad = cov @ daily - (daily @ cov @ daily) / (e @ daily) * e
    assert max(kkt_violation(grad, daily, lo, hi, np.ones(30))) < 1e-9
    np.testing.assert_allclose(daily, max_sharpe(mu, cov, rf, bounds), atol=1e-9)


# ---------------------------------------------------------------------------
# efficient_risk (bounded)
# ---------------------------------------------------------------------------


def test_efficient_risk_binding_target():
    cov, mu = COV_CASES["well"]
    frontier = efficient_frontier(mu, cov, 25)
    target = float(frontier["volatility"].iloc[12])
    w = efficient_risk(mu, cov, target).to_numpy()
    assert math.sqrt(w @ cov @ w) == pytest.approx(target, rel=1e-10)
    assert w @ mu == pytest.approx(frontier["return"].iloc[12], abs=1e-10)
    # No frontier point within the risk budget earns more.
    within = frontier[frontier["volatility"] <= target]
    assert within["return"].max() <= w @ mu + 1e-12


def test_efficient_risk_non_binding_returns_max_return_portfolio():
    cov, mu = SMALL_COV, SMALL_MU
    w = efficient_risk(mu, cov, 10.0)
    expected = np.zeros(5)
    expected[np.argmax(mu)] = 1.0
    np.testing.assert_allclose(w, expected, atol=1e-12)


def test_efficient_risk_at_minimum_volatility_is_min_variance():
    cov, mu = COV_CASES["well"]
    w_mvp = min_variance(cov)
    vol = portfolio_volatility(w_mvp, cov)
    np.testing.assert_allclose(efficient_risk(mu, cov, vol), w_mvp, atol=1e-12)
    with pytest.raises(ValueError, match="below the minimum-variance volatility"):
        efficient_risk(mu, cov, 0.99 * vol)
    vol_unbounded = portfolio_volatility(min_variance(cov, bounds=None), cov)
    assert vol_unbounded < vol  # shorting can only lower the minimum risk
    with pytest.raises(ValueError, match="below the minimum-variance volatility"):
        efficient_risk(mu, cov, 0.99 * vol_unbounded, bounds=None)


def test_efficient_risk_daily_scale_with_nearly_equal_returns():
    # Regression: the root finder on the return used an ABSOLUTE tolerance
    # (1e-15). With nearly equal returns (a flat frontier, σ changing fast per
    # unit of return) in daily units that left the volatility at the root
    # above the target by more than 1e-9, and a valid problem that solves in
    # annual units raised OptimizationError in daily units.
    mu = np.array([0.05996379996100452, 0.06006909023107993, 0.06008993356142445])
    cov = np.array(
        [
            [0.05088052498292478, 0.01133025960094482, 0.03173578846150665],
            [0.01133025960094482, 0.00905894914102235, 0.01919667630383306],
            [0.03173578846150665, 0.01919667630383306, 0.05504483748898056],
        ]
    )
    w_mvp = min_variance(cov).to_numpy()
    w_top = efficient_return(mu, cov, float(mu.max())).to_numpy()
    vol_mvp, vol_top = math.sqrt(w_mvp @ cov @ w_mvp), math.sqrt(w_top @ cov @ w_top)
    day = 1.0 / TRADING_DAYS
    for frac in (0.05, 0.3, 0.9):
        target = vol_mvp + frac * (vol_top - vol_mvp)
        annual = efficient_risk(mu, cov, target).to_numpy()
        daily = efficient_risk(mu * day, cov * day, target * math.sqrt(day)).to_numpy()
        np.testing.assert_allclose(daily, annual, atol=1e-9)
        for w, s in ((annual, 1.0), (daily, day)):
            vol = math.sqrt(w @ (cov * s) @ w)
            assert vol <= target * math.sqrt(s) * (1.0 + 1e-12)  # within the budget
            assert vol == pytest.approx(target * math.sqrt(s), rel=1e-9)  # and it binds


def test_efficient_risk_stays_within_budget_at_the_return_resolution_limit():
    # Returns 1e-9 apart: one ulp of the return (~7e-18) is ~1e-8 of the
    # attainable range, so no representable target return puts σ(r) exactly
    # on the budget and the root finder's last iterate can overshoot it by
    # more than 1e-9 (this raised OptimizationError). The closest evaluated
    # frontier point inside the budget must be returned instead.
    cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.0225, 0.003], [0.0, 0.003, 0.01]])
    mu = 0.06 + 1e-9 * np.array([1.0, 0.5, -1.0])
    w_mvp = min_variance(cov).to_numpy()
    w_top = efficient_return(mu, cov, float(mu.max())).to_numpy()
    vol_mvp, vol_top = math.sqrt(w_mvp @ cov @ w_mvp), math.sqrt(w_top @ cov @ w_top)
    for frac in (0.5, 0.75, 0.9):
        target = vol_mvp + frac * (vol_top - vol_mvp)
        w = efficient_risk(mu, cov, target).to_numpy()
        vol = math.sqrt(w @ cov @ w)
        assert vol <= target
        assert vol >= target * (1.0 - 1e-6)
        # Independent answer: on this stretch asset 2 is out, so the frontier
        # is the two-asset curve σ²(w0) = a w0² + b w0 + c11 of assets 0 and 1;
        # take the upper-branch root of σ(w0) = target.
        a = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
        b = 2.0 * (cov[0, 1] - cov[1, 1])
        w0 = (-b + math.sqrt(b * b - 4.0 * a * (cov[1, 1] - target**2))) / (2.0 * a)
        np.testing.assert_allclose(w, [w0, 1.0 - w0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["well", "ill"])
def test_frontier_structure_and_shape(case):
    cov, mu = COV_CASES[case]
    rf = 0.01
    frontier = efficient_frontier(mu, cov, n_points=20, rf=rf)
    assert list(frontier.columns[:3]) == ["return", "volatility", "sharpe"]
    assert list(frontier.columns[3:]) == list(range(30))
    assert len(frontier) == 20
    weights = frontier.iloc[:, 3:].to_numpy()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    assert weights.min() >= 0.0 and weights.max() <= 1.0
    # Endpoints: the minimum-variance portfolio and the max-return asset.
    np.testing.assert_allclose(weights[0], min_variance(cov), atol=1e-12)
    assert frontier["return"].iloc[-1] == pytest.approx(mu.max(), abs=1e-12)
    # Statistics are those of the weights.
    np.testing.assert_allclose(frontier["return"], weights @ mu, atol=1e-15)
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    np.testing.assert_allclose(frontier["volatility"], vols, rtol=1e-12)
    np.testing.assert_allclose(frontier["sharpe"], (weights @ mu - rf) / vols, rtol=1e-12)
    # Monotone: return increasing, volatility non-decreasing.
    assert np.all(np.diff(frontier["return"]) > 0)
    assert np.all(np.diff(frontier["volatility"]) >= 0)
    # Convex: the minimum variance is a convex function of the target return
    # (equally spaced returns => second differences >= 0).
    second = np.diff(frontier["volatility"].to_numpy() ** 2, n=2)
    assert second.min() >= -1e-12


def test_unbounded_frontier_is_the_merton_hyperbola():
    cov, mu = COV_CASES["well"]
    a, b, c, d, _ = merton_abcd(mu, cov)
    frontier = efficient_frontier(mu, cov, n_points=15, bounds=None)
    r = frontier["return"].to_numpy()
    np.testing.assert_allclose(
        frontier["volatility"] ** 2, (a * r**2 - 2 * b * r + c) / d, rtol=1e-9
    )
    assert r[0] == pytest.approx(b / a, abs=1e-13)
    assert r[-1] == pytest.approx(mu.max() + (mu.max() - b / a), abs=1e-13)


def test_frontier_max_sharpe_point_matches_tangency():
    cov, mu = SMALL_COV, SMALL_MU
    rf = 0.02
    tangency = max_sharpe(mu, cov, rf)
    best = portfolio_sharpe(tangency, mu, cov, rf)
    r_tan = portfolio_return(tangency, mu)
    r_mvp = portfolio_return(min_variance(cov), mu)
    half = 0.5 * min(r_tan - r_mvp, mu.max() - r_tan)
    assert half > 1e-3
    frontier = efficient_frontier(
        mu, cov, n_points=41, rf=rf, return_range=(r_tan - half, r_tan + half)
    )
    assert frontier["sharpe"].max() <= best + 1e-13
    centre = frontier.iloc[20]  # the target grid is centred on the tangency return
    assert centre["sharpe"] == pytest.approx(best, rel=1e-12)
    np.testing.assert_allclose(centre.iloc[3:].to_numpy(dtype=float), tangency, atol=1e-10)


def test_frontier_return_range_and_validation():
    cov, mu = SMALL_COV, SMALL_MU
    r_mvp = portfolio_return(min_variance(cov), mu)
    frontier = efficient_frontier(mu, cov, 5, return_range=(r_mvp, 0.1))
    np.testing.assert_allclose(frontier["return"], np.linspace(r_mvp, 0.1, 5), atol=1e-13)
    with pytest.raises(ValueError, match="inefficient branch"):
        efficient_frontier(mu, cov, 5, return_range=(r_mvp - 0.01, 0.1))
    with pytest.raises(ValueError, match="unattainable"):
        efficient_frontier(mu, cov, 5, return_range=(r_mvp, 0.2))
    with pytest.raises(ValueError, match="exceeds"):
        efficient_frontier(mu, cov, 5, return_range=(0.1, 0.09))
    # Not a (low, high) pair: a ValueError, not a TypeError from len().
    for bad_range in (0.1, np.float64(0.1), (r_mvp,), (r_mvp, 0.09, 0.1)):
        with pytest.raises(ValueError, match="return_range"):
            efficient_frontier(mu, cov, 5, return_range=bad_range)
    for bad in (1, 0, 2.5, True):
        with pytest.raises(ValueError, match="n_points"):
            efficient_frontier(mu, cov, bad)
    labelled = pd.DataFrame(cov, index=["return", *"BCDE"], columns=["return", *"BCDE"])
    with pytest.raises(ValueError, match="clash"):
        efficient_frontier(mu, labelled, 5)


def test_frontier_box_bounds_upper_end_is_max_attainable_return():
    cov, mu = SMALL_COV, SMALL_MU
    frontier = efficient_frontier(mu, cov, 6, bounds=(0.0, 0.4))
    # Greedy LP: 0.4 in 11%, 0.4 in 9%, 0.2 in 7%.
    assert frontier["return"].iloc[-1] == pytest.approx(0.4 * 0.11 + 0.4 * 0.09 + 0.2 * 0.07)
    weights = frontier.iloc[:, 3:].to_numpy()
    assert weights.max() <= 0.4


def test_frontier_with_singular_covariance_stays_monotone():
    # T < N sample covariance: a flat zero-variance stretch, then rising.
    rng = np.random.default_rng(3)
    sample = np.cov(rng.normal(size=(5, 8)) * 0.01, rowvar=False) * TRADING_DAYS
    frontier = efficient_frontier(rng.uniform(0.02, 0.1, 8), sample, 8)
    assert np.all(np.diff(frontier["volatility"]) >= 0)
    np.testing.assert_allclose(frontier.iloc[:, 3:].sum(axis=1), 1.0, atol=1e-10)
    # No row is dominated: returns strictly rise, so every row after the first
    # must carry strictly more risk than the first (the flat stretch is skipped).
    assert np.all(np.diff(frontier["return"]) > 0)
    assert np.all(frontier["volatility"].iloc[1:] > frontier["volatility"].iloc[0] + 1e-6)


# Two share classes of one asset (identical risk, correlation 1) make the
# covariance singular: every split between them is a minimum-variance portfolio,
# but only the one holding the higher-return class is efficient.
DUP_COV = np.array([[0.04, 0.04, 0.0], [0.04, 0.04, 0.0], [0.0, 0.0, 0.09]])
DUP_MU = np.array([0.05, 0.10, 0.07])


def test_efficient_risk_at_minimum_volatility_picks_efficient_end_of_mvp_set():
    vol_mvp = portfolio_volatility(min_variance(DUP_COV), DUP_COV)
    w = efficient_risk(DUP_MU, DUP_COV, vol_mvp).to_numpy()
    # Minimum variance splits 9/13 into the duplicated risk and 4/13 into asset 3;
    # the efficient version puts the whole 9/13 into the 10% class.
    np.testing.assert_allclose(w, [0.0, 9 / 13, 4 / 13], atol=1e-9)
    assert portfolio_volatility(w, DUP_COV) <= vol_mvp * (1 + 1e-9)


def test_frontier_with_duplicated_asset_starts_at_efficient_end():
    frontier = efficient_frontier(DUP_MU, DUP_COV, 4)
    assert frontier["return"].iloc[0] == pytest.approx((0.10 * 9 + 0.07 * 4) / 13, abs=1e-10)
    assert np.all(np.diff(frontier["volatility"]) > 0)
    weights = frontier[[0, 1, 2]].to_numpy()
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, DUP_COV, weights))
    np.testing.assert_allclose(frontier["volatility"], vols, rtol=1e-8)


def test_efficient_risk_near_numerically_zero_volatility_does_not_leak_scipy_errors():
    # T < N: the minimum volatility is numerically zero, so a target of 1.001x it
    # is within solver noise; the answer is the efficient zero-variance portfolio.
    rng = np.random.default_rng(1)
    sample = np.cov(rng.normal(size=(6, 14)) * 0.01, rowvar=False) * TRADING_DAYS
    mu = rng.uniform(0.02, 0.1, 14)
    vol_mvp = portfolio_volatility(min_variance(sample), sample)
    w = efficient_risk(mu, sample, 1.001 * vol_mvp)
    assert w.sum() == pytest.approx(1.0)
    assert portfolio_volatility(w, sample) < 1e-6 * math.sqrt(np.trace(sample) / 14)
    assert portfolio_return(w, mu) >= portfolio_return(min_variance(sample), mu) - 1e-12


@pytest.mark.parametrize("gap", [1e-9, 1e-10, 3e-11, 2e-11])
def test_max_sharpe_with_rf_a_rounding_distance_below_the_best_return(gap):
    # cond(cov) = 1e6 with box bounds: the convex reformulation is too
    # ill-conditioned here, so the limit (min variance at r_max) is returned.
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    cov = (q * (0.04 * np.logspace(0, -6, 30))) @ q.T
    cov = 0.5 * (cov + cov.T)
    mu = np.random.default_rng(1001).uniform(0.02, 0.12, 30)
    bounds = (-0.1, 0.25)
    top = float(np.sort(mu)[::-1] @ np.r_[np.full(11, 0.25), 0.05, np.full(18, -0.1)])
    w = max_sharpe(mu, cov, top - gap, bounds)
    assert portfolio_return(w, mu) == pytest.approx(top, abs=1e-9)
    assert portfolio_return(w, mu) - (top - gap) > 0
    np.testing.assert_allclose(w, efficient_return(mu, cov, top, bounds), atol=1e-8)


def _best_zero_risk_return(mu, cov, bounds):
    """Independent LP: highest return among the minimum-variance portfolios.

    Every minimizer of w'Σw shares Σw, so the minimum-variance set is
    {w feasible : U'w = U'w_mvp} with U spanning range(Σ).
    """
    lam, vecs = np.linalg.eigh(cov)
    u = vecs[:, lam > 1e-9 * lam[-1]]
    w0 = min_variance(cov, bounds).to_numpy()
    res = optimize.linprog(
        -mu,
        A_eq=np.vstack([u.T, np.ones(len(mu))]),
        b_eq=np.r_[u.T @ w0, 1.0],
        bounds=[bounds] * len(mu),
        method="highs",
    )
    assert res.status == 0
    return -res.fun


def _rank_deficient_case(seed, n, t):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.12, n)
    cov = np.cov(rng.normal(size=(t, n)) * 0.01, rowvar=False) * TRADING_DAYS
    return mu, cov


@pytest.mark.parametrize(
    "seed, n, t, bounds",
    [
        (39, 30, 15, (-0.2, 0.6)),
        (11, 5, 2, (-0.2, 0.6)),
        (18, 11, 5, (-0.2, 0.6)),
        (20, 11, 4, (0.0, 0.4)),
        (8, 9, 4, (0.0, 0.4)),
        (3, 8, 5, (0.0, 1.0)),
    ],
)
def test_singular_frontier_starts_at_best_minimum_variance_portfolio(seed, n, t, bounds):
    mu, cov = _rank_deficient_case(seed, n, t)
    best = _best_zero_risk_return(mu, cov, bounds)
    frontier = efficient_frontier(mu, cov, 5, bounds=bounds)
    assert frontier["return"].iloc[0] >= best - 1e-9
    # A risk budget equal to the minimum volatility must reach the same return.
    noise = 1e-6 * math.sqrt(np.trace(cov) / n)
    target = portfolio_volatility(min_variance(cov, bounds), cov)
    w = efficient_risk(mu, cov, target, bounds)
    assert portfolio_return(w, mu) >= best - 1e-9
    assert portfolio_volatility(w, cov) <= target * (1 + 1e-9) + noise


def test_return_range_on_a_singular_covariance_accepts_the_flat_face():
    r_end = (0.10 * 9 + 0.07 * 4) / 13
    exact = efficient_frontier(DUP_MU, DUP_COV, 3, return_range=(r_end, 0.095))
    assert exact["return"].iloc[0] == pytest.approx(r_end, abs=1e-12)
    # The solver's own minimum-variance return lies on the dominated flat face:
    # the frontier starts at the face's efficient end instead of raising.
    r_solver = portfolio_return(min_variance(DUP_COV), DUP_MU)
    clamped = efficient_frontier(DUP_MU, DUP_COV, 3, return_range=(r_solver, 0.095))
    assert clamped["return"].iloc[0] == pytest.approx(r_end, abs=1e-12)
    # Below the face's lowest return (all 9/13 in the 5% class) is the inefficient branch.
    with pytest.raises(ValueError, match="inefficient branch"):
        efficient_frontier(DUP_MU, DUP_COV, 3, return_range=(0.05, 0.095))


def test_unbounded_return_on_the_minimum_variance_face_is_an_arbitrage_error():
    bounds = [(None, None), (None, None), (0.0, 1.0)]
    with pytest.raises(ValueError, match="riskless arbitrage"):
        efficient_frontier(DUP_MU, DUP_COV, 4, bounds=bounds)


def test_max_sharpe_with_near_tied_top_returns_is_shift_invariant():
    cov = np.diag([0.09, 0.01])
    mu = np.array([0.08 + 1e-10, 0.08])
    rf = float(mu.max()) - 5e-10
    # Diagonal covariance: tangency w ∝ (μ - rf) / σ², both excesses positive.
    excess = mu - rf
    expected = (excess / np.diag(cov)) / (excess / np.diag(cov)).sum()
    np.testing.assert_allclose(max_sharpe(mu, cov, rf), expected, atol=1e-6)
    np.testing.assert_allclose(max_sharpe(mu - 0.08, cov, rf - 0.08), expected, atol=1e-6)


def test_max_sharpe_near_the_top_still_reports_an_unbounded_sharpe_ratio():
    # Asset 3 is riskless and has the highest return: the Sharpe ratio is unbounded
    # for any rf below 8%, including one a rounding distance below it.
    cov = np.diag([0.04, 0.09, 0.0])
    mu = np.array([0.05, 0.07, 0.08])
    for rf in (0.07, 0.08 - 1e-10):
        with pytest.raises(ValueError, match="riskless"):
            max_sharpe(mu, cov, rf)


@pytest.mark.parametrize(
    "bounds", [(np.nan, 1.0), (0.0, np.nan), [(0, 1), (np.nan, np.nan), (0, 1)]]
)
def test_nan_bounds_are_rejected_as_input_errors(bounds):
    with pytest.raises(ValueError, match="NaN"):
        min_variance(DUP_COV, bounds)
    with pytest.raises(ValueError, match="NaN"):
        max_sharpe(DUP_MU, DUP_COV, 0.0, bounds)


def test_frontier_collapses_when_all_expected_returns_are_equal():
    cov = SMALL_COV
    frontier = efficient_frontier(np.full(5, 0.05), cov, 4)
    np.testing.assert_allclose(frontier.iloc[:, 3:], np.tile(min_variance(cov), (4, 1)), atol=1e-14)
    np.testing.assert_allclose(
        efficient_return(np.full(5, 0.05), cov, 0.05), min_variance(cov), atol=1e-15
    )
    with pytest.raises(ValueError, match="unattainable"):
        efficient_return(np.full(5, 0.05), cov, 0.06)
    with pytest.raises(ValueError, match="unattainable"):
        efficient_return(np.full(5, 0.05), cov, 0.06, bounds=None)


# ---------------------------------------------------------------------------
# Attainable return range (the LP max/min μ'w over the feasible set)
# ---------------------------------------------------------------------------


def vertex_enumeration_extreme(mu: np.ndarray, lo: np.ndarray, hi: np.ndarray, sense: float):
    """max (sense=+1) or min (sense=-1) of μ'w over {1'w = 1, lo <= w <= hi} by brute force.

    Every vertex of this polytope has at most one weight strictly inside its
    bounds, so trying each "pivot" asset with every lower/upper pattern of the
    others visits all vertices; the LP optimum is attained at one of them.
    Patterns that would need an infinite weight are skipped.
    """
    n, best = len(mu), -np.inf
    for p in range(n):
        others = [i for i in range(n) if i != p]
        for pattern in itertools.product((0, 1), repeat=n - 1):
            w = np.empty(n)
            for i, at_hi in zip(others, pattern, strict=True):
                w[i] = hi[i] if at_hi else lo[i]
            if not np.isfinite(w[others]).all():
                continue
            w[p] = 1.0 - math.fsum(w[others])
            if lo[p] - 1e-12 <= w[p] <= hi[p] + 1e-12:
                best = max(best, sense * math.fsum(mu * w))
    return sense * best


@pytest.mark.parametrize(
    "level, spread",
    [(0.05, 1e-2), (0.05, 1e-8), (0.05, 1e-10), (1e-7, 1e-7), (0.06 / TRADING_DAYS, 1e-7)],
    ids=["spread", "near_tied_1e-8", "near_tied_1e-10", "tiny_scale", "daily"],
)
def test_return_extreme_matches_vertex_enumeration(level, spread):
    # Regression: the range came from HiGHS linprog, whose ~1e-7 optimality
    # tolerance (in cost units) treats returns closer than that as ties and
    # returned the wrong vertex (even the max for the min) for near-equal,
    # tiny or daily-scale μ. The greedy knapsack rule compares exactly.
    rng = np.random.default_rng(17)
    checked = 0
    for _ in range(60):
        n = int(rng.integers(2, 6))
        mu = level + spread * rng.uniform(-1.0, 1.0, n)
        lo = rng.choice([-0.5, 0.0, 0.1], n)
        hi = np.maximum(rng.choice([0.3, 0.6, 1.0], n), lo)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        for sense in (1.0, -1.0):
            ext = markowitz._return_extreme(mu, lo, hi, sense)
            v = ext.vertex
            assert abs(v.sum() - 1.0) < 1e-14
            assert np.all(v >= lo) and np.all(v <= hi)
            assert ext.value == pytest.approx(float(mu @ v), abs=1e-17 * level)
            ref = vertex_enumeration_extreme(mu, lo, hi, sense)
            # Far below the 1e-10..1e-2 spread of μ, i.e. the right vertex.
            assert abs(ext.value - ref) <= 1e-14 * abs(level)
            checked += 1
    assert checked > 40


def test_return_extreme_with_infinite_bounds():
    inf = np.inf
    # Unbounded above: weight moves without limit from asset 0 into asset 1.
    ext = markowitz._return_extreme(
        np.array([0.05, 0.08]), np.array([-inf, 0.0]), np.array([1, inf]), 1.0
    )
    assert ext.value == math.inf and ext.vertex is None
    np.testing.assert_array_equal(ext.ray, [-1.0, 1.0])
    # ... but bounded below (the improving transfer would need hi_0 = +inf).
    ext = markowitz._return_extreme(
        np.array([0.05, 0.08]), np.array([-inf, 0.0]), np.array([1, inf]), -1.0
    )
    assert ext.value == pytest.approx(0.05, abs=1e-17)
    # Two tied assets unbounded both ways: no vertex exists, yet the optimum is
    # finite (0.5 in the 8% asset capped at 0.5, the rest at 5%).
    mu = np.array([0.05, 0.05, 0.08, 0.01])
    lo, hi = np.array([-inf, -inf, 0.0, 0.0]), np.array([inf, inf, 0.5, 0.5])
    top = markowitz._return_extreme(mu, lo, hi, 1.0)
    bottom = markowitz._return_extreme(mu, lo, hi, -1.0)
    assert top.value == pytest.approx(0.065, abs=1e-16)
    assert bottom.value == pytest.approx(0.5 * 0.05 + 0.5 * 0.01, abs=1e-16)
    for ext in (top, bottom):
        assert ext.vertex.sum() == pytest.approx(1.0, abs=1e-15)
        assert np.all(ext.vertex >= lo) and np.all(ext.vertex <= hi)
    # Partially infinite but bounded: matches the enumeration of finite vertices.
    mu = np.array([0.05, 0.08, 0.05, 0.02])
    lo, hi = np.array([-inf, 0.0, 0.0, -0.2]), np.array([0.3, 1.0, inf, 0.4])
    for sense in (1.0, -1.0):
        ext = markowitz._return_extreme(mu, lo, hi, sense)
        assert ext.value == pytest.approx(vertex_enumeration_extreme(mu, lo, hi, sense), abs=1e-16)


def test_near_tied_and_tiny_expected_returns():
    # Regression for the LP tolerance: each case raised "unattainable" or
    # "no feasible portfolio has expected return above rf" before.
    cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.0225, 0.003], [0.0, 0.003, 0.01]])
    sub = cov[1:, 1:]
    mvp_23 = np.linalg.solve(sub, np.ones(2))
    mvp_23 /= mvp_23.sum()

    mu = np.array([0.05 + 1e-8, 0.05, 0.05])
    # min(μ) is attainable long-only: the least risky mix of the two 5% assets.
    # (A return gap of 1e-8 pins weights only to ~1 ulp / 1e-8 ≈ 1e-9.)
    np.testing.assert_allclose(efficient_return(mu, cov, 0.05), np.r_[0.0, mvp_23], atol=1e-9)
    frontier = efficient_frontier(mu, cov, 5)
    weights = frontier.iloc[:, 3:].to_numpy()
    np.testing.assert_allclose(weights[0], min_variance(cov), atol=1e-12)
    np.testing.assert_allclose(weights[-1], [1.0, 0.0, 0.0], atol=1e-8)
    assert np.all(np.diff(frontier["volatility"]) >= 0)

    mu = np.array([0.05, 0.05, 0.05 + 5e-8])
    np.testing.assert_allclose(efficient_return(mu, cov, mu.max()), [0.0, 0.0, 1.0], atol=1e-9)
    rf = 0.05 + 1e-8  # only asset 2 beats rf, by 4e-8
    w = max_sharpe(mu, cov, rf).to_numpy()
    e = mu - rf
    assert w @ e > 0
    grad = cov @ w - (w @ cov @ w) / (e @ w) * e
    stationarity, wrong_sign = kkt_violation(grad, w, np.zeros(3), np.ones(3), np.ones(3))
    assert stationarity < 1e-9 and wrong_sign < 1e-9

    # Tiny scale: the same investor problem in units of 1e-6 has the same weights.
    mu = np.array([0.08, 0.06, 0.04])
    np.testing.assert_allclose(
        efficient_frontier(mu * 1e-6, cov * 1e-6, 4).iloc[:, 3:],
        efficient_frontier(mu, cov, 4).iloc[:, 3:],
        atol=1e-9,
    )

    # Daily-scale returns that differ by ~1e-7: the ends of the range are single assets.
    mu = np.array([0.06000144773063554, 0.0600255198115711, 0.06001731910407308]) / TRADING_DAYS
    daily_cov = cov / TRADING_DAYS
    np.testing.assert_allclose(efficient_return(mu, daily_cov, mu.max()), np.eye(3)[1], atol=1e-9)
    np.testing.assert_allclose(efficient_return(mu, daily_cov, mu.min()), np.eye(3)[0], atol=1e-9)
    vol_mvp = portfolio_volatility(min_variance(daily_cov), daily_cov)
    w = efficient_risk(mu, daily_cov, 1.05 * vol_mvp)
    assert portfolio_volatility(w, daily_cov) == pytest.approx(1.05 * vol_mvp, rel=1e-9)


# ---------------------------------------------------------------------------
# Invariances
# ---------------------------------------------------------------------------

OPTIMIZERS = {
    "min_variance": lambda mu, cov, rf, b: min_variance(cov, b),
    "max_sharpe": lambda mu, cov, rf, b: max_sharpe(mu, cov, rf, b),
    "efficient_return": lambda mu, cov, rf, b: efficient_return(mu, cov, 1.2 * mu.mean(), b),
    "efficient_risk": lambda mu, cov, rf, b: efficient_risk(
        mu, cov, 1.3 * portfolio_volatility(min_variance(cov, b), cov), b
    ),
}


@pytest.mark.parametrize("name", list(OPTIMIZERS))
@pytest.mark.parametrize("bounds", [(0.0, 1.0), (-0.2, 0.5), None], ids=["lo", "box", "none"])
def test_weights_invariant_to_annualization_scale(name, bounds):
    # Daily inputs (μ/252, Σ/252, rf/252) describe the same investor problem
    # as annual ones: the weights must not change. Annualized Σ entries are
    # O(1e-2) and daily ones O(1e-4); this catches absolute-tolerance and
    # objective-scaling problems in the solver. The Sharpe ratio scales by
    # sqrt(252), which catches a wrong annualization power.
    cov, mu = COV_CASES["well"]
    rf = 0.01
    annual = OPTIMIZERS[name](mu, cov, rf, bounds)
    daily = OPTIMIZERS[name](mu / TRADING_DAYS, cov / TRADING_DAYS, rf / TRADING_DAYS, bounds)
    np.testing.assert_allclose(daily, annual, atol=1e-9)
    s_annual = portfolio_sharpe(annual, mu, cov, rf)
    s_daily = portfolio_sharpe(daily, mu / TRADING_DAYS, cov / TRADING_DAYS, rf / TRADING_DAYS)
    assert s_annual == pytest.approx(s_daily * math.sqrt(TRADING_DAYS), rel=1e-9)


def test_utility_scale_invariance_with_rescaled_risk_aversion():
    # max w'(cμ) - (δ/2) w'(cΣ)w has the same solution as for (μ, Σ, δ).
    cov, mu = COV_CASES["well"]
    for bounds in [(0.0, 1.0), None]:
        np.testing.assert_allclose(
            mean_variance_utility(mu / TRADING_DAYS, cov / TRADING_DAYS, 3.0, bounds),
            mean_variance_utility(mu, cov, 3.0, bounds),
            atol=1e-10,
        )


def test_permutation_equivariance_and_labels():
    cov, mu = COV_CASES["well"]
    assets = [f"A{i:02d}" for i in range(30)]
    cov_df = pd.DataFrame(cov, index=assets, columns=assets)
    mu_s = pd.Series(mu, index=assets)
    perm = np.random.default_rng(4).permutation(30)
    perm_assets = [assets[i] for i in perm]
    cov_perm = cov_df.loc[perm_assets, perm_assets]
    for solve in (
        lambda m, c: max_sharpe(m, c, 0.01),
        lambda m, c: min_variance(c, (-0.1, 0.3)),
        lambda m, c: efficient_return(m, c, 0.09),
    ):
        base = solve(mu_s, cov_df)
        permuted = solve(mu_s, cov_perm)  # μ is aligned to the permuted Σ by label
        assert list(base.index) == assets
        assert list(permuted.index) == perm_assets
        np.testing.assert_allclose(permuted.reindex(assets), base, atol=1e-10)


def test_numpy_and_pandas_inputs_agree():
    cov, mu = SMALL_COV, SMALL_MU
    assets = list("VWXYZ")
    w_np = max_sharpe(mu, cov, 0.01)
    w_pd = max_sharpe(pd.Series(mu, index=assets), pd.DataFrame(cov, assets, assets), 0.01)
    assert list(w_np.index) == list(range(5))
    assert list(w_pd.index) == assets
    np.testing.assert_array_equal(w_np.to_numpy(), w_pd.to_numpy())
    # Unlabelled Σ with a labelled μ: Σ borrows μ's labels.
    w_mixed = max_sharpe(pd.Series(mu, index=assets), cov, 0.01)
    pd.testing.assert_series_equal(w_mixed, w_pd)


def test_constant_shift_of_mu_moves_efficient_portfolios_by_the_same_return():
    cov, mu = SMALL_COV, SMALL_MU
    for bounds in [(0.0, 1.0), None]:
        np.testing.assert_allclose(
            efficient_return(mu + 0.03, cov, 0.11, bounds),
            efficient_return(mu, cov, 0.08, bounds),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_asset():
    assert min_variance([[0.04]]).tolist() == [1.0]
    assert min_variance([[0.04]], bounds=None).tolist() == [1.0]
    assert max_sharpe([0.05], [[0.04]]).tolist() == [1.0]
    assert efficient_return([0.05], [[0.04]], 0.05).tolist() == [1.0]
    assert efficient_risk([0.05], [[0.04]], 0.3).tolist() == [1.0]
    assert mean_variance_utility([0.05], [[0.04]], 2.0).tolist() == [1.0]
    frontier = efficient_frontier([0.05], [[0.04]], 3)
    np.testing.assert_allclose(frontier["volatility"], 0.2)
    with pytest.raises(ValueError):
        max_sharpe([0.05], [[0.04]], rf=0.06)


def test_fixed_weights_are_respected():
    cov, mu = SMALL_COV, SMALL_MU
    bounds = [(0.2, 0.2), (0.0, 1.0), (0.0, 1.0), (0.1, 0.1), (0.0, 1.0)]
    for w in (
        min_variance(cov, bounds),
        max_sharpe(mu, cov, 0.0, bounds),
        efficient_return(mu, cov, 0.07, bounds),
        mean_variance_utility(mu, cov, 2.0, bounds),
    ):
        assert w.iloc[0] == pytest.approx(0.2, abs=1e-15)
        assert w.iloc[3] == pytest.approx(0.1, abs=1e-15)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
    every = [(0.2, 0.2), (0.3, 0.3), (0.1, 0.1), (0.25, 0.25), (0.15, 0.15)]
    np.testing.assert_allclose(min_variance(cov, every), [0.2, 0.3, 0.1, 0.25, 0.15], atol=1e-15)


def test_efficient_return_at_the_edges_of_the_attainable_range():
    cov, mu = SMALL_COV, SMALL_MU
    # Long-only: the max/min return portfolios are single assets (degenerate vertices).
    np.testing.assert_allclose(efficient_return(mu, cov, mu.max()), np.eye(5)[3], atol=1e-14)
    np.testing.assert_allclose(efficient_return(mu, cov, mu.min()), np.eye(5)[4], atol=1e-14)
    # Ties at the top: the least risky mix of the tied assets.
    tied = np.array([0.11, 0.09, 0.07, 0.11, 0.04])
    w = efficient_return(tied, cov, 0.11).to_numpy()
    sub = cov[np.ix_([0, 3], [0, 3])]
    expected = np.linalg.solve(sub, np.ones(2))
    expected /= expected.sum()
    np.testing.assert_allclose(w[[0, 3]], expected, atol=1e-12)
    np.testing.assert_allclose(w[[1, 2, 4]], 0.0, atol=1e-15)


def test_partially_unbounded_weights():
    cov, mu = SMALL_COV, SMALL_MU
    bounds = [(None, None), (0.0, None), (0.0, None), (0.0, None), (0.0, None)]
    w = efficient_return(mu, cov, 0.3, bounds).to_numpy()  # beyond max(μ): needs leverage
    assert w @ mu == pytest.approx(0.3, abs=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(w[1:] >= 0)
    lo = np.array([-np.inf, 0, 0, 0, 0])
    hi = np.full(5, np.inf)
    stationarity, wrong_sign = kkt_violation(cov @ w, w, lo, hi, np.vstack([np.ones(5), mu]))
    assert stationarity < 1e-9 and wrong_sign < 1e-9


def test_singular_covariance_is_fine_with_bounds_but_not_without():
    rng = np.random.default_rng(3)
    sample = np.cov(rng.normal(size=(5, 8)) * 0.01, rowvar=False) * TRADING_DAYS
    w = min_variance(sample)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert portfolio_volatility(w, sample) < 1e-6  # a (near) riskless combination exists
    with pytest.raises(ValueError, match="positive definite"):
        min_variance(sample, bounds=None)


# ---------------------------------------------------------------------------
# Solver fallbacks
# ---------------------------------------------------------------------------


def failed_slsqp(fun, x0, **kwargs):
    """Stand-in for scipy.optimize.minimize: an SLSQP run that failed far away."""
    return optimize.OptimizeResult(
        x=np.full_like(np.asarray(x0, dtype=float), 10.0), success=False, message="forced failure"
    )


@pytest.mark.parametrize("bounds", SMALL_BOUNDS, ids=["long_only", "box", "per_asset"])
def test_active_set_solves_from_cold_start_when_slsqp_fails(monkeypatch, bounds):
    # Without a usable SLSQP point the active-set method must reach the optimum
    # on its own from the feasible start (equal weight projected into the bounds).
    monkeypatch.setattr(markowitz.optimize, "minimize", failed_slsqp)
    lo, hi = small_bound_arrays(bounds)
    ones = np.ones((1, 5))
    ref = brute_force_qp(SMALL_COV, np.zeros(5), ones, np.ones(1), lo, hi)
    np.testing.assert_allclose(min_variance(SMALL_COV, bounds), ref, atol=1e-12)
    ref = brute_force_qp(
        SMALL_COV, np.zeros(5), np.vstack([ones, SMALL_MU]), np.array([1.0, 0.08]), lo, hi
    )
    np.testing.assert_allclose(efficient_return(SMALL_MU, SMALL_COV, 0.08, bounds), ref, atol=1e-12)
    ref = brute_force_qp(4.0 * SMALL_COV, SMALL_MU, ones, np.ones(1), lo, hi)
    np.testing.assert_allclose(
        mean_variance_utility(SMALL_MU, SMALL_COV, 4.0, bounds), ref, atol=1e-12
    )


@pytest.mark.parametrize("case", ["well", "ill"])
def test_active_set_cold_start_large_problem_satisfies_kkt(monkeypatch, case):
    monkeypatch.setattr(markowitz.optimize, "minimize", failed_slsqp)
    cov, mu = COV_CASES[case]
    lo, hi = bounds_arrays(0.0, 1.0, 30)
    w = min_variance(cov).to_numpy()
    assert max(kkt_violation(cov @ w, w, lo, hi, np.ones(30))) < 1e-9
    w = max_sharpe(mu, cov, 0.02).to_numpy()
    e = mu - 0.02
    grad = cov @ w - (w @ cov @ w) / (e @ w) * e
    assert max(kkt_violation(grad, w, lo, hi, np.ones(30))) < 1e-9


@pytest.mark.parametrize("scale", [1.0, 1.0 / TRADING_DAYS], ids=["annual", "daily"])
@pytest.mark.parametrize("case", ["well", "ill"])
def test_degenerate_vertex_targets_without_slsqp(monkeypatch, case, scale):
    # At the ends of the long-only return range the feasible set is a single
    # vertex where N + 1 constraints are active in N dimensions; old SciPy
    # SLSQP versions give up there ("Inequality constraints incompatible").
    # The active-set method must still return the vertex exactly.
    monkeypatch.setattr(markowitz.optimize, "minimize", failed_slsqp)
    cov, mu = COV_CASES[case]
    eye = np.eye(30)
    w = efficient_return(mu * scale, cov * scale, float(mu.max() * scale))
    np.testing.assert_allclose(w, eye[np.argmax(mu)], atol=1e-14)
    w = efficient_return(mu * scale, cov * scale, float(mu.min() * scale))
    np.testing.assert_allclose(w, eye[np.argmin(mu)], atol=1e-14)


def test_active_set_certificate_does_not_loosen_with_a_far_start():
    # Regression: the activity and multiplier tolerances of the active-set
    # method were scaled by the START point. From the |y0| ~ 1e15 start of a
    # max-Sharpe QP (rf = mean(μ), so the equal-weight start has a
    # rounding-noise excess) they grew to ~1e6, and a point with an O(1) KKT
    # violation was returned as certified optimal. They now follow the iterate.
    cov, mu = COV_CASES["well"]
    n = 30
    rf = float(mu.mean())
    e = mu - rf
    e_hat = e / np.abs(e).max()
    w_eq = np.full(n, 1.0 / n)
    assert 0.0 < e_hat @ w_eq < 1e-12  # deterministic data: a tiny positive excess
    y0 = w_eq / (e_hat @ w_eq)
    # Cornuejols-Tütüncü QP for long-only weights (k = 1'y):
    #   min y'Σy  s.t.  ê'y = 1,  y_i >= 0,  k - y_i >= 0.
    G = np.vstack([np.eye(n), np.ones((n, n)) - np.eye(n)])
    H = cov / (np.trace(cov) / n)
    y = markowitz._active_set_qp(H, np.zeros(n), e_hat[None, :], np.ones(1), G, np.zeros(2 * n), y0)
    assert y is not None
    w = y / y.sum()
    grad = cov @ w - (w @ cov @ w) / (e @ w) * e
    lo, hi = bounds_arrays(0.0, 1.0, n)
    assert max(kkt_violation(grad, w, lo, hi, np.ones(n))) < 1e-9
    np.testing.assert_allclose(w, max_sharpe(mu, cov, rf), atol=1e-9)


def test_successful_slsqp_point_is_used_when_refinement_fails(monkeypatch):
    monkeypatch.setattr(markowitz, "_active_set_qp", lambda *args: None)
    lo, hi = bounds_arrays(0.0, 1.0, 5)
    ref = brute_force_qp(SMALL_COV, np.zeros(5), np.ones((1, 5)), np.ones(1), lo, hi)
    w = min_variance(SMALL_COV)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(w, ref, atol=1e-5)  # SLSQP accuracy, not exact


def test_optimization_error_when_every_solver_fails(monkeypatch):
    monkeypatch.setattr(markowitz.optimize, "minimize", failed_slsqp)
    monkeypatch.setattr(markowitz, "_active_set_qp", lambda *args: None)
    with pytest.raises(OptimizationError, match="forced failure"):
        min_variance(SMALL_COV)
    with pytest.raises(OptimizationError):
        max_sharpe(SMALL_MU, SMALL_COV)
    # Closed forms do not use the numerical solver.
    assert min_variance(SMALL_COV, bounds=None).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_covariance_inputs():
    with pytest.raises(ValueError, match="square"):
        min_variance(np.ones((2, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        min_variance(np.array([[0.04, 0.01], [0.0, 0.09]]))
    with pytest.raises(ValueError, match="NaN"):
        min_variance(np.array([[0.04, np.nan], [np.nan, 0.09]]))
    with pytest.raises(ValueError, match="positive semi-definite"):
        min_variance(np.array([[0.04, 0.05], [0.05, 0.04]]))
    with pytest.raises(ValueError, match="positive definite"):
        min_variance(np.array([[0.04, 0.04], [0.04, 0.04]]), bounds=None)
    with pytest.raises(ValueError, match="zero"):
        min_variance(np.zeros((2, 2)))
    duplicated = pd.DataFrame(SMALL_COV[:2, :2], index=["A", "A"], columns=["A", "A"])
    with pytest.raises(ValueError, match="unique"):
        min_variance(duplicated)


def test_invalid_expected_returns():
    cov = pd.DataFrame(SMALL_COV, index=list("ABCDE"), columns=list("ABCDE"))
    with pytest.raises(ValueError, match="expected 5 values"):
        max_sharpe(SMALL_MU[:4], cov)
    with pytest.raises(ValueError, match="NaN"):
        max_sharpe(np.r_[SMALL_MU[:4], np.nan], cov)
    with pytest.raises(ValueError, match="missing"):
        max_sharpe(pd.Series(SMALL_MU[:4], index=list("ABCD")), cov)
    with pytest.raises(ValueError, match="not in the asset universe"):
        max_sharpe(pd.Series(0.05, index=list("ABCDEF")), cov)


def test_invalid_bounds():
    cov, mu = SMALL_COV, SMALL_MU
    with pytest.raises(ValueError, match="infeasible"):
        min_variance(cov, bounds=(0.0, 0.1))  # 5 assets cannot reach 100%
    with pytest.raises(ValueError, match="infeasible"):
        max_sharpe(mu, cov, bounds=(0.3, 1.0))
    with pytest.raises(ValueError, match="exceeds"):
        min_variance(cov, bounds=(0.5, 0.1))
    with pytest.raises(ValueError, match="bound pairs"):
        min_variance(cov, bounds=[(0, 1)] * 4)


def test_invalid_scalars_and_targets():
    cov, mu = SMALL_COV, SMALL_MU
    with pytest.raises(ValueError, match="unattainable"):
        efficient_return(mu, cov, mu.max() + 0.01)
    with pytest.raises(ValueError, match="unattainable"):
        efficient_return(mu, cov, mu.min() - 0.01)
    with pytest.raises(ValueError, match="unattainable"):
        efficient_return(mu, cov, mu.max() + 1e-7)  # the tolerance is rounding-sized
    np.testing.assert_allclose(
        efficient_return(mu, cov, mu.max() + 1e-13), np.eye(5)[3], atol=1e-14
    )
    with pytest.raises(ValueError, match="unattainable"):
        efficient_return(mu, cov, 0.2, bounds=(0.0, 0.4))
    with pytest.raises(ValueError, match="target_return"):
        efficient_return(mu, cov, float("inf"))
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="risk_aversion"):
            mean_variance_utility(mu, cov, bad)
        with pytest.raises(ValueError, match="target_volatility"):
            efficient_risk(mu, cov, bad)
    with pytest.raises(ValueError, match="rf"):
        max_sharpe(mu, cov, rf=float("nan"))
    with pytest.raises(ValueError, match="rf"):
        efficient_frontier(mu, cov, rf="x")
