"""Mean-variance (Markowitz) portfolio optimization.

Conventions follow :mod:`quantfin.portfolio._common`: ``mu`` (expected simple
returns), ``cov`` (covariance) and ``rf`` (risk-free rate) are ANNUALIZED;
weights are a ``pd.Series`` indexed by asset that sums to 1. Inputs may be numpy
or pandas; a labelled ``mu``/``weights`` Series is aligned to the covariance
labels (it must carry exactly the same assets).

Problems solved (Markowitz 1952), all with the budget constraint ``1'w = 1``::

    min_variance          min  w'Σw
    efficient_return      min  w'Σw            s.t.  w'μ = r*
    efficient_risk        max  w'μ             s.t.  sqrt(w'Σw) <= σ*
    mean_variance_utility max  w'μ - (δ/2) w'Σw
    max_sharpe            max  (w'μ - rf) / sqrt(w'Σw)

and, optionally, box bounds ``lo_i <= w_i <= hi_i`` (default long-only
``(0, 1)``; ``bounds=None`` allows unlimited short selling).

Unbounded problems have closed forms (Merton 1972). With
``A = 1'Σ⁻¹1``, ``B = 1'Σ⁻¹μ``, ``C = μ'Σ⁻¹μ`` and ``D = AC - B²`` the
minimum-variance portfolio is ``w_mvp = Σ⁻¹1 / A`` with return ``B/A`` and
variance ``1/A``, and every frontier portfolio is

    w(r) = w_mvp + (r - B/A) Σ⁻¹(μ - (B/A)1) / q,     q = D/A,
    σ²(r) = 1/A + (r - B/A)² / q,

a hyperbola in (σ, r) space; any two frontier portfolios span the whole
frontier (two-fund separation, Tobin 1958). The code uses this centred form
(``μ - (B/A)1`` and ``q = ‖L⁻¹(μ - (B/A)1)‖²`` with Cholesky factor ``L``)
because ``D = AC - B²`` suffers catastrophic cancellation.

Bounded problems are convex quadratic programs. They are solved with SciPy's
SLSQP (Kraft 1988) using analytic gradients, on a RESCALED problem (the
covariance is divided by its average variance so the objective is O(1) instead
of O(1e-2), which makes SLSQP's absolute ``ftol`` meaningful), started from an
exactly feasible point. The SLSQP point then warm-starts a primal active-set
method (Nocedal & Wright 2006, Algorithm 16.3) that solves the KKT system of
the identified active set exactly and certifies optimality (feasibility plus
non-negative inequality multipliers). The refinement matters for
ill-conditioned covariances: SLSQP stops when the objective stalls, and with a
condition number of 1e6 an objective error of 1e-12 still allows weight errors
of order sqrt(1e-12 / λ_min) ≈ 1e-3 along the low-variance eigenvectors.

The key practical lesson (Michaud 1989; Best & Grauer 1991; Chopra & Ziemba
1993): the optimizer effectively applies Σ⁻¹ to μ, so it *maximizes* estimation
error. Small errors in μ along low-variance eigenvectors of Σ become huge,
confident bets; the "optimal" portfolio's in-sample Sharpe ratio overstates its
out-of-sample Sharpe ratio (Broadie 1993). Shrinking Σ (Ledoit-Wolf) and
constraining weights both reduce the damage.

References:
    H. Markowitz (1952), "Portfolio Selection", Journal of Finance 7(1), 77-91.
    J. Tobin (1958), "Liquidity preference as behavior towards risk", Review of
    Economic Studies 25(2), 65-86.
    G. B. Dantzig (1957), "Discrete-variable extremum problems", Operations
    Research 5(2), 266-288 (greedy solution of the continuous knapsack).
    W. F. Sharpe (1966), "Mutual fund performance", Journal of Business 39(1).
    R. C. Merton (1972), "An analytic derivation of the efficient portfolio
    frontier", Journal of Financial and Quantitative Analysis 7(4), 1851-1872.
    D. Kraft (1988), "A software package for sequential quadratic programming",
    DFVLR-FB 88-28.
    R. Michaud (1989), "The Markowitz optimization enigma: is 'optimized'
    optimal?", Financial Analysts Journal 45(1), 31-42.
    M. Broadie (1993), "Computing efficient frontiers using estimated
    parameters", Annals of Operations Research 45, 21-58.
    J. Nocedal and S. J. Wright (2006), Numerical Optimization, 2nd ed.,
    Springer, ch. 16.
    G. Cornuejols and R. Tütüncü (2007), Optimization Methods in Finance,
    Cambridge University Press, §8.2 (maximum Sharpe ratio as a convex QP).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from quantfin.portfolio._common import (
    Bounds,
    OptimizationError,
    as_cov_frame,
    as_series,
    normalize_bounds,
)

__all__ = [
    "efficient_frontier",
    "efficient_return",
    "efficient_risk",
    "max_sharpe",
    "mean_variance_utility",
    "min_variance",
    "portfolio_return",
    "portfolio_sharpe",
    "portfolio_volatility",
]

VectorLike = pd.Series | np.ndarray | Sequence[float]
MatrixLike = pd.DataFrame | np.ndarray

_EPS = float(np.finfo(float).eps)

# SLSQP settings from the spec. ftol is an ABSOLUTE tolerance on the objective,
# which is why every bounded problem is rescaled to an O(1) objective first.
_SLSQP_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}

# Largest constraint violation (in the units of the rescaled problem) accepted
# from a solver before its point is used or returned.
_FEAS_TOL = 1e-8

# An eigenvalue ratio lambda_min / lambda_max at or below this many multiples of
# N * eps is rounding noise: the matrix is singular for practical purposes.
_PD_FACTOR = 1e3

# When rf is within this RELATIVE distance of the highest attainable return, the
# tangency portfolio is taken as its limit, the minimum-variance portfolio at
# that return (the convex reformulation is too ill-conditioned to solve there).
_DEGENERATE_EXCESS = 1e-8

_FRONTIER_COLUMNS = ("return", "volatility", "sharpe")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Problem:
    """Validated, aligned inputs of one optimization problem."""

    assets: pd.Index
    cov: np.ndarray
    mu: np.ndarray | None
    lo: np.ndarray | None  # None means unbounded (short selling allowed)
    hi: np.ndarray | None

    @property
    def n(self) -> int:
        return self.cov.shape[0]

    @property
    def bounded(self) -> bool:
        return self.lo is not None


def _cov_frame(cov: MatrixLike, *label_sources: object) -> pd.DataFrame:
    """Validate ``cov``; an unlabelled matrix borrows labels from the first Series given."""
    labels = None
    if not isinstance(cov, pd.DataFrame):
        for source in label_sources:
            if isinstance(source, pd.Series):
                labels = list(source.index)
                break
    frame = as_cov_frame(cov, labels)
    if not frame.index.is_unique:
        raise ValueError("asset labels must be unique")
    return frame


def _vector(x: VectorLike, assets: pd.Index, name: str) -> np.ndarray:
    """Align ``x`` to ``assets``; a Series must carry exactly the same labels."""
    if isinstance(x, pd.Series):
        if not x.index.is_unique:
            raise ValueError(f"{name} has duplicate labels")
        known = set(assets)
        extra = [label for label in x.index if label not in known]
        if extra:
            raise ValueError(f"{name} has labels that are not in the asset universe: {extra}")
    try:
        return as_series(x, assets).to_numpy(dtype=float)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None


def _scalar(x: float, name: str) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {x!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return value


def _problem(cov: MatrixLike, mu: VectorLike | None = None, bounds: Bounds = None) -> _Problem:
    frame = _cov_frame(cov, mu)
    assets = frame.index
    mu_values = None if mu is None else _vector(mu, assets, "mu")
    pairs = normalize_bounds(bounds, len(assets))
    lo = hi = None
    if pairs is not None:
        lo = np.array([pair[0] for pair in pairs], dtype=float)
        hi = np.array([pair[1] for pair in pairs], dtype=float)
        # All-infinite bounds are the unbounded problem: use the exact closed forms.
        if np.isneginf(lo).all() and np.isposinf(hi).all():
            lo = hi = None
    return _Problem(assets, frame.to_numpy(dtype=float), mu_values, lo, hi)


def _weights_series(w: np.ndarray, prob: _Problem) -> pd.Series:
    return pd.Series(w, index=prob.assets.copy(), dtype=float)


def _eigen_range(cov: np.ndarray) -> tuple[float, float]:
    eigenvalues = np.linalg.eigvalsh(cov)
    return float(eigenvalues[0]), float(eigenvalues[-1])


def _cholesky_pd(cov: np.ndarray, what: str) -> np.ndarray:
    """Lower Cholesky factor of a positive definite ``cov`` (ValueError if singular)."""
    lam_min, lam_max = _eigen_range(cov)
    threshold = _PD_FACTOR * max(cov.shape[0], 1) * _EPS * max(lam_max, 0.0)
    if not lam_max > 0.0 or lam_min <= threshold:
        raise ValueError(
            f"{what} with bounds=None needs a positive definite covariance matrix "
            f"(eigenvalues span [{lam_min:.3g}, {lam_max:.3g}]); use bounds or a "
            "shrinkage estimator"
        )
    return np.linalg.cholesky(cov)


def _check_psd(cov: np.ndarray) -> float:
    """Raise unless ``cov`` is positive semi-definite and non-zero; return its scale.

    The scale (average variance, trace / N) is used to rescale objectives to O(1).
    """
    lam_min, lam_max = _eigen_range(cov)
    if not lam_max > 0.0:
        raise ValueError("covariance matrix is zero or negative definite")
    if lam_min < -_PD_FACTOR * cov.shape[0] * _EPS * lam_max:
        raise ValueError(
            f"covariance matrix is not positive semi-definite (smallest eigenvalue {lam_min:.3g})"
        )
    return float(np.trace(cov)) / cov.shape[0]


def _return_tol(mu: np.ndarray) -> float:
    """Absolute tolerance for comparing portfolio returns."""
    return 1e-10 * max(float(np.abs(mu).max()), 1e-12)


def _mu_is_constant(mu: np.ndarray) -> bool:
    # Every fully invested portfolio then has the same return; the return
    # constraint duplicates the budget constraint.
    return float(np.ptp(mu)) <= 64 * _EPS * float(np.abs(mu).max())


def _is_singular(cov: np.ndarray) -> bool:
    """Whether ``cov`` is singular for practical purposes (see ``_PD_FACTOR``)."""
    lam_min, lam_max = _eigen_range(cov)
    return lam_min <= _PD_FACTOR * cov.shape[0] * _EPS * max(lam_max, 0.0)


def _variance_noise(cov: np.ndarray) -> float:
    """Variance differences below this are numerical noise (1e-12 of the average variance).

    Solvers on singular covariances only resolve the variance to roughly this level,
    so comparisons near the minimum variance must allow for it.
    """
    return 1e-12 * float(np.trace(cov)) / cov.shape[0]


# ---------------------------------------------------------------------------
# Closed forms (unbounded)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TwoFund:
    """Merton (1972) frontier in centred form: w(r) = w_mvp + (r - r_mvp) v / q."""

    w_mvp: np.ndarray
    r_mvp: float
    var_mvp: float
    v: np.ndarray  # Σ⁻¹(μ - r_mvp 1): a zero-cost (1'v = 0) long-short portfolio
    q: float  # (μ - r_mvp 1)'Σ⁻¹(μ - r_mvp 1) = D / A >= 0


def _mvp_closed_form(cov: np.ndarray, chol: np.ndarray) -> np.ndarray:
    s1 = linalg.cho_solve((chol, True), np.ones(cov.shape[0]))
    # 1's1 = A > 0 for positive definite Σ; dividing by the computed sum also
    # makes the budget hold to rounding.
    return s1 / s1.sum()


def _two_fund(cov: np.ndarray, mu: np.ndarray, chol: np.ndarray) -> _TwoFund:
    w_mvp = _mvp_closed_form(cov, chol)
    r_mvp = float(mu @ w_mvp)
    centred = mu - r_mvp
    z = linalg.solve_triangular(chol, centred, lower=True)
    q = float(z @ z)  # a sum of squares: never negative, no AC - B² cancellation
    v = linalg.cho_solve((chol, True), centred)
    # 1'v = 0 in exact arithmetic; remove the rounding residue along w_mvp,
    # which keeps v in span{Σ⁻¹1, Σ⁻¹μ} (so optimality is unaffected).
    v = v - v.sum() * w_mvp
    return _TwoFund(w_mvp, r_mvp, float(w_mvp @ cov @ w_mvp), v, q)


def _normalized(w: np.ndarray, what: str) -> np.ndarray:
    total = float(w.sum())
    if not np.isfinite(w).all() or abs(total - 1.0) > 1e-6:
        raise OptimizationError(f"{what}: closed form lost the budget constraint (sum {total})")
    return w / total


# ---------------------------------------------------------------------------
# Feasible points and linear programs over {1'w = 1, lo <= w <= hi}
# ---------------------------------------------------------------------------


def _feasible_center(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Euclidean projection of the equal-weight portfolio onto the feasible set.

    The projection of 1/N onto {1'w = 1, lo <= w <= hi} is clip(1/N + λ, lo, hi)
    with the scalar λ chosen so the weights sum to 1; the sum is non-decreasing
    in λ, so bisection finds it. This is "equal weight clipped into the bounds"
    done so that it stays feasible.
    """
    n = lo.size
    base = np.full(n, 1.0 / n)

    def excess(lam: float) -> float:
        return float(np.clip(base + lam, lo, hi).sum()) - 1.0

    left, right = -1.0, 1.0
    for _ in range(64):
        if excess(left) <= 0.0:
            break
        left *= 2.0
    for _ in range(64):
        if excess(right) >= 0.0:
            break
        right *= 2.0
    for _ in range(200):
        mid = 0.5 * (left + right)
        if mid in (left, right):
            break
        if excess(mid) < 0.0:
            left = mid
        else:
            right = mid
    w = np.clip(base + right, lo, hi)
    return _repair_budget(w, lo, hi)


def _repair_budget(w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Spread a rounding-size budget residual over the assets not at a bound."""
    residual = 1.0 - float(w.sum())
    if residual != 0.0:
        free = (w > lo) & (w < hi)
        if free.any():
            w = w.copy()
            w[free] += residual / free.sum()
            w = np.clip(w, lo, hi)
    return w


@dataclass(frozen=True)
class _Extreme:
    """Optimum of max/min μ'w over the feasible set (an LP)."""

    value: float  # +-inf when the LP is unbounded
    vertex: np.ndarray | None  # an optimal vertex (finite value)
    ray: np.ndarray | None  # an improving recession direction (infinite value)


def _return_extreme(mu: np.ndarray, lo: np.ndarray, hi: np.ndarray, sense: float) -> _Extreme:
    """Maximize (sense=+1) or minimize (sense=-1) μ'w s.t. 1'w = 1, lo <= w <= hi.

    The LP is unbounded exactly when weight can be moved without limit from an
    asset j with lo_j = -inf into another asset i with hi_i = +inf and
    sense·μ_i > sense·μ_j (the recession cone is generated by such transfers).

    Otherwise it is a continuous knapsack, solved exactly by the greedy rule
    (Dantzig 1957): rank the assets by sense·μ, fill the best ones to their
    upper bounds and leave the rest at their lower bounds, with one "pivot"
    asset taking the remaining budget. Writing s_k for the k-th ranked asset
    and G(t) = Σ_{k<t} hi_{s_k} + Σ_{k>=t} lo_{s_k} (non-decreasing in t), the
    pivot is the last position p with G(p) <= 1; then 1 <= G(p+1) puts the
    pivot's weight inside its bounds. Optimality: for any feasible w with
    pivot score s*, sense·μ'w = s* + Σ (score_i - s*) w_i, and every term is
    maximized by w_i = hi_i above s* and w_i = lo_i below it.

    An LP solver would be no better here: its default optimality tolerances
    (about 1e-7 in cost units) treat returns that differ by less than that as
    ties and can return the wrong end of the range for near-equal or
    daily-scale returns; the greedy rule compares returns exactly.
    """
    score = sense * mu
    best_gain, best_pair = 0.0, None
    for i in np.flatnonzero(np.isposinf(hi)):
        for j in np.flatnonzero(np.isneginf(lo)):
            gain = float(score[i] - score[j])
            if i != j and gain > best_gain:
                best_gain, best_pair = gain, (i, j)
    if best_pair is not None:
        ray = np.zeros(mu.size)
        ray[best_pair[0]], ray[best_pair[1]] = 1.0, -1.0
        return _Extreme(sense * math.inf, None, ray)
    # Without an improving ray every asset with hi = +inf scores at most as
    # much as every other asset with lo = -inf. Among equal scores, rank the
    # lo = -inf assets first and the hi = +inf assets last; then no prefix
    # sum of hi meets a suffix sum of lo with opposite infinities (except
    # between tied assets unbounded on both sides, handled below).
    tie_break = np.where(np.isneginf(lo), 0, 1) + np.where(np.isposinf(hi), 1, 0)
    order = np.lexsort((tie_break, -score))
    lo_s, hi_s = lo[order], hi[order]
    n = mu.size
    with np.errstate(invalid="ignore"):
        # inf - inf (two tied doubly unbounded assets) gives NaN, which
        # compares False below and so is never chosen as the pivot.
        head = np.concatenate([[0.0], np.cumsum(hi_s)[:-1]])  # Σ_{k<t} hi
        tail = np.cumsum(lo_s[::-1])[::-1]  # Σ_{k>=t} lo
        g = head + tail
    candidates = np.flatnonzero(g <= 1.0)
    pivot = int(candidates[-1]) if candidates.size else 0
    w_sorted = np.where(np.arange(n) < pivot, hi_s, lo_s)
    # A non-finite entry away from the pivot is a doubly unbounded asset tied
    # with the (doubly unbounded) pivot: any finite weight is optimal for it.
    w_sorted[~np.isfinite(w_sorted)] = 0.0
    w_sorted[pivot] = 0.0
    w_sorted[pivot] = 1.0 - float(w_sorted.sum())
    vertex = np.empty(n)
    vertex[order] = w_sorted
    vertex = _repair_budget(np.clip(vertex, lo, hi), lo, hi)
    return _Extreme(float(mu @ vertex), vertex, None)


def _point_with_return(
    mu: np.ndarray,
    target: float,
    center: np.ndarray,
    top: _Extreme,
    bottom: _Extreme,
) -> np.ndarray:
    """A feasible portfolio with μ'w = target (target must lie in the attainable range).

    Convex combination of the feasible centre with the max- or min-return
    vertex, or a step along an improving ray when the range is unbounded.
    """
    r_center = float(mu @ center)
    extreme = top if target >= r_center else bottom
    if extreme.vertex is not None:
        span = float(mu @ extreme.vertex) - r_center
        if span == 0.0:
            return center.copy()
        t = min(max((target - r_center) / span, 0.0), 1.0)
        return center + t * (extreme.vertex - center)
    ray = extreme.ray
    return center + ((target - r_center) / float(mu @ ray)) * ray


# ---------------------------------------------------------------------------
# Convex QP solver: SLSQP + active-set refinement
# ---------------------------------------------------------------------------


def _box_rows(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Write finite box bounds as rows of G x >= h."""
    n = lo.size
    eye = np.eye(n)
    has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
    rows = np.vstack([eye[has_lo], -eye[has_hi]]) if n else np.empty((0, n))
    rhs = np.concatenate([lo[has_lo], -hi[has_hi]])
    return rows.reshape(-1, n), rhs


def _solve_eqp(
    H: np.ndarray, c: np.ndarray, C: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve min ½x'Hx - c'x s.t. Cx = d by the null-space method.

    With the QR factorization C' = [Y Z] [R; 0], every feasible point is
    x = Y R⁻ᵀd + Z u, and u solves the reduced system (Z'HZ) u = Z'(c - H x_p)
    by Cholesky (Nocedal & Wright 2006, §16.2). Unlike an LU of the indefinite
    KKT matrix this never mixes the scales of H and C, and at a vertex (no
    null space) x comes from the constraints alone, exactly.

    Returns (x, λ) with Hx - c = C'λ (λ = R⁻¹Y'(Hx - c)), or None if the
    constraints are dependent or H is singular on the null space of C.
    """
    n, m = H.shape[0], C.shape[0]
    if m > n:
        return None
    q, r = np.linalg.qr(C.T, mode="complete")
    r = r[:m, :m]
    pivots = np.abs(np.diag(r))
    if m and not pivots.min() > 1e-10 * pivots.max():
        return None
    y, z = q[:, :m], q[:, m:]
    x = y @ linalg.solve_triangular(r, d, trans="T", lower=False)
    if z.shape[1]:
        reduced = z.T @ H @ z
        try:
            factor = linalg.cholesky(0.5 * (reduced + reduced.T), lower=True)
        except linalg.LinAlgError:
            return None
        diag = np.diag(factor) ** 2
        # Pivots this small mean Z'HZ is singular to working precision.
        if not diag.min() > 1e-14 * diag.max():
            return None
        x = x + z @ linalg.cho_solve((factor, True), z.T @ (c - H @ x))
    lam = linalg.solve_triangular(r, y.T @ (H @ x - c), lower=False)
    return x, lam


def _kkt_certified(
    H: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    x: np.ndarray,
    act_tol: float,
    tol: float,
) -> bool:
    """Is x a KKT point? Searches multipliers over ALL active constraints.

    Solves min ‖A'ν + G_act'μ - (Hx - c)‖ over ν free, μ >= 0 (bounded least
    squares). Unlike the multipliers of one working set this is robust at
    degenerate vertices, where more constraints are active than there are
    free dimensions and a given working set can show a spurious negative μ.
    """
    gradient = H @ x - c
    active = np.flatnonzero(G @ x - h <= act_tol)
    M = np.hstack([A.T, G[active].T])
    lower = np.concatenate([np.full(A.shape[0], -np.inf), np.zeros(active.size)])
    res = optimize.lsq_linear(M, gradient, bounds=(lower, np.inf), method="bvls")
    return float(np.abs(M @ res.x - gradient).max()) <= tol


def _active_set_qp(
    H: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    x_start: np.ndarray,
) -> np.ndarray | None:
    """Primal active-set method for min ½x'Hx - c'x s.t. Ax = b, Gx >= h.

    Nocedal & Wright (2006), Algorithm 16.3, in "solve for the iterate" form:
    each iteration solves the equality-constrained QP of the working set W
    exactly, then either (i) steps toward it until the first blocking
    constraint (which joins W), or (ii) reaches it and stops if the
    multipliers of W are non-negative (a KKT point, hence the global optimum
    of this convex QP), otherwise drops the most negative one.

    ``x_start`` must be feasible up to a small tolerance. Returns None on
    failure (singular KKT system, no convergence) so callers can fall back.
    """
    n, m_eq = x_start.size, A.shape[0]
    norms = np.linalg.norm(G, axis=1)
    nonzero = norms > 0.0
    if np.any(h[~nonzero] > 0.0):
        return None
    # Unit-norm rows make every slack a Euclidean distance, so one tolerance fits all.
    G = G[nonzero] / norms[nonzero, None]
    h = h[nonzero] / norms[nonzero]
    x = x_start.astype(float).copy()
    x_scale = max(1.0, float(np.abs(x).max()))
    act_tol = 1e-9 * x_scale
    slack = G @ x - h
    if slack.size and slack.min() < -1e-6 * x_scale:
        return None
    if np.linalg.matrix_rank(A) < m_eq:
        return None

    # Initial working set: the constraints active at x_start, most active
    # first, keeping only rows independent of the equalities and each other.
    working: list[int] = []
    basis, rank = A, m_eq
    for i in np.argsort(slack, kind="stable"):
        if slack[i] > act_tol or rank >= n:
            break
        trial = np.vstack([basis, G[i]])
        if np.linalg.matrix_rank(trial) > rank:
            basis, rank = trial, rank + 1
            working.append(int(i))
    # Move x onto the working constraints exactly (minimum-norm correction);
    # afterwards every step keeps them satisfied.
    residual = np.concatenate([b, h[working]]) - basis @ x
    x = x + np.linalg.lstsq(basis, residual, rcond=None)[0]

    def tolerances(point: np.ndarray) -> tuple[float, float, float]:
        # Taken at the CURRENT iterate, not at x_start: a start far from the
        # optimum (large |x|) must not loosen the activity and multiplier
        # tests that certify the final point.
        scale = max(1.0, float(np.abs(point).max()))
        g_scale = max(1.0, float(np.abs(c).max(initial=0.0)), float(np.abs(H @ point).max()))
        return 1e-9 * scale, 1e-9 * g_scale, scale

    for _ in range(10 * (n + G.shape[0]) + 50):
        act_tol, mult_tol, x_scale = tolerances(x)
        C = np.vstack([A, G[working]])
        solved = _solve_eqp(H, c, C, np.concatenate([b, h[working]]))
        if solved is None:
            # Singular KKT system (e.g. a singular covariance): x may still be optimal.
            return x if _kkt_certified(H, c, A, G, h, x, act_tol, mult_tol) else None
        x_eqp, lam = solved
        step = x_eqp - x
        step_size = float(np.abs(step).max())
        alpha, blocking = 1.0, -1
        if G.shape[0] and step_size > 1e-15 * x_scale:
            g_step = G @ step
            candidates = np.ones(G.shape[0], dtype=bool)
            candidates[working] = False
            candidates &= g_step < -1e-12 * step_size
            idx = np.flatnonzero(candidates)
            ratios = np.maximum(G[idx] @ x - h[idx], 0.0) / -g_step[idx]
            for k in np.argsort(ratios, kind="stable"):
                if ratios[k] >= 1.0:
                    break
                # A row dependent on the working set cannot block: C p = 0 forces
                # G_j p = 0 in exact arithmetic, so its "blocking" is rounding
                # noise at a degenerate vertex (more active constraints than
                # free dimensions). Adding it would make the KKT system singular.
                if np.linalg.matrix_rank(np.vstack([C, G[idx[k]]])) > C.shape[0]:
                    alpha, blocking = float(ratios[k]), int(idx[k])
                    break
        if blocking >= 0:
            x = x + alpha * step
            working.append(blocking)
            continue
        x = x_eqp
        act_tol, mult_tol, x_scale = tolerances(x)
        multipliers = lam[m_eq:]
        if multipliers.size == 0 or multipliers.min() >= -mult_tol:
            return x
        if _kkt_certified(H, c, A, G, h, x, act_tol, mult_tol):
            return x
        working.pop(int(np.argmin(multipliers)))
    return None


def _solve_qp(
    H: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    x0: np.ndarray,
    label: str,
    G: np.ndarray | None = None,
    h: np.ndarray | None = None,
) -> np.ndarray:
    """Solve min ½x'Hx - c'x s.t. Ax = b, lo <= x <= hi, Gx >= h (H PSD, scaled O(1)).

    SLSQP with analytic gradients and Jacobians from the feasible start x0,
    then an exact active-set refinement warm-started at the SLSQP point (or at
    x0 if SLSQP wandered off). A refined point that is feasible is returned;
    otherwise a successful, feasible SLSQP point; otherwise OptimizationError.
    """
    n = x0.size
    if G is None or h is None:
        G, h = np.empty((0, n)), np.empty(0)

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        hx = H @ x
        return 0.5 * float(x @ hx) - float(c @ x), hx - c

    constraints = [{"type": "eq", "fun": lambda x: A @ x - b, "jac": lambda x: A}]
    if G.shape[0]:
        constraints.append({"type": "ineq", "fun": lambda x: G @ x - h, "jac": lambda x: G})
    box = [
        (None if np.isneginf(a) else float(a), None if np.isposinf(z) else float(z))
        for a, z in zip(lo, hi, strict=True)
    ]
    # SciPy may warn about clipping iterates into the bounds; the result is
    # verified below, so its warnings carry no information we need.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = optimize.minimize(
            objective,
            x0,
            jac=True,
            method="SLSQP",
            bounds=box,
            constraints=constraints,
            options=dict(_SLSQP_OPTIONS),
        )
    x_slsqp = np.asarray(res.x, dtype=float)

    box_G, box_h = _box_rows(lo, hi)
    G_all, h_all = np.vstack([box_G, G]), np.concatenate([box_h, h])

    def violation(x: np.ndarray) -> float:
        worst = float(np.abs(A @ x - b).max())
        if G_all.shape[0]:
            worst = max(worst, float(np.max(h_all - G_all @ x)))
        return worst

    starts = []
    if np.isfinite(x_slsqp).all():
        clipped = np.clip(x_slsqp, lo, hi)
        if violation(clipped) <= 1e-6 * max(1.0, float(np.abs(clipped).max())):
            starts.append(clipped)
    starts.append(x0)
    for start in starts:
        x = _active_set_qp(H, c, A, b, G_all, h_all, start)
        if x is not None and violation(x) <= 1e-9 * max(1.0, float(np.abs(x).max())):
            return np.clip(x, lo, hi)
    if res.success and np.isfinite(x_slsqp).all():
        x = np.clip(x_slsqp, lo, hi)
        if violation(x) <= _FEAS_TOL * max(1.0, float(np.abs(x).max())):
            return x
    raise OptimizationError(
        f"{label}: SLSQP failed ({res.message}) and the active-set refinement did not converge"
    )


def _finalize_weights(w: np.ndarray, lo: np.ndarray, hi: np.ndarray, label: str) -> np.ndarray:
    """Clip tiny bound violations, restore the budget, and verify the result."""
    w = np.clip(w, lo, hi)
    if not np.isfinite(w).all() or abs(1.0 - float(w.sum())) > _FEAS_TOL:
        raise OptimizationError(f"{label}: solution violates the budget constraint")
    w = _repair_budget(w, lo, hi)
    if abs(1.0 - float(w.sum())) > 1e-10:
        raise OptimizationError(f"{label}: could not restore the budget constraint")
    return w


# ---------------------------------------------------------------------------
# Problem cores (validated inputs in, weight arrays out)
# ---------------------------------------------------------------------------


def _min_variance_core(prob: _Problem) -> np.ndarray:
    if not prob.bounded:
        chol = _cholesky_pd(prob.cov, "min_variance")
        return _normalized(_mvp_closed_form(prob.cov, chol), "min_variance")
    scale = _check_psd(prob.cov)
    x0 = _feasible_center(prob.lo, prob.hi)
    x = _solve_qp(
        prob.cov / scale,
        np.zeros(prob.n),
        np.ones((1, prob.n)),
        np.ones(1),
        prob.lo,
        prob.hi,
        x0,
        "min_variance",
    )
    return _finalize_weights(x, prob.lo, prob.hi, "min_variance")


@dataclass(frozen=True)
class _ReturnRange:
    center: np.ndarray
    top: _Extreme
    bottom: _Extreme


def _return_range(prob: _Problem, mu: np.ndarray) -> _ReturnRange:
    return _ReturnRange(
        _feasible_center(prob.lo, prob.hi),
        _return_extreme(mu, prob.lo, prob.hi, +1.0),
        _return_extreme(mu, prob.lo, prob.hi, -1.0),
    )


def _efficient_return_core(
    prob: _Problem, target: float, attainable: _ReturnRange | None = None
) -> np.ndarray:
    mu = prob.mu
    tol = _return_tol(mu)
    if _mu_is_constant(mu):
        if abs(target - float(mu.mean())) > tol:
            raise ValueError(
                f"target return {target} is unattainable: every asset has expected "
                f"return {float(mu.mean())}"
            )
        return _min_variance_core(prob)
    if not prob.bounded:
        chol = _cholesky_pd(prob.cov, "efficient_return")
        tf = _two_fund(prob.cov, mu, chol)
        return _normalized(tf.w_mvp + ((target - tf.r_mvp) / tf.q) * tf.v, "efficient_return")

    scale = _check_psd(prob.cov)
    rng = attainable if attainable is not None else _return_range(prob, mu)
    r_max, r_min = rng.top.value, rng.bottom.value
    if target > r_max + tol or target < r_min - tol:
        raise ValueError(
            f"target return {target} is unattainable under the bounds; "
            f"attainable range is [{r_min}, {r_max}]"
        )
    target = min(max(target, r_min), r_max)
    x0 = _point_with_return(mu, target, rng.center, rng.top, rng.bottom)
    # Scale the return row like the budget row so both equality residuals are O(1).
    mu_scale = float(np.abs(mu).max())
    x = _solve_qp(
        prob.cov / scale,
        np.zeros(prob.n),
        np.vstack([np.ones(prob.n), mu / mu_scale]),
        np.array([1.0, target / mu_scale]),
        prob.lo,
        prob.hi,
        x0,
        "efficient_return",
    )
    w = _finalize_weights(x, prob.lo, prob.hi, "efficient_return")
    if abs(float(mu @ w) - target) > 1e-8 * max(mu_scale, abs(target)):
        raise OptimizationError("efficient_return: solution misses the target return")
    return w


def _volatility(w: np.ndarray, cov: np.ndarray) -> float:
    return math.sqrt(max(float(w @ cov @ w), 0.0))


def _efficient_mvp(
    prob: _Problem, w_mvp: np.ndarray, attainable: _ReturnRange | None = None
) -> tuple[float, np.ndarray]:
    """The highest-return portfolio among the minimum-variance portfolios.

    A positive definite covariance has a unique minimum-variance portfolio, which is
    returned unchanged. A singular one (T < N sample covariance, or two assets with
    identical risk) has a whole set of them, and the solver may return any member;
    only the member with the highest return is efficient, the rest are dominated.
    Beyond ``r_mvp`` the frontier variance is convex and non-decreasing in the target
    return, so it stays at its minimum on ``[r_mvp, r_low]``: bisection finds
    ``r_low``, the efficient end of that flat stretch.

    Returns:
        ``(r_low, w_low)``.
    """
    r_mvp = float(prob.mu @ w_mvp)
    if not prob.bounded or _mu_is_constant(prob.mu) or not _is_singular(prob.cov):
        return r_mvp, w_mvp
    cov = prob.cov
    var_cap = float(w_mvp @ cov @ w_mvp) * (1.0 + 1e-9) + _variance_noise(cov)
    rng = attainable if attainable is not None else _return_range(prob, prob.mu)

    def on_flat(r: float) -> tuple[bool, np.ndarray]:
        w = _efficient_return_core(prob, r, rng)
        return float(w @ cov @ w) <= var_cap, w

    r_hi = rng.top.value
    if not math.isfinite(r_hi):
        step = max(float(np.ptp(prob.mu)), 1e-12)
        for _ in range(200):
            r_hi = r_mvp + step
            if not on_flat(r_hi)[0]:
                break
            step *= 2.0
        else:
            raise OptimizationError("could not bound the minimum-variance set")
    flat, w_hi = on_flat(r_hi)
    if flat:
        return r_hi, w_hi
    r_lo, w_lo = r_mvp, w_mvp
    resolution = 1e-12 * max(r_hi - r_mvp, abs(r_hi), float(np.finfo(float).tiny))
    while r_hi - r_lo > resolution:
        mid = 0.5 * (r_lo + r_hi)
        flat, w = on_flat(mid)
        if flat:
            r_lo, w_lo = mid, w
        else:
            r_hi = mid
    return r_lo, w_lo


def _efficient_risk_core(prob: _Problem, target: float) -> np.ndarray:
    mu, cov = prob.mu, prob.cov
    if not prob.bounded:
        chol = _cholesky_pd(prob.cov, "efficient_risk")
        tf = _two_fund(cov, mu, chol)
        vol_mvp = math.sqrt(tf.var_mvp)
        _check_risk_target(target, vol_mvp)
        if _mu_is_constant(mu) or tf.q == 0.0:
            return tf.w_mvp
        # On the frontier σ² = σ²_mvp + (r - r_mvp)²/q; take the upper branch.
        shift = math.sqrt(max(target * target - tf.var_mvp, 0.0) / tf.q)
        return _normalized(tf.w_mvp + shift * tf.v, "efficient_risk")

    w_mvp = _min_variance_core(prob)
    vol_mvp = _volatility(w_mvp, cov)
    _check_risk_target(target, vol_mvp)
    attainable = _return_range(prob, mu)
    # With a singular covariance the solver's minimum-variance portfolio may be a
    # dominated member of the minimum-variance set; start from its efficient end.
    r_mvp, w_mvp = _efficient_mvp(prob, w_mvp, attainable)
    if target <= vol_mvp * (1.0 + 1e-12):
        return w_mvp
    vol_noise = math.sqrt(_variance_noise(cov))
    # The best evaluated frontier point within the risk budget (highest return
    # with σ <= target), so the answer never depends on which side of the
    # root the root finder happens to stop.
    best_r, best_w = r_mvp, w_mvp

    def frontier_vol(r: float) -> float:
        nonlocal best_r, best_w
        w_r = _efficient_return_core(prob, r, attainable)
        vol = _volatility(w_r, cov)
        if vol <= target and r >= best_r:
            best_r, best_w = r, w_r
        return vol

    r_max = attainable.top.value
    if math.isfinite(r_max):
        w_top = _efficient_return_core(prob, r_max, attainable)
        if _volatility(w_top, cov) <= target:
            # The risk budget does not bind: the best attainable return is affordable.
            return w_top
        r_high = r_max
    else:
        step = max(float(np.ptp(mu)), float(np.abs(mu).max()), 1e-12)
        for _ in range(200):
            r_high = r_mvp + step
            if frontier_vol(r_high) > target:
                break
            step *= 2.0
        else:
            raise OptimizationError("efficient_risk: could not bracket the target volatility")
    # σ(r) is continuous and increasing on [r_mvp, r_high] (the minimum
    # variance at a given return is a convex function of that return), so
    # the root is unique. The tolerance is RELATIVE to the size of the
    # returns, i.e. a few ulps of r: an absolute xtol would be loose for
    # daily-scale or nearly equal returns, where σ changes fast per unit of
    # return, and the volatility at the root would then miss the target.
    xtol = max(4.0 * _EPS * max(abs(r_mvp), abs(r_high)), float(np.finfo(float).tiny))
    if frontier_vol(r_mvp) >= target:
        # Only possible when the target is within solver noise of the minimum
        # volatility (singular covariance): the bracket [r_mvp, r_high] has no sign
        # change, and the efficient end of the minimum-variance set is the answer.
        w = w_mvp
    else:
        r_star = optimize.brentq(
            lambda r: frontier_vol(r) - target,
            r_mvp,
            r_high,
            xtol=xtol,
            rtol=4 * _EPS,
            maxiter=200,
        )
        w = _efficient_return_core(prob, r_star, attainable)
        if _volatility(w, cov) > target:
            # The root finder stopped a rounding step above the root: take the
            # closest evaluated point below it, which respects the risk budget.
            w = best_w
    if _volatility(w, cov) > target * (1.0 + 1e-9) + vol_noise:
        raise OptimizationError("efficient_risk: solution exceeds the target volatility")
    return w


def _check_risk_target(target: float, vol_mvp: float) -> None:
    if target < vol_mvp * (1.0 - 1e-10):
        raise ValueError(
            f"target volatility {target} is below the minimum-variance volatility {vol_mvp}"
        )


def _utility_core(prob: _Problem, risk_aversion: float) -> np.ndarray:
    mu, cov = prob.mu, prob.cov
    if not prob.bounded:
        chol = _cholesky_pd(cov, "mean_variance_utility")
        tf = _two_fund(cov, mu, chol)
        return _normalized(tf.w_mvp + tf.v / risk_aversion, "mean_variance_utility")
    scale = _check_psd(cov)
    # Divide the objective by κ so both the quadratic and the linear term are O(1).
    kappa = max(risk_aversion * scale, float(np.abs(mu).max()))
    x = _solve_qp(
        (risk_aversion / kappa) * cov,
        mu / kappa,
        np.ones((1, prob.n)),
        np.ones(1),
        prob.lo,
        prob.hi,
        _feasible_center(prob.lo, prob.hi),
        "mean_variance_utility",
    )
    return _finalize_weights(x, prob.lo, prob.hi, "mean_variance_utility")


def _max_sharpe_core(prob: _Problem, rf: float) -> np.ndarray:
    mu, cov = prob.mu, prob.cov
    excess = mu - rf
    if not prob.bounded:
        chol = _cholesky_pd(cov, "max_sharpe")
        z = linalg.cho_solve((chol, True), excess)
        total = float(z.sum())
        if not total > 1e-12 * float(np.abs(z).sum()):
            raise ValueError(
                "tangency portfolio is undefined: 1'Σ⁻¹(μ - rf) <= 0, so rf is at or above "
                "the minimum-variance return and the line from rf only touches the "
                "inefficient branch of the frontier"
            )
        return _normalized(z / total, "max_sharpe")

    scale = _check_psd(cov)
    lo, hi = prob.lo, prob.hi
    rng = _return_range(prob, mu)
    best_excess = rng.top.value - rf
    if not best_excess > _return_tol(mu):
        raise ValueError(
            f"no feasible portfolio has expected return above rf={rf} (best attainable "
            f"{rng.top.value}); the maximum-Sharpe portfolio is undefined"
        )
    if best_excess <= _DEGENERATE_EXCESS * max(float(np.abs(mu).max()), abs(rf)):
        # rf is a rounding distance below the highest attainable return. The QP
        # below then needs y ~ 1/excess and is too ill-conditioned to solve, but
        # the answer is known: as rf rises to r_max only portfolios near r_max
        # keep a positive excess, and among them the Sharpe ratio is maximized
        # at r_max itself, by its minimum-variance portfolio.
        return _efficient_return_core(prob, rng.top.value, rng)

    # Feasible start with a comfortably positive excess return. The QP start
    # is y0 = w0 / (ê'w0), so ê'w0 must not be tiny: a centre whose excess is
    # rounding noise (rf ≈ its return, e.g. rf = mean(μ) with long-only
    # bounds) would give |y0| ~ 1e15, and SLSQP then fails ("Inequality
    # constraints incompatible"). Otherwise aim halfway to the best excess.
    center_excess = float(excess @ rng.center)
    if math.isfinite(best_excess):
        wanted = 0.5 * best_excess
    else:
        # Unbounded return range (an improving ray exists, so ptp(μ) > 0):
        # any excess on the scale of the spread of μ will do.
        wanted = max(center_excess, float(np.ptp(mu)))
    if center_excess >= 0.5 * wanted:
        w0 = rng.center
    else:
        w0 = _point_with_return(mu, rf + wanted, rng.center, rng.top, rng.bottom)

    # Cornuejols & Tütüncü (2007, §8.2): with y = w / (excess'w) and
    # k = 1'y > 0, maximizing the Sharpe ratio is the convex QP
    #     min y'Σy  s.t.  excess'y = 1,  lo_i k <= y_i <= hi_i k,  k >= 0.
    # The excess vector is rescaled to max |.| = 1 so that y is O(1).
    e_hat = excess / float(np.abs(excess).max())
    y0 = w0 / float(e_hat @ w0)
    n = prob.n
    eye, ones = np.eye(n), np.ones(n)
    rows, y_lo, y_hi = [], np.full(n, -np.inf), np.full(n, np.inf)
    for i in range(n):
        # Sign constraints are exact consequences of k >= 0; as variable bounds
        # SLSQP keeps them satisfied at every iterate.
        if lo[i] >= 0.0:
            y_lo[i] = 0.0
        if hi[i] <= 0.0:
            y_hi[i] = 0.0
        if np.isfinite(lo[i]) and lo[i] != 0.0:
            rows.append(eye[i] - lo[i] * ones)  # y_i - lo_i k >= 0
        if np.isfinite(hi[i]) and hi[i] != 0.0:
            rows.append(hi[i] * ones - eye[i])  # hi_i k - y_i >= 0
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        rows.append(ones)  # k >= 0 (implied when every bound is finite)
    G = np.array(rows).reshape(-1, n)
    y = _solve_qp(
        cov / scale,
        np.zeros(n),
        e_hat[None, :],
        np.ones(1),
        y_lo,
        y_hi,
        y0,
        "max_sharpe",
        G=G,
        h=np.zeros(G.shape[0]),
    )
    k = float(y.sum())
    if not k > 1e-12 * float(np.abs(y).sum()):
        raise ValueError(
            "maximum Sharpe ratio is only approached by an infinitely leveraged "
            "zero-cost portfolio under these bounds"
        )
    w = _finalize_weights(y / k, lo, hi, "max_sharpe")
    # A volatility below 1e-5 of the average asset volatility is numerically
    # zero: the covariance is singular (e.g. T < N) and admits a (near-)riskless
    # portfolio with positive excess return, i.e. an unbounded Sharpe ratio.
    if not float(w @ cov @ w) > 1e-10 * scale:
        raise ValueError(
            "covariance is singular: a riskless portfolio with positive excess return "
            "exists, so the Sharpe ratio is unbounded"
        )
    return w


# ---------------------------------------------------------------------------
# Public API: portfolio statistics
# ---------------------------------------------------------------------------


def portfolio_return(weights: VectorLike, mu: VectorLike) -> float:
    """Expected portfolio return ``w'μ``.

    Args:
        weights: portfolio weights (Series aligned by label, or array).
        mu: expected (annualized) returns per asset.

    Returns:
        The expected portfolio return in the units of ``mu``.

    Raises:
        ValueError: on length or label mismatch or non-finite values.
    """
    if isinstance(mu, pd.Series):
        assets = mu.index
    elif isinstance(weights, pd.Series):
        assets = weights.index
    else:
        assets = pd.RangeIndex(np.asarray(mu, dtype=float).reshape(-1).shape[0])
    if not pd.Index(assets).is_unique:
        raise ValueError("asset labels must be unique")
    m = _vector(mu, assets, "mu")
    w = _vector(weights, assets, "weights")
    return float(w @ m)


def portfolio_volatility(weights: VectorLike, cov: MatrixLike) -> float:
    """Portfolio volatility ``σ_p = sqrt(w'Σw)``.

    Args:
        weights: portfolio weights (Series aligned to the covariance labels, or array).
        cov: (annualized) covariance matrix.

    Returns:
        The portfolio standard deviation in the units of ``sqrt(cov)``.

    Raises:
        ValueError: on shape/label mismatch, non-finite values, or if
            ``w'Σw`` is materially negative (``cov`` not positive semi-definite).
    """
    frame = _cov_frame(cov, weights)
    w = _vector(weights, frame.index, "weights")
    values = frame.to_numpy()
    variance = float(w @ values @ w)
    if variance < 0.0:
        magnitude = float(np.abs(w) @ np.abs(values) @ np.abs(w))
        if variance < -1e-12 * magnitude:
            raise ValueError(f"w'Σw = {variance:.3g} < 0: covariance is not positive semi-definite")
        variance = 0.0  # rounding noise of a (near-)riskless portfolio
    return math.sqrt(variance)


def portfolio_sharpe(
    weights: VectorLike, mu: VectorLike, cov: MatrixLike, rf: float = 0.0
) -> float:
    """Sharpe ratio ``(w'μ - rf) / sqrt(w'Σw)`` (Sharpe 1966).

    Args:
        weights: portfolio weights.
        mu: expected (annualized) returns.
        cov: (annualized) covariance matrix.
        rf: (annualized) risk-free rate.

    Returns:
        The Sharpe ratio. A riskless portfolio gives ``+inf``/``-inf`` for a
        positive/negative excess return and ``nan`` for zero excess return.

    Raises:
        ValueError: on invalid or misaligned inputs.
    """
    rf = _scalar(rf, "rf")
    frame = _cov_frame(cov, mu, weights)
    assets = frame.index
    m = _vector(mu, assets, "mu")
    w = _vector(weights, assets, "weights")
    excess = float(w @ m) - rf
    vol = portfolio_volatility(w, frame.to_numpy())
    if vol == 0.0:
        return math.copysign(math.inf, excess) if excess != 0.0 else math.nan
    return excess / vol


# ---------------------------------------------------------------------------
# Public API: optimizers
# ---------------------------------------------------------------------------


def min_variance(cov: MatrixLike, bounds: Bounds = (0.0, 1.0)) -> pd.Series:
    """Global minimum-variance portfolio: ``min w'Σw  s.t.  1'w = 1`` (+ bounds).

    With ``bounds=None`` the Lagrangian ``w'Σw - 2λ(1'w - 1)`` gives the closed
    form (Merton 1972)::

        w = Σ⁻¹1 / (1'Σ⁻¹1),        σ² = 1 / (1'Σ⁻¹1),

    computed with a Cholesky solve. With bounds the convex QP is solved by
    SLSQP and certified/refined by an active-set step (see module docstring).
    It needs no expected returns, which is why it is far less exposed to
    estimation error than the other mean-variance portfolios.

    Args:
        cov: (annualized) covariance matrix, numpy or labelled DataFrame.
        bounds: ``(lo, hi)`` for every asset, one pair per asset, or None
            (unbounded). Default long-only ``(0, 1)``.

    Returns:
        Weights indexed by asset, summing to 1.

    Raises:
        ValueError: invalid input, infeasible bounds, a singular covariance with
            ``bounds=None``, or a covariance that is not positive semi-definite.
        OptimizationError: if the numerical solver fails.
    """
    prob = _problem(cov, None, bounds)
    return _weights_series(_min_variance_core(prob), prob)


def max_sharpe(
    mu: VectorLike, cov: MatrixLike, rf: float = 0.0, bounds: Bounds = (0.0, 1.0)
) -> pd.Series:
    """Tangency (maximum Sharpe ratio) portfolio: ``max (w'μ - rf)/sqrt(w'Σw)``, ``1'w = 1``.

    ``bounds=None``: the first-order condition of the scale-invariant Sharpe
    ratio gives ``Σw ∝ μ - rf·1`` (Tobin 1958; Sharpe 1966), so::

        w = Σ⁻¹(μ - rf 1) / 1'Σ⁻¹(μ - rf 1),   max Sharpe = sqrt((μ - rf 1)'Σ⁻¹(μ - rf 1)).

    If ``1'Σ⁻¹(μ - rf 1) <= 0`` (rf at or above the minimum-variance return)
    the line from rf touches only the inefficient branch and no tangency
    exists, so ValueError is raised.

    With bounds, the convex reformulation of Cornuejols & Tütüncü (2007, §8.2)
    is solved: with ``y = w / (μ - rf)'w`` and ``k = 1'y``,

        min y'Σy  s.t.  (μ - rf 1)'y = 1,  lo_i k <= y_i <= hi_i k,  k >= 0,

    then ``w = y / k``. It is valid whenever some feasible portfolio has a
    positive excess return (for long-only bounds: some ``μ_i > rf``).

    Args:
        mu: expected (annualized) returns.
        cov: (annualized) covariance matrix.
        rf: (annualized) risk-free rate.
        bounds: weight bounds (default long-only), or None for unbounded.

    Returns:
        Weights indexed by asset, summing to 1.

    Raises:
        ValueError: invalid input; no feasible portfolio beats ``rf``; the
            unbounded tangency is undefined; or a riskless portfolio with a
            positive excess return exists (unbounded Sharpe ratio).
        OptimizationError: if the numerical solver fails.
    """
    rf = _scalar(rf, "rf")
    prob = _problem(cov, mu, bounds)
    return _weights_series(_max_sharpe_core(prob, rf), prob)


def efficient_return(
    mu: VectorLike, cov: MatrixLike, target_return: float, bounds: Bounds = (0.0, 1.0)
) -> pd.Series:
    """Minimum-variance portfolio for a target return: ``min w'Σw  s.t.  w'μ = r*, 1'w = 1``.

    ``bounds=None``: Merton's (1972) two-fund solution. With ``A = 1'Σ⁻¹1``,
    ``B = 1'Σ⁻¹μ``, ``C = μ'Σ⁻¹μ``, ``D = AC - B²``::

        w = [(C - B r*) Σ⁻¹1 + (A r* - B) Σ⁻¹μ] / D,   σ² = (A r*² - 2B r* + C) / D,

    evaluated in the cancellation-free centred form
    ``w = w_mvp + (r* - B/A) Σ⁻¹(μ - (B/A)1) / q`` with ``q = D/A``. Targets
    below ``B/A`` are allowed (they lie on the inefficient branch).

    With bounds the convex QP is solved numerically; the target must lie in
    the attainable range ``[min μ'w, max μ'w]`` over the feasible set (for
    long-only bounds ``[min μ, max μ]``).

    Args:
        mu: expected (annualized) returns.
        cov: (annualized) covariance matrix.
        target_return: required (annualized) portfolio return.
        bounds: weight bounds (default long-only), or None for unbounded.

    Returns:
        Weights indexed by asset, summing to 1, with ``w'μ = target_return``.

    Raises:
        ValueError: invalid input or an unattainable target (including any
            target other than the common return when all ``μ_i`` are equal).
        OptimizationError: if the numerical solver fails.
    """
    target = _scalar(target_return, "target_return")
    prob = _problem(cov, mu, bounds)
    return _weights_series(_efficient_return_core(prob, target), prob)


def efficient_risk(
    mu: VectorLike, cov: MatrixLike, target_volatility: float, bounds: Bounds = (0.0, 1.0)
) -> pd.Series:
    """Maximum-return portfolio for a risk budget: ``max w'μ  s.t.  sqrt(w'Σw) <= σ*, 1'w = 1``.

    ``bounds=None``: the constraint binds and, inverting the frontier
    hyperbola ``σ² = 1/A + (r - B/A)²/q``, the solution is the upper-branch
    frontier portfolio with return ``r = B/A + sqrt(q (σ*² - 1/A))``.

    With bounds the solution is the bounded efficient-frontier portfolio whose
    volatility equals ``σ*`` (found by Brent's method on the return, since the
    frontier volatility is increasing above the minimum-variance return, to a
    few ulps of the return; if the volatility there is a rounding step above
    ``σ*``, the closest evaluated frontier point inside the budget is taken),
    or, if ``σ*`` exceeds the volatility of the least risky maximum-return
    portfolio, that portfolio (the risk budget is then not binding).

    Args:
        mu: expected (annualized) returns.
        cov: (annualized) covariance matrix.
        target_volatility: (annualized) volatility budget, > 0.
        bounds: weight bounds (default long-only), or None for unbounded.

    Returns:
        Weights indexed by asset, summing to 1.

    Raises:
        ValueError: invalid input, or a target below the minimum-variance volatility.
        OptimizationError: if the numerical solver fails.
    """
    target = _scalar(target_volatility, "target_volatility")
    if not target > 0.0:
        raise ValueError(f"target_volatility must be positive, got {target_volatility!r}")
    prob = _problem(cov, mu, bounds)
    return _weights_series(_efficient_risk_core(prob, target), prob)


def mean_variance_utility(
    mu: VectorLike, cov: MatrixLike, risk_aversion: float, bounds: Bounds = (0.0, 1.0)
) -> pd.Series:
    """Quadratic-utility portfolio: ``max w'μ - (δ/2) w'Σw  s.t.  1'w = 1`` (+ bounds).

    ``bounds=None``: the first-order condition ``μ - δΣw - λ1 = 0`` with the
    budget gives ``λ = (B - δ)/A`` and::

        w = Σ⁻¹(μ - λ1)/δ = w_mvp + Σ⁻¹(μ - (B/A)1) / δ,

    the minimum-variance portfolio plus ``1/δ`` units of the zero-cost
    "speculative" portfolio (Merton 1972). Expected return
    ``B/A + q/δ`` with ``q = D/A``; ``δ → ∞`` recovers ``w_mvp``.

    Args:
        mu: expected (annualized) returns.
        cov: (annualized) covariance matrix.
        risk_aversion: δ > 0.
        bounds: weight bounds (default long-only), or None for unbounded.

    Returns:
        Weights indexed by asset, summing to 1.

    Raises:
        ValueError: invalid input or non-positive ``risk_aversion``.
        OptimizationError: if the numerical solver fails.
    """
    delta = _scalar(risk_aversion, "risk_aversion")
    if not delta > 0.0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion!r}")
    prob = _problem(cov, mu, bounds)
    return _weights_series(_utility_core(prob, delta), prob)


def efficient_frontier(
    mu: VectorLike,
    cov: MatrixLike,
    n_points: int = 50,
    bounds: Bounds = (0.0, 1.0),
    rf: float = 0.0,
    return_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Efficient (upper) branch of the mean-variance frontier.

    Each row is ``efficient_return(mu, cov, r, bounds)`` for ``n_points``
    equally spaced target returns ``r``. By default ``r`` runs from the
    minimum-variance portfolio's return ``r_mvp`` (for a singular covariance,
    the highest-return portfolio among the minimum-variance portfolios, so no
    row is dominated) up to the highest attainable
    return under the bounds (``max μ`` for long-only bounds); for
    ``bounds=None`` (or bounds that leave the return unbounded) the upper end is
    ``max μ + (max μ - r_mvp)``. Along this branch the volatility is
    non-decreasing and the variance is a convex function of the return (it is
    the value function of a parametric convex QP).

    Args:
        mu: expected (annualized) returns.
        cov: (annualized) covariance matrix.
        n_points: number of frontier points (>= 2).
        bounds: weight bounds (default long-only), or None for unbounded.
        rf: (annualized) risk-free rate for the Sharpe ratio column.
        return_range: optional ``(low, high)`` target-return range; ``low``
            must not be below the minimum-variance return (that part of the
            curve is the inefficient branch) and ``high`` must be attainable.

    Returns:
        DataFrame with one row per point, sorted by return, and columns
        ``"return"``, ``"volatility"``, ``"sharpe"`` followed by one weight
        column per asset.

    Raises:
        ValueError: invalid input, an invalid or unattainable ``return_range``,
            or an asset label that clashes with a statistics column.
        OptimizationError: if the numerical solver fails.
    """
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError(f"n_points must be an integer, got {n_points!r}")
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    rf = _scalar(rf, "rf")
    prob = _problem(cov, mu, bounds)
    clash = [a for a in prob.assets if isinstance(a, str) and a in _FRONTIER_COLUMNS]
    if clash:
        raise ValueError(f"asset labels clash with frontier statistic columns: {clash}")
    mu_values, cov_values = prob.mu, prob.cov

    attainable = _return_range(prob, mu_values) if prob.bounded else None
    # With a singular covariance, start at the efficient end of the minimum-variance
    # set; the portfolios before it have the same risk and less return (dominated).
    r_mvp, w_mvp = _efficient_mvp(prob, _min_variance_core(prob), attainable)
    tol = _return_tol(mu_values)
    if return_range is None:
        low = r_mvp
        if attainable is not None and math.isfinite(attainable.top.value):
            high = max(attainable.top.value, r_mvp)
        else:
            span = float(mu_values.max()) - r_mvp
            if span <= 0.0:
                span = float(np.ptp(mu_values))
            high = r_mvp + 2.0 * span
    else:
        try:
            pair = list(return_range)
        except TypeError:
            pair = []  # a scalar (or other non-iterable) is not a pair
        if len(pair) != 2:
            raise ValueError(f"return_range must be a (low, high) pair, got {return_range!r}")
        low = _scalar(pair[0], "return_range[0]")
        high = _scalar(pair[1], "return_range[1]")
        if low > high:
            raise ValueError(f"return_range low {low} exceeds high {high}")
        if low < r_mvp - tol:
            raise ValueError(
                f"return_range starts at {low}, below the minimum-variance return {r_mvp}: "
                "that part of the curve is the inefficient branch"
            )
        low = max(low, r_mvp)
        high = max(high, low)

    targets = np.linspace(low, high, n_points)
    rows = []
    for target in targets:
        if abs(target - r_mvp) <= tol:
            w = w_mvp
        else:
            w = _efficient_return_core(prob, float(target), attainable)
        rows.append(w)
    weights = np.vstack(rows)
    returns = weights @ mu_values
    variances = np.einsum("ij,jk,ik->i", weights, cov_values, weights)
    # Along the efficient branch the minimum variance is non-decreasing in the
    # target return. Allow only rounding-level decreases: they occur on flat
    # segments of a singular covariance (e.g. T < N sample covariance), where
    # many portfolios share the same (possibly zero) variance.
    noise = 1e-10 * float(np.trace(cov_values)) / prob.n
    noise *= max(1.0, float(np.abs(weights).sum(axis=1).max()) ** 2)
    if np.any(np.diff(variances) < -noise):
        raise OptimizationError("efficient_frontier: variance decreases along the frontier")
    vols = np.maximum.accumulate(np.sqrt(np.maximum(variances, 0.0)))
    excess = returns - rf
    riskless = vols == 0.0
    sharpe = excess / np.where(riskless, 1.0, vols)
    # Riskless points: avoid 0 * inf (a RuntimeWarning) by assigning each case.
    sharpe[riskless & (excess > 0.0)] = math.inf
    sharpe[riskless & (excess < 0.0)] = -math.inf
    sharpe[riskless & (excess == 0.0)] = math.nan

    frame = pd.DataFrame(weights, columns=prob.assets.copy())
    frame.insert(0, "sharpe", sharpe)
    frame.insert(0, "volatility", vols)
    frame.insert(0, "return", returns)
    return frame
