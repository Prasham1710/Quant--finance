"""Risk-based allocation: risk contributions, inverse-volatility and risk parity.

Conventions follow :mod:`quantfin.portfolio._common`: ``cov`` is an ANNUALIZED
covariance matrix (any common scale works -- every allocation here is invariant
to multiplying ``cov`` by a positive constant), weights are a ``pd.Series``
indexed by asset that sums to 1, and numpy input is labelled 0..N-1.

Risk decomposition (Euler allocation). Portfolio volatility
``σ(w) = sqrt(w'Σw)`` is positively homogeneous of degree one, so Euler's
theorem splits it exactly into per-asset contributions::

    σ(w) = Σ_i w_i ∂σ/∂w_i = Σ_i RC_i,       RC_i = w_i (Σw)_i / σ(w).

``RC_i / σ(w) = w_i (Σw)_i / w'Σw`` is asset ``i``'s share of portfolio
variance (Litterman 1996; Qian 2006 shows it is also the expected share of a
portfolio loss attributable to asset ``i``).

Allocations that need no expected-return estimate:

* inverse volatility ``w_i ∝ 1/σ_i`` -- the risk-parity portfolio when all
  correlations are equal;
* inverse variance ``w_i ∝ 1/σ_i²`` -- the minimum-variance portfolio when all
  correlations are zero;
* risk budgeting / equal risk contribution (Maillard, Roncalli & Teiletche
  2010; Bruder & Roncalli 2012): the long-only portfolio whose variance shares
  equal prescribed budgets ``b`` (``b_i = 1/N`` for ERC).

Why these matter for the package's key lesson: a mean-variance optimizer
applies ``Σ⁻¹`` to the expected returns ``μ``, which are estimated with huge
relative error, so it amplifies that error into extreme, unstable weights.
Risk-based portfolios never use ``μ`` and depend on ``Σ`` only through
quantities that are estimated far more precisely, so they typically hold up
much better out of sample. Their volatility always lies between the
minimum-variance and the equally weighted portfolio's (Maillard et al. 2010).

References:
    R. Litterman (1996), "Hot Spots and Hedges", Goldman Sachs Risk Management
    Series.
    E. Qian (2006), "On the financial interpretation of risk contribution: risk
    budgets do add up", Journal of Investment Management 4(4), 41-51.
    S. Maillard, T. Roncalli and J. Teiletche (2010), "The properties of
    equally weighted risk contribution portfolios", Journal of Portfolio
    Management 36(4), 60-70.
    B. Bruder and T. Roncalli (2012), "Managing risk exposures using the risk
    budgeting approach", SSRN 2009778.
    F. Spinu (2013), "An algorithm for computing risk parity weights",
    SSRN 2297383.
    M. López de Prado (2016), "Building diversified portfolios that outperform
    out of sample", Journal of Portfolio Management 42(4), 59-69.
    Y. Nesterov (2004), "Introductory Lectures on Convex Optimization",
    Kluwer, Section 4.1 (self-concordant functions, damped Newton method).
    S. Boyd and L. Vandenberghe (2004), "Convex Optimization", Cambridge
    University Press, Sections 9.2-9.6 (backtracking line search, Newton
    decrement, self-concordance).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import linalg

from quantfin.portfolio._common import OptimizationError, as_cov_frame, as_series

__all__ = [
    "equal_risk_contribution",
    "inverse_variance",
    "inverse_volatility",
    "risk_contributions",
]

_EPS = float(np.finfo(float).eps)

# The computed portfolio variance w'Σw carries a rounding error of roughly
# N * eps * |w|'|Σ||w|. A variance below a generous multiple of that bound is
# indistinguishable from zero, and dividing by it would return pure noise.
_ZERO_VARIANCE_FACTOR = 64.0

# A valid correlation matrix is PSD; its eigenvalues lie in [0, N]. Negative
# eigenvalues smaller in magnitude than this (times N) are rounding noise from
# however the matrix was built; anything more negative is not a covariance.
_PSD_TOL = 1e-10

# Final, independent acceptance test from the spec: every relative risk
# contribution must match its budget to this accuracy.
_BUDGET_CHECK_TOL = 1e-8

# As for the variance above, the computed risk-budget residual y_i (Σy)_i - b_i
# is only accurate to about N * eps * y_i (|Σ||y|)_i, so the iteration stops
# once the residual is within a multiple of that floor even if ``tol`` is
# tighter than double precision can deliver for an ill-conditioned matrix
# (strong negative correlations make (Σy)_i a cancelling sum).
_RESIDUAL_ROUNDING_FACTOR = 8.0

# Backtracking line search parameters (Boyd & Vandenberghe 2004, Alg. 9.2).
_ARMIJO_C = 1e-4
_MAX_HALVINGS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weights_for(weights: pd.Series | np.ndarray | Sequence[float], cov: pd.DataFrame) -> pd.Series:
    """Align ``weights`` to the covariance labels, rejecting unknown assets."""
    if isinstance(weights, pd.Series):
        known = set(cov.index)
        extra = [a for a in weights.index if a not in known]
        if extra:
            raise ValueError(f"weights contain assets not in the covariance matrix: {extra}")
    return as_series(weights, cov.index)


def _positive_variance(w: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (w'Σw, Σw); raise ValueError unless w'Σw is clearly above rounding noise."""
    sigma_w = s @ w
    variance = float(w @ sigma_w)
    rounding = _ZERO_VARIANCE_FACTOR * len(w) * _EPS * float(np.abs(w) @ np.abs(s) @ np.abs(w))
    if not variance > rounding:
        raise ValueError(
            f"portfolio variance {variance:.3g} is not strictly positive; "
            "risk contributions are undefined"
        )
    return variance, sigma_w


def _volatilities(sigma: pd.DataFrame) -> np.ndarray:
    """Asset volatilities sqrt(Σ_ii); every variance must be strictly positive."""
    variances = np.diag(sigma.to_numpy())
    bad = [a for a, v in zip(sigma.index, variances, strict=True) if not v > 0.0]
    if bad:
        raise ValueError(f"every asset needs a strictly positive variance; offending: {bad}")
    return np.sqrt(variances)


def _correlation(s: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """Correlation matrix D⁻¹ΣD⁻¹ (vols > 0), validated to be positive semi-definite.

    The test is scale-free: Σ is PSD iff its correlation matrix is.
    """
    # Divide in two steps so that D⁻¹ΣD⁻¹ cannot underflow the denominator
    # for tiny variances; an overflow can only come from |ρ| >> 1.
    with np.errstate(over="ignore"):
        corr = s / vols[:, None] / vols[None, :]
    if not np.isfinite(corr).all():
        raise ValueError("covariance matrix is not positive semi-definite (|correlation| > 1)")
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    n = corr.shape[0]
    min_eig = float(linalg.eigvalsh(corr)[0])
    if min_eig < -_PSD_TOL * n:
        raise ValueError(
            "covariance matrix is not positive semi-definite "
            f"(smallest eigenvalue of the correlation matrix is {min_eig:.3g})"
        )
    return corr


def _check_covariance(sigma: pd.DataFrame) -> None:
    """Raise ValueError unless Σ is a valid covariance: variances >= 0 and Σ PSD.

    Zero variances (cash, or an asset held constant) are allowed. PSD-ness
    is tested without reference to the units: an asset with zero variance must
    have zero covariances (otherwise the 2x2 principal minor
    Σ_ii Σ_jj - Σ_ij² is negative), and the block of the other assets is PSD
    iff its correlation matrix is.
    """
    s = sigma.to_numpy()
    variances = np.diag(s)
    negative = [a for a, v in zip(sigma.index, variances, strict=True) if v < 0.0]
    if negative:
        raise ValueError(f"variances must be non-negative; offending assets: {negative}")
    zero = variances == 0.0
    if (s[zero] != 0.0).any():
        raise ValueError(
            "covariance matrix is not positive semi-definite "
            "(an asset with zero variance has non-zero covariances)"
        )
    positive = ~zero
    if positive.any():
        _correlation(s[np.ix_(positive, positive)], np.sqrt(variances[positive]))


def _budgets_for(
    risk_budgets: pd.Series | np.ndarray | Sequence[float] | None, assets: pd.Index
) -> np.ndarray:
    """Positive risk budgets aligned to ``assets`` and normalized to sum to 1."""
    if risk_budgets is None:
        return np.full(len(assets), 1.0 / len(assets))
    if isinstance(risk_budgets, pd.Series):
        known = set(assets)
        extra = [a for a in risk_budgets.index if a not in known]
        if extra:
            raise ValueError(f"risk_budgets contain assets not in the covariance matrix: {extra}")
    b = as_series(risk_budgets, assets).to_numpy()
    if not (b > 0.0).all():
        raise ValueError(
            "risk budgets must be strictly positive (drop an asset from the universe "
            "instead of giving it a zero budget)"
        )
    return b / b.sum()


def _check_solver_settings(tol: float, max_iter: int) -> tuple[float, int]:
    try:
        tol_ok = bool(np.isfinite(float(tol)) and float(tol) > 0.0)
    except (TypeError, ValueError):
        tol_ok = False
    if not tol_ok:
        raise ValueError(f"tol must be a positive finite number, got {tol!r}")
    try:
        iter_ok = not isinstance(max_iter, bool) and int(max_iter) == max_iter and max_iter >= 1
    except (TypeError, ValueError, OverflowError):
        iter_ok = False
    if not iter_ok:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
    return float(tol), int(max_iter)


def _armijo_holds(
    b: np.ndarray,
    x: np.ndarray,
    step: np.ndarray,
    alpha: float,
    curvature: float,
    decrement_sq: float,
) -> bool:
    """Armijo test f(x - αΔ) <= f(x) - c α g'Δ for Spinu's objective, free of cancellation.

    With f(x) = ½ x'Cx - Σ_i b_i ln x_i, g = ∇f = Cx - b/x, d = -αΔ and
    u = d/x (elementwise; u > -1 because x + d > 0), expanding exactly gives::

        f(x + d) - f(x) = d'g + ½ d'Cd + Σ_i b_i ψ(u_i),   ψ(u) = u - ln(1 + u) >= 0,

    and d'g = -α g'Δ, so the Armijo condition is equivalent to::

        ½ α² Δ'CΔ + Σ_i b_i ψ(-α Δ_i / x_i) <= (1 - c) α g'Δ.

    Why not subtract two objective values: f is O(1) while the terms that
    involve an asset with budget b_i are O(b_i), so f(x - αΔ) - f(x) is lost
    in rounding once those terms fall below eps·|f| -- with a budget near
    1e-14 the search then halves α to nothing and the iteration stalls before
    the budgets are met. Here both sides are of the size of the decrease
    itself and every left-hand term is non-negative, so the test stays
    meaningful until the step is at machine precision.

    Args:
        b: risk budgets (sum to 1).
        x: current iterate (> 0).
        step: descent direction Δ (the Newton step H⁻¹g in the solver).
        alpha: trial step length, with x - αΔ > 0.
        curvature: Δ'CΔ.
        decrement_sq: g'Δ (> 0; the squared Newton decrement for the
            Newton step).

    Returns:
        True if the trial step decreases f sufficiently.
    """
    # x - αΔ > 0 in floating point means αΔ_i < x_i exactly (α is a power of
    # two), so fl(αΔ_i / x_i) < 1, u > -1 and log1p(u) is finite.
    u = -alpha * step / x
    excess = 0.5 * alpha * alpha * curvature + float(b @ (u - np.log1p(u)))
    return excess <= (1.0 - _ARMIJO_C) * alpha * decrement_sq


def _line_search(
    corr: np.ndarray, b: np.ndarray, x: np.ndarray, step: np.ndarray, decrement_sq: float
) -> float:
    """Step length α for the update x ← x - αΔ (Boyd & Vandenberghe 2004, Alg. 9.2).

    Starting from α = 1, halve α until x - αΔ > 0 (f is +inf outside the
    positive orthant, so feasibility comes first), then keep halving until the
    Armijo condition f(x - αΔ) <= f(x) - c α g'Δ holds (see
    ``_armijo_holds``). ``step`` is any descent direction with
    ``decrement_sq = g'Δ > 0``; the solver passes the Newton step.

    Raises:
        OptimizationError: if no acceptable α is found in ``_MAX_HALVINGS``
            halvings (Δ is a descent direction, g'Δ > 0, so in exact
            arithmetic a small enough α always works).
    """
    alpha = 1.0
    halvings = 0
    while not (x - alpha * step > 0.0).all():
        alpha *= 0.5
        halvings += 1
        if halvings > _MAX_HALVINGS:
            raise OptimizationError("risk parity: no positive Newton step found")
    curvature = float(step @ corr @ step)
    while not _armijo_holds(b, x, step, alpha, curvature, decrement_sq):
        alpha *= 0.5
        halvings += 1
        if halvings > _MAX_HALVINGS:
            raise OptimizationError("risk parity: line search failed to decrease the objective")
    return alpha


def _spinu_newton(corr: np.ndarray, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    """Minimize f(x) = ½ x'Cx - Σ_i b_i ln x_i over x > 0 by damped Newton.

    Works on the correlation matrix C (unit diagonal), which makes the problem
    dimensionless and well scaled whatever the units of the covariance.

    Globalization: Newton's method with a backtracking line search (Boyd &
    Vandenberghe 2004, Alg. 9.5). Let Δ = H⁻¹g be the Newton step. The step
    is first halved until x - αΔ > 0 (f is +inf outside the positive orthant),
    then halved until the Armijo condition holds (evaluated without
    cancellation, see ``_armijo_holds``). With b_min = min_i b_i, f/b_min is
    standard self-concordant (a convex quadratic plus terms -a ln x with
    a >= 1), and the Armijo test is invariant to rescaling f, so the classical
    analysis applies: whenever a solution exists the iteration converges from
    any start; the full step is accepted once the Newton decrement
    λ = sqrt(g'Δ / b_min) is below (1 - 2c)/2, and convergence is quadratic
    for λ <= (1 - 2c)/4 (Boyd & Vandenberghe 2004, Section 9.6.4; Nesterov
    2004, Section 4.1).

    Stopping rule on the unit-free risk-budget residual r_i = x_i (Cx)_i - b_i:
    max_i |r_i| <= min(max(tol, rounding floor), cap). The cap guarantees the
    final budget check: the variance shares are (b_i + r_i) / (1 + Σ_j r_j), so
    |share_i - b_i| <= (|r_i| + b_i Σ_j |r_j|) / (1 - N max|r|)
    <= 2 (N + 1) max|r| whenever N max|r| <= 1/2; with
    cap = _BUDGET_CHECK_TOL / (4 (N + 1)) that is at most half the check's
    tolerance. It also stops a rounding floor that grows with diverging
    iterates (no solution exists) from ever being mistaken for convergence.

    Raises:
        OptimizationError: if not converged within ``max_iter`` Newton steps,
            the Hessian factorization fails, or a floating-point error occurs
            (e.g. the iterates diverge because some long-only combination of
            the assets has zero variance, in which case no solution exists).
    """
    n = b.shape[0]
    abs_corr = np.abs(corr)
    residual_cap = _BUDGET_CHECK_TOL / (4.0 * (n + 1))
    ray_variance = float(b @ corr @ b)
    if not ray_variance > _ZERO_VARIANCE_FACTOR * n * _EPS * float(b @ abs_corr @ b):
        # f(c b) = ½c² b'Cb - ln c - const -> -inf as c -> inf: unbounded below.
        raise OptimizationError(
            "risk parity has no solution: the budget-weighted portfolio has zero variance "
            "(the covariance admits a riskless long-only combination)"
        )
    # Start on the ray x ∝ b at its optimal scale: d/dc f(c b) = 0 gives
    # c = 1/sqrt(b'Cb). This is the spec's y0 ∝ b/σ in covariance units, and
    # it is the exact solution when all correlations are equal and b = 1/N.
    x = b / np.sqrt(ray_variance)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # max_iter Newton steps; the extra pass only tests the final iterate.
            for iteration in range(max_iter + 1):
                cx = corr @ x
                residual = float(np.max(np.abs(x * cx - b)))
                floor = _RESIDUAL_ROUNDING_FACTOR * n * _EPS * float(np.max(x * (abs_corr @ x)))
                if residual <= min(max(tol, floor), residual_cap):
                    return x
                if iteration == max_iter:
                    break
                grad = cx - b / x
                hess = corr + np.diag(b / x**2)
                step = linalg.cho_solve(linalg.cho_factor(hess, lower=True), grad)
                decrement_sq = float(grad @ step)
                if not decrement_sq > 0.0:
                    # H is positive definite, so g'H⁻¹g <= 0 means g is zero up
                    # to rounding while the residual test failed: we cannot
                    # make progress, and must not report convergence.
                    break
                x = x - _line_search(corr, b, x, step, decrement_sq) * step
    except (FloatingPointError, linalg.LinAlgError) as exc:
        raise OptimizationError(
            f"risk parity Newton iteration broke down ({exc}); the covariance may be "
            "singular in a way that admits no long-only risk-budgeting portfolio"
        ) from exc
    raise OptimizationError(
        f"risk parity Newton iteration did not converge in {max_iter} iterations "
        f"(max risk-budget residual {residual:.3g}, tol {tol:.3g})"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def risk_contributions(
    weights: pd.Series | np.ndarray | Sequence[float],
    cov: pd.DataFrame | np.ndarray,
    relative: bool = False,
) -> pd.Series:
    """Euler risk contributions of each asset to portfolio volatility.

    Volatility ``σ_p = sqrt(w'Σw)`` is homogeneous of degree one in ``w``, so
    by Euler's theorem ``σ_p = Σ_i w_i ∂σ_p/∂w_i`` with marginal risk
    ``∂σ_p/∂w_i = (Σw)_i / σ_p``. Hence::

        RC_i = w_i (Σw)_i / σ_p,             Σ_i RC_i = σ_p
        RC_i / σ_p = w_i (Σw)_i / (w'Σw),    Σ_i RC_i / σ_p = 1   (relative)

    The relative form is asset ``i``'s share of portfolio variance (Litterman
    1996; Qian 2006). Contributions are in the units of ``cov``: with an
    annualized covariance they sum to the annualized volatility. Weights need
    not sum to one and may be negative (a hedge has a negative contribution).

    Args:
        weights: portfolio weights; a Series is aligned to the covariance
            labels and must cover exactly those assets.
        cov: covariance matrix (N x N), positive semi-definite. Assets with
            zero variance (e.g. cash) are allowed.
        relative: if True return shares of risk ``RC_i / σ_p`` (sum to 1).

    Returns:
        Series of risk contributions indexed by asset.

    Raises:
        ValueError: on malformed inputs, a matrix that is not a covariance
            (a negative variance, or not positive semi-definite), weights for
            unknown assets, or a portfolio variance that is not strictly
            positive (risk contributions are then undefined).
    """
    sigma = as_cov_frame(cov)
    # Without this an invalid matrix still yields numbers whenever w'Σw > 0,
    # e.g. a negative "variance share" for a long position.
    _check_covariance(sigma)
    w = _weights_for(weights, sigma).to_numpy()
    variance, sigma_w = _positive_variance(w, sigma.to_numpy())
    contributions = w * sigma_w
    contributions = contributions / variance if relative else contributions / np.sqrt(variance)
    return pd.Series(contributions, index=sigma.index.copy())


def inverse_volatility(cov: pd.DataFrame | np.ndarray) -> pd.Series:
    """Inverse-volatility ("naive risk parity") weights ``w_i ∝ 1/σ_i``.

    ``w_i = σ_i⁻¹ / Σ_j σ_j⁻¹`` with ``σ_i = sqrt(Σ_ii)``. Correlations are
    ignored. Equivalently, every asset gets the same stand-alone risk
    ``w_i σ_i``; when all pairwise correlations are equal this is exactly the
    equal-risk-contribution portfolio (Maillard, Roncalli & Teiletche 2010).

    Args:
        cov: covariance matrix (N x N); only the diagonal is used.

    Returns:
        Series of weights indexed by asset, positive and summing to 1.

    Raises:
        ValueError: on a malformed matrix or a non-positive variance.
    """
    sigma = as_cov_frame(cov)
    vols = _volatilities(sigma)
    # Scaled by the smallest volatility so 1/σ cannot overflow for tiny variances.
    inv = vols.min() / vols
    return pd.Series(inv / inv.sum(), index=sigma.index.copy())


def inverse_variance(cov: pd.DataFrame | np.ndarray) -> pd.Series:
    """Inverse-variance weights ``w_i ∝ 1/σ_i²``.

    ``w_i = σ_i⁻² / Σ_j σ_j⁻²``. This is the minimum-variance portfolio
    ``Σ⁻¹1 / 1'Σ⁻¹1`` when ``Σ`` is diagonal (zero correlations), and the
    allocation HRP uses within a cluster (López de Prado 2016).

    Args:
        cov: covariance matrix (N x N); only the diagonal is used.

    Returns:
        Series of weights indexed by asset, positive and summing to 1.

    Raises:
        ValueError: on a malformed matrix or a non-positive variance.
    """
    sigma = as_cov_frame(cov)
    vols = _volatilities(sigma)
    # Normalize by the smallest variance first so that tiny (e.g. daily)
    # variances cannot overflow 1/σ² before the weights are normalized.
    inv = (vols.min() / vols) ** 2
    return pd.Series(inv / inv.sum(), index=sigma.index.copy())


def equal_risk_contribution(
    cov: pd.DataFrame | np.ndarray,
    risk_budgets: pd.Series | np.ndarray | Sequence[float] | None = None,
    tol: float = 1e-12,
    max_iter: int = 500,
) -> pd.Series:
    """Long-only risk-budgeting portfolio; equal risk contribution by default.

    Finds the unique ``w > 0``, ``1'w = 1``, whose shares of portfolio
    variance equal the budgets ``b`` (Maillard, Roncalli & Teiletche 2010;
    Bruder & Roncalli 2012)::

        w_i (Σw)_i / (w'Σw) = b_i   for every i.

    Algorithm (Spinu 2013). Solve the strictly convex problem::

        min_{y > 0}  f(y) = ½ y'Σy - Σ_i b_i ln y_i

    whose stationarity condition ``∇f = Σy - b/y = 0`` reads
    ``y_i (Σy)_i = b_i``. Setting ``w = y / 1'y`` gives
    ``w_i (Σw)_i = b_i / (1'y)²`` and, summing over ``i`` (``Σb = 1``),
    ``w'Σw = 1/(1'y)²``, so the variance shares are exactly ``b``. The
    Hessian ``Σ + diag(b/y²)`` is positive definite, so the minimizer is unique,
    and ``f → +∞`` at the boundary of the orthant (log barrier) and, for
    positive-definite ``Σ``, at infinity, so it exists. A singular ``Σ`` still
    has a solution unless some nonzero long-only combination ``v >= 0`` has
    ``v'Σv = 0``; then ``f(y + t v) → -∞`` and no risk-budgeting portfolio
    exists (the iteration fails with ``OptimizationError``).

    The problem is solved in correlation units: with ``D = diag(σ)``,
    ``C = D⁻¹ΣD⁻¹`` and ``x = D y``, ``f(y) = ½ x'Cx - Σ b_i ln x_i + const``.
    The iteration is then identical for daily and annualized covariances (and
    for any rescaling of an asset's units) and the Hessian is well scaled. It
    starts from ``x ∝ b`` (i.e. ``y ∝ b/σ``) and takes damped Newton steps
    ``x ← x - α H⁻¹∇f`` with gradient ``Cx - b/x`` and Hessian
    ``C + diag(b/x²)``; ``α`` is chosen by backtracking so that ``x`` stays
    strictly positive and ``f`` decreases sufficiently (Armijo; the decrease
    is evaluated in a cancellation-free form, so the search keeps working when
    some budgets are tiny, e.g. 1e-14, and their effect on ``f`` is below
    rounding). Near the solution full steps are accepted and
    convergence is quadratic (``f`` is self-concordant up to scaling).
    Typical problems converge in 5-15 iterations, including N = 50 assets
    with correlations of 0.99.

    Args:
        cov: covariance matrix (N x N), positive semi-definite with strictly
            positive variances. Any positive multiple gives the same weights.
        risk_budgets: positive budgets, one per asset (a Series is aligned by
            label); normalized to sum to 1. Default: equal budgets ``1/N``.
        tol: convergence tolerance on the unit-free risk-budget residual
            ``max_i |y_i (Σy)_i - b_i|``. It is relaxed to the floating-point
            accuracy of that residual when tighter than double precision can
            resolve, and never looser than ``1e-8 / (4(N+1))``, which
            guarantees the final 1e-8 budget check. The tolerance is
            absolute, so a budget far below it is met only to that absolute
            accuracy, not relative to its own size.
        max_iter: maximum number of Newton iterations.

    Returns:
        Series of positive weights indexed by asset, summing to 1.

    Raises:
        ValueError: on malformed inputs, a non-positive variance, a matrix
            that is not positive semi-definite, non-positive budgets, or
            invalid ``tol``/``max_iter``.
        OptimizationError: if the Newton iteration does not converge (this
            includes covariances for which no solution exists, i.e. some
            long-only combination of the assets has zero variance) or the
            result fails the final check ``max_i |RC_i/σ_p - b_i| < 1e-8``.
    """
    sigma = as_cov_frame(cov)
    tol, max_iter = _check_solver_settings(tol, max_iter)
    b = _budgets_for(risk_budgets, sigma.index)
    vols = _volatilities(sigma)
    corr = _correlation(sigma.to_numpy(), vols)

    x = _spinu_newton(corr, b, tol, max_iter)
    # Back to covariance units: y = D⁻¹x, then normalize to the budget constraint.
    y = x / vols
    w = y / y.sum()

    try:
        variance, sigma_w = _positive_variance(w, sigma.to_numpy())
    except ValueError as exc:
        raise OptimizationError(f"risk parity solution is degenerate: {exc}") from exc
    shares = w * sigma_w / variance
    error = float(np.max(np.abs(shares - b)))
    if not error < _BUDGET_CHECK_TOL:
        raise OptimizationError(
            f"risk parity solution misses its risk budgets by {error:.3g} "
            f"(tolerance {_BUDGET_CHECK_TOL:g})"
        )
    return pd.Series(w, index=sigma.index.copy())
