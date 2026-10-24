"""Estimators of expected returns and covariance matrices.

Every estimator takes a table of simple per-period returns (rows = dates in
ascending order, columns = assets; numpy input is labelled 0..N-1) and returns
ANNUALIZED, asset-labelled pandas objects, scaling per-period moments by
``periods_per_year`` (default ``TRADING_DAYS`` = 252). The one exception is
:func:`ledoit_wolf_shrinkage`, which works on the raw per-period scale and
returns numpy arrays exactly as in the papers; :func:`cov_ledoit_wolf` is its
annualized, labelled wrapper.

Why shrinkage matters for portfolio optimization: the sample covariance of N
assets has N(N+1)/2 free parameters estimated from T observations. When N/T is
not small its extreme eigenvalues are badly biased (the largest too large, the
smallest too small), and a mean-variance optimizer, which effectively inverts
the matrix, loads up on exactly those spuriously low-risk directions. Shrinking
toward a structured target trades a little bias for a large variance reduction.

References:
    O. Ledoit and M. Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices", Journal of Multivariate Analysis
    88(2), 365-411.
    O. Ledoit and M. Wolf (2004), "Honey, I shrunk the sample covariance
    matrix", Journal of Portfolio Management 30(4), 110-119 (working paper
    2003; reference MATLAB code ``covCor.m``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from quantfin.portfolio._common import TRADING_DAYS, as_cov_frame, as_returns_frame

ShrinkageTarget = Literal["constant_correlation", "identity"]

_TARGETS = ("constant_correlation", "identity")

# ||S - F||_F^2 below this fraction of ||S||_F^2 means the sample covariance
# already equals the target up to floating-point rounding (e.g. N = 2 with the
# constant-correlation target, where F == S identically). The shrinkage
# intensity is then undefined (0/0) and irrelevant, so it is reported as 0.
_DEGENERATE_GAMMA_RTOL = 1e-20

# Cholesky pivots L_ii^2 / A_ii at or below this many multiples of N * eps are
# rounding noise: a rank-deficient matrix can "pass" LAPACK's Cholesky by luck.
_PD_PIVOT_FACTOR = 1e3


def _check_periods_per_year(periods_per_year: float) -> float:
    ppy = float(periods_per_year)
    if not np.isfinite(ppy) or ppy <= 0:
        raise ValueError(f"periods_per_year must be a positive number, got {periods_per_year!r}")
    return ppy


def _returns_for_cov(returns: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    frame = as_returns_frame(returns)
    if frame.shape[0] < 2:
        raise ValueError(
            f"at least 2 observations are needed to estimate a covariance, got {frame.shape[0]}"
        )
    return frame


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _labelled_cov(values: np.ndarray, assets: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(_symmetrize(values), index=assets.copy(), columns=assets.copy())


# ---------------------------------------------------------------------------
# Expected returns
# ---------------------------------------------------------------------------


def mean_historical(
    returns: pd.DataFrame | np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    compounding: bool = False,
) -> pd.Series:
    """Annualized historical mean return of each asset.

    Arithmetic (default):  mu_i = ppy * (1/T) sum_t r_ti.

    Compounding (geometric / CAGR):  mu_i = prod_t (1 + r_ti) ** (ppy / T) - 1,
    computed as ``expm1(ppy/T * sum_t log1p(r_ti))`` for accuracy. An asset with
    a -100% return in the sample has been wiped out and gets exactly -1.

    The arithmetic mean is the right input for a one-period mean-variance
    optimizer (it is the expectation of the next period's return); the
    geometric mean is always <= the arithmetic one (AM-GM) and describes
    realized growth.

    Args:
        returns: T x N simple returns.
        periods_per_year: periods per year used to annualize (ppy).
        compounding: use the geometric (compounded) mean instead.

    Returns:
        Series of annualized expected returns indexed by asset.

    Raises:
        ValueError: on invalid returns, non-positive ``periods_per_year``, or a
            return below -100% when ``compounding`` is True.
    """
    frame = as_returns_frame(returns)
    ppy = _check_periods_per_year(periods_per_year)
    values = frame.to_numpy(dtype=float)
    n_obs = values.shape[0]
    if not compounding:
        return pd.Series(values.mean(axis=0) * ppy, index=frame.columns.copy())
    if (values < -1.0).any():
        raise ValueError("simple returns below -100% are impossible; cannot compound them")
    wiped_out = (values == -1.0).any(axis=0)
    # log1p(-1) = -inf would raise a RuntimeWarning; those assets are set to -1 below.
    safe = np.where(values > -1.0, values, 0.0)
    log_growth = np.log1p(safe).sum(axis=0) * (ppy / n_obs)
    # Annualizing a handful of huge returns can exceed float range; inf is the honest answer.
    with np.errstate(over="ignore"):
        annual = np.expm1(log_growth)
    annual[wiped_out] = -1.0
    return pd.Series(annual, index=frame.columns.copy())


def mean_ewma(
    returns: pd.DataFrame | np.ndarray,
    span: int = 252,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Annualized exponentially weighted mean return (most recent observations weigh most).

    With alpha = 2 / (span + 1) and the observation of age k = T-1-t getting
    weight (1 - alpha)^k, the estimate is

        mu_i = ppy * sum_t (1-alpha)^(T-1-t) r_ti / sum_t (1-alpha)^(T-1-t),

    i.e. the last value of ``pandas.DataFrame.ewm(span=span, adjust=True).mean()``.
    ``span = 1`` puts all weight on the last observation.

    Args:
        returns: T x N simple returns.
        span: decay span in periods (>= 1); the centre of mass is (span - 1) / 2.
        periods_per_year: periods per year used to annualize.

    Returns:
        Series of annualized expected returns indexed by asset.

    Raises:
        ValueError: on invalid returns, ``span < 1`` or non-positive ``periods_per_year``.
    """
    frame = as_returns_frame(returns)
    ppy = _check_periods_per_year(periods_per_year)
    span_f = float(span)
    if not np.isfinite(span_f) or span_f < 1.0:
        raise ValueError(f"span must be >= 1, got {span!r}")
    last = frame.ewm(span=span_f, adjust=True).mean().iloc[-1]
    return pd.Series(last.to_numpy(dtype=float) * ppy, index=frame.columns.copy())


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def cov_sample(
    returns: pd.DataFrame | np.ndarray, periods_per_year: int = TRADING_DAYS
) -> pd.DataFrame:
    """Annualized unbiased sample covariance.

    Sigma = ppy * X'X / (T - 1), where X holds the returns demeaned column by
    column (ddof = 1).

    Args:
        returns: T x N simple returns (T >= 2).
        periods_per_year: periods per year used to annualize.

    Returns:
        N x N symmetric covariance DataFrame labelled by asset.

    Raises:
        ValueError: on invalid returns, T < 2 or non-positive ``periods_per_year``.
    """
    frame = _returns_for_cov(returns)
    ppy = _check_periods_per_year(periods_per_year)
    values = frame.to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    cov = centered.T @ centered / (values.shape[0] - 1)
    return _labelled_cov(cov * ppy, frame.columns)


def cov_ewma(
    returns: pd.DataFrame | np.ndarray,
    halflife: float = 63,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Annualized exponentially weighted covariance (RiskMetrics-style, with a mean).

    Observation t (0-based, oldest first) gets weight
    w_t = v_t / sum_s v_s with v_t = 0.5 ** ((T - 1 - t) / halflife), so an
    observation ``halflife`` periods old counts half as much as the latest one.
    With the weighted mean m = sum_t w_t r_t,

        Sigma = ppy * sum_t w_t (r_t - m)(r_t - m)' / (1 - sum_t w_t^2).

    The divisor 1 - sum w^2 is the reliability-weights bias correction (the
    weighted analogue of ddof = 1, with 1 / sum w^2 the effective sample
    size). It makes the estimator unbiased for i.i.d. data, makes it equal to
    :func:`cov_sample` in the limit halflife -> infinity, and equals the last
    block of ``pandas.DataFrame.ewm(halflife=halflife).cov()`` (bias=False).
    The result is symmetric positive semi-definite (a positive combination of
    outer products).

    Args:
        returns: T x N simple returns (T >= 2).
        halflife: decay half-life in periods (> 0).
        periods_per_year: periods per year used to annualize.

    Returns:
        N x N symmetric PSD covariance DataFrame labelled by asset.

    Raises:
        ValueError: on invalid returns, T < 2, non-positive ``halflife`` or
            ``periods_per_year``, or a half-life so short that all weight sits on
            a single observation (the covariance is then not identifiable).
    """
    frame = _returns_for_cov(returns)
    ppy = _check_periods_per_year(periods_per_year)
    halflife_f = float(halflife)
    if not np.isfinite(halflife_f) or halflife_f <= 0:
        raise ValueError(f"halflife must be a positive number, got {halflife!r}")
    values = frame.to_numpy(dtype=float)
    n_obs = values.shape[0]
    age = np.arange(n_obs - 1, -1, -1, dtype=float)
    raw = np.exp2(-age / halflife_f)  # newest weight is exactly 1; old ones may underflow to 0
    # Write sum(v) = 1 + a and sum(v^2) = 1 + b (the newest observation has v = 1), so
    # 1 - sum(w^2) = (2a + a^2 - b) / (1 + a)^2 without catastrophic cancellation.
    a = float(raw[:-1].sum())
    b = float((raw[:-1] ** 2).sum())
    if a <= 1e-12:
        raise ValueError(
            f"halflife={halflife_f} is too short for this sample: essentially all weight "
            "is on the last observation, so the covariance cannot be estimated"
        )
    weights = raw / (1.0 + a)
    one_minus_sum_w2 = (2.0 * a + a * a - b) / (1.0 + a) ** 2
    mean = weights @ values
    centered = values - mean
    cov = (centered * weights[:, None]).T @ centered / one_minus_sum_w2
    return _labelled_cov(cov * ppy, frame.columns)


def _lw_identity_target(x: np.ndarray, sample: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004, JMVA) target m*I and its optimal intensity (see caller)."""
    n_obs, n_assets = x.shape
    m = float(np.trace(sample)) / n_assets
    prior = m * np.eye(n_assets)
    d2 = float(np.sum((sample - prior) ** 2))
    if d2 <= _DEGENERATE_GAMMA_RTOL * float(np.sum(sample**2)):
        return prior, 0.0
    # sum_t ||x_t x_t' - S||_F^2 = sum_t (x_t'x_t)^2 - T ||S||_F^2 (the cross term is
    # -2 sum_t <x_t x_t', S> = -2 T ||S||_F^2), which avoids forming T outer products.
    row_norms_sq = np.sum(x * x, axis=1)
    b2_bar = (float(np.sum(row_norms_sq**2)) / n_obs - float(np.sum(sample**2))) / n_obs
    b2 = min(max(b2_bar, 0.0), d2)
    return prior, b2 / d2


def _lw_constant_correlation_target(x: np.ndarray, sample: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2003/2004, "Honey") target and intensity, transcribing covCor.m."""
    n_obs, n_assets = x.shape
    var = np.diag(sample).copy()
    if n_assets == 1:
        # The target equals the sample variance; there is nothing to shrink.
        return sample.copy(), 0.0
    sqrtvar = np.sqrt(var)
    sd_outer = np.outer(sqrtvar, sqrtvar)
    corr = sample / sd_outer
    off_diagonal = ~np.eye(n_assets, dtype=bool)
    # covCor.m computes (sum(corr) - n); summing only off-diagonal entries is the same
    # quantity without relying on the diagonal of corr being exactly 1 in floating point.
    rbar = float(corr[off_diagonal].sum()) / (n_assets * (n_assets - 1))
    prior = rbar * sd_outer
    np.fill_diagonal(prior, var)

    gamma = float(np.sum((sample - prior) ** 2))
    if gamma <= _DEGENERATE_GAMMA_RTOL * float(np.sum(sample**2)):
        return prior, 0.0

    # pi-hat: sum of asymptotic variances of sqrt(T) s_ij.
    y = x**2
    xtx = x.T @ x
    phi_mat = y.T @ y / n_obs - 2.0 * xtx * sample / n_obs + sample**2
    phi = float(phi_mat.sum())

    # rho-hat: sum of asymptotic covariances between sqrt(T) f_ij and sqrt(T) s_ij.
    term1 = (x**3).T @ x / n_obs
    help_ = xtx / n_obs
    help_diag = np.diag(help_)
    term2 = help_diag[:, None] * sample  # repmat(helpDiag, 1, n) .* sample
    term3 = help_ * var[:, None]  # help .* repmat(var, 1, n)
    term4 = var[:, None] * sample  # repmat(var, 1, n) .* sample
    theta_mat = term1 - term2 - term3 + term4
    np.fill_diagonal(theta_mat, 0.0)
    ratio = np.outer(1.0 / sqrtvar, sqrtvar)  # (i, j) entry: sd_j / sd_i
    rho = float(np.trace(phi_mat)) + rbar * float(np.sum(ratio * theta_mat))

    kappa = (phi - rho) / gamma
    delta = max(0.0, min(1.0, kappa / n_obs))
    return prior, delta


def ledoit_wolf_shrinkage(
    returns: pd.DataFrame | np.ndarray,
    target: ShrinkageTarget = "constant_correlation",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Ledoit-Wolf shrinkage of the sample covariance toward a structured target.

    Works on the raw (per-period, NOT annualized) scale. With X the T x N matrix
    of column-demeaned returns and S = X'X / T the maximum-likelihood sample
    covariance (divisor T, as in the papers), the estimator is

        Sigma_hat = delta * F + (1 - delta) * S,

    with the intensity delta in [0, 1] chosen to minimize the (estimated)
    expected squared Frobenius loss E||Sigma_hat - Sigma||^2.

    ``target="identity"`` -- Ledoit & Wolf (2004), "A well-conditioned estimator
    for large-dimensional covariance matrices", JMVA 88: F = m I with
    m = tr(S) / N, and

        d^2 = ||S - m I||_F^2,
        b_bar^2 = (1/T^2) sum_t ||x_t x_t' - S||_F^2,
        b^2 = min(b_bar^2, d^2),           delta = b^2 / d^2.

    (The paper's 1/N norm normalization cancels in the ratio.) This matches
    ``sklearn.covariance.ledoit_wolf``.

    ``target="constant_correlation"`` -- Ledoit & Wolf (2004), "Honey, I shrunk
    the sample covariance matrix", JPM 30(4), a line-by-line transcription of
    their ``covCor.m``. With s_ii = diag(S), sd = sqrt(s_ii) and the average
    sample correlation rbar = sum_{i != j} s_ij / (sd_i sd_j) / (N(N-1)):

        F_ij = rbar sd_i sd_j (i != j),   F_ii = s_ii;
        pi_ij = (1/T) sum_t (x_ti x_tj - s_ij)^2,        pi = sum_ij pi_ij;
        theta_ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - s_ij)
                 = (1/T) sum_t x_ti^3 x_tj - s_ii s_ij;
        rho = sum_i pi_ii + rbar sum_{i != j} (sd_j / sd_i) theta_ij;
        gamma = ||F - S||_F^2;   kappa = (pi - rho) / gamma;
        delta = max(0, min(1, kappa / T)).

    Here pi estimates the asymptotic variance of the sample covariance, rho
    the part of it the target shares (its covariance with the target), and
    gamma the squared misspecification of the target, so the optimal
    intensity shrinks more when S is noisy and less when F is biased; it
    decays like 1/T as the sample grows.

    If S already equals F (||S - F||_F^2 <= 1e-20 ||S||_F^2, e.g. any 2-asset
    problem with the constant-correlation target), delta is 0.

    Args:
        returns: T x N simple returns (T >= 2).
        target: ``"constant_correlation"`` (default) or ``"identity"``.

    Returns:
        ``(shrunk_cov, target_matrix, delta)``: the shrunk covariance and the
        target F as N x N numpy arrays on the per-period scale, and delta.

    Raises:
        ValueError: on invalid returns, T < 2, an unknown target, or (for the
            constant-correlation target) an asset with constant returns, whose
            correlations are undefined.
    """
    if target not in _TARGETS:
        raise ValueError(f"target must be one of {_TARGETS}, got {target!r}")
    frame = _returns_for_cov(returns)
    values = frame.to_numpy(dtype=float)
    n_obs = values.shape[0]
    x = values - values.mean(axis=0)
    sample = _symmetrize(x.T @ x / n_obs)
    if target == "identity":
        prior, delta = _lw_identity_target(x, sample)
    else:
        # Checked on the raw data: demeaning a constant column can leave rounding dust
        # that would masquerade as a tiny but non-zero variance.
        constant = np.ptp(values, axis=0) == 0.0
        if constant.any():
            names = list(frame.columns[constant])
            raise ValueError(
                f"assets {names} have constant returns; their correlations are undefined, "
                "so the constant-correlation target cannot be built"
            )
        prior, delta = _lw_constant_correlation_target(x, sample)
    prior = _symmetrize(prior)
    shrunk = _symmetrize(delta * prior + (1.0 - delta) * sample)
    return shrunk, prior, float(delta)


def cov_ledoit_wolf(
    returns: pd.DataFrame | np.ndarray,
    target: ShrinkageTarget = "constant_correlation",
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Annualized Ledoit-Wolf shrunk covariance, labelled by asset.

    Sigma = ppy * (delta F + (1 - delta) S); see :func:`ledoit_wolf_shrinkage`
    for the estimator, the targets and the references. Because S uses the
    divisor T, the diagonal of the constant-correlation estimate is the
    ddof = 0 sample variance.

    Args:
        returns: T x N simple returns (T >= 2).
        target: ``"constant_correlation"`` (default) or ``"identity"``.
        periods_per_year: periods per year used to annualize.

    Returns:
        N x N symmetric covariance DataFrame labelled by asset; positive
        definite whenever delta > 0 and the target is positive definite, even
        when T <= N and the sample covariance is singular.

    Raises:
        ValueError: as :func:`ledoit_wolf_shrinkage`, or on non-positive
            ``periods_per_year``.
    """
    ppy = _check_periods_per_year(periods_per_year)
    frame = _returns_for_cov(returns)
    shrunk, _, _ = ledoit_wolf_shrinkage(frame, target=target)
    return _labelled_cov(shrunk * ppy, frame.columns)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def cov_to_corr(cov: pd.DataFrame | np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    """Split a covariance matrix into correlations and volatilities.

    sigma_i = sqrt(Sigma_ii) and rho_ij = Sigma_ij / (sigma_i sigma_j), so that
    Sigma = diag(sigma) rho diag(sigma). The diagonal of rho is set to exactly 1
    and entries within 1e-8 of +-1 are clipped into [-1, 1] (rounding).

    Args:
        cov: N x N symmetric covariance (numpy or labelled DataFrame).

    Returns:
        ``(corr, vols)``: the correlation DataFrame and the volatility Series,
        both labelled like ``cov``. Volatilities are on the scale of ``cov``
        (annualized in, annualized out).

    Raises:
        ValueError: if ``cov`` is invalid (see ``as_cov_frame``), has a
            non-positive variance, or implies |correlation| > 1 (not a
            covariance matrix).
    """
    frame = as_cov_frame(cov)
    values = frame.to_numpy(dtype=float)
    variances = np.diag(values)
    if (variances <= 0).any():
        bad = list(frame.index[variances <= 0])
        raise ValueError(f"variances must be positive to define correlations; bad assets: {bad}")
    vols = np.sqrt(variances)
    corr = values / np.outer(vols, vols)
    if np.abs(corr).max() > 1.0 + 1e-8:
        raise ValueError("matrix implies |correlation| > 1; it is not a valid covariance matrix")
    corr = np.clip(_symmetrize(corr), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    labels = frame.index
    return (
        pd.DataFrame(corr, index=labels.copy(), columns=labels.copy()),
        pd.Series(vols, index=labels.copy()),
    )


def is_positive_definite(cov: pd.DataFrame | np.ndarray) -> bool:
    """Whether a symmetric matrix is (numerically) positive definite, via Cholesky.

    A symmetric A is positive definite iff it has a Cholesky factorization
    A = L L' with L_ii > 0. In floating point a singular PSD matrix (e.g. a
    sample covariance with T <= N) sometimes factors with pivots that are pure
    rounding noise, so we additionally require every pivot to be clearly
    non-zero relative to its diagonal: L_ii^2 / A_ii > 1e3 * N * eps. This
    ratio is invariant to rescaling the assets, and any matrix whose
    correlation matrix has condition number below ~1e12 / N passes.

    Args:
        cov: N x N symmetric matrix (numpy or labelled DataFrame).

    Returns:
        True if positive definite, False otherwise.

    Raises:
        ValueError: if ``cov`` is not a finite symmetric square matrix.
    """
    values = as_cov_frame(cov).to_numpy(dtype=float)
    try:
        chol = np.linalg.cholesky(values)
    except np.linalg.LinAlgError:
        return False
    # A successful Cholesky implies every diagonal entry is positive, so the division is safe.
    pivots = np.diag(chol) ** 2 / np.diag(values)
    threshold = _PD_PIVOT_FACTOR * values.shape[0] * np.finfo(float).eps
    return bool(np.all(pivots > threshold))
