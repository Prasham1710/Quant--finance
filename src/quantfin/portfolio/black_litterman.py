"""Black-Litterman: blend market-equilibrium returns with an investor's views.

Why this exists: a mean-variance optimizer treats estimated expected returns as
exact, and because it effectively inverts the covariance matrix it turns small
errors in ``mu`` into huge, unstable bets. Black and Litterman (1992) replace the
raw estimate with a Bayesian posterior that is anchored to a neutral prior -- the
returns that would make the market portfolio optimal -- and that moves away from
it only as far as the investor's confidence in each view justifies.

The model (all expected returns are EXCESS returns over ``rf``, annualized):

* Prior. Reverse optimization (Sharpe, 1974) of a mean-variance investor with
  risk aversion ``delta`` who holds the market portfolio gives the implied
  equilibrium returns ``pi = delta * Sigma @ w_mkt``. The unknown mean is
  ``mu ~ N(pi, tau * Sigma)``; the scalar ``tau`` says how uncertain the
  equilibrium estimate is (typically 0.01-0.05, He and Litterman use 0.05).
* Views. ``K`` linear views ``P @ mu = Q + eps`` with ``eps ~ N(0, Omega)``:
  row ``k`` of the pick matrix ``P`` is a portfolio, ``Q_k`` its expected excess
  return and ``Omega_kk`` the variance of the error in that view.
* Posterior (Theil-Goldberger mixed estimation / Gaussian conditioning):
  ``mu | Q ~ N(mu_BL, M)`` and the predictive covariance of returns used for
  optimization is ``Sigma + M`` (He and Litterman, 1999).

References:
    F. Black and R. Litterman (1992), "Global Portfolio Optimization",
    Financial Analysts Journal 48(5), 28-43.
    W. F. Sharpe (1974), "Imputing expected security returns from portfolio
    composition", Journal of Financial and Quantitative Analysis 9(3), 463-472.
    H. Theil and A. S. Goldberger (1961), "On pure and mixed statistical
    estimation in economics", International Economic Review 2(1), 65-78.
    G. He and R. Litterman (1999), "The Intuition Behind Black-Litterman Model
    Portfolios", Goldman Sachs Investment Management Research.
    T. M. Idzorek (2005), "A step-by-step guide to the Black-Litterman model:
    incorporating user-specified confidence levels", Ibbotson Associates
    (reprinted in S. Satchell (ed.), Forecasting Expected Returns in the
    Financial Markets, 2007).
    J. Walters (2014), "The Black-Litterman Model in Detail", SSRN 1314585.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg

from quantfin.portfolio._common import TRADING_DAYS, as_cov_frame, as_series

# Relative tolerance for "positive semidefinite". For a covariance matrix,
# eigenvalues above -_PSD_RTOL * (largest |eigenvalue|) are rounding noise (a
# rank-deficient sample covariance has eigenvalues of order -1e-17 * its
# scale). Omega is judged view by view instead (see _check_omega_psd), because
# one huge omega_k -- the usual way to all but ignore a view -- would otherwise
# set the scale and wave through negative variances in every other view.
_PSD_RTOL = 1e-10

# P tau Sigma P' + Omega is rescaled to unit diagonal before it is factorized.
# The rescaled matrix has eigenvalues in [0, K]; one this close to zero means
# two views are (numerically) linearly dependent AND held with certainty, so
# the posterior is undefined rather than merely ill-conditioned.
_SINGULAR_TOL = 1e-10

# A view portfolio's prior std sqrt(p tau Sigma p') is compared with the largest
# std it could have under any correlation, sqrt(tau) * sum_i |p_i| sigma_i. Below
# this fraction (a variance ratio of 1e-12) the computed variance is rounding
# noise -- e.g. a spread between two assets with identical risk -- and dividing
# by it would turn that noise into the posterior.
_DEGENERATE_STD_RTOL = 1e-6

# Simple returns derived from prices carry an absolute rounding error of a few
# eps (that of the gross return 1 + r), so a series whose range is below this
# many eps is constant and its computed variance is pure noise.
_CONSTANT_RANGE_EPS = 64.0

# Market weights must sum to one within this tolerance.
_WEIGHT_SUM_TOL = 1e-6


@dataclass(frozen=True)
class BlackLittermanResult:
    """Posterior of the Black-Litterman model.

    Attributes:
        mu: posterior expected EXCESS returns ``mu_BL`` (annualized), by asset.
        cov: covariance of returns to optimize with, ``Sigma + M`` where ``M``
            is the posterior covariance of the mean (He and Litterman, 1999).
        prior: the prior (equilibrium) excess returns ``pi``, by asset.
        omega: the ``K x K`` view-uncertainty matrix actually used (``0 x 0``
            when there are no views).
        tau: the prior-uncertainty scalar.
    """

    mu: pd.Series
    cov: pd.DataFrame
    prior: pd.Series
    omega: np.ndarray
    tau: float


def market_implied_risk_aversion(
    market_returns: pd.Series | np.ndarray,
    rf: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Risk aversion implied by the market portfolio's risk premium.

    If the market portfolio is the optimal portfolio of a mean-variance
    investor with utility ``w'mu - (delta/2) w'Sigma w``, its first-order
    condition ``mu - rf = delta * Sigma w_mkt`` premultiplied by ``w_mkt'``
    gives (Black and Litterman, 1992)::

        delta = (E[r_m] - rf) / sigma_m^2

    with ``E[r_m] = periods_per_year * mean(r)`` and
    ``sigma_m^2 = periods_per_year * var(r)`` (sample variance, ``ddof=1``,
    matching ``estimators.cov_sample``). When ``rf == 0`` the annualization
    cancels; with ``rf != 0`` it does not, so ``rf`` must be annual.

    A sample in which the market underperformed ``rf`` yields ``delta <= 0``.
    That value is returned as estimated (it is informative), but it is not a
    valid input to :func:`implied_equilibrium_returns`.

    Args:
        market_returns: per-period simple returns of the market portfolio
            (1-D, or a single-column table).
        rf: annual risk-free rate.
        periods_per_year: periods per year used to annualize.

    Returns:
        The implied risk-aversion coefficient ``delta``.

    Raises:
        ValueError: if the series is not 1-D, has fewer than two observations,
            contains NaN/inf, has zero variance (it is constant up to rounding:
            its range is at most ``64 * eps * max(1, max|r|)``), or
            ``rf``/``periods_per_year`` are invalid.
    """
    if isinstance(market_returns, pd.DataFrame):
        if market_returns.shape[1] != 1:
            raise ValueError("market_returns must be a single series (one column)")
        values = market_returns.iloc[:, 0].to_numpy(dtype=float)
    else:
        values = np.asarray(market_returns, dtype=float)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"market_returns must be 1-D, got shape {values.shape}")
    if values.shape[0] < 2:
        raise ValueError("market_returns needs at least two observations")
    if not np.isfinite(values).all():
        raise ValueError("market_returns contain NaN or infinite values")
    rf = _finite(rf, "rf")
    ppy = _positive(periods_per_year, "periods_per_year")
    # np.mean of a constant array is generally not that constant (summation
    # rounding), so its computed variance is ~1e-35 rather than 0 and delta
    # would come out ~1e34. Test the range against rounding noise instead.
    noise = _CONSTANT_RANGE_EPS * np.finfo(float).eps * max(1.0, float(np.abs(values).max()))
    variance = float(np.var(values, ddof=1))
    if float(np.ptp(values)) <= noise or not variance > 0.0:
        raise ValueError(
            "market_returns have zero variance (constant up to rounding); "
            "risk aversion is undefined"
        )
    return (ppy * float(np.mean(values)) - rf) / (ppy * variance)


def implied_equilibrium_returns(
    cov: pd.DataFrame | np.ndarray,
    market_weights: pd.Series | np.ndarray | Sequence[float],
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Equilibrium excess returns by reverse optimization.

    The unconstrained mean-variance optimum ``w* = (delta Sigma)^-1 (mu - rf)``
    equals ``w_mkt`` exactly when (Sharpe, 1974; Black and Litterman, 1992)::

        pi = delta * Sigma @ w_mkt

    so ``pi`` is the vector of expected excess returns under which the market
    portfolio is optimal. Unlike historical means it is smooth, consistent with
    ``Sigma``, and yields well-diversified optimal portfolios.

    Args:
        cov: annualized covariance matrix ``Sigma`` (N x N).
        market_weights: market-capitalization weights summing to 1 (a Series
            is aligned to the covariance labels).
        risk_aversion: ``delta > 0``; 2.5 is the conventional choice (He and
            Litterman, 1999), or estimate it with
            :func:`market_implied_risk_aversion`.

    Returns:
        ``pi`` as a Series labelled like ``cov`` (annualized excess returns).

    Raises:
        ValueError: if ``cov`` is not a symmetric PSD matrix, the weights are
            misaligned/non-finite or do not sum to 1, or ``risk_aversion`` is
            not a positive finite number.
    """
    sigma = as_cov_frame(cov)
    _check_psd(sigma.to_numpy(), "cov")
    weights = as_series(market_weights, sigma.index)
    total = float(weights.sum())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(f"market_weights must sum to 1 (got {total:.8g}); normalize them first")
    delta = _positive(risk_aversion, "risk_aversion")
    pi = delta * (sigma.to_numpy() @ weights.to_numpy())
    return pd.Series(pi, index=sigma.index)


def build_views(
    assets: Sequence,
    absolute: Mapping[str, float] | None = None,
    relative: Sequence[tuple[str, str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the pick matrix ``P`` and view vector ``Q`` from readable views.

    * Absolute view ``{"A": 0.05}``: "A's expected excess return is 5%" -- a
      row of ``P`` with 1 on A, ``Q_k = 0.05``.
    * Relative view ``("A", "B", 0.02)``: "A outperforms B by 2%" -- a row
      with +1 on A and -1 on B (a zero-cost long-short portfolio),
      ``Q_k = 0.02``. A negative spread means A underperforms B.

    Rows are ordered as given: absolute views first, then relative ones. Row
    labels are ``"A"`` for absolute and ``"A - B"`` for relative views.

    Args:
        assets: asset labels; they become the columns of ``P`` in this order.
        absolute: mapping asset -> expected excess return (a Series works).
        relative: sequence of (outperformer, underperformer, spread) tuples.

    Returns:
        ``(P, Q)``: ``P`` is a K x N DataFrame (columns = ``assets``), ``Q`` a
        length-K Series sharing ``P``'s row labels.

    Raises:
        ValueError: on empty or duplicate asset labels, unknown assets,
            malformed relative views, a relative view of an asset against
            itself, non-finite view values, or when no view is given.
    """
    if isinstance(assets, str):
        raise ValueError("assets must be a sequence of labels, not a single string")
    labels = list(assets)
    if not labels:
        raise ValueError("assets must not be empty")
    if len(set(labels)) != len(labels):
        raise ValueError("asset labels must be unique")
    position = {asset: i for i, asset in enumerate(labels)}

    rows: list[np.ndarray] = []
    values: list[float] = []
    names: list[str] = []

    if absolute is not None and not hasattr(absolute, "items"):
        raise ValueError("absolute views must be a mapping asset -> expected excess return")
    for asset, value in absolute.items() if absolute is not None else ():
        if asset not in position:
            raise ValueError(f"unknown asset in absolute view: {asset!r}")
        row = np.zeros(len(labels))
        row[position[asset]] = 1.0
        rows.append(row)
        values.append(_finite(value, f"absolute view on {asset!r}"))
        names.append(str(asset))

    for view in relative if relative is not None else ():
        if isinstance(view, str) or not isinstance(view, Sequence) or len(view) != 3:
            raise ValueError(
                f"relative view {view!r} must be an (outperformer, underperformer, spread) tuple"
            )
        winner, loser, value = view
        for asset in (winner, loser):
            if asset not in position:
                raise ValueError(f"unknown asset in relative view: {asset!r}")
        if winner == loser:
            raise ValueError(f"relative view compares {winner!r} with itself")
        row = np.zeros(len(labels))
        row[position[winner]] = 1.0
        row[position[loser]] = -1.0
        rows.append(row)
        values.append(_finite(value, f"relative view {winner!r} vs {loser!r}"))
        names.append(f"{winner} - {loser}")

    if not rows:
        raise ValueError("no views given: pass absolute and/or relative views")
    columns = assets if isinstance(assets, pd.Index) else pd.Index(labels)
    index = pd.Index(names)
    pick = pd.DataFrame(np.vstack(rows), index=index, columns=columns)
    return pick, pd.Series(values, index=index, dtype=float)


def he_litterman_omega(
    cov: pd.DataFrame | np.ndarray,
    P: pd.DataFrame | np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """View uncertainty proportional to the prior variance of each view portfolio.

    He and Litterman (1999) set::

        Omega = diag(tau * P Sigma P')

    i.e. each view is exactly as uncertain as the prior estimate of the same
    portfolio's return. For a single view the posterior view return is then the
    midpoint of the prior ``p'pi`` and the view ``q``; and because ``Omega`` is
    proportional to ``tau``, the posterior MEAN does not depend on ``tau``.

    Args:
        cov: annualized covariance matrix (N x N).
        P: pick matrix (K x N, or a length-N vector for one view). A DataFrame
            is aligned to the covariance labels by column name.
        tau: prior-uncertainty scalar, > 0.

    Returns:
        The diagonal ``K x K`` matrix ``Omega``.

    Raises:
        ValueError: on invalid ``cov``, ``P`` or ``tau``.
    """
    sigma = as_cov_frame(cov)
    _check_psd(sigma.to_numpy(), "cov")
    pick, _ = _as_pick_matrix(P, sigma.index)
    tau = _positive(tau, "tau")
    return np.diag(tau * _view_variances(sigma.to_numpy(), pick))


def idzorek_omega(
    cov: pd.DataFrame | np.ndarray,
    P: pd.DataFrame | np.ndarray,
    confidences: Sequence[float] | np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """View uncertainty from intuitive confidence levels (Idzorek's method).

    Idzorek (2005) lets the user state a confidence ``c_k`` in (0, 1] for each
    view and searches for the ``omega_k`` at which the portfolio tilt caused by
    that view alone is ``c_k`` times the tilt at 100% confidence. Walters (2014)
    shows the search has a closed form::

        omega_k = ((1 - c_k) / c_k) * p_k (tau Sigma) p_k'

    Derivation for one view with ``s = p (tau Sigma) p'``: the posterior mean
    is ``pi + tau Sigma p' (q - p'pi) / (s + omega)`` and the unconstrained
    optimum ``(delta Sigma)^-1 mu`` is ``w_mkt + tau p' (q - p'pi) /
    (delta (s + omega))``. Its tilt relative to ``omega = 0`` is
    ``s / (s + omega)``, which equals ``c`` exactly for the ``omega`` above.
    (The identity is exact when the optimizer uses ``Sigma``; with the
    posterior ``Sigma + M`` it is approximate.) Consequently ``p'mu_BL`` moves
    exactly the fraction ``c`` of the way from ``p'pi`` to ``q``.

    ``c = 1`` gives ``omega = 0`` (the view holds with certainty) and
    ``c -> 0`` sends ``omega -> inf`` (the view is ignored); ``c = 0.5`` equals
    :func:`he_litterman_omega`.

    Args:
        cov: annualized covariance matrix (N x N).
        P: pick matrix (K x N, or a length-N vector for one view).
        confidences: one confidence per view, each in (0, 1]. A Series or
            mapping keyed by view label is aligned to the row labels of a
            labelled ``P`` (as :func:`black_litterman` aligns ``Q``); any other
            sequence is taken in the order of ``P``'s rows.
        tau: prior-uncertainty scalar, > 0.

    Returns:
        The diagonal ``K x K`` matrix ``Omega``.

    Raises:
        ValueError: on invalid ``cov``, ``P`` or ``tau``, a wrong number of
            confidences, labelled confidences whose labels do not match ``P``'s
            rows (or a mapping with an unlabelled ``P``), non-numeric
            confidences, a confidence outside (0, 1], or one so close to zero
            that ``omega`` overflows.
    """
    sigma = as_cov_frame(cov)
    _check_psd(sigma.to_numpy(), "cov")
    pick, view_labels = _as_pick_matrix(P, sigma.index)
    tau = _positive(tau, "tau")
    conf = _as_confidences(confidences, view_labels)
    if conf.shape[0] != pick.shape[0]:
        raise ValueError(
            f"expected {pick.shape[0]} confidences (one per view), got {conf.shape[0]}"
        )
    if not np.isfinite(conf).all() or (conf <= 0.0).any() or (conf > 1.0).any():
        raise ValueError("confidences must lie in (0, 1]")
    # Tiny confidences overflow (1 - c) / c; that is reported, not returned as inf.
    with np.errstate(over="ignore"):
        omega_diag = (1.0 - conf) / conf * (tau * _view_variances(sigma.to_numpy(), pick))
    if not np.isfinite(omega_diag).all():
        raise ValueError("a confidence is too close to 0; drop that view instead")
    return np.diag(omega_diag)


def black_litterman(
    cov: pd.DataFrame | np.ndarray,
    prior: pd.Series | np.ndarray | Sequence[float],
    P: pd.DataFrame | np.ndarray | None = None,
    Q: pd.Series | np.ndarray | Sequence[float] | None = None,
    omega: pd.DataFrame | np.ndarray | None = None,
    tau: float = 0.05,
) -> BlackLittermanResult:
    """Black-Litterman posterior expected returns and covariance.

    With prior ``mu ~ N(pi, tau Sigma)`` and views ``P mu = Q + eps``,
    ``eps ~ N(0, Omega)``, Gaussian conditioning (equivalently Theil-Goldberger
    mixed estimation) gives::

        A     = P (tau Sigma) P' + Omega
        mu_BL = pi + (tau Sigma) P' A^-1 (Q - P pi)
        M     = tau Sigma - (tau Sigma) P' A^-1 P (tau Sigma)
        Sigma_BL = Sigma + M

    (Black and Litterman, 1992; He and Litterman, 1999; Walters, 2014). This
    form needs only ``A`` to be invertible, so it stays valid for views held
    with certainty (``Omega = 0``, then ``P mu_BL = Q`` exactly and
    ``P M = 0``), where the textbook precision form
    ``[(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 Q]``
    breaks down. ``A`` is factorized after rescaling to unit diagonal, so a
    nearly-ignored view (huge ``omega_k``) does not degrade the others.

    ``Omega`` must be PSD up to rounding, judged view by view so that a huge
    ``omega_k`` cannot hide a negative variance elsewhere: with
    ``c_k = max(omega_kk, p_k tau Sigma p_k')`` it requires
    ``Omega + 1e-10 diag(c) >= 0``. A negative ``omega_kk`` would mean "more
    certain than certain": ``p_k mu_BL`` would overshoot ``Q_k`` and ``M``
    would not be PSD.

    Without views (``P`` and ``Q`` None) the posterior is the prior:
    ``mu_BL = pi`` and ``Sigma_BL = (1 + tau) Sigma``.

    Args:
        cov: annualized covariance of returns ``Sigma`` (N x N, PSD).
        prior: prior expected excess returns ``pi`` (usually
            :func:`implied_equilibrium_returns`); a Series is aligned to the
            covariance labels.
        P: pick matrix (K x N, or a length-N vector for one view). A DataFrame
            must have exactly the covariance assets as columns (any order).
        Q: expected excess return of each view portfolio (length K). If both
            ``P`` and ``Q`` are labelled, ``Q`` is aligned to ``P``'s rows by
            label; otherwise by position.
        omega: K x K symmetric PSD view-uncertainty matrix, or a length-K
            vector of view variances taken as its diagonal. Like ``Q``, a
            labelled ``omega`` (a DataFrame with identical index and columns,
            or a Series of variances) is aligned to ``P``'s rows by label when
            ``P`` is labelled; otherwise it is taken by position. Default:
            :func:`he_litterman_omega`.
        tau: prior-uncertainty scalar, > 0.

    Returns:
        A :class:`BlackLittermanResult` with labelled posterior mean (excess
        returns) and covariance, the prior, the ``Omega`` used, and ``tau``.

    Raises:
        ValueError: on mismatched shapes or labels, non-finite input, a
            covariance or ``Omega`` that is not symmetric PSD (for ``Omega``,
            judged per view as above), a zero row in
            ``P``, ``tau <= 0``, only one of ``P``/``Q`` given, ``omega``
            without views, ``A`` singular -- a view held with certainty (or
            with the default He-Litterman ``Omega``) on a portfolio whose prior
            variance is zero up to rounding, e.g. a spread between two assets
            with identical risk, or certain views that are linearly
            dependent -- or ``Omega`` so large that ``A`` overflows.
    """
    sigma_frame = as_cov_frame(cov)
    assets = sigma_frame.index
    sigma = sigma_frame.to_numpy()
    _check_psd(sigma, "cov")
    # Copied so the frozen result never shares memory with the caller's array
    # (pandas 2.x wraps a float64 ndarray without copying it).
    pi = as_series(prior, assets).to_numpy().copy()
    tau = _positive(tau, "tau")

    if P is None and Q is None:
        if omega is not None:
            raise ValueError("omega was given without views (P and Q)")
        return BlackLittermanResult(
            mu=pd.Series(pi.copy(), index=assets),
            cov=pd.DataFrame((1.0 + tau) * sigma, index=assets, columns=assets),
            prior=pd.Series(pi, index=assets),
            omega=np.zeros((0, 0)),
            tau=tau,
        )
    if P is None or Q is None:
        raise ValueError("P and Q must be given together (or both omitted for no views)")

    pick, view_labels = _as_pick_matrix(P, assets)
    n_views = pick.shape[0]
    q = _as_view_returns(Q, n_views, view_labels)
    if omega is None:
        omega_used = np.diag(tau * _view_variances(sigma, pick))
    else:
        omega_used = _as_omega(omega, n_views, view_labels)

    tau_sigma = tau * sigma
    # Row k of P tau Sigma is the prior covariance of view portfolio k with each asset.
    view_cov = pick @ tau_sigma
    # A finite but huge omega_k (a stand-in for "ignore this view") or huge P
    # entries can overflow; that is reported below rather than left to warn.
    with np.errstate(over="ignore", invalid="ignore"):
        prior_view_cov = view_cov @ pick.T
        a = _symmetrize(prior_view_cov + omega_used)
        # Largest prior std view k could have under any correlation:
        # sqrt(tau) * sum_i |p_ki| sigma_i (sigma_i clipped at 0 for rounding).
        max_view_std = np.sqrt(tau) * (np.abs(pick) @ np.sqrt(np.maximum(np.diag(sigma), 0.0)))
    if not np.isfinite(a).all():
        raise ValueError(
            "P tau Sigma P' + Omega overflows: omega (or P) entries are too large; "
            "drop views you want ignored instead of giving them a near-infinite omega"
        )
    if omega is not None:
        # Finite a (with finite Omega) implies P tau Sigma P' is finite too.
        _check_omega_psd(omega_used, np.diag(prior_view_cov), max_view_std)
    solved = _solve_view_system(a, np.column_stack([q - pick @ pi, view_cov]), max_view_std)
    mu = pi + view_cov.T @ solved[:, 0]
    m = _symmetrize(tau_sigma - view_cov.T @ solved[:, 1:])
    return BlackLittermanResult(
        mu=pd.Series(mu, index=assets),
        cov=pd.DataFrame(_symmetrize(sigma + m), index=assets, columns=assets),
        prior=pd.Series(pi, index=assets),
        omega=omega_used,
        tau=tau,
    )


def _finite(value: float, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _positive(value: float, name: str) -> float:
    out = _finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return out


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    # Halve first: 0.5 * (M + M') overflows for finite entries above ~9e307.
    return 0.5 * matrix + 0.5 * matrix.T


def _check_psd(matrix: np.ndarray, name: str) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = float(np.abs(eigenvalues).max())
    if eigenvalues[0] < -_PSD_RTOL * scale:
        raise ValueError(
            f"{name} must be positive semidefinite (smallest eigenvalue {eigenvalues[0]:.3g})"
        )


def _view_variances(sigma: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """diag(P Sigma P') without forming the K x K product; clipped at 0 for rounding."""
    return np.maximum(np.einsum("ki,ij,kj->k", pick, sigma, pick), 0.0)


def _as_pick_matrix(
    P: pd.DataFrame | np.ndarray, assets: pd.Index
) -> tuple[np.ndarray, list | None]:
    """Validate ``P`` and return it as a K x N array in ``assets`` order plus its row labels."""
    labels: list | None = None
    if isinstance(P, pd.Series):
        # One view given as a Series of asset weights.
        P = P.to_frame().T
    if isinstance(P, pd.DataFrame):
        columns = list(P.columns)
        if len(set(columns)) != len(columns):
            raise ValueError("P has duplicate column labels")
        known, given = set(assets), set(columns)
        extra = [c for c in columns if c not in known]
        missing = [a for a in assets if a not in given]
        if extra or missing:
            raise ValueError(
                f"P columns must be exactly the covariance assets "
                f"(unknown: {extra}, missing: {missing})"
            )
        pick = P.reindex(columns=assets).to_numpy(dtype=float)
        labels = list(P.index)
    else:
        pick = np.asarray(P, dtype=float)
        if pick.ndim == 1:
            pick = pick.reshape(1, -1)
        if pick.ndim != 2 or pick.shape[1] != len(assets):
            raise ValueError(f"P must have shape (K, {len(assets)}), got {np.shape(P)}")
    if pick.shape[0] == 0:
        raise ValueError("P has no views (zero rows)")
    if not np.isfinite(pick).all():
        raise ValueError("P contains NaN or infinite values")
    zero_rows = np.flatnonzero(~pick.any(axis=1))
    if zero_rows.size:
        raise ValueError(f"P has all-zero view rows at positions {zero_rows.tolist()}")
    return pick, labels


def _label_order(index: pd.Index, view_labels: list, name: str) -> np.ndarray | None:
    """Positions that reorder a labelled ``index`` to ``view_labels`` (None = keep order)."""
    if list(index) == view_labels:
        return None
    if (
        index.is_unique
        and len(set(view_labels)) == len(view_labels)
        and set(index) == set(view_labels)
    ):
        return index.get_indexer(view_labels)
    raise ValueError(
        f"{name} labels do not match the row labels of P; pass {name} as a plain array "
        "to align it by position"
    )


def _as_view_returns(Q, n_views: int, view_labels: list | None) -> np.ndarray:
    if isinstance(Q, pd.Series) and view_labels is not None:
        order = _label_order(Q.index, view_labels, "Q")
        values = Q.to_numpy(dtype=float)
        if order is not None:
            values = values[order]
    else:
        values = np.asarray(Q, dtype=float).reshape(-1)
    if values.shape[0] != n_views:
        raise ValueError(f"Q must have one entry per view ({n_views}), got {values.shape[0]}")
    if not np.isfinite(values).all():
        raise ValueError("Q contains NaN or infinite values")
    return values


def _as_confidences(confidences, view_labels: list | None) -> np.ndarray:
    """Idzorek confidences as a flat array in the order of ``P``'s rows.

    Labelled input (a Series or mapping) is aligned to ``view_labels`` like
    ``Q`` is; a mapping with an unlabelled ``P`` has nothing to align to.
    """
    if isinstance(confidences, Mapping) and not isinstance(confidences, pd.Series):
        if view_labels is None:
            raise ValueError(
                "confidences given as a mapping need a labelled P (a DataFrame); "
                "pass a plain sequence to align them by position"
            )
        confidences = pd.Series(list(confidences.values()), index=list(confidences.keys()))
    order = None
    if isinstance(confidences, pd.Series) and view_labels is not None:
        order = _label_order(confidences.index, view_labels, "confidences")
    try:
        values = np.asarray(confidences, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confidences must be numbers in (0, 1], got {confidences!r}") from exc
    return values if order is None else values[order]


def _as_omega(omega, n_views: int, view_labels: list | None) -> np.ndarray:
    """``omega`` as a finite symmetric K x K array in the order of ``P``'s rows.

    Labelled input -- a DataFrame, or a Series of view variances -- is aligned
    to ``view_labels`` like ``Q`` is. Positive semidefiniteness needs the view
    scales and is checked separately by :func:`_check_omega_psd`.
    """
    if isinstance(omega, pd.DataFrame):
        values = omega.to_numpy(dtype=float)
        if view_labels is not None and values.shape == (n_views, n_views):
            if list(omega.index) != list(omega.columns):
                raise ValueError("omega DataFrame must have identical index and columns")
            order = _label_order(omega.index, view_labels, "omega")
            if order is not None:
                values = values[np.ix_(order, order)]
    elif isinstance(omega, pd.Series) and view_labels is not None:
        order = _label_order(omega.index, view_labels, "omega")
        values = omega.to_numpy(dtype=float)
        if order is not None:
            values = values[order]
    else:
        values = np.asarray(omega, dtype=float)
    if values.ndim == 1:
        values = np.diag(values)
    if values.shape != (n_views, n_views):
        raise ValueError(f"omega must have shape ({n_views}, {n_views}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("omega contains NaN or infinite values")
    scale = max(1.0, float(np.abs(values).max()))
    # Half-differences cannot overflow, unlike values - values.T near the float maximum.
    if float(np.abs(0.5 * values - 0.5 * values.T).max()) > 0.5e-10 * scale:
        raise ValueError("omega must be symmetric")
    return _symmetrize(values)


def _check_omega_psd(
    omega: np.ndarray, prior_view_var: np.ndarray, max_view_std: np.ndarray
) -> None:
    """Raise unless the symmetric ``omega`` is PSD up to rounding, judged per view.

    The covariance test (eigenvalues above ``-_PSD_RTOL * max|eigenvalue|``
    are rounding noise) does not work here: one huge ``omega_k`` (the usual
    stand-in for "ignore this view") sets that scale and waves through
    negative variances of order ``1e-10 * omega_k`` in every other view. Those
    make ``p_j mu_BL`` overshoot ``Q_j`` and ``M`` indefinite. Instead each
    view ``k`` is measured on its own variance scale::

        c_k = max(omega_kk, p_k tau Sigma p_k', (1e-6 * max_view_std_k)^2)

    (the last term is the noise floor below which the solver treats a prior
    view variance as zero), and the test is ``Omega + _PSD_RTOL * diag(c) >= 0``,
    i.e. ``lambda_min(C^-1/2 Omega C^-1/2) >= -_PSD_RTOL``. By Sylvester's law
    of inertia the rescaling never changes WHETHER ``Omega`` is PSD. It only
    decides what counts as rounding, and does so in the units in which
    ``Omega`` enters the model: ``A = P tau Sigma P' + Omega`` has
    ``a_kk >= c_k`` (up to the floor), so a negative variance is tolerated
    only at 1e-10 of the view's own scale, where it cannot visibly move the
    posterior. The verdict is therefore unchanged by restating any view as a
    multiple of itself (scaling its row of ``P``, ``Q`` and ``omega``).

    ``c_k = 0`` happens only for a view on riskless assets with
    ``omega_kk <= 0``. PSD then requires that view's whole row of ``omega``
    to be zero.
    """
    diagonal = np.diag(omega)
    with np.errstate(over="ignore"):
        noise_floor = np.square(_DEGENERATE_STD_RTOL * max_view_std)
    scale = np.maximum(diagonal, np.maximum(prior_view_var, noise_floor))
    # Written without dividing by scale so a riskless view (scale 0) is covered.
    negative = np.flatnonzero(diagonal < -_PSD_RTOL * scale)
    if negative.size:
        raise ValueError(
            f"omega must be positive semidefinite: views at positions {negative.tolist()} "
            "have a negative variance (omega_kk < 0, beyond rounding relative to the "
            "view's prior variance), i.e. they would be more than certain"
        )
    unscaled = ~(scale > 0.0)
    coupled = np.flatnonzero(unscaled & (omega != 0.0).any(axis=1))
    if coupled.size:
        raise ValueError(
            f"omega must be positive semidefinite: views at positions {coupled.tolist()} "
            "have zero variance but a nonzero covariance with other views"
        )
    keep = np.flatnonzero(~unscaled)
    if keep.size == 0:
        return
    inv_sqrt = 1.0 / np.sqrt(scale[keep])
    # Rows then columns, as in _solve_view_system. Every diagonal entry of the
    # result is <= 1, so an off-diagonal that overflows is itself proof that
    # Omega is indefinite (|r_kj| <= sqrt(r_kk r_jj) <= 1 for a PSD matrix).
    with np.errstate(over="ignore", invalid="ignore"):
        r = _symmetrize((inv_sqrt[:, None] * omega[np.ix_(keep, keep)]) * inv_sqrt[None, :])
    smallest = float(np.linalg.eigvalsh(r)[0]) if np.isfinite(r).all() else -np.inf
    if smallest < -_PSD_RTOL:
        raise ValueError(
            "omega must be positive semidefinite (smallest eigenvalue "
            f"{smallest:.3g} after rescaling each view to its own variance)"
        )


def _solve_view_system(a: np.ndarray, rhs: np.ndarray, max_view_std: np.ndarray) -> np.ndarray:
    """Solve ``a x = rhs`` for the SPD view matrix ``a = P tau Sigma P' + Omega``.

    ``a`` is rescaled to ``R = D a D`` with ``D = diag(a)^-1/2`` (unit diagonal)
    and factorized by Cholesky, so wildly different view scales -- e.g. one
    view with a near-infinite ``omega_k`` -- neither trip the singularity test
    nor cost accuracy for the other views.

    Because ``R`` always has a unit diagonal, the rescaling would hide a view
    whose ``a_kk`` is rounding noise. So before rescaling, ``sqrt(a_kk)`` is
    compared with ``max_view_std[k]``, the largest prior std view ``k`` could
    have under any correlation (``sqrt(tau) sum_i |p_ki| sigma_i``).
    """
    diagonal = np.diag(a)
    # Written as "not above" so a zero (or rounding-negative) diagonal with a
    # zero bound -- a view on riskless assets held with certainty -- is caught.
    usable = np.sqrt(np.maximum(diagonal, 0.0)) > _DEGENERATE_STD_RTOL * max_view_std
    if not usable.all():
        bad = np.flatnonzero(~usable).tolist()
        raise ValueError(
            f"views at positions {bad} have zero prior variance and zero uncertainty "
            "(up to rounding), so P tau Sigma P' + Omega is singular: the prior already "
            "fixes the return of such a view portfolio (e.g. a spread between two assets "
            "with identical risk); drop the view or give it a positive omega"
        )
    scale = 1.0 / np.sqrt(diagonal)
    # Scale rows then columns: forming outer(scale, scale) first would go
    # subnormal for a_kk near the float maximum.
    r = _symmetrize((scale[:, None] * a) * scale[None, :])
    if np.linalg.eigvalsh(r)[0] <= _SINGULAR_TOL:
        raise ValueError(
            "P tau Sigma P' + Omega is singular: some views are linearly dependent and "
            "held with (near) certainty; remove redundant views or give them uncertainty"
        )
    try:
        factor = scipy.linalg.cho_factor(r, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError("P tau Sigma P' + Omega is not positive definite") from exc
    return scale[:, None] * scipy.linalg.cho_solve(factor, scale[:, None] * rhs)
