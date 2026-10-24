"""Hierarchical Risk Parity (López de Prado, 2016).

HRP builds a diversified long-only portfolio from a covariance matrix in three
stages, none of which inverts the covariance matrix:

1. **Tree clustering.** Correlations become distances
   ``d_ij = sqrt((1 - ρ_ij) / 2)``, and the assets are clustered on the
   *distance of distances* ``d̃_ij = ||d_·i - d_·j||₂`` (the Euclidean distance
   between columns ``i`` and ``j`` of ``d``), so two assets are close when
   their whole correlation profiles against the universe are similar, not
   only their pairwise correlation (:func:`correlation_distance`,
   :func:`hrp_linkage`).
2. **Quasi-diagonalization.** The leaves of the dendrogram are read left to
   right; reordering the covariance matrix in this order places similar assets
   next to each other, so large covariances concentrate around the diagonal
   (:func:`quasi_diagonal_order`).
3. **Recursive bisection.** Starting from the whole ordered list, every
   cluster is cut into two halves *by position*; the halves split the
   cluster's weight in inverse proportion to their variances, each half's
   variance being that of its own inverse-variance portfolio
   (:func:`recursive_bisection`).

Why HRP belongs in a comparison with mean-variance optimization. A Markowitz
optimizer applies ``Σ⁻¹``; when ``Σ`` is ill-conditioned (many correlated
assets, short samples) the inverse amplifies estimation error into extreme,
unstable weights -- "Markowitz's curse" in López de Prado's words. HRP only
ever uses variances of inverse-variance sub-portfolios (from halves of the
universe down to single assets), so it works even with a singular covariance
matrix. Its weights always lie in [0, 1] and are far less erratic than
``Σ⁻¹``-based weights, but they are only *piecewise* smooth in the inputs: a
change in the cluster tree (e.g. nearly tied distances resolving the other
way) changes which assets end up in which half and can make the weights jump.
With three equally correlated assets, a 1e-9 change in one correlation moves
weights by about 4 percentage points. López de Prado (2016) reports lower
out-of-sample variance than the minimum-variance portfolio in Monte Carlo
experiments.

Inverse-variance special case. If all correlations are zero, HRP returns the
inverse-variance portfolio ``w_i ∝ 1/σ_i²`` *for any ordering*. Proof: let
``S_C = Σ_{i∈C} σ_i⁻²``. The inverse-variance portfolio of a cluster ``C`` has
weights ``σ_i⁻² / S_C`` and, with a diagonal covariance, variance
``V_C = Σ_{i∈C} σ_i⁻⁴ σ_i² / S_C² = 1/S_C``. A bisection ``C = L ∪ R`` gives
the left half ``α = 1 - V_L/(V_L + V_R) = S_L/(S_L + S_R)`` of the cluster's
weight, i.e. exactly its share of the cluster's total inverse variance, so by
induction every asset ends with ``σ_i⁻² / Σ_j σ_j⁻²``.

The implementation follows the paper's reference code (``correlDist``,
``getQuasiDiag``, ``getRecBipart``, ``getClusterVar``, ``getIVP``) exactly,
including the floor split ``|L₁| = ⌊|L|/2⌋`` of odd-sized clusters. Note that
the bisection follows positions in the quasi-diagonal order, not the branches
of the dendrogram; later variants that cut along the tree are not implemented.

Conventions follow :mod:`quantfin.portfolio._common`: ``cov`` is an ANNUALIZED
covariance matrix (HRP is invariant to multiplying ``cov`` by a positive
constant, so daily or annual units give identical weights), numpy input is
labelled 0..N-1 and weights are returned as a ``pd.Series`` in the covariance
matrix's original asset order.

References:
    M. López de Prado (2016), "Building diversified portfolios that outperform
    out of sample", Journal of Portfolio Management 42(4), 59-69.
    M. López de Prado (2018), "Advances in Financial Machine Learning",
    Wiley, Chapter 16.
    R. Mantegna (1999), "Hierarchical structure in financial markets",
    European Physical Journal B 11, 193-197 (the correlation distance).
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.cluster import hierarchy
from scipy.spatial import distance

from quantfin.portfolio._common import as_cov_frame

__all__ = [
    "correlation_distance",
    "hrp_linkage",
    "hrp_weights",
    "quasi_diagonal_order",
    "recursive_bisection",
]

_EPS = float(np.finfo(float).eps)

# Methods accepted by scipy.cluster.hierarchy.linkage. The centroid, median and
# Ward updates are only meaningful for Euclidean distances; d̃ is one (it is
# the Euclidean distance between the columns of d), so all of them are valid.
_LINKAGE_METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")

# A correlation matrix must have a unit diagonal and entries in [-1, 1]. Values
# outside by less than this are rounding noise from however the matrix was
# computed and are clipped; anything further out is rejected.
_CORR_TOL = 1e-8

# A valid correlation matrix is PSD with eigenvalues in [0, N]. Negative
# eigenvalues smaller in magnitude than this (times N) are rounding noise.
_PSD_TOL = 1e-10

# The computed variance w'Σw of a cluster carries a rounding error of roughly
# N * eps * w'|Σ|w (w >= 0 here). Two cluster variances that are both below a
# generous multiple of that bound cannot be told apart from zero, and the
# split α = V_R / (V_L + V_R) would then be pure noise.
_ZERO_VARIANCE_FACTOR = 64.0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_method(method: str) -> str:
    if not isinstance(method, str) or method not in _LINKAGE_METHODS:
        raise ValueError(f"method must be one of {_LINKAGE_METHODS}, got {method!r}")
    return method


def _check_psd(corr: np.ndarray, what: str) -> None:
    """Reject a (unit-diagonal) matrix with a clearly negative eigenvalue."""
    min_eig = float(linalg.eigvalsh(corr)[0])
    if min_eig < -_PSD_TOL * corr.shape[0]:
        raise ValueError(
            f"{what} is not positive semi-definite "
            f"(smallest eigenvalue of the correlation matrix is {min_eig:.3g})"
        )


def _corr_frame(corr: pd.DataFrame | np.ndarray, assets: Sequence | None = None) -> pd.DataFrame:
    """``as_cov_frame`` for a correlation matrix, with errors that blame ``corr``.

    ``as_cov_frame``'s messages speak of a "covariance"; when the matrix at
    fault is the correlation (possibly passed next to a perfectly valid
    ``cov``), the user must be pointed at the right argument.
    """
    try:
        return as_cov_frame(corr, assets=assets)
    except ValueError as exc:
        message = str(exc).replace("covariance", "correlation")
        raise ValueError(f"invalid corr: {message}") from exc


def _correlation_values(frame: pd.DataFrame) -> np.ndarray:
    """Validate a square symmetric frame as a correlation matrix; return clean values.

    The returned array has an exact unit diagonal, entries clipped to [-1, 1]
    (so ``sqrt((1 - ρ)/2)`` can never see a negative argument) and is exactly
    symmetric (``as_cov_frame`` symmetrized it).
    """
    values = frame.to_numpy(dtype=float, copy=True)
    diag = np.diag(values)
    if not (np.abs(diag - 1.0) <= _CORR_TOL).all():
        raise ValueError(
            "correlation matrix must have a unit diagonal "
            f"(diagonal ranges from {diag.min():.6g} to {diag.max():.6g}); "
            "did you pass a covariance matrix?"
        )
    largest = float(np.abs(values).max())
    if not largest <= 1.0 + _CORR_TOL:
        raise ValueError(f"correlations must lie in [-1, 1], found |ρ| = {largest:.6g}")
    np.fill_diagonal(values, 1.0)
    _check_psd(values, "correlation matrix")
    np.clip(values, -1.0, 1.0, out=values)
    return values


def _covariance(cov: pd.DataFrame | np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Validated covariance frame and its implied correlation matrix.

    Requires strictly positive variances (inverse-variance weights need them)
    and a positive semi-definite matrix (otherwise a cluster variance could
    be negative and push the split α outside [0, 1]).
    """
    sigma = as_cov_frame(cov)
    values = sigma.to_numpy()
    if not np.isfinite(values).all():
        # as_cov_frame checks finiteness before it symmetrizes with
        # 0.5 * (Σ + Σ'), which overflows for entries above ~8.99e307. Say
        # so instead of misreporting the inf/NaN downstream as a PSD failure.
        raise ValueError(
            "covariance entries are too large to process (|cov_ij| must stay below about "
            "8.9e307); rescale it -- HRP weights do not change when cov is multiplied by a "
            "positive constant"
        )
    variances = np.diag(values)
    bad = [a for a, v in zip(sigma.index, variances, strict=True) if not v > 0.0]
    if bad:
        raise ValueError(f"every asset needs a strictly positive variance; offending: {bad}")
    vols = np.sqrt(variances)
    # Divide by one volatility at a time: the outer product σ_i σ_j could
    # underflow to zero for tiny variances even though each ratio is fine.
    # A matrix far from PSD can still have |Σ_ij| / (σ_i σ_j) beyond the float
    # range; that overflow is a PSD violation (|ρ_ij| > 1), reported as such
    # below rather than as a numpy warning or a NaN error from the eigensolver.
    with np.errstate(over="ignore", invalid="ignore"):
        corr = values / vols[:, None] / vols[None, :]
        corr = 0.5 * (corr + corr.T)
    if not np.isfinite(corr).all():
        raise ValueError(
            "covariance matrix is not positive semi-definite "
            "(some |cov_ij| / sqrt(cov_ii cov_jj) overflows, i.e. |ρ_ij| > 1)"
        )
    np.fill_diagonal(corr, 1.0)
    # PSD check before clipping: clipping could hide a correlation far above 1.
    _check_psd(corr, "covariance matrix")
    np.clip(corr, -1.0, 1.0, out=corr)
    return sigma, corr


def _aligned_corr(corr: pd.DataFrame | np.ndarray, sigma: pd.DataFrame) -> pd.DataFrame:
    """Return ``corr`` as a frame in the covariance matrix's asset order."""
    labels = list(sigma.index)
    if isinstance(corr, pd.DataFrame):
        frame = _corr_frame(corr)
        corr_labels = list(frame.index)
        if corr_labels == labels:
            return frame
        unique = len(set(labels)) == len(labels) and len(set(corr_labels)) == len(corr_labels)
        if not unique or set(corr_labels) != set(labels):
            raise ValueError("corr must be labelled with exactly the same assets as cov")
        return frame.loc[labels, labels]
    values = np.asarray(corr, dtype=float)
    if values.shape != (len(labels), len(labels)):
        raise ValueError(
            f"corr must be a {len(labels)}x{len(labels)} matrix like cov, got shape {values.shape}"
        )
    return _corr_frame(values, assets=labels)


def _check_order(order: Sequence[int], n_assets: int) -> list[int]:
    """Validate that ``order`` is a permutation of the positions 0..N-1."""
    try:
        items = list(order)
        if any(isinstance(i, (bool, np.bool_)) for i in items):
            raise TypeError("booleans are not positions")
        positions = [operator.index(i) for i in items]
    except TypeError as exc:
        raise ValueError(
            f"order must be a sequence of integer positions 0..{n_assets - 1}"
        ) from exc
    if sorted(positions) != list(range(n_assets)):
        raise ValueError(
            f"order must be a permutation of the positions 0..{n_assets - 1}, got {positions}"
        )
    return positions


# ---------------------------------------------------------------------------
# Core computations (inputs already validated)
# ---------------------------------------------------------------------------


def _distance_values(rho: np.ndarray) -> np.ndarray:
    # rho is clipped to [-1, 1], so the argument of the square root is in [0, 1].
    return np.sqrt(0.5 * (1.0 - rho))


def _linkage(rho: np.ndarray, method: str) -> np.ndarray:
    d = _distance_values(rho)
    # Distance of distances: each column of d is an asset's distance profile
    # against the whole universe; pdist returns the condensed form of d̃, i.e.
    # squareform(d̃). This is what the paper's code computes implicitly when it
    # passes the square matrix d to scipy's linkage as an observation matrix.
    d_tilde = distance.pdist(d.T, metric="euclidean")
    return hierarchy.linkage(d_tilde, method=method)


def _cluster_variance(cov: np.ndarray, items: np.ndarray) -> tuple[float, float]:
    """Inverse-variance portfolio variance of a cluster and its rounding-error bound."""
    sub = cov[np.ix_(items, items)]
    variances = np.diag(sub)
    # Scale by the smallest variance so 1/σ² cannot overflow for tiny variances.
    inv = variances.min() / variances
    w = inv / inv.sum()
    variance = float(w @ sub @ w)
    rounding = _ZERO_VARIANCE_FACTOR * len(items) * _EPS * float(w @ np.abs(sub) @ w)
    # Σ is PSD, so a negative value is rounding noise around zero.
    return max(variance, 0.0), rounding


def _bisect(cov: np.ndarray, order: Sequence[int]) -> np.ndarray:
    weights = np.ones(cov.shape[0])
    pending = [np.asarray(order, dtype=np.intp)]
    # Each split depends only on Σ and the cluster's members, never on the
    # weights accumulated so far, so a depth-first stack yields exactly the
    # paper's level-by-level result.
    while pending:
        items = pending.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2  # the paper's int(len/2): the left half is the smaller one
        left, right = items[:half], items[half:]
        v_left, r_left = _cluster_variance(cov, left)
        v_right, r_right = _cluster_variance(cov, right)
        total = v_left + v_right
        if not total > r_left + r_right:
            raise ValueError(
                "both halves of a cluster have zero variance (a long-only combination of the "
                "assets is riskless), so recursive bisection cannot split the weight"
            )
        # α = 1 - V_L/(V_L + V_R) = V_R/(V_L + V_R); the ratio form keeps a
        # tiny share accurate instead of rounding 1 - (1 - ε) to zero.
        weights[left] *= v_right / total
        weights[right] *= v_left / total
        pending.extend((left, right))
    return weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def correlation_distance(corr: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Correlation distance ``d_ij = sqrt((1 - ρ_ij) / 2)`` (López de Prado 2016, Stage 1).

    ``d`` maps ρ = 1 to 0, ρ = 0 to sqrt(½) and ρ = -1 to 1. It is a proper
    metric (Mantegna 1999; López de Prado 2016, Appendix): a PSD correlation
    matrix is a Gram matrix ``ρ_ij = x_i'x_j`` of unit vectors, so
    ``||x_i - x_j||² = 2 - 2ρ_ij`` and ``d_ij = ||x_i - x_j|| / 2`` is a
    Euclidean distance.

    Args:
        corr: correlation matrix (N x N): symmetric, unit diagonal, entries in
            [-1, 1] and positive semi-definite. Rounding errors up to 1e-8 are
            tolerated and clipped.

    Returns:
        Symmetric DataFrame of distances in [0, 1] with a zero diagonal,
        labelled like ``corr``.

    Raises:
        ValueError: if ``corr`` is not a valid correlation matrix.
    """
    frame = _corr_frame(corr)
    d = _distance_values(_correlation_values(frame))
    return pd.DataFrame(d, index=frame.index.copy(), columns=frame.columns.copy())


def hrp_linkage(corr: pd.DataFrame | np.ndarray, method: str = "single") -> np.ndarray:
    """Hierarchical clustering of the assets on the distance of distances (Stage 1).

    With the correlation distance ``d_ij = sqrt((1 - ρ_ij)/2)``, the paper
    clusters on the Euclidean distance between the *columns* of ``d``::

        d̃_ij = sqrt( Σ_k (d_ki - d_kj)² )

    so two assets are close when they are similarly correlated with every
    asset in the universe. ``d̃`` is passed in condensed form (equivalently
    ``squareform(d̃, checks=False)``) to ``scipy.cluster.hierarchy.linkage``.
    The paper uses single linkage; with it the merge heights are the edge
    weights of the minimum spanning tree of ``d̃``.

    Args:
        corr: correlation matrix (N x N, N >= 2), validated as in
            :func:`correlation_distance`.
        method: scipy linkage method: "single" (paper), "complete",
            "average", "weighted", "centroid", "median" or "ward" (the last
            three are valid because ``d̃`` is a Euclidean distance).

    Returns:
        SciPy linkage matrix of shape (N - 1, 4): row ``k = [a, b, height, size]``
        records the merge of clusters ``a`` and ``b`` into cluster ``N + k``;
        ids 0..N-1 are the leaves, i.e. the asset positions in ``corr``.

    Raises:
        ValueError: for an invalid correlation matrix, fewer than two assets,
            or an unknown method.

    References:
        López de Prado (2016), "Building diversified portfolios that
        outperform out of sample", Journal of Portfolio Management 42(4),
        Stage 1 (tree clustering).
    """
    method = _check_method(method)
    frame = _corr_frame(corr)
    if frame.shape[0] < 2:
        raise ValueError("at least two assets are needed to build a hierarchy")
    return _linkage(_correlation_values(frame), method)


def quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Leaf order of a linkage tree: the paper's ``getQuasiDiag`` (Stage 2).

    Starting from the root (last row), each cluster id ``c >= N`` is replaced
    by its two children ``link[c - N, 0]`` then ``link[c - N, 1]`` until only
    original items (``< N``) remain. This is a left-to-right, depth-first
    read of the dendrogram's leaves; reordering the covariance matrix in this
    order puts similar assets next to each other (quasi-diagonalization).

    Args:
        link: SciPy-style linkage matrix of shape (N - 1, 4), N >= 2. Only
            the child ids (columns 0 and 1) and the cluster sizes (column 3)
            are used; the tree structure and sizes are validated.

    Returns:
        A permutation of the positions 0..N-1 as a list of Python ints.

    Raises:
        ValueError: if ``link`` is not a valid linkage matrix (wrong shape,
            non-finite or non-integer ids, a cluster used before it is formed
            or merged twice, inconsistent sizes).

    References:
        López de Prado (2016), Stage 2 (quasi-diagonalization), code snippet
        ``getQuasiDiag``.
    """
    try:
        z = np.asarray(link, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("link must be a numeric linkage matrix") from exc
    if z.ndim != 2 or z.shape[1] != 4 or z.shape[0] < 1:
        raise ValueError(f"link must have shape (N - 1, 4) with N >= 2, got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("link contains NaN or infinite values")
    children = z[:, :2]
    if not (children == np.round(children)).all():
        raise ValueError("cluster ids in the first two columns of link must be integers")
    n_merges = z.shape[0]
    n_items = n_merges + 1
    ids = np.empty((n_merges, 2), dtype=np.int64)
    sizes = np.ones(n_items + n_merges)
    used = np.zeros(n_items + n_merges - 1, dtype=bool)  # every id except the root's
    for row in range(n_merges):
        for col in range(2):
            child = float(children[row, col])
            # Row r creates cluster N + r, so it may only use ids formed earlier.
            # The check runs on the float: casting an id outside the int64
            # range (e.g. 1e20) first would be undefined and warn.
            if not 0 <= child < n_items + row:
                raise ValueError(f"link row {row} uses cluster {child:.0f} before it is formed")
            child_id = int(child)
            if used[child_id]:
                raise ValueError(f"link merges cluster {child_id} more than once")
            used[child_id] = True
            ids[row, col] = child_id
        sizes[n_items + row] = sizes[ids[row, 0]] + sizes[ids[row, 1]]
        if z[row, 3] != sizes[n_items + row]:
            raise ValueError(
                f"link row {row} reports cluster size {z[row, 3]:g}, "
                f"but its children contain {sizes[n_items + row]:g} items"
            )
    # Each of the 2(N-1) non-root ids was used at most once in 2(N-1) child
    # slots, hence exactly once: the rows form one binary tree over all items.
    order: list[int] = []
    stack = [n_items + n_merges - 1]
    while stack:
        node = stack.pop()
        if node < n_items:
            order.append(int(node))
        else:
            left, right = ids[node - n_items]
            stack.append(int(right))  # pushed first so the left child is read first
            stack.append(int(left))
    return order


def recursive_bisection(cov: pd.DataFrame | np.ndarray, order: Sequence[int]) -> pd.Series:
    """HRP weights by top-down recursive bisection: the paper's ``getRecBipart`` (Stage 3).

    Algorithm (López de Prado 2016, Stage 3):

    1. Put all items in one cluster ``L`` in the given ``order``; ``w_n = 1``.
    2. Cut every cluster with more than one item into ``L₁`` (its first
       ``⌊|L|/2⌋`` items) and ``L₂`` (the rest), preserving the order.
    3. Give each half ``j`` the variance of its inverse-variance portfolio::

           Ṽ_j = w̃_j' Σ_j w̃_j,     w̃_j = diag(Σ_j)⁻¹ 1 / tr(diag(Σ_j)⁻¹)

       where ``Σ_j`` is the sub-matrix of ``Σ`` on the half's items.
    4. Split the cluster's weight inversely to the halves' variances::

           α = 1 - Ṽ₁ / (Ṽ₁ + Ṽ₂);   w_n *= α (n in L₁),  w_n *= 1 - α (n in L₂)

    5. Repeat on every half until all clusters are single items.

    Every factor lies in [0, 1] and each split preserves the cluster's total
    weight, so the weights are non-negative and sum to one. Only variances of
    sub-portfolios are needed: ``Σ`` is never inverted and may be singular.
    Covariances *between* the two halves of a split do not enter that split.

    Args:
        cov: covariance matrix (N x N), positive semi-definite with strictly
            positive variances. Any positive scaling (daily or annualized)
            gives the same weights.
        order: the processing order as integer *positions* 0..N-1 into
            ``cov`` (e.g. from :func:`quasi_diagonal_order`), not labels.

    Returns:
        Series of weights indexed like ``cov`` in its original asset order
        (not in ``order``), non-negative and summing to 1.

    Raises:
        ValueError: on a malformed or non-PSD covariance, a non-positive
            variance, an ``order`` that is not a permutation of 0..N-1, or a
            split whose two halves both have zero variance.

    References:
        López de Prado (2016), Stage 3 (recursive bisection), code snippets
        ``getRecBipart``, ``getClusterVar`` and ``getIVP``.
    """
    sigma, _ = _covariance(cov)
    positions = _check_order(order, sigma.shape[0])
    weights = _bisect(sigma.to_numpy(), positions)
    return pd.Series(weights, index=sigma.index.copy())


def hrp_weights(
    cov: pd.DataFrame | np.ndarray,
    corr: pd.DataFrame | np.ndarray | None = None,
    method: str = "single",
) -> pd.Series:
    """Hierarchical Risk Parity portfolio (López de Prado 2016).

    Runs the three stages end to end::

        order = quasi_diagonal_order(hrp_linkage(corr, method))
        w     = recursive_bisection(cov, order)

    i.e. tree clustering on the distance of distances of
    ``d_ij = sqrt((1 - ρ_ij)/2)``, quasi-diagonalization, and recursive
    bisection with inverse-variance cluster portfolios. The result is
    long-only and fully invested; with zero correlations it is exactly the
    inverse-variance portfolio, and with two assets it is ``w_i ∝ 1/σ_i²``
    whatever their correlation.

    Args:
        cov: covariance matrix (N x N), positive semi-definite with strictly
            positive variances; it may be singular.
        corr: correlation matrix used for the clustering. Default: the one
            implied by ``cov``, ``ρ_ij = Σ_ij / (σ_i σ_j)``. A DataFrame is
            aligned to ``cov`` by label; an array must be in ``cov``'s order.
        method: scipy linkage method (see :func:`hrp_linkage`); the paper
            uses "single".

    Returns:
        Series of weights indexed like ``cov`` in its original asset order,
        non-negative and summing to 1. A single asset gets weight 1.

    Raises:
        ValueError: on invalid ``cov``/``corr`` (shape, labels, symmetry,
            positive semi-definiteness, unit diagonal), a non-positive
            variance, or an unknown linkage method.

    References:
        M. López de Prado (2016), "Building diversified portfolios that
        outperform out of sample", Journal of Portfolio Management 42(4),
        59-69.
    """
    method = _check_method(method)
    sigma, implied_corr = _covariance(cov)
    rho = implied_corr if corr is None else _correlation_values(_aligned_corr(corr, sigma))
    if sigma.shape[0] == 1:
        return pd.Series([1.0], index=sigma.index.copy())
    order = quasi_diagonal_order(_linkage(rho, method))
    return pd.Series(_bisect(sigma.to_numpy(), order), index=sigma.index.copy())
