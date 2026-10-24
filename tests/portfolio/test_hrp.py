"""Tests for quantfin.portfolio.hrp (Hierarchical Risk Parity, López de Prado 2016).

The math is checked against independent answers:

* correlation distance: closed-form values, and the identity
  d_ij = ||x_i - x_j|| / 2 for a Gram factorization ρ = XX' (so d is a metric);
* linkage: a hand-computed 3-asset distance-of-distances example, the paper's
  own code path (scipy linkage on the square distance matrix used as an
  observation matrix), and the minimum-spanning-tree characterization of
  single linkage with d̃ computed by explicit loops;
* quasi-diagonalization: a hand-made linkage, the paper's ``getQuasiDiag``
  transcribed literally, a recursive tree walk and ``scipy`` ``leaves_list``;
* recursive bisection: hand-computed 3- and 4-asset examples, the paper's
  ``getRecBipart``/``getClusterVar``/``getIVP`` transcribed literally, and
  the proof that zero correlations give inverse-variance weights;
* end to end: the paper's full pipeline, closed forms (two assets, diagonal
  covariance, a cloned asset), block structure recovery, and invariances
  (scale/annualization/ddof, permutations, numpy vs pandas).
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from scipy.sparse.csgraph import minimum_spanning_tree

import quantfin.portfolio.hrp as hrp_module
from quantfin.portfolio._common import TRADING_DAYS
from quantfin.portfolio.hrp import (
    correlation_distance,
    hrp_linkage,
    hrp_weights,
    quasi_diagonal_order,
    recursive_bisection,
)

METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")

# ---------------------------------------------------------------------------
# Fixtures
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


def corr_of(cov: np.ndarray) -> np.ndarray:
    """Correlation implied by a covariance, with an exact unit diagonal."""
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return 0.5 * (corr + corr.T)


def block_corr(
    sizes: list[int], rng: np.random.Generator, between: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Factor-model correlation with strongly correlated blocks.

    x_i = sqrt(between) m + sqrt(load_i) f_block(i) + idiosyncratic, so
    ρ_ij = between + sqrt(load_i load_j) inside a block and ``between``
    across blocks; PSD by construction. Returns (corr, block id per asset).
    """
    labels = np.repeat(np.arange(len(sizes)), sizes)
    load = rng.uniform(0.6, 0.85, labels.size)
    loadings = np.zeros((labels.size, len(sizes) + 1))
    loadings[:, 0] = np.sqrt(between)
    loadings[np.arange(labels.size), labels + 1] = np.sqrt(load)
    corr = loadings @ loadings.T
    np.fill_diagonal(corr, 1.0)
    return corr, labels


def riskless_quad_corr(seed: int) -> np.ndarray:
    """Gram matrix of four unit vectors in R³ with x0 + x1 + x2 + x3 = 0.

    C1 = 0, so with equal volatilities the equal-weight (= inverse-variance)
    portfolio of the four assets is riskless; computed in floating point its
    variance comes out as ±1e-18 noise.
    """
    rng = np.random.default_rng(seed)
    x0, x1 = (v / np.linalg.norm(v) for v in rng.standard_normal((2, 3)))
    m = x0 + x1
    u = np.cross(m, rng.standard_normal(3))
    u *= np.sqrt(1.0 - m @ m / 4.0) / np.linalg.norm(u)
    x = np.array([x0, x1, -m / 2.0 + u, -m / 2.0 - u])
    corr = x @ x.T
    np.fill_diagonal(corr, 1.0)
    return 0.5 * (corr + corr.T)


# ---------------------------------------------------------------------------
# Independent reference implementations: the paper's code, transcribed
# literally (Python 3: integer division, pd.concat for Series.append, .loc
# for label-based assignment).
# ---------------------------------------------------------------------------


def paper_correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    return ((1 - corr) / 2.0) ** 0.5


def paper_get_ivp(cov: pd.DataFrame) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def paper_get_cluster_var(cov: pd.DataFrame, c_items: list) -> float:
    cov_ = cov.loc[c_items, c_items]
    w_ = paper_get_ivp(cov_).reshape(-1, 1)
    return float(np.dot(np.dot(w_.T, cov_), w_)[0, 0])


def paper_get_quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.to_numpy() - num_items
        sort_ix.loc[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return [int(v) for v in sort_ix.tolist()]


def paper_get_rec_bipart(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = paper_get_cluster_var(cov, c_items0)
            c_var1 = paper_get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w.loc[c_items0] *= alpha
            w.loc[c_items1] *= 1 - alpha
    return w


def paper_linkage(corr: pd.DataFrame, method: str) -> np.ndarray:
    """The paper's ``sch.linkage(dist, 'single')``: the square matrix d is used
    as an observation matrix, so scipy computes the Euclidean distances between
    its rows -- the distance of distances."""
    dist = paper_correl_dist(corr)
    with warnings.catch_warnings():
        # scipy warns that the square matrix looks like a distance matrix;
        # treating it as observations is exactly the paper's intent.
        warnings.simplefilter("ignore", hierarchy.ClusterWarning)
        return hierarchy.linkage(dist.to_numpy(), method)


def paper_hrp(cov: pd.DataFrame, corr: pd.DataFrame, method: str = "single") -> pd.Series:
    link = paper_linkage(corr, method)
    sort_ix = paper_get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()
    return paper_get_rec_bipart(cov, sort_ix)


def recursive_leaves(link: np.ndarray) -> list[int]:
    """Pandas-free recursive reading of the dendrogram's leaves, left child first."""
    n = link.shape[0] + 1

    def leaves(node: int) -> list[int]:
        if node < n:
            return [node]
        left, right = (int(v) for v in link[node - n, :2])
        return leaves(left) + leaves(right)

    return leaves(2 * n - 2)


def labelled(values: np.ndarray, prefix: str = "A") -> pd.DataFrame:
    labels = [f"{prefix}{k:02d}" for k in range(values.shape[0])]
    return pd.DataFrame(values, index=labels, columns=labels)


# Hand-computed 3-asset correlation: d01 = 0.1, d02 = 0.5, d12 = 0.55, i.e.
# ρ = 1 - 2 d² = 0.98, 0.5, 0.395 (PSD: determinant 0.020675 > 0).
CORR3 = np.array([[1.0, 0.98, 0.5], [0.98, 1.0, 0.395], [0.5, 0.395, 1.0]])
# Distance of distances (columns of d = [[0,.1,.5],[.1,0,.55],[.5,.55,0]]):
# d̃01 = sqrt(.1² + .1² + .05²) = sqrt(.0225) = 0.15
# d̃02 = sqrt(.5² + .45² + .5²) = sqrt(.7025)
# d̃12 = sqrt(.4² + .55² + .55²) = sqrt(.765)
D01, D02, D12 = 0.15, np.sqrt(0.7025), np.sqrt(0.765)


# ---------------------------------------------------------------------------
# correlation_distance
# ---------------------------------------------------------------------------


class TestCorrelationDistance:
    @pytest.mark.parametrize(
        ("rho", "expected"),
        [
            (1.0, 0.0),
            (0.5, 0.5),
            (0.0, np.sqrt(0.5)),
            (-0.5, np.sqrt(0.75)),
            (-1.0, 1.0),
        ],
    )
    def test_closed_form_values(self, rho: float, expected: float) -> None:
        # A sign error (1 + ρ) would swap the ρ = ±1 and ±0.5 cases.
        d = correlation_distance(np.array([[1.0, rho], [rho, 1.0]]))
        assert d.iloc[0, 1] == pytest.approx(expected, abs=1e-15)
        assert d.iloc[1, 0] == pytest.approx(expected, abs=1e-15)
        assert d.iloc[0, 0] == 0.0
        assert d.iloc[1, 1] == 0.0

    def test_is_half_the_euclidean_distance_of_gram_vectors(self) -> None:
        # ρ = XX' with unit-norm rows x_i  =>  ||x_i - x_j||² = 2 - 2ρ_ij.
        rng = np.random.default_rng(1)
        x = rng.standard_normal((9, 4))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        corr = x @ x.T
        d = correlation_distance(corr).to_numpy()
        expected = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2) / 2.0
        np.testing.assert_allclose(d, expected, atol=1e-12)
        # Hence a metric: the triangle inequality holds for every triple.
        for i, j, k in itertools.permutations(range(9), 3):
            assert d[i, k] <= d[i, j] + d[j, k] + 1e-12

    def test_labels_range_and_symmetry(self) -> None:
        rng = np.random.default_rng(2)
        corr = labelled(corr_of(random_cov(6, rng)), prefix="X")
        d = correlation_distance(corr)
        assert list(d.index) == list(corr.index)
        assert list(d.columns) == list(corr.columns)
        values = d.to_numpy()
        np.testing.assert_array_equal(values, values.T)
        np.testing.assert_array_equal(np.diag(values), 0.0)
        assert (values >= 0.0).all()
        assert (values <= 1.0).all()
        np.testing.assert_allclose(values, np.sqrt((1.0 - corr.to_numpy()) / 2.0), atol=1e-15)

    def test_rounding_just_above_one_is_clipped(self) -> None:
        # ρ = 1 + 1e-12 (rounding noise of perfectly correlated assets) must
        # give d = 0, not NaN from sqrt of a negative number.
        rho = 1.0 + 1e-12
        d = correlation_distance(np.array([[1.0, rho], [rho, 1.0]]))
        assert d.iloc[0, 1] == 0.0

    @pytest.mark.parametrize(
        ("corr", "match"),
        [
            (np.array([[0.04, 0.01], [0.01, 0.09]]), "unit diagonal"),
            (np.array([[1.0, 1.5], [1.5, 1.0]]), r"\[-1, 1\]"),
            (
                np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]),
                "positive semi-definite",
            ),
            (np.array([[1.0, np.nan], [np.nan, 1.0]]), "NaN"),
            (np.array([[1.0, 0.2], [0.3, 1.0]]), "symmetric"),
            (np.ones((2, 3)), "square"),
        ],
    )
    def test_validation(self, corr: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            correlation_distance(corr)


# The cov passed next to a bad corr is valid, so an error that blames the
# "covariance" would point the user at the wrong argument.
_GOOD_COV = labelled(np.diag([0.04, 0.09]))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: correlation_distance(np.array([[1.0, 0.2], [0.3, 1.0]])), "symmetric"),
        (lambda: correlation_distance(np.array([[1.0, np.nan], [np.nan, 1.0]])), "NaN"),
        (lambda: correlation_distance(np.ones((2, 3))), "square"),
        (
            lambda: correlation_distance(
                pd.DataFrame(np.eye(2), index=["a", "b"], columns=["b", "a"])
            ),
            "identical index and columns",
        ),
        (lambda: hrp_linkage(np.array([[1.0, 0.2], [0.3, 1.0]])), "symmetric"),
        (lambda: hrp_linkage(np.array([[1.0, np.inf], [np.inf, 1.0]])), "infinite"),
        (
            lambda: hrp_weights(
                _GOOD_COV,
                corr=pd.DataFrame(np.eye(2), index=["A00", "A01"], columns=["A01", "A00"]),
            ),
            "identical index and columns",
        ),
        (
            lambda: hrp_weights(_GOOD_COV, corr=np.array([[1.0, 0.2], [0.3, 1.0]])),
            "symmetric",
        ),
        (
            lambda: hrp_weights(_GOOD_COV, corr=np.array([[1.0, np.nan], [np.nan, 1.0]])),
            "NaN",
        ),
    ],
)
def test_correlation_errors_name_the_correlation(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match) as info:
        call()
    message = str(info.value)
    assert "corr" in message
    assert "covariance" not in message


# ---------------------------------------------------------------------------
# hrp_linkage
# ---------------------------------------------------------------------------


class TestHrpLinkage:
    @pytest.mark.parametrize(
        ("method", "root_height"),
        [
            ("single", min(D02, D12)),
            ("complete", max(D02, D12)),
            ("average", 0.5 * (D02 + D12)),
        ],
    )
    def test_hand_computed_distance_of_distances(self, method: str, root_height: float) -> None:
        # Without the distance-of-distances step the first merge would happen
        # at d01 = 0.1 instead of d̃01 = 0.15.
        link = hrp_linkage(CORR3, method=method)
        expected = np.array([[0.0, 1.0, D01, 2.0], [2.0, 3.0, root_height, 3.0]])
        np.testing.assert_allclose(link, expected, rtol=1e-14, atol=1e-15)

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n", [2, 3, 7, 16])
    def test_matches_paper_observation_matrix_route(self, method: str, n: int) -> None:
        rng = np.random.default_rng(100 + n)
        corr = labelled(corr_of(random_cov(n, rng)))
        np.testing.assert_allclose(
            hrp_linkage(corr, method=method), paper_linkage(corr, method), rtol=1e-12, atol=1e-14
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_single_linkage_heights_are_mst_of_explicit_d_tilde(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = 10
        corr = corr_of(random_cov(n, rng))
        d = np.sqrt((1.0 - corr) / 2.0)
        d_tilde = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d_tilde[i, j] = np.sqrt(sum((d[k, i] - d[k, j]) ** 2 for k in range(n)))
        mst_weights = np.sort(minimum_spanning_tree(d_tilde).data)
        link = hrp_linkage(corr)
        np.testing.assert_allclose(link[:, 2], mst_weights, rtol=1e-12)
        # The check discriminates: clustering on d itself gives other heights.
        assert not np.allclose(np.sort(minimum_spanning_tree(d).data), mst_weights)

    def test_structure(self) -> None:
        rng = np.random.default_rng(3)
        n = 12
        link = hrp_linkage(corr_of(random_cov(n, rng)))
        assert link.shape == (n - 1, 4)
        assert link[-1, 3] == n
        assert (np.diff(link[:, 2]) >= 0).all()  # single linkage is monotone
        assert hierarchy.is_valid_linkage(link)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="method"):
            hrp_linkage(CORR3, method="nearest")
        with pytest.raises(ValueError, match="method"):
            hrp_linkage(CORR3, method=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least two"):
            hrp_linkage(np.array([[1.0]]))
        with pytest.raises(ValueError, match="unit diagonal"):
            hrp_linkage(CORR3 * 0.04)


# ---------------------------------------------------------------------------
# quasi_diagonal_order
# ---------------------------------------------------------------------------


class TestQuasiDiagonalOrder:
    def test_hand_made_linkage(self) -> None:
        # N = 4: cluster 4 = (3, 1), cluster 5 = (0, 2), root 6 = (4, 5).
        # Reading left child before right child gives [3, 1, 0, 2]; swapping
        # the columns would give [2, 0, 1, 3].
        link = np.array([[3, 1, 0.1, 2], [0, 2, 0.2, 2], [4, 5, 0.3, 4]], dtype=float)
        assert quasi_diagonal_order(link) == [3, 1, 0, 2]
        assert quasi_diagonal_order(np.array([[0, 1, 0.5, 2]], dtype=float)) == [0, 1]
        assert quasi_diagonal_order(np.array([[1, 0, 0.5, 2]], dtype=float)) == [1, 0]

    def test_three_asset_example(self) -> None:
        assert quasi_diagonal_order(hrp_linkage(CORR3)) == [2, 0, 1]

    @pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
    @pytest.mark.parametrize("n", [2, 3, 4, 9, 25, 40])
    def test_matches_paper_get_quasi_diag(self, method: str, n: int) -> None:
        rng = np.random.default_rng(7 * n)
        link = hierarchy.linkage(rng.standard_normal((n, 3)), method=method)
        order = quasi_diagonal_order(link)
        assert order == paper_get_quasi_diag(link)
        assert order == recursive_leaves(link)
        assert order == hierarchy.leaves_list(link).tolist()
        assert sorted(order) == list(range(n))
        assert all(type(i) is int for i in order)

    def test_long_chain_needs_no_recursion(self) -> None:
        # Single linkage on points on a line merges one leaf at a time: a
        # depth-(N-1) tree that would overflow a naive recursive walk.
        n = 3000
        link = hierarchy.linkage(np.cumsum(np.arange(1.0, n + 1)).reshape(-1, 1), "single")
        assert quasi_diagonal_order(link) == hierarchy.leaves_list(link).tolist()

    @pytest.mark.parametrize(
        ("link", "match"),
        [
            (np.zeros((0, 4)), "shape"),
            (np.zeros((2, 3)), "shape"),
            (np.zeros(4), "shape"),
            (np.array([[0, 1, np.nan, 2]], dtype=float), "NaN"),
            (np.array([[0, 1.5, 0.1, 2]]), "integers"),
            (np.array([[0, 3, 0.1, 2], [1, 2, 0.2, 3]], dtype=float), "before it is formed"),
            (np.array([[0, 1, 0.1, 2], [3, 3, 0.2, 4]], dtype=float), "more than once"),
            (np.array([[0, 1, 0.1, 2], [0, 3, 0.2, 3]], dtype=float), "more than once"),
            (np.array([[0, -1, 0.1, 2]], dtype=float), "before it is formed"),
            (np.array([[0, 1, 0.1, 2], [2, 3, 0.2, 4]], dtype=float), "size"),
            (np.array([["a", "b", "c", "d"]]), "numeric"),
        ],
    )
    def test_validation(self, link: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            quasi_diagonal_order(link)

    @pytest.mark.parametrize("bad_id", [1e20, -1e20, 2.0**63, 1e300])
    def test_ids_beyond_int64_are_rejected_before_casting(self, bad_id: float) -> None:
        # Casting such an id to int64 is undefined (numpy warns "invalid value
        # encountered in cast" and yields -2**63); it must be range-checked as
        # a float and reported as given.
        link = np.array([[0.0, bad_id, 0.1, 2.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ValueError, match="before it is formed") as info:
                quasi_diagonal_order(link)
        assert f"cluster {bad_id:.0f} " in str(info.value)


# ---------------------------------------------------------------------------
# recursive_bisection
# ---------------------------------------------------------------------------

# 4 assets in order [0, 1, 2, 3]. Halves {0, 1} and {2, 3}:
#   {0,1}: σ² = (1, 4), cov 0.5 -> IVP (4/5, 1/5), V = 16/25 + 4/25 + 2·(4/25)·0.5 = 24/25
#   {2,3}: σ² = (1, 1), cov 0.5 -> IVP (1/2, 1/2), V = 1/4 + 1/4 + 2·(1/4)·0.5 = 3/4
#   α = 1 - (24/25)/(24/25 + 3/4) = 25/57 to {0, 1}, 32/57 to {2, 3}
#   {0} vs {1}: α = 1 - 1/5 = 4/5   -> w0 = 20/57, w1 = 5/57
#   {2} vs {3}: α = 1/2              -> w2 = w3 = 16/57
# Covariances across the two halves (x) never enter a split.
COV4_WITHIN = np.array(
    [[1.0, 0.5, 0.0, 0.0], [0.5, 4.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.5, 1.0]]
)
W4 = np.array([20.0, 5.0, 16.0, 16.0]) / 57.0

# 3 assets in order [0, 1, 2]: the paper splits at floor(3/2) = 1 -> {0} | {1, 2}.
#   {1,2}: σ² = (1, 4), cov 0.5 -> V = 24/25; {0}: V = 1
#   α = 1 - 1/(1 + 24/25) = 24/49 -> w0 = 24/49; {1} vs {2}: 4/5, 1/5 of 25/49
COV3 = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.5], [-0.2, 0.5, 4.0]])
W3 = np.array([24.0, 20.0, 5.0]) / 49.0


class TestRecursiveBisection:
    @pytest.mark.parametrize(
        "cross", [(0.0, 0.0, 0.0, 0.0), (0.1, -0.1, 0.2, 0.0), (0.3, 0.2, -0.4, 0.25)]
    )
    def test_hand_computed_four_assets(self, cross: tuple[float, float, float, float]) -> None:
        cov = COV4_WITHIN.copy()
        cov[0, 2], cov[0, 3], cov[1, 2], cov[1, 3] = cross
        cov = np.triu(cov) + np.triu(cov, 1).T
        assert np.linalg.eigvalsh(cov).min() > 0.0
        w = recursive_bisection(cov, [0, 1, 2, 3])
        np.testing.assert_allclose(w.to_numpy(), W4, rtol=1e-14)

    def test_hand_computed_three_assets_floor_split(self) -> None:
        # A ceil split ({0, 1} | {2}) or a reversed α would give other numbers;
        # the lower-variance half must receive the larger share.
        w = recursive_bisection(COV3, [0, 1, 2])
        np.testing.assert_allclose(w.to_numpy(), W3, rtol=1e-14)

    def test_order_is_positional_and_output_in_asset_order(self) -> None:
        # Same economics, assets relabelled and listed in a different order:
        # positions [2, 0, 1] of the permuted matrix are the original [0, 1, 2].
        perm = [1, 2, 0]  # new position k holds original asset perm[k]
        cov = COV3[np.ix_(perm, perm)]
        labels = ["B", "C", "A"]
        w = recursive_bisection(pd.DataFrame(cov, index=labels, columns=labels), [2, 0, 1])
        assert list(w.index) == labels
        np.testing.assert_allclose(w[["A", "B", "C"]].to_numpy(), W3, rtol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 11, 17])
    def test_matches_paper_get_rec_bipart(self, n: int) -> None:
        rng = np.random.default_rng(1000 + n)
        cov = labelled(random_cov(n, rng))
        for _ in range(5):
            order = rng.permutation(n).tolist()
            ours = recursive_bisection(cov, order)
            ref = paper_get_rec_bipart(cov, cov.index[order].tolist()).reindex(cov.index)
            np.testing.assert_allclose(ours.to_numpy(), ref.to_numpy(), rtol=1e-12, atol=1e-15)

    def test_every_order_of_four_assets_matches_paper(self) -> None:
        rng = np.random.default_rng(11)
        cov = labelled(random_cov(4, rng))
        results = set()
        for order in itertools.permutations(range(4)):
            ours = recursive_bisection(cov, order)
            ref = paper_get_rec_bipart(cov, cov.index[list(order)].tolist()).reindex(cov.index)
            np.testing.assert_allclose(ours.to_numpy(), ref.to_numpy(), rtol=1e-12)
            results.add(tuple(np.round(ours.to_numpy(), 12)))
        # Only the partition into halves matters, not the order inside a half.
        assert len(results) == 3

    @pytest.mark.parametrize("seed", range(4))
    def test_diagonal_covariance_gives_inverse_variance_for_any_order(self, seed: int) -> None:
        # Proof in the module docstring: V_C = 1/S_C, so α = S_L / (S_L + S_R).
        rng = np.random.default_rng(seed)
        n = 9
        variances = rng.uniform(0.01, 0.5, n)
        ivp = (1.0 / variances) / (1.0 / variances).sum()
        w = recursive_bisection(np.diag(variances), rng.permutation(n))
        np.testing.assert_allclose(w.to_numpy(), ivp, rtol=1e-13)

    def test_permutation_equivariance(self) -> None:
        rng = np.random.default_rng(5)
        n = 7
        cov = random_cov(n, rng)
        order = rng.permutation(n)
        w = recursive_bisection(cov, order).to_numpy()
        perm = rng.permutation(n)  # new position k holds old asset perm[k]
        inverse = np.argsort(perm)  # old asset i sits at new position inverse[i]
        w_perm = recursive_bisection(cov[np.ix_(perm, perm)], inverse[order]).to_numpy()
        np.testing.assert_allclose(w_perm, w[perm], rtol=1e-13)

    def test_singular_covariance(self) -> None:
        # N > T: the sample covariance is singular; HRP never inverts it.
        rng = np.random.default_rng(6)
        returns = rng.standard_normal((10, 30)) * 0.01
        cov = np.cov(returns, rowvar=False)
        assert np.linalg.matrix_rank(cov) < 30
        w = recursive_bisection(cov, range(30))
        assert (w > 0.0).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-14)

    def test_zero_variance_half_gets_no_weight(self) -> None:
        # Assets 0 and 1 have equal volatility and ρ = -1: their equal-weight
        # (= inverse-variance) portfolio is riskless, so it takes all weight.
        # (x1 = -x0, so cov(x1, x2) = -cov(x0, x2): the matrix is PSD and singular.)
        cov = np.array([[0.04, -0.04, 0.01], [-0.04, 0.04, -0.01], [0.01, -0.01, 0.09]])
        assert np.linalg.eigvalsh(cov).min() > -1e-15
        w = recursive_bisection(cov, [2, 0, 1])
        np.testing.assert_allclose(w.to_numpy(), [0.5, 0.5, 0.0], atol=1e-15)
        assert (w >= 0.0).all()

    def test_rounding_negative_cluster_variance_is_clipped(self) -> None:
        # Half {0..3} is riskless; its computed variance is often a tiny
        # negative number. Unclipped, α would leave [0, 1] and the other half
        # would get tiny negative weights.
        rng = np.random.default_rng(14)
        other = random_cov(4, rng)
        negative = 0
        for seed in range(20):
            quad = riskless_quad_corr(seed) * 0.04
            cov = np.zeros((8, 8))
            cov[:4, :4] = quad
            cov[4:, 4:] = other
            ew = np.full(4, 0.25)
            negative += float(ew @ quad @ ew) < 0.0
            w = recursive_bisection(cov, range(8)).to_numpy()
            assert (w >= 0.0).all()
            np.testing.assert_allclose(w[4:], 0.0, atol=1e-15)
            # The riskless half takes all the weight and splits it as it would alone.
            np.testing.assert_allclose(w[:4], recursive_bisection(quad, range(4)), rtol=1e-12)
        assert negative > 0  # the fixture really exercises a negative rounded variance

    def test_both_halves_zero_variance_raises(self) -> None:
        pair = np.array([[0.04, -0.04], [-0.04, 0.04]])
        cov = np.block([[pair, np.zeros((2, 2))], [np.zeros((2, 2)), pair]])
        with pytest.raises(ValueError, match="zero variance"):
            recursive_bisection(cov, [0, 1, 2, 3])
        # Splitting the riskless pairs apart is fine.
        w = recursive_bisection(cov, [0, 2, 1, 3])
        np.testing.assert_allclose(w.to_numpy(), 0.25, rtol=1e-14)

    @pytest.mark.parametrize(
        "order",
        [
            [0, 1],
            [0, 1, 1],
            [0, 1, 3],
            [0, 1, 2, 3],
            [0.0, 1.0, 2.0],
            [True, False, 2],
            ["0", "1", "2"],
            None,
            5,
        ],
    )
    def test_order_validation(self, order: object) -> None:
        with pytest.raises(ValueError, match="order"):
            recursive_bisection(COV3, order)  # type: ignore[arg-type]

    def test_accepts_numpy_integer_orders(self) -> None:
        w = recursive_bisection(COV3, np.array([0, 1, 2], dtype=np.int32))
        np.testing.assert_allclose(w.to_numpy(), W3, rtol=1e-14)

    @pytest.mark.parametrize(
        ("cov", "match"),
        [
            (np.array([[0.04, 0.0], [0.0, 0.0]]), "positive variance"),
            (np.array([[0.04, 0.0], [0.0, -0.01]]), "positive variance"),
            (np.array([[0.04, 0.05], [0.05, 0.04]]), "positive semi-definite"),
            (np.array([[0.04, np.inf], [np.inf, 0.04]]), "NaN or infinite"),
            (np.array([[0.04, 0.01], [0.02, 0.04]]), "symmetric"),
        ],
    )
    def test_covariance_validation(self, cov: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            recursive_bisection(cov, [0, 1])

    @pytest.mark.parametrize("func", [hrp_weights, lambda c: recursive_bisection(c, [0, 1])])
    def test_overflowing_correlation_is_reported_as_not_psd(
        self, func: Callable[[np.ndarray], pd.Series]
    ) -> None:
        # |Σ_01| / (σ_0 σ_1) = 1e600 overflows. Such a matrix is grossly not PSD
        # and must be rejected with the module's own message, with no numpy
        # overflow warning first.
        cov = np.array([[1e-300, 1e300], [1e300, 1e-300]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ValueError, match="covariance matrix is not positive semi-definite"):
                func(cov)

    def test_entries_too_large_to_symmetrize_are_not_misreported(self) -> None:
        # diag(1.7e308, 1.7e308) is PSD, but _common.as_cov_frame's 0.5*(Σ + Σ')
        # overflows to inf (and warns -- outside this module). It must not be
        # reported as a PSD failure or a riskless split.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(ValueError, match="too large"):
                hrp_weights(np.diag([1.7e308, 1.7e308]))
        # Just inside the range the weights are still exact.
        w = hrp_weights(np.diag([8e307, 2e307]))
        np.testing.assert_allclose(w.to_numpy(), [0.2, 0.8], rtol=1e-14)


# ---------------------------------------------------------------------------
# hrp_weights
# ---------------------------------------------------------------------------


class TestHrpWeights:
    @pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
    @pytest.mark.parametrize("n", [2, 3, 5, 8, 12, 21])
    def test_matches_paper_pipeline(self, method: str, n: int) -> None:
        rng = np.random.default_rng(n)
        cov = labelled(random_cov(n, rng))
        corr = labelled(corr_of(cov.to_numpy()))
        ref = paper_hrp(cov, corr, method).reindex(cov.index)
        w = hrp_weights(cov, method=method)
        np.testing.assert_allclose(w.to_numpy(), ref.to_numpy(), rtol=1e-12, atol=1e-15)
        assert list(w.index) == list(cov.index)

    @pytest.mark.parametrize("seed", range(5))
    def test_long_only_fully_invested(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        w = hrp_weights(random_cov(15, rng))
        assert (w > 0.0).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("method", METHODS)
    def test_diagonal_covariance_is_inverse_variance(self, method: str) -> None:
        # Zero correlations: every d̃_ij is equal (ties), whatever the order
        # chosen, HRP reduces to the inverse-variance portfolio.
        variances = np.array([0.04, 0.01, 0.09, 0.0225, 0.0625, 0.0016, 0.0324])
        w = hrp_weights(np.diag(variances), method=method)
        ivp = (1.0 / variances) / (1.0 / variances).sum()
        np.testing.assert_allclose(w.to_numpy(), ivp, rtol=1e-13)

    @pytest.mark.parametrize("rho", [-1.0, -0.9, -0.3, 0.0, 0.4, 0.95, 1.0])
    def test_two_assets_inverse_variance_for_any_correlation(self, rho: float) -> None:
        s1, s2 = 0.15, 0.35
        cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
        w = hrp_weights(cov)
        expected = np.array([s2**2, s1**2]) / (s1**2 + s2**2)
        np.testing.assert_allclose(w.to_numpy(), expected, rtol=1e-14)

    def test_hand_computed_three_assets(self) -> None:
        # CORR3 clusters {0, 1} first; the order is [2, 0, 1], so the root
        # split is {2} | {0, 1}. With vols σ = (0.1, 0.2, 0.3):
        #   {0,1}: IVP (4/5, 1/5), V = (16·0.01 + 0.04 + 8·0.98·0.02)/25
        #   w2 = V / (V + 0.09), then {0} vs {1} at 4/5 vs 1/5.
        vols = np.array([0.1, 0.2, 0.3])
        cov = CORR3 * np.outer(vols, vols)
        v01 = (16 * 0.01 + 0.04 + 8 * 0.98 * 0.02) / 25
        w2 = v01 / (v01 + 0.09)
        expected = [(1 - w2) * 0.8, (1 - w2) * 0.2, w2]
        np.testing.assert_allclose(hrp_weights(cov).to_numpy(), expected, rtol=1e-13)

    def test_weights_jump_when_the_cluster_tree_changes(self) -> None:
        # HRP weights are only piecewise smooth: with all three correlations
        # tied at 0.5, a 1e-9 nudge decides which pair merges first, and so
        # which positional halves the root split forms.
        vols = np.array([0.1, 0.2, 0.4])

        def cov_with(r01: float, r02: float) -> np.ndarray:
            corr = np.array([[1.0, r01, r02], [r01, 1.0, 0.5], [r02, 0.5, 1.0]])
            return corr * np.outer(vols, vols)

        # {0, 1} merge first -> order [2, 0, 1], root split {2} | {0, 1}:
        #   IVP (4/5, 1/5), V01 = .64·.01 + .04·.04 + 2·.8·.2·(.5·.1·.2) = .0064 + .0016 + .0032
        v01 = 0.0112
        w2 = v01 / (v01 + 0.16)
        tree_a = np.array([0.8 * (1 - w2), 0.2 * (1 - w2), w2])
        # {0, 2} merge first -> order [1, 0, 2], root split {1} | {0, 2}:
        #   IVP (16/17, 1/17), V02 = (256·.01 + .16 + 2·16·(.5·.1·.4)) / 289 = 3.36 / 289
        v02 = 3.36 / 289
        w1 = v02 / (v02 + 0.04)
        tree_b = np.array([16 / 17 * (1 - w1), w1, 1 / 17 * (1 - w1)])

        a = hrp_weights(cov_with(0.5 + 1e-9, 0.5)).to_numpy()
        a2 = hrp_weights(cov_with(0.5 + 2e-9, 0.5)).to_numpy()
        b = hrp_weights(cov_with(0.5, 0.5 + 1e-9)).to_numpy()
        np.testing.assert_allclose(a, tree_a, rtol=1e-7)
        np.testing.assert_allclose(b, tree_b, rtol=1e-7)
        # Same tree: the weights move by O(1e-9) ...
        assert np.abs(a2 - a).max() < 1e-8
        # ... different tree: they jump by several percentage points.
        assert np.abs(a - b).max() > 0.03
        # The module documentation must not promise smooth weights (the
        # comparison write-up draws on it) and must describe the jump.
        doc = " ".join((hrp_module.__doc__ or "").split())
        assert "move smoothly" not in doc
        assert "jump" in doc

    def test_cloned_asset_shares_its_weight(self) -> None:
        # A and its perfect clone A' sit at distance 0 and merge first; the
        # root split is then B | {A, A'} with V_{AA'} = σ_A², so B keeps its
        # two-asset inverse-variance weight and the clones split the rest.
        sa, sb, rho = 0.2, 0.1, 0.3
        cov = np.array(
            [
                [sa**2, sa**2, rho * sa * sb],
                [sa**2, sa**2, rho * sa * sb],
                [rho * sa * sb, rho * sa * sb, sb**2],
            ]
        )
        w = hrp_weights(cov)
        w_b = sa**2 / (sa**2 + sb**2)
        np.testing.assert_allclose(w.to_numpy(), [(1 - w_b) / 2, (1 - w_b) / 2, w_b], rtol=1e-14)

    @pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
    def test_block_structure_is_recovered(self, method: str) -> None:
        rng = np.random.default_rng(42)
        corr, blocks = block_corr([3, 4, 5, 2], rng)
        shuffle = rng.permutation(blocks.size)
        corr, blocks = corr[np.ix_(shuffle, shuffle)], blocks[shuffle]
        vols = rng.uniform(0.05, 0.4, blocks.size)
        cov = corr * np.outer(vols, vols)
        order = quasi_diagonal_order(hrp_linkage(corr, method=method))
        ordered_blocks = blocks[order]
        # Each block occupies one contiguous run of the quasi-diagonal order.
        runs = [b for k, b in enumerate(ordered_blocks) if k == 0 or b != ordered_blocks[k - 1]]
        assert sorted(runs) == sorted(set(blocks.tolist()))
        # The reordered correlation matrix is block diagonal (quasi-diagonal).
        reordered = corr[np.ix_(order, order)]
        same = ordered_blocks[:, None] == ordered_blocks[None, :]
        assert reordered[same].min() > 0.6 > 0.1 > reordered[~same].max()
        w = hrp_weights(cov, method=method)
        np.testing.assert_allclose(w.to_numpy(), recursive_bisection(cov, order), rtol=1e-14)

    def test_explicit_corr_drives_the_clustering(self) -> None:
        rng = np.random.default_rng(8)
        n = 9
        cov = labelled(random_cov(n, rng))
        other = labelled(corr_of(random_cov(n, rng)))
        expected = recursive_bisection(cov, quasi_diagonal_order(hrp_linkage(other)))
        pd.testing.assert_series_equal(hrp_weights(cov, corr=other), expected)
        # A labelled corr is aligned by label; an array is taken in cov's order.
        shuffled = other.iloc[::-1, ::-1]
        pd.testing.assert_series_equal(hrp_weights(cov, corr=shuffled), expected)
        pd.testing.assert_series_equal(hrp_weights(cov, corr=other.to_numpy()), expected)
        # The default corr is the one implied by cov.
        implied = labelled(corr_of(cov.to_numpy()))
        np.testing.assert_allclose(hrp_weights(cov), hrp_weights(cov, corr=implied), rtol=1e-14)
        assert not np.allclose(hrp_weights(cov), expected)

    def test_scale_annualization_and_ddof_invariance(self) -> None:
        # HRP depends on Σ only through correlations and variance ratios, so
        # daily vs annualized (x252) and ddof 0 vs 1 give identical weights.
        rng = np.random.default_rng(9)
        returns = rng.standard_normal((300, 8)) @ rng.standard_normal((8, 8)) * 0.01
        daily = np.cov(returns, rowvar=False, ddof=1)
        base = hrp_weights(daily).to_numpy()
        for cov in (
            daily * TRADING_DAYS,
            np.cov(returns, rowvar=False, ddof=0),
            daily * 1e-150,
            daily * 1e150,
        ):
            np.testing.assert_allclose(hrp_weights(cov).to_numpy(), base, rtol=1e-12)

    def test_unit_changes_move_only_the_allocation(self) -> None:
        # Rescaling asset 0 (e.g. quoting it in another currency unit) leaves
        # correlations, hence the order, unchanged, but its variance -- and so
        # its inverse-variance share -- changes.
        rng = np.random.default_rng(10)
        cov = random_cov(6, rng)
        scale = np.ones(6)
        scale[0] = 3.0
        cov2 = cov * np.outer(scale, scale)
        link, link2 = hrp_linkage(corr_of(cov)), hrp_linkage(corr_of(cov2))
        np.testing.assert_allclose(link2, link, rtol=1e-12)
        assert quasi_diagonal_order(link2) == quasi_diagonal_order(link)
        assert hrp_weights(cov2)[0] < hrp_weights(cov)[0]

    def test_numpy_and_pandas_agree_and_deterministic(self) -> None:
        rng = np.random.default_rng(12)
        values = random_cov(10, rng)
        frame = labelled(values, prefix="Z")
        w_np = hrp_weights(values)
        w_pd = hrp_weights(frame)
        assert list(w_np.index) == list(range(10))
        assert list(w_pd.index) == list(frame.index)
        np.testing.assert_array_equal(w_np.to_numpy(), w_pd.to_numpy())
        np.testing.assert_array_equal(hrp_weights(values).to_numpy(), w_np.to_numpy())

    def test_single_asset(self) -> None:
        w = hrp_weights(pd.DataFrame([[0.04]], index=["X"], columns=["X"]))
        pd.testing.assert_series_equal(w, pd.Series([1.0], index=["X"]))
        with pytest.raises(ValueError, match="positive variance"):
            hrp_weights(np.array([[0.0]]))

    def test_singular_covariance_and_perfect_correlation(self) -> None:
        # More assets than observations, plus an exact duplicate column:
        # ρ = 1 up to rounding must not produce NaNs or RuntimeWarnings.
        rng = np.random.default_rng(13)
        returns = rng.standard_normal((12, 25)) * 0.01
        returns[:, 1] = returns[:, 0]
        cov = np.cov(returns, rowvar=False)
        w = hrp_weights(cov)
        assert np.isfinite(w).all()
        assert (w > 0.0).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-14)
        # The clones merge first at distance 0, so they are neighbours in the
        # quasi-diagonal order (a positional cut may still separate them).
        order = quasi_diagonal_order(hrp_linkage(corr_of(cov)))
        assert abs(order.index(0) - order.index(1)) == 1

    def test_validation(self) -> None:
        cov = labelled(COV3)
        with pytest.raises(ValueError, match="method"):
            hrp_weights(cov, method="median-ish")
        with pytest.raises(ValueError, match="same assets"):
            hrp_weights(cov, corr=labelled(CORR3, prefix="B"))
        with pytest.raises(ValueError, match="3x3"):
            hrp_weights(cov, corr=np.eye(2))
        with pytest.raises(ValueError, match="unit diagonal"):
            hrp_weights(cov, corr=COV3)
        with pytest.raises(ValueError, match="positive semi-definite"):
            hrp_weights(cov, corr=np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]]))
        with pytest.raises(ValueError, match="positive semi-definite"):
            hrp_weights(np.array([[0.04, 0.05], [0.05, 0.04]]))
        with pytest.raises(ValueError, match="identical index and columns"):
            hrp_weights(pd.DataFrame(COV3, index=["a", "b", "c"], columns=["a", "b", "d"]))
        with pytest.raises(ValueError, match="square"):
            hrp_weights(np.ones((2, 3)))
