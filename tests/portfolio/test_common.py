"""Tests for the shared validators in ``quantfin.portfolio._common``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfin.portfolio._common import (
    as_cov_frame,
    as_returns_frame,
    as_series,
    normalize_bounds,
)


def test_as_returns_frame_accepts_arrays_series_and_frames():
    frame = as_returns_frame(np.array([[0.01, 0.02], [0.0, -0.01]]))
    assert list(frame.columns) == [0, 1] and frame.dtypes.eq(float).all()
    labelled = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
    pd.testing.assert_frame_equal(as_returns_frame(labelled), labelled.astype(float))
    assert as_returns_frame(pd.Series([0.1, 0.2], name="x")).columns.tolist() == ["x"]


@pytest.mark.parametrize(
    "bad", [np.empty((0, 2)), np.array([[0.1, np.nan]]), np.array([[np.inf, 0.0]])]
)
def test_as_returns_frame_rejects_empty_or_non_finite(bad):
    with pytest.raises(ValueError):
        as_returns_frame(bad)


def test_as_cov_frame_symmetrizes_rounding_and_labels():
    cov = np.array([[0.04, 0.01 + 1e-13], [0.01, 0.09]])
    frame = as_cov_frame(cov, ["a", "b"])
    assert list(frame.index) == ["a", "b"] == list(frame.columns)
    assert frame.loc["a", "b"] == frame.loc["b", "a"]


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0.04, 0.02], [0.01, 0.09]]),  # not symmetric
        np.ones((2, 3)),  # not square
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        pd.DataFrame(np.eye(2), index=["a", "b"], columns=["b", "a"]),
    ],
)
def test_as_cov_frame_rejects_bad_matrices(bad):
    with pytest.raises(ValueError):
        as_cov_frame(bad)


def test_as_series_aligns_by_label_and_checks_length():
    s = as_series(pd.Series({"b": 2.0, "a": 1.0, "c": 9.0}), ["a", "b"])
    assert s.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="missing"):
        as_series(pd.Series({"a": 1.0}), ["a", "b"])
    with pytest.raises(ValueError, match="expected 2"):
        as_series([1.0], ["a", "b"])


def test_normalize_bounds_expands_and_keeps_none():
    assert normalize_bounds(None, 3) is None
    assert normalize_bounds((0.0, 1.0), 2) == [(0.0, 1.0), (0.0, 1.0)]
    assert normalize_bounds([(None, 1), (0, None)], 2) == [(-np.inf, 1.0), (0.0, np.inf)]


@pytest.mark.parametrize(
    "bounds, match",
    [
        ((np.nan, 1.0), "NaN"),
        ([(0, 1), (0, np.nan), (0, 1)], "NaN"),
        ((0.6, 0.4), "exceeds"),
        ((0.0, 0.3), "infeasible"),
        ((0.5, 1.0), "infeasible"),
        ([(0, 1)], "expected 3"),
    ],
)
def test_normalize_bounds_rejects_invalid(bounds, match):
    with pytest.raises(ValueError, match=match):
        normalize_bounds(bounds, 3)
