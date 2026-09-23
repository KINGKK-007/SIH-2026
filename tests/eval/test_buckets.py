"""T3.5: distance buckets (L16) and the first density helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from foveamap.eval.buckets import (
    OUT_OF_RANGE,
    bucket_counts,
    bucket_labels,
    bucket_of_range,
    horizontal_range,
)
from foveamap.eval.density import points_per_range_bin
from foveamap.io.sequence import Sequence


def test_bucket_boundaries_half_open_last_closed() -> None:
    r = np.array([0.0, 9.999, 10.0, 29.999, 30.0, 59.99, 60.0, 99.99, 100.0, 100.0001, -0.5, np.nan, np.inf])
    np.testing.assert_array_equal(bucket_of_range(r), [0, 0, 1, 1, 2, 2, 3, 3, 3, -1, -1, -1, -1])
    assert bucket_of_range(r).dtype == np.int8


def test_uses_horizontal_range_only() -> None:
    xyz = np.array([[6.0, 8.0, -50.0], [0.0, 0.0, 99.0]])
    np.testing.assert_allclose(horizontal_range(xyz), [10.0, 0.0])
    np.testing.assert_array_equal(bucket_of_range(horizontal_range(xyz)), [1, 0])


def test_custom_edges_and_validation() -> None:
    np.testing.assert_array_equal(bucket_of_range(np.array([1.0, 6.0, 12.0]), (0, 5, 12)), [0, 1, 1])
    with pytest.raises(ValueError):
        bucket_of_range(np.array([1.0]), (0, 10, 10))


def test_labels_and_counts() -> None:
    assert bucket_labels() == ["0-10 m", "10-30 m", "30-60 m", "60-100 m"]
    np.testing.assert_array_equal(bucket_counts(np.array([1, 2, 15, 45, 99, 150.0])), [2, 1, 1, 1])


@settings(max_examples=300, derandomize=True)
@given(st.floats(min_value=0, max_value=100, allow_nan=False))
def test_every_in_range_value_has_exactly_the_right_bucket(r: float) -> None:
    b = int(bucket_of_range(np.array([r]))[0])
    edges = (0, 10, 30, 60, 100)
    assert b != OUT_OF_RANGE
    assert edges[b] <= r and (r < edges[b + 1] or (b == 3 and r == 100))


def test_points_per_range_bin_on_synthetic(synthetic_root: Path) -> None:
    seq = Sequence(synthetic_root, "08", frame_stride=4)
    out = points_per_range_bin((s.xyz for s in seq), bin_m=10.0, max_m=100.0)
    assert out["n_scans"] == [3.0]
    assert len(out["counts"]) == 10 and sum(out["counts"]) > 0
    density = np.array(out["points_per_scan_per_m2"])
    assert density[1] > density[5] > density[9]  # density falls with range


def test_points_per_range_bin_requires_scans() -> None:
    with pytest.raises(ValueError):
        points_per_range_bin([])
