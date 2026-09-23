"""T5.3: rasterisation accounting — invariants I1 (point conservation) and I10 (saturation), README 6.5.4-6.5.6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foveamap.grid.accumulators import INT32_MAX, INT32_MIN, RingAccumulators
from foveamap.grid.backends.numpy_backend import NumpyBackend
from foveamap.grid.engine import locate, rasterize
from foveamap.grid.presets import GridSpec
from foveamap.io.labels import raw_to_super
from foveamap.io.sequence import Sequence

from .conftest import labelled_points

GROUND = (1, 2)
OBSTACLE = (3, 4)


def _metres(xyz_mm: np.ndarray) -> np.ndarray:
    return xyz_mm.astype(np.float64) / 1000.0


def _check_conservation(
    spec: GridSpec,
    xyz_m: np.ndarray,
    cls: np.ndarray,
    moving: np.ndarray,
    conf: np.ndarray,
    min_range_mm: int,
) -> None:
    acc = rasterize(spec, xyz_m, cls, moving, conf, min_range_mm=min_range_mm)
    c = acc.counters
    xyz_m = np.asarray(xyz_m, dtype=np.float64)  # KITTI scans are float32; quantise in float64 (L6)
    assert c.n_raw == len(xyz_m)
    assert c.n_raw == c.n_invalid + c.n_in_grid + c.n_out_of_grid

    # independent recomputation of which points are valid and in grid
    finite = np.isfinite(xyz_m).all(axis=1)
    rng_mm = np.where(finite, np.hypot(xyz_m[:, 0], xyz_m[:, 1]) * 1000.0, -1.0)
    valid = finite & (rng_mm >= min_range_mm)
    assert c.n_invalid == int((~valid).sum())
    q = np.rint(xyz_m[valid] * 1000.0).astype(np.int64)
    ring, _, _ = locate(spec, q[:, 0], q[:, 1])
    in_grid = ring >= 0
    assert c.n_in_grid == int(in_grid.sum())

    cls_in, mov_in, conf_in = cls[valid][in_grid], moving[valid][in_grid], conf[valid][in_grid]
    z_in = np.clip(q[in_grid, 2], -32767, 32767)
    totals = {f: 0 for f in ("n_total", "n_moving", "conf_sum", "g_sum")}
    per_class = np.zeros(5, dtype=np.int64)
    for r in acc.rings:
        totals["n_total"] += int(r.n_total.sum(dtype=np.int64))
        totals["n_moving"] += int(r.n_moving.sum(dtype=np.int64))
        totals["conf_sum"] += int(r.conf_sum.sum(dtype=np.int64))
        totals["g_sum"] += int(r.g_sum.sum())
        per_class += r.n_cls.sum(axis=0, dtype=np.int64)
        np.testing.assert_array_equal(r.n_cls.sum(axis=1, dtype=np.int64), r.n_total)
    assert totals["n_total"] == c.n_in_grid
    np.testing.assert_array_equal(per_class, np.bincount(cls_in, minlength=5))
    assert totals["n_moving"] == int(mov_in.sum())
    assert totals["conf_sum"] == int(conf_in.astype(np.int64).sum())
    assert totals["g_sum"] == int(z_in[np.isin(cls_in, GROUND)].sum())


@pytest.mark.parametrize("name", ["tiny_test", "tiny_ps_test", "fovea_default", "ps_literal", "uniform_40cm"])
def test_conservation_random_points(presets: dict[str, GridSpec], name: str) -> None:
    spec = presets[name]
    rng = np.random.default_rng(42)
    xyz_mm, cls, moving, conf = labelled_points(rng, 30_000, spec.extent_mm, margin_mm=spec.extent_mm // 4)
    xyz = _metres(xyz_mm)
    xyz[::997] = np.nan  # invalid: non-finite
    xyz[5::1009, :2] = 0.3  # invalid: inside the ego-body radius
    _check_conservation(spec, xyz, cls, moving, conf, min_range_mm=1_000)


def test_conservation_on_synthetic_scan(presets: dict[str, GridSpec], synthetic_root: Path) -> None:
    scan = Sequence(synthetic_root, "08")[4]
    assert scan.raw_labels is not None
    cls, moving = raw_to_super(scan.raw_labels)
    conf = np.full(len(cls), 255, dtype=np.uint8)
    for name in ("fovea_default", "ps_literal", "uniform_20cm", "uniform_5cm"):
        _check_conservation(presets[name], scan.xyz, cls, moving, conf, min_range_mm=1_000)


def test_every_point_in_exactly_one_cell(presets: dict[str, GridSpec]) -> None:
    spec = presets["tiny_test"]
    rng = np.random.default_rng(3)
    xyz_mm, cls, moving, conf = labelled_points(rng, 5_000, spec.extent_mm)
    acc = rasterize(spec, _metres(xyz_mm), cls, moving, conf, min_range_mm=0)
    ring, ix, iy = locate(spec, xyz_mm[:, 0], xyz_mm[:, 1])
    for k, r in enumerate(acc.rings):
        sel = ring == k
        flat = iy[sel] * r.side + ix[sel]
        cells, counts = np.unique(flat, return_counts=True)
        np.testing.assert_array_equal(r.cells, cells)
        np.testing.assert_array_equal(r.n_total, counts)


def test_min_max_and_sentinels(presets: dict[str, GridSpec]) -> None:
    spec = presets["tiny_test"]
    # three points in the same 5 cm cell (ring 0): ground z=-1.700/-1.650, obstacle z=0.200
    xyz = np.array([[0.01, 0.01, -1.7], [0.02, 0.03, -1.65], [0.04, 0.04, 0.2], [0.51, 0.51, 1.0]])
    cls = np.array([1, 2, 3, 4], dtype=np.uint8)
    acc = rasterize(spec, xyz, cls, np.array([0, 0, 0, 1], bool), np.array([10, 20, 30, 40], np.uint8), 0)
    r0 = acc.rings[0]
    assert r0.n_cells == 2
    i = 0 if r0.n_total[0] == 3 else 1
    assert (r0.g_min[i], r0.g_max[i], r0.g_sum[i]) == (-1700, -1650, -3350)
    assert (r0.o_min[i], r0.o_max[i]) == (200, 200)
    assert r0.conf_sum[i] == 60 and r0.n_moving[i] == 0
    j = 1 - i  # the lone dynamic point: no ground -> sentinels
    assert (r0.g_min[j], r0.g_max[j], r0.g_sum[j]) == (INT32_MAX, INT32_MIN, 0)
    assert (r0.o_min[j], r0.o_max[j], r0.n_moving[j]) == (1000, 1000, 1)


def test_dtypes(presets: dict[str, GridSpec]) -> None:
    rng = np.random.default_rng(1)
    xyz_mm, cls, moving, conf = labelled_points(rng, 2_000, 2_000)
    r = rasterize(presets["tiny_test"], _metres(xyz_mm), cls, moving, conf, 0).rings[0]
    expected = {
        "cells": np.int64, "n_total": np.uint32, "n_cls": np.uint16, "n_moving": np.uint16,
        "g_sum": np.int64, "g_min": np.int32, "g_max": np.int32, "o_min": np.int32, "o_max": np.int32,
        "conf_sum": np.uint32,
    }  # fmt: skip
    for field, dtype in expected.items():
        assert getattr(r, field).dtype == dtype, field
    assert r.n_cls.shape == (r.n_cells, 5)
    assert np.all(np.diff(r.cells) > 0)  # sorted, unique


def test_z_saturation_counted_not_wrapped(presets: dict[str, GridSpec]) -> None:
    """I10: z outside the int16 mm range is clamped to +-32.767 m and counted."""
    spec = presets["tiny_test"]
    xyz = np.array([[1.0, 1.0, 40.0], [1.01, 1.01, -50.0], [1.02, 1.02, 32.767], [3.0, 3.0, -1.7]])
    cls = np.array([3, 1, 3, 1], dtype=np.uint8)
    acc = rasterize(spec, xyz, cls, np.zeros(4, bool), np.zeros(4, np.uint8), 0)
    assert acc.counters.n_z_saturated == 2
    r0 = acc.rings[0]
    cell = int(np.argmax(r0.n_total))
    assert r0.o_max[cell] == 32767 and r0.g_min[cell] == -32767  # clamped, never wrapped


def test_empty_input(presets: dict[str, GridSpec]) -> None:
    acc = rasterize(presets["fovea_default"], np.zeros((0, 3)), np.zeros(0, np.uint8), np.zeros(0, bool),
                    np.zeros(0, np.uint8), 1_000)  # fmt: skip
    assert acc.counters.n_raw == 0 and all(r.n_cells == 0 for r in acc.rings)


def test_all_invalid(presets: dict[str, GridSpec]) -> None:
    xyz = np.full((10, 3), np.nan)
    acc = rasterize(
        presets["tiny_test"], xyz, np.zeros(10, np.uint8), np.zeros(10, bool), np.zeros(10, np.uint8), 0
    )
    assert acc.counters.n_invalid == 10 and acc.counters.n_in_grid == 0


def test_count_overflow_is_an_error(presets: dict[str, GridSpec]) -> None:
    n = 70_000  # > uint16 max points of one class in one cell
    xyz = np.tile([[1.0, 1.0, -1.7]], (n, 1))
    with pytest.raises(OverflowError, match="65535"):
        rasterize(
            presets["tiny_test"], xyz, np.ones(n, np.uint8), np.zeros(n, bool), np.zeros(n, np.uint8), 0
        )


def test_backend_rejects_mismatched_lengths(presets: dict[str, GridSpec]) -> None:
    with pytest.raises(ValueError, match="length"):
        NumpyBackend().rasterize(presets["tiny_test"], np.zeros((3, 3), np.int32), np.zeros(2, np.uint8),
                                 np.zeros(3, bool), np.zeros(3, np.uint8))  # fmt: skip


def test_to_dense_round_trip(presets: dict[str, GridSpec]) -> None:
    spec = presets["tiny_ps_test"]
    rng = np.random.default_rng(5)
    xyz_mm, cls, moving, conf = labelled_points(rng, 3_000, spec.extent_mm)
    for r in rasterize(spec, _metres(xyz_mm), cls, moving, conf, 0).rings:
        dense = r.to_dense()
        assert dense["n_total"].shape == (r.side * r.side,)
        assert int(dense["n_total"].sum()) == int(r.n_total.sum())
        np.testing.assert_array_equal(np.flatnonzero(dense["n_total"]), r.cells)
        assert (dense["g_min"][dense["n_total"] == 0] == INT32_MAX).all()
        assert RingAccumulators.from_dense(r.ring, r.cell_mm, r.side, dense).equals(r)
