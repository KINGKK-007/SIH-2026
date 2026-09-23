"""Integer-exact per-cell accumulators and fine->coarse reduction (README 6.5.4, tasks T5.3, T6.1).

Accumulators are stored **sparse** (D-019): only occupied cells, as sorted unique flat indices
``cells = iy * N_k + ix`` with one entry per field. ``to_dense`` gives the full-square view of README 9.5.

Two independent exact integer routines produce accumulators: :func:`reduce_points` (points -> cells, used
by the rasteriser; bincount sums, sorted min/max) and :func:`reduce_cells` (cells -> coarser cells, used by
:func:`reduce_block`). Invariant I4 compares the two paths bit for bit; I7 checks both against a naive
dict implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_CLASSES = 5
INT32_MAX = int(np.iinfo(np.int32).max)
INT32_MIN = int(np.iinfo(np.int32).min)
UINT16_MAX = int(np.iinfo(np.uint16).max)
UINT32_MAX = int(np.iinfo(np.uint32).max)
Z_SATURATION_MM = 32767  # int16 range without INT16_MIN, which layers reserve as "no data" (I10)

GROUND_CLASSES = (1, 2)  # DRIVABLE, NON_DRIVABLE_TERRAIN
OBSTACLE_CLASSES = (3, 4)  # STATIC_OBSTACLE, DYNAMIC

SUM_FIELDS = ("n_total", "n_cls", "n_moving", "g_sum", "conf_sum")
MIN_FIELDS = ("g_min", "o_min")
MAX_FIELDS = ("g_max", "o_max")
FIELDS = ("n_total", "n_cls", "n_moving", "g_sum", "g_min", "g_max", "o_min", "o_max", "conf_sum")
DTYPES = {
    "n_total": np.uint32, "n_cls": np.uint16, "n_moving": np.uint16, "g_sum": np.int64,
    "g_min": np.int32, "g_max": np.int32, "o_min": np.int32, "o_max": np.int32, "conf_sum": np.uint32,
}  # fmt: skip
EMPTY = {"g_min": INT32_MAX, "g_max": INT32_MIN, "o_min": INT32_MAX, "o_max": INT32_MIN}
_LIMITS = {"n_total": UINT32_MAX, "n_cls": UINT16_MAX, "n_moving": UINT16_MAX, "conf_sum": UINT32_MAX}


@dataclass
class FrameCounters:
    """Per-frame accounting (README 6.5.6): ``n_raw = n_invalid + n_in_grid + n_out_of_grid``."""

    n_raw: int = 0
    n_invalid: int = 0
    n_in_grid: int = 0
    n_out_of_grid: int = 0
    n_z_saturated: int = 0


@dataclass(eq=False)
class RingAccumulators:
    """Sparse accumulators of one ring (fields per README 6.5.4)."""

    ring: int
    cell_mm: int
    side: int
    cells: np.ndarray  # (m,) int64, sorted unique flat indices iy * side + ix
    n_total: np.ndarray  # (m,) uint32
    n_cls: np.ndarray  # (m, 5) uint16, points per super-class
    n_moving: np.ndarray  # (m,) uint16
    g_sum: np.ndarray  # (m,) int64, sum of ground z (mm)
    g_min: np.ndarray  # (m,) int32, INT32_MAX if no ground
    g_max: np.ndarray  # (m,) int32, INT32_MIN if no ground
    o_min: np.ndarray  # (m,) int32, INT32_MAX if no obstacle
    o_max: np.ndarray  # (m,) int32, INT32_MIN if no obstacle
    conf_sum: np.ndarray  # (m,) uint32

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def ix(self) -> np.ndarray:
        return self.cells % self.side

    @property
    def iy(self) -> np.ndarray:
        return self.cells // self.side

    def fields(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in FIELDS}

    def equals(self, other: RingAccumulators) -> bool:
        """Bit-identical: same ring geometry, same cells, same values and dtypes in every field."""
        if (self.ring, self.cell_mm, self.side) != (other.ring, other.cell_mm, other.side):
            return False
        if not np.array_equal(self.cells, other.cells):
            return False
        return all(
            a.dtype == b.dtype and np.array_equal(a, b)
            for a, b in zip(self.fields().values(), other.fields().values(), strict=True)
        )

    def to_dense(self) -> dict[str, np.ndarray]:
        """Full-square arrays of length ``side**2`` (``n_cls``: ``(side**2, 5)``); empty cells hold sentinels."""
        n = self.side * self.side
        dense = {}
        for name, values in self.fields().items():
            shape = (n, N_CLASSES) if name == "n_cls" else (n,)
            arr = np.full(shape, EMPTY.get(name, 0), dtype=DTYPES[name])
            arr[self.cells] = values
            dense[name] = arr
        return dense

    @classmethod
    def from_dense(cls, ring: int, cell_mm: int, side: int, dense: dict[str, np.ndarray]) -> RingAccumulators:
        cells = np.flatnonzero(dense["n_total"]).astype(np.int64)
        return cls(ring, cell_mm, side, cells, **{name: dense[name][cells] for name in FIELDS})

    @classmethod
    def empty(cls, ring: int, cell_mm: int, side: int) -> RingAccumulators:
        values = {
            name: np.zeros((0, N_CLASSES) if name == "n_cls" else (0,), dtype=DTYPES[name]) for name in FIELDS
        }
        return cls(ring, cell_mm, side, np.zeros(0, dtype=np.int64), **values)


@dataclass
class GridAccumulators:
    rings: list[RingAccumulators]
    counters: FrameCounters = field(default_factory=FrameCounters)


def point_fields(
    z_mm: np.ndarray, super_cls: np.ndarray, moving: np.ndarray, conf: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-point values as int64 one-point "cells", ready for :func:`reduce_cells`."""
    z = np.asarray(z_mm, dtype=np.int64)
    cls = np.asarray(super_cls, dtype=np.intp)
    if cls.size and cls.max(initial=0) >= N_CLASSES:
        raise ValueError(f"super-class IDs must be < {N_CLASSES}")
    ground = (cls == GROUND_CLASSES[0]) | (cls == GROUND_CLASSES[1])
    obstacle = (cls == OBSTACLE_CLASSES[0]) | (cls == OBSTACLE_CLASSES[1])
    one_hot = np.zeros((len(cls), N_CLASSES), dtype=np.int64)
    one_hot[np.arange(len(cls)), cls] = 1
    return {
        "n_total": np.ones(len(z), dtype=np.int64),
        "n_cls": one_hot,
        "n_moving": np.asarray(moving, dtype=bool).astype(np.int64),
        "g_sum": np.where(ground, z, 0),
        "g_min": np.where(ground, z, INT32_MAX),
        "g_max": np.where(ground, z, INT32_MIN),
        "o_min": np.where(obstacle, z, INT32_MAX),
        "o_max": np.where(obstacle, z, INT32_MIN),
        "conf_sum": np.asarray(conf, dtype=np.int64),
    }


def reduce_cells(keys: np.ndarray, values: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Group items by ``keys`` and reduce every field exactly (int64): sum, min or max by field name.

    Returns the sorted unique keys and the reduced fields (still int64). Independent of input order (I6).
    """
    keys = np.asarray(keys, dtype=np.int64)
    if keys.size == 0:
        return keys.copy(), {n: np.zeros((0,) + v.shape[1:], dtype=np.int64) for n, v in values.items()}
    order = np.argsort(keys)  # any order within a group is fine: sums, mins and maxes commute
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    reduced = {}
    for name, v in values.items():
        v = np.asarray(v, dtype=np.int64)[order]
        if name in SUM_FIELDS:
            reduced[name] = np.add.reduceat(v, starts, axis=0)
        elif name in MIN_FIELDS:
            reduced[name] = np.minimum.reduceat(v, starts, axis=0)
        elif name in MAX_FIELDS:
            reduced[name] = np.maximum.reduceat(v, starts, axis=0)
        else:
            raise KeyError(f"unknown accumulator field {name!r}")
    return sorted_keys[starts], reduced


def reduce_points(
    keys: np.ndarray, z_mm: np.ndarray, super_cls: np.ndarray, moving: np.ndarray, conf: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Points -> per-cell int64 fields (the rasteriser's fast path; same result as ``reduce_cells`` on
    :func:`point_fields`). Sums use ``bincount`` with float64 weights, exact because every sum is far below
    2**53 (at most 2**32 points x 32767 mm); min/max use one sort.
    """
    keys = np.asarray(keys, dtype=np.int64)
    if keys.size == 0:
        return reduce_cells(keys, point_fields(z_mm, super_cls, moving, conf))
    cls = np.asarray(super_cls, dtype=np.intp)
    if cls.max(initial=0) >= N_CLASSES:
        raise ValueError(f"super-class IDs must be < {N_CLASSES}")
    z = np.asarray(z_mm, dtype=np.int32)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    new_group = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    starts = np.flatnonzero(new_group)
    m = len(starts)
    group = np.empty(len(keys), dtype=np.intp)
    group[order] = np.cumsum(new_group) - 1

    def exact_sum(weights: np.ndarray) -> np.ndarray:
        return np.rint(np.bincount(group, weights=weights, minlength=m)).astype(np.int64)

    ground = (cls == GROUND_CLASSES[0]) | (cls == GROUND_CLASSES[1])
    obstacle = (cls == OBSTACLE_CLASSES[0]) | (cls == OBSTACLE_CLASSES[1])
    zs, gs, obs = z[order], ground[order], obstacle[order]
    reduced = {
        "n_total": np.diff(np.r_[starts, len(keys)]).astype(np.int64),
        "n_cls": np.bincount(group * N_CLASSES + cls, minlength=m * N_CLASSES)
        .reshape(m, N_CLASSES)
        .astype(np.int64),
        "n_moving": exact_sum(np.asarray(moving, dtype=np.float64)),
        "g_sum": exact_sum(np.where(ground, z, 0).astype(np.float64)),
        "g_min": np.minimum.reduceat(np.where(gs, zs, INT32_MAX), starts).astype(np.int64),
        "g_max": np.maximum.reduceat(np.where(gs, zs, INT32_MIN), starts).astype(np.int64),
        "o_min": np.minimum.reduceat(np.where(obs, zs, INT32_MAX), starts).astype(np.int64),
        "o_max": np.maximum.reduceat(np.where(obs, zs, INT32_MIN), starts).astype(np.int64),
        "conf_sum": exact_sum(np.asarray(conf, dtype=np.float64)),
    }
    return sorted_keys[starts], reduced


def make_ring(
    ring: int, cell_mm: int, side: int, cells: np.ndarray, reduced: dict[str, np.ndarray]
) -> RingAccumulators:
    """Cast reduced int64 fields to their storage dtypes, refusing (never wrapping) overflow."""
    out = {}
    for name in FIELDS:
        values = reduced[name]
        limit = _LIMITS.get(name)
        if limit is not None and values.size and int(values.max()) > limit:
            raise OverflowError(
                f"ring {ring}: {name} value {int(values.max())} exceeds {limit} "
                f"({'uint16' if limit == UINT16_MAX else 'uint32'}); refusing to wrap"
            )
        out[name] = values.astype(DTYPES[name])
    return RingAccumulators(ring, cell_mm, side, np.asarray(cells, dtype=np.int64), **out)


def reduce_block(acc: RingAccumulators, factor: int) -> RingAccumulators:
    """Sum counts/sums and take min/max over each ``factor x factor`` block (fine -> coarse)."""
    if factor < 1 or acc.side % factor:
        raise ValueError(f"factor {factor} must divide the ring side {acc.side}")
    new_side = acc.side // factor
    keys = (acc.iy // factor) * new_side + acc.ix // factor
    cells, reduced = reduce_cells(keys, acc.fields())
    return make_ring(acc.ring, acc.cell_mm * factor, new_side, cells, reduced)
