"""Packed 12-byte per-cell layers derived from accumulators (README 6.6, task T6.2).

Layout (L10): ground_z int16 mm, top_z int16 mm, clearance int16 mm, cls uint8, moving_frac uint8,
count uint16, conf uint8, flags uint8 = 12 bytes. ``INT16_MIN`` means "no data". Every rounding is
integer half-up: ``(a + n // 2) // n`` (README 6.6 for ground_z; used identically for moving_frac/conf).

Class rule (applied identically at every resolution, baselines included):

* ``safety`` (default): DYNAMIC if ``n_dyn >= max(dyn_min_points, dyn_min_frac * n)``, else STATIC_OBSTACLE if
  ``n_static >= max(obs_min_points, obs_min_frac * n)``, else the argmax of {NON_DRIVABLE_TERRAIN, UNKNOWN,
  DRIVABLE} with ties resolved in that order (D-021: unknown is not free). Fractions are compared exactly.
* ``majority`` (ablation): argmax of all five counts, ties resolved DYNAMIC > STATIC > NON_DRIVABLE >
  UNKNOWN > DRIVABLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from foveamap.config import GridConfig
from foveamap.grid.accumulators import GridAccumulators, RingAccumulators
from foveamap.grid.presets import GridSpec
from foveamap.io.labels import DRIVABLE, DYNAMIC, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, UNKNOWN

LAYER_DTYPE = np.dtype(
    [("ground_z", "<i2"), ("top_z", "<i2"), ("clearance", "<i2"), ("cls", "u1"),
     ("moving_frac", "u1"), ("count", "<u2"), ("conf", "u1"), ("flags", "u1")]
)  # fmt: skip
BYTES_PER_CELL = LAYER_DTYPE.itemsize
assert BYTES_PER_CELL == 12, "L10: packed layout must be 12 bytes"

INT16_MIN = int(np.iinfo(np.int16).min)
INT16_MAX = int(np.iinfo(np.int16).max)

FLAG_HAS_GROUND = 1 << 0
FLAG_HAS_OBSTACLE = 1 << 1
FLAG_HAS_OVERHANG = 1 << 2
FLAG_TRAVERSABLE = 1 << 3  # set by the derived-layer stage (Phase 11)
FLAG_KERB = 1 << 4
FLAG_STEEP = 1 << 5
FLAG_LOW_CLEARANCE = 1 << 6

GROUND_TIE_ORDER = (NON_DRIVABLE_TERRAIN, UNKNOWN, DRIVABLE)
MAJORITY_TIE_ORDER = (DYNAMIC, STATIC_OBSTACLE, NON_DRIVABLE_TERRAIN, UNKNOWN, DRIVABLE)

EMPTY_CELL = np.array((INT16_MIN, INT16_MIN, INT16_MIN, UNKNOWN, 0, 0, 0, 0), dtype=LAYER_DTYPE)


@dataclass
class GridLayers:
    """Dense packed layers per ring: ``rings[k]`` is an ``(N_k, N_k)`` structured array indexed ``[iy, ix]``."""

    spec: GridSpec | None
    rings: list[np.ndarray]

    def layer(self, name: str, ring: int) -> np.ndarray:
        return self.rings[ring][name]

    def nbytes(self) -> int:
        """Measured memory of all allocated rings (README 6.9, representation 4)."""
        return int(sum(g.nbytes for g in self.rings))


def _round_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Integer half-up division ``(num + den // 2) // den`` (floor division, so negatives round consistently)."""
    return np.floor_divide(num + den // 2, den)


def _meets(count: np.ndarray, total: np.ndarray, min_points: int, min_frac: float) -> np.ndarray:
    """``count >= max(min_points, min_frac * total)`` evaluated exactly with a rational fraction."""
    frac = Fraction(str(min_frac))
    return (count >= min_points) & (count * frac.denominator >= frac.numerator * total)


def _argmax_with_ties(counts: dict[int, np.ndarray], order: tuple[int, ...]) -> np.ndarray:
    stacked = np.stack([counts[c] for c in order], axis=1)
    return np.asarray(order, dtype=np.uint8)[np.argmax(stacked, axis=1)]  # argmax picks the first maximum


def classify(n_cls: np.ndarray, cfg: GridConfig) -> np.ndarray:
    """Cell super-class from per-class point counts ``(m, 5)`` by ``cfg.class_rule``."""
    n_cls = n_cls.astype(np.int64)
    counts = {c: n_cls[:, c] for c in range(5)}
    total = n_cls.sum(axis=1)
    if cfg.class_rule == "majority":
        return _argmax_with_ties(counts, MAJORITY_TIE_ORDER)
    t = cfg.class_thresholds
    dyn = _meets(counts[DYNAMIC], total, t.dyn_min_points, t.dyn_min_frac)
    static = _meets(counts[STATIC_OBSTACLE], total, t.obs_min_points, t.obs_min_frac)
    ground = _argmax_with_ties(counts, GROUND_TIE_ORDER)
    return np.where(dyn, DYNAMIC, np.where(static, STATIC_OBSTACLE, ground)).astype(np.uint8)


def cell_layers(acc: RingAccumulators, cfg: GridConfig) -> np.ndarray:
    """Packed layers for the occupied cells of one ring, aligned with ``acc.cells`` (``(m,)`` LAYER_DTYPE)."""
    n = acc.n_total.astype(np.int64)
    n_cls = acc.n_cls.astype(np.int64)
    n_ground = n_cls[:, 1] + n_cls[:, 2]
    has_ground = n_ground > 0
    has_obstacle = (n_cls[:, 3] + n_cls[:, 4]) > 0

    safe_ng = np.where(has_ground, n_ground, 1)
    if cfg.ground_estimator == "mean":
        ground_est = _round_div(acc.g_sum, safe_ng)
    else:
        ground_est = acc.g_min.astype(np.int64)
    ground_z = np.where(has_ground, ground_est, INT16_MIN)
    top_z = np.where(has_obstacle, acc.o_max.astype(np.int64), INT16_MIN)

    gap = acc.o_min.astype(np.int64) - ground_z
    both = has_ground & has_obstacle
    overhang = both & (gap >= cfg.contact_height_mm)
    clearance = np.where(overhang, np.minimum(gap, INT16_MAX), 0)
    clearance = np.where(has_ground, clearance, INT16_MIN)

    safe_n = np.maximum(n, 1)
    out = np.empty(acc.n_cells, dtype=LAYER_DTYPE)
    out["ground_z"] = np.clip(ground_z, INT16_MIN, INT16_MAX)
    out["top_z"] = np.clip(top_z, INT16_MIN, INT16_MAX)
    out["clearance"] = clearance
    out["cls"] = classify(acc.n_cls, cfg)
    out["moving_frac"] = _round_div(255 * acc.n_moving.astype(np.int64), safe_n)
    out["count"] = np.minimum(n, 65535)
    out["conf"] = _round_div(acc.conf_sum.astype(np.int64), safe_n)
    out["flags"] = (
        has_ground * FLAG_HAS_GROUND + has_obstacle * FLAG_HAS_OBSTACLE + overhang * FLAG_HAS_OVERHANG
    ).astype(np.uint8)
    return out


def finalize(acc: GridAccumulators, cfg: GridConfig, spec: GridSpec | None = None) -> GridLayers:
    """Accumulators -> dense packed 12-byte layers per ring; empty cells are ``EMPTY_CELL``."""
    rings = []
    for r in acc.rings:
        grid = np.empty(r.side * r.side, dtype=LAYER_DTYPE)
        grid[:] = EMPTY_CELL
        grid[r.cells] = cell_layers(r, cfg)
        rings.append(grid.reshape(r.side, r.side))
    return GridLayers(spec if spec is not None else acc.spec, rings)
