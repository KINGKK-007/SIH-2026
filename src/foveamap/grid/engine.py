"""Integer-only cell addressing (README 6.5.3, task T5.2).

For a quantised point ``(x, y)`` in mm (L6)::

    ring(x, y) = min k such that -R_k <= x < R_k and -R_k <= y < R_k        (L7, half-open)
    ix = floor_div(x, s_k) + R_k / s_k,   iy = floor_div(y, s_k) + R_k / s_k
    flat = iy * N_k + ix,                  N_k = 2 R_k / s_k
    corner = ((ix - R_k/s_k) * s_k, (iy - R_k/s_k) * s_k),   centre = corner + s_k // 2

No floating-point arithmetic touches the boundary path; ``np.floor_divide`` floors negatives correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from foveamap.grid.accumulators import INT32_MAX, Z_SATURATION_MM, GridAccumulators
from foveamap.grid.presets import GridSpec

if TYPE_CHECKING:
    from foveamap.grid.backends import GridBackend


def quantize_mm(xyz: np.ndarray) -> np.ndarray:
    """``np.rint(xyz * 1000).astype(np.int32)`` (L6); rejects non-finite values and int32 overflow."""
    mm = np.rint(np.asarray(xyz, dtype=np.float64) * 1000.0)
    if not np.isfinite(mm).all():
        raise ValueError("quantize_mm needs finite coordinates (filter invalid points first)")
    if mm.size and np.abs(mm).max() > INT32_MAX:
        raise ValueError("coordinates exceed the int32 millimetre range")
    return mm.astype(np.int32)


def locate(spec: GridSpec, x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised addressing. Returns ``(ring int8, ix int64, iy int64)``; ``-1`` everywhere if out of grid.

    For integers, ``-R <= v < R`` is exactly ``max(v, -v - 1) < R``, so the smallest containing ring is one
    ``searchsorted`` of ``max(x, -x-1, y, -y-1)`` over the ring radii (still exact integer arithmetic, L7).
    """
    x = np.asarray(x_mm, dtype=np.int64)
    y = np.asarray(y_mm, dtype=np.int64)
    radii = np.array([r.r_max_mm for r in spec.rings], dtype=np.int64)
    a = np.maximum(np.maximum(x, -x - 1), np.maximum(y, -y - 1))
    k = np.searchsorted(radii, a, side="right")
    inside = k < len(radii)
    ring = np.where(inside, k, -1).astype(np.int8)
    cell = np.array([r.cell_mm for r in spec.rings], dtype=np.int64)
    offset = np.array([r.offset for r in spec.rings], dtype=np.int64)
    kk = np.where(inside, k, 0)
    ix = np.where(inside, np.floor_divide(x, cell[kk]) + offset[kk], -1)
    iy = np.where(inside, np.floor_divide(y, cell[kk]) + offset[kk], -1)
    return ring, ix, iy


def world_to_cell(spec: GridSpec, x_mm: int, y_mm: int) -> tuple[int, int, int] | None:
    """Return ``(ring, ix, iy)`` or ``None`` if the point is out of grid."""
    for k, rs in enumerate(spec.rings):
        r = rs.r_max_mm
        if -r <= x_mm < r and -r <= y_mm < r:
            return k, x_mm // rs.cell_mm + rs.offset, y_mm // rs.cell_mm + rs.offset
    return None


def cell_to_corner_mm(spec: GridSpec, ring: int, ix: int, iy: int) -> tuple[int, int]:
    """Lower-left corner ``(x, y)`` of a cell in mm; always a multiple of the ring's cell size (I3)."""
    rs = spec.rings[ring]
    return (ix - rs.offset) * rs.cell_mm, (iy - rs.offset) * rs.cell_mm


def cell_to_center_mm(spec: GridSpec, ring: int, ix: int, iy: int) -> tuple[int, int]:
    """Cell centre in mm (``corner + s // 2``; exact for the even cell sizes of every preset)."""
    cx, cy = cell_to_corner_mm(spec, ring, ix, iy)
    half = spec.rings[ring].cell_mm // 2
    return cx + half, cy + half


def is_logical_cell(spec: GridSpec, ring: int, ix: int, iy: int) -> bool:
    """True if the cell belongs to the ring's annulus, i.e. not to the masked hole of the inner ring.

    V4 guarantees no cell straddles the inner boundary, so a cell is in the hole iff it lies entirely
    inside ``[-R_{k-1}, R_{k-1})^2``.
    """
    rs = spec.rings[ring]
    if not (0 <= ix < rs.side and 0 <= iy < rs.side):
        return False
    inner = rs.r_min_mm
    cx, cy = cell_to_corner_mm(spec, ring, ix, iy)
    in_hole = -inner <= cx and cx + rs.cell_mm <= inner and -inner <= cy and cy + rs.cell_mm <= inner
    return not in_hole


def logical_mask(spec: GridSpec, ring: int) -> np.ndarray:
    """``(N_k, N_k)`` bool mask indexed ``[iy, ix]``; False on the inner ring's hole."""
    rs = spec.rings[ring]
    corners = (np.arange(rs.side, dtype=np.int64) - rs.offset) * rs.cell_mm
    inside_1d = (corners >= -rs.r_min_mm) & (corners + rs.cell_mm <= rs.r_min_mm)
    return ~(inside_1d[:, None] & inside_1d[None, :])


def prepare_points(xyz_m: np.ndarray, min_range_mm: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Filter and quantise a scan (README 6.2, L6).

    Drops non-finite points and points with horizontal range below ``min_range_mm`` (counted as invalid),
    quantises to integer mm once, and clamps z to +-32767 mm (I10). x/y are clamped to the int32 range;
    such points are far outside any grid and stay out of grid. Returns ``(xyz_mm int32, valid mask,
    n_z_saturated)``.
    """
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got shape {xyz.shape}")
    finite = np.isfinite(xyz).all(axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
        r2_m2 = xyz[:, 0] * xyz[:, 0] + xyz[:, 1] * xyz[:, 1]
    valid = finite & (r2_m2 * 1e6 >= float(min_range_mm) ** 2)
    mm = np.rint(xyz[valid] * 1000.0)
    saturated = np.abs(mm[:, 2]) > Z_SATURATION_MM
    mm[:, 2] = np.clip(mm[:, 2], -Z_SATURATION_MM, Z_SATURATION_MM)
    mm[:, :2] = np.clip(mm[:, :2], -INT32_MAX, INT32_MAX)
    return mm.astype(np.int32), valid, int(saturated.sum())


def rasterize(
    spec: GridSpec,
    xyz_m: np.ndarray,
    super_cls: np.ndarray,
    moving: np.ndarray,
    conf: np.ndarray,
    min_range_mm: int = 0,
    backend: GridBackend | None = None,
) -> GridAccumulators:
    """Scan (metres, Velodyne frame) -> accumulators with complete frame counters (README 6.5.4-6.5.6)."""
    from foveamap.grid.backends import get_backend

    super_cls, moving, conf = np.asarray(super_cls), np.asarray(moving), np.asarray(conf)
    if not (len(super_cls) == len(moving) == len(conf) == len(xyz_m)):
        raise ValueError("xyz, super_cls, moving and conf must have the same length")
    xyz_mm, valid, n_saturated = prepare_points(xyz_m, min_range_mm)
    acc = (backend or get_backend()).rasterize(spec, xyz_mm, super_cls[valid], moving[valid], conf[valid])
    acc.counters.n_raw = len(xyz_m)
    acc.counters.n_invalid = int((~valid).sum())
    acc.counters.n_z_saturated = n_saturated
    c = acc.counters
    assert c.n_raw == c.n_invalid + c.n_in_grid + c.n_out_of_grid, "point conservation violated (I1)"
    return acc
