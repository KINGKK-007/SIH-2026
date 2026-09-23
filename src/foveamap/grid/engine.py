"""Integer-only cell addressing (README 6.5.3, task T5.2).

For a quantised point ``(x, y)`` in mm (L6)::

    ring(x, y) = min k such that -R_k <= x < R_k and -R_k <= y < R_k        (L7, half-open)
    ix = floor_div(x, s_k) + R_k / s_k,   iy = floor_div(y, s_k) + R_k / s_k
    flat = iy * N_k + ix,                  N_k = 2 R_k / s_k
    corner = ((ix - R_k/s_k) * s_k, (iy - R_k/s_k) * s_k),   centre = corner + s_k // 2

No floating-point arithmetic touches the boundary path; ``np.floor_divide`` floors negatives correctly.
"""

from __future__ import annotations

import numpy as np

from foveamap.grid.presets import GridSpec

INT32_MAX = np.iinfo(np.int32).max


def quantize_mm(xyz: np.ndarray) -> np.ndarray:
    """``np.rint(xyz * 1000).astype(np.int32)`` (L6); rejects non-finite values and int32 overflow."""
    mm = np.rint(np.asarray(xyz, dtype=np.float64) * 1000.0)
    if not np.isfinite(mm).all():
        raise ValueError("quantize_mm needs finite coordinates (filter invalid points first)")
    if mm.size and np.abs(mm).max() > INT32_MAX:
        raise ValueError("coordinates exceed the int32 millimetre range")
    return mm.astype(np.int32)


def locate(spec: GridSpec, x_mm: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised addressing. Returns ``(ring int8, ix int64, iy int64)``; ``-1`` everywhere if out of grid."""
    x = np.asarray(x_mm, dtype=np.int64)
    y = np.asarray(y_mm, dtype=np.int64)
    ring = np.full(x.shape, -1, dtype=np.int8)
    for k in range(len(spec.rings) - 1, -1, -1):  # reverse so the smallest containing ring wins
        r = spec.rings[k].r_max_mm
        ring[(x >= -r) & (x < r) & (y >= -r) & (y < r)] = k
    ix = np.full(x.shape, -1, dtype=np.int64)
    iy = np.full(x.shape, -1, dtype=np.int64)
    for k, rs in enumerate(spec.rings):
        sel = ring == k
        ix[sel] = np.floor_divide(x[sel], rs.cell_mm) + rs.offset
        iy[sel] = np.floor_divide(y[sel], rs.cell_mm) + rs.offset
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
