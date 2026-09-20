"""foveamap.grid.aggregate — Numba scatter-reduce kernel for the grid engine.

Alignment invariants satisfied
--------------------------------
I3 — Each point lands in exactly one cell via the rounding-guarded floor formula::

    ix = int(floor((x + half_extent) / cell_m + GUARD))

    GUARD = 1e-9 ensures that a point sitting precisely on a boundary
    (e.g. x = 10.0000 with half_extent = 10, cell_m = 0.05) maps to the
    correct outer-ring cell rather than the inner-ring's last cell.
    Computed in float64 throughout.

I2 — Ring assignment uses cell-based Chebyshev distance with half-open
    intervals [inner, outer).  A point is tested against ``max(|px|, |py|)``
    against the outer ring's coarse-cell-aligned boundaries.

Notes
-----
- The Numba kernel is compiled once on first call; subsequent calls are fast.
- Set environment variable ``FOVEAMAP_PARALLEL_GRID=1`` to enable ``parallel=True``
  (disabled by default to avoid Numba threading issues inside pytest).
- The pure-Python fallback (``_scatter_reduce_numpy``) is used if Numba is
  unavailable or during testing with ``FOVEAMAP_NO_NUMBA=1``.
"""

from __future__ import annotations

import math
import os

import numpy as np
from numpy.typing import NDArray

from foveamap.io.labels import (
    DRIVABLE,
    DYNAMIC,
    NON_DRIVABLE_TERRAIN,
    STATIC_OBSTACLE,
    UNKNOWN,
)
from foveamap.grid.layers import (
    FLAG_MOVING,
    FLAG_VRU,
    FLAG_OBSERVED,
    MOVING_RAW_ID_MIN,
    MOVING_RAW_ID_MAX,
    VRU_RAW_IDS,
    OBSTACLE_HEIGHT_THRESH_M,
    OVERHANG_HEIGHT_THRESH_M,
)

# ─────────────────────────────────────────────────────────────────────────────
# G13 rounding guard (master plan §11 G13)
# ─────────────────────────────────────────────────────────────────────────────

_GUARD: float = 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# Numba import (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

_USE_NUMBA = os.environ.get("FOVEAMAP_NO_NUMBA", "0") != "1"
_PARALLEL   = os.environ.get("FOVEAMAP_PARALLEL_GRID", "0") == "1"

try:
    if _USE_NUMBA:
        from numba import njit
        # Probe compile: tiny no-op to trigger JIT validation early
        @njit(cache=True)
        def _probe() -> int:
            return 0
        _probe()
        _NUMBA_OK = True
    else:
        _NUMBA_OK = False
except Exception:
    _NUMBA_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Index computation helper (shared by Numba and NumPy paths)
# ─────────────────────────────────────────────────────────────────────────────

def _cell_index(coord: float, half_extent: float, cell_m: float) -> int:
    """Map a 1-D coordinate to a cell index within a ring's square array.

    Uses the rounding guard to handle exact boundary values deterministically.

    Parameters
    ----------
    coord : float
        World-frame coordinate (x or y), in metres.
    half_extent : float
        Outer half-extent of the ring, in metres.
    cell_m : float
        Cell side length, in metres.

    Returns
    -------
    int
        Cell index ∈ [0, side), where side = round(2 * half_extent / cell_m).
        Returns -1 if the coordinate is out of range (should not happen after
        ring assignment filtering).
    """
    raw = (float(coord) + half_extent) / cell_m + _GUARD
    idx = int(math.floor(raw))
    side = round(2.0 * half_extent / cell_m)
    if 0 <= idx < side:
        return idx
    # Clamp edge case: a point at exactly the outer boundary due to float noise
    return max(0, min(side - 1, idx))


def _chebyshev(px: float, py: float) -> float:
    """Chebyshev (L∞) distance from origin."""
    return max(abs(px), abs(py))


# ─────────────────────────────────────────────────────────────────────────────
# NumPy (pure-Python) scatter-reduce — used when Numba is unavailable
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_reduce_numpy(
    points: NDArray[np.float32],    # (N, 4)  x, y, z, intensity
    super_cls: NDArray[np.uint8],   # (N,)
    moving: NDArray[np.bool_],      # (N,)
    vru: NDArray[np.bool_],         # (N,)
    conf_raw: NDArray[np.float16],  # (N,)
    ring_arrays: dict[str, NDArray],
    accum: dict[str, NDArray],
    half_extent: float,
    cell_m: float,
    inner_half: float,
    z_min: float,
    z_max: float,
    min_dyn_points: int,
    min_obs_points: int,
) -> int:
    """Scatter points into a single ring's accumulation arrays.

    Returns
    -------
    int
        Number of points assigned to this ring.
    """
    side = round(2.0 * half_extent / cell_m)
    n_assigned = 0

    for i in range(len(points)):
        px = float(points[i, 0])
        py = float(points[i, 1])
        pz = float(points[i, 2])

        # Ring assignment (I2): half-open Chebyshev interval [inner_half, half_extent)
        cheb = _chebyshev(px, py)
        if cheb < inner_half or cheb >= half_extent:
            continue
        # z range filter
        if pz < z_min or pz > z_max:
            continue

        ix = _cell_index(px, half_extent, cell_m)
        iy = _cell_index(py, half_extent, cell_m)

        sc    = int(super_cls[i])
        is_mv = bool(moving[i])
        is_vr = bool(vru[i])
        cf    = float(conf_raw[i])

        # count (saturating at 255)
        if ring_arrays["count"][iy, ix] < 255:
            ring_arrays["count"][iy, ix] += 1

        # flags
        if is_mv:
            ring_arrays["flags"][iy, ix] |= FLAG_MOVING
            accum["_n_moving"][iy, ix]   += 1
        if is_vr:
            ring_arrays["flags"][iy, ix] |= FLAG_VRU
            accum["_n_vru"][iy, ix]      += 1
        ring_arrays["flags"][iy, ix] |= FLAG_OBSERVED

        # superclass accumulators
        if sc == DYNAMIC or is_mv or is_vr:
            accum["_n_dyn"][iy, ix] += 1
        elif sc == STATIC_OBSTACLE:
            accum["_n_obs"][iy, ix] += 1
        elif sc == DRIVABLE:
            accum["_n_drivable"][iy, ix] += 1
        elif sc == NON_DRIVABLE_TERRAIN:
            accum["_n_ndt"][iy, ix] += 1

        # elevation layers
        is_ground = (sc == DRIVABLE or sc == NON_DRIVABLE_TERRAIN)
        if is_ground:
            accum["_sum_z"][iy, ix]    += pz
            accum["_n_ground"][iy, ix] += 1
        else:
            # top_z: max z of non-ground
            cur_top = ring_arrays["top_z"][iy, ix]
            if np.isnan(cur_top) or pz > cur_top:
                ring_arrays["top_z"][iy, ix] = np.float32(pz)

        # confidence accumulator
        accum["_sum_conf"][iy, ix] += cf

        n_assigned += 1

    # ── Finalise ground_z (mean approximation) ───────────────────────────
    ng = accum["_n_ground"]
    has_ground = ng > 0
    ring_arrays["ground_z"][has_ground] = (
        accum["_sum_z"][has_ground] / ng[has_ground]
    ).astype(np.float32)

    # ── overhang_z ────────────────────────────────────────────────────────
    gnd = ring_arrays["ground_z"]
    top = ring_arrays["top_z"]
    has_both = has_ground & ~np.isnan(top)
    above = has_both & ((top - gnd) > OVERHANG_HEIGHT_THRESH_M)
    ring_arrays["overhang_z"][above] = top[above]

    # ── conf (uint8, mean × 255) ──────────────────────────────────────────
    cnt = ring_arrays["count"].astype(np.float64)
    has_pts = cnt > 0
    ring_arrays["conf"][has_pts] = np.clip(
        accum["_sum_conf"][has_pts] / cnt[has_pts] * 255.0, 0, 255
    ).astype(np.uint8)

    # ── dyn_frac (fraction of dynamic points × 255) ───────────────────────
    n_dyn = accum["_n_dyn"]
    ring_arrays["dyn_frac"][has_pts] = np.clip(
        n_dyn[has_pts].astype(np.float64) / cnt[has_pts] * 255.0, 0, 255
    ).astype(np.uint8)

    return n_assigned


# ─────────────────────────────────────────────────────────────────────────────
# Numba kernel (compiled on first call)
# ─────────────────────────────────────────────────────────────────────────────

if _NUMBA_OK:
    from numba import njit as _njit

    @_njit(cache=True, parallel=False)
    def _scatter_reduce_numba_inner(
        x_arr: NDArray[np.float64],
        y_arr: NDArray[np.float64],
        z_arr: NDArray[np.float64],
        super_cls: NDArray[np.uint8],
        moving: NDArray[np.bool_],
        vru: NDArray[np.bool_],
        conf_f32: NDArray[np.float32],
        count: NDArray[np.uint8],
        flags: NDArray[np.uint8],
        top_z: NDArray[np.float32],
        n_dyn: NDArray[np.int32],
        n_obs: NDArray[np.int32],
        n_drivable: NDArray[np.int32],
        n_ndt: NDArray[np.int32],
        n_ground: NDArray[np.int32],
        n_moving: NDArray[np.int32],
        n_vru: NDArray[np.int32],
        sum_z: NDArray[np.float64],
        sum_conf: NDArray[np.float64],
        half_extent: float,
        cell_m: float,
        inner_half: float,
        z_min: float,
        z_max: float,
        side: int,
        guard: float,
        flag_moving: int,
        flag_vru: int,
        flag_observed: int,
        sc_dyn: int,
        sc_static: int,
        sc_driv: int,
        sc_ndt: int,
    ) -> int:
        """Inner Numba kernel: scatter N points into one ring."""
        n_assigned = 0
        for i in range(len(x_arr)):
            px = x_arr[i]
            py = y_arr[i]
            pz = z_arr[i]

            # Chebyshev ring test (I2)
            cheb = abs(px) if abs(px) > abs(py) else abs(py)
            if cheb < inner_half or cheb >= half_extent:
                continue
            if pz < z_min or pz > z_max:
                continue

            # I3 rounding-guarded cell index
            raw_x = (px + half_extent) / cell_m + guard
            raw_y = (py + half_extent) / cell_m + guard
            ix = int(math.floor(raw_x))
            iy = int(math.floor(raw_y))
            if ix < 0: ix = 0
            if ix >= side: ix = side - 1
            if iy < 0: iy = 0
            if iy >= side: iy = side - 1

            sc    = int(super_cls[i])
            is_mv = moving[i]
            is_vr = vru[i]
            cf    = float(conf_f32[i])

            # count (saturating)
            if count[iy, ix] < 255:
                count[iy, ix] += np.uint8(1)

            # flags
            if is_mv:
                flags[iy, ix] |= np.uint8(flag_moving)
                n_moving[iy, ix] += 1
            if is_vr:
                flags[iy, ix] |= np.uint8(flag_vru)
                n_vru[iy, ix]   += 1
            flags[iy, ix] |= np.uint8(flag_observed)

            # super-class accumulators
            is_dyn_point = (sc == sc_dyn) or is_mv or is_vr
            if is_dyn_point:
                n_dyn[iy, ix] += 1
            elif sc == sc_static:
                n_obs[iy, ix] += 1
            elif sc == sc_driv:
                n_drivable[iy, ix] += 1
            elif sc == sc_ndt:
                n_ndt[iy, ix] += 1

            # elevation
            is_ground = (sc == sc_driv) or (sc == sc_ndt)
            if is_ground:
                sum_z[iy, ix]   += pz
                n_ground[iy, ix] += 1
            else:
                cur = top_z[iy, ix]
                if math.isnan(cur) or pz > cur:
                    top_z[iy, ix] = np.float32(pz)

            sum_conf[iy, ix] += cf
            n_assigned += 1

        return n_assigned


def _scatter_reduce_numba(
    points: NDArray[np.float32],
    super_cls: NDArray[np.uint8],
    moving: NDArray[np.bool_],
    vru: NDArray[np.bool_],
    conf_raw: NDArray[np.float16],
    ring_arrays: dict[str, NDArray],
    accum: dict[str, NDArray],
    half_extent: float,
    cell_m: float,
    inner_half: float,
    z_min: float,
    z_max: float,
    min_dyn_points: int,
    min_obs_points: int,
) -> int:
    """Wrapper around the Numba inner kernel, with NumPy finalisation."""
    side = round(2.0 * half_extent / cell_m)

    x64 = points[:, 0].astype(np.float64)
    y64 = points[:, 1].astype(np.float64)
    z64 = points[:, 2].astype(np.float64)
    cf32 = conf_raw.astype(np.float32)

    n_assigned = _scatter_reduce_numba_inner(
        x64, y64, z64, super_cls, moving, vru, cf32,
        ring_arrays["count"],
        ring_arrays["flags"],
        ring_arrays["top_z"],
        accum["_n_dyn"],
        accum["_n_obs"],
        accum["_n_drivable"],
        accum["_n_ndt"],
        accum["_n_ground"],
        accum["_n_moving"],
        accum["_n_vru"],
        accum["_sum_z"],
        accum["_sum_conf"],
        float(half_extent),
        float(cell_m),
        float(inner_half),
        float(z_min),
        float(z_max),
        int(side),
        _GUARD,
        int(FLAG_MOVING),
        int(FLAG_VRU),
        int(FLAG_OBSERVED),
        int(DYNAMIC),
        int(STATIC_OBSTACLE),
        int(DRIVABLE),
        int(NON_DRIVABLE_TERRAIN),
    )

    # ── Finalise (NumPy vectorised, same as pure path) ────────────────────
    ng = accum["_n_ground"]
    has_ground = ng > 0
    ring_arrays["ground_z"][has_ground] = (
        accum["_sum_z"][has_ground] / ng[has_ground]
    ).astype(np.float32)

    top = ring_arrays["top_z"]
    gnd = ring_arrays["ground_z"]
    has_both = has_ground & ~np.isnan(top)
    above = has_both & ((top - gnd) > OVERHANG_HEIGHT_THRESH_M)
    ring_arrays["overhang_z"][above] = top[above]

    cnt = ring_arrays["count"].astype(np.float64)
    has_pts = cnt > 0
    ring_arrays["conf"][has_pts] = np.clip(
        accum["_sum_conf"][has_pts] / cnt[has_pts] * 255.0, 0, 255
    ).astype(np.uint8)

    n_dyn = accum["_n_dyn"]
    ring_arrays["dyn_frac"][has_pts] = np.clip(
        n_dyn[has_pts].astype(np.float64) / cnt[has_pts] * 255.0, 0, 255
    ).astype(np.uint8)

    return int(n_assigned)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def scatter_reduce(
    points: NDArray[np.float32],
    super_cls: NDArray[np.uint8],
    moving: NDArray[np.bool_],
    vru: NDArray[np.bool_],
    conf_raw: NDArray[np.float16],
    ring_arrays: dict[str, NDArray],
    accum: dict[str, NDArray],
    half_extent: float,
    cell_m: float,
    inner_half: float,
    z_min: float,
    z_max: float,
    min_dyn_points: int = 2,
    min_obs_points: int = 2,
) -> int:
    """Scatter N points into one ring's accumulation arrays.

    Dispatches to the Numba kernel if available, otherwise falls back to the
    pure-NumPy implementation.

    Parameters
    ----------
    points : (N, 4) float32
        x, y, z, intensity in the Velodyne frame.
    super_cls : (N,) uint8
        Per-point super-class IDs.
    moving : (N,) bool
        Per-point moving flag (True if raw ID ∈ 252–259 or motion-module flag).
    vru : (N,) bool
        Per-point VRU flag (True if raw ID ∈ {30, 31, 32}).
    conf_raw : (N,) float16
        Per-point confidence in [0, 1].
    ring_arrays : dict
        Output layer arrays (allocated by :func:`layers.make_ring_arrays`).
    accum : dict
        Intermediate accumulation arrays (allocated by :func:`layers.make_accum_arrays`).
    half_extent : float
        Outer half-extent of this ring (metres).
    cell_m : float
        Cell side length (metres).
    inner_half : float
        Inner half-extent of this ring; 0.0 for the innermost ring.
    z_min, z_max : float
        Velodyne-frame z clipping range.
    min_dyn_points, min_obs_points : int
        Thresholds for safety_priority aggregation (not used here, passed for
        completeness).

    Returns
    -------
    int
        Number of points assigned to this ring.
    """
    fn = _scatter_reduce_numba if _NUMBA_OK else _scatter_reduce_numpy
    return fn(
        points, super_cls, moving, vru, conf_raw,
        ring_arrays, accum,
        half_extent, cell_m, inner_half,
        z_min, z_max,
        min_dyn_points, min_obs_points,
    )
