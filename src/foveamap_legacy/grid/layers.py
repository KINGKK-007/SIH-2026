"""foveamap.grid.layers — Cell data structure constants, dtypes, and aggregation.

Cell memory layout (8 bytes / cell in default config)
------------------------------------------------------
Each ring is stored as a collection of 2-D square numpy arrays, one per layer.
All arrays share the same shape ``(side, side)`` where ``side = 2 * n_half_cells``.

Layer arrays
~~~~~~~~~~~~
- ``ground_z``  : float32  — median z of ground-labelled points (NaN = unobserved)
- ``top_z``     : float32  — max z of non-ground points (NaN = unobserved)
- ``overhang_z``: float32  — lowest z of points >0.3 m above local ground (NaN=none)
- ``cls``       : uint8    — dominant super-class (UNKNOWN=255 if unobserved)
- ``dyn_frac``  : uint8    — fraction of moving/VRU points × 255 (0–255)
- ``count``     : uint8    — point count, saturating at 255
- ``conf``      : uint8    — mean network confidence × 255 (0–255)
- ``flags``     : uint8    — bitmask: bit0=moving, bit1=vru, bit2=observed,
                             bit3=traversable (Phase 5)

Aggregation rules (master plan §5.5 H8)
----------------------------------------
safety_priority (default)
    1. DYNAMIC if ≥ min_dyn_points moving/VRU points.
    2. STATIC_OBSTACLE if ≥ min_obs_points non-ground points > 0.15 m above
       local ground.
    3. Majority of ground-class points (DRIVABLE or NON_DRIVABLE_TERRAIN).
    4. UNKNOWN.

majority
    Plain mode: super-class with the highest point count. UNKNOWN if no points.

Alignment invariants satisfied here
-------------------------------------
I3 — only the aggregate.py kernel determines which cell a point lands in.
I4 — layer arrays accumulate sums/counts/min/max; final values computed once.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from foveamap_legacy.io.labels import DRIVABLE, DYNAMIC, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, UNKNOWN

# ─────────────────────────────────────────────────────────────────────────────
# Layer names and per-cell byte budget
# ─────────────────────────────────────────────────────────────────────────────

#: Names of all layers stored per cell.
LAYER_NAMES: Final[tuple[str, ...]] = (
    "ground_z",
    "top_z",
    "overhang_z",
    "cls",
    "dyn_frac",
    "count",
    "conf",
    "flags",
)

#: Dtype of each layer (used for memory accounting).
LAYER_DTYPES: Final[dict[str, np.dtype]] = {
    "ground_z":   np.dtype("float32"),
    "top_z":      np.dtype("float32"),
    "overhang_z": np.dtype("float32"),
    "cls":        np.dtype("uint8"),
    "dyn_frac":   np.dtype("uint8"),
    "count":      np.dtype("uint8"),
    "conf":       np.dtype("uint8"),
    "flags":      np.dtype("uint8"),
}

#: Bytes per cell when all 8 default layers are stored.
BYTES_PER_CELL: Final[int] = sum(d.itemsize for d in LAYER_DTYPES.values())
# 3×4 + 5×1 = 12 + 5 = 17 bytes raw; the master plan quotes 8 B/cell which
# refers to a compact packed struct.  We store float32 elevations separately
# and report the 8 B/cell target for the *semantic* layers (cls+dyn_frac+count
# +conf+flags = 5 B), rounding the elevation block as needed.  For E3 memory
# accounting we report all 17 B/cell honestly and also show 8 B/cell target.
BYTES_PER_CELL_TARGET: Final[int] = 8   # master-plan headline number

# ─────────────────────────────────────────────────────────────────────────────
# Flag bit positions
# ─────────────────────────────────────────────────────────────────────────────

FLAG_MOVING:      Final[int] = 0b00000001   # bit 0
FLAG_VRU:         Final[int] = 0b00000010   # bit 1
FLAG_OBSERVED:    Final[int] = 0b00000100   # bit 2
FLAG_TRAVERSABLE: Final[int] = 0b00001000   # bit 3

# ─────────────────────────────────────────────────────────────────────────────
# Physical thresholds
# ─────────────────────────────────────────────────────────────────────────────

#: A non-ground point must rise at least this far above local ground to count
#: as a STATIC_OBSTACLE candidate in safety_priority mode.
OBSTACLE_HEIGHT_THRESH_M: Final[float] = 0.15

#: Minimum height above local ground to register as overhang.
OVERHANG_HEIGHT_THRESH_M: Final[float] = 0.30

#: Raw semantic IDs 252–259 are the SemanticKITTI "moving-*" variants.
MOVING_RAW_ID_MIN: Final[int] = 252
MOVING_RAW_ID_MAX: Final[int] = 259

#: Raw IDs for persons/VRUs that are always DYNAMIC regardless of motion flag.
VRU_RAW_IDS: Final[frozenset[int]] = frozenset({30, 31, 32})

# ─────────────────────────────────────────────────────────────────────────────
# Allocation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_ring_arrays(side: int) -> dict[str, NDArray]:
    """Allocate zero-initialised layer arrays for a single ring.

    Parameters
    ----------
    side : int
        Number of cells along one axis of the square.  Total cells = side².

    Returns
    -------
    dict[str, NDArray]
        Layer-name → numpy array of shape ``(side, side)``.
    """
    arrays: dict[str, NDArray] = {}
    for name, dtype in LAYER_DTYPES.items():
        if np.issubdtype(dtype, np.floating):
            arr = np.full((side, side), np.nan, dtype=dtype)
        else:
            arr = np.zeros((side, side), dtype=dtype)
        arrays[name] = arr
    return arrays


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation: safety_priority
# ─────────────────────────────────────────────────────────────────────────────

def apply_safety_priority(
    ring_arrays: dict[str, NDArray],
    min_dyn_points: int = 2,
    min_obs_points: int = 2,
) -> None:
    """Apply safety_priority aggregation rule to a ring's accumulated arrays.

    This operates *in-place* on ``ring_arrays`` after the scatter-reduce kernel
    has finished populating the intermediate accumulation arrays.

    Aggregation steps (master plan §5.5 H8):
    1. DYNAMIC  if n_dyn ≥ min_dyn_points
    2. STATIC_OBSTACLE  if n_obstacle ≥ min_obs_points
    3. Majority ground class (DRIVABLE or NON_DRIVABLE_TERRAIN)
    4. UNKNOWN

    Parameters
    ----------
    ring_arrays : dict[str, NDArray]
        Must contain the intermediate accumulation arrays produced by
        :func:`aggregate.scatter_reduce`.  The ``cls`` and ``flags`` arrays
        are updated in place.
    min_dyn_points, min_obs_points : int
        Minimum point counts for DYNAMIC / STATIC_OBSTACLE classification.
    """
    n_dyn       = ring_arrays["_n_dyn"]
    n_obs       = ring_arrays["_n_obs"]
    n_drivable  = ring_arrays["_n_drivable"]
    n_ndt       = ring_arrays["_n_ndt"]

    cls = ring_arrays["cls"]

    # Start from UNKNOWN
    cls[:] = UNKNOWN

    # Step 3: majority ground
    ground_mask = (n_drivable + n_ndt) > 0
    cls[ground_mask & (n_drivable >= n_ndt)] = DRIVABLE
    cls[ground_mask & (n_ndt > n_drivable)]  = NON_DRIVABLE_TERRAIN

    # Step 2: STATIC_OBSTACLE wins over ground
    cls[n_obs >= min_obs_points] = STATIC_OBSTACLE

    # Step 1: DYNAMIC wins over everything
    cls[n_dyn >= min_dyn_points] = DYNAMIC


def apply_majority(ring_arrays: dict[str, NDArray]) -> None:
    """Apply majority-vote aggregation rule to a ring's accumulated arrays.

    Parameters
    ----------
    ring_arrays : dict[str, NDArray]
        Must contain the intermediate accumulation arrays.
    """
    n_dyn      = ring_arrays["_n_dyn"]
    n_static   = ring_arrays["_n_obs"]       # re-use _n_obs as all-static count
    n_drivable = ring_arrays["_n_drivable"]
    n_ndt      = ring_arrays["_n_ndt"]
    total      = ring_arrays["count"].astype(np.int32)

    cls = ring_arrays["cls"]
    cls[:] = UNKNOWN

    # Stack counts; argmax gives dominant class
    stack = np.stack([n_drivable, n_ndt, n_static, n_dyn], axis=-1).astype(np.int32)
    has_any = total > 0
    dominant = np.argmax(stack, axis=-1)  # 0=drivable,1=ndt,2=static,3=dyn

    mapping = np.array([DRIVABLE, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, DYNAMIC], dtype=np.uint8)
    cls[has_any] = mapping[dominant[has_any]]


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate accumulation array names
# (not part of the public layer API but needed by aggregate.py)
# ─────────────────────────────────────────────────────────────────────────────

#: Extra arrays needed during scatter-reduce (not part of LAYER_NAMES).
ACCUM_NAMES: Final[tuple[str, ...]] = (
    "_n_dyn",       # int32 — count of DYNAMIC points
    "_n_obs",       # int32 — count of obstacle points (>OBSTACLE_HEIGHT_THRESH above ground)
    "_n_drivable",  # int32 — count of DRIVABLE points
    "_n_ndt",       # int32 — count of NON_DRIVABLE_TERRAIN points
    "_sum_z",       # float64 — sum of ground z for median approximation
    "_sum_conf",    # float64 — sum of confidence values
    "_n_ground",    # int32 — count of ground points
    "_n_moving",    # int32 — count of points with moving flag set
    "_n_vru",       # int32 — count of VRU points
)


def make_accum_arrays(side: int) -> dict[str, NDArray]:
    """Allocate intermediate accumulation arrays for a ring.

    Parameters
    ----------
    side : int
        Ring square side length in cells.

    Returns
    -------
    dict[str, NDArray]
    """
    accum: dict[str, NDArray] = {}
    int_names  = {"_n_dyn", "_n_obs", "_n_drivable", "_n_ndt",
                  "_n_ground", "_n_moving", "_n_vru"}
    float_names = {"_sum_z", "_sum_conf"}
    for name in ACCUM_NAMES:
        if name in float_names:
            accum[name] = np.zeros((side, side), dtype=np.float64)
        else:
            accum[name] = np.zeros((side, side), dtype=np.int32)
    return accum
