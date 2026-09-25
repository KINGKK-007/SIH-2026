"""3-state traversability (unknown is not free) (README 6.7, task T11.5).

Evaluates drivability based on super-class, slope, kerb/step, overhang clearance,
and obstacles in cell. Sets FLAG_TRAVERSABLE in cell flags.
"""

from __future__ import annotations

import numpy as np

from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_HAS_OBSTACLE,
    FLAG_KERB,
    FLAG_LOW_CLEARANCE,
    FLAG_STEEP,
    FLAG_TRAVERSABLE,
    GridLayers,
)
from foveamap.io.labels import DRIVABLE, UNKNOWN

UNKNOWN_CLASS = 0
TRAVERSABLE_CLASS = 1
NON_TRAVERSABLE_CLASS = 2


def compute_traversability(layers: GridLayers, cfg: object) -> GridLayers:
    """Evaluate traversability and update FLAG_TRAVERSABLE in layer flags."""
    new_rings = [r.copy() for r in layers.rings]

    for ring in new_rings:
        is_drivable = ring["cls"] == DRIVABLE
        has_g = (ring["flags"] & FLAG_HAS_GROUND) != 0
        has_obs = (ring["flags"] & FLAG_HAS_OBSTACLE) != 0
        is_steep = (ring["flags"] & FLAG_STEEP) != 0
        is_kerb = (ring["flags"] & FLAG_KERB) != 0
        is_low_clr = (ring["flags"] & FLAG_LOW_CLEARANCE) != 0

        traversable = (
            is_drivable
            & has_g
            & (~has_obs)
            & (~is_steep)
            & (~is_kerb)
            & (~is_low_clr)
        )

        ring["flags"] = np.where(
            traversable,
            ring["flags"] | FLAG_TRAVERSABLE,
            ring["flags"] & (255 - FLAG_TRAVERSABLE),
        )

    return GridLayers(spec=layers.spec, rings=new_rings)


def traversability_map(layers: GridLayers) -> list[np.ndarray]:
    """Return 3-state categories (0: UNKNOWN, 1: TRAVERSABLE, 2: NON_TRAVERSABLE) per ring."""
    maps: list[np.ndarray] = []
    for ring in layers.rings:
        has_g = (ring["flags"] & FLAG_HAS_GROUND) != 0
        is_trav = (ring["flags"] & FLAG_TRAVERSABLE) != 0
        is_unk = (~has_g) | (ring["cls"] == UNKNOWN)

        cat = np.full(ring.shape, NON_TRAVERSABLE_CLASS, dtype=np.uint8)
        cat[is_unk] = UNKNOWN_CLASS
        cat[is_trav] = TRAVERSABLE_CLASS
        maps.append(cat)
    return maps
