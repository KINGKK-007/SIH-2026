"""Overhang clearance and low_clearance flag (README 6.7, task T11.4).

Computes vertical gap under overhead obstacles.
Sets FLAG_LOW_CLEARANCE if clearance < vehicle_height_m + clearance_margin_m.
"""

from __future__ import annotations

import numpy as np

from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_HAS_OBSTACLE,
    FLAG_HAS_OVERHANG,
    FLAG_LOW_CLEARANCE,
    INT16_MIN,
    GridLayers,
)


def compute_clearance(layers: GridLayers, cfg: object) -> GridLayers:
    """Evaluate clearance under overhead obstacles and update FLAG_LOW_CLEARANCE."""
    veh_h = getattr(cfg, "vehicle_height_mm", int(getattr(cfg, "vehicle_height_m", 2.0) * 1000))
    margin = getattr(cfg, "clearance_margin_mm", int(getattr(cfg, "clearance_margin_m", 0.2) * 1000))
    min_headway_mm = veh_h + margin

    new_rings = [r.copy() for r in layers.rings]

    for ring in new_rings:
        has_oh = (ring["flags"] & FLAG_HAS_OVERHANG) != 0
        clr = ring["clearance"].astype(np.int64)

        # Clearance < headway threshold means danger of collision
        is_low = has_oh & (clr > 0) & (clr < min_headway_mm)

        ring["flags"] = np.where(
            is_low,
            ring["flags"] | FLAG_LOW_CLEARANCE,
            ring["flags"] & (255 - FLAG_LOW_CLEARANCE),
        )

    return GridLayers(spec=layers.spec, rings=new_rings)
