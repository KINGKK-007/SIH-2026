"""Central-difference slope, masked where neighbours lack ground (README 6.7, task T11.2).

slope = atan(sqrt(gx² + gy²)) where gx, gy are central differences of ground_z / s_k.
Sets FLAG_STEEP in cell flags where slope > max_slope_deg.
"""

from __future__ import annotations

import numpy as np

from foveamap.derived.halo import extract_padded_ring
from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_STEEP,
    GridLayers,
)


def compute_slope_deg(layers: GridLayers, cfg: object) -> list[np.ndarray]:
    """Compute local ground surface slope in degrees for each ring."""
    slopes: list[np.ndarray] = []
    spec = layers.spec

    for r_idx, ring in enumerate(layers.rings):
        side = ring.shape[0]
        padded = extract_padded_ring(layers, r_idx, cfg)

        cell_mm = float(spec.rings[r_idx].cell_mm if spec is not None else 100)

        center_flags = padded["flags"][1:-1, 1:-1]
        center_has_g = (center_flags & FLAG_HAS_GROUND) != 0
        center_gz = padded["ground_z"][1:-1, 1:-1].astype(np.float64)

        # Neighbours in X
        right_flags = padded["flags"][1:-1, 2:]
        right_has_g = (right_flags & FLAG_HAS_GROUND) != 0
        right_gz = padded["ground_z"][1:-1, 2:].astype(np.float64)

        left_flags = padded["flags"][1:-1, :-2]
        left_has_g = (left_flags & FLAG_HAS_GROUND) != 0
        left_gz = padded["ground_z"][1:-1, :-2].astype(np.float64)

        both_x = right_has_g & left_has_g
        only_right = right_has_g & ~left_has_g & center_has_g
        only_left = left_has_g & ~right_has_g & center_has_g

        gx = np.zeros((side, side), dtype=np.float64)
        gx[both_x] = (right_gz[both_x] - left_gz[both_x]) / (2.0 * cell_mm)
        gx[only_right] = (right_gz[only_right] - center_gz[only_right]) / cell_mm
        gx[only_left] = (center_gz[only_left] - left_gz[only_left]) / cell_mm

        # Neighbours in Y
        top_flags = padded["flags"][2:, 1:-1]
        top_has_g = (top_flags & FLAG_HAS_GROUND) != 0
        top_gz = padded["ground_z"][2:, 1:-1].astype(np.float64)

        bot_flags = padded["flags"][:-2, 1:-1]
        bot_has_g = (bot_flags & FLAG_HAS_GROUND) != 0
        bot_gz = padded["ground_z"][:-2, 1:-1].astype(np.float64)

        both_y = top_has_g & bot_has_g
        only_top = top_has_g & ~bot_has_g & center_has_g
        only_bot = bot_has_g & ~top_has_g & center_has_g

        gy = np.zeros((side, side), dtype=np.float64)
        gy[both_y] = (top_gz[both_y] - bot_gz[both_y]) / (2.0 * cell_mm)
        gy[only_top] = (top_gz[only_top] - center_gz[only_top]) / cell_mm
        gy[only_bot] = (center_gz[only_bot] - bot_gz[only_bot]) / cell_mm

        grad_mag = np.sqrt(gx * gx + gy * gy)
        deg = np.degrees(np.arctan(grad_mag))
        deg[~center_has_g] = np.nan
        slopes.append(deg)

    return slopes


def compute_slope(layers: GridLayers, cfg: object) -> GridLayers:
    """Compute slope and update FLAG_STEEP in layer flags."""
    slopes = compute_slope_deg(layers, cfg)
    max_deg = getattr(cfg, "max_slope_deg", 15.0)

    new_rings = [r.copy() for r in layers.rings]
    for r_idx, ring in enumerate(new_rings):
        deg = slopes[r_idx]
        has_g = (ring["flags"] & FLAG_HAS_GROUND) != 0
        is_steep = has_g & (np.nan_to_num(deg, nan=0.0) > max_deg)

        ring["flags"] = np.where(
            is_steep,
            ring["flags"] | FLAG_STEEP,
            ring["flags"] & (255 - FLAG_STEEP),
        )

    return GridLayers(spec=layers.spec, rings=new_rings)
