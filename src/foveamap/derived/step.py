"""Step/kerb detector (README 6.7, task T11.3).

For each cell with ground, computes max absolute ground_z difference to its 4-neighbours
that also have ground and at least min_points_step points.
Sets FLAG_KERB if step in [kerb_min_m, kerb_max_m].
"""

from __future__ import annotations

import numpy as np

from foveamap.derived.halo import extract_padded_ring
from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_KERB,
    FLAG_POTHOLE,
    GridLayers,
)
from foveamap.io.labels import DRIVABLE


def compute_step_height_mm(layers: GridLayers, cfg: object) -> list[np.ndarray]:
    """Compute maximum absolute step difference across valid 4-neighbours in mm."""
    steps: list[np.ndarray] = []
    min_pts = getattr(cfg, "min_points_step", 2)

    for r_idx, ring in enumerate(layers.rings):
        side = ring.shape[0]
        padded = extract_padded_ring(layers, r_idx, cfg)

        center_flags = padded["flags"][1:-1, 1:-1]
        center_has_g = (center_flags & FLAG_HAS_GROUND) != 0
        center_gz = padded["ground_z"][1:-1, 1:-1].astype(np.int64)

        max_step = np.zeros((side, side), dtype=np.int64)

        # 4-neighbour relative offsets (dy, dx)
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dy, dx in offsets:
            ny0, ny1 = 1 + dy, 1 + dy + side
            nx0, nx1 = 1 + dx, 1 + dx + side

            nbr_flags = padded["flags"][ny0:ny1, nx0:nx1]
            nbr_count = padded["count"][ny0:ny1, nx0:nx1]
            nbr_has_g = ((nbr_flags & FLAG_HAS_GROUND) != 0) & (nbr_count >= min_pts)
            nbr_gz = padded["ground_z"][ny0:ny1, nx0:nx1].astype(np.int64)

            valid_pair = center_has_g & nbr_has_g
            diff = np.where(valid_pair, np.abs(center_gz - nbr_gz), 0)
            max_step = np.maximum(max_step, diff)

        max_step[~center_has_g] = 0
        steps.append(max_step)

    return steps


def compute_step(layers: GridLayers, cfg: object) -> GridLayers:
    """Detect kerbs and obstacle steps (FLAG_KERB) and potholes/depressions (FLAG_POTHOLE)."""
    steps = compute_step_height_mm(layers, cfg)

    kerb_min_mm = getattr(cfg, "kerb_min_mm", int(getattr(cfg, "kerb_min_m", 0.06) * 1000))
    kerb_max_mm = getattr(cfg, "kerb_max_mm", int(getattr(cfg, "kerb_max_m", 0.25) * 1000))
    # Pothole: drivable cell whose ground is ≥ pothole_depth_mm below the neighbourhood mean
    pothole_depth_mm = int(getattr(cfg, "pothole_depth_mm", 80))  # 8 cm default

    new_rings = [r.copy() for r in layers.rings]
    for r_idx, ring in enumerate(new_rings):
        step_mm = steps[r_idx]
        has_g = (ring["flags"] & FLAG_HAS_GROUND) != 0
        is_kerb = has_g & (step_mm >= kerb_min_mm) & (step_mm <= kerb_max_mm)

        ring["flags"] = np.where(
            is_kerb,
            ring["flags"] | FLAG_KERB,
            ring["flags"] & (255 - FLAG_KERB),
        )

        # Pothole detection: drivable cell with ground significantly BELOW neighbours.
        # We reuse the padded ring to compute neighbourhood mean ground height.
        padded = extract_padded_ring(layers, r_idx, cfg)
        side = ring.shape[0]
        center_gz = padded["ground_z"][1:-1, 1:-1].astype(np.int32)
        center_cls = ring["cls"]
        is_drivable = center_cls == DRIVABLE

        # Compute neighbourhood mean from 4-neighbours that have ground
        nbr_sum = np.zeros((side, side), dtype=np.int32)
        nbr_cnt = np.zeros((side, side), dtype=np.int32)
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny0, ny1 = 1 + dy, 1 + dy + side
            nx0, nx1 = 1 + dx, 1 + dx + side
            nbr_flags = padded["flags"][ny0:ny1, nx0:nx1]
            nbr_has_g = (nbr_flags & FLAG_HAS_GROUND) != 0
            nbr_gz = padded["ground_z"][ny0:ny1, nx0:nx1].astype(np.int32)
            nbr_sum += np.where(nbr_has_g, nbr_gz, 0)
            nbr_cnt += nbr_has_g.astype(np.int32)

        has_nbrs = nbr_cnt > 0
        nbr_mean = np.where(has_nbrs, nbr_sum // np.maximum(nbr_cnt, 1), center_gz)
        depression = nbr_mean.astype(np.int32) - center_gz  # positive = cell is below neighbours

        is_pothole = has_g & is_drivable & has_nbrs & (depression >= pothole_depth_mm)
        ring["flags"] = np.where(
            is_pothole,
            ring["flags"] | FLAG_POTHOLE,
            ring["flags"] & ~np.uint8(FLAG_POTHOLE),
        ).astype(np.uint8)

    return GridLayers(spec=layers.spec, rings=new_rings)
