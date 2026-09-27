"""Display-only semantic groups aligned to the occupied adaptive-grid cells.

The packed 12-byte map remains unchanged. This optional view layer retains the
vehicle distinction that the five navigation superclasses intentionally merge.
"""

from __future__ import annotations

import numpy as np

from foveamap.grid.accumulators import GridAccumulators
from foveamap.grid.engine import locate, prepare_points
from foveamap.grid.layers import GridLayers
from foveamap.grid.presets import GridSpec

# Dashboard categories: unknown, drivable, static vehicle, moving vehicle,
# static non-vehicle obstacle, other classified points.
VEHICLE_IDS = np.array([10, 11, 13, 15, 16, 18, 20, 31, 32, 252, 253, 255, 256, 257, 258, 259])


def display_groups(
    spec: GridSpec,
    layers: GridLayers,
    acc: GridAccumulators,
    xyz_m: np.ndarray,
    raw_ids: np.ndarray,
    super_cls: np.ndarray,
    moving: np.ndarray,
    min_range_mm: int,
) -> list[np.ndarray]:
    """Return one uint8 category per occupied cell, in sparse protocol order.

    For obstacle cells, vehicle and other obstacle points vote separately.
    Motion comes from the pipeline's final per-point motion state. Terrain and
    unknown cells retain their grid superclass. No extra field is allocated in
    the map itself or included in its memory comparison.
    """
    xyz_mm, valid, _ = prepare_points(xyz_m, min_range_mm)
    point_ring, ix, iy = locate(spec, xyz_mm[:, 0], xyz_mm[:, 1])
    raw = np.asarray(raw_ids)[valid] & 0xFFFF
    moved = np.asarray(moving, dtype=bool)[valid]
    obstacle = np.isin(np.asarray(super_cls)[valid], [3, 4])
    vehicle = np.isin(raw, VEHICLE_IDS)
    result: list[np.ndarray] = []

    for k, ring_acc in enumerate(acc.rings):
        cells = ring_acc.cells
        grid = layers.rings[k]
        cls = grid[ring_acc.iy, ring_acc.ix]["cls"]
        category = np.choose(cls, [0, 1, 5, 4, 5]).astype(np.uint8)
        if cells.size == 0:
            result.append(category)
            continue

        in_ring = point_ring == k
        point_cells = iy[in_ring] * spec.rings[k].side + ix[in_ring]
        positions = np.searchsorted(cells, point_cells)
        matches = (positions < cells.size) & (cells[np.minimum(positions, cells.size - 1)] == point_cells)
        positions = positions[matches]
        is_vehicle = vehicle[in_ring][matches]
        is_moving = moved[in_ring][matches]
        is_obstacle = obstacle[in_ring][matches]
        vehicle_static = np.bincount(positions[is_vehicle & ~is_moving], minlength=cells.size)
        vehicle_moving = np.bincount(positions[is_vehicle & is_moving], minlength=cells.size)
        other_points = np.bincount(positions[~is_vehicle & is_obstacle], minlength=cells.size)

        static_cells = cls == 3
        dynamic_cells = cls == 4
        category[static_cells & (vehicle_static > other_points)] = 2
        category[dynamic_cells & (vehicle_moving > other_points)] = 3
        result.append(category)
    return result
