"""foveamap.derive.layers — Derived geometry layers (slope, step, clearance, traversability).

Per master plan §5.6:
Derived safety layers are computed purely from grid geometry without learning:
1. Slope          : Local gradient of ground_z (in degrees) per ring.
2. Step height    : Maximum elevation difference to 8 neighbors (detects kerbs/potholes).
3. Clearance      : Vertical headway clearance (overhang_z - ground_z).
4. Traversability : Passable cells satisfying:
                    drivable_class ∧ slope < 15° ∧ step < 10 cm ∧ clearance ≥ 2.0 m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from foveamap_legacy.io.labels import DRIVABLE

if TYPE_CHECKING:
    from foveamap_legacy.grid.clipmap import ClipmapGrid


@dataclass(frozen=True)
class DerivedLayers:
    """Container holding all derived geometric safety layers across all rings.

    Attributes
    ----------
    slope_deg : list[NDArray[float32]]
        Ground slope in degrees for each ring, shape ``(side, side)``.
    step_height_m : list[NDArray[float32]]
        Maximum step height across 8 neighbors in metres, shape ``(side, side)``.
    clearance_m : list[NDArray[float32]]
        Vertical clearance in metres, shape ``(side, side)``.
    traversable : list[NDArray[bool]]
        Boolean passability mask for each ring, shape ``(side, side)``.
    """

    slope_deg: list[NDArray[np.float32]]
    step_height_m: list[NDArray[np.float32]]
    clearance_m: list[NDArray[np.float32]]
    traversable: list[NDArray[np.bool_]]

    @property
    def n_rings(self) -> int:
        """Number of rings in the clipmap."""
        return len(self.traversable)


def compute_slope(grid: ClipmapGrid) -> list[NDArray[np.float32]]:
    """Compute local ground surface slope in degrees for each ring.

    Parameters
    ----------
    grid : ClipmapGrid
        Populated FoveaMap clipmap grid.

    Returns
    -------
    slopes : list[NDArray[float32]]
        Per-ring 2D arrays of slope values in degrees.
    """
    slopes: list[NDArray[np.float32]] = []

    for r_idx in range(grid.n_rings):
        z = grid.layer("ground_z", r_idx)
        cell_m = grid.spec.rings[r_idx].cell_m

        # Handle NaNs: replace with nearest valid neighbor or nan-safe diffs
        # 2D Central differences with cell resolution spacing
        valid = ~np.isnan(z)

        # Pad with nan border for unified difference calculation
        padded = np.pad(z, 1, mode="constant", constant_values=np.nan)

        # Horizontal diff: (right - left) / (2 * cell_m)
        dz_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * cell_m)
        # Fallback to forward/backward if one side is NaN
        fwd_x = (padded[1:-1, 2:] - padded[1:-1, 1:-1]) / cell_m
        bwd_x = (padded[1:-1, 1:-1] - padded[1:-1, :-2]) / cell_m
        dz_dx = np.where(~np.isnan(dz_dx), dz_dx, np.where(~np.isnan(fwd_x), fwd_x, bwd_x))

        # Vertical diff: (bottom - top) / (2 * cell_m)
        dz_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * cell_m)
        fwd_y = (padded[2:, 1:-1] - padded[1:-1, 1:-1]) / cell_m
        bwd_y = (padded[1:-1, 1:-1] - padded[:-2, 1:-1]) / cell_m
        dz_dy = np.where(~np.isnan(dz_dy), dz_dy, np.where(~np.isnan(fwd_y), fwd_y, bwd_y))

        # Gradient magnitude
        grad_mag_sq = np.nan_to_num(dz_dx * dz_dx, nan=0.0) + np.nan_to_num(dz_dy * dz_dy, nan=0.0)
        grad_mag = np.sqrt(grad_mag_sq)

        # Convert to degrees: arctan(slope) * 180 / pi
        slope_deg = np.degrees(np.arctan(grad_mag)).astype(np.float32)
        slope_deg[~valid] = np.nan
        slopes.append(slope_deg)

    return slopes


def compute_step_height(grid: ClipmapGrid) -> list[NDArray[np.float32]]:
    """Compute maximum absolute step height across 8 adjacent neighbors.

    Identifies kerbs, potholes, and terrain discontinuities.

    Parameters
    ----------
    grid : ClipmapGrid
        Populated FoveaMap clipmap grid.

    Returns
    -------
    steps : list[NDArray[float32]]
        Per-ring 2D arrays of maximum step difference in metres.
    """
    steps: list[NDArray[np.float32]] = []

    for r_idx in range(grid.n_rings):
        z = grid.layer("ground_z", r_idx)
        valid = ~np.isnan(z)

        # Pad with NaN borders
        padded = np.pad(z, 1, mode="constant", constant_values=np.nan)
        center = padded[1:-1, 1:-1]

        max_step = np.zeros_like(z, dtype=np.float32)

        # 8-neighbor offsets
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbor = padded[1 + di : 1 + di + z.shape[0], 1 + dj : 1 + dj + z.shape[1]]
                diff = np.abs(center - neighbor)
                # Ignore NaN neighbors
                valid_diff = np.where(np.isnan(diff), 0.0, diff)
                max_step = np.maximum(max_step, valid_diff.astype(np.float32))

        max_step[~valid] = np.nan
        steps.append(max_step)

    return steps


def compute_clearance(
    grid: ClipmapGrid,
    default_clearance_m: float = 10.0,
) -> list[NDArray[np.float32]]:
    """Compute overhead clearance (overhang_z - ground_z) for each cell.

    Parameters
    ----------
    grid : ClipmapGrid
        Populated FoveaMap clipmap grid.
    default_clearance_m : float, default 10.0
        Clearance assigned when no overhead points are present in the column.

    Returns
    -------
    clearance : list[NDArray[float32]]
        Per-ring 2D arrays of vertical clearance in metres.
    """
    clearances: list[NDArray[np.float32]] = []

    for r_idx in range(grid.n_rings):
        gz = grid.layer("ground_z", r_idx)
        oz = grid.layer("overhang_z", r_idx)
        valid_ground = ~np.isnan(gz)

        # Where overhang exists: overhang_z - ground_z
        # Where no overhang exists: default_clearance_m (clear to sky)
        has_overhang = ~np.isnan(oz) & valid_ground
        clr = np.full_like(gz, default_clearance_m, dtype=np.float32)
        clr[has_overhang] = oz[has_overhang] - gz[has_overhang]

        # Cells with no ground observed remain NaN
        clr[~valid_ground] = np.nan
        clearances.append(clr)

    return clearances


def compute_traversability(
    grid: ClipmapGrid,
    max_slope_deg: float = 15.0,
    max_step_m: float = 0.10,
    min_clearance_m: float = 2.0,
) -> list[NDArray[np.bool_]]:
    """Compute boolean vehicle passability mask across all rings.

    Criteria for traversability:
    1. Class is DRIVABLE (0).
    2. Cell has at least 1 point (count > 0).
    3. Ground slope < max_slope_deg (default 15°).
    4. Step height < max_step_m (default 10 cm).
    5. Headway clearance ≥ min_clearance_m (default 2.0 m).

    Parameters
    ----------
    grid : ClipmapGrid
        Populated FoveaMap clipmap grid.
    max_slope_deg : float, default 15.0
        Maximum traversable slope angle in degrees.
    max_step_m : float, default 0.10
        Maximum passable vertical step height in metres.
    min_clearance_m : float, default 2.0
        Minimum vehicle headway clearance in metres.

    Returns
    -------
    traversable : list[NDArray[bool]]
        Per-ring 2D boolean masks. True for safe passable cells.
    """
    slopes = compute_slope(grid)
    steps = compute_step_height(grid)
    clearances = compute_clearance(grid)

    masks: list[NDArray[np.bool_]] = []

    for r_idx in range(grid.n_rings):
        cls_layer = grid.layer("cls", r_idx)
        cnt_layer = grid.layer("count", r_idx)

        slope = slopes[r_idx]
        step = steps[r_idx]
        clr = clearances[r_idx]

        is_drivable = (cls_layer == DRIVABLE)
        is_observed = (cnt_layer > 0)
        safe_slope = np.nan_to_num(slope, nan=999.0) <= max_slope_deg
        safe_step = np.nan_to_num(step, nan=999.0) <= max_step_m
        safe_clearance = np.nan_to_num(clr, nan=0.0) >= min_clearance_m

        trav = is_drivable & is_observed & safe_slope & safe_step & safe_clearance
        masks.append(trav)

    return masks


def compute_derived_layers(
    grid: ClipmapGrid,
    max_slope_deg: float = 15.0,
    max_step_m: float = 0.10,
    min_clearance_m: float = 2.0,
) -> DerivedLayers:
    """Compute all derived geometric layers in a single pass.

    Parameters
    ----------
    grid : ClipmapGrid
        Populated FoveaMap clipmap grid.
    max_slope_deg : float, default 15.0
        Slope threshold in degrees.
    max_step_m : float, default 0.10
        Step height threshold in metres.
    min_clearance_m : float, default 2.0
        Clearance threshold in metres.

    Returns
    -------
    DerivedLayers
        Container with slope, step_height, clearance, and traversability masks.
    """
    slopes = compute_slope(grid)
    steps = compute_step_height(grid)
    clearances = compute_clearance(grid)

    trav_masks: list[NDArray[np.bool_]] = []
    for r_idx in range(grid.n_rings):
        cls_layer = grid.layer("cls", r_idx)
        cnt_layer = grid.layer("count", r_idx)

        is_drivable = (cls_layer == DRIVABLE)
        is_observed = (cnt_layer > 0)
        safe_slope = np.nan_to_num(slopes[r_idx], nan=999.0) <= max_slope_deg
        safe_step = np.nan_to_num(steps[r_idx], nan=999.0) <= max_step_m
        safe_clearance = np.nan_to_num(clearances[r_idx], nan=0.0) >= min_clearance_m

        trav = is_drivable & is_observed & safe_slope & safe_step & safe_clearance
        trav_masks.append(trav)

    return DerivedLayers(
        slope_deg=slopes,
        step_height_m=steps,
        clearance_m=clearances,
        traversable=trav_masks,
    )
