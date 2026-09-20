"""tests.test_derive — Verification suite for derived geometric safety layers.

Verifies:
1. `compute_slope` returns the exact mathematical gradient (e.g. 45° slope -> 1.0 gradient magnitude and 45.0°).
2. `compute_step_height` highlights boundary cells of a vertical step discontinuity.
3. `compute_clearance` computes overhead clearance under low obstacle overhangs.
4. `compute_traversability` safely masks out steep slopes, steps, and low clearances.
5. `compute_derived_layers` returns the complete DerivedLayers container.
"""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.derive.layers import (
    DerivedLayers,
    compute_clearance,
    compute_derived_layers,
    compute_slope,
    compute_step_height,
    compute_traversability,
)
from foveamap.grid.clipmap import ClipmapGrid
from foveamap.grid.spec import GridSpec, Ring, load_spec_from_preset
from foveamap.io.labels import DRIVABLE, STATIC_OBSTACLE


@pytest.fixture
def ring0_spec() -> GridSpec:
    """1-ring test specification for fast, exact coordinate evaluation."""
    return GridSpec(
        rings=(Ring(cell_m=0.10, half_extent_m=6.0),),
        z_range_m=(-3.0, 5.0),
    )


class TestDerivedLayers:
    """Test suite for derived geometric safety layers."""

    def test_compute_slope_mathematical_gradient(self, ring0_spec: GridSpec) -> None:
        """Verify compute_slope on a 45-degree ramp (dz/dx = 1.0) and a flat plane."""
        # Sample points at cell centers (cell_m = 0.10m, centers at k * 0.10 + 0.05)
        xs = np.arange(-4.0 + 0.05, 4.0, 0.10, dtype=np.float32)
        ys = np.arange(-4.0 + 0.05, 4.0, 0.10, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        zz = xx.copy()  # z = x -> dz/dx = 1.0, dz/dy = 0.0 -> tan(theta) = 1.0 -> theta = 45 deg

        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel(), np.zeros(xx.size, dtype=np.float32)])
        n = len(pts)
        sc = np.full(n, DRIVABLE, dtype=np.uint8)
        mv = np.zeros(n, dtype=np.bool_)
        vru = np.zeros(n, dtype=np.bool_)
        cf = np.ones(n, dtype=np.float16)

        grid = ClipmapGrid.build(pts, sc, mv, vru, cf, ring0_spec)
        slopes = compute_slope(grid)
        slope_r0 = slopes[0]

        # Extract interior cells well within [-3.0, 3.0] where data is continuous
        cx = ring0_spec.rings[0].half_extent_m
        res = ring0_spec.rings[0].cell_m
        i_min = int((-3.0 + cx) / res)
        i_max = int((3.0 + cx) / res)
        interior = slope_r0[i_min:i_max, i_min:i_max]

        # Assert slope angle is 45 degrees
        np.testing.assert_allclose(
            interior,
            45.0,
            atol=0.5,
            err_msg="Interior cells of z=x ramp must have 45.0 degree slope",
        )

        # Assert mathematical gradient magnitude is 1.0
        grad_mag = np.tan(np.radians(interior))
        np.testing.assert_allclose(
            grad_mag,
            1.0,
            atol=0.02,
            err_msg="Mathematical gradient tan(slope) must equal 1.0",
        )

        # 2. Perfectly flat plane (z = 0)
        zz_flat = np.zeros_like(xx)
        pts_flat = np.column_stack([xx.ravel(), yy.ravel(), zz_flat.ravel(), np.zeros(xx.size, dtype=np.float32)])
        grid_flat = ClipmapGrid.build(pts_flat, sc, mv, vru, cf, ring0_spec)
        slopes_flat = compute_slope(grid_flat)
        interior_flat = slopes_flat[0][i_min:i_max, i_min:i_max]

        np.testing.assert_allclose(
            interior_flat,
            0.0,
            atol=1e-5,
            err_msg="Flat plane interior slope must be 0.0 degrees",
        )

    def test_compute_step_height_boundary(self, ring0_spec: GridSpec) -> None:
        """Assert compute_step_height detects the boundary of a 30 cm vertical step."""
        xs = np.linspace(-4.0, 4.0, 161, dtype=np.float32)
        ys = np.linspace(-4.0, 4.0, 161, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        # Step at x = 0: z = 0 for x < 0; z = 0.30 for x >= 0
        step_height = 0.30
        zz = np.where(xx < 0.0, 0.0, step_height).astype(np.float32)

        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel(), np.zeros(xx.size, dtype=np.float32)])
        n = len(pts)
        sc = np.full(n, DRIVABLE, dtype=np.uint8)
        mv = np.zeros(n, dtype=np.bool_)
        vru = np.zeros(n, dtype=np.bool_)
        cf = np.ones(n, dtype=np.float16)

        grid = ClipmapGrid.build(pts, sc, mv, vru, cf, ring0_spec)
        steps = compute_step_height(grid)
        step_r0 = steps[0]

        cx = ring0_spec.rings[0].half_extent_m
        res = ring0_spec.rings[0].cell_m

        # Column index corresponding to x = 0
        ix_boundary = int((0.0 + cx) / res)

        # Boundary cells (columns adjacent to boundary) should register ~0.30m step
        boundary_col = step_r0[int((-2.0 + cx) / res) : int((2.0 + cx) / res), ix_boundary]
        np.testing.assert_allclose(
            boundary_col,
            step_height,
            atol=0.01,
            err_msg="Boundary cells at step must record 0.30m step height",
        )

        # Interior cells far from step (x = -2.0m) should have 0.0m step height
        ix_interior_left = int((-2.0 + cx) / res)
        interior_col = step_r0[int((-2.0 + cx) / res) : int((2.0 + cx) / res), ix_interior_left]
        np.testing.assert_allclose(
            interior_col,
            0.0,
            atol=1e-5,
            err_msg="Flat interior cells must have 0.0m step height",
        )

    def test_compute_clearance_under_overhang(self, ring0_spec: GridSpec) -> None:
        """Clearance under overhang is overhang_z - ground_z; sky cells have default clearance."""
        # Ground at Z = 0
        xs = np.linspace(-3.0, 3.0, 121, dtype=np.float32)
        ys = np.linspace(-3.0, 3.0, 121, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.zeros_like(xx)

        ground_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel(), np.zeros(xx.size, dtype=np.float32)])
        n_ground = len(ground_pts)
        ground_sc = np.full(n_ground, DRIVABLE, dtype=np.uint8)

        # Overhang points at (x in [0, 1], y in [-1, 1]) at Z = 1.5m
        oxs = np.linspace(0.0, 1.0, 25, dtype=np.float32)
        oys = np.linspace(-1.0, 1.0, 25, dtype=np.float32)
        oxx, oyy = np.meshgrid(oxs, oys)
        ozz = np.full_like(oxx, 1.5)
        overhang_pts = np.column_stack([oxx.ravel(), oyy.ravel(), ozz.ravel(), np.zeros(oxx.size, dtype=np.float32)])
        n_overhang = len(overhang_pts)
        overhang_sc = np.full(n_overhang, STATIC_OBSTACLE, dtype=np.uint8)

        all_pts = np.vstack([ground_pts, overhang_pts])
        all_sc = np.concatenate([ground_sc, overhang_sc])
        all_mv = np.zeros(len(all_pts), dtype=np.bool_)
        all_vru = np.zeros(len(all_pts), dtype=np.bool_)
        all_cf = np.ones(len(all_pts), dtype=np.float16)

        grid = ClipmapGrid.build(all_pts, all_sc, all_mv, all_vru, all_cf, ring0_spec)
        clearances = compute_clearance(grid, default_clearance_m=10.0)
        clr_r0 = clearances[0]

        cx = ring0_spec.rings[0].half_extent_m
        res = ring0_spec.rings[0].cell_m

        # Cells under overhang (e.g. x=0.5, y=0.0) should have clearance ~ 1.5m
        ix_oh = int((0.5 + cx) / res)
        iy_oh = int((0.0 + cx) / res)
        assert np.isclose(clr_r0[iy_oh, ix_oh], 1.5, atol=0.05)

        # Open-sky cells (e.g. x=-2.0, y=0.0) should have default clearance 10.0m
        ix_sky = int((-2.0 + cx) / res)
        iy_sky = int((0.0 + cx) / res)
        assert np.isclose(clr_r0[iy_sky, ix_sky], 10.0, atol=1e-5)

    def test_compute_traversability_masks_hazards(self, ring0_spec: GridSpec) -> None:
        """Traversability mask allows flat drivable ground but rejects steep slope, step, and low clearance."""
        # Create 3 distinct zones:
        # Zone A: Flat drivable (x in [-4, -2]) -> should be Traversable (True)
        # Zone B: Steep 45° slope (x in [-1, 1], z = 1.0 * x) -> slope > 15° -> Non-traversable (False)
        # Zone C: Flat with 30cm step boundary (x in [2, 4], step at x=3.0) -> step > 0.10m -> Non-traversable (False)
        xs = np.linspace(-4.0, 4.0, 201, dtype=np.float32)
        ys = np.linspace(-2.0, 2.0, 101, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        zz = np.zeros_like(xx)
        # Apply 45 deg slope in x in [-1, 1]
        mask_slope = (xx >= -1.0) & (xx <= 1.0)
        zz[mask_slope] = 1.0 * (xx[mask_slope] - (-1.0))
        # Keep elevated plateau after x=1.0 up to 2.0
        mask_plateau = (xx > 1.0) & (xx < 2.5)
        zz[mask_plateau] = 2.0
        # Vertical 30cm step at x = 3.0 (from 0 to 0.30)
        mask_step_left = (xx >= 2.5) & (xx < 3.0)
        zz[mask_step_left] = 0.0
        mask_step_right = xx >= 3.0
        zz[mask_step_right] = 0.30

        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel(), np.zeros(xx.size, dtype=np.float32)])
        n = len(pts)
        sc = np.full(n, DRIVABLE, dtype=np.uint8)
        mv = np.zeros(n, dtype=np.bool_)
        vru = np.zeros(n, dtype=np.bool_)
        cf = np.ones(n, dtype=np.float16)

        grid = ClipmapGrid.build(pts, sc, mv, vru, cf, ring0_spec)
        traversable_masks = compute_traversability(
            grid, max_slope_deg=15.0, max_step_m=0.10, min_clearance_m=2.0
        )
        trav_r0 = traversable_masks[0]

        cx = ring0_spec.rings[0].half_extent_m
        res = ring0_spec.rings[0].cell_m

        # Zone A: Flat drivable (x = -3.0, y = 0.0) -> MUST be Traversable
        ix_flat = int((-3.0 + cx) / res)
        iy_mid = int((0.0 + cx) / res)
        assert bool(trav_r0[iy_mid, ix_flat]) is True

        # Zone B: Steep 45° slope (x = 0.0, y = 0.0) -> MUST NOT be Traversable
        ix_slope = int((0.0 + cx) / res)
        assert bool(trav_r0[iy_mid, ix_slope]) is False

        # Zone C: Step boundary (x = 3.0, y = 0.0) -> MUST NOT be Traversable
        ix_step = int((3.0 + cx) / res)
        assert bool(trav_r0[iy_mid, ix_step]) is False

    def test_compute_derived_layers_container(self) -> None:
        """compute_derived_layers returns DerivedLayers container with all 4 rings."""
        spec = load_spec_from_preset("fovea_4ring")
        pts = np.zeros((100, 4), dtype=np.float32)
        sc = np.full(100, DRIVABLE, dtype=np.uint8)
        mv = np.zeros(100, dtype=np.bool_)
        vru = np.zeros(100, dtype=np.bool_)
        cf = np.ones(100, dtype=np.float16)

        grid = ClipmapGrid.build(pts, sc, mv, vru, cf, spec)
        derived = compute_derived_layers(grid)

        assert isinstance(derived, DerivedLayers)
        assert derived.n_rings == 4
        assert len(derived.slope_deg) == 4
        assert len(derived.step_height_m) == 4
        assert len(derived.clearance_m) == 4
        assert len(derived.traversable) == 4
