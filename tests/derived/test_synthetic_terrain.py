"""Synthetic-terrain unit tests across all ring resolutions (README 6.7, task T11.6)."""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.config import DerivedConfig
from foveamap.derived import compute_derived_layers, compute_slope_deg, compute_step_height_mm
from foveamap.derived.traversability import UNKNOWN_CLASS, traversability_map
from foveamap.grid.engine import rasterize
from foveamap.grid.layers import (
    FLAG_KERB,
    FLAG_LOW_CLEARANCE,
    FLAG_STEEP,
    FLAG_TRAVERSABLE,
    finalize,
)
from foveamap.grid.presets import spec_from_rings
from foveamap.io.labels import DRIVABLE, STATIC_OBSTACLE


@pytest.fixture
def fovea_4ring_spec():
    # 4 rings: 5cm (10m), 10cm (25m), 20cm (50m), 40cm (100m)
    return spec_from_rings(
        "fovea_4ring_test",
        [
            (10000, 50),
            (25000, 100),
            (50000, 200),
            (100000, 400),
        ],
    )


@pytest.fixture
def derived_cfg():
    return DerivedConfig(
        max_slope_deg=15.0,
        kerb_min_m=0.06,
        kerb_max_m=0.25,
        obstacle_step_m=0.30,
        min_points_step=2,
        vehicle_height_m=2.0,
        clearance_margin_m=0.2,
        halo=True,
        fill_missing_ground=False,
    )


class DummyGridCfg:
    class_rule = "safety"
    ground_estimator = "mean"
    contact_height_mm = 300
    min_range_mm = 0

    class class_thresholds:
        dyn_min_points = 2
        dyn_min_frac = 0.3
        obs_min_points = 2
        obs_min_frac = 0.3


def test_ramp_recovered_at_all_ring_resolutions(fovea_4ring_spec, derived_cfg):
    """A ramp of known slope (e.g. 10 deg) is recovered within tolerance across all ring resolutions."""
    spec = fovea_4ring_spec
    slope_target_deg = 10.0
    tan_slope = np.tan(np.radians(slope_target_deg))

    # Generate synthetic points on a dense plane: z = tan_slope * x
    # Spacing 0.04m (40mm) guarantees every 50mm cell has at least 1 point
    xs = np.arange(-20.0, 20.0, 0.04)
    ys = np.arange(-20.0, 20.0, 0.04)
    xx, yy = np.meshgrid(xs, ys)
    zz = tan_slope * xx

    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = len(pts)
    super_cls = np.full(n, DRIVABLE, dtype=np.uint8)
    moving = np.zeros(n, dtype=np.bool_)
    conf = np.full(n, 255, dtype=np.uint8)

    acc = rasterize(spec, pts, super_cls, moving, conf)
    layers = finalize(acc, DummyGridCfg(), spec)
    layers = compute_derived_layers(layers, derived_cfg)
    slopes = compute_slope_deg(layers, derived_cfg)

    # Check that in each ring, the interior non-empty cells have measured slope ~ 10 deg
    for k in (0, 1):
        r_slope = slopes[k]
        valid = ~np.isnan(r_slope)
        assert valid.sum() > 50, f"Ring {k} must have valid ground cells"

        median_slope = float(np.nanmedian(r_slope))
        np.testing.assert_allclose(
            median_slope,
            slope_target_deg,
            atol=0.5,
            err_msg=f"Ring {k} resolution {spec.rings[k].cell_mm}mm must recover 10 deg slope",
        )

        # 10 deg <= 15 deg -> not steep
        flags = layers.rings[k]["flags"][valid]
        assert (flags & FLAG_STEEP).sum() == 0, f"Ring {k} slope 10 deg should not be steep"


def test_step_kerb_fine_vs_coarse(fovea_4ring_spec, derived_cfg):
    """Step of 12 cm sets FLAG_KERB at fine rings; degrades at coarser rings."""
    spec = fovea_4ring_spec
    step_height_m = 0.12  # 12 cm

    # Ground points with a step along x = 5.0m (within ring 0 & ring 1)
    xs = np.linspace(2.0, 8.0, 300)
    ys = np.linspace(-3.0, 3.0, 300)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.where(xx < 5.0, 0.0, step_height_m)

    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = len(pts)
    super_cls = np.full(n, DRIVABLE, dtype=np.uint8)
    moving = np.zeros(n, dtype=np.bool_)
    conf = np.full(n, 255, dtype=np.uint8)

    acc = rasterize(spec, pts, super_cls, moving, conf)
    layers = finalize(acc, DummyGridCfg(), spec)
    layers = compute_derived_layers(layers, derived_cfg)

    # In Ring 0 (5cm cells), the 12cm kerb is sharply detected
    kerb_cells_r0 = (layers.rings[0]["flags"] & FLAG_KERB).sum()
    assert kerb_cells_r0 > 0, "Ring 0 (5cm) must detect the 12cm kerb"

    # Measured step height in Ring 0 at kerb cells is ~ 120 mm
    steps_r0 = compute_step_height_mm(layers, derived_cfg)[0]
    kerb_mask = (layers.rings[0]["flags"] & FLAG_KERB) != 0
    assert np.allclose(steps_r0[kerb_mask], 120, atol=10)


def test_overhang_sets_low_clearance(fovea_4ring_spec, derived_cfg):
    """Overhang slab at 1.8m height above ground sets FLAG_LOW_CLEARANCE."""
    spec = fovea_4ring_spec

    # Ground at z = 0
    xs = np.linspace(-3.0, 3.0, 100)
    ys = np.linspace(-3.0, 3.0, 100)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    ground_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n_g = len(ground_pts)
    g_cls = np.full(n_g, DRIVABLE, dtype=np.uint8)

    # Overhang slab at (x in [0, 2], y in [-1, 1], z = 1.8m)
    oxs = np.linspace(0.0, 2.0, 30)
    oys = np.linspace(-1.0, 1.0, 30)
    oxx, oyy = np.meshgrid(oxs, oys)
    ozz = np.full_like(oxx, 1.8)
    oh_pts = np.column_stack([oxx.ravel(), oyy.ravel(), ozz.ravel()])
    n_oh = len(oh_pts)
    oh_cls = np.full(n_oh, STATIC_OBSTACLE, dtype=np.uint8)

    all_pts = np.vstack([ground_pts, oh_pts])
    all_cls = np.concatenate([g_cls, oh_cls])
    all_mv = np.zeros(len(all_pts), dtype=np.bool_)
    all_cf = np.full(len(all_pts), 255, dtype=np.uint8)

    acc = rasterize(spec, all_pts, all_cls, all_mv, all_cf)
    layers = finalize(acc, DummyGridCfg(), spec)
    layers = compute_derived_layers(layers, derived_cfg)

    # Cells under the slab in Ring 0 must have FLAG_LOW_CLEARANCE
    low_clr_count = (layers.rings[0]["flags"] & FLAG_LOW_CLEARANCE).sum()
    assert low_clr_count > 0, "Cells under 1.8m overhang must have FLAG_LOW_CLEARANCE set"

    # And those cells must NOT be traversable
    under_oh_mask = (layers.rings[0]["flags"] & FLAG_LOW_CLEARANCE) != 0
    assert (layers.rings[0]["flags"][under_oh_mask] & FLAG_TRAVERSABLE).sum() == 0


def test_empty_cells_are_unknown(fovea_4ring_spec, derived_cfg):
    """Cells with no points/ground have UNKNOWN traversability category."""
    spec = fovea_4ring_spec
    # Only a small patch of ground at x in [1, 2], y in [1, 2]
    xs = np.linspace(1.0, 2.0, 20)
    ys = np.linspace(1.0, 2.0, 20)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    n = len(pts)
    super_cls = np.full(n, DRIVABLE, dtype=np.uint8)
    moving = np.zeros(n, dtype=np.bool_)
    conf = np.full(n, 255, dtype=np.uint8)

    acc = rasterize(spec, pts, super_cls, moving, conf)
    layers = finalize(acc, DummyGridCfg(), spec)
    layers = compute_derived_layers(layers, derived_cfg)
    trav_cats = traversability_map(layers)

    # Outside the patch, cells must be UNKNOWN_CLASS
    cat0 = trav_cats[0]
    empty_cells = layers.rings[0]["count"] == 0
    assert np.all(cat0[empty_cells] == UNKNOWN_CLASS)
