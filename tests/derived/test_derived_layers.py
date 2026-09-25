"""Tests for derived geometric layers: slope, step, clearance, traversability (README 6.7, T11.2 - T11.6)."""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.config import DerivedConfig
from foveamap.derived.clearance import compute_clearance
from foveamap.derived.halo import compute_halo
from foveamap.derived.slope import compute_slope, compute_slope_deg
from foveamap.derived.step import compute_step, compute_step_height_mm
from foveamap.derived.traversability import (
    TRAVERSABLE_CLASS,
    NON_TRAVERSABLE_CLASS,
    UNKNOWN_CLASS,
    compute_traversability,
    traversability_map,
)
from foveamap.grid.layers import (
    EMPTY_CELL,
    FLAG_HAS_GROUND,
    FLAG_HAS_OBSTACLE,
    FLAG_HAS_OVERHANG,
    FLAG_KERB,
    FLAG_LOW_CLEARANCE,
    FLAG_STEEP,
    FLAG_TRAVERSABLE,
    LAYER_DTYPE,
    GridLayers,
)
from foveamap.grid.presets import spec_from_rings
from foveamap.io.labels import DRIVABLE, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, UNKNOWN


@pytest.fixture
def one_ring_spec():
    # Ring 0: [-5000, 5000) mm, cell 100 mm -> side 100
    return spec_from_rings("one_ring_test", [(5000, 100)])


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


def test_slope_mathematical_gradient(one_ring_spec, derived_cfg):
    spec = one_ring_spec
    side = spec.rings[0].side
    ring = np.empty((side, side), dtype=LAYER_DTYPE)
    ring[:] = EMPTY_CELL

    # 45-degree ramp in X: z = x mm (dz/dx = 1.0 -> 45 deg)
    xs = (np.arange(side) - spec.rings[0].offset) * 100 + 50
    xx, _ = np.meshgrid(xs, xs)

    ring["ground_z"] = xx
    ring["cls"] = DRIVABLE
    ring["count"] = 5
    ring["flags"] = FLAG_HAS_GROUND

    layers = GridLayers(spec=spec, rings=[ring])
    layers_out = compute_slope(layers, derived_cfg)
    slopes = compute_slope_deg(layers, derived_cfg)[0]

    # Interior cells (avoid outer borders) should have ~45 degrees
    interior = slopes[10:90, 10:90]
    np.testing.assert_allclose(interior, 45.0, atol=0.5)

    # 45 deg > 15 deg max_slope -> FLAG_STEEP must be set
    flags_interior = layers_out.rings[0]["flags"][10:90, 10:90]
    assert np.all((flags_interior & FLAG_STEEP) != 0)


def test_step_kerb_detector(one_ring_spec, derived_cfg):
    spec = one_ring_spec
    side = spec.rings[0].side
    ring = np.empty((side, side), dtype=LAYER_DTYPE)
    ring[:] = EMPTY_CELL

    # Flat ground at z = 0 for x < 0, z = 120 mm (12 cm kerb) for x >= 0
    xs = (np.arange(side) - spec.rings[0].offset) * 100 + 50
    xx, _ = np.meshgrid(xs, xs)

    ring["ground_z"] = np.where(xx < 0, 0, 120)
    ring["cls"] = DRIVABLE
    ring["count"] = 5
    ring["flags"] = FLAG_HAS_GROUND

    layers = GridLayers(spec=spec, rings=[ring])
    layers_out = compute_step(layers, derived_cfg)
    steps = compute_step_height_mm(layers, derived_cfg)[0]

    # At the step boundary (x ~ 0, col = offset - 1 and offset)
    col_left = spec.rings[0].offset - 1
    col_right = spec.rings[0].offset

    # Both columns adjacent to boundary record 120 mm step
    assert np.all(steps[10:90, col_left] == 120)
    assert np.all(steps[10:90, col_right] == 120)

    # Kerb is in [60, 250] mm -> FLAG_KERB must be set on boundary cells
    flags_bnd = layers_out.rings[0]["flags"][10:90, col_left]
    assert np.all((flags_bnd & FLAG_KERB) != 0)

    # Far interior cells have 0 step and no FLAG_KERB
    assert np.all(steps[10:90, 10] == 0)
    assert np.all((layers_out.rings[0]["flags"][10:90, 10] & FLAG_KERB) == 0)


def test_clearance_under_overhang(one_ring_spec, derived_cfg):
    spec = one_ring_spec
    side = spec.rings[0].side
    ring = np.empty((side, side), dtype=LAYER_DTYPE)
    ring[:] = EMPTY_CELL

    ring["ground_z"] = 0
    ring["cls"] = DRIVABLE
    ring["count"] = 5
    ring["flags"] = FLAG_HAS_GROUND

    # Cell with low overhang: clearance = 1800 mm (< 2200 mm vehicle + margin)
    ring["clearance"][20, 20] = 1800
    ring["flags"][20, 20] |= (FLAG_HAS_OBSTACLE | FLAG_HAS_OVERHANG)

    # Cell with high overhang: clearance = 3500 mm (>= 2200 mm)
    ring["clearance"][20, 30] = 3500
    ring["flags"][20, 30] |= (FLAG_HAS_OBSTACLE | FLAG_HAS_OVERHANG)

    layers = GridLayers(spec=spec, rings=[ring])
    layers_out = compute_clearance(layers, derived_cfg)

    # Low overhang cell has FLAG_LOW_CLEARANCE set
    assert (layers_out.rings[0]["flags"][20, 20] & FLAG_LOW_CLEARANCE) != 0

    # High overhang cell does not have FLAG_LOW_CLEARANCE set
    assert (layers_out.rings[0]["flags"][20, 30] & FLAG_LOW_CLEARANCE) == 0


def test_traversability_logic(one_ring_spec, derived_cfg):
    spec = one_ring_spec
    side = spec.rings[0].side
    ring = np.empty((side, side), dtype=LAYER_DTYPE)
    ring[:] = EMPTY_CELL

    # Cell 0: Perfect flat drivable -> Traversable
    ring[10, 10]["cls"] = DRIVABLE
    ring[10, 10]["ground_z"] = 0
    ring[10, 10]["count"] = 5
    ring[10, 10]["flags"] = FLAG_HAS_GROUND

    # Cell 1: Steep slope flag set -> Non-traversable
    ring[10, 20]["cls"] = DRIVABLE
    ring[10, 20]["ground_z"] = 0
    ring[10, 20]["count"] = 5
    ring[10, 20]["flags"] = FLAG_HAS_GROUND | FLAG_STEEP

    # Cell 2: Kerb flag set -> Non-traversable
    ring[10, 30]["cls"] = DRIVABLE
    ring[10, 30]["ground_z"] = 0
    ring[10, 30]["count"] = 5
    ring[10, 30]["flags"] = FLAG_HAS_GROUND | FLAG_KERB

    # Cell 3: Obstacle present -> Non-traversable
    ring[10, 40]["cls"] = STATIC_OBSTACLE
    ring[10, 40]["ground_z"] = 0
    ring[10, 40]["count"] = 5
    ring[10, 40]["flags"] = FLAG_HAS_GROUND | FLAG_HAS_OBSTACLE

    # Cell 4: Empty / No ground -> Unknown
    # ring[10, 50] remains EMPTY_CELL

    layers = GridLayers(spec=spec, rings=[ring])
    layers_out = compute_traversability(layers, derived_cfg)
    trav_cat = traversability_map(layers_out)[0]

    assert (layers_out.rings[0]["flags"][10, 10] & FLAG_TRAVERSABLE) != 0
    assert trav_cat[10, 10] == TRAVERSABLE_CLASS

    assert (layers_out.rings[0]["flags"][10, 20] & FLAG_TRAVERSABLE) == 0
    assert trav_cat[10, 20] == NON_TRAVERSABLE_CLASS

    assert (layers_out.rings[0]["flags"][10, 30] & FLAG_TRAVERSABLE) == 0
    assert trav_cat[10, 30] == NON_TRAVERSABLE_CLASS

    assert (layers_out.rings[0]["flags"][10, 40] & FLAG_TRAVERSABLE) == 0
    assert trav_cat[10, 40] == NON_TRAVERSABLE_CLASS

    assert (layers_out.rings[0]["flags"][10, 50] & FLAG_TRAVERSABLE) == 0
    assert trav_cat[10, 50] == UNKNOWN_CLASS
