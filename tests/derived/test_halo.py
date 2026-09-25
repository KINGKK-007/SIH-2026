"""Tests for derived halo computation (README 6.7, task T11.1)."""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.config import DerivedConfig
from foveamap.derived.halo import compute_halo, extract_padded_ring
from foveamap.grid.layers import (
    EMPTY_CELL,
    FLAG_HAS_GROUND,
    LAYER_DTYPE,
    GridLayers,
)
from foveamap.grid.presets import spec_from_rings
from foveamap.io.labels import DRIVABLE


@pytest.fixture
def two_ring_spec():
    # Ring 0: [-2000, 2000) mm, cell 100 mm -> side 40
    # Ring 1: [-6000, 6000) mm, cell 200 mm -> side 60, inner hole [-2000, 2000) mm (side 20)
    return spec_from_rings("two_ring_test", [(2000, 100), (6000, 200)])


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


def test_halo_fills_inner_hole_from_fine_ring(two_ring_spec, derived_cfg):
    spec = two_ring_spec
    r0_side = spec.rings[0].side  # 40
    r1_side = spec.rings[1].side  # 60

    r0 = np.empty((r0_side, r0_side), dtype=LAYER_DTYPE)
    r0[:] = EMPTY_CELL
    r1 = np.empty((r1_side, r1_side), dtype=LAYER_DTYPE)
    r1[:] = EMPTY_CELL

    # Populate ring 0 with a constant ground height = 500 mm
    r0["ground_z"] = 500
    r0["cls"] = DRIVABLE
    r0["count"] = 5
    r0["flags"] = FLAG_HAS_GROUND

    # Ring 1 annulus is populated with 500 mm, but hole is EMPTY_CELL
    r1["ground_z"] = 500
    r1["cls"] = DRIVABLE
    r1["count"] = 5
    r1["flags"] = FLAG_HAS_GROUND

    # Clear ring 1's hole (x in [-2000, 2000), y in [-2000, 2000))
    # In ring 1: offset is 30, cell is 200 mm -> hole is [20:40, 20:40]
    r1[20:40, 20:40] = EMPTY_CELL
    assert (r1[20:40, 20:40]["flags"] & FLAG_HAS_GROUND).sum() == 0

    layers = GridLayers(spec=spec, rings=[r0, r1])
    halo_layers = compute_halo(layers, derived_cfg)

    # After compute_halo, ring 1's hole should be filled with ground_z=500 from ring 0
    hole_ground = halo_layers.rings[1]["ground_z"][20:40, 20:40]
    assert np.all(hole_ground == 500)
    assert np.all(halo_layers.rings[1]["flags"][20:40, 20:40] & FLAG_HAS_GROUND)


def test_extract_padded_ring_coarse_to_fine(two_ring_spec, derived_cfg):
    spec = two_ring_spec
    r0_side = spec.rings[0].side  # 40
    r1_side = spec.rings[1].side  # 60

    r0 = np.empty((r0_side, r0_side), dtype=LAYER_DTYPE)
    r0[:] = EMPTY_CELL
    r1 = np.empty((r1_side, r1_side), dtype=LAYER_DTYPE)
    r1[:] = EMPTY_CELL

    r0["ground_z"] = 100
    r0["flags"] = FLAG_HAS_GROUND
    r1["ground_z"] = 200
    r1["flags"] = FLAG_HAS_GROUND

    layers = GridLayers(spec=spec, rings=[r0, r1])

    # Extract padded ring 0 (shape 42, 42)
    padded0 = extract_padded_ring(layers, ring_idx=0, cfg=derived_cfg)
    assert padded0.shape == (42, 42)

    # Interior [1:-1, 1:-1] matches ring 0
    assert np.all(padded0["ground_z"][1:-1, 1:-1] == 100)

    # Outer border is filled from coarser ring 1 (ground_z == 200)
    assert np.all(padded0["ground_z"][0, :] == 200)
    assert np.all(padded0["ground_z"][-1, :] == 200)
    assert np.all(padded0["ground_z"][:, 0] == 200)
    assert np.all(padded0["ground_z"][:, -1] == 200)
