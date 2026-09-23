"""T5.2: integer-only addressing — invariants I2 (exact partition) and I5 (round trip), README 6.5.3."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from foveamap.grid.engine import (
    cell_to_center_mm,
    cell_to_corner_mm,
    is_logical_cell,
    locate,
    quantize_mm,
    world_to_cell,
)
from foveamap.grid.presets import GridSpec

from .conftest import adversarial_points_mm

PRESET_NAMES = ["tiny_test", "tiny_ps_test", "fovea_default", "ps_literal", "uniform_20cm"]


def smallest_ring(spec: GridSpec, x: int, y: int) -> int | None:
    """Independent restatement of L7."""
    for k, ring in enumerate(spec.rings):
        r = ring.r_max_mm
        if -r <= x < r and -r <= y < r:
            return k
    return None


def check_point(spec: GridSpec, x: int, y: int) -> None:
    cell = world_to_cell(spec, x, y)
    expected_ring = smallest_ring(spec, x, y)
    if expected_ring is None:
        assert cell is None
        return
    assert cell is not None
    ring, ix, iy = cell
    assert ring == expected_ring
    s = spec.rings[ring].cell_mm
    cx, cy = cell_to_corner_mm(spec, ring, ix, iy)
    assert cx <= x < cx + s and cy <= y < cy + s  # the point lies in its own cell
    assert 0 <= ix < spec.rings[ring].side and 0 <= iy < spec.rings[ring].side
    assert is_logical_cell(spec, ring, ix, iy)  # never inside the ring's hole


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_adversarial_points_partition(presets: dict[str, GridSpec], name: str) -> None:
    spec = presets[name]
    for x, y in adversarial_points_mm(spec).tolist():
        check_point(spec, x, y)


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_vectorised_locate_matches_scalar(presets: dict[str, GridSpec], name: str) -> None:
    spec = presets[name]
    pts = adversarial_points_mm(spec)
    ring, ix, iy = locate(spec, pts[:, 0], pts[:, 1])
    assert ring.dtype == np.int8 and ix.dtype == np.int64
    for (x, y), r, i, j in zip(pts.tolist(), ring.tolist(), ix.tolist(), iy.tolist(), strict=True):
        cell = world_to_cell(spec, x, y)
        assert (cell is None and r == -1 and i == -1 and j == -1) or cell == (r, i, j)


@settings(max_examples=300, derandomize=True)
@given(st.sampled_from(PRESET_NAMES), st.integers(-130_000, 130_000), st.integers(-130_000, 130_000))
def test_random_points_partition(presets: dict[str, GridSpec], name: str, x: int, y: int) -> None:
    spec = presets[name]
    scale = spec.extent_mm / 100_000  # keep the tiny presets' out-of-grid fraction similar
    check_point(spec, int(x * scale), int(y * scale))


def test_bulk_random_points_partition(presets: dict[str, GridSpec]) -> None:
    rng = np.random.default_rng(0)
    for name in PRESET_NAMES:
        spec = presets[name]
        r = spec.extent_mm
        pts = rng.integers(-r - 5_000, r + 5_000, size=(20_000, 2))
        ring, ix, iy = locate(spec, pts[:, 0], pts[:, 1])
        inside = ((pts >= -r) & (pts < r)).all(axis=1)  # half-open [-R, R)^2 (L4)
        np.testing.assert_array_equal(ring >= 0, inside)
        for k, rs in enumerate(spec.rings):
            sel = ring == k
            corner_x = (ix[sel] - rs.offset) * rs.cell_mm
            corner_y = (iy[sel] - rs.offset) * rs.cell_mm
            assert ((corner_x <= pts[sel, 0]) & (pts[sel, 0] < corner_x + rs.cell_mm)).all()
            assert ((corner_y <= pts[sel, 1]) & (pts[sel, 1] < corner_y + rs.cell_mm)).all()


@pytest.mark.parametrize("name", ["tiny_test", "tiny_ps_test"])
def test_round_trip_every_logical_cell(presets: dict[str, GridSpec], name: str) -> None:
    spec = presets[name]
    n_logical = 0
    for k, ring in enumerate(spec.rings):
        s = ring.cell_mm
        for ix in range(ring.side):
            for iy in range(ring.side):
                if not is_logical_cell(spec, k, ix, iy):
                    continue
                n_logical += 1
                corner = cell_to_corner_mm(spec, k, ix, iy)
                center = cell_to_center_mm(spec, k, ix, iy)
                assert world_to_cell(spec, *corner) == (k, ix, iy)
                assert world_to_cell(spec, *center) == (k, ix, iy)
                last = (corner[0] + s - 1, corner[1] + s - 1)  # last millimetre inside the cell
                assert world_to_cell(spec, *last) == (k, ix, iy)
                assert center == (corner[0] + s // 2, corner[1] + s // 2)
    assert n_logical == spec.logical_cells()


def test_cell_corners_are_multiples_of_cell_size(presets: dict[str, GridSpec]) -> None:
    spec = presets["fovea_default"]
    for k, ring in enumerate(spec.rings):
        for ix, iy in [(0, 0), (ring.side - 1, 0), (ring.offset, ring.offset), (7, ring.side - 3)]:
            cx, cy = cell_to_corner_mm(spec, k, ix, iy)
            assert cx % ring.cell_mm == 0 and cy % ring.cell_mm == 0  # I3: corners on the lattice


def test_origin_and_negative_zero(presets: dict[str, GridSpec]) -> None:
    spec = presets["fovea_default"]
    assert world_to_cell(spec, 0, 0) == (0, 200, 200)
    assert world_to_cell(spec, -1, -1) == (0, 199, 199)  # floor division, not truncation
    xyz = quantize_mm(np.array([[-0.0, 0.0, -0.0004]]))
    assert xyz.tolist() == [[0, 0, 0]]


def test_quantize_mm() -> None:
    xyz = np.array([[1.0004, -2.0006, 0.05], [-0.0016, 0.0026, 32.767]], dtype=np.float64)
    q = quantize_mm(xyz)
    assert q.dtype == np.int32
    assert q.tolist() == [[1000, -2001, 50], [-2, 3, 32767]]
    np.testing.assert_array_equal(quantize_mm(xyz.astype(np.float32))[0], [1000, -2001, 50])
    with pytest.raises(ValueError, match="finite"):
        quantize_mm(np.array([[np.nan, 0, 0]]))
    with pytest.raises(ValueError, match="int32"):
        quantize_mm(np.array([[3e6, 0, 0]]))


def test_logical_mask_matches_cell_test_and_counts(presets: dict[str, GridSpec]) -> None:
    from foveamap.grid.engine import logical_mask

    for name in ("tiny_test", "tiny_ps_test", "fovea_default", "ps_literal"):
        spec = presets[name]
        counts = [int(logical_mask(spec, k).sum()) for k in range(len(spec.rings))]
        assert counts == spec.logical_cells_per_ring(), name
    spec = presets["tiny_test"]
    for k, ring in enumerate(spec.rings):
        mask = logical_mask(spec, k)
        for ix in range(0, ring.side, 7):
            for iy in range(0, ring.side, 5):
                assert mask[iy, ix] == is_logical_cell(spec, k, ix, iy)
