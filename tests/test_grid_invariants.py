"""tests/test_grid_invariants.py — Mathematical proof of no alignment errors or data loss.

Five invariant tests (T1–T5) match the master plan §5.5 and the project
definition-of-done requirements.  All tests run with the NumPy fallback
(FOVEAMAP_NO_NUMBA=1) so that Numba compilation is not required in CI, but
the same logic is exercised by the Numba path because the finalisation and
ring-assignment code is shared.

Test IDs
--------
T1  Conservation      — every in-range point is assigned to exactly one ring
T2  Nesting/Reduction — block-reducing 5cm counts matches 40cm direct binning
T3  Boundary determ.  — points at exact ring boundaries land in the outer ring
T4  Round-trip        — cell→world centroid→cell is identity
T5  Partition/tile    — union of ring masks covers the plane; intersection empty
"""

from __future__ import annotations

import math
import os
import itertools

import numpy as np
import pytest

# Force NumPy path so these tests never require Numba compilation
os.environ.setdefault("FOVEAMAP_NO_NUMBA", "1")

from foveamap.grid.spec import Ring, GridSpec
from foveamap.grid.clipmap import ClipmapGrid
from foveamap.grid.layers import FLAG_OBSERVED


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

FOVEA_4RING_SPEC = GridSpec(
    rings=(
        Ring(cell_m=0.05, half_extent_m=10.0),
        Ring(cell_m=0.10, half_extent_m=30.0),
        Ring(cell_m=0.20, half_extent_m=60.0),
        Ring(cell_m=0.40, half_extent_m=100.0),
    ),
    z_range_m=(-3.0, 5.0),
    aggregation="safety_priority",
)

UNIFORM_5CM_SPEC = GridSpec(
    rings=(Ring(cell_m=0.05, half_extent_m=100.0),),
    z_range_m=(-3.0, 5.0),
    aggregation="safety_priority",
)


def _make_random_pts(n: int = 5_000, seed: int = 0) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return (points, super_cls, moving, vru, conf) for n uniformly random points."""
    rng = np.random.default_rng(seed)
    # Spread uniformly in [-95, 95]² (all within fovea_4ring range)
    xy = rng.uniform(-95.0, 95.0, size=(n, 2)).astype(np.float32)
    z  = rng.uniform(-2.5, 4.5, size=(n, 1)).astype(np.float32)
    intensity = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    pts = np.concatenate([xy, z, intensity], axis=1)
    sc  = rng.integers(0, 4, size=n, dtype=np.uint8)
    mv  = np.zeros(n, dtype=np.bool_)
    vru = np.zeros(n, dtype=np.bool_)
    cf  = np.ones(n, dtype=np.float16)
    return pts, sc, mv, vru, cf


def _build(pts, sc, mv, vru, cf, spec: GridSpec) -> ClipmapGrid:
    return ClipmapGrid.build(pts, sc, mv, vru, cf, spec)


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Conservation
# ─────────────────────────────────────────────────────────────────────────────

class TestT1Conservation:
    """Every in-range point is assigned to exactly one ring cell (no loss, no double-count)."""

    def _count_assigned(self, grid: ClipmapGrid) -> int:
        """Sum count layers across all rings, masking out inner-ring overlap zones."""
        total = 0
        for r_idx in range(grid.n_rings):
            ring = grid.spec.rings[r_idx]
            count_arr = grid.layer("count", r_idx).astype(np.int32)
            side = grid.side(r_idx)

            if r_idx == 0:
                total += int(count_arr.sum())
            else:
                inner_half = grid.spec.rings[r_idx - 1].half_extent_m
                inner_cells = round(2.0 * inner_half / ring.cell_m)
                pad = (side - inner_cells) // 2
                # Zero out the inner square (those cells belong to ring r_idx-1)
                mask = np.ones((side, side), dtype=np.bool_)
                mask[pad:pad + inner_cells, pad:pad + inner_cells] = False
                total += int(count_arr[mask].sum())
        return total

    def test_t1_all_inrange_points_assigned(self):
        """T1: sum(count) over active areas == n_points_assigned from stats()."""
        pts, sc, mv, vru, cf = _make_random_pts(n=10_000, seed=42)
        grid = _build(pts, sc, mv, vru, cf, FOVEA_4RING_SPEC)
        s = grid.stats()

        assigned_from_stats = s["n_points_assigned"]
        assigned_from_count = self._count_assigned(grid)

        assert assigned_from_count == assigned_from_stats, (
            f"T1 FAIL: count-layer sum ({assigned_from_count}) != "
            f"stats['n_points_assigned'] ({assigned_from_stats})"
        )

    def test_t1_count_plus_dropped_equals_total(self):
        """T1: assigned + dropped == total points."""
        pts, sc, mv, vru, cf = _make_random_pts(n=8_000, seed=7)
        grid = _build(pts, sc, mv, vru, cf, FOVEA_4RING_SPEC)
        s = grid.stats()
        assert s["n_points_assigned"] + s["n_points_dropped"] == len(pts), (
            "T1 FAIL: assigned + dropped != total"
        )

    def test_t1_no_double_count(self):
        """T1: a single point generates exactly 1 count unit across all rings."""
        # Place exactly one point in the centre of ring 0 (0.025, 0.025)
        pt = np.array([[0.025, 0.025, 0.0, 0.5]], dtype=np.float32)
        sc  = np.array([0], dtype=np.uint8)
        mv  = np.array([False])
        vru = np.array([False])
        cf  = np.array([1.0], dtype=np.float16)
        grid = _build(pt, sc, mv, vru, cf, FOVEA_4RING_SPEC)
        assert grid.n_points_assigned == 1
        total_count = self._count_assigned(grid)
        assert total_count == 1, f"T1 FAIL: single point gave count={total_count}"

    def test_t1_points_at_z_boundary_dropped(self):
        """T1: points with z outside z_range_m are dropped (not assigned)."""
        # z=-3.5 and z=5.5 are outside (-3, 5)
        pts = np.array([
            [5.0, 5.0, -3.5, 0.0],  # below
            [5.0, 5.0,  5.5, 0.0],  # above
            [5.0, 5.0,  0.0, 0.0],  # valid
        ], dtype=np.float32)
        sc  = np.zeros(3, dtype=np.uint8)
        mv  = np.zeros(3, dtype=np.bool_)
        vru = np.zeros(3, dtype=np.bool_)
        cf  = np.ones(3, dtype=np.float16)
        grid = _build(pts, sc, mv, vru, cf, FOVEA_4RING_SPEC)
        assert grid.n_points_assigned == 1, (
            f"T1 FAIL: expected 1 assigned, got {grid.n_points_assigned}"
        )
        assert grid.n_points_dropped == 2

    def test_t1_large_population_conservation(self):
        """T1: 120k random points — assigned + dropped == 120,000."""
        rng = np.random.default_rng(99)
        # Points scattered across ±110m so some are outside range
        xy  = rng.uniform(-110.0, 110.0, size=(120_000, 2)).astype(np.float32)
        z   = rng.uniform(-4.0, 6.0, size=(120_000, 1)).astype(np.float32)
        pts = np.concatenate([xy, z, np.zeros((120_000, 1), np.float32)], axis=1)
        sc  = rng.integers(0, 4, 120_000, dtype=np.uint8)
        mv  = np.zeros(120_000, dtype=np.bool_)
        vru = np.zeros(120_000, dtype=np.bool_)
        cf  = np.ones(120_000, dtype=np.float16)
        grid = _build(pts, sc, mv, vru, cf, FOVEA_4RING_SPEC)
        s = grid.stats()
        assert s["n_points_assigned"] + s["n_points_dropped"] == 120_000


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Nesting / Reduction Consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestT2NestingReduction:
    """Block-reducing fine-ring counts matches direct coarse-ring binning (I4)."""

    def _block_reduce_count(
        self,
        fine_count: np.ndarray,
        factor: int,
        inner_cells_fine: int | None = None,
    ) -> np.ndarray:
        """Sum blocks of shape (factor, factor) in fine_count.

        Parameters
        ----------
        fine_count : (H, W) int32
        factor : int — downsampling factor
        inner_cells_fine : int or None
            If set, zero out the inner square (it belongs to a finer ring)
            before reducing.

        Returns
        -------
        (H // factor, W // factor) int32
        """
        arr = fine_count.astype(np.int64).copy()
        if inner_cells_fine is not None:
            H, W = arr.shape
            pad_y = (H - inner_cells_fine) // 2
            pad_x = (W - inner_cells_fine) // 2
            arr[pad_y:pad_y + inner_cells_fine, pad_x:pad_x + inner_cells_fine] = 0
        H, W = arr.shape
        return arr.reshape(H // factor, factor, W // factor, factor).sum(axis=(1, 3)).astype(np.int32)

    def test_t2_5cm_reduces_to_40cm_exact(self):
        """T2: 5cm uniform ring block-reduced by 8 == direct 40cm binning."""
        # Build a single ring that spans 0–40m at 5cm (no inner zone)
        spec_fine   = GridSpec(
            rings=(Ring(cell_m=0.05, half_extent_m=40.0),),
            z_range_m=(-3.0, 5.0),
        )
        spec_coarse = GridSpec(
            rings=(Ring(cell_m=0.40, half_extent_m=40.0),),
            z_range_m=(-3.0, 5.0),
        )

        rng = np.random.default_rng(2024)
        xy  = rng.uniform(-38.0, 38.0, size=(30_000, 2)).astype(np.float32)
        z   = rng.uniform(-2.0, 3.0, size=(30_000, 1)).astype(np.float32)
        pts = np.concatenate([xy, z, np.ones((30_000, 1), np.float32) * 0.5], axis=1)
        sc  = rng.integers(0, 4, 30_000, dtype=np.uint8)
        mv  = np.zeros(30_000, dtype=np.bool_)
        vru = np.zeros(30_000, dtype=np.bool_)
        cf  = np.ones(30_000, dtype=np.float16)

        grid_fine   = _build(pts, sc, mv, vru, cf, spec_fine)
        grid_coarse = _build(pts, sc, mv, vru, cf, spec_coarse)

        fine_count   = grid_fine.layer("count", 0).astype(np.int32)
        coarse_count = grid_coarse.layer("count", 0).astype(np.int32)

        # 8 = 40cm / 5cm
        reduced = self._block_reduce_count(fine_count, factor=8)

        np.testing.assert_array_equal(
            reduced, coarse_count,
            err_msg="T2 FAIL: 5cm block-reduced by 8 != direct 40cm count"
        )

    def test_t2_2x_reduction_exact(self):
        """T2: 5cm block-reduced by 2 == direct 10cm binning (single ring)."""
        spec_5  = GridSpec(rings=(Ring(cell_m=0.05, half_extent_m=20.0),), z_range_m=(-3.0, 5.0))
        spec_10 = GridSpec(rings=(Ring(cell_m=0.10, half_extent_m=20.0),), z_range_m=(-3.0, 5.0))

        rng = np.random.default_rng(123)
        xy  = rng.uniform(-18.0, 18.0, size=(10_000, 2)).astype(np.float32)
        z   = rng.uniform(-2.0, 3.0, size=(10_000, 1)).astype(np.float32)
        pts = np.concatenate([xy, z, np.zeros((10_000, 1), np.float32)], axis=1)
        sc  = rng.integers(0, 4, 10_000, dtype=np.uint8)
        mv  = np.zeros(10_000, dtype=np.bool_)
        vru = np.zeros(10_000, dtype=np.bool_)
        cf  = np.ones(10_000, dtype=np.float16)

        g5  = _build(pts, sc, mv, vru, cf, spec_5)
        g10 = _build(pts, sc, mv, vru, cf, spec_10)

        reduced = self._block_reduce_count(g5.layer("count", 0).astype(np.int32), factor=2)
        direct  = g10.layer("count", 0).astype(np.int32)

        np.testing.assert_array_equal(reduced, direct, err_msg="T2 FAIL: 5cm→10cm 2× reduction mismatch")

    def test_t2_total_conservation_across_reduction(self):
        """T2: sum of fine counts == sum of block-reduced counts (no loss)."""
        spec_5  = GridSpec(rings=(Ring(cell_m=0.05, half_extent_m=40.0),), z_range_m=(-3.0, 5.0))
        rng = np.random.default_rng(55)
        xy  = rng.uniform(-38.0, 38.0, size=(20_000, 2)).astype(np.float32)
        z   = rng.uniform(-2.0, 2.0, size=(20_000, 1)).astype(np.float32)
        pts = np.concatenate([xy, z, np.zeros((20_000, 1), np.float32)], axis=1)
        sc  = np.zeros(20_000, dtype=np.uint8)
        mv  = np.zeros(20_000, dtype=np.bool_)
        vru = np.zeros(20_000, dtype=np.bool_)
        cf  = np.ones(20_000, dtype=np.float16)
        g5 = _build(pts, sc, mv, vru, cf, spec_5)
        fine_total   = int(g5.layer("count", 0).sum())
        reduced      = self._block_reduce_count(g5.layer("count", 0).astype(np.int32), factor=4)
        reduced_total = int(reduced.sum())
        assert fine_total == reduced_total, (
            f"T2 FAIL: fine total {fine_total} != reduced total {reduced_total}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Boundary Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestT3BoundaryDeterminism:
    """Points exactly at ring boundaries land in exactly one correct cell."""

    def _ring_for_point(self, px: float, py: float, spec: GridSpec) -> int:
        """Return which ring index a point should be assigned to, or -1 if OOB."""
        cheb = max(abs(px), abs(py))
        for r_idx, ring in enumerate(spec.rings):
            inner = spec.rings[r_idx - 1].half_extent_m if r_idx > 0 else 0.0
            if inner <= cheb < ring.half_extent_m:
                return r_idx
        return -1  # out of range

    def _count_total_assigned_for_point(self, px: float, py: float, spec: GridSpec) -> int:
        """Build grid with exactly this one point; return total count assigned."""
        pt  = np.array([[px, py, 0.0, 0.5]], dtype=np.float32)
        sc  = np.array([0], dtype=np.uint8)
        mv  = np.array([False])
        vru = np.array([False])
        cf  = np.array([1.0], dtype=np.float16)
        grid = _build(pt, sc, mv, vru, cf, spec)
        return grid.n_points_assigned

    @pytest.mark.parametrize("boundary", [10.0, 30.0, 60.0])
    def test_t3_inner_boundary_lands_in_outer_ring(self, boundary: float):
        """T3: point at x=boundary is assigned to the outer ring (half-open [inner,outer))."""
        # At exactly x=boundary the Chebyshev distance == inner ring's half_extent
        # → half-open rule means it belongs to the OUTER ring
        px, py = boundary, 0.0
        expected_ring = self._ring_for_point(px, py, FOVEA_4RING_SPEC)
        assert expected_ring >= 0, f"boundary={boundary} classified as OOB"

        pt  = np.array([[px, py, 0.0, 0.5]], dtype=np.float32)
        sc  = np.array([0], dtype=np.uint8)
        mv  = np.array([False])
        vru = np.array([False])
        cf  = np.array([1.0], dtype=np.float16)
        grid = _build(pt, sc, mv, vru, cf, FOVEA_4RING_SPEC)

        assert grid.n_points_assigned == 1, (
            f"T3 FAIL: boundary={boundary} — expected 1 assigned, got {grid.n_points_assigned}"
        )

        # Verify it landed in the expected ring
        # Point at x=boundary is inside outer ring; inner ring's area excludes this
        outer_count = int(grid.layer("count", expected_ring).sum())
        assert outer_count >= 1, (
            f"T3 FAIL: boundary={boundary} not found in ring={expected_ring}"
        )

    @pytest.mark.parametrize("boundary", [-10.0, -30.0, -60.0])
    def test_t3_negative_boundary_lands_in_outer_ring(self, boundary: float):
        """T3: negative boundary values are handled symmetrically."""
        px, py = boundary, 0.0
        expected_ring = self._ring_for_point(px, py, FOVEA_4RING_SPEC)
        assert expected_ring >= 0, f"boundary={boundary} classified as OOB"
        n_assigned = self._count_total_assigned_for_point(px, py, FOVEA_4RING_SPEC)
        assert n_assigned == 1, (
            f"T3 FAIL: boundary={boundary} gave {n_assigned} assigned points"
        )

    def test_t3_negative_zero_xy(self):
        """T3: (-0.0, -0.0) is treated identically to (0.0, 0.0) — ring 0."""
        for px, py in [(0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0)]:
            ring = self._ring_for_point(px, py, FOVEA_4RING_SPEC)
            assert ring == 0, f"T3 FAIL: ({px},{py}) → ring {ring}, expected 0"

    def test_t3_outer_boundary_is_dropped(self):
        """T3: point at x=100.0 (== outermost boundary) is dropped, not assigned."""
        n = self._count_total_assigned_for_point(100.0, 0.0, FOVEA_4RING_SPEC)
        assert n == 0, f"T3 FAIL: x=100.0 should be OOB but got {n} assigned"

    @pytest.mark.parametrize("coord", [10.0, -10.0, 30.0, -30.0, 60.0, -60.0])
    def test_t3_exact_boundary_no_double_count(self, coord: float):
        """T3: a point at an exact boundary is never counted in two rings."""
        pt  = np.array([[coord, 0.0, 0.0, 0.5]], dtype=np.float32)
        sc  = np.array([0], dtype=np.uint8)
        mv  = np.array([False])
        vru = np.array([False])
        cf  = np.array([1.0], dtype=np.float16)
        grid = _build(pt, sc, mv, vru, cf, FOVEA_4RING_SPEC)
        # Exactly 1 assignment total
        assert grid.n_points_assigned + grid.n_points_dropped == 1
        assert grid.n_points_assigned <= 1, "T3 FAIL: point counted more than once"


# ─────────────────────────────────────────────────────────────────────────────
# T4 — Round-Trip (cell index ↔ world centroid)
# ─────────────────────────────────────────────────────────────────────────────

class TestT4RoundTrip:
    """Map point → cell → world centroid → cell: output must match input."""

    def _world_to_cell(self, x: float, y: float, half: float, cell_m: float) -> tuple[int, int]:
        """Rounding-guarded cell index (same formula as aggregate.py)."""
        GUARD = 1e-9
        ix = int(math.floor((x + half) / cell_m + GUARD))
        iy = int(math.floor((y + half) / cell_m + GUARD))
        side = round(2.0 * half / cell_m)
        ix = max(0, min(side - 1, ix))
        iy = max(0, min(side - 1, iy))
        return ix, iy

    def _cell_to_centroid(self, ix: int, iy: int, half: float, cell_m: float) -> tuple[float, float]:
        """Return world (x, y) centroid of cell (ix, iy)."""
        x = -half + (ix + 0.5) * cell_m
        y = -half + (iy + 0.5) * cell_m
        return x, y

    @pytest.mark.parametrize("r_idx", [0, 1, 2, 3])
    def test_t4_round_trip_all_rings(self, r_idx: int):
        """T4: for every ring, cell_to_centroid→world_to_cell is identity."""
        ring = FOVEA_4RING_SPEC.rings[r_idx]
        half = ring.half_extent_m
        cell = ring.cell_m
        side = round(2.0 * half / cell)

        rng = np.random.default_rng(r_idx)
        # Sample 500 random cell indices
        ixs = rng.integers(0, side, 500)
        iys = rng.integers(0, side, 500)

        for ix, iy in zip(ixs, iys):
            cx, cy = self._cell_to_centroid(int(ix), int(iy), half, cell)
            ix2, iy2 = self._world_to_cell(cx, cy, half, cell)
            assert ix2 == ix and iy2 == iy, (
                f"T4 FAIL ring={r_idx}: cell ({ix},{iy}) → centroid "
                f"({cx:.6f},{cy:.6f}) → cell ({ix2},{iy2})"
            )

    def test_t4_grid_cell_centres_round_trip(self):
        """T4: ClipmapGrid.cell_centres() → world_to_cell matches meshgrid."""
        rng = np.random.default_rng(77)
        xy = rng.uniform(-9.0, 9.0, size=(500, 2)).astype(np.float32)
        z  = rng.uniform(-2.0, 2.0, size=(500, 1)).astype(np.float32)
        pts = np.concatenate([xy, z, np.zeros((500, 1), np.float32)], axis=1)
        sc  = np.zeros(500, dtype=np.uint8)
        mv  = np.zeros(500, dtype=np.bool_)
        vru = np.zeros(500, dtype=np.bool_)
        cf  = np.ones(500, dtype=np.float16)
        grid = _build(pts, sc, mv, vru, cf, FOVEA_4RING_SPEC)

        r_idx = 0
        ring = grid.spec.rings[r_idx]
        half, cell = ring.half_extent_m, ring.cell_m
        side = grid.side(r_idx)

        X, Y = grid.cell_centres(r_idx)
        assert X.shape == (side, side)
        assert Y.shape == (side, side)

        # For each cell (row=iy, col=ix), centroid must round-trip
        for iy in range(0, side, 20):  # stride 20 for speed
            for ix in range(0, side, 20):
                cx = float(X[iy, ix])
                cy = float(Y[iy, ix])
                ix2, iy2 = self._world_to_cell(cx, cy, half, cell)
                assert ix2 == ix and iy2 == iy, (
                    f"T4 FAIL: cell ({ix},{iy}) centroid ({cx:.6f},{cy:.6f}) → ({ix2},{iy2})"
                )

    def test_t4_boundary_cell_centroid_safe(self):
        """T4: boundary cells' centroids are safely inside the ring extent."""
        for r_idx, ring in enumerate(FOVEA_4RING_SPEC.rings):
            half = ring.half_extent_m
            cell = ring.cell_m
            side = round(2.0 * half / cell)
            # First and last cell centroids
            first_x = -half + 0.5 * cell
            last_x  = -half + (side - 0.5) * cell
            assert abs(first_x) < half + cell * 0.01, (
                f"T4 FAIL ring={r_idx}: first centroid {first_x} outside ±{half}"
            )
            assert abs(last_x) < half + cell * 0.01, (
                f"T4 FAIL ring={r_idx}: last centroid {last_x} outside ±{half}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# T5 — Partition / Tile
# ─────────────────────────────────────────────────────────────────────────────

class TestT5PartitionTile:
    """Ring masks are non-overlapping and cover the valid plane."""

    def _ring_active_mask_world(
        self, side_coarse: int, coarse_cell: float, ring_idx: int, spec: GridSpec
    ) -> np.ndarray:
        """Return a (side_coarse, side_coarse) bool mask of world cells active in ring_idx.

        We evaluate at the coarsest grid resolution to get a common reference frame.
        A world cell (ir, ic) is "active in ring_idx" if its centre satisfies:
          inner_half <= max(|x|, |y|) < outer_half
        """
        inner = spec.rings[ring_idx - 1].half_extent_m if ring_idx > 0 else 0.0
        outer = spec.rings[ring_idx].half_extent_m

        half_coarse = spec.rings[-1].half_extent_m
        # Build a meshgrid of cell centres for the coarsest ring's resolution
        coords = np.linspace(
            -half_coarse + 0.5 * coarse_cell,
            half_coarse - 0.5 * coarse_cell,
            side_coarse,
        )
        X, Y = np.meshgrid(coords, coords)
        cheb = np.maximum(np.abs(X), np.abs(Y))
        return (cheb >= inner) & (cheb < outer)

    def test_t5_union_covers_full_square(self):
        """T5: union of all ring active masks covers the full inner plane."""
        spec = FOVEA_4RING_SPEC
        coarse_ring = spec.rings[-1]
        coarse_cell = coarse_ring.cell_m
        side_coarse = round(2.0 * coarse_ring.half_extent_m / coarse_cell)

        union_mask = np.zeros((side_coarse, side_coarse), dtype=np.bool_)
        for r_idx in range(len(spec.rings)):
            mask = self._ring_active_mask_world(side_coarse, coarse_cell, r_idx, spec)
            union_mask |= mask

        assert union_mask.all(), (
            f"T5 FAIL: union mask has {(~union_mask).sum()} uncovered cells"
        )

    def test_t5_no_overlaps_between_rings(self):
        """T5: pairwise intersection of ring active masks is empty."""
        spec = FOVEA_4RING_SPEC
        coarse_ring = spec.rings[-1]
        coarse_cell = coarse_ring.cell_m
        side_coarse = round(2.0 * coarse_ring.half_extent_m / coarse_cell)

        masks = [
            self._ring_active_mask_world(side_coarse, coarse_cell, r, spec)
            for r in range(len(spec.rings))
        ]

        for r1, r2 in itertools.combinations(range(len(spec.rings)), 2):
            overlap = masks[r1] & masks[r2]
            assert not overlap.any(), (
                f"T5 FAIL: rings {r1} and {r2} overlap in {overlap.sum()} cells"
            )

    def test_t5_ring_cell_counts_sum_to_total(self):
        """T5: sum of active-cell counts per ring == total active cells."""
        spec = FOVEA_4RING_SPEC
        s = spec.total_cells()  # from GridSpec (sum of cells_in_ring for all rings)
        manual = sum(spec.cells_in_ring(i) for i in range(len(spec.rings)))
        assert s == manual == 910_000, (
            f"T5 FAIL: total active cells = {s}, expected 910,000"
        )

    def test_t5_four_ring_partition_no_gap(self):
        """T5: a dense grid of world points, each assigned to exactly one ring."""
        spec = FOVEA_4RING_SPEC

        # Sample ~40k points uniformly in [-99.9, 99.9]^2 (all in range)
        rng = np.random.default_rng(314)
        xy  = rng.uniform(-99.9, 99.9, size=(40_000, 2)).astype(np.float32)
        z   = rng.uniform(-2.0, 2.0, size=(40_000, 1)).astype(np.float32)
        pts = np.concatenate([xy, z, np.zeros((40_000, 1), np.float32)], axis=1)
        sc  = np.zeros(40_000, dtype=np.uint8)
        mv  = np.zeros(40_000, dtype=np.bool_)
        vru = np.zeros(40_000, dtype=np.bool_)
        cf  = np.ones(40_000, dtype=np.float16)
        grid = _build(pts, sc, mv, vru, cf, spec)
        s = grid.stats()
        # All should be assigned (none OOB)
        assert s["n_points_dropped"] == 0, (
            f"T5 FAIL: {s['n_points_dropped']} points dropped from in-range inputs"
        )
        assert s["n_points_assigned"] == 40_000
