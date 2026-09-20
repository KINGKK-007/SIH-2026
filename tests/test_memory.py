"""tests/test_memory.py — Memory accounting validation (§5.7).

Tests verify that:
- fovea_4ring has exactly 910,000 active cells
- uniform_5cm over ±100 m has exactly 16,000,000 cells
- active_cells correctly subtracts inner-ring overlap from outer rings
- allocated vs active bytes are computed correctly
- all 4 baseline specs are consistent with §5.5 table
"""

from __future__ import annotations

import os
import pytest
import numpy as np

os.environ.setdefault("FOVEAMAP_NO_NUMBA", "1")

from foveamap.grid.spec import Ring, GridSpec
from foveamap.grid.clipmap import ClipmapGrid
from foveamap.grid.layers import BYTES_PER_CELL, BYTES_PER_CELL_TARGET


# ─────────────────────────────────────────────────────────────────────────────
# Specs (matching §5.5 table)
# ─────────────────────────────────────────────────────────────────────────────

FOVEA_4RING = GridSpec(
    rings=(
        Ring(cell_m=0.05, half_extent_m=10.0),
        Ring(cell_m=0.10, half_extent_m=30.0),
        Ring(cell_m=0.20, half_extent_m=60.0),
        Ring(cell_m=0.40, half_extent_m=100.0),
    ),
    z_range_m=(-3.0, 5.0),
    aggregation="safety_priority",
    preset="fovea_4ring",
)

# PS-literal: 5 cm within 10 m, 50 cm out to 100 m
PS_LITERAL_2RING = GridSpec(
    rings=(
        Ring(cell_m=0.05, half_extent_m=10.0),
        Ring(cell_m=0.50, half_extent_m=100.0),
    ),
    z_range_m=(-3.0, 5.0),
    aggregation="safety_priority",
    preset="ps_literal_2ring",
)

UNIFORM_5CM = GridSpec(
    rings=(Ring(cell_m=0.05, half_extent_m=100.0),),
    z_range_m=(-3.0, 5.0),
    aggregation="safety_priority",
    preset="uniform_5cm",
)

UNIFORM_20CM = GridSpec(
    rings=(Ring(cell_m=0.20, half_extent_m=100.0),),
    z_range_m=(-3.0, 5.0),
    aggregation="safety_priority",
    preset="uniform_20cm",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dummy_grid(spec: GridSpec, n: int = 100) -> ClipmapGrid:
    """Build a minimal ClipmapGrid (no real data needed for stats)."""
    pts = np.zeros((n, 4), dtype=np.float32)
    sc  = np.zeros(n, dtype=np.uint8)
    mv  = np.zeros(n, dtype=np.bool_)
    vru = np.zeros(n, dtype=np.bool_)
    cf  = np.ones(n, dtype=np.float16)
    return ClipmapGrid.build(pts, sc, mv, vru, cf, spec)


def _cells_in_ring_manual(ring: Ring, prev: Ring | None) -> int:
    """Manual annulus cell count."""
    outer_side = round(2.0 * ring.half_extent_m / ring.cell_m)
    if prev is None:
        return outer_side * outer_side
    inner_cells = round(2.0 * prev.half_extent_m / ring.cell_m)
    return outer_side * outer_side - inner_cells * inner_cells


# ─────────────────────────────────────────────────────────────────────────────
# Test: cell count constants
# ─────────────────────────────────────────────────────────────────────────────

class TestCellCountConstants:
    """Exact cell counts from §5.5 table."""

    def test_fovea_4ring_ring0_cells(self):
        """Ring 0 (0–10m, 5cm): side=400, cells=160,000."""
        ring0 = FOVEA_4RING.rings[0]
        side = round(2.0 * ring0.half_extent_m / ring0.cell_m)
        assert side == 400
        assert side * side == 160_000

    def test_fovea_4ring_ring1_annulus(self):
        """Ring 1 (10–30m, 10cm): side=600, annulus=600²−200²=360000−40000=320,000."""
        ring0, ring1 = FOVEA_4RING.rings[0], FOVEA_4RING.rings[1]
        outer_side   = round(2.0 * ring1.half_extent_m / ring1.cell_m)   # 600
        inner_cells  = round(2.0 * ring0.half_extent_m / ring1.cell_m)   # 200
        assert outer_side == 600
        assert inner_cells == 200
        assert outer_side**2 - inner_cells**2 == 320_000

    def test_fovea_4ring_ring2_annulus(self):
        """Ring 2 (30–60m, 20cm): side=600, inner=300 → 360000−90000=270,000."""
        ring1, ring2 = FOVEA_4RING.rings[1], FOVEA_4RING.rings[2]
        outer_side  = round(2.0 * ring2.half_extent_m / ring2.cell_m)    # 600
        inner_cells = round(2.0 * ring1.half_extent_m / ring2.cell_m)    # 300
        assert outer_side == 600
        assert inner_cells == 300
        assert outer_side**2 - inner_cells**2 == 270_000

    def test_fovea_4ring_ring3_annulus(self):
        """Ring 3 (60–100m, 40cm): side=500, inner=300 → 250000−90000=160,000."""
        ring2, ring3 = FOVEA_4RING.rings[2], FOVEA_4RING.rings[3]
        outer_side  = round(2.0 * ring3.half_extent_m / ring3.cell_m)    # 500
        inner_cells = round(2.0 * ring2.half_extent_m / ring3.cell_m)    # 300
        assert outer_side == 500
        assert inner_cells == 300
        assert outer_side**2 - inner_cells**2 == 160_000

    def test_fovea_4ring_total_active_exactly_910k(self):
        """Master plan §5.5: fovea_4ring active cells == exactly 910,000."""
        total = FOVEA_4RING.total_cells()
        assert total == 910_000, f"Expected 910,000, got {total}"

    def test_uniform_5cm_total_exactly_16m(self):
        """Uniform 5cm over ±100m: 4000² == exactly 16,000,000."""
        side = round(2.0 * UNIFORM_5CM.rings[0].half_extent_m / UNIFORM_5CM.rings[0].cell_m)
        assert side == 4000
        assert UNIFORM_5CM.total_cells() == 16_000_000

    def test_reduction_factor_17x(self):
        """FoveaMap uses ~17.6× fewer cells than uniform 5cm (§5.5)."""
        ratio = UNIFORM_5CM.total_cells() / FOVEA_4RING.total_cells()
        assert ratio > 17.0, f"Reduction ratio {ratio:.2f} < 17× (expected ~17.6)"
        assert ratio < 20.0, f"Reduction ratio {ratio:.2f} unexpectedly large"

    def test_ps_literal_2ring_ring0(self):
        """PS-literal ring 0 (5cm, 10m): side=400, cells=160,000."""
        ring0 = PS_LITERAL_2RING.rings[0]
        side = round(2.0 * ring0.half_extent_m / ring0.cell_m)
        assert side == 400
        assert side * side == 160_000

    def test_ps_literal_2ring_ring1(self):
        """PS-literal ring 1 (50cm, 100m): outer=400, inner=40 → 160000−1600=158,400.

        inner_cells = round(2 × 10m / 0.50m) = round(40) = 40
        annulus = 400² − 40² = 160000 − 1600 = 158,400.
        """
        ring0, ring1 = PS_LITERAL_2RING.rings[0], PS_LITERAL_2RING.rings[1]
        outer_side  = round(2.0 * ring1.half_extent_m / ring1.cell_m)    # 400
        inner_cells = round(2.0 * ring0.half_extent_m / ring1.cell_m)    # 40
        assert outer_side == 400
        assert inner_cells == 40  # 2×10m / 0.50m = 40 coarse cells
        assert outer_side**2 - inner_cells**2 == 158_400

    def test_uniform_20cm_total(self):
        """Uniform 20cm over ±100m: 1000² == 1,000,000."""
        total = UNIFORM_20CM.total_cells()
        assert total == 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Test: GridSpec cell-count API
# ─────────────────────────────────────────────────────────────────────────────

class TestGridSpecCellCountAPI:
    """GridSpec.cells_in_ring() and total_cells() are internally consistent."""

    def test_cells_in_ring_matches_manual(self):
        rings = FOVEA_4RING.rings
        for i, ring in enumerate(rings):
            prev = rings[i - 1] if i > 0 else None
            expected = _cells_in_ring_manual(ring, prev)
            actual = FOVEA_4RING.cells_in_ring(i)
            assert actual == expected, (
                f"ring {i}: cells_in_ring={actual}, expected={expected}"
            )

    def test_total_cells_equals_sum_of_rings(self):
        manual = sum(FOVEA_4RING.cells_in_ring(i) for i in range(len(FOVEA_4RING.rings)))
        assert FOVEA_4RING.total_cells() == manual

    def test_memory_bytes_formula(self):
        """memory_bytes() uses BYTES_PER_CELL_TARGET (8 B/cell)."""
        expected = FOVEA_4RING.total_cells() * 8
        assert FOVEA_4RING.memory_bytes() == expected


# ─────────────────────────────────────────────────────────────────────────────
# Test: ClipmapGrid.stats() memory accounting
# ─────────────────────────────────────────────────────────────────────────────

class TestClipmapGridStats:
    """ClipmapGrid.stats() returns correct accounting values."""

    def test_stats_active_cells_fovea4ring(self):
        """stats()['active_cells'] == 910,000 for fovea_4ring."""
        grid = _dummy_grid(FOVEA_4RING, n=1)
        s = grid.stats()
        assert s["active_cells"] == 910_000

    def test_stats_allocated_cells_fovea4ring(self):
        """stats()['allocated_cells'] == sum of all full squares."""
        rings = FOVEA_4RING.rings
        expected_alloc = sum(
            round(2.0 * r.half_extent_m / r.cell_m) ** 2 for r in rings
        )
        grid = _dummy_grid(FOVEA_4RING, n=1)
        s = grid.stats()
        assert s["allocated_cells"] == expected_alloc

    def test_stats_active_lt_allocated(self):
        """Active cells < allocated cells for multi-ring grid."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        assert s["active_cells"] < s["allocated_cells"], (
            "Active cells should be less than allocated (overlap subtracted)"
        )

    def test_stats_single_ring_active_equals_allocated(self):
        """Single-ring: active == allocated (no inner-ring overlap to subtract)."""
        grid = _dummy_grid(UNIFORM_5CM, n=1)
        s = grid.stats()
        assert s["active_cells"] == s["allocated_cells"]

    def test_stats_active_bytes_formula(self):
        """active_bytes == active_cells × bytes_per_cell."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        assert s["active_bytes"] == s["active_cells"] * BYTES_PER_CELL

    def test_stats_allocated_bytes_formula(self):
        """allocated_bytes == allocated_cells × bytes_per_cell."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        assert s["allocated_bytes"] == s["allocated_cells"] * BYTES_PER_CELL

    def test_stats_target_bytes_formula(self):
        """active_bytes_target == active_cells × 8 (target layout)."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        assert s["active_bytes_target"] == s["active_cells"] * BYTES_PER_CELL_TARGET

    def test_stats_mb_values_reasonable(self):
        """Memory in MB is plausible (>1 MB, <100 MB for fovea_4ring)."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        assert 1.0 < s["active_mb"] < 100.0, (
            f"active_mb={s['active_mb']:.2f} outside expected range [1, 100]"
        )

    def test_stats_rings_list_length(self):
        """stats()['rings'] has one entry per ring."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        assert len(s["rings"]) == 4

    def test_stats_per_ring_active_sum(self):
        """Sum of per-ring active_cells == total active_cells."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        per_ring_sum = sum(r["active_cells"] for r in s["rings"])
        assert per_ring_sum == s["active_cells"]

    def test_stats_per_ring_allocated_sum(self):
        """Sum of per-ring allocated_cells == total allocated_cells."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        per_ring_sum = sum(r["allocated_cells"] for r in s["rings"])
        assert per_ring_sum == s["allocated_cells"]

    def test_stats_required_keys_present(self):
        """stats() contains all required keys."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        required = {
            "allocated_cells", "active_cells",
            "allocated_bytes", "active_bytes",
            "allocated_bytes_target", "active_bytes_target",
            "allocated_mb", "active_mb",
            "bytes_per_cell", "bytes_per_cell_target",
            "rings", "build_time_ms",
            "n_points_assigned", "n_points_dropped",
        }
        missing = required - s.keys()
        assert not missing, f"stats() missing keys: {missing}"

    @pytest.mark.parametrize("r_idx,expected_active", [
        (0, 160_000),
        (1, 320_000),
        (2, 270_000),
        (3, 160_000),
    ])
    def test_stats_per_ring_active_exact(self, r_idx: int, expected_active: int):
        """stats()['rings'][r]['active_cells'] matches §5.5 table."""
        grid = _dummy_grid(FOVEA_4RING)
        s = grid.stats()
        actual = s["rings"][r_idx]["active_cells"]
        assert actual == expected_active, (
            f"Ring {r_idx}: expected {expected_active:,}, got {actual:,}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test: four-baseline comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestFourBaselineComparison:
    """Compare all four baseline memory footprints (§3.6)."""

    def test_ordering_fovea_fewer_than_uniform_5cm(self):
        """fovea_4ring has fewer active cells than uniform_5cm (17× reduction)."""
        assert FOVEA_4RING.total_cells() < UNIFORM_5CM.total_cells()

    def test_ordering_ps_literal_fewer_than_uniform_5cm(self):
        """ps_literal_2ring also has fewer cells than uniform_5cm."""
        assert PS_LITERAL_2RING.total_cells() < UNIFORM_5CM.total_cells()

    def test_ps_literal_total_cells(self):
        """ps_literal_2ring total = ring0 + ring1 annulus = 160,000 + 158,400 = 318,400."""
        total = PS_LITERAL_2RING.total_cells()
        assert total == 318_400, f"Expected 318,400, got {total}"

    def test_ordering_uniform_5cm_largest(self):
        """uniform_5cm has more active cells than all other presets."""
        assert UNIFORM_5CM.total_cells() > FOVEA_4RING.total_cells()
        assert UNIFORM_5CM.total_cells() > PS_LITERAL_2RING.total_cells()
        assert UNIFORM_5CM.total_cells() > UNIFORM_20CM.total_cells()

    def test_uniform_5cm_over_uniform_20cm_ratio(self):
        """uniform_5cm has exactly 25× more cells than uniform_20cm (0.20/0.05=4 → 4²=16×)."""
        # Actually (100/5)² / (100/20)² = 4000²/1000² = 16
        ratio = UNIFORM_5CM.total_cells() / UNIFORM_20CM.total_cells()
        assert ratio == 16.0, f"Expected 16× ratio, got {ratio}"

    def test_all_specs_have_positive_memory(self):
        """All specs have memory_bytes() > 0."""
        for spec in [FOVEA_4RING, PS_LITERAL_2RING, UNIFORM_5CM, UNIFORM_20CM]:
            assert spec.memory_bytes() > 0
