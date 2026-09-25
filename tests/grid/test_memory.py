"""T7.1: memory accounting invariant I9 (accounting) and closed-form value checks.

I9: allocated counts == actual array sizes; logical cells <= allocated cells for every spec.
"""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.config import GridConfig, load_config
from foveamap.grid.baselines import uniform_spec
from foveamap.grid.engine import rasterize
from foveamap.grid.layers import BYTES_PER_CELL, GridLayers, finalize
from foveamap.grid.memory import MemoryReport, memory_report
from foveamap.grid.presets import GridSpec, load_preset, spec_from_rings

CONFIG_DIR = __import__("pathlib").Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="module")
def grid_cfg() -> GridConfig:
    return load_config(CONFIG_DIR).grid


@pytest.fixture(scope="module")
def fovea_spec(grid_cfg: GridConfig) -> GridSpec:
    return load_preset("fovea_default", grid_cfg)


@pytest.fixture(scope="module")
def uniform_5cm_spec() -> GridSpec:
    return uniform_spec(50, 100_000)  # 5 cm cell, 100 m extent


@pytest.fixture(scope="module")
def uniform_20cm_spec() -> GridSpec:
    return uniform_spec(200, 100_000)  # 20 cm cell, 100 m extent


# ── I9: allocated counts == array sizes, logical <= allocated ─────────────────


def test_i9_logical_le_allocated_all_presets(grid_cfg: GridConfig) -> None:
    """I9 (part 1): logical_cells() <= allocated_cells() for every preset."""
    for name in grid_cfg.presets:
        spec = load_preset(name, grid_cfg)
        assert spec.logical_cells() <= spec.allocated_cells(), (
            f"preset {name}: logical {spec.logical_cells()} > allocated {spec.allocated_cells()}"
        )


def test_i9_logical_equals_allocated_for_single_ring() -> None:
    """Single-ring presets have no hole: logical == allocated."""
    spec = uniform_spec(50, 10_000)  # 5 cm, 10 m
    assert spec.logical_cells() == spec.allocated_cells()


def test_i9_allocated_matches_array_sizes_after_finalize(grid_cfg: GridConfig, fovea_spec: GridSpec) -> None:
    """I9 (part 2): after finalize(), total array size == allocated_cells * BYTES_PER_CELL."""
    rng = np.random.default_rng(42)
    n = 1000
    xyz_m = rng.uniform(-90, 90, (n, 3)).astype(np.float32)
    super_cls = rng.integers(0, 5, n, dtype=np.uint8)
    moving = rng.random(n) < 0.2
    conf = rng.integers(0, 256, n, dtype=np.uint8)

    acc = rasterize(fovea_spec, xyz_m, super_cls, moving, conf)
    layers = finalize(acc, grid_cfg)

    for k, ring_arr in enumerate(layers.rings):
        expected = fovea_spec.rings[k].side ** 2
        assert ring_arr.size == expected, (
            f"ring {k}: finalize produced {ring_arr.size} cells, expected {expected}"
        )
    assert layers.nbytes() == fovea_spec.allocated_cells() * BYTES_PER_CELL


# ── memory_report: four representations ──────────────────────────────────────


def test_memory_report_returns_dataclass(grid_cfg: GridConfig, fovea_spec: GridSpec) -> None:
    rng = np.random.default_rng(7)
    xyz_m = rng.uniform(-90, 90, (500, 3)).astype(np.float32)
    super_cls = rng.integers(0, 5, 500, dtype=np.uint8)
    moving = rng.random(500) < 0.2
    conf = rng.integers(0, 256, 500, dtype=np.uint8)

    acc = rasterize(fovea_spec, xyz_m, super_cls, moving, conf)
    layers = finalize(acc, grid_cfg)

    report = memory_report(fovea_spec, layers, grid_cfg)
    assert isinstance(report, MemoryReport)


def test_memory_report_none_layers_ok(grid_cfg: GridConfig, fovea_spec: GridSpec) -> None:
    """memory_report with layers=None still returns theoretical estimates."""
    report = memory_report(fovea_spec, None, grid_cfg)
    assert isinstance(report, MemoryReport)
    assert report.fovea_bytes > 0
    assert report.uniform25d_bytes > 0
    assert report.dense3d_bytes > 0


def test_memory_report_values_positive(grid_cfg: GridConfig, fovea_spec: GridSpec) -> None:
    report = memory_report(fovea_spec, None, grid_cfg)
    assert report.dense3d_bytes > 0
    assert report.uniform25d_bytes > 0
    assert report.fovea_bytes > 0


def test_memory_report_fovea_le_uniform_5cm(grid_cfg: GridConfig, fovea_spec: GridSpec, uniform_5cm_spec: GridSpec) -> None:
    """FoveaMap must use less memory than uniform 5 cm at the same extent (R1: derived, not hard-coded)."""
    report_fovea = memory_report(fovea_spec, None, grid_cfg)
    report_uni = memory_report(uniform_5cm_spec, None, grid_cfg)
    # logical fovea < logical uniform_5cm
    assert fovea_spec.logical_cells() < uniform_5cm_spec.logical_cells(), (
        "fovea preset must have fewer logical cells than uniform_5cm"
    )
    assert report_fovea.fovea_bytes < report_uni.uniform25d_bytes


def test_memory_report_uniform25d_matches_closed_form(grid_cfg: GridConfig, uniform_5cm_spec: GridSpec) -> None:
    """uniform_5cm 2.5D theoretical == allocated_cells * BYTES_PER_CELL (closed form)."""
    report = memory_report(uniform_5cm_spec, None, grid_cfg)
    expected = uniform_5cm_spec.allocated_cells() * BYTES_PER_CELL
    assert report.uniform25d_bytes == expected


def test_memory_report_measured_matches_nbytes(grid_cfg: GridConfig, fovea_spec: GridSpec) -> None:
    """When layers are provided, fovea_bytes == layers.nbytes() (I9 measured path)."""
    rng = np.random.default_rng(99)
    xyz_m = rng.uniform(-90, 90, (800, 3)).astype(np.float32)
    super_cls = rng.integers(0, 5, 800, dtype=np.uint8)
    moving = rng.random(800) < 0.2
    conf = rng.integers(0, 256, 800, dtype=np.uint8)

    acc = rasterize(fovea_spec, xyz_m, super_cls, moving, conf)
    layers = finalize(acc, grid_cfg)

    report = memory_report(fovea_spec, layers, grid_cfg)
    assert report.fovea_bytes == layers.nbytes()
    assert report.basis == "allocated"


# ── uniform_spec ─────────────────────────────────────────────────────────────


def test_uniform_spec_single_ring(uniform_5cm_spec: GridSpec) -> None:
    assert len(uniform_5cm_spec.rings) == 1
    assert uniform_5cm_spec.rings[0].cell_mm == 50
    assert uniform_5cm_spec.rings[0].r_max_mm == 100_000


def test_uniform_spec_passes_validation() -> None:
    from foveamap.grid.presets import validate_preset
    spec = uniform_spec(200, 50_000)
    validate_preset(spec)  # must not raise


def test_uniform_spec_roundtrip_via_spec_from_rings() -> None:
    spec = uniform_spec(100, 10_000)
    assert spec.logical_cells() == spec.allocated_cells()
    expected_side = 2 * 10_000 // 100
    assert spec.rings[0].side == expected_side
