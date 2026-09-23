"""Grid test fixtures: presets from configs/grid.yaml and labelled random / boundary-adversarial points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foveamap.config import GridConfig, load_config
from foveamap.grid.presets import GridSpec, load_preset

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="session")
def grid_cfg() -> GridConfig:
    return load_config(CONFIG_DIR).grid


@pytest.fixture(scope="session")
def presets(grid_cfg: GridConfig) -> dict[str, GridSpec]:
    return {name: load_preset(name, grid_cfg) for name in grid_cfg.presets}


def adversarial_coords_mm(spec: GridSpec) -> np.ndarray:
    """Boundary-adversarial 1-D coordinates (I2): +-R_k, +-R_k -+ 1 mm, multiples of every s_k, 0."""
    values = {0, 1, -1}
    for ring in spec.rings:
        r, s = ring.r_max_mm, ring.cell_mm
        values |= {r, -r, r - 1, -r + 1, r + 1, -r - 1}
        for m in (1, 2, 3, 7):
            values |= {m * s, -m * s, m * s - 1, -m * s + 1, r - m * s, -r + m * s}
    return np.array(sorted(values), dtype=np.int64)


def adversarial_points_mm(spec: GridSpec) -> np.ndarray:
    """All pairs of adversarial coordinates as (N, 2) int64 x/y in millimetres."""
    c = adversarial_coords_mm(spec)
    xx, yy = np.meshgrid(c, c, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def labelled_points(
    rng: np.random.Generator, n: int, extent_mm: int, margin_mm: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random quantised points ``(xyz_mm int32, super_cls uint8, moving bool, conf uint8)``.

    Coordinates cover ``[-extent - margin, extent + margin)`` so some points are out of grid; z spans the
    realistic range; classes, motion flags and confidences are random.
    """
    lo, hi = -extent_mm - margin_mm, extent_mm + margin_mm
    xy = rng.integers(lo, hi, size=(n, 2))
    z = rng.integers(-3_000, 5_000, size=(n, 1))
    xyz = np.hstack([xy, z]).astype(np.int32)
    super_cls = rng.integers(0, 5, size=n).astype(np.uint8)
    moving = rng.random(n) < 0.2
    conf = rng.integers(0, 256, size=n).astype(np.uint8)
    return xyz, super_cls, moving, conf
