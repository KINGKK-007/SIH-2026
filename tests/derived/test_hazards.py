"""Tests for synthetic hazard injection and evaluation (README 6.8, tasks T11.7 - T11.8)."""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.config import HazardConfig, KerbHazardConfig, OverhangConfig, PotholeConfig
from foveamap.derived.hazards import Hazard, inject_hazards, check_hazard_detected
from foveamap.eval.hazard_eval import wilson_score_interval
from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_KERB,
    FLAG_LOW_CLEARANCE,
    FLAG_TRAVERSABLE,
    LAYER_DTYPE,
    GridLayers,
)
from foveamap.grid.presets import spec_from_rings
from foveamap.io.labels import DRIVABLE, STATIC_OBSTACLE
from foveamap.pipeline.records import ClassifiedScan, Scan


@pytest.fixture
def hazards_cfg():
    return HazardConfig(
        seed=1337,
        per_bucket_injections=10,
        min_ground_points=3,
        pothole=PotholeConfig(radius_m=(0.3, 0.6), depth_m=(0.08, 0.15), detect_frac=0.5),
        kerb=KerbHazardConfig(width_m=0.3, length_m=4.0, height_m=(0.10, 0.20)),
        overhang=OverhangConfig(size_m=(3.0, 1.0), height_above_ground_m=(1.6, 2.4)),
    )


@pytest.fixture
def sample_classified_scan():
    # Synthetic flat drivable ground plane
    xs = np.linspace(-10.0, 10.0, 100)
    ys = np.linspace(-10.0, 10.0, 100)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    xyz = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    n = len(xyz)

    scan = Scan(
        seq="08",
        idx=0,
        xyz=xyz,
        remission=np.zeros(n, dtype=np.float32),
        raw_labels=None,
        pose=np.eye(4),
        timestamp=0.0,
    )
    return ClassifiedScan(
        scan=scan,
        super_cls=np.full(n, DRIVABLE, dtype=np.uint8),
        moving=np.zeros(n, dtype=np.bool_),
        conf=np.full(n, 255, dtype=np.uint8),
    )


def test_inject_hazards_pothole_kerb_overhang(sample_classified_scan, hazards_cfg):
    rng = np.random.default_rng(hazards_cfg.seed)
    injected_scan, hazards = inject_hazards(sample_classified_scan, hazards_cfg, rng)

    assert len(hazards) == 3
    types = {h.type for h in hazards}
    assert types == {"pothole", "kerb", "overhang"}

    # Overhang adds points
    assert len(injected_scan.scan.xyz) > len(sample_classified_scan.scan.xyz)

    # Pothole dropped z in its footprint
    pothole = next(h for h in hazards if h.type == "pothole")
    px, py = pothole.center_xy_m
    pr = pothole.params["radius_m"]
    d_sq = (sample_classified_scan.scan.xyz[:, 0] - px) ** 2 + (sample_classified_scan.scan.xyz[:, 1] - py) ** 2
    in_hole = d_sq <= pr ** 2
    assert np.any(in_hole)
    n_orig = len(sample_classified_scan.scan.xyz)
    assert np.all(injected_scan.scan.xyz[:n_orig][in_hole, 2] < -0.05)


def test_wilson_score_interval():
    # 0 out of 100
    low, high = wilson_score_interval(0, 100)
    assert 0.0 <= low <= high
    assert high < 0.05

    # 100 out of 100
    low, high = wilson_score_interval(100, 100)
    assert low > 0.95
    assert high <= 1.0

    # 50 out of 100
    low, high = wilson_score_interval(50, 100)
    assert np.isclose((low + high) / 2, 0.5, atol=0.02)
