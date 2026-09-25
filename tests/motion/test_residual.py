"""Unit tests for ego-compensated range residual and per-point motion votes (T10.1)."""

from types import SimpleNamespace

import numpy as np

from foveamap.io.synthetic import Box, SensorModel, render_scan
from foveamap.motion.residual import range_residual_votes


def test_residual_static_planar_scene_zero_votes() -> None:
    """A static ground plane observed in both scans should yield zero moving votes."""
    cfg = SimpleNamespace(window=3, tau0_m=0.30, tau1_rel=0.02, occlusion_rule=False)
    sensor = SensorModel(n_beams=32, n_azimuth=256, range_noise_m=0.0)
    rng = np.random.default_rng(42)

    xyz_static, _, _ = render_scan([], np.eye(4), 0.0, rng, sensor)

    # Current scan and previous scan in current frame are identical (static scene)
    votes = range_residual_votes(xyz_static, xyz_static, cfg, proj_H=64, proj_W=1024)

    # All points should be static-consistent (0) or unobserved (-1), zero moving (1)
    n_moving = int(np.sum(votes == 1))
    assert n_moving == 0
    # Overwhelming majority of planar points should be static-consistent (0)
    assert np.mean(votes == 0) > 0.95


def test_residual_translated_box_votes_on_box() -> None:
    """A static scene with one box translating by 2.0 m between scans.

    Moving votes should be concentrated specifically on the translated box points.
    With occlusion_rule=True, disoccluded background ground points are filtered.
    """
    cfg = SimpleNamespace(window=3, tau0_m=0.30, tau1_rel=0.02, occlusion_rule=True)
    sensor = SensorModel(n_beams=32, n_azimuth=256, range_noise_m=0.0)
    rng = np.random.default_rng(42)

    # Box translating at 20 m/s along Y (moves 2.0 m in dt=0.1 s)
    box = Box(
        lo=np.array([8.0, -1.5, -0.8]),
        hi=np.array([12.0, 1.5, 0.8]),
        label=10,
        velocity=np.array([0.0, 20.0, 0.0]),
    )

    xyz_prev, _, raw_prev = render_scan([box], np.eye(4), 0.0, rng, sensor)
    xyz_cur, _, raw_cur = render_scan([box], np.eye(4), 0.1, rng, sensor)

    votes = range_residual_votes(xyz_cur, xyz_prev, cfg, proj_H=64, proj_W=1024)

    # Ground points should have virtually zero moving votes
    is_ground = (raw_cur & 0xFFFF) != 10
    is_box = (raw_cur & 0xFFFF) == 10

    ground_votes = votes[is_ground]
    box_votes = votes[is_box]

    # Ground points should be static-consistent with occlusion handling
    assert np.sum(ground_votes == 1) == 0

    # Moving box points should vote moving (exceeding vote_frac threshold)
    assert np.sum(box_votes == 1) > 0
    assert np.mean(box_votes == 1) >= 0.30

