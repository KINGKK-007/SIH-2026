"""Unit tests for ClusterTracker with temporal hysteresis (T10.4)."""

import numpy as np

from foveamap.config import HysteresisConfig
from foveamap.motion.tracker import ClusterTracker
from foveamap.pipeline.records import ObjectBox


def test_tracker_velocity_and_hysteresis() -> None:
    h_cfg = HysteresisConfig(enabled=True, window=3, min_hits=2, gate_m=3.0, v_min_mps=0.5)
    tracker = ClusterTracker(type("Cfg", (), {"hysteresis": h_cfg})())

    # Frame 0: object at (10, 0, 0)
    obj0 = ObjectBox(
        id=0,
        cls_name="car",
        center=(10.0, 0.0, 0.0),
        size=(4.0, 2.0, 1.5),
        yaw=0.0,
        n_points=50,
        mean_conf=200.0,
        moving=False,
        vote_frac=0.1,
        speed_mps=None,
        safety_critical=False,
    )
    tracked0 = tracker.update([obj0], T_prev_to_cur=None, dt=0.1)
    assert len(tracked0) == 1
    assert tracked0[0].speed_mps is None
    assert tracked0[0].moving is False

    # Frame 1: ego-static (T = identity), object moved to (10, 0.2, 0) -> 0.2 m in 0.1 s = 2.0 m/s
    T_ident = np.eye(4)
    obj1 = ObjectBox(
        id=0,
        cls_name="car",
        center=(10.0, 0.2, 0.0),
        size=(4.0, 2.0, 1.5),
        yaw=0.0,
        n_points=50,
        mean_conf=200.0,
        moving=False,  # Single-frame vote was False
        vote_frac=0.1,
        speed_mps=None,
        safety_critical=False,
    )
    tracked1 = tracker.update([obj1], T_prev_to_cur=T_ident, dt=0.1)
    assert len(tracked1) == 1
    # Speed 2.0 m/s > v_min_mps (0.5 m/s) -> promoted to moving
    assert tracked1[0].speed_mps is not None
    assert np.isclose(tracked1[0].speed_mps, 2.0, atol=0.01)
    assert tracked1[0].moving is True


def test_tracker_vulnerable_road_users_safety_critical() -> None:
    tracker = ClusterTracker()
    ped = ObjectBox(
        id=0,
        cls_name="person",
        center=(5.0, 2.0, 0.0),
        size=(0.6, 0.6, 1.7),
        yaw=0.0,
        n_points=30,
        mean_conf=250.0,
        moving=False,
        vote_frac=0.0,
        speed_mps=None,
        safety_critical=False,
    )
    tracked = tracker.update([ped])
    assert tracked[0].safety_critical is True
    assert tracked[0].moving is True
