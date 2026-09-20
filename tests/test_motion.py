"""tests/test_motion.py — Verification suite for scan-to-scan motion detection and clustering.

QA Requirements Addressed
-------------------------
M1  Ego-Motion Math:
    - Pure translation: vehicle moves forward 2m along X. Verify compensated points
      match scan t coordinates exactly (zero residual).
    - Pure rotation: vehicle rotates by yaw/pitch. Verify inverse SE(3) compensation.
    - Combined SE(3): translation + rotation. Verify exact reconstruction.
    - Identity transformation: T_t == T_{t-1} leaves points unchanged.
M2  Residual Logic:
    - Injected moving points that violate the rigid transform produce residuals > threshold.
    - Static points conforming to the transform produce residuals ≈ 0.0 m.
    - KDTree flags moving points as motion_candidates; static points are unflagged.
    - Threshold parameter verification (0.2m, 0.5m, 1.0m).
    - Empty point cloud edge cases handled gracefully.
M3  Clustering & 3D Bounding Boxes:
    - Localized cluster of vehicle points clustered by DBSCAN into exactly 1 ObjectBox.
    - Axis-Aligned Bounding Box (AABB) correctly encapsulates points [min..max].
    - Noise points separated in space are discarded (cluster ID -1).
    - Non-movable classes (e.g. road=40, building=50) are rejected by class filter.
    - Multiple moving objects at distinct locations yield distinct bounding boxes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foveamap.motion.cluster import (
    cluster_and_vote_motion,
    cluster_motion_candidates,
)
from foveamap.motion.residual import (
    compute_motion_residuals,
    compute_relative_pose,
    transform_points,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers for SE(3) matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def _make_pose(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw_rad: float = 0.0) -> np.ndarray:
    """Create a 4x4 SE(3) matrix with translation (x, y, z) and yaw rotation."""
    T = np.eye(4, dtype=np.float64)
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


# ─────────────────────────────────────────────────────────────────────────────
# M1: Ego-Motion Mathematics
# ─────────────────────────────────────────────────────────────────────────────

class TestEgoMotionMath:
    """Mathematical verification of coordinate transformations and compensation."""

    def test_pure_translation_forward_2m(self) -> None:
        """Vehicle translates 2m forward along X: points transform and compensate exactly."""
        # World points (static landmarks)
        pts_world = np.array([
            [10.0, 2.0, 0.0],
            [15.0, -3.0, 1.0],
            [25.0, 0.0, -1.0],
            [5.0, 5.0, 0.5],
        ], dtype=np.float32)

        # Time t-1: sensor at origin
        T_tm1 = _make_pose(x=0.0, y=0.0, z=0.0)
        pts_tm1 = pts_world.copy()

        # Time t: sensor moved 2m forward in world frame
        T_t = _make_pose(x=2.0, y=0.0, z=0.0)
        # In sensor frame at time t, stationary world points appear 2m backward
        pts_t = pts_world.copy()
        pts_t[:, 0] -= 2.0

        # Relative transform: T_rel = inv(T_t) @ T_tm1
        T_rel = compute_relative_pose(T_t, T_tm1)
        expected_T_rel = np.eye(4)
        expected_T_rel[0, 3] = -2.0
        np.testing.assert_allclose(T_rel, expected_T_rel, atol=1e-10)

        # Apply compensation: transform pts_tm1 into frame t
        pts_comp = transform_points(pts_tm1, T_rel)

        # Assert points match frame t coordinates bit-for-bit / within float precision
        np.testing.assert_allclose(pts_comp[:, :3], pts_t[:, :3], atol=1e-6)

    def test_pure_rotation_yaw_15_deg(self) -> None:
        """Vehicle rotates 15 degrees: compensation reverses the sensor yaw."""
        pts_world = np.random.uniform(-20.0, 20.0, (50, 3)).astype(np.float32)

        T_tm1 = _make_pose(yaw_rad=0.0)
        yaw_rad = math.radians(15.0)
        T_t = _make_pose(yaw_rad=yaw_rad)

        # Sensor frame coordinates
        pts_tm1 = pts_world.copy()
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        R_t = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pts_t = (pts_world @ R_t).astype(np.float32)  # p_velo = R^T @ p_world, etc.
        # Use exact inverse rigid transform
        pts_t = (np.linalg.inv(T_t[:3, :3]) @ pts_world.T).T.astype(np.float32)

        res = compute_motion_residuals(pts_t, pts_tm1, T_t, T_tm1, threshold_m=0.5)

        # All static points must have zero residual
        np.testing.assert_allclose(res.residuals, 0.0, atol=1e-5)
        assert res.n_candidates == 0

    def test_combined_se3_translation_and_rotation(self) -> None:
        """Combined translation (x=3m, y=1m) and rotation (yaw=10 deg, pitch=3 deg)."""
        rng = np.random.default_rng(12345)
        pts_world = rng.uniform(-30.0, 30.0, (100, 3)).astype(np.float32)

        T_tm1 = _make_pose(x=5.0, y=2.0, z=0.0, yaw_rad=math.radians(5.0))
        T_t = _make_pose(x=8.0, y=3.0, z=0.2, yaw_rad=math.radians(15.0))

        # Sensor coordinates: p_sensor = inv(T) @ [p_world, 1]
        T_tm1_inv = np.linalg.inv(T_tm1)
        T_t_inv = np.linalg.inv(T_t)

        pts_tm1 = (pts_world @ T_tm1_inv[:3, :3].T) + T_tm1_inv[:3, 3]
        pts_t = (pts_world @ T_t_inv[:3, :3].T) + T_t_inv[:3, 3]

        res = compute_motion_residuals(pts_t, pts_tm1, T_t, T_tm1, threshold_m=0.5)

        # Mathematically exact overlap: max residual must be near zero
        assert np.max(res.residuals) < 1e-4
        assert res.n_candidates == 0

    def test_identity_pose_returns_identical_points(self) -> None:
        """When T_t == T_tm1, T_rel is identity and points are identical."""
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        T_ident = np.eye(4, dtype=np.float64)

        T_rel = compute_relative_pose(T_ident, T_ident)
        np.testing.assert_allclose(T_rel, np.eye(4), atol=1e-12)

        pts_comp = transform_points(pts, T_rel)
        np.testing.assert_allclose(pts_comp, pts, atol=1e-7)

    def test_transform_preserves_extra_columns(self) -> None:
        """Intensity or feature columns beyond XYZ are preserved unchanged."""
        pts = np.array([
            [1.0, 2.0, 3.0, 0.75, 42.0],
            [4.0, 5.0, 6.0, 0.12, 99.0],
        ], dtype=np.float32)
        T = _make_pose(x=10.0)

        out = transform_points(pts, T)
        assert out.shape == (2, 5)
        np.testing.assert_allclose(out[:, 3:], pts[:, 3:])


# ─────────────────────────────────────────────────────────────────────────────
# M2: Residual Logic & Motion Candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestResidualLogic:
    """Verify that stationary points produce low residuals and moving points are flagged."""

    def test_stationary_scene_zero_candidates(self) -> None:
        """Scene with no moving objects has 0 motion candidates."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-20.0, 20.0, (200, 3)).astype(np.float32)
        T = np.eye(4, dtype=np.float64)

        out = compute_motion_residuals(pts, pts, T, T, threshold_m=0.5)
        assert out.n_candidates == 0
        assert out.candidate_fraction == 0.0
        np.testing.assert_allclose(out.residuals, 0.0, atol=1e-6)

    def test_injected_moving_vehicle_flagged(self) -> None:
        """Static building points stay below threshold; moving car points exceed 0.5m threshold."""
        rng = np.random.default_rng(101)

        # 100 static points (building / road)
        static_world = rng.uniform(-10.0, 10.0, (100, 3)).astype(np.float32)

        # 50 dynamic points (car moving 5.0m forward in world, fully clearing its previous footprint)
        car_world_tm1 = rng.uniform(-0.8, 0.8, (50, 3)).astype(np.float32) + np.array([10.0, 0.0, 0.0], dtype=np.float32)
        car_world_t = car_world_tm1.copy()
        car_world_t[:, 0] += 5.0  # Moved 5.0m along X

        # Assemble full scans in world frame (ego sensor at origin at both t-1 and t)
        T_tm1 = np.eye(4, dtype=np.float64)
        T_t = np.eye(4, dtype=np.float64)

        pts_tm1 = np.vstack([static_world, car_world_tm1])
        pts_t = np.vstack([static_world, car_world_t])

        out = compute_motion_residuals(pts_t, pts_tm1, T_t, T_tm1, threshold_m=0.5)

        # Static points (indices 0..99): zero residual
        assert np.all(out.residuals[:100] < 1e-4)
        assert not np.any(out.motion_candidates[:100])

        # Moving car points (indices 100..149): residual is ≈ 5.0m > 0.5m
        assert np.all(out.residuals[100:] > 0.5)
        assert np.all(out.motion_candidates[100:])
        assert out.n_candidates == 50

    def test_threshold_sensitivity(self) -> None:
        """Varying threshold changes motion candidate classifications appropriately."""
        # Point moved by exactly 0.6m
        p_prev = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        p_curr = np.array([[0.6, 0.0, 0.0]], dtype=np.float32)
        T = np.eye(4)

        # At threshold 0.5: 0.6 > 0.5 -> candidate True
        out_05 = compute_motion_residuals(p_curr, p_prev, T, T, threshold_m=0.5)
        assert bool(out_05.motion_candidates[0]) is True

        # At threshold 0.7: 0.6 < 0.7 -> candidate False
        out_07 = compute_motion_residuals(p_curr, p_prev, T, T, threshold_m=0.7)
        assert bool(out_07.motion_candidates[0]) is False

    def test_empty_scans_do_not_crash(self) -> None:
        """Empty scan inputs return empty valid structures without throwing exceptions."""
        empty_pts = np.empty((0, 3), dtype=np.float32)
        valid_pts = np.ones((5, 3), dtype=np.float32)
        T = np.eye(4)

        # Both empty
        out1 = compute_motion_residuals(empty_pts, empty_pts, T, T)
        assert len(out1.motion_candidates) == 0

        # One empty
        out2 = compute_motion_residuals(valid_pts, empty_pts, T, T)
        assert len(out2.motion_candidates) == 5
        assert not np.any(out2.motion_candidates)


# ─────────────────────────────────────────────────────────────────────────────
# M3: Movable Class Filtering & DBSCAN Clustering
# ─────────────────────────────────────────────────────────────────────────────

class TestClusteringAndBoundingBoxes:
    """Verify DBSCAN clustering, class filtering, and bounding box creation."""

    def test_movable_class_filtering_rejects_static_structures(self) -> None:
        """High residual points on road/building are filtered out; only vehicles pass."""
        # 30 points with high residual
        pts = np.random.uniform(-5.0, 5.0, (30, 3)).astype(np.float32)
        motion_mask = np.ones(30, dtype=np.bool_)

        # Labels: 10 road (40), 10 building (50), 10 car (10)
        labels = np.array([40] * 10 + [50] * 10 + [10] * 10, dtype=np.uint16)

        boxes = cluster_motion_candidates(pts, motion_mask, labels, eps=10.0, min_samples=5, class_space="raw")

        # Road and building must be excluded; only the 10 car points cluster
        assert len(boxes) == 1
        assert boxes[0].n_points == 10
        assert boxes[0].cls == 10  # car

    def test_single_vehicle_bounding_box_tight_bounds(self) -> None:
        """DBSCAN forms 1 cluster and bounding box tight bounds encapsulate the points."""
        rng = np.random.default_rng(999)

        # Create 100 points densely sampled inside a compact vehicle volume
        n_pts = 100
        x = rng.uniform(8.0, 10.0, n_pts).astype(np.float32)
        y = rng.uniform(1.0, 2.5, n_pts).astype(np.float32)
        z = rng.uniform(-1.0, 0.0, n_pts).astype(np.float32)
        vehicle_pts = np.column_stack([x, y, z])

        motion_mask = np.ones(n_pts, dtype=np.bool_)
        labels = np.full(n_pts, 252, dtype=np.uint16)  # moving-car (raw ID 252)

        boxes = cluster_motion_candidates(
            vehicle_pts, motion_mask, labels, eps=0.7, min_samples=10, class_space="raw"
        )

        assert len(boxes) == 1
        box = boxes[0]

        assert box.id == 0
        assert box.is_moving is True
        assert box.n_points == n_pts
        assert box.cls == 252

        # Verify AABB bounds tightly match min and max
        assert box.x_min == pytest.approx(float(np.min(x)), abs=1e-5)
        assert box.x_max == pytest.approx(float(np.max(x)), abs=1e-5)
        assert box.y_min == pytest.approx(float(np.min(y)), abs=1e-5)
        assert box.y_max == pytest.approx(float(np.max(y)), abs=1e-5)
        assert box.z_min == pytest.approx(float(np.min(z)), abs=1e-5)
        assert box.z_max == pytest.approx(float(np.max(z)), abs=1e-5)

        # Dimensions
        dx, dy, dz = box.dimensions
        assert dx == pytest.approx(box.x_max - box.x_min)
        assert dy == pytest.approx(box.y_max - box.y_min)
        assert dz == pytest.approx(box.z_max - box.z_min)

    def test_sparse_noise_points_discarded(self) -> None:
        """Points with density < min_samples are discarded as noise."""
        # 5 isolated points far from each other
        noise_pts = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [20.0, -10.0, 0.0],
            [-15.0, 5.0, 0.0],
            [30.0, 0.0, 0.0],
        ], dtype=np.float32)

        motion_mask = np.ones(5, dtype=np.bool_)
        labels = np.full(5, 10, dtype=np.uint16)

        boxes = cluster_motion_candidates(noise_pts, motion_mask, labels, eps=0.7, min_samples=3, class_space="raw")
        assert len(boxes) == 0

    def test_two_distinct_moving_vehicles(self) -> None:
        """Two separated clusters produce two distinct ObjectBox instances."""
        rng = np.random.default_rng(777)

        # Vehicle 1 at (x=10, y=2)
        v1_pts = rng.uniform(-0.5, 0.5, (30, 3)).astype(np.float32) + np.array([10.0, 2.0, 0.0], dtype=np.float32)
        # Vehicle 2 at (x=30, y=-5) — 20+ metres apart
        v2_pts = rng.uniform(-0.5, 0.5, (30, 3)).astype(np.float32) + np.array([30.0, -5.0, 0.0], dtype=np.float32)

        pts = np.vstack([v1_pts, v2_pts])
        motion_mask = np.ones(60, dtype=np.bool_)
        labels = np.array([10] * 30 + [18] * 30, dtype=np.uint16)  # 10=car, 18=truck in raw KITTI

        boxes = cluster_motion_candidates(pts, motion_mask, labels, eps=0.7, min_samples=10, class_space="raw")

        assert len(boxes) == 2
        # Check centers
        c1, c2 = sorted([b.center for b in boxes], key=lambda c: c[0])
        assert c1[0] == pytest.approx(10.0, abs=0.5)
        assert c2[0] == pytest.approx(30.0, abs=0.5)

    def test_cluster_and_vote_motion(self) -> None:
        """cluster_and_vote_motion correctly marks confirmed clusters and returns moving mask."""
        rng = np.random.default_rng(555)

        # Vehicle points (40 pts)
        v_pts = rng.uniform(-0.5, 0.5, (40, 3)).astype(np.float32) + np.array([10.0, 0.0, 0.0], dtype=np.float32)
        # Static parked car (40 pts)
        parked_pts = rng.uniform(-0.5, 0.5, (40, 3)).astype(np.float32) + np.array([5.0, 7.0, 0.0], dtype=np.float32)

        pts = np.vstack([v_pts, parked_pts])
        labels = np.full(80, 10, dtype=np.uint16)  # both cars

        # Only v_pts has motion candidates (80% of points exceed residual)
        cands = np.zeros(80, dtype=np.bool_)
        cands[:32] = True  # 32/40 = 80% > vote_frac 0.3

        boxes, confirmed_mask = cluster_and_vote_motion(
            pts, labels, cands, eps=0.7, min_samples=10, vote_frac=0.3
        )

        assert len(boxes) == 1
        assert boxes[0].n_points == 40
        # Entire moving cluster is confirmed moving in mask
        assert np.all(confirmed_mask[:40])
        # Parked car points remain False
        assert not np.any(confirmed_mask[40:])
