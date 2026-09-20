"""tests.test_hazards — Verification suite for synthetic hazard injection.

Verifies:
1. Pothole region Z-elevation depressed by exactly 15 cm.
2. Kerb region Z-elevation elevated by exactly 20 cm.
3. Overhang points placed at local ground + 1.5m within vehicle corridor.
4. Super-class and raw SemanticKITTI label preservation.
5. Empty and edge-case point clouds.
"""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.derive.hazards import HazardReport, inject_synthetic_hazards
from foveamap.io.labels import DRIVABLE, STATIC_OBSTACLE


@pytest.fixture
def flat_ground_plane() -> tuple[np.ndarray, np.ndarray]:
    """Generate a dense, perfectly flat ground plane at Z=0.0 in Velodyne frame."""
    # Regular 2D grid: [-20, 20]m with 10 cm resolution -> 401 x 401 = 160,801 points
    x = np.linspace(-20.0, 20.0, 401, dtype=np.float32)
    y = np.linspace(-20.0, 20.0, 401, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx, dtype=np.float32)
    intensity = np.full_like(xx, 0.5, dtype=np.float32)

    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel(), intensity.ravel()])
    lbls = np.full(len(pts), DRIVABLE, dtype=np.uint8)
    return pts, lbls


class TestSyntheticHazards:
    """Test suite for inject_synthetic_hazards."""

    def test_pothole_and_kerb_elevation_on_flat_plane(
        self, flat_ground_plane: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Pothole points must have Z = -0.15m and kerb points must have Z = +0.20m."""
        pts, lbls = flat_ground_plane
        pothole_center = (7.0, 0.0)
        pothole_radius = 1.0
        pothole_depth = 0.15
        kerb_x = (5.0, 15.0)
        kerb_y = (1.8, 2.5)
        kerb_height = 0.20

        out_pts, out_lbls, report = inject_synthetic_hazards(
            pts,
            lbls,
            pothole_center=pothole_center,
            pothole_radius_m=pothole_radius,
            pothole_depth_m=pothole_depth,
            kerb_x_range=kerb_x,
            kerb_y_range=kerb_y,
            kerb_height_m=kerb_height,
            num_overhang_points=100,
        )

        assert isinstance(report, HazardReport)
        assert report.pothole_points_modified > 0
        assert report.kerb_points_modified > 0
        assert report.overhang_points_added == 100

        # Points in original cloud (before overhang points appended)
        n_orig = len(pts)
        orig_out_pts = out_pts[:n_orig]

        # 1. Check Pothole region
        dx = orig_out_pts[:, 0] - pothole_center[0]
        dy = orig_out_pts[:, 1] - pothole_center[1]
        dist_sq = dx * dx + dy * dy
        pothole_mask = dist_sq <= (pothole_radius * pothole_radius)

        pothole_z = orig_out_pts[pothole_mask, 2]
        np.testing.assert_allclose(
            pothole_z,
            -pothole_depth,
            atol=1e-5,
            err_msg="All pothole points must be depressed by exactly -0.15m",
        )

        # 2. Check Kerb region (excluding any pothole overlap)
        kerb_mask = (
            (orig_out_pts[:, 0] >= kerb_x[0])
            & (orig_out_pts[:, 0] <= kerb_x[1])
            & (orig_out_pts[:, 1] >= kerb_y[0])
            & (orig_out_pts[:, 1] <= kerb_y[1])
            & (~pothole_mask)
        )
        kerb_z = orig_out_pts[kerb_mask, 2]
        np.testing.assert_allclose(
            kerb_z,
            kerb_height,
            atol=1e-5,
            err_msg="All kerb points must be raised by exactly +0.20m",
        )

        # 3. Check unaffected ground points
        unaffected_mask = (~pothole_mask) & (~kerb_mask)
        unaffected_z = orig_out_pts[unaffected_mask, 2]
        np.testing.assert_allclose(
            unaffected_z,
            0.0,
            atol=1e-5,
            err_msg="Points outside pothole and kerb must maintain original Z=0.0m",
        )

    def test_overhang_points_geometry_and_label(
        self, flat_ground_plane: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Overhang points must be placed at Z ~ 1.5m with STATIC_OBSTACLE label."""
        pts, lbls = flat_ground_plane
        num_overhang = 120

        out_pts, out_lbls, report = inject_synthetic_hazards(
            pts,
            lbls,
            overhang_x_range=(8.0, 9.5),
            overhang_y_range=(-1.5, 1.5),
            overhang_z_above_ground=1.5,
            num_overhang_points=num_overhang,
        )

        n_orig = len(pts)
        overhang_pts = out_pts[n_orig:]
        overhang_lbls = out_lbls[n_orig:]

        assert len(overhang_pts) == num_overhang
        assert len(overhang_lbls) == num_overhang

        # Overhang coordinates within requested bounds
        assert np.all(overhang_pts[:, 0] >= 8.0)
        assert np.all(overhang_pts[:, 0] <= 9.5)
        assert np.all(overhang_pts[:, 1] >= -1.5)
        assert np.all(overhang_pts[:, 1] <= 1.5)

        # Ground was at Z=0.0, so overhang Z ~ 1.5m ± small jitter (0.02m)
        assert np.isclose(np.mean(overhang_pts[:, 2]), 1.5, atol=0.05)
        assert np.all(np.abs(overhang_pts[:, 2] - 1.5) < 0.15)

        # Super-class label should be STATIC_OBSTACLE (2)
        assert np.all(overhang_lbls == STATIC_OBSTACLE)

    def test_empty_point_cloud(self) -> None:
        """Empty point cloud must return empty arrays without crashing."""
        empty_pts = np.zeros((0, 4), dtype=np.float32)
        empty_lbls = np.zeros((0,), dtype=np.uint8)

        out_pts, out_lbls, report = inject_synthetic_hazards(empty_pts, empty_lbls)
        assert len(out_pts) == 0
        assert len(out_lbls) == 0
        assert report.pothole_points_modified == 0
        assert report.kerb_points_modified == 0
        assert report.overhang_points_added == 0

    def test_raw_semkitti_uint32_labels_handled(self) -> None:
        """Packed SemanticKITTI uint32 labels are unpacked and overhang gets raw building ID."""
        pts = np.zeros((100, 3), dtype=np.float32)
        pts[:, 0] = np.linspace(6.0, 8.0, 100)
        pts[:, 1] = 0.0
        pts[:, 2] = -1.73  # Typical Velodyne ground elevation

        raw_labels = np.full(100, 40, dtype=np.uint32)

        out_pts, out_lbls, report = inject_synthetic_hazards(
            pts, raw_labels, num_overhang_points=25
        )

        assert len(out_pts) == 125
        assert len(out_lbls) == 125
        assert out_lbls.dtype == np.uint32

        # Pothole points were depressed from -1.73m by -0.15m -> -1.88m
        assert report.pothole_points_modified > 0

        # Overhang points should have raw SemanticKITTI obstacle ID (50 = building)
        overhang_lbls = out_lbls[100:]
        assert np.all(overhang_lbls == 50)
