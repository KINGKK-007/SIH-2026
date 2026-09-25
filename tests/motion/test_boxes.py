"""Unit tests for minimum-area oriented bounding box in the ground plane (T10.2)."""

import numpy as np
import pytest

from foveamap.motion.boxes import oriented_box


def test_oriented_box_empty_and_single() -> None:
    center, size, yaw = oriented_box(np.zeros((0, 3)))
    assert center == (0.0, 0.0, 0.0)
    assert size == (0.0, 0.0, 0.0)
    assert yaw == 0.0

    p = np.array([[3.0, 4.0, 1.5]])
    center, size, yaw = oriented_box(p)
    assert center == (3.0, 4.0, 1.5)
    assert size == (0.0, 0.0, 0.0)
    assert yaw == 0.0


@pytest.mark.parametrize("true_yaw_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0])
def test_oriented_box_recovery_with_yaw(true_yaw_deg: float) -> None:
    """A synthetic 3D box of known dimensions and yaw should be recovered within angle_step_deg."""
    L, W, H = 4.5, 2.0, 1.6
    cx, cy, cz = 10.0, -5.0, 0.5
    theta = np.radians(true_yaw_deg)

    # Generate grid of points on the box surface / volume
    xs = np.linspace(-L / 2, L / 2, 20)
    ys = np.linspace(-W / 2, W / 2, 12)
    zs = np.linspace(-H / 2, H / 2, 8)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    local_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # Rotate by theta around z-axis and translate
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_x = local_pts[:, 0] * cos_t - local_pts[:, 1] * sin_t + cx
    rot_y = local_pts[:, 0] * sin_t + local_pts[:, 1] * cos_t + cy
    rot_z = local_pts[:, 2] + cz
    pts = np.column_stack([rot_x, rot_y, rot_z])

    angle_step = 0.5
    rec_center, rec_size, rec_yaw = oriented_box(pts, angle_step_deg=angle_step)

    # Center recovery
    assert np.isclose(rec_center[0], cx, atol=0.05)
    assert np.isclose(rec_center[1], cy, atol=0.05)
    assert np.isclose(rec_center[2], cz, atol=0.01)

    # Size recovery (L >= W)
    rec_l, rec_w, rec_h = rec_size
    assert np.isclose(rec_l, L, atol=0.1)
    assert np.isclose(rec_w, W, atol=0.1)
    assert np.isclose(rec_h, H, atol=0.01)

    # Yaw recovery: angle error within angle_step_deg + small numerical tolerance
    rec_yaw_deg = np.degrees(rec_yaw)
    angle_diff = abs((rec_yaw_deg - true_yaw_deg + 180) % 180 - 180)
    # Since boxes are symmetric under 180 deg, check modulo 180 or 90
    angle_diff_symm = min(abs(rec_yaw_deg - true_yaw_deg), abs(abs(rec_yaw_deg - true_yaw_deg) - 180))
    assert angle_diff_symm <= angle_step + 0.1
