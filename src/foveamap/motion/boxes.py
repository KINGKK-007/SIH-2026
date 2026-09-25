"""Minimum-area oriented boxes by angle sweep (README 6.4 step 7, task T10.2)."""

from __future__ import annotations

import numpy as np


def oriented_box(
    xyz: np.ndarray, angle_step_deg: float = 1.0
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return ``(center, size (l, w, h), yaw)`` of the minimum-area rectangle in the ground plane.

    Sweeps angles from 0 to 90 degrees in ``angle_step_deg`` increments, finds the rotation
    that minimizes the 2D bounding box area in the xy-plane, and combines with z-range.
    Yaw is returned in radians, aligned with the box's major dimension (l >= w).

    Args:
        xyz: (N, 3) point coordinates in metres.
        angle_step_deg: Angular resolution for the sweep in degrees (default 1.0).

    Returns:
        center: (cx, cy, cz) in metres.
        size: (l, w, h) in metres, where l >= w.
        yaw: Box orientation in radians in [-pi/2, pi/2].
    """
    n_pts = len(xyz)
    if n_pts == 0:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)

    xyz = np.asarray(xyz, dtype=np.float64)
    if n_pts == 1:
        p = xyz[0]
        return ((float(p[0]), float(p[1]), float(p[2])), (0.0, 0.0, 0.0), 0.0)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    cz = (z_min + z_max) / 2.0
    h = z_max - z_min

    # Angle sweep in [0, 90) degrees
    step = max(0.01, float(angle_step_deg))
    angles_deg = np.arange(0.0, 90.0, step, dtype=np.float64)
    angles_rad = np.radians(angles_deg)

    cos_a = np.cos(angles_rad)  # (K,)
    sin_a = np.sin(angles_rad)  # (K,)

    # Rotate 2D points: for each angle theta:
    # x_rot = x * cos(theta) + y * sin(theta)
    # y_rot = -x * sin(theta) + y * cos(theta)
    # Vectorized: (N, K)
    x_rot = np.outer(x, cos_a) + np.outer(y, sin_a)
    y_rot = -np.outer(x, sin_a) + np.outer(y, cos_a)

    x_min = np.min(x_rot, axis=0)  # (K,)
    x_max = np.max(x_rot, axis=0)
    y_min = np.min(y_rot, axis=0)
    y_max = np.max(y_rot, axis=0)

    lengths = x_max - x_min
    widths = y_max - y_min
    areas = lengths * widths

    best_idx = int(np.argmin(areas))
    theta = angles_rad[best_idx]

    c_x_rot = (x_min[best_idx] + x_max[best_idx]) / 2.0
    c_y_rot = (y_min[best_idx] + y_max[best_idx]) / 2.0

    # Rotate center back to original frame
    cx = c_x_rot * np.cos(theta) - c_y_rot * np.sin(theta)
    cy = c_x_rot * np.sin(theta) + c_y_rot * np.cos(theta)

    len_val = float(lengths[best_idx])
    width_val = float(widths[best_idx])

    if len_val >= width_val:
        l = len_val
        w = width_val
        yaw = float(theta)
    else:
        l = width_val
        w = len_val
        yaw = float(theta + np.pi / 2.0)

    # Normalize yaw to [-pi, pi]
    yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi

    return ((float(cx), float(cy), float(cz)), (float(l), float(w), float(h)), float(yaw))
