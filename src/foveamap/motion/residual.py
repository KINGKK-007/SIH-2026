"""Ego-compensated range-image residual and per-point votes (README 6.4 steps 1-3, task T10.1)."""

from __future__ import annotations

from typing import Any

import numpy as np


def project_to_range_image(
    xyz: np.ndarray,
    proj_H: int = 64,
    proj_W: int = 1024,
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to spherical range image coordinates.

    Returns:
        r: (N,) float32 range in metres.
        u: (N,) int32 horizontal pixel coordinate in [0, proj_W - 1].
        v: (N,) int32 vertical pixel coordinate in [0, proj_H - 1].
        valid: (N,) bool mask of points with r > 0.1 and finite coordinates.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(r) & (r > 0.1)

    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))

    fov_up = np.radians(fov_up_deg)
    fov_down = np.radians(fov_down_deg)
    total_fov = abs(fov_down) + abs(fov_up)

    # Horizontal coordinate [0, W - 1]
    u = np.floor(0.5 * (yaw / np.pi + 1.0) * proj_W).astype(np.int32)
    u = np.clip(u, 0, proj_W - 1)

    # Vertical coordinate [0, H - 1]
    v = np.floor((1.0 - (pitch + abs(fov_down)) / total_fov) * proj_H).astype(np.int32)
    v = np.clip(v, 0, proj_H - 1)

    return r, u, v, valid


def range_residual_votes(
    cur_xyz: np.ndarray,
    prev_xyz_in_cur: np.ndarray,
    cfg: Any,
    proj_H: int = 64,
    proj_W: int = 1024,
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
) -> np.ndarray:
    """Per-point vote: 1 moving, 0 static-consistent, -1 unobserved.

    Args:
        cur_xyz: (N, 3) points in current Velodyne frame.
        prev_xyz_in_cur: (M, 3) previous scan transformed into current Velodyne frame.
        cfg: MotionConfig or object with window, tau0_m, tau1_rel, occlusion_rule, same_object_radius_m.
        proj_H: Range image height (default 64 for HDL-64E).
        proj_W: Range image width (default 1024).
        fov_up_deg: Vertical FOV upper limit in degrees (default +3.0).
        fov_down_deg: Vertical FOV lower limit in degrees (default -25.0).

    Returns:
        votes: (N,) int8 array where 1 = moving, 0 = static, -1 = unobserved.
    """
    n_pts = len(cur_xyz)
    if n_pts == 0:
        return np.zeros(0, dtype=np.int8)

    votes = np.full(n_pts, -1, dtype=np.int8)
    if len(prev_xyz_in_cur) == 0:
        return votes

    window = getattr(cfg, "window", 3)
    tau0 = getattr(cfg, "tau0_m", 0.30)
    tau1 = getattr(cfg, "tau1_rel", 0.02)
    occlusion_rule = getattr(cfg, "occlusion_rule", False)

    # Project current points
    cur_r, cur_u, cur_v, cur_valid = project_to_range_image(
        cur_xyz, proj_H=proj_H, proj_W=proj_W, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg
    )

    # Project previous points
    prev_r, prev_u, prev_v, prev_valid = project_to_range_image(
        prev_xyz_in_cur, proj_H=proj_H, proj_W=proj_W, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg
    )

    # Build previous range image with nearest depth at each pixel
    prev_range_img = np.full((proj_H, proj_W), np.inf, dtype=np.float32)
    if np.any(prev_valid):
        v_valid = prev_v[prev_valid]
        u_valid = prev_u[prev_valid]
        r_valid = prev_r[prev_valid]
        np.minimum.at(prev_range_img, (v_valid, u_valid), r_valid)

    # Window search: min |cur_r - r_prev| over win x win window
    half_w = window // 2
    best_diff = np.full(n_pts, np.inf, dtype=np.float32)
    best_prev_r = np.full(n_pts, np.nan, dtype=np.float32)

    for dv in range(-half_w, half_w + 1):
        v_idx = np.clip(cur_v + dv, 0, proj_H - 1)
        for du in range(-half_w, half_w + 1):
            u_idx = (cur_u + du) % proj_W
            sampled_prev_r = prev_range_img[v_idx, u_idx]
            valid_sample = np.isfinite(sampled_prev_r)
            diff = np.abs(cur_r - sampled_prev_r)

            closer = cur_valid & valid_sample & (diff < best_diff)
            best_diff[closer] = diff[closer]
            best_prev_r[closer] = sampled_prev_r[closer]

    has_valid = cur_valid & np.isfinite(best_diff)
    tau = tau0 + tau1 * cur_r

    # Static-consistent: d <= tau(r)
    static_mask = has_valid & (best_diff <= tau)
    votes[static_mask] = 0

    # Moving candidates: d > tau(r)
    moving_mask = has_valid & (best_diff > tau)

    if occlusion_rule:
        # Occlusion handling: if previous view was nearer, it was occluded
        disoccluded = moving_mask & (best_prev_r < (cur_r - tau))
        # Points that are disoccluded are static-consistent (no motion vote)
        votes[moving_mask & ~disoccluded] = 1
        votes[disoccluded] = 0
    else:
        votes[moving_mask] = 1

    return votes
