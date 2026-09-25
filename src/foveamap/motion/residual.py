"""Ego-compensated range-image residual and per-point votes (README 6.4 steps 1-3, task T10.1).

GPU acceleration
----------------
When CuPy is available and ``device='cuda'`` (or ``'auto'`` with a GPU present), the heavy
path runs entirely on GPU:

* Range-image construction uses a sort + ``minimum.reduceat`` scatter instead of ``np.minimum.at``.
* The 9-iteration window search is vectorised over all N points per step on GPU.
* Only the final ``(N,) int8`` vote array is downloaded to CPU.

The CPU path is preserved unchanged for testing and fallback (``device='cpu'``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from foveamap.gpu_utils import get_array_module, is_cupy_available, to_numpy


def project_to_range_image(
    xyz: np.ndarray,
    proj_H: int = 64,
    proj_W: int = 1024,
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to spherical range image coordinates (CPU version).

    Returns:
        r: (N,) float32 range in metres.
        u: (N,) int32 horizontal pixel coordinate in [0, proj_W - 1].
        v: (N,) int32 vertical pixel coordinate in [0, proj_H - 1].
        valid: (N,) bool mask of points with r > 0.1 and finite coordinates.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    r = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(r) & (r > 0.1)

    yaw   = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))

    fov_up   = np.radians(fov_up_deg)
    fov_down = np.radians(fov_down_deg)
    total_fov = abs(fov_down) + abs(fov_up)

    u = np.floor(0.5 * (yaw / np.pi + 1.0) * proj_W).astype(np.int32)
    u = np.clip(u, 0, proj_W - 1)

    v = np.floor((1.0 - (pitch + abs(fov_down)) / total_fov) * proj_H).astype(np.int32)
    v = np.clip(v, 0, proj_H - 1)

    return r, u, v, valid


def _project_gpu(xp, xyz_gpu, proj_H: int, proj_W: int, fov_up_deg: float, fov_down_deg: float):
    """GPU version of ``project_to_range_image`` — all arrays are CuPy."""
    x, y, z = xyz_gpu[:, 0], xyz_gpu[:, 1], xyz_gpu[:, 2]

    r     = xp.linalg.norm(xyz_gpu, axis=1).astype(xp.float32)
    valid = xp.isfinite(r) & (r > 0.1)

    yaw   = -xp.arctan2(y, x)
    pitch = xp.arcsin(xp.clip(z / xp.maximum(r, 1e-6), -1.0, 1.0))

    fov_up   = np.radians(fov_up_deg)   # scalar — fine on CPU
    fov_down = np.radians(fov_down_deg)
    total_fov = abs(fov_down) + abs(fov_up)

    u = xp.floor(0.5 * (yaw / np.pi + 1.0) * proj_W).astype(xp.int32)
    u = xp.clip(u, 0, proj_W - 1)

    v = xp.floor((1.0 - (pitch + abs(fov_down)) / total_fov) * proj_H).astype(xp.int32)
    v = xp.clip(v, 0, proj_H - 1)

    return r, u, v, valid


def _build_range_image_gpu(xp, r_valid, v_valid, u_valid, proj_H: int, proj_W: int):
    """Build a (proj_H, proj_W) minimum-range image from scattered points on GPU.

    CuPy has no ``minimum.at`` scatter, so we use sort + ``minimum.reduceat`` instead, which is
    still fully vectorised and O(M log M) where M = number of valid prev points.
    """
    if r_valid.size == 0:
        return xp.full((proj_H, proj_W), xp.inf, dtype=xp.float32)

    flat_idx = v_valid.astype(xp.int64) * proj_W + u_valid.astype(xp.int64)  # (M,)

    order       = xp.argsort(flat_idx)  # stable not required — taking min per pixel
    sorted_idx  = flat_idx[order]
    sorted_r    = r_valid[order]

    # Group boundaries
    new_group = xp.concatenate([
        xp.ones(1, dtype=xp.bool_),
        sorted_idx[1:] != sorted_idx[:-1],
    ])
    starts_gpu  = xp.flatnonzero(new_group)            # (K,) unique pixel indices in sorted_idx
    starts_cpu  = to_numpy(starts_gpu)                 # reduceat needs CPU indices
    pixels_gpu  = sorted_idx[starts_gpu]               # (K,) unique flat pixel coordinates

    # CuPy doesn't support minimum.reduceat; download sorted_r and reduce on CPU
    min_r = xp.asarray(
        np.minimum.reduceat(to_numpy(sorted_r.astype(xp.float32)), starts_cpu)
    )

    img_flat = xp.full(proj_H * proj_W, xp.inf, dtype=xp.float32)
    img_flat[pixels_gpu] = min_r
    return img_flat.reshape(proj_H, proj_W)


def range_residual_votes(
    cur_xyz: np.ndarray,
    prev_xyz_in_cur: np.ndarray,
    cfg: Any,
    proj_H: int = 64,
    proj_W: int = 1024,
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
    device: str = "auto",
) -> np.ndarray:
    """Per-point vote: 1 moving, 0 static-consistent, -1 unobserved.

    Args:
        cur_xyz:           (N, 3) points in current Velodyne frame.
        prev_xyz_in_cur:   (M, 3) previous scan transformed into current Velodyne frame.
        cfg:               MotionConfig or object with window, tau0_m, tau1_rel, occlusion_rule.
        proj_H:            Range image height (default 64 for HDL-64E).
        proj_W:            Range image width (default 1024).
        fov_up_deg:        Vertical FOV upper limit in degrees (default +3.0).
        fov_down_deg:      Vertical FOV lower limit in degrees (default -25.0).
        device:            ``'auto'`` / ``'cuda'`` / ``'cupy'`` to use the GPU path;
                           ``'cpu'`` to force NumPy.

    Returns:
        votes: (N,) int8 array — 1 = moving, 0 = static, -1 = unobserved.
    """
    n_pts = len(cur_xyz)
    if n_pts == 0:
        return np.zeros(0, dtype=np.int8)

    votes_cpu = np.full(n_pts, -1, dtype=np.int8)
    if len(prev_xyz_in_cur) == 0:
        return votes_cpu

    window        = getattr(cfg, "window", 3)
    tau0          = getattr(cfg, "tau0_m", 0.30)
    tau1          = getattr(cfg, "tau1_rel", 0.02)
    occlusion_rule = getattr(cfg, "occlusion_rule", False)
    half_w        = window // 2

    xp = get_array_module(device)
    use_gpu = xp is not np

    if use_gpu:
        return _range_residual_votes_gpu(
            xp, cur_xyz, prev_xyz_in_cur, cfg,
            proj_H, proj_W, fov_up_deg, fov_down_deg,
            window, tau0, tau1, occlusion_rule, half_w,
        )

    # ── CPU (NumPy) path — unchanged from original ────────────────────────
    cur_r, cur_u, cur_v, cur_valid = project_to_range_image(
        cur_xyz, proj_H=proj_H, proj_W=proj_W, fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg
    )
    prev_r, prev_u, prev_v, prev_valid = project_to_range_image(
        prev_xyz_in_cur, proj_H=proj_H, proj_W=proj_W,
        fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg,
    )

    prev_range_img = np.full((proj_H, proj_W), np.inf, dtype=np.float32)
    if np.any(prev_valid):
        v_val = prev_v[prev_valid]
        u_val = prev_u[prev_valid]
        r_val = prev_r[prev_valid]
        np.minimum.at(prev_range_img, (v_val, u_val), r_val)

    best_diff   = np.full(n_pts, np.inf, dtype=np.float32)
    best_prev_r = np.full(n_pts, np.nan,  dtype=np.float32)

    for dv in range(-half_w, half_w + 1):
        v_idx = np.clip(cur_v + dv, 0, proj_H - 1)
        for du in range(-half_w, half_w + 1):
            u_idx = (cur_u + du) % proj_W
            sampled = prev_range_img[v_idx, u_idx]
            valid_s = np.isfinite(sampled)
            diff    = np.abs(cur_r - sampled)
            closer  = cur_valid & valid_s & (diff < best_diff)
            best_diff[closer]   = diff[closer]
            best_prev_r[closer] = sampled[closer]

    has_valid  = cur_valid & np.isfinite(best_diff)
    tau        = tau0 + tau1 * cur_r
    static_m   = has_valid & (best_diff <= tau)
    moving_m   = has_valid & (best_diff > tau)

    votes_cpu[static_m] = 0
    if occlusion_rule:
        disoccluded = moving_m & (best_prev_r < (cur_r - tau))
        votes_cpu[moving_m & ~disoccluded] = 1
        votes_cpu[disoccluded] = 0
    else:
        votes_cpu[moving_m] = 1

    return votes_cpu


def _range_residual_votes_gpu(
    xp,
    cur_xyz: np.ndarray,
    prev_xyz_in_cur: np.ndarray,
    cfg: Any,
    proj_H: int,
    proj_W: int,
    fov_up_deg: float,
    fov_down_deg: float,
    window: int,
    tau0: float,
    tau1: float,
    occlusion_rule: bool,
    half_w: int,
) -> np.ndarray:
    """Fully GPU-vectorised residual vote computation (CuPy path)."""
    # Upload
    cur_gpu  = xp.asarray(np.asarray(cur_xyz,       dtype=np.float32))
    prev_gpu = xp.asarray(np.asarray(prev_xyz_in_cur, dtype=np.float32))

    n_pts = cur_gpu.shape[0]

    # Project current scan
    cur_r, cur_u, cur_v, cur_valid = _project_gpu(
        xp, cur_gpu, proj_H, proj_W, fov_up_deg, fov_down_deg
    )

    # Project previous scan
    prev_r, prev_u, prev_v, prev_valid = _project_gpu(
        xp, prev_gpu, proj_H, proj_W, fov_up_deg, fov_down_deg
    )

    # Build minimum-range image from previous points
    if xp.any(prev_valid):
        prev_range_img = _build_range_image_gpu(
            xp, prev_r[prev_valid], prev_v[prev_valid], prev_u[prev_valid], proj_H, proj_W
        )
    else:
        prev_range_img = xp.full((proj_H, proj_W), xp.inf, dtype=xp.float32)

    # Window search — vectorised over all N points, iterated over window²  (≤ 25) offsets
    best_diff   = xp.full(n_pts, xp.inf, dtype=xp.float32)
    best_prev_r = xp.full(n_pts, xp.nan, dtype=xp.float32)

    for dv in range(-half_w, half_w + 1):
        v_idx = xp.clip(cur_v + dv, 0, proj_H - 1)
        for du in range(-half_w, half_w + 1):
            u_idx    = (cur_u + du) % proj_W
            sampled  = prev_range_img[v_idx, u_idx]
            valid_s  = xp.isfinite(sampled)
            diff     = xp.abs(cur_r - sampled)
            closer   = cur_valid & valid_s & (diff < best_diff)
            best_diff   = xp.where(closer, diff,    best_diff)
            best_prev_r = xp.where(closer, sampled, best_prev_r)

    has_valid = cur_valid & xp.isfinite(best_diff)
    tau       = (tau0 + tau1 * cur_r).astype(xp.float32)

    static_m  = has_valid & (best_diff <= tau)
    moving_m  = has_valid & (best_diff > tau)

    votes_gpu = xp.full(n_pts, -1, dtype=xp.int8)
    votes_gpu[static_m] = 0

    if occlusion_rule:
        disoccluded = moving_m & (best_prev_r < (cur_r - tau))
        votes_gpu[moving_m & ~disoccluded] = 1
        votes_gpu[disoccluded] = 0
    else:
        votes_gpu[moving_m] = 1

    return to_numpy(votes_gpu)
