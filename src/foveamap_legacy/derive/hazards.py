"""foveamap.derive.hazards — Synthetic hazard injection for terrain & clearance evaluation.

Per master plan §5.6:
SemanticKITTI has no ground truth for pothole depth, kerb height, or low overhangs.
We programmatically inject calibrated synthetic hazards into real or synthetic scans:
1. Pothole : Drop Z by 15 cm inside a 1.0 m radius disc on drivable ground.
2. Kerb    : Raise Z by 20 cm along a 0.5 m wide linear boundary along the roadway.
3. Overhang: Insert synthetic points forming a low overhead obstacle at 1.5 m height
             (low branch or bridge) directly in the ego vehicle's corridor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from foveamap_legacy.io.labels import (
    DRIVABLE,
    STATIC_OBSTACLE,
    to_superclass,
    unpack_kitti_labels,
)


@dataclass(frozen=True)
class HazardReport:
    """Summary of injected synthetic hazards."""

    pothole_center: tuple[float, float]
    pothole_radius_m: float
    pothole_depth_m: float
    pothole_points_modified: int

    kerb_x_range: tuple[float, float]
    kerb_y_range: tuple[float, float]
    kerb_height_m: float
    kerb_points_modified: int

    overhang_bounds: tuple[float, float, float, float, float, float]
    overhang_points_added: int


def inject_synthetic_hazards(
    points: NDArray[np.floating],
    labels: NDArray,
    poses: Any = None,
    pothole_center: tuple[float, float] = (7.0, 0.0),
    pothole_radius_m: float = 1.0,
    pothole_depth_m: float = 0.15,
    kerb_x_range: tuple[float, float] = (5.0, 15.0),
    kerb_y_range: tuple[float, float] = (1.8, 2.5),
    kerb_height_m: float = 0.20,
    overhang_x_range: tuple[float, float] = (8.0, 9.5),
    overhang_y_range: tuple[float, float] = (-1.5, 1.5),
    overhang_z_above_ground: float = 1.5,
    num_overhang_points: int = 150,
) -> tuple[NDArray[np.float32], NDArray, HazardReport]:
    """Inject synthetic pothole, kerb, and low overhang hazards into a point cloud.

    Parameters
    ----------
    points : NDArray[floating]
        Shape ``(N, 3)`` or ``(N, 4)`` in Velodyne frame.
    labels : NDArray
        Shape ``(N,)`` semantic labels (raw SemanticKITTI or super-classes).
    poses : Any, optional
        Sensor pose matrix or trajectory (unused for single-scan injection,
        reserved for world-frame trajectory alignment).
    pothole_center : tuple[float, float], default (7.0, 0.0)
        (x, y) center coordinates of the pothole in metres.
    pothole_radius_m : float, default 1.0
        Pothole radius in metres.
    pothole_depth_m : float, default 0.15
        Depth to drop ground points inside the pothole (15 cm).
    kerb_x_range : tuple[float, float], default (5.0, 15.0)
        Longitudinal bounds [x_min, x_max] for the kerb line.
    kerb_y_range : tuple[float, float], default (1.8, 2.5)
        Lateral bounds [y_min, y_max] for the kerb line.
    kerb_height_m : float, default 0.20
        Height step to raise ground points along the kerb (20 cm).
    overhang_x_range : tuple[float, float], default (8.0, 9.5)
        Corridor longitudinal bounds for the low overhead obstacle.
    overhang_y_range : tuple[float, float], default (-1.5, 1.5)
        Corridor lateral bounds for the low overhead obstacle.
    overhang_z_above_ground : float, default 1.5
        Height of the overhang above local ground level (1.5 m < 2.0 m vehicle height).
    num_overhang_points : int, default 150
        Number of synthetic points to generate on the overhang surface.

    Returns
    -------
    out_points : NDArray[float32]
        Shape ``(N + num_overhang_points, 3)`` or ``(N + num_overhang_points, 4)``.
    out_labels : NDArray
        Shape ``(N + num_overhang_points,)`` with original dtype preserved.
    report : HazardReport
        Summary of modified and added hazard points.
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    lbls = np.asarray(labels).copy()
    N = len(pts)

    if N == 0:
        empty_report = HazardReport(
            pothole_center=pothole_center,
            pothole_radius_m=pothole_radius_m,
            pothole_depth_m=pothole_depth_m,
            pothole_points_modified=0,
            kerb_x_range=kerb_x_range,
            kerb_y_range=kerb_y_range,
            kerb_height_m=kerb_height_m,
            kerb_points_modified=0,
            overhang_bounds=(0, 0, 0, 0, 0, 0),
            overhang_points_added=0,
        )
        return pts, lbls, empty_report

    # Detect ground points (z < -1.0 or super-class DRIVABLE / TERRAIN)
    if lbls.dtype == np.uint32 or np.max(lbls) > 255:
        sem_ids, _ = unpack_kitti_labels(lbls.astype(np.uint32))
        sc = to_superclass(sem_ids)
        is_ground = (sc == DRIVABLE) | (pts[:, 2] < -1.0)
    elif np.max(lbls) <= 3:
        is_ground = (lbls == DRIVABLE) | (pts[:, 2] < -1.0)
    else:
        is_ground = pts[:, 2] < -1.0

    # Local ground reference elevation (typically ~ -1.73 m in Velodyne frame)
    ground_z_ref = float(np.median(pts[is_ground, 2])) if np.any(is_ground) else -1.73

    # ── 1. Inject Pothole ────────────────────────────────────────────────────
    # Drop Z by 15 cm within radius R of center
    dx = pts[:, 0] - pothole_center[0]
    dy = pts[:, 1] - pothole_center[1]
    dist_sq = dx * dx + dy * dy
    pothole_mask = is_ground & (dist_sq <= (pothole_radius_m * pothole_radius_m))
    pts[pothole_mask, 2] -= float(pothole_depth_m)
    n_pothole = int(np.sum(pothole_mask))

    # ── 2. Inject Kerb ───────────────────────────────────────────────────────
    # Raise Z by 20 cm along linear strip
    kerb_mask = (
        is_ground
        & (pts[:, 0] >= kerb_x_range[0])
        & (pts[:, 0] <= kerb_x_range[1])
        & (pts[:, 1] >= kerb_y_range[0])
        & (pts[:, 1] <= kerb_y_range[1])
        & (~pothole_mask)  # Don't overlap with pothole
    )
    pts[kerb_mask, 2] += float(kerb_height_m)
    n_kerb = int(np.sum(kerb_mask))

    # ── 3. Inject Overhang ───────────────────────────────────────────────────
    # Low branch or barrier: Z = ground_z_ref + 1.5m directly in vehicle path
    overhang_z = ground_z_ref + overhang_z_above_ground

    rng = np.random.default_rng(42)
    ox = rng.uniform(overhang_x_range[0], overhang_x_range[1], num_overhang_points).astype(np.float32)
    oy = rng.uniform(overhang_y_range[0], overhang_y_range[1], num_overhang_points).astype(np.float32)
    oz = np.full(num_overhang_points, overhang_z, dtype=np.float32) + rng.normal(0, 0.02, num_overhang_points).astype(np.float32)

    has_extra_cols = (pts.shape[1] > 3)
    if has_extra_cols:
        extra_dim = pts.shape[1] - 3
        extra_cols = np.full((num_overhang_points, extra_dim), 0.5, dtype=np.float32)
        overhang_pts = np.column_stack([ox, oy, oz, extra_cols])
    else:
        overhang_pts = np.column_stack([ox, oy, oz])

    # Assign obstacle label
    if lbls.dtype == np.uint32 or np.max(lbls) > 255:
        # Raw SemanticKITTI ID: 50 (building / structure) or 70 (vegetation)
        overhang_lbl = np.full(num_overhang_points, 50, dtype=lbls.dtype)
    elif np.max(lbls) <= 3:
        overhang_lbl = np.full(num_overhang_points, STATIC_OBSTACLE, dtype=lbls.dtype)
    else:
        # 19-class ID: 2 (vegetation) or 17 (building)
        overhang_lbl = np.full(num_overhang_points, 17, dtype=lbls.dtype)

    out_points = np.vstack([pts, overhang_pts])
    out_labels = np.concatenate([lbls, overhang_lbl])

    report = HazardReport(
        pothole_center=pothole_center,
        pothole_radius_m=pothole_radius_m,
        pothole_depth_m=pothole_depth_m,
        pothole_points_modified=n_pothole,
        kerb_x_range=kerb_x_range,
        kerb_y_range=kerb_y_range,
        kerb_height_m=kerb_height_m,
        kerb_points_modified=n_kerb,
        overhang_bounds=(
            overhang_x_range[0],
            overhang_x_range[1],
            overhang_y_range[0],
            overhang_y_range[1],
            float(overhang_z - 0.1),
            float(overhang_z + 0.1),
        ),
        overhang_points_added=num_overhang_points,
    )

    return out_points, out_labels, report
