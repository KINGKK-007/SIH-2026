"""foveamap.motion.residual — Scan-to-scan geometric motion residual detection.

Implements geometric ego-motion compensation and nearest-neighbour residual
distance calculation between consecutive LiDAR scans.

Math & Geometry
---------------
1. **Ego-motion compensation:**
   Given Velodyne-frame sensor poses :math:`T_{velo}(t)` and :math:`T_{velo}(t-1)`
   (computed via :func:`~foveamap.io.poses.load_poses`), the relative transform
   mapping coordinates from frame :math:`t-1` into frame :math:`t` is:

   .. math::
       T_{rel} = T_{velo}(t)^{-1} \\cdot T_{velo}(t-1)

   The previous scan points are transformed into the current frame:

   .. math::
       p_{t-1}^{(t)} = R_{rel} \\cdot p_{t-1} + t_{rel}

2. **Residual calculation:**
   Using a spatial :class:`scipy.spatial.cKDTree` built on the compensated previous
   scan :math:`p_{t-1}^{(t)}`, we query the nearest-neighbour Euclidean distance
   :math:`d_i` for every point in the current scan :math:`p_{t, i}`.

   Points with distance :math:`d_i > \\text{threshold}` (default 0.5 m) indicate
   that no corresponding physical surface was present in that location in the
   previous scan, flagging them as ``motion_candidates``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class MotionResidualOutput:
    """Output of the scan-to-scan geometric residual calculation.

    Attributes
    ----------
    motion_candidates : NDArray[bool]
        Boolean mask of shape ``(N_t,)``. True for points whose nearest-neighbour
        distance to the ego-compensated previous scan exceeds ``threshold_m``.
    residuals : NDArray[float32]
        Nearest-neighbour Euclidean distance (in metres) for each point in
        ``points_t``, shape ``(N_t,)``.
    compensated_prev_points : NDArray[float32]
        Points from scan :math:`t-1` transformed into scan :math:`t`'s coordinate
        frame, shape ``(N_{t-1}, 3)`` or ``(N_{t-1}, 4)``.
    T_rel : NDArray[float64]
        4x4 relative transformation matrix :math:`T_{velo}(t)^{-1} \\cdot T_{velo}(t-1)`.
    threshold_m : float
        The distance threshold used to classify candidates.
    """

    motion_candidates: NDArray[np.bool_]
    residuals: NDArray[np.float32]
    compensated_prev_points: NDArray[np.float32]
    T_rel: NDArray[np.float64]
    threshold_m: float

    @property
    def n_candidates(self) -> int:
        """Total number of motion candidate points."""
        return int(np.sum(self.motion_candidates))

    @property
    def candidate_fraction(self) -> float:
        """Fraction of points in current scan flagged as motion candidates."""
        if len(self.motion_candidates) == 0:
            return 0.0
        return float(np.mean(self.motion_candidates))


def transform_points(
    points: NDArray[np.floating],
    T: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Apply a 4x4 rigid transformation to a point cloud.

    Parameters
    ----------
    points : NDArray[floating]
        Shape ``(N, 3)`` or ``(N, 4)`` [x, y, z, ...].
    T : NDArray[floating]
        Shape ``(4, 4)`` homogeneous transformation matrix.

    Returns
    -------
    transformed : NDArray[float32]
        Same shape as ``points``, with XYZ rotated and translated by ``T``.
        Additional columns (e.g. intensity) are preserved unchanged.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.copy()

    T_mat = np.asarray(T, dtype=np.float64)
    R = T_mat[:3, :3]
    t = T_mat[:3, 3]

    xyz = pts[:, :3].astype(np.float64)
    # Vectorised transform: (N, 3) @ R.T + t
    xyz_trans = (xyz @ R.T) + t

    out = pts.copy()
    out[:, :3] = xyz_trans.astype(np.float32)
    return out


def compute_relative_pose(
    T_velo_t: NDArray[np.floating],
    T_velo_tm1: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Compute relative transformation mapping frame t-1 into frame t.

    .. math::
        T_{rel} = T_{velo}(t)^{-1} \\cdot T_{velo}(t-1)

    Parameters
    ----------
    T_velo_t : NDArray[floating]
        4x4 sensor pose at time :math:`t`.
    T_velo_tm1 : NDArray[floating]
        4x4 sensor pose at time :math:`t-1`.

    Returns
    -------
    T_rel : NDArray[float64]
        4x4 relative transformation matrix.
    """
    T_t = np.asarray(T_velo_t, dtype=np.float64)
    T_prev = np.asarray(T_velo_tm1, dtype=np.float64)

    T_t_inv = np.linalg.inv(T_t)
    T_rel = T_t_inv @ T_prev
    return T_rel


def compute_motion_residuals(
    points_t: NDArray[np.floating],
    points_tm1: NDArray[np.floating],
    T_velo_t: NDArray[np.floating],
    T_velo_tm1: NDArray[np.floating],
    threshold_m: float = 0.5,
) -> MotionResidualOutput:
    """Compute scan-to-scan geometric motion residuals.

    Parameters
    ----------
    points_t : NDArray[floating]
        Current scan point cloud at time :math:`t`, shape ``(N_t, 3)`` or ``(N_t, 4)``.
    points_tm1 : NDArray[floating]
        Previous scan point cloud at time :math:`t-1`, shape ``(N_{t-1}, 3)`` or ``(N_{t-1}, 4)``.
    T_velo_t : NDArray[floating]
        4x4 Velodyne-frame pose at time :math:`t`.
    T_velo_tm1 : NDArray[floating]
        4x4 Velodyne-frame pose at time :math:`t-1`.
    threshold_m : float, default 0.5
        Distance threshold in metres above which a point is marked as a
        motion candidate.

    Returns
    -------
    MotionResidualOutput
        Contains boolean mask ``motion_candidates``, distance array ``residuals``,
        the compensated previous points, and ``T_rel``.
    """
    pts_t = np.asarray(points_t, dtype=np.float32)
    pts_tm1 = np.asarray(points_tm1, dtype=np.float32)
    N_t = len(pts_t)

    # Compute relative ego-motion transform
    T_rel = compute_relative_pose(T_velo_t, T_velo_tm1)

    if N_t == 0 or len(pts_tm1) == 0:
        return MotionResidualOutput(
            motion_candidates=np.zeros(N_t, dtype=np.bool_),
            residuals=np.zeros(N_t, dtype=np.float32),
            compensated_prev_points=np.empty((0, 3), dtype=np.float32),
            T_rel=T_rel,
            threshold_m=threshold_m,
        )

    # Ego-motion compensation: transform previous scan points into frame t
    prev_comp = transform_points(pts_tm1, T_rel)

    # Spatial query: build KDTree on compensated previous XYZ
    tree = cKDTree(prev_comp[:, :3])
    distances, _ = tree.query(pts_t[:, :3], k=1, workers=-1)
    residuals = distances.astype(np.float32)

    # Residual thresholding
    motion_candidates = residuals > float(threshold_m)

    return MotionResidualOutput(
        motion_candidates=motion_candidates,
        residuals=residuals,
        compensated_prev_points=prev_comp,
        T_rel=T_rel,
        threshold_m=threshold_m,
    )
