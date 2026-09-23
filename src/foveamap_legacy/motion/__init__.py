"""foveamap.motion — Scan-to-scan residual motion detection and object clustering."""

from foveamap_legacy.motion.cluster import (
    ObjectBox,
    cluster_and_vote_motion,
    cluster_motion_candidates,
    is_movable_class,
)
from foveamap_legacy.motion.residual import (
    MotionResidualOutput,
    compute_motion_residuals,
    compute_relative_pose,
    transform_points,
)

__all__ = [
    "MotionResidualOutput",
    "compute_motion_residuals",
    "compute_relative_pose",
    "transform_points",
    "ObjectBox",
    "cluster_motion_candidates",
    "cluster_and_vote_motion",
    "is_movable_class",
]
