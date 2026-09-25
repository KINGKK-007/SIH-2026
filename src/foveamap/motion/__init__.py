"""Geometric motion estimation and object boxes (README 6.4, 9.4)."""

from foveamap.motion.boxes import oriented_box
from foveamap.motion.cluster import cluster_points
from foveamap.motion.pipeline import estimate_motion
from foveamap.motion.residual import range_residual_votes
from foveamap.motion.tracker import ClusterTracker

__all__ = [
    "ClusterTracker",
    "cluster_points",
    "estimate_motion",
    "oriented_box",
    "range_residual_votes",
]
