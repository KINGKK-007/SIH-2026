"""foveamap.motion.cluster — DBSCAN clustering of moving objects and bounding box estimation.

Implements object clustering and 3D Axis-Aligned Bounding Box (AABB) extraction
for motion candidates filtered by movable classes (vehicles, pedestrians, cyclists).

Math & Strategy
---------------
1. **Movable class filtering:**
   Residual motion noise often appears at the edges of static buildings or tree
   foliage due to LiDAR beam divergence and small calibration inaccuracies.
   To eliminate false positives, motion candidates are filtered by semantic class:
   only points belonging to movable categories (cars, trucks, buses, motorcycles,
   bicycles, pedestrians, motorcyclists) are admitted into the clustering stage.

2. **DBSCAN Clustering:**
   :class:`sklearn.cluster.DBSCAN` groups spatially coherent candidate points
   without requiring a predetermined number of objects (unlike k-means).
   Noise points (:math:`-1`) are discarded.

3. **3D Bounding Box Extraction:**
   For each valid cluster, an axis-aligned 3D bounding box (AABB) is computed
   along with majority-vote semantic classification, point count, and moving status.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN

from foveamap.io.labels import (
    DYNAMIC,
    STATIC_OBSTACLE,
    UNKNOWN,
)

# SemanticKITTI raw IDs for movable classes:
# Vehicles (static + moving variants) and VRUs
_MOVABLE_RAW_IDS: frozenset[int] = frozenset({
    # Vehicles (static annotations)
    10,  # car
    11,  # bicycle
    13,  # bus
    15,  # motorcycle
    16,  # on-rails
    18,  # truck
    20,  # other-vehicle
    # VRUs
    30,  # person
    31,  # bicyclist
    32,  # motorcyclist
    # Vehicles & VRUs (moving annotations)
    252,  # moving-car
    253,  # moving-bicyclist
    254,  # moving-person
    255,  # moving-motorcyclist
    256,  # moving-on-rails
    257,  # moving-bus
    258,  # moving-truck
    259,  # moving-other-vehicle
})

# 19-class benchmark IDs for movable classes
_MOVABLE_19_IDS: frozenset[int] = frozenset({
    0,   # car
    1,   # bicycle
    3,   # motorcycle
    4,   # bus
    5,   # on-rails
    6,   # truck
    7,   # other-vehicle
    8,   # person
    9,   # bicyclist
    10,  # motorcyclist
})


@dataclass(frozen=True)
class ObjectBox:
    """3D Axis-Aligned Bounding Box (AABB) for a detected moving object.

    Attributes
    ----------
    id : int
        Cluster identifier.
    cls : int | str
        Semantic class ID or name (super-class or original class).
    x_min, x_max : float
        Longitudinal bounds in metres (Velodyne frame: +x forward).
    y_min, y_max : float
        Lateral bounds in metres (Velodyne frame: +y left).
    z_min, z_max : float
        Vertical bounds in metres (Velodyne frame: +z up).
    is_moving : bool
        Whether this object is confirmed moving (default True).
    n_points : int
        Number of LiDAR points supporting this cluster.
    confidence : float
        Confidence of cluster classification in ``[0, 1]``.
    """

    id: int
    cls: int | str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    is_moving: bool = True
    n_points: int = 0
    confidence: float = 1.0

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return bounds as ``(x_min, x_max, y_min, y_max, z_min, z_max)``."""
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    @property
    def center(self) -> tuple[float, float, float]:
        """Return cluster centroid ``(cx, cy, cz)``."""
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
            (self.z_min + self.z_max) / 2.0,
        )

    @property
    def dimensions(self) -> tuple[float, float, float]:
        """Return bounding box dimensions ``(dx, dy, dz)``: length, width, height."""
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )

    @property
    def volume(self) -> float:
        """Return 3D bounding box volume in cubic metres."""
        dx, dy, dz = self.dimensions
        return float(dx * dy * dz)


def is_movable_class(
    labels: NDArray,
    class_space: str = "auto",
) -> NDArray[np.bool_]:
    """Identify which points belong to physically movable classes.

    Parameters
    ----------
    labels : NDArray
        Array of semantic IDs (raw SemanticKITTI IDs, 19-class IDs, or super-class IDs).
    class_space : {"auto", "raw", "19class", "superclass"}, default "auto"
        Format of ``labels``. When "auto", the format is inferred from the max label.

    Returns
    -------
    movable_mask : NDArray[bool]
        Boolean array of shape ``(N,)``; True for movable classes.
    """
    arr = np.asarray(labels)
    N = len(arr)
    if N == 0:
        return np.empty(0, dtype=np.bool_)

    max_val = int(np.max(arr))

    if class_space == "auto":
        if max_val > 255:
            # Packed SemanticKITTI labels (upper 16 = instance, lower 16 = semantic)
            class_space = "raw"
            arr = (arr & 0xFFFF).astype(np.uint16)
        elif max_val > 21 and max_val != UNKNOWN:
            class_space = "raw"
        elif max_val <= 3 or (max_val == UNKNOWN and np.all((arr <= 3) | (arr == UNKNOWN))):
            class_space = "superclass"
        else:
            class_space = "19class"

    movable = np.zeros(N, dtype=np.bool_)
    if class_space == "raw":
        for raw_id in _MOVABLE_RAW_IDS:
            movable |= (arr == raw_id)
    elif class_space == "19class":
        for c19 in _MOVABLE_19_IDS:
            movable |= (arr == c19)
    elif class_space == "superclass":
        # In super-class space, DYNAMIC is movable, and STATIC_OBSTACLE may contain vehicles
        movable = (arr == DYNAMIC) | (arr == STATIC_OBSTACLE)
    else:
        raise ValueError(f"Unknown class_space: {class_space}")

    return movable


def cluster_motion_candidates(
    points: NDArray[np.floating],
    motion_candidates: NDArray[np.bool_],
    labels: NDArray,
    eps: float = 0.7,
    min_samples: int = 10,
    class_space: str = "auto",
) -> list[ObjectBox]:
    """Filter motion candidates to movable classes and cluster into 3D bounding boxes.

    Parameters
    ----------
    points : NDArray[floating]
        LiDAR points, shape ``(N, 3)`` or ``(N, 4)`` in Velodyne frame.
    motion_candidates : NDArray[bool]
        Boolean mask indicating points exceeding the motion residual threshold.
    labels : NDArray
        Point-wise semantic labels.
    eps : float, default 0.7
        DBSCAN maximum distance between two samples to be considered neighbors.
    min_samples : int, default 10
        DBSCAN minimum number of points required to form a dense cluster.
    class_space : {"auto", "raw", "19class", "superclass"}, default "auto"
        Format of ``labels``.

    Returns
    -------
    boxes : list[ObjectBox]
        List of detected 3D bounding boxes for moving objects.
    """
    pts = np.asarray(points, dtype=np.float32)
    cands = np.asarray(motion_candidates, dtype=np.bool_)
    lbls = np.asarray(labels)

    if len(pts) == 0 or len(cands) == 0:
        return []

    # Step 1: Filter to movable classes
    movable = is_movable_class(lbls, class_space=class_space)
    active_mask = cands & movable

    n_active = int(np.sum(active_mask))
    if n_active < min_samples:
        return []

    # Step 2: Extract active XYZ and corresponding labels
    active_xyz = pts[active_mask, :3]
    active_lbls = lbls[active_mask]

    # Step 3: Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(active_xyz)
    cluster_labels = clustering.labels_

    # Step 4: Extract bounding box per valid cluster
    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)  # Remove noise

    boxes: list[ObjectBox] = []
    for cid in sorted(unique_clusters):
        c_mask = cluster_labels == cid
        c_pts = active_xyz[c_mask]
        c_lbls = active_lbls[c_mask]

        x_min, y_min, z_min = np.min(c_pts, axis=0)
        x_max, y_max, z_max = np.max(c_pts, axis=0)

        # Majority semantic class
        unique_l, counts = np.unique(c_lbls, return_counts=True)
        majority_label = unique_l[np.argmax(counts)]

        # Class confidence is the fraction of points agreeing with majority
        conf = float(np.max(counts) / len(c_lbls))

        box = ObjectBox(
            id=int(cid),
            cls=int(majority_label) if np.issubdtype(type(majority_label), np.integer) else majority_label,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            z_min=float(z_min),
            z_max=float(z_max),
            is_moving=True,
            n_points=len(c_pts),
            confidence=conf,
        )
        boxes.append(box)

    return boxes


def cluster_and_vote_motion(
    points: NDArray[np.floating],
    labels: NDArray,
    motion_candidates: NDArray[np.bool_],
    eps: float = 0.7,
    min_samples: int = 10,
    vote_frac: float = 0.3,
    class_space: str = "auto",
) -> tuple[list[ObjectBox], NDArray[np.bool_]]:
    """Cluster all movable points and confirm motion if candidate vote exceeds fraction.

    Per master plan §5.4: clustering all movable points and voting suppresses
    edge noise and segments the entire object rather than only the moving boundary.

    Parameters
    ----------
    points : NDArray[floating]
        LiDAR points, shape ``(N, 3)`` or ``(N, 4)``.
    labels : NDArray
        Point-wise semantic labels.
    motion_candidates : NDArray[bool]
        Boolean mask of residual-exceeding points.
    eps : float, default 0.7
        DBSCAN clustering radius.
    min_samples : int, default 10
        DBSCAN density threshold.
    vote_frac : float, default 0.3
        Fraction of candidate points within cluster required to declare the
        object moving.
    class_space : str, default "auto"
        Label format.

    Returns
    -------
    boxes : list[ObjectBox]
        List of moving object bounding boxes.
    moving_points_mask : NDArray[bool]
        Updated full-scan boolean mask where all points in confirmed moving
        clusters are marked True.
    """
    pts = np.asarray(points, dtype=np.float32)
    lbls = np.asarray(labels)
    cands = np.asarray(motion_candidates, dtype=np.bool_)
    N = len(pts)

    confirmed_moving_mask = np.zeros(N, dtype=np.bool_)

    if N == 0:
        return [], confirmed_moving_mask

    # Select all points of movable classes
    movable_mask = is_movable_class(lbls, class_space=class_space)
    n_movable = int(np.sum(movable_mask))
    if n_movable < min_samples:
        return [], confirmed_moving_mask

    movable_indices = np.flatnonzero(movable_mask)
    movable_xyz = pts[movable_mask, :3]
    movable_lbls = lbls[movable_mask]
    movable_cands = cands[movable_mask]

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(movable_xyz)
    cluster_labels = clustering.labels_

    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)

    boxes: list[ObjectBox] = []
    box_id = 0
    for cid in sorted(unique_clusters):
        c_mask = cluster_labels == cid
        c_cand_frac = float(np.mean(movable_cands[c_mask]))

        # Check motion vote
        if c_cand_frac >= vote_frac:
            # Mark all points in this cluster as moving
            orig_indices = movable_indices[c_mask]
            confirmed_moving_mask[orig_indices] = True

            c_pts = movable_xyz[c_mask]
            c_lbls = movable_lbls[c_mask]

            x_min, y_min, z_min = np.min(c_pts, axis=0)
            x_max, y_max, z_max = np.max(c_pts, axis=0)

            unique_l, counts = np.unique(c_lbls, return_counts=True)
            majority_label = unique_l[np.argmax(counts)]
            conf = float(np.max(counts) / len(c_lbls))

            boxes.append(
                ObjectBox(
                    id=box_id,
                    cls=int(majority_label) if np.issubdtype(type(majority_label), np.integer) else majority_label,
                    x_min=float(x_min),
                    x_max=float(x_max),
                    y_min=float(y_min),
                    y_max=float(y_max),
                    z_min=float(z_min),
                    z_max=float(z_max),
                    is_moving=True,
                    n_points=len(c_pts),
                    confidence=conf,
                )
            )
            box_id += 1

    return boxes, confirmed_moving_mask
