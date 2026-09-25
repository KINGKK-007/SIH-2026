"""Range-scaled Euclidean clustering of movable points (README 6.4 step 4, task T10.2)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def cluster_points(xyz: np.ndarray, cfg: Any) -> np.ndarray:
    """Return a cluster id per point (-1 = noise).

    Uses range-scaled Euclidean clustering:
    eps(r) = eps0_m + eps1_rel * r
    Connected components with size >= min_points are assigned ids >= 0.

    Args:
        xyz: (N, 3) point coordinates in metres.
        cfg: MotionClusterConfig or object with eps0_m, eps1_rel, min_points.

    Returns:
        cluster_ids: (N,) int32 array where values >= 0 are cluster IDs, -1 is noise.
    """
    n_pts = len(xyz)
    if n_pts == 0:
        return np.zeros(0, dtype=np.int32)

    min_points = int(getattr(cfg, "min_points", 5))
    if n_pts < min_points:
        return np.full(n_pts, -1, dtype=np.int32)

    eps0 = float(getattr(cfg, "eps0_m", 0.5))
    eps1 = float(getattr(cfg, "eps1_rel", 0.01))

    # Range per point
    r = np.linalg.norm(xyz, axis=1)
    eps = eps0 + eps1 * r

    # Build k-d tree and query neighbours within point-specific radius
    tree = cKDTree(xyz)
    neighbour_lists = tree.query_ball_point(xyz, r=eps)

    # Disjoint connected components via scipy.sparse
    lens = [len(lst) for lst in neighbour_lists]
    total_edges = sum(lens)
    if total_edges == 0:
        return np.full(n_pts, -1, dtype=np.int32)

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    row = np.repeat(np.arange(n_pts, dtype=np.int32), lens)
    col = np.concatenate(neighbour_lists).astype(np.int32)
    adj = coo_matrix((np.ones(len(col), dtype=bool), (row, col)), shape=(n_pts, n_pts))

    _, labels = connected_components(adj, directed=False)

    counts = np.bincount(labels)
    valid_mask = counts >= min_points

    # Map valid labels to sequential IDs sorted by cluster size (descending)
    valid_components = np.flatnonzero(valid_mask)
    sorted_components = valid_components[np.argsort(-counts[valid_components])]

    label_map = np.full(len(counts), -1, dtype=np.int32)
    for next_id, comp_id in enumerate(sorted_components):
        label_map[comp_id] = next_id

    return label_map[labels]

