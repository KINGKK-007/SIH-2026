"""Unit tests for range-scaled Euclidean clustering (T10.2)."""

from types import SimpleNamespace

import numpy as np

from foveamap.motion.cluster import cluster_points


def test_cluster_empty_and_noise() -> None:
    cfg = SimpleNamespace(eps0_m=0.5, eps1_rel=0.01, min_points=5)

    assert len(cluster_points(np.zeros((0, 3)), cfg)) == 0

    # 4 points when min_points is 5 -> all noise (-1)
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]])
    c_ids = cluster_points(pts, cfg)
    assert np.all(c_ids == -1)


def test_cluster_separation_and_grouping() -> None:
    """Two well-separated point clusters should be assigned distinct IDs >= 0."""
    cfg = SimpleNamespace(eps0_m=0.5, eps1_rel=0.01, min_points=5)

    rng = np.random.default_rng(42)
    # Cluster 1 around (10, 0, 0)
    c1 = rng.uniform(-0.2, 0.2, size=(20, 3)) + np.array([10.0, 0.0, 0.0])
    # Cluster 2 around (10, 15, 0) -> separated by 15 m >> eps(18m) ~ 0.68m
    c2 = rng.uniform(-0.2, 0.2, size=(25, 3)) + np.array([10.0, 15.0, 0.0])
    # Isolated noise point at (0, 10, 0)
    noise = np.array([[0.0, 10.0, 0.0]])

    all_pts = np.vstack([c1, c2, noise])
    c_ids = cluster_points(all_pts, cfg)

    # Noise point is -1
    assert c_ids[-1] == -1

    # Cluster 1 points all have the same ID >= 0
    id1 = c_ids[0]
    assert id1 >= 0
    assert np.all(c_ids[:20] == id1)

    # Cluster 2 points all have the same ID >= 0
    id2 = c_ids[20]
    assert id2 >= 0
    assert np.all(c_ids[20:45] == id2)

    # They must be distinct clusters
    assert id1 != id2
