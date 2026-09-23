"""Point density vs range (README 12.7). First version (T3.5): points per range bin.

Nearest-neighbour spacing and cell occupancy vs range are added in Phase 12 (T12.6).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from foveamap.eval.buckets import horizontal_range


def points_per_range_bin(
    xyz_scans: Iterable[np.ndarray], bin_m: float = 5.0, max_m: float = 100.0
) -> dict[str, list[float]]:
    """Pool scans and count points per ``bin_m`` horizontal-range bin up to ``max_m``.

    Returns bin edges, total counts, mean points per scan, and mean points per scan per square metre
    of the bin's annulus (the density that falls off roughly as 1/r^2 for a spinning LiDAR).
    """
    if bin_m <= 0 or max_m <= 0:
        raise ValueError("bin_m and max_m must be positive")
    edges = np.arange(0.0, max_m + bin_m / 2, bin_m)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    n_scans = 0
    for xyz in xyz_scans:
        hist, _ = np.histogram(horizontal_range(xyz), bins=edges)
        counts += hist
        n_scans += 1
    if n_scans == 0:
        raise ValueError("no scans given")
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    per_scan = counts / n_scans
    return {
        "edges_m": edges.tolist(),
        "counts": counts.tolist(),
        "points_per_scan": per_scan.tolist(),
        "points_per_scan_per_m2": (per_scan / area).tolist(),
        "n_scans": [float(n_scans)],
    }
