"""Distance buckets (L16): horizontal Euclidean range, 0-10/10-30/30-60/60-100 m (task T3.5).

Point-level metrics bucket by ``sqrt(x^2 + y^2)`` of the *un-quantised* coordinates (README 6.2).
Buckets are half-open ``[a, b)`` except the last, which is closed so a point at exactly 100 m counts;
points beyond the last edge get bucket ``-1`` and are excluded from per-bucket metrics.
Cell-level metrics bucket by ring id instead (see :func:`bucket_labels` for the shared names).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_EDGES_M: tuple[float, ...] = (0.0, 10.0, 30.0, 60.0, 100.0)
OUT_OF_RANGE = -1


def horizontal_range(xyz: np.ndarray) -> np.ndarray:
    """``sqrt(x^2 + y^2)`` in metres, float64."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return np.hypot(xyz[:, 0], xyz[:, 1])


def bucket_of_range(r: np.ndarray, edges_m: Sequence[float] = DEFAULT_EDGES_M) -> np.ndarray:
    """Bucket index per range value (int8): 0..len(edges)-2, or -1 outside ``[edges[0], edges[-1]]``."""
    edges = np.asarray(edges_m, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"edges must be strictly increasing with >= 2 entries, got {list(edges_m)}")
    r = np.asarray(r, dtype=np.float64)
    idx = np.searchsorted(edges, r, side="right") - 1
    idx = np.where(r == edges[-1], len(edges) - 2, idx)  # close the last bucket
    out = (r < edges[0]) | (r > edges[-1]) | ~np.isfinite(r)
    return np.where(out, OUT_OF_RANGE, idx).astype(np.int8)


def bucket_labels(edges_m: Sequence[float] = DEFAULT_EDGES_M) -> list[str]:
    """Human-readable bucket names, e.g. ``"0-10 m"``."""
    return [f"{a:g}-{b:g} m" for a, b in zip(edges_m[:-1], edges_m[1:], strict=True)]


def bucket_counts(r: np.ndarray, edges_m: Sequence[float] = DEFAULT_EDGES_M) -> np.ndarray:
    """Number of values per bucket (out-of-range values are not counted)."""
    b = bucket_of_range(r, edges_m)
    return np.bincount(b[b >= 0].astype(np.intp), minlength=len(edges_m) - 1)
