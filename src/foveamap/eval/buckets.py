"""Distance buckets (L16): horizontal Euclidean range, 0-10/10-30/30-60/60-100 m (task T3.5)."""

from __future__ import annotations

import numpy as np


def bucket_of_range(r: np.ndarray) -> np.ndarray:
    """Bucket index 0..3 per range value (L16)."""
    raise NotImplementedError("Implemented in Phase 3, T3.5 (docs/PHASES.md).")
