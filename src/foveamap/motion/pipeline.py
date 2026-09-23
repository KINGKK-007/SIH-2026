"""``estimate_motion``: README 6.4 steps 1-5, 7-8 (task T10.3)."""

from __future__ import annotations

import numpy as np

from foveamap.pipeline.records import ClassifiedScan, Prediction, Scan


def estimate_motion(
    cur: ClassifiedScan,
    prev_scans: list[tuple[Scan, Prediction]],
    transforms: list[np.ndarray],
    cfg: object,
) -> ClassifiedScan:
    """Return ``cur`` with ``moving``/``super_cls`` promoted and ``objects`` filled."""
    raise NotImplementedError("Implemented in Phase 10, T10.3 (docs/PHASES.md).")
