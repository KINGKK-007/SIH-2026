"""Point back-projection accuracy per bucket (README 12.2, 12.4)."""

from __future__ import annotations

import numpy as np  # noqa: F401


def point_backprojection_accuracy(frame_result: object, scan: object, super_gt: np.ndarray) -> dict:
    raise NotImplementedError("Implemented in Phase 7, T7.3 (docs/PHASES.md).")
