"""Minimum-area oriented boxes by angle sweep (README 6.4 step 7, task T10.2)."""

from __future__ import annotations

import numpy as np


def oriented_box(
    xyz: np.ndarray, angle_step_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return ``(center, size (l, w, h), yaw)`` of the minimum-area rectangle in the ground plane."""
    raise NotImplementedError("Implemented in Phase 10, T10.2 (docs/PHASES.md).")
