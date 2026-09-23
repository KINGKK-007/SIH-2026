"""Ego-compensated range-image residual and per-point votes (README 6.4 steps 1-3, task T10.1)."""

from __future__ import annotations

import numpy as np


def range_residual_votes(cur_xyz: np.ndarray, prev_xyz_in_cur: np.ndarray, cfg: object) -> np.ndarray:
    """Per-point vote: 1 moving, 0 static-consistent, -1 unobserved."""
    raise NotImplementedError("Implemented in Phase 10, T10.1 (docs/PHASES.md).")
