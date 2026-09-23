"""Calibration, poses and the exact frame transform of README 5.3 (task T3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calib:
    """Parsed ``calib.txt``: projection matrices P0..P3 (3x4) and ``Tr`` (4x4, Velodyne -> camera-0)."""

    P: tuple[np.ndarray, ...]
    Tr: np.ndarray


def load_calib(path: str | Path) -> Calib:
    """Parse ``calib.txt`` (lines ``P0:``..``P3:`` and ``Tr:``)."""
    raise NotImplementedError("Implemented in Phase 3, T3.2 (docs/PHASES.md).")


def load_poses(path: str | Path) -> np.ndarray:
    """Parse ``poses.txt`` into (T,4,4) camera-0 poses."""
    raise NotImplementedError("Implemented in Phase 3, T3.2 (docs/PHASES.md).")


def relative_transform(calib: Calib, poses: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return ``T_vel_i_from_vel_j = inv(Tr) @ inv(P_i) @ P_j @ Tr`` (4x4)."""
    raise NotImplementedError("Implemented in Phase 3, T3.2 (docs/PHASES.md).")
