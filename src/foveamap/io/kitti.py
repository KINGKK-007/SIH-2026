"""Binary scan and label loading (README 5.2, task T3.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_scan_bin(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Velodyne ``.bin`` scan. Returns ``xyz`` (N,3) float32 and ``remission`` (N,) float32."""
    raise NotImplementedError("Implemented in Phase 3, T3.1 (docs/PHASES.md).")


def load_label(path: str | Path) -> np.ndarray:
    """Load a SemanticKITTI ``.label`` file as (N,) uint32 (semantic = raw & 0xFFFF, instance = raw >> 16)."""
    raise NotImplementedError("Implemented in Phase 3, T3.1 (docs/PHASES.md).")
