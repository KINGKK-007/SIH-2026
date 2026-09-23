"""Integer-only cell addressing (README 6.5.3, task T5.2)."""

from __future__ import annotations

import numpy as np

from foveamap.grid.presets import GridSpec


def quantize_mm(xyz: np.ndarray) -> np.ndarray:
    """``np.rint(xyz * 1000).astype(np.int32)`` (L6)."""
    raise NotImplementedError("Implemented in Phase 5, T5.2 (docs/PHASES.md).")


def world_to_cell(spec: GridSpec, x_mm: int, y_mm: int) -> tuple[int, int, int] | None:
    """Return ``(ring, ix, iy)`` or ``None`` if the point is out of grid."""
    raise NotImplementedError("Implemented in Phase 5, T5.2 (docs/PHASES.md).")


def cell_to_corner_mm(spec: GridSpec, ring: int, ix: int, iy: int) -> tuple[int, int]:
    raise NotImplementedError("Implemented in Phase 5, T5.2 (docs/PHASES.md).")


def cell_to_center_mm(spec: GridSpec, ring: int, ix: int, iy: int) -> tuple[int, int]:
    raise NotImplementedError("Implemented in Phase 5, T5.2 (docs/PHASES.md).")
