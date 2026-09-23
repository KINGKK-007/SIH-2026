"""Grid backends: NumPy reference (always) and optional C++ (README 6.5.7)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from foveamap.grid.accumulators import GridAccumulators
from foveamap.grid.presets import GridSpec


class GridBackend(Protocol):
    def rasterize(
        self,
        spec: GridSpec,
        xyz_mm: np.ndarray,
        super_cls: np.ndarray,
        moving: np.ndarray,
        conf: np.ndarray,
    ) -> GridAccumulators: ...
