"""Reference NumPy rasteriser, the correctness oracle (README 6.5.4, task T5.3)."""

from __future__ import annotations

import numpy as np

from foveamap.grid.accumulators import GridAccumulators
from foveamap.grid.presets import GridSpec


class NumpyBackend:
    name = "numpy"

    def rasterize(
        self,
        spec: GridSpec,
        xyz_mm: np.ndarray,
        super_cls: np.ndarray,
        moving: np.ndarray,
        conf: np.ndarray,
    ) -> GridAccumulators:
        raise NotImplementedError("Implemented in Phase 5, T5.3 (docs/PHASES.md).")
