"""Optional pybind11 backend ``foveamap._fovea_cpp`` (README 6.5.7, task T7.5, tier P2)."""

from __future__ import annotations

import numpy as np

from foveamap.grid.accumulators import GridAccumulators
from foveamap.grid.presets import GridSpec


def is_available() -> bool:
    """True if the compiled ``foveamap._fovea_cpp`` module imports."""
    try:
        import foveamap._fovea_cpp  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


class CppBackend:
    name = "cpp"

    def rasterize(
        self,
        spec: GridSpec,
        xyz_mm: np.ndarray,
        super_cls: np.ndarray,
        moving: np.ndarray,
        conf: np.ndarray,
    ) -> GridAccumulators:
        raise NotImplementedError("Implemented in Phase 7, T7.5 (docs/PHASES.md).")
