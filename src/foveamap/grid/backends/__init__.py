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


def get_backend(name: str = "auto") -> GridBackend:
    """``auto`` uses the C++ backend if importable, else NumPy (L8, README 6.5.7)."""
    from foveamap.grid.backends import cpp_backend
    from foveamap.grid.backends.numpy_backend import NumpyBackend

    if name == "numpy" or (name == "auto" and not cpp_backend.is_available()):
        return NumpyBackend()
    if name in ("cpp", "auto"):
        if not cpp_backend.is_available():
            raise RuntimeError("backend 'cpp' requested but foveamap._fovea_cpp is not built")
        return cpp_backend.CppBackend()
    raise ValueError(f"unknown grid backend {name!r}")
