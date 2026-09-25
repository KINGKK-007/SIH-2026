"""Grid backends: NumPy reference (always), optional CuPy GPU, and optional C++ (README 6.5.7).

Auto-selection priority (highest first):

1. ``cupy``  — GPU via CuPy; requires ``cupy`` + a CUDA device.
2. ``cpp``   — compiled pybind11 extension ``foveamap._fovea_cpp`` (Phase 7 T7.5, P2).
3. ``numpy`` — pure-NumPy reference; always available; used as fallback.

Pass an explicit name to ``get_backend()`` to skip auto-selection.
"""

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
    """Return the requested grid backend, falling back to NumPy when unavailable.

    ``name`` is one of ``'auto'``, ``'cupy'``, ``'cpp'``, or ``'numpy'``:

    * ``'auto'``: try CuPy → C++ → NumPy in order.
    * ``'cupy'``: GPU via CuPy; raises if CuPy is absent.
    * ``'cpp'``: pybind11 extension; raises if not built.
    * ``'numpy'``: always available.
    """
    from foveamap.grid.backends import cpp_backend, cupy_backend
    from foveamap.grid.backends.numpy_backend import NumpyBackend

    if name == "numpy":
        return NumpyBackend()

    if name == "cupy":
        if not cupy_backend.is_available():
            raise RuntimeError(
                "backend 'cupy' requested but CuPy is not installed or no CUDA device found. "
                "Install CuPy: pip install cupy-cuda12x  (match your CUDA version)"
            )
        return cupy_backend.CupyBackend()

    if name == "cpp":
        if not cpp_backend.is_available():
            raise RuntimeError("backend 'cpp' requested but foveamap._fovea_cpp is not built")
        return cpp_backend.CppBackend()

    if name == "auto":
        # Priority: CuPy (GPU) > C++ > NumPy
        if cupy_backend.is_available():
            return cupy_backend.CupyBackend()
        if cpp_backend.is_available():
            return cpp_backend.CppBackend()
        return NumpyBackend()

    raise ValueError(f"unknown grid backend {name!r}; choose from 'auto', 'cupy', 'cpp', 'numpy'")
