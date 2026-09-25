"""GPU device utilities — thin wrapper so the rest of the codebase can write ``xp.array(...)``
and remain NumPy-compatible when CuPy is absent (README 3.1 "never guess values from checkpoints"
applies equally to the processing device: we must degrade gracefully).

Usage pattern::

    from foveamap.gpu_utils import get_array_module, to_numpy

    xp = get_array_module(device)   # cp or np
    arr_gpu = xp.asarray(arr_cpu)
    result_cpu = to_numpy(result_gpu)
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "is_cupy_available",
    "get_array_module",
    "to_numpy",
    "to_device",
    "device_to_backend_name",
]

def _ensure_cuda_dlls_on_windows() -> None:
    """Windows-only: add PyTorch's bundled CUDA DLLs to the DLL search path.

    On Windows, CuPy's ``cuda-pathfinder`` locates runtime libraries (curand, cublas, …) by
    scanning ``PATH`` and known NVIDIA Toolkit directories.  When the full CUDA Toolkit is not
    installed — but PyTorch is — the required DLLs already exist inside ``torch/lib``.  We add
    that directory to both ``os.environ["PATH"]`` and (Python ≥ 3.8) ``os.add_dll_directory``
    so the loader finds them automatically, with no separate Toolkit download required.
    """
    if sys.platform != "win32":
        return
    try:
        import torch  # already installed in uav_env

        torch_lib = Path(torch.__file__).parent / "lib"
        if not torch_lib.is_dir():
            return
        lib_str = str(torch_lib)
        # Make ctypes / os.CDLL find DLLs in this directory (Python 3.8+)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib_str)
        # Also add to PATH so cuda-pathfinder's own filesystem scan picks it up
        path_env = os.environ.get("PATH", "")
        if lib_str not in path_env:
            os.environ["PATH"] = lib_str + os.pathsep + path_env
    except Exception:
        pass  # non-fatal — CuPy will emit its own error if it still can't find the DLLs


# Run immediately at module load so CUDA DLLs are on the search path before
# any code does ``import cupy`` — even code that doesn't go through our helpers.
_ensure_cuda_dlls_on_windows()

# Cached probe result so we only import / call CUDA once per process.
_CUPY_AVAILABLE: bool | None = None


def is_cupy_available() -> bool:
    """Return True if CuPy is installed **and** at least one CUDA device is visible."""
    global _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy as cp  # noqa: F401

            n = cp.cuda.runtime.getDeviceCount()
            _CUPY_AVAILABLE = n > 0
        except Exception:
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


def get_array_module(device: str = "auto") -> Any:
    """Return ``cupy`` for ``'cuda'`` / ``'cupy'`` / ``'auto'`` when CuPy+GPU is available, else ``numpy``.

    Args:
        device: ``'auto'`` (default) picks GPU when available; ``'cuda'`` / ``'cupy'`` forces GPU
            and emits a warning on fallback; ``'cpu'`` always returns ``numpy``.

    Returns:
        The ``cupy`` or ``numpy`` module.
    """
    if device == "cpu":
        return np
    if device in ("cuda", "cupy", "auto"):
        if is_cupy_available():
            import cupy as cp

            return cp
        if device != "auto":
            warnings.warn(
                f"CuPy / CUDA not available (device={device!r}); falling back to NumPy. "
                "Install CuPy matching your CUDA toolkit: pip install cupy-cuda12x",
                stacklevel=2,
            )
    return np


def to_numpy(arr: Any) -> np.ndarray:
    """Copy a GPU array to CPU NumPy if necessary; return a NumPy array in all cases."""
    if hasattr(arr, "get"):  # CuPy array
        return arr.get()
    return np.asarray(arr)


def to_device(arr: np.ndarray, xp: Any) -> Any:
    """Upload ``arr`` to ``xp``'s device (no-op when ``xp is numpy``)."""
    return xp.asarray(arr)


def device_to_backend_name(device: str, current_backend: str = "auto") -> str:
    """Map a ``--device`` CLI flag to the grid backend name.

    ``'auto'`` and ``'cuda'`` resolve to ``'cupy'`` when CuPy is importable, else they leave the
    current backend unchanged so ``get_backend()`` can fall through to NumPy / C++.
    ``'cpu'`` forces ``'numpy'``.
    """
    if device == "cpu":
        return "numpy"
    if device in ("cuda", "cupy"):
        return "cupy"
    # "auto" — honour whatever was already in the config
    return current_backend
