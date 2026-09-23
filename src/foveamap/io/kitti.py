"""Binary scan and label loading (README 5.2, task T3.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCAN_RECORD_BYTES = 16  # float32 x, y, z, remission
LABEL_RECORD_BYTES = 4  # uint32 per point


def load_scan_bin(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Velodyne ``.bin`` scan.

    Returns ``xyz`` (N,3) float32 in metres (Velodyne frame) and ``remission`` (N,) float32,
    both C-contiguous.
    """
    path = Path(path)
    size = path.stat().st_size
    if size % SCAN_RECORD_BYTES:
        raise ValueError(f"{path}: size {size} is not a multiple of 16 bytes (N x 4 float32)")
    pts = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    return np.ascontiguousarray(pts[:, :3]), np.ascontiguousarray(pts[:, 3])


def load_label(path: str | Path) -> np.ndarray:
    """Load a SemanticKITTI ``.label`` file as (N,) uint32 (semantic = raw & 0xFFFF, instance = raw >> 16)."""
    path = Path(path)
    size = path.stat().st_size
    if size % LABEL_RECORD_BYTES:
        raise ValueError(f"{path}: size {size} is not a multiple of 4 bytes (N x uint32)")
    return np.fromfile(path, dtype="<u4")


def split_label(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split raw uint32 labels into ``(semantic, instance)``, both uint16."""
    raw = np.asarray(raw, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint16), (raw >> 16).astype(np.uint16)
