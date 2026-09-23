"""Calibration, poses and the exact frame transform of README 5.3 (task T3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIB_KEYS = ("P0", "P1", "P2", "P3", "Tr")


@dataclass(frozen=True)
class Calib:
    """Parsed ``calib.txt``: projection matrices P0..P3 (3x4) and ``Tr`` (4x4, Velodyne -> camera-0)."""

    P: tuple[np.ndarray, ...]
    Tr: np.ndarray


def _homogeneous(values: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :4] = values.reshape(3, 4)
    return T


def load_calib(path: str | Path) -> Calib:
    """Parse ``calib.txt`` (lines ``P0:``..``P3:`` and ``Tr:``, 12 floats each)."""
    path = Path(path)
    entries: dict[str, np.ndarray] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        key, _, rest = line.partition(":")
        values = np.array(rest.split(), dtype=np.float64)
        if values.size != 12:
            raise ValueError(f"{path}:{lineno}: {key.strip()} has {values.size} values, expected 12 values")
        entries[key.strip()] = values
    missing = [k for k in CALIB_KEYS if k not in entries]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    return Calib(P=tuple(entries[f"P{i}"].reshape(3, 4) for i in range(4)), Tr=_homogeneous(entries["Tr"]))


def load_poses(path: str | Path) -> np.ndarray:
    """Parse ``poses.txt`` into (T,4,4) float64 camera-0 poses (one 3x4 row-major matrix per line)."""
    path = Path(path)
    poses = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = np.array(line.split(), dtype=np.float64)
        if values.size != 12:
            raise ValueError(f"{path}: line {lineno} has {values.size} values, expected 12")
        poses.append(_homogeneous(values))
    return np.stack(poses) if poses else np.zeros((0, 4, 4))


def relative_transform(calib: Calib, poses: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return ``T_vel_i_from_vel_j = inv(Tr) @ inv(P_i) @ P_j @ Tr`` (4x4).

    Maps homogeneous points from the Velodyne frame of scan ``j`` into the Velodyne frame of scan ``i``.
    """
    inv_tr = np.linalg.inv(calib.Tr)
    return inv_tr @ np.linalg.inv(poses[i]) @ poses[j] @ calib.Tr


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N,3) points; returns float64 (N,3)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(T: np.ndarray) -> float:
    """Rotation angle of the 3x3 block of ``T`` in degrees (0 for identity)."""
    cos = (np.trace(T[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
