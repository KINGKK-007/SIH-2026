"""foveamap.io.poses — Calibration and pose utilities.

Coordinate frame conventions
------------------------------
SemanticKITTI / KITTI Odometry uses **two** frames that must not be confused:

Camera-0 frame
    ``poses.txt`` lists ``T_cam(t)`` — the 3×4 rigid-body pose of the Camera-0
    at time *t* in a global world frame, expressed in the Camera-0 coordinate
    system (x right, y down, z forward).

Velodyne frame
    Scans (``.bin`` files) are natively in the Velodyne frame
    (x forward, y left, z up).  The sensor sits ≈ 1.73 m above the road.

Conversion
    ``calib.txt`` contains ``Tr`` — the 4×4 rigid-body transform that maps
    **Velodyne → Camera-0** (i.e. ``p_cam = Tr @ p_velo``).

    To get the pose of the Velodyne sensor in the world frame at time *t*::

        T_velo(t) = Tr⁻¹ · pose(t) · Tr

    where ``pose(t)`` is the 4×4 homogeneous version of the Camera-0 pose row.
    Skipping this step silently produces wrong ego-motion residuals (G6 in the
    master plan risk register).

References
----------
- Master plan §5.2
- https://github.com/PRBonn/semantic-kitti-api
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_calib(calib_path: str | Path) -> NDArray[np.float64]:
    """Parse a KITTI ``calib.txt`` and return the Velodyne→Camera-0 transform.

    Parameters
    ----------
    calib_path:
        Path to ``calib.txt`` (e.g.
        ``data/semkitti/sequences/08/calib.txt``).

    Returns
    -------
    Tr : NDArray[float64]
        Shape ``(4, 4)``.  Homogeneous rigid-body transform mapping a point
        from the Velodyne frame to the Camera-0 frame:
        ``p_cam = Tr @ [x, y, z, 1]ᵀ``.

    Raises
    ------
    FileNotFoundError
        If ``calib_path`` does not exist.
    KeyError
        If the ``Tr:`` row is missing from the file.
    ValueError
        If the parsed row does not contain exactly 12 float values.

    Notes
    -----
    ``calib.txt`` contains rows like::

        P0: f00 f01 ... f11
        P1: ...
        Tr: f00 f01 ... f11

    The ``Tr`` row holds the upper 3×4 of the transform.  We append a
    ``[0, 0, 0, 1]`` row to form a full 4×4 matrix.
    """
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"calib.txt not found: {calib_path}")

    data: dict[str, NDArray[np.float64]] = {}
    with calib_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, values = line.split(":", 1)
            data[key.strip()] = np.fromstring(values, sep=" ", dtype=np.float64)

    if "Tr" not in data:
        raise KeyError(f"'Tr' key not found in calib file: {calib_path}")

    tr_flat = data["Tr"]
    if tr_flat.size != 12:
        raise ValueError(
            f"Expected 12 values for Tr, got {tr_flat.size} in {calib_path}"
        )

    Tr = np.eye(4, dtype=np.float64)
    Tr[:3, :4] = tr_flat.reshape(3, 4)
    return Tr


def load_poses(
    poses_path: str | Path,
    Tr: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Parse ``poses.txt`` and convert Camera-0 poses to the Velodyne frame.

    Parameters
    ----------
    poses_path:
        Path to ``poses.txt`` (e.g.
        ``data/semkitti/sequences/08/poses.txt``).
        Each line holds 12 floats — the upper 3×4 of a Camera-0 pose matrix
        (row-major).
    Tr:
        The 4×4 Velodyne→Camera-0 calibration matrix from
        :func:`load_calib`.

    Returns
    -------
    poses_velo : list[NDArray[float64]]
        One 4×4 matrix per frame.  Each matrix is the **Velodyne-frame** pose::

            T_velo(t) = Tr⁻¹ · pose_cam(t) · Tr

        such that transforming a point from the Velodyne frame of scan *t* into
        the Velodyne frame of scan *s* is::

            p_s = T_velo(s)⁻¹ · T_velo(t) · p_t

    Raises
    ------
    FileNotFoundError
        If ``poses_path`` does not exist.
    ValueError
        If any line does not contain exactly 12 values.

    Notes
    -----
    The inverse of ``Tr`` is computed once with :func:`numpy.linalg.inv` and
    reused for all frames.  For a rigid-body matrix ``Tr = [R | t]`` the
    analytic inverse is cheaper, but ``np.linalg.inv`` is numerically safe
    for well-conditioned calibration matrices.
    """
    poses_path = Path(poses_path)
    if not poses_path.exists():
        raise FileNotFoundError(f"poses.txt not found: {poses_path}")

    Tr_inv = np.linalg.inv(Tr)

    poses_velo: list[NDArray[np.float64]] = []
    with poses_path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            vals = np.fromstring(line, sep=" ", dtype=np.float64)
            if vals.size != 12:
                raise ValueError(
                    f"Expected 12 values per pose row, got {vals.size} "
                    f"at line {lineno} of {poses_path}"
                )
            T_cam = np.eye(4, dtype=np.float64)
            T_cam[:3, :4] = vals.reshape(3, 4)

            # Convert Camera-0 pose → Velodyne frame
            T_velo = Tr_inv @ T_cam @ Tr
            poses_velo.append(T_velo)

    return poses_velo
