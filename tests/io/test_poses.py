"""T3.2: calib/poses parsing and relative_transform (README 5.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from foveamap.io.poses import (
    Calib,
    load_calib,
    load_poses,
    relative_transform,
    rotation_angle_deg,
)


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _rigid(rng: np.random.Generator, t_scale: float = 5.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rot(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
    T[:3, 3] = rng.normal(scale=t_scale, size=3)
    return T


def _kitti_like_tr() -> np.ndarray:
    """Velodyne (x fwd, y left, z up) -> camera (x right, y down, z fwd), slightly rotated and offset."""
    T = np.eye(4)
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    T[:3, :3] = _rot(np.array([0.3, 1.0, -0.2]), 0.02) @ base
    T[:3, 3] = [-0.004, -0.076, -0.272]
    return T


def _write_calib(path: Path, Tr: np.ndarray) -> None:
    lines = [f"P{i}: " + " ".join(f"{v:.12e}" for v in np.eye(3, 4).ravel()) for i in range(4)]
    lines.append("Tr: " + " ".join(f"{v:.12e}" for v in Tr[:3].ravel()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_poses(path: Path, poses: np.ndarray) -> None:
    rows = [" ".join(f"{v:.12e}" for v in P[:3].ravel()) for P in poses]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_calib(tmp_path: Path) -> None:
    Tr = _kitti_like_tr()
    _write_calib(tmp_path / "calib.txt", Tr)
    calib = load_calib(tmp_path / "calib.txt")
    assert len(calib.P) == 4 and all(p.shape == (3, 4) for p in calib.P)
    assert calib.Tr.shape == (4, 4)
    np.testing.assert_allclose(calib.Tr, Tr, atol=1e-11)
    np.testing.assert_array_equal(calib.Tr[3], [0, 0, 0, 1])


def test_load_calib_missing_tr(tmp_path: Path) -> None:
    (tmp_path / "calib.txt").write_text("P0: " + " ".join(["0"] * 12) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Tr"):
        load_calib(tmp_path / "calib.txt")


def test_load_calib_wrong_count(tmp_path: Path) -> None:
    (tmp_path / "calib.txt").write_text("Tr: 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 12 values"):
        load_calib(tmp_path / "calib.txt")


def test_load_poses(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    poses = np.stack([_rigid(rng) for _ in range(5)])
    _write_poses(tmp_path / "poses.txt", poses)
    loaded = load_poses(tmp_path / "poses.txt")
    assert loaded.shape == (5, 4, 4) and loaded.dtype == np.float64
    np.testing.assert_allclose(loaded, poses, atol=1e-11)


def test_load_poses_bad_line(tmp_path: Path) -> None:
    (tmp_path / "poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_poses(tmp_path / "poses.txt")


def _setup(seed: int, n: int = 6) -> tuple[Calib, np.ndarray]:
    rng = np.random.default_rng(seed)
    calib = Calib(P=tuple(np.eye(3, 4) for _ in range(4)), Tr=_kitti_like_tr())
    return calib, np.stack([_rigid(rng) for _ in range(n)])


def test_identity_for_same_scan() -> None:
    calib, poses = _setup(2)
    for i in range(len(poses)):
        np.testing.assert_allclose(relative_transform(calib, poses, i, i), np.eye(4), atol=1e-9)


def test_inverse_pair() -> None:
    calib, poses = _setup(3)
    T_ij = relative_transform(calib, poses, 1, 4)
    T_ji = relative_transform(calib, poses, 4, 1)
    np.testing.assert_allclose(T_ij @ T_ji, np.eye(4), atol=1e-9)


def test_result_is_rigid() -> None:
    calib, poses = _setup(4)
    T = relative_transform(calib, poses, 0, 5)
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_array_equal(T[3], [0, 0, 0, 1])


@settings(max_examples=100, derandomize=True)
@given(st.integers(0, 2**31 - 1))
def test_maps_points_between_velodyne_frames(seed: int) -> None:
    """A world point seen from scan j lands where scan i sees it (README 5.3 semantics)."""
    calib, poses = _setup(seed, n=2)
    rng = np.random.default_rng(seed + 1)
    world_cam = np.vstack([rng.normal(scale=20, size=(3, 50)), np.ones((1, 50))])  # camera-0 world frame
    inv_tr = np.linalg.inv(calib.Tr)
    seen_by_j = inv_tr @ np.linalg.inv(poses[1]) @ world_cam  # Velodyne frame of scan j
    seen_by_i = inv_tr @ np.linalg.inv(poses[0]) @ world_cam  # Velodyne frame of scan i
    np.testing.assert_allclose(relative_transform(calib, poses, 0, 1) @ seen_by_j, seen_by_i, atol=1e-8)


def test_wrong_formula_would_fail() -> None:
    """Guard: omitting the Tr conjugation gives a different transform when Tr is not identity."""
    calib, poses = _setup(5, n=2)
    naive = np.linalg.inv(poses[0]) @ poses[1]
    assert not np.allclose(relative_transform(calib, poses, 0, 1), naive, atol=1e-3)


def test_rotation_angle() -> None:
    T = np.eye(4)
    T[:3, :3] = _rot(np.array([0.0, 0.0, 1.0]), np.deg2rad(7.5))
    assert rotation_angle_deg(T) == pytest.approx(7.5, abs=1e-9)
    assert rotation_angle_deg(np.eye(4)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.data
def test_real_consecutive_scans_plausible(real_data_root: Path, real_sequences: list[str]) -> None:
    """Consecutive scans (0.1 s apart): translation < 3 m and rotation < 5 degrees (README 5.3)."""
    for seq in real_sequences:
        seq_dir = real_data_root / "sequences" / seq
        calib = load_calib(seq_dir / "calib.txt")
        pose_path = seq_dir / "poses.txt"
        if not pose_path.is_file():
            pose_path = real_data_root / "poses" / f"{seq}.txt"
        poses = load_poses(pose_path)
        for i in range(1, len(poses)):
            T = relative_transform(calib, poses, i, i - 1)
            assert np.linalg.norm(T[:3, 3]) < 3.0, (seq, i)
            assert rotation_angle_deg(T) < 5.0, (seq, i)
