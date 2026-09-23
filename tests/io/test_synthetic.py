"""The synthetic KITTI-format generator used as a fixture (D-015)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from foveamap.io.kitti import load_label, load_scan_bin, split_label
from foveamap.io.poses import load_calib, load_poses, relative_transform
from foveamap.io.synthetic import write_kitti_sequence


def test_deterministic(tmp_path: Path) -> None:
    a = write_kitti_sequence(tmp_path / "a", n_frames=2, seed=3)
    b = write_kitti_sequence(tmp_path / "b", n_frames=2, seed=3)
    for sub in ("velodyne/000001.bin", "labels/000001.label", "poses.txt", "calib.txt"):
        assert (a.root / "sequences/08" / sub).read_bytes() == (b.root / "sequences/08" / sub).read_bytes()


def test_files_are_kitti_format(synthetic_root: Path) -> None:
    seq = synthetic_root / "sequences" / "08"
    xyz, rem = load_scan_bin(seq / "velodyne" / "000000.bin")
    raw = load_label(seq / "labels" / "000000.label")
    assert len(xyz) == len(rem) == len(raw) > 30_000
    sem, inst = split_label(raw)
    for needed in (40, 48, 50, 80, 10, 252, 254):  # road, sidewalk, building, pole, car, movers
        assert (sem == needed).any(), needed
    assert inst[sem == 10].max() > 0  # cars carry instance IDs
    assert np.abs(np.hypot(xyz[:, 0], xyz[:, 1])).max() <= 120.0
    assert len((seq / "times.txt").read_text().split()) == len(load_poses(seq / "poses.txt"))


def test_poses_decode_to_true_vehicle_motion(tmp_path: Path) -> None:
    info = write_kitti_sequence(tmp_path, n_frames=5, seed=11)
    seq = tmp_path / "sequences" / "08"
    calib, poses = load_calib(seq / "calib.txt"), load_poses(seq / "poses.txt")
    V = info.vehicle_poses
    for i, j in [(1, 0), (4, 2), (0, 3)]:
        truth = np.linalg.inv(V[i]) @ V[j]
        np.testing.assert_allclose(relative_transform(calib, poses, i, j), truth, atol=1e-9)


def test_poses_dir_variant(synthetic_root: Path) -> None:
    assert (synthetic_root / "poses" / "04.txt").is_file()
    assert not (synthetic_root / "sequences" / "04" / "poses.txt").exists()
