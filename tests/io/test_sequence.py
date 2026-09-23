"""T3.4: the lazy, strided Sequence reader."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from foveamap.io.sequence import Sequence
from foveamap.pipeline.records import Scan


def test_basic_access(synthetic_root: Path) -> None:
    seq = Sequence(synthetic_root, "08")
    assert len(seq) == seq.n_frames_total == 12
    scan = seq[3]
    assert isinstance(scan, Scan)
    assert scan.seq == "08" and scan.idx == 3
    assert scan.xyz.dtype == np.float32 and scan.xyz.shape[1] == 3
    assert scan.raw_labels is not None and len(scan.raw_labels) == len(scan.xyz)
    assert scan.pose.shape == (4, 4)
    assert scan.timestamp == pytest.approx(0.3)


def test_stride_keeps_original_indices(synthetic_root: Path) -> None:
    seq = Sequence(synthetic_root, "08", frame_stride=5)
    assert len(seq) == 3  # frames 0, 5, 10
    assert [s.idx for s in seq] == [0, 5, 10]
    assert seq[-1].idx == 10
    with pytest.raises(IndexError):
        seq[3]


def test_without_labels(synthetic_root: Path) -> None:
    assert Sequence(synthetic_root, "07", with_labels=False)[0].raw_labels is None


def test_poses_found_in_poses_dir(synthetic_root: Path) -> None:
    seq = Sequence(synthetic_root, "04")
    assert len(seq.poses) == len(seq)


def test_relative_transform_matches_io(synthetic_root: Path) -> None:
    seq = Sequence(synthetic_root, "08")
    np.testing.assert_allclose(seq.relative_transform(2, 2), np.eye(4), atol=1e-9)
    assert np.linalg.norm(seq.relative_transform(1, 0)[:3, 3]) == pytest.approx(0.8, abs=0.01)  # 8 m/s


def test_missing_label_detected(tmp_path: Path, synthetic_root: Path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(synthetic_root / "sequences" / "07", root / "sequences" / "07")
    (root / "sequences" / "07" / "labels" / "000004.label").unlink()
    with pytest.raises(FileNotFoundError, match="label"):
        Sequence(root, "07")
    assert len(Sequence(root, "07", with_labels=False)) == 12


def test_missing_times_falls_back_with_warning(tmp_path: Path, synthetic_root: Path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(synthetic_root / "sequences" / "07", root / "sequences" / "07")
    (root / "sequences" / "07" / "times.txt").unlink()
    with pytest.warns(UserWarning, match="times.txt"):
        seq = Sequence(root, "07")
    assert seq[2].timestamp == pytest.approx(0.2)


def test_pose_count_mismatch(tmp_path: Path, synthetic_root: Path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(synthetic_root / "sequences" / "07", root / "sequences" / "07")
    poses = root / "sequences" / "07" / "poses.txt"
    poses.write_text("".join(poses.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(ValueError, match="poses"):
        Sequence(root, "07")


def test_bad_stride(synthetic_root: Path) -> None:
    with pytest.raises(ValueError):
        Sequence(synthetic_root, "08", frame_stride=0)


@pytest.mark.data
def test_real_sequence_loads(real_data_root: Path, real_sequences: list[str]) -> None:
    for name in real_sequences:
        seq = Sequence(real_data_root, name, frame_stride=100)
        scan = seq[0]
        assert scan.raw_labels is not None and len(scan.raw_labels) == len(scan.xyz)
