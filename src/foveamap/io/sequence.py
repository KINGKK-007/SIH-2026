"""Lazy, indexable access to one SemanticKITTI sequence (README 9.2, task T3.4)."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from foveamap.io.kitti import load_label, load_scan_bin
from foveamap.io.poses import Calib, load_calib, load_poses, relative_transform
from foveamap.pipeline.records import Scan

SENSOR_PERIOD_S = 0.1  # L3: 10 Hz


def find_poses_file(root: Path, seq: str) -> Path | None:
    """``sequences/<seq>/poses.txt`` or, as shipped by ``data_odometry_poses.zip``, ``poses/<seq>.txt``."""
    for candidate in (root / "sequences" / seq / "poses.txt", root / "poses" / f"{seq}.txt"):
        if candidate.is_file():
            return candidate
    return None


class Sequence:
    """Lazy, indexable sequence of :class:`Scan` records.

    ``frame_stride = k`` exposes every k-th frame; ``Scan.idx`` is always the original frame index
    (``seq[i].idx == i * k``). Scans are read from disk on access; nothing is cached.
    """

    def __init__(self, root: str | Path, seq: str, with_labels: bool = True, frame_stride: int = 1) -> None:
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        self.root = Path(root)
        self.seq = seq
        self.with_labels = with_labels
        self.frame_stride = frame_stride
        self.seq_dir = self.root / "sequences" / seq
        self._scan_paths = sorted((self.seq_dir / "velodyne").glob("*.bin"))
        if not self._scan_paths:
            raise FileNotFoundError(f"no scans in {self.seq_dir / 'velodyne'}")
        self.calib: Calib = load_calib(self.seq_dir / "calib.txt")
        poses_file = find_poses_file(self.root, seq)
        if poses_file is None:
            raise FileNotFoundError(f"no poses for sequence {seq} under {self.root}")
        self.poses = load_poses(poses_file)
        if len(self.poses) != len(self._scan_paths):
            raise ValueError(f"{poses_file}: {len(self.poses)} poses for {len(self._scan_paths)} scans")
        self.times = self._load_times()
        if with_labels:
            missing = [p.stem for p in self._scan_paths if not self._label_path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} label files missing in {self.seq_dir / 'labels'}")

    def _label_path(self, scan_path: Path) -> Path:
        return self.seq_dir / "labels" / (scan_path.stem + ".label")

    def _load_times(self) -> np.ndarray:
        n = len(self._scan_paths)
        times_file = self.seq_dir / "times.txt"
        if not times_file.is_file():
            warnings.warn(f"{times_file} missing; assuming {SENSOR_PERIOD_S} s spacing", stacklevel=3)
            return np.arange(n) * SENSOR_PERIOD_S
        times = np.loadtxt(times_file, dtype=np.float64, ndmin=1)
        if len(times) != n:
            raise ValueError(f"{times_file}: {len(times)} timestamps for {n} scans")
        return times

    @property
    def n_frames_total(self) -> int:
        """Number of frames on disk, ignoring the stride."""
        return len(self._scan_paths)

    def frame_index(self, i: int) -> int:
        """Original frame index of the i-th strided item (negative ``i`` counts from the end)."""
        n = len(self)
        if not -n <= i < n:
            raise IndexError(f"index {i} out of range for {n} frames")
        return (i % n) * self.frame_stride

    def __len__(self) -> int:
        return -(-self.n_frames_total // self.frame_stride)

    def __getitem__(self, i: int) -> Scan:
        return self.load_frame(self.frame_index(i))

    def __iter__(self) -> Iterator[Scan]:
        for i in range(len(self)):
            yield self[i]

    def load_frame(self, frame: int) -> Scan:
        """Load an original frame index, ignoring the stride."""
        path = self._scan_paths[frame]
        xyz, remission = load_scan_bin(path)
        raw = None
        if self.with_labels:
            raw = load_label(self._label_path(path))
            if len(raw) != len(xyz):
                raise ValueError(f"{path.stem}: {len(raw)} labels for {len(xyz)} points")
        return Scan(
            seq=self.seq,
            idx=frame,
            xyz=xyz,
            remission=remission,
            raw_labels=raw,
            pose=self.poses[frame],
            timestamp=float(self.times[frame]),
        )

    def relative_transform(self, i: int, j: int) -> np.ndarray:
        """``T_vel_i_from_vel_j`` between original frame indices (README 5.3)."""
        return relative_transform(self.calib, self.poses, i, j)
