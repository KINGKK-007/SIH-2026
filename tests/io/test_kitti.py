"""T3.1: binary scan and label loading (README 5.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foveamap.io.kitti import load_label, load_scan_bin, split_label


def _write_bin(path: Path, pts: np.ndarray) -> Path:
    pts.astype(np.float32).tofile(path)
    return path


def test_load_scan_bin_shapes_and_values(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(1000, 4)).astype(np.float32)
    xyz, rem = load_scan_bin(_write_bin(tmp_path / "000000.bin", pts))
    assert xyz.shape == (1000, 3) and xyz.dtype == np.float32
    assert rem.shape == (1000,) and rem.dtype == np.float32
    np.testing.assert_array_equal(xyz, pts[:, :3])
    np.testing.assert_array_equal(rem, pts[:, 3])
    assert xyz.flags.c_contiguous and rem.flags.c_contiguous


def test_load_scan_bin_empty(tmp_path: Path) -> None:
    xyz, rem = load_scan_bin(_write_bin(tmp_path / "e.bin", np.zeros((0, 4))))
    assert xyz.shape == (0, 3) and rem.shape == (0,)


def test_load_scan_bin_rejects_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\x00" * 20)
    with pytest.raises(ValueError, match="multiple of 16"):
        load_scan_bin(path)


def test_load_label_dtype_and_split(tmp_path: Path) -> None:
    raw = np.array([0, 40, (7 << 16) | 252, (65535 << 16) | 10, 99], dtype=np.uint32)
    path = tmp_path / "000000.label"
    raw.tofile(path)
    loaded = load_label(path)
    assert loaded.dtype == np.uint32 and loaded.shape == (5,)
    np.testing.assert_array_equal(loaded, raw)
    semantic, instance = split_label(loaded)
    assert semantic.dtype == np.uint16 and instance.dtype == np.uint16
    np.testing.assert_array_equal(semantic, [0, 40, 252, 10, 99])
    np.testing.assert_array_equal(instance, [0, 0, 7, 65535, 0])


def test_load_label_rejects_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.label"
    path.write_bytes(b"\x00" * 6)
    with pytest.raises(ValueError, match="multiple of 4"):
        load_label(path)


@pytest.mark.data
def test_real_scan_and_label_lengths_match(real_data_root: Path, real_sequences: list[str]) -> None:
    for seq in real_sequences:
        seq_dir = real_data_root / "sequences" / seq
        scan = sorted((seq_dir / "velodyne").glob("*.bin"))[0]
        xyz, rem = load_scan_bin(scan)
        labels = load_label(seq_dir / "labels" / (scan.stem + ".label"))
        assert xyz.dtype == np.float32 and rem.dtype == np.float32 and labels.dtype == np.uint32
        assert len(labels) == len(xyz) > 10_000
        assert np.isfinite(xyz).all()
