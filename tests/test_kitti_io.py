"""tests/test_kitti_io.py — Verification suite for foveamap.io.kitti and synthetic.

QA requirements addressed
--------------------------
R1  Binary I/O round-trip: write synthetic points and labels to temporary .bin /
    .label files; reload via load_velodyne_bin / load_labels; assert exact
    equality (np.array_equal for labels, np.allclose for float32 points).
R2  Shape (N, 4) and float32 dtype for all point arrays.
R3  Physical bounds: z_min ≥ −3.0 m (sensor height budget), z_max ≤ +15 m,
    intensity ∈ [0, 1].
R4  Label dtype uint32 and shape (N,) with N matching point count.
R5  Corrupt / bad-format error paths (FileNotFoundError, ValueError).
R6  Determinism: same seed → identical arrays.
R7  Super-class coverage: synthetic scan contains ≥ 1 point of each of
    DRIVABLE, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, DYNAMIC.
R8  No pass blocks: every test body contains at least one non-trivial assertion.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from foveamap.io.kitti import load_labels, load_velodyne_bin


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (no logic hidden behind pass)
# ─────────────────────────────────────────────────────────────────────────────

def _write_bin(path: Path, points: np.ndarray) -> None:
    """Write an (N, 4) float32 array as a Velodyne .bin file."""
    assert points.ndim == 2 and points.shape[1] == 4
    points.astype(np.float32).tofile(path)


def _write_label(path: Path, labels: np.ndarray) -> None:
    """Write an (N,) uint32 array as a SemanticKITTI .label file."""
    assert labels.ndim == 1
    labels.astype(np.uint32).tofile(path)


# ─────────────────────────────────────────────────────────────────────────────
# R2 + R5 — load_velodyne_bin: shape, dtype, values, errors
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadVelodyneBin:
    def test_shape_100_points(self, tmp_path: Path) -> None:
        pts = np.random.default_rng(0).random((100, 4)).astype(np.float32)
        f = tmp_path / "000000.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert result.shape == (100, 4), f"Expected (100, 4), got {result.shape}"

    def test_dtype_is_float32(self, tmp_path: Path) -> None:
        pts = np.ones((50, 4), dtype=np.float32)
        f = tmp_path / "test.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_values_allclose_after_roundtrip(self, tmp_path: Path) -> None:
        """Float32 values must survive write→read with near-exact precision."""
        pts = np.array([[1.0, 2.0, -1.73, 0.5]], dtype=np.float32)
        f = tmp_path / "precise.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        np.testing.assert_allclose(result, pts, rtol=1e-6,
                                   err_msg="Float values changed during binary round-trip")

    def test_column_order_xyzintensity(self, tmp_path: Path) -> None:
        """Columns must be in order x, y, z, intensity."""
        pts = np.array([[1.1, 2.2, 3.3, 0.77]], dtype=np.float32)
        f = tmp_path / "cols.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert result[0, 0] == pytest.approx(1.1, abs=1e-5)  # x
        assert result[0, 1] == pytest.approx(2.2, abs=1e-5)  # y
        assert result[0, 2] == pytest.approx(3.3, abs=1e-5)  # z
        assert result[0, 3] == pytest.approx(0.77, abs=1e-5) # intensity

    def test_120k_points_shape(self, tmp_path: Path) -> None:
        """Typical HDL-64E scan size must be handled without truncation."""
        pts = np.random.default_rng(1).random((120_000, 4)).astype(np.float32)
        f = tmp_path / "big.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert result.shape == (120_000, 4)

    def test_single_point(self, tmp_path: Path) -> None:
        pts = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        f = tmp_path / "single.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert result.shape == (1, 4)
        assert result[0, 3] == pytest.approx(1.0)

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        pts = np.ones((8, 4), dtype=np.float32)
        f = tmp_path / "str.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(str(f))
        assert result.shape == (8, 4)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Velodyne bin not found"):
            load_velodyne_bin("/no/such/file.bin")

    def test_corrupt_file_not_multiple_of_16_bytes(self, tmp_path: Path) -> None:
        """7 float32 values = 28 bytes, which is not divisible by 16."""
        f = tmp_path / "corrupt.bin"
        f.write_bytes(struct.pack("7f", *range(7)))
        with pytest.raises(ValueError, match="multiple of 16"):
            load_velodyne_bin(f)

    def test_corrupt_3_floats(self, tmp_path: Path) -> None:
        """3 float32 = 12 bytes — also not a multiple of 16."""
        f = tmp_path / "three.bin"
        f.write_bytes(struct.pack("3f", 1.0, 2.0, 3.0))
        with pytest.raises(ValueError):
            load_velodyne_bin(f)

    def test_negative_z_preserved(self, tmp_path: Path) -> None:
        """Negative z values (ground points) must survive the round-trip."""
        pts = np.array([[0.0, 0.0, -1.73, 0.3]], dtype=np.float32)
        f = tmp_path / "neg_z.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert result[0, 2] == pytest.approx(-1.73, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# R4 + R5 — load_labels: shape, dtype, values, errors
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadLabels:
    def test_shape_5_labels(self, tmp_path: Path) -> None:
        labels = np.array([40, 40, 10, 252, 30], dtype=np.uint32)
        f = tmp_path / "000000.label"
        _write_label(f, labels)
        result = load_labels(f)
        assert result.shape == (5,), f"Expected (5,), got {result.shape}"

    def test_dtype_is_uint32(self, tmp_path: Path) -> None:
        labels = np.array([40], dtype=np.uint32)
        f = tmp_path / "test.label"
        _write_label(f, labels)
        result = load_labels(f)
        assert result.dtype == np.uint32, f"Expected uint32, got {result.dtype}"

    def test_values_array_equal(self, tmp_path: Path) -> None:
        """Label values must be bit-for-bit identical after round-trip."""
        labels = np.array([0x0003000A, 0x00000028, 0x000000FC], dtype=np.uint32)
        f = tmp_path / "vals.label"
        _write_label(f, labels)
        result = load_labels(f)
        np.testing.assert_array_equal(result, labels,
                                      err_msg="Label values changed during binary round-trip")

    def test_120k_labels(self, tmp_path: Path) -> None:
        labels = np.full(120_000, 40, dtype=np.uint32)
        f = tmp_path / "big.label"
        _write_label(f, labels)
        result = load_labels(f)
        assert result.shape == (120_000,)
        assert np.all(result == 40)

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        labels = np.array([72], dtype=np.uint32)
        f = tmp_path / "str.label"
        _write_label(f, labels)
        result = load_labels(str(f))
        assert result.shape == (1,)
        assert int(result[0]) == 72

    def test_moving_car_252_preserved(self, tmp_path: Path) -> None:
        """Moving-car ID (252 = 0xFC) must survive without truncation."""
        labels = np.array([252], dtype=np.uint32)
        f = tmp_path / "mover.label"
        _write_label(f, labels)
        result = load_labels(f)
        assert int(result[0]) == 252, f"Expected 252, got {int(result[0])}"

    def test_max_uint32_preserved(self, tmp_path: Path) -> None:
        """0xFFFFFFFF must survive the round-trip (no truncation to uint16)."""
        labels = np.array([0xFFFFFFFF], dtype=np.uint32)
        f = tmp_path / "max.label"
        _write_label(f, labels)
        result = load_labels(f)
        assert int(result[0]) == 0xFFFFFFFF

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Label file not found"):
            load_labels("/no/such/file.label")

    def test_mixed_raw_ids_preserved(self, tmp_path: Path) -> None:
        """A mix of all raw semantic IDs from the master plan table."""
        raw_ids = [0, 1, 10, 30, 40, 48, 50, 72, 252, 259]
        labels = np.array(raw_ids, dtype=np.uint32)
        f = tmp_path / "mix.label"
        _write_label(f, labels)
        result = load_labels(f)
        np.testing.assert_array_equal(result, labels)


# ─────────────────────────────────────────────────────────────────────────────
# R1 — Binary I/O round-trip (explicit np.array_equal / np.allclose)
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryRoundTrip:
    """Write data → read back → assert exact equality or numerical closeness."""

    def test_points_allclose(self, tmp_path: Path) -> None:
        """Points (float32): np.allclose with tight tolerance."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-50, 50, (5000, 4)).astype(np.float32)
        f = tmp_path / "rt.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert np.allclose(result, pts, rtol=1e-6, atol=0), (
            "Point values changed beyond float32 tolerance during round-trip"
        )

    def test_points_array_equal(self, tmp_path: Path) -> None:
        """float32 round-trip must be bitwise identical (no precision loss)."""
        rng = np.random.default_rng(7)
        pts = rng.standard_normal((1000, 4)).astype(np.float32)
        f = tmp_path / "exact.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert np.array_equal(result, pts), (
            "float32 binary round-trip is not bitwise identical"
        )

    def test_labels_array_equal(self, tmp_path: Path) -> None:
        """Labels (uint32): np.array_equal — must be bit-perfect."""
        rng = np.random.default_rng(13)
        labels = rng.integers(0, 2**32, 10_000, dtype=np.uint64).astype(np.uint32)
        f = tmp_path / "rt.label"
        _write_label(f, labels)
        result = load_labels(f)
        assert np.array_equal(result, labels), (
            "uint32 labels changed during binary round-trip"
        )

    def test_joint_scan_label_roundtrip(self, tmp_path: Path) -> None:
        """Write both .bin and .label, reload, check both are identical and aligned."""
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, labels = generate_synthetic_scan(num_points=500, seed=5)
        bin_f = tmp_path / "scan.bin"
        lbl_f = tmp_path / "scan.label"
        _write_bin(bin_f, pts)
        _write_label(lbl_f, labels)
        pts_rt = load_velodyne_bin(bin_f)
        labels_rt = load_labels(lbl_f)
        assert np.array_equal(pts_rt, pts), "Points changed during round-trip"
        assert np.array_equal(labels_rt, labels), "Labels changed during round-trip"
        assert len(pts_rt) == len(labels_rt), "Points and labels misaligned"


# ─────────────────────────────────────────────────────────────────────────────
# R2 + R3 + R6 + R7 — Synthetic scan
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticScan:
    # ── R2: Shape and dtype ────────────────────────────────────────────────

    @pytest.mark.parametrize("N", [100, 500, 1000, 5000, 120_000])
    def test_exact_shape_various_N(self, N: int) -> None:
        """Output must have EXACTLY (N, 4) points and (N,) labels."""
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, labels = generate_synthetic_scan(num_points=N)
        assert pts.shape == (N, 4), f"N={N}: pts.shape={pts.shape}, expected ({N}, 4)"
        assert labels.shape == (N,), f"N={N}: labels.shape={labels.shape}, expected ({N},)"

    def test_dtype_float32(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, _ = generate_synthetic_scan(num_points=1000)
        assert pts.dtype == np.float32, f"Expected float32, got {pts.dtype}"

    def test_dtype_uint32_labels(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        _, labels = generate_synthetic_scan(num_points=1000)
        assert labels.dtype == np.uint32, f"Expected uint32, got {labels.dtype}"

    def test_4_columns(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, _ = generate_synthetic_scan(num_points=200)
        assert pts.shape[1] == 4, "Points must have exactly 4 columns (x, y, z, intensity)"

    # ── R3: Physical bounds ────────────────────────────────────────────────

    def test_z_min_physical_bound(self) -> None:
        """z must be ≥ −3.0 m (Velodyne sensor height budget per config z_range_m)."""
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, _ = generate_synthetic_scan(num_points=10_000)
        z_min = float(pts[:, 2].min())
        assert z_min >= -3.0, (
            f"z_min={z_min:.3f} m is below the physical sensor budget of -3.0 m"
        )

    def test_z_ground_near_minus_1_73(self) -> None:
        """Ground points (label 40 = road) must cluster near z ≈ −1.73 m."""
        from foveamap.io.synthetic import generate_synthetic_scan
        from foveamap.io.labels import unpack_kitti_labels
        pts, labels = generate_synthetic_scan(num_points=20_000)
        sem, _ = unpack_kitti_labels(labels)
        road_mask = sem == 40
        assert road_mask.sum() > 0, "No road points found"
        road_z = pts[road_mask, 2]
        z_median = float(np.median(road_z))
        assert -2.0 <= z_median <= -1.5, (
            f"Road z median={z_median:.3f} m; expected ≈ −1.73 m (±0.27 tolerance)"
        )

    def test_z_max_physical_bound(self) -> None:
        """No point should exceed z = +15 m in the synthetic scene."""
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, _ = generate_synthetic_scan(num_points=10_000)
        z_max = float(pts[:, 2].max())
        assert z_max <= 15.0, f"z_max={z_max:.3f} m exceeds physical plausibility of 15 m"

    def test_intensity_range_0_to_1(self) -> None:
        """All intensity values must be in [0, 1]."""
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, _ = generate_synthetic_scan(num_points=5000)
        intensity = pts[:, 3]
        assert float(intensity.min()) >= 0.0, f"Intensity below 0: min={intensity.min()}"
        assert float(intensity.max()) <= 1.0, f"Intensity above 1: max={intensity.max()}"

    def test_x_range_plausible(self) -> None:
        """X must be within ±200 m (far beyond the scene, but sanity check)."""
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, _ = generate_synthetic_scan(num_points=5000)
        assert float(pts[:, 0].min()) >= -200.0
        assert float(pts[:, 0].max()) <= 200.0

    def test_y_range_plausible(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, _ = generate_synthetic_scan(num_points=5000)
        assert float(pts[:, 1].min()) >= -200.0
        assert float(pts[:, 1].max()) <= 200.0

    # ── R6: Determinism ────────────────────────────────────────────────────

    def test_same_seed_same_output_points(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        pts1, _ = generate_synthetic_scan(num_points=3000, seed=17)
        pts2, _ = generate_synthetic_scan(num_points=3000, seed=17)
        assert np.array_equal(pts1, pts2), "Same seed must produce identical point arrays"

    def test_same_seed_same_output_labels(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        _, lab1 = generate_synthetic_scan(num_points=3000, seed=17)
        _, lab2 = generate_synthetic_scan(num_points=3000, seed=17)
        assert np.array_equal(lab1, lab2), "Same seed must produce identical label arrays"

    def test_different_seeds_different_points(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        pts1, _ = generate_synthetic_scan(num_points=1000, seed=1)
        pts2, _ = generate_synthetic_scan(num_points=1000, seed=2)
        assert not np.array_equal(pts1, pts2), "Different seeds must produce different points"

    def test_default_seed_deterministic(self) -> None:
        """Calling with default seed=42 twice must give the same result."""
        from foveamap.io.synthetic import generate_synthetic_scan
        pts1, lab1 = generate_synthetic_scan()
        pts2, lab2 = generate_synthetic_scan()
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(lab1, lab2)

    # ── R7: Super-class coverage ───────────────────────────────────────────

    def test_contains_drivable(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        from foveamap.io.labels import unpack_kitti_labels, to_superclass, DRIVABLE
        pts, raw = generate_synthetic_scan(num_points=5000)
        sem, _ = unpack_kitti_labels(raw)
        sc = to_superclass(sem)
        n = int(np.sum(sc == DRIVABLE))
        assert n > 0, f"Expected DRIVABLE points, got 0"

    def test_contains_non_drivable_terrain(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        from foveamap.io.labels import unpack_kitti_labels, to_superclass, NON_DRIVABLE_TERRAIN
        pts, raw = generate_synthetic_scan(num_points=5000)
        sem, _ = unpack_kitti_labels(raw)
        sc = to_superclass(sem)
        n = int(np.sum(sc == NON_DRIVABLE_TERRAIN))
        assert n > 0, f"Expected NON_DRIVABLE_TERRAIN points, got 0"

    def test_contains_static_obstacle(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        from foveamap.io.labels import unpack_kitti_labels, to_superclass, STATIC_OBSTACLE
        pts, raw = generate_synthetic_scan(num_points=5000)
        sem, _ = unpack_kitti_labels(raw)
        sc = to_superclass(sem)
        n = int(np.sum(sc == STATIC_OBSTACLE))
        assert n > 0, f"Expected STATIC_OBSTACLE points, got 0"

    def test_contains_dynamic(self) -> None:
        from foveamap.io.synthetic import generate_synthetic_scan
        from foveamap.io.labels import unpack_kitti_labels, to_superclass, DYNAMIC
        pts, raw = generate_synthetic_scan(num_points=5000)
        sem, _ = unpack_kitti_labels(raw)
        sc = to_superclass(sem)
        n = int(np.sum(sc == DYNAMIC))
        assert n > 0, f"Expected DYNAMIC points, got 0"

    def test_all_four_superclasses_present(self) -> None:
        """Single call that verifies all 4 super-classes are present."""
        from foveamap.io.synthetic import generate_synthetic_scan
        from foveamap.io.labels import (
            DRIVABLE, DYNAMIC, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE,
            unpack_kitti_labels, to_superclass,
        )
        pts, raw = generate_synthetic_scan(num_points=10_000)
        sem, _ = unpack_kitti_labels(raw)
        sc = to_superclass(sem)
        missing = [
            name for name, sc_id in [
                ("DRIVABLE", DRIVABLE),
                ("NON_DRIVABLE_TERRAIN", NON_DRIVABLE_TERRAIN),
                ("STATIC_OBSTACLE", STATIC_OBSTACLE),
                ("DYNAMIC", DYNAMIC),
            ]
            if not np.any(sc == sc_id)
        ]
        assert missing == [], f"Missing super-classes in synthetic scan: {missing}"

    def test_no_unknown_points(self) -> None:
        """Synthetic scan must not contain any UNKNOWN points (all IDs are known)."""
        from foveamap.io.synthetic import generate_synthetic_scan
        from foveamap.io.labels import unpack_kitti_labels, to_superclass, UNKNOWN
        pts, raw = generate_synthetic_scan(num_points=5000)
        sem, _ = unpack_kitti_labels(raw)
        sc = to_superclass(sem)
        n_unknown = int(np.sum(sc == UNKNOWN))
        assert n_unknown == 0, (
            f"Found {n_unknown} UNKNOWN points — all synthetic IDs should be mapped"
        )

    # ── Integration: round-trip with kitti IO ─────────────────────────────

    def test_roundtrip_points_array_equal(self, tmp_path: Path) -> None:
        """Synthetic points must survive a .bin write→read with bitwise equality."""
        from foveamap.io.synthetic import generate_synthetic_scan
        pts, _ = generate_synthetic_scan(num_points=500, seed=3)
        f = tmp_path / "rt.bin"
        _write_bin(f, pts)
        result = load_velodyne_bin(f)
        assert np.array_equal(result, pts), (
            "float32 points changed during .bin round-trip (not bitwise identical)"
        )

    def test_roundtrip_labels_array_equal(self, tmp_path: Path) -> None:
        """Synthetic labels must survive a .label write→read with exact equality."""
        from foveamap.io.synthetic import generate_synthetic_scan
        _, labels = generate_synthetic_scan(num_points=500, seed=3)
        f = tmp_path / "rt.label"
        _write_label(f, labels)
        result = load_labels(f)
        assert np.array_equal(result, labels), (
            "uint32 labels changed during .label round-trip"
        )

    def test_points_labels_count_match_after_roundtrip(self, tmp_path: Path) -> None:
        """N points must equal N labels after round-trip through file system."""
        from foveamap.io.synthetic import generate_synthetic_scan
        N = 800
        pts, labels = generate_synthetic_scan(num_points=N, seed=9)
        bin_f = tmp_path / "scan.bin"
        lbl_f = tmp_path / "scan.label"
        _write_bin(bin_f, pts)
        _write_label(lbl_f, labels)
        pts_rt = load_velodyne_bin(bin_f)
        labels_rt = load_labels(lbl_f)
        assert len(pts_rt) == len(labels_rt) == N, (
            f"Mismatch: pts={len(pts_rt)}, labels={len(labels_rt)}, expected {N}"
        )
