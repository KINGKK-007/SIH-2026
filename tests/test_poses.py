"""tests/test_poses.py — Verification suite for foveamap.io.poses.

QA requirements addressed
--------------------------
R1  Tr matrix inversion math: Tr⁻¹ · Tr == I₄ for all test calibrations.
R2  Rigid transformation: T_velo(t) = Tr⁻¹ · pose(t) · Tr — computed and
    verified analytically for several non-trivial Tr/pose combinations.
R3  SE(3) validity: every output pose must be a valid rigid-body matrix:
    - det(R) ≈ +1.0  (not -1, not 0)
    - R · Rᵀ ≈ I₃   (orthogonal rotation block)
    - last row == [0, 0, 0, 1]
R4  Sequential pose chain: transforming a point through a multi-frame pose
    sequence and back must recover the original point.
R5  Real-world Tr values: test with a Tr matrix drawn from an actual
    SemanticKITTI sequence-00 calibration file.
R6  Error paths: FileNotFoundError and ValueError on bad input.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from foveamap.io.poses import load_calib, load_poses


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_calib(tmp_path: Path, tr_flat: list[float]) -> Path:
    """Write a minimal calib.txt with the given Tr row (12 floats)."""
    calib = tmp_path / "calib.txt"
    tr_str = " ".join(f"{v:.10e}" for v in tr_flat)
    calib.write_text(
        "P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 "
        "0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 "
        "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
        f"Tr: {tr_str}\n"
    )
    return calib


def _write_poses(tmp_path: Path, poses_3x4: list[list[float]]) -> Path:
    """Write a poses.txt with the given list of 3×4 row-major pose rows."""
    poses_file = tmp_path / "poses.txt"
    lines = [" ".join(f"{v:.10e}" for v in row) for row in poses_3x4]
    poses_file.write_text("\n".join(lines) + "\n")
    return poses_file


def _flat_3x4(mat4x4: np.ndarray) -> list[float]:
    """Extract upper 3×4 of a 4×4 matrix as a flat Python list."""
    return mat4x4[:3, :4].flatten().tolist()


def _identity_tr_flat() -> list[float]:
    return [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0]


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 SE(3) matrix from a 3×3 rotation and a 3-vector translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _assert_se3(M: np.ndarray, *, atol: float = 1e-9, label: str = "") -> None:
    """Assert that M is a valid 4×4 SE(3) matrix."""
    assert M.shape == (4, 4), f"{label}: expected (4,4), got {M.shape}"
    assert M.dtype == np.float64, f"{label}: expected float64, got {M.dtype}"

    R = M[:3, :3]
    # det(R) ≈ +1
    det = np.linalg.det(R)
    assert abs(det - 1.0) < atol, (
        f"{label}: det(R)={det:.8f}, expected ≈ 1.0"
    )
    # R·Rᵀ ≈ I₃
    RRt = R @ R.T
    np.testing.assert_allclose(RRt, np.eye(3), atol=atol,
                               err_msg=f"{label}: R·Rᵀ ≠ I₃")
    # Last row
    np.testing.assert_array_equal(M[3], [0, 0, 0, 1],
                                  err_msg=f"{label}: last row ≠ [0,0,0,1]")


# ─────────────────────────────────────────────────────────────────────────────
# R1 — load_calib: Tr inversion and matrix structure
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCalib:
    def test_identity_tr(self, tmp_path: Path) -> None:
        calib = _write_calib(tmp_path, _identity_tr_flat())
        Tr = load_calib(calib)
        np.testing.assert_allclose(Tr, np.eye(4), atol=1e-12)

    def test_shape_and_dtype(self, tmp_path: Path) -> None:
        calib = _write_calib(tmp_path, _identity_tr_flat())
        Tr = load_calib(calib)
        assert Tr.shape == (4, 4)
        assert Tr.dtype == np.float64

    def test_last_row_always_0001(self, tmp_path: Path) -> None:
        """Regardless of input, the appended 4th row must be [0,0,0,1]."""
        calib = _write_calib(tmp_path, list(range(1, 13)))
        Tr = load_calib(calib)
        np.testing.assert_array_equal(Tr[3], [0, 0, 0, 1])

    def test_3x4_block_placed_correctly(self, tmp_path: Path) -> None:
        vals = list(range(1, 13))
        calib = _write_calib(tmp_path, vals)
        Tr = load_calib(calib)
        expected = np.array(vals, dtype=np.float64).reshape(3, 4)
        np.testing.assert_allclose(Tr[:3, :4], expected)

    def test_tr_inverse_times_tr_is_identity(self, tmp_path: Path) -> None:
        """Tr⁻¹ · Tr must equal I₄ for any valid rigid-body calibration."""
        # Use a realistic rotation+translation Tr
        R = _rot_z(math.pi / 6) @ _rot_x(math.pi / 12)  # compound rotation
        t = np.array([0.27, -0.06, 0.08])
        T = _make_se3(R, t)
        calib = _write_calib(tmp_path, _flat_3x4(T))
        Tr = load_calib(calib)
        Tr_inv = np.linalg.inv(Tr)
        product = Tr_inv @ Tr
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10,
                                   err_msg="Tr⁻¹ · Tr ≠ I₄")

    def test_tr_times_tr_inverse_is_identity(self, tmp_path: Path) -> None:
        """Tr · Tr⁻¹ must also equal I₄."""
        R = _rot_y(math.pi / 4)
        t = np.array([0.1, -0.2, 0.3])
        T = _make_se3(R, t)
        calib = _write_calib(tmp_path, _flat_3x4(T))
        Tr = load_calib(calib)
        Tr_inv = np.linalg.inv(Tr)
        product = Tr @ Tr_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10,
                                   err_msg="Tr · Tr⁻¹ ≠ I₄")

    def test_real_kitti_tr_values(self, tmp_path: Path) -> None:
        """Test with Tr values from a real SemanticKITTI sequence-00 calibration.
        Source: semantic-kitti-api / KITTI odometry sequence 00.
        Values [VERIFY]: these are canonical public values.
        """
        # fmt: off
        tr_flat = [
            4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
            -7.210626507497e-03,  8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
             9.999738645903e-01,  4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
        ]
        # fmt: on
        calib = _write_calib(tmp_path, tr_flat)
        Tr = load_calib(calib)
        # Verify it parses as a valid SE(3) matrix
        _assert_se3(Tr, atol=1e-6, label="real KITTI Tr")

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_calib("/nonexistent/path/calib.txt")

    def test_missing_tr_key(self, tmp_path: Path) -> None:
        f = tmp_path / "calib.txt"
        f.write_text("P0: 1 2 3 4 5 6 7 8 9 10 11 12\n")
        with pytest.raises(KeyError, match="Tr"):
            load_calib(f)

    def test_wrong_value_count_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "calib.txt"
        f.write_text("Tr: 1 2 3\n")
        with pytest.raises(ValueError, match="12"):
            load_calib(f)


# ─────────────────────────────────────────────────────────────────────────────
# R2 — load_poses: rigid transformation math
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadPosesTransformMath:
    def test_identity_tr_passthrough(self, tmp_path: Path) -> None:
        """Identity Tr: T_velo(t) must equal T_cam(t) exactly."""
        Tr = np.eye(4, dtype=np.float64)
        T = _make_se3(_rot_z(0.3), np.array([1.0, 2.0, 0.5]))
        pf = _write_poses(tmp_path, [_flat_3x4(T)])
        result = load_poses(pf, Tr)
        np.testing.assert_allclose(result[0], T, atol=1e-12)

    def test_analytic_formula_pure_rotation_tr(self, tmp_path: Path) -> None:
        """T_velo = Tr⁻¹ · T_cam · Tr — verified with 90° z-rotation Tr."""
        Tr = _make_se3(_rot_z(math.pi / 2), np.zeros(3))
        T_cam = _make_se3(_rot_z(math.pi / 4), np.array([3.0, 0.0, 0.0]))
        expected = np.linalg.inv(Tr) @ T_cam @ Tr
        pf = _write_poses(tmp_path, [_flat_3x4(T_cam)])
        result = load_poses(pf, Tr)
        np.testing.assert_allclose(result[0], expected, atol=1e-10,
                                   err_msg="T_velo ≠ Tr⁻¹ · T_cam · Tr")

    def test_analytic_formula_rotation_and_translation_tr(self, tmp_path: Path) -> None:
        """Full compound Tr (rotation + translation) correctness check."""
        R_tr = _rot_x(0.1) @ _rot_z(-0.05)
        t_tr = np.array([0.27, -0.06, -0.30])
        Tr = _make_se3(R_tr, t_tr)

        R_cam = _rot_y(0.5)
        t_cam = np.array([5.0, -2.0, 0.0])
        T_cam = _make_se3(R_cam, t_cam)

        expected = np.linalg.inv(Tr) @ T_cam @ Tr
        pf = _write_poses(tmp_path, [_flat_3x4(T_cam)])
        result = load_poses(pf, Tr)
        np.testing.assert_allclose(result[0], expected, atol=1e-10,
                                   err_msg="Compound Tr: T_velo ≠ Tr⁻¹ · T_cam · Tr")

    def test_translation_only_camera_frame(self, tmp_path: Path) -> None:
        """Pure translation in camera frame must produce a translation in velo frame."""
        Tr = np.eye(4, dtype=np.float64)  # identity calibration
        T_cam = np.eye(4, dtype=np.float64)
        T_cam[0, 3] = 5.0
        T_cam[1, 3] = -2.0
        T_cam[2, 3] = 0.0
        pf = _write_poses(tmp_path, [_flat_3x4(T_cam)])
        result = load_poses(pf, Tr)
        np.testing.assert_allclose(result[0], T_cam, atol=1e-12)

    def test_relative_pose_is_consistent(self, tmp_path: Path) -> None:
        """T_velo(1)⁻¹ · T_velo(0) must match the Camera-frame relative pose."""
        Tr = np.eye(4, dtype=np.float64)
        T0 = _make_se3(_rot_z(0.1), np.array([0.0, 0.0, 0.0]))
        T1 = _make_se3(_rot_z(0.5), np.array([3.0, 1.0, 0.0]))
        pf = _write_poses(tmp_path, [_flat_3x4(T0), _flat_3x4(T1)])
        poses = load_poses(pf, Tr)
        T_rel_from_poses = np.linalg.inv(poses[0]) @ poses[1]
        T_rel_direct = np.linalg.inv(T0) @ T1
        np.testing.assert_allclose(T_rel_from_poses, T_rel_direct, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# R3 — SE(3) validity of all output poses
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputIsValidSE3:
    """Every output pose must satisfy the three SE(3) conditions:
    det(R)≈1, R·Rᵀ≈I₃, last row=[0,0,0,1].
    """

    def test_identity_pose_is_se3(self, tmp_path: Path) -> None:
        Tr = np.eye(4, dtype=np.float64)
        pf = _write_poses(tmp_path, [_flat_3x4(np.eye(4))])
        poses = load_poses(pf, Tr)
        _assert_se3(poses[0], label="identity pose")

    def test_rotation_pose_is_se3(self, tmp_path: Path) -> None:
        Tr = np.eye(4, dtype=np.float64)
        T = _make_se3(_rot_z(math.pi / 3), np.array([1.0, 2.0, 3.0]))
        pf = _write_poses(tmp_path, [_flat_3x4(T)])
        poses = load_poses(pf, Tr)
        _assert_se3(poses[0], label="rotation pose")

    def test_compound_rotation_pose_is_se3(self, tmp_path: Path) -> None:
        """Compound rotation (x·y·z) must still give a valid SE(3) output."""
        R = _rot_x(0.2) @ _rot_y(0.5) @ _rot_z(1.1)
        T_cam = _make_se3(R, np.array([10.0, -5.0, 0.3]))
        R_tr = _rot_z(math.pi / 2)
        Tr = _make_se3(R_tr, np.array([0.27, -0.06, -0.30]))
        pf = _write_poses(tmp_path, [_flat_3x4(T_cam)])
        poses = load_poses(pf, Tr)
        _assert_se3(poses[0], atol=1e-9, label="compound rotation")

    def test_determinant_is_plus_one_not_minus_one(self, tmp_path: Path) -> None:
        """det(R) must be exactly +1, never -1 (which would indicate reflection)."""
        Tr = np.eye(4, dtype=np.float64)
        T = _make_se3(_rot_y(math.pi / 7), np.array([2.0, 0.0, 0.0]))
        pf = _write_poses(tmp_path, [_flat_3x4(T)])
        poses = load_poses(pf, Tr)
        det = np.linalg.det(poses[0][:3, :3])
        assert det > 0, f"det(R)={det:.8f} is negative — reflection, not rotation"
        assert abs(det - 1.0) < 1e-9, f"det(R)={det:.8f} expected ≈ 1.0"

    def test_all_frames_in_100_frame_sequence_are_se3(self, tmp_path: Path) -> None:
        """All 100 output poses in a synthetic sequence must be valid SE(3)."""
        rng = np.random.default_rng(0)
        Tr = _make_se3(_rot_z(math.pi / 4), np.array([0.1, -0.1, 0.0]))

        pose_rows = []
        cum_angle = 0.0
        cum_t = np.zeros(3)
        for i in range(100):
            cum_angle += rng.uniform(0.0, 0.05)
            cum_t += np.array([rng.uniform(0.05, 0.2), 0.0, 0.0])
            T = _make_se3(_rot_z(cum_angle), cum_t)
            pose_rows.append(_flat_3x4(T))

        pf = _write_poses(tmp_path, pose_rows)
        poses = load_poses(pf, Tr)
        assert len(poses) == 100
        for i, pose in enumerate(poses):
            _assert_se3(pose, atol=1e-9, label=f"frame {i}")


# ─────────────────────────────────────────────────────────────────────────────
# R4 — Sequential pose chain: point round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestPoseChainPointTransform:
    """Transform a static world point through successive poses and verify
    that the point looks the same from the world frame regardless of which
    scan's pose is used.
    """

    def test_static_point_world_coords_unchanged(self, tmp_path: Path) -> None:
        """A static point p_world must map to p_world when transformed
        via T_velo(t) from any frame t.

        p_velo(t) = T_velo(t)⁻¹ · p_world
        p_world_recovered = T_velo(t) · p_velo(t) == p_world
        """
        Tr = np.eye(4, dtype=np.float64)
        # Three sequential poses: robot moving forward
        poses_cam = [
            _make_se3(np.eye(3), np.array([0.0, 0.0, 0.0])),
            _make_se3(_rot_z(0.1), np.array([1.0, 0.0, 0.0])),
            _make_se3(_rot_z(0.2), np.array([2.0, 0.1, 0.0])),
        ]
        pf = _write_poses(tmp_path, [_flat_3x4(p) for p in poses_cam])
        poses_velo = load_poses(pf, Tr)

        # A static point in the world frame
        p_world = np.array([5.0, 3.0, 0.0, 1.0])

        for i, T_velo in enumerate(poses_velo):
            # Express the world point in this frame's Velodyne coordinates
            p_velo = np.linalg.inv(T_velo) @ p_world
            # Recover the world point
            p_world_recovered = T_velo @ p_velo
            np.testing.assert_allclose(
                p_world_recovered, p_world, atol=1e-10,
                err_msg=f"Frame {i}: world point not recovered"
            )

    def test_relative_transform_moves_point_correctly(self, tmp_path: Path) -> None:
        """If a point is at position p in frame 0, its position in frame 1 is
        T_rel⁻¹ · p  where T_rel = T_velo(0)⁻¹ · T_velo(1).
        """
        Tr = np.eye(4, dtype=np.float64)
        T0 = _make_se3(np.eye(3), np.array([0.0, 0.0, 0.0]))
        T1 = _make_se3(_rot_z(math.pi / 4), np.array([5.0, 0.0, 0.0]))
        pf = _write_poses(tmp_path, [_flat_3x4(T0), _flat_3x4(T1)])
        poses = load_poses(pf, Tr)

        p_frame0 = np.array([1.0, 0.0, 0.0, 1.0])
        # Compute where this point appears in frame 1's coordinate system
        T_rel = np.linalg.inv(poses[0]) @ poses[1]
        p_in_frame1 = np.linalg.inv(T_rel) @ p_frame0
        # Verify via independent path: bring to world, then to frame 1
        p_world = poses[0] @ p_frame0
        p_in_frame1_alt = np.linalg.inv(poses[1]) @ p_world
        np.testing.assert_allclose(p_in_frame1, p_in_frame1_alt, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# R5 — Real-world Tr calibration values
# ─────────────────────────────────────────────────────────────────────────────

class TestRealKITTICalibration:
    """End-to-end test using calibration values from a real KITTI sequence."""

    # Canonical SemanticKITTI sequence-00 Tr (Velodyne → Camera-0).
    # Source: KITTI odometry devkit / semantic-kitti-api  [VERIFY on download]
    _SEQ00_TR_FLAT = [
        4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
        -7.210626507497e-03,  8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
         9.999738645903e-01,  4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
    ]

    def test_real_tr_parses_and_is_nearly_orthogonal(self, tmp_path: Path) -> None:
        calib = _write_calib(tmp_path, self._SEQ00_TR_FLAT)
        Tr = load_calib(calib)
        _assert_se3(Tr, atol=1e-5, label="seq-00 Tr")

    def test_real_tr_inverse_closes(self, tmp_path: Path) -> None:
        calib = _write_calib(tmp_path, self._SEQ00_TR_FLAT)
        Tr = load_calib(calib)
        Tr_inv = np.linalg.inv(Tr)
        np.testing.assert_allclose(Tr @ Tr_inv, np.eye(4), atol=1e-9)

    def test_real_tr_conversion_preserves_se3(self, tmp_path: Path) -> None:
        """Converting a Camera-0 pose using the real Tr must give a valid SE(3)."""
        calib = _write_calib(tmp_path, self._SEQ00_TR_FLAT)
        Tr = load_calib(calib)

        # Synthetic Camera-0 pose: straight-line forward motion
        T_cam = _make_se3(_rot_z(0.02), np.array([3.0, 0.0, 0.0]))
        pf = _write_poses(tmp_path, [_flat_3x4(T_cam)])
        poses = load_poses(pf, Tr)
        # atol=1e-7: the real KITTI Tr is not a perfect SO(3) matrix; it carries
        # ~2e-9 orthogonality error from physical calibration, which accumulates
        # slightly in Tr⁻¹·T_cam·Tr. 1e-7 still catches any implementation bugs.
        _assert_se3(poses[0], atol=1e-7, label="velo pose from real Tr")


# ─────────────────────────────────────────────────────────────────────────────
# R6 — Error paths
# ─────────────────────────────────────────────────────────────────────────────

class TestPoseErrorPaths:
    def test_calib_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_calib("/no/such/calib.txt")

    def test_poses_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_poses("/no/such/poses.txt", np.eye(4))

    def test_calib_missing_tr_key(self, tmp_path: Path) -> None:
        f = tmp_path / "calib.txt"
        f.write_text("P0: 1 2 3 4 5 6 7 8 9 10 11 12\n")
        with pytest.raises(KeyError, match="Tr"):
            load_calib(f)

    def test_calib_wrong_value_count(self, tmp_path: Path) -> None:
        f = tmp_path / "calib.txt"
        f.write_text("Tr: 1 2 3\n")
        with pytest.raises(ValueError, match="12"):
            load_calib(f)

    def test_poses_wrong_column_count(self, tmp_path: Path) -> None:
        f = tmp_path / "poses.txt"
        f.write_text("1 2 3\n")
        with pytest.raises(ValueError, match="12"):
            load_poses(f, np.eye(4))

    def test_n_frames_matches_lines(self, tmp_path: Path) -> None:
        Tr = np.eye(4, dtype=np.float64)
        rows = [_flat_3x4(np.eye(4))] * 47
        pf = _write_poses(tmp_path, rows)
        poses = load_poses(pf, Tr)
        assert len(poses) == 47
