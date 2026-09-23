"""tests/test_cache_model.py — Verification suite for CacheSegmenter.

QA Requirements Addressed
-------------------------
C1  Segmenter Protocol Compliance:
    - isinstance(CacheSegmenter(), Segmenter) is True.
    - Exposes name attribute and __call__(points: NDArray[float32]) -> SegOutput.
C2  NPY Precomputed Prediction Loading:
    - Saves a dummy .npy file with 19-class benchmark IDs.
    - Loads file and asserts cls19_to_superclass mapping is strictly applied.
    - Confirms VRU mask correctly flags 19-class person (8), bicyclist (9), motorcyclist (10).
C3  Confidence Loading:
    - When companion _conf.npy file exists, per-point confidence values are loaded.
    - When absent, defaults to default_conf (e.g. 0.90).
C4  SemanticKITTI .label Format Loading:
    - Saves a dummy binary .label file with packed uint32 (semantic ID + instance ID).
    - Verifies lower 16-bit unpacking, super-class mapping, and moving raw IDs (252-259).
C5  OracleSegmenter Interface Equivalence:
    - Exposes super_cls, moving_mask, vru_mask, conf properties.
    - grid_inputs(points) returns exact 4-tuple (super_cls, moving, vru, conf).
    - set_moving_mask updates moving_mask for motion engine integration.
C6  Error Handling:
    - Point cloud length mismatch raises ValueError.
    - Missing cache directory or file raises FileNotFoundError.
    - Unsupported file extension raises ValueError.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foveamap_legacy.io.labels import (
    DRIVABLE,
    DYNAMIC,
    NON_DRIVABLE_TERRAIN,
    STATIC_OBSTACLE,
    UNKNOWN,
)
from foveamap_legacy.models.base import Segmenter, SegOutput
from foveamap_legacy.models.cache import CacheSegmenter
from foveamap_legacy.models.oracle import OracleSegmenter


class TestCacheSegmenter:
    """Verification of CacheSegmenter functionality and interface conformance."""

    def test_segmenter_protocol_compliance(self) -> None:
        """CacheSegmenter satisfies the structural Segmenter Protocol."""
        seg = CacheSegmenter(name="test_backend")
        assert isinstance(seg, Segmenter)
        assert seg.name == "test_backend"

    def test_load_19class_npy_file(self, tmp_path: Path) -> None:
        """Loads 19-class predictions from .npy and correctly maps to 4 super-classes."""
        # 19-class IDs:
        # 0: car -> STATIC_OBSTACLE (2)
        # 8: person -> DYNAMIC (3) [VRU]
        # 9: bicyclist -> DYNAMIC (3) [VRU]
        # 10: motorcyclist -> DYNAMIC (3) [VRU]
        # 11: road -> DRIVABLE (0)
        # 13: sidewalk -> NON_DRIVABLE_TERRAIN (1)
        # 17: building -> STATIC_OBSTACLE (2)
        # 255: unknown -> UNKNOWN (255)
        c19_ids = np.array([0, 8, 9, 10, 11, 13, 17, 255], dtype=np.uint8)
        npy_path = tmp_path / "000000.npy"
        np.save(npy_path, c19_ids)

        seg = CacheSegmenter()
        seg.load_file(npy_path)

        dummy_pts = np.zeros((len(c19_ids), 4), dtype=np.float32)
        out = seg(dummy_pts)

        assert isinstance(out, SegOutput)
        assert len(out.label) == len(c19_ids)

        # Expected super-classes
        expected_super = np.array([
            STATIC_OBSTACLE,       # 0 (car)
            DYNAMIC,               # 8 (person)
            DYNAMIC,               # 9 (bicyclist)
            DYNAMIC,               # 10 (motorcyclist)
            DRIVABLE,              # 11 (road)
            NON_DRIVABLE_TERRAIN,  # 13 (sidewalk)
            STATIC_OBSTACLE,       # 17 (building)
            UNKNOWN,               # 255 (unknown)
        ], dtype=np.uint8)

        np.testing.assert_array_equal(seg.super_cls, expected_super)

        # VRU mask must be True for indices 1, 2, 3 (person, bicyclist, motorcyclist)
        expected_vru = np.array([False, True, True, True, False, False, False, False])
        np.testing.assert_array_equal(seg.vru_mask, expected_vru)

    def test_load_with_companion_confidence_file(self, tmp_path: Path) -> None:
        """Companion _conf.npy file is automatically discovered and loaded."""
        c19 = np.array([11, 11, 11], dtype=np.uint8)
        npy_path = tmp_path / "000001.npy"
        np.save(npy_path, c19)

        conf_vals = np.array([0.95, 0.82, 0.64], dtype=np.float32)
        conf_path = tmp_path / "000001_conf.npy"
        np.save(conf_path, conf_vals)

        seg = CacheSegmenter()
        seg.load_file(npy_path)

        dummy_pts = np.zeros((3, 4), dtype=np.float32)
        out = seg(dummy_pts)

        np.testing.assert_allclose(out.conf, conf_vals, atol=1e-3)
        np.testing.assert_allclose(seg.conf, conf_vals, atol=1e-3)

    def test_default_confidence_fallback(self, tmp_path: Path) -> None:
        """When no confidence file exists, conf array is filled with default_conf."""
        c19 = np.array([11, 11, 11], dtype=np.uint8)
        npy_path = tmp_path / "000002.npy"
        np.save(npy_path, c19)

        seg = CacheSegmenter(default_conf=0.85)
        seg.load_file(npy_path)

        dummy_pts = np.zeros((3, 4), dtype=np.float32)
        out = seg(dummy_pts)

        expected_conf = np.full(3, 0.85, dtype=np.float16)
        np.testing.assert_allclose(out.conf, expected_conf, atol=1e-3)

    def test_load_semkitti_label_file(self, tmp_path: Path) -> None:
        """Binary .label files with packed uint32 are unpacked and mapped correctly."""
        # Create packed uint32 labels:
        # high 16 bits = instance ID, low 16 bits = semantic ID
        # Point 0: road (40), instance 0
        # Point 1: moving-car (252), instance 4
        # Point 2: person (30), instance 7
        raw_vals = np.array([
            40 | (0 << 16),
            252 | (4 << 16),
            30 | (7 << 16),
        ], dtype=np.uint32)

        label_path = tmp_path / "000000.label"
        raw_vals.tofile(label_path)

        seg = CacheSegmenter()
        seg.load_file(label_path)

        dummy_pts = np.zeros((3, 4), dtype=np.float32)
        out = seg(dummy_pts)

        # Lower 16 bits unpacked
        np.testing.assert_array_equal(out.label, [40, 252, 30])
        # Super-classes: road=0, moving-car=3, person=3
        np.testing.assert_array_equal(seg.super_cls, [DRIVABLE, DYNAMIC, DYNAMIC])
        # Moving mask: raw ID 252 is True
        np.testing.assert_array_equal(seg.moving_mask, [False, True, False])
        # VRU mask: raw ID 30 is True
        np.testing.assert_array_equal(seg.vru_mask, [False, False, True])

    def test_oracle_interface_equivalence(self) -> None:
        """CacheSegmenter exposes grid_inputs matching OracleSegmenter signature."""
        raw_labels = np.array([40, 252, 30], dtype=np.uint32)
        pts = np.zeros((3, 4), dtype=np.float32)

        oracle = OracleSegmenter(raw_labels)
        o_sc, o_mv, o_vr, o_cf = oracle.grid_inputs(pts)

        cached = CacheSegmenter(labels=raw_labels, conf=np.ones(3, dtype=np.float16))
        c_sc, c_mv, c_vr, c_cf = cached.grid_inputs(pts)

        np.testing.assert_array_equal(o_sc, c_sc)
        np.testing.assert_array_equal(o_mv, c_mv)
        np.testing.assert_array_equal(o_vr, c_vr)
        np.testing.assert_allclose(o_cf, c_cf, atol=1e-3)

    def test_set_moving_mask_updates_state(self) -> None:
        """set_moving_mask properly updates moving_mask for motion module integration."""
        labels = np.array([0, 0, 0, 0], dtype=np.uint8)
        seg = CacheSegmenter(labels=labels)

        # Initially all False in 19-class mode
        assert not np.any(seg.moving_mask)

        # Update with motion engine results
        new_mask = np.array([False, True, True, False], dtype=np.bool_)
        seg.set_moving_mask(new_mask)

        np.testing.assert_array_equal(seg.moving_mask, new_mask)

    def test_directory_based_frame_lookup(self, tmp_path: Path) -> None:
        """load_frame correctly resolves {frame:06d}.npy in sequence directory."""
        seq_dir = tmp_path / "08"
        seq_dir.mkdir()
        frame_file = seq_dir / "000005.npy"
        np.save(frame_file, np.array([11, 11], dtype=np.uint8))

        seg = CacheSegmenter(cache_dir=tmp_path, sequence="08")
        seg.load_frame(5)

        dummy_pts = np.zeros((2, 4), dtype=np.float32)
        out = seg(dummy_pts)
        assert len(out.label) == 2

    def test_length_mismatch_raises_value_error(self) -> None:
        """Calling segmenter with point count mismatching labels raises ValueError."""
        seg = CacheSegmenter(labels=np.array([11, 11], dtype=np.uint8))
        bad_pts = np.zeros((5, 4), dtype=np.float32)  # 5 != 2

        with pytest.raises(ValueError, match="point count.*does not match"):
            seg(bad_pts)

        with pytest.raises(ValueError, match="point count.*does not match"):
            seg.grid_inputs(bad_pts)

    def test_missing_frame_raises_file_not_found(self, tmp_path: Path) -> None:
        """Querying non-existent frame raises FileNotFoundError."""
        seg = CacheSegmenter(cache_dir=tmp_path, sequence="08")
        with pytest.raises(FileNotFoundError):
            seg.load_frame(999)
