"""tests/test_labels.py — Verification suite for foveamap.io.labels.

QA requirements addressed
--------------------------
R1  Bit-unpacking: upper-16 instance ID and lower-16 semantic ID extracted
    correctly for all bit-pattern combinations.
R2  Super-class coverage: every one of the 19 official SemanticKITTI benchmark
    classes (raw IDs in the master plan §5.1 table) maps deterministically to
    exactly one of {DRIVABLE, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, DYNAMIC}.
    None may map to UNKNOWN.
R3  Moving variants (raw IDs 252–259) all map to DYNAMIC — zero exceptions.
R4  LUT consistency: to_superclass result equals the explicit _RAW_TO_SUPER dict.
R5  19-class pathway: cls19_to_superclass covers all 22 defined 19-class IDs.
R6  No false positives: every assertion is grounded in a concrete expected value,
    not just "not UNKNOWN".
"""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.io.labels import (
    DRIVABLE,
    DYNAMIC,
    NON_DRIVABLE_TERRAIN,
    STATIC_OBSTACLE,
    UNKNOWN,
    _CLS19_TO_SUPER,
    _RAW_TO_SUPER,
    cls19_to_superclass,
    superclass_histogram,
    to_19class,
    to_superclass,
    unpack_kitti_labels,
)


# ─────────────────────────────────────────────────────────────────────────────
# R1 — Bit-unpacking correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestUnpackKittiLabels:
    """Verify that semantic (bits 0-15) and instance (bits 16-31) are split correctly."""

    def test_zero_label(self) -> None:
        """All-zero label must produce semantic=0, instance=0."""
        raw = np.array([0x00000000], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 0
        assert int(inst[0]) == 0

    def test_semantic_only_road(self) -> None:
        """Semantic=40 (0x28), instance=0."""
        raw = np.array([0x00000028], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 40
        assert int(inst[0]) == 0

    def test_instance_only_nonzero(self) -> None:
        """Instance=7, semantic=0 — upper bits nonzero, lower bits zero."""
        raw = np.array([0x00070000], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 0, "lower 16 bits must be 0"
        assert int(inst[0]) == 7, "upper 16 bits must be 7"

    def test_both_fields_set(self) -> None:
        """Semantic=10 (car=0x0A), instance=3 (0x0003)."""
        raw = np.array([0x0003000A], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 10
        assert int(inst[0]) == 3

    def test_max_semantic_id(self) -> None:
        """Semantic=0xFFFF (65535), instance=0."""
        raw = np.array([0x0000FFFF], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 65535
        assert int(inst[0]) == 0

    def test_max_instance_id(self) -> None:
        """Instance=0xFFFF (65535), semantic=0."""
        raw = np.array([0xFFFF0000], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 0
        assert int(inst[0]) == 65535

    def test_all_ones(self) -> None:
        """0xFFFFFFFF: both fields at maximum."""
        raw = np.array([0xFFFFFFFF], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 65535
        assert int(inst[0]) == 65535

    def test_moving_car_semantic_252(self) -> None:
        """Moving-car = raw semantic ID 252 = 0x00FC."""
        raw = np.array([0x000000FC], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 252
        assert int(inst[0]) == 0

    def test_moving_car_with_instance(self) -> None:
        """Moving-car (252) with instance 15."""
        raw = np.array([0x000F00FC], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert int(sem[0]) == 252
        assert int(inst[0]) == 15

    def test_output_dtypes_uint16(self) -> None:
        raw = np.array([40, 10], dtype=np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert sem.dtype == np.uint16, f"Expected uint16, got {sem.dtype}"
        assert inst.dtype == np.uint16, f"Expected uint16, got {inst.dtype}"

    def test_batch_length_preserved(self) -> None:
        N = 5000
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32)
        sem, inst = unpack_kitti_labels(raw)
        assert sem.shape == (N,)
        assert inst.shape == (N,)

    def test_round_trip_all_fields(self) -> None:
        """Pack semantic + instance into uint32 and unpack; must recover originals."""
        sem_orig  = np.array([40, 10, 252, 30, 50, 0], dtype=np.uint32)
        inst_orig = np.array([0,   5,   0,  2,  7, 65535], dtype=np.uint32)
        packed = (inst_orig << 16) | sem_orig
        sem_out, inst_out = unpack_kitti_labels(packed)
        np.testing.assert_array_equal(sem_out,  sem_orig.astype(np.uint16))
        np.testing.assert_array_equal(inst_out, inst_orig.astype(np.uint16))

    def test_instance_does_not_bleed_into_semantic(self) -> None:
        """A nonzero instance ID must not corrupt the semantic field."""
        for inst_id in [1, 255, 1000, 65535]:
            for sem_id in [0, 10, 40, 252]:
                packed = np.array([(inst_id << 16) | sem_id], dtype=np.uint32)
                sem_out, inst_out = unpack_kitti_labels(packed)
                assert int(sem_out[0]) == sem_id, (
                    f"Semantic bleed: inst={inst_id:#06x} sem={sem_id:#06x} "
                    f"→ sem_out={int(sem_out[0])}"
                )
                assert int(inst_out[0]) == inst_id, (
                    f"Instance wrong: inst={inst_id:#06x} sem={sem_id:#06x} "
                    f"→ inst_out={int(inst_out[0])}"
                )

    def test_semantic_does_not_bleed_into_instance(self) -> None:
        """A high semantic ID must not corrupt the instance field."""
        packed = np.array([0x0000FFFF], dtype=np.uint32)  # sem=65535, inst=0
        sem_out, inst_out = unpack_kitti_labels(packed)
        assert int(inst_out[0]) == 0, (
            f"High semantic bled into instance: inst_out={int(inst_out[0])}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# R2 — Full 19-class super-class coverage (exhaustive)
# ─────────────────────────────────────────────────────────────────────────────

# Official SemanticKITTI benchmark classes (master plan §5.1 / semantic-kitti-api)
# Raw-ID → expected super-class, covering every class in the master plan table.
_GROUND_TRUTH_SUPERCLASS: dict[int, int] = {
    # ignore
    0:   UNKNOWN,           # unlabeled
    1:   UNKNOWN,           # outlier
    # vehicles (not moving)
    10:  STATIC_OBSTACLE,   # car
    11:  STATIC_OBSTACLE,   # bicycle
    13:  STATIC_OBSTACLE,   # bus
    15:  STATIC_OBSTACLE,   # motorcycle
    16:  STATIC_OBSTACLE,   # on-rails
    18:  STATIC_OBSTACLE,   # truck
    20:  STATIC_OBSTACLE,   # other-vehicle
    # VRUs (always DYNAMIC per H6)
    30:  DYNAMIC,           # person
    31:  DYNAMIC,           # bicyclist
    32:  DYNAMIC,           # motorcyclist
    # ground classes
    40:  DRIVABLE,                  # road
    44:  DRIVABLE,                  # parking
    48:  NON_DRIVABLE_TERRAIN,      # sidewalk
    49:  NON_DRIVABLE_TERRAIN,      # other-ground
    60:  DRIVABLE,                  # lane-marking
    72:  NON_DRIVABLE_TERRAIN,      # terrain
    # structures & objects
    50:  STATIC_OBSTACLE,   # building
    51:  STATIC_OBSTACLE,   # fence
    52:  STATIC_OBSTACLE,   # other-structure
    70:  STATIC_OBSTACLE,   # vegetation
    71:  STATIC_OBSTACLE,   # trunk
    80:  STATIC_OBSTACLE,   # pole
    81:  STATIC_OBSTACLE,   # traffic-sign
    99:  STATIC_OBSTACLE,   # other-object
    # moving variants (252-259 → always DYNAMIC)
    252: DYNAMIC,           # moving-car
    253: DYNAMIC,           # moving-bicyclist
    254: DYNAMIC,           # moving-person
    255: DYNAMIC,           # moving-motorcyclist
    256: DYNAMIC,           # moving-on-rails
    257: DYNAMIC,           # moving-bus
    258: DYNAMIC,           # moving-truck
    259: DYNAMIC,           # moving-other-vehicle
}

# Named super-classes for error messages
_SC_NAMES = {
    DRIVABLE: "DRIVABLE",
    NON_DRIVABLE_TERRAIN: "NON_DRIVABLE_TERRAIN",
    STATIC_OBSTACLE: "STATIC_OBSTACLE",
    DYNAMIC: "DYNAMIC",
    UNKNOWN: "UNKNOWN",
}


class TestToSuperclassExhaustive:
    """Exhaustive parametrized tests — one test case per raw semantic ID."""

    @pytest.mark.parametrize("raw_id,expected_sc", list(_GROUND_TRUTH_SUPERCLASS.items()))
    def test_ground_truth_mapping(self, raw_id: int, expected_sc: int) -> None:
        """Every raw ID must map to exactly the documented super-class."""
        ids = np.array([raw_id], dtype=np.uint16)
        result = int(to_superclass(ids)[0])
        assert result == expected_sc, (
            f"raw_id={raw_id}: expected {_SC_NAMES[expected_sc]}={expected_sc}, "
            f"got {_SC_NAMES.get(result, '?')}={result}"
        )

    def test_lut_matches_dict_for_all_known_ids(self) -> None:
        """LUT values must exactly match the _RAW_TO_SUPER dict for every known ID."""
        for raw_id, expected_sc in _RAW_TO_SUPER.items():
            ids = np.array([raw_id], dtype=np.uint16)
            lut_val = int(to_superclass(ids)[0])
            assert lut_val == expected_sc, (
                f"LUT mismatch at raw_id={raw_id}: dict says {expected_sc}, LUT says {lut_val}"
            )

    def test_no_known_id_maps_to_unknown_except_ignore(self) -> None:
        """IDs 10–259 in the master plan must NEVER map to UNKNOWN.
        Only IDs 0 and 1 (unlabeled, outlier) are permitted to be UNKNOWN.
        """
        valid_ids_that_must_not_be_unknown = [
            10, 11, 13, 15, 16, 18, 20,   # vehicles (static)
            30, 31, 32,                    # VRUs
            40, 44, 48, 49, 60, 72,        # ground
            50, 51, 52, 70, 71, 80, 81, 99,# structures
            252, 253, 254, 255, 256, 257, 258, 259,  # moving
        ]
        ids = np.array(valid_ids_that_must_not_be_unknown, dtype=np.uint16)
        result = to_superclass(ids)
        unknowns = [
            (raw_id, int(sc))
            for raw_id, sc in zip(valid_ids_that_must_not_be_unknown, result)
            if sc == UNKNOWN
        ]
        assert unknowns == [], (
            f"These raw IDs unexpectedly mapped to UNKNOWN: {unknowns}"
        )

    def test_all_19_benchmark_classes_covered(self) -> None:
        """Every one of the 19 SemanticKITTI benchmark classes must have
        a defined super-class in _CLS19_TO_SUPER (no UNKNOWN for valid classes).
        """
        # The 19 benchmark class IDs are 0..18 plus trunk (19), pole (20),
        # traffic-sign (21) in extended configs. We verify all IDs 0–21.
        valid_cls19 = list(range(22))
        ids = np.array(valid_cls19, dtype=np.uint8)
        result = cls19_to_superclass(ids)
        unknowns = [
            (c, int(sc)) for c, sc in zip(valid_cls19, result) if sc == UNKNOWN
        ]
        assert unknowns == [], (
            f"19-class IDs unexpectedly map to UNKNOWN: {unknowns}"
        )

    def test_output_is_uint8(self) -> None:
        ids = np.array(list(_GROUND_TRUTH_SUPERCLASS.keys()), dtype=np.uint16)
        result = to_superclass(ids)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_vectorised_batch_matches_individual(self) -> None:
        """Batch lookup must match single-element lookups element-wise."""
        raw_ids = np.array(list(_GROUND_TRUTH_SUPERCLASS.keys()), dtype=np.uint16)
        batch_result = to_superclass(raw_ids)
        for i, raw_id in enumerate(raw_ids):
            single = int(to_superclass(np.array([raw_id], dtype=np.uint16))[0])
            assert int(batch_result[i]) == single, (
                f"Batch/single mismatch at index {i}, raw_id={raw_id}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# R3 — Moving variants 252–259 all → DYNAMIC (explicit, zero exceptions)
# ─────────────────────────────────────────────────────────────────────────────

class TestMovingVariantsDynamic:
    """R3: All 8 moving-* raw IDs must resolve to DYNAMIC. No exceptions allowed."""

    MOVING_IDS = [252, 253, 254, 255, 256, 257, 258, 259]
    MOVING_NAMES = {
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    @pytest.mark.parametrize("raw_id", MOVING_IDS)
    def test_individual_moving_id_is_dynamic(self, raw_id: int) -> None:
        ids = np.array([raw_id], dtype=np.uint16)
        result = int(to_superclass(ids)[0])
        assert result == DYNAMIC, (
            f"{self.MOVING_NAMES[raw_id]} (raw_id={raw_id}) mapped to "
            f"{_SC_NAMES.get(result, '?')}={result}, expected DYNAMIC={DYNAMIC}"
        )

    def test_all_moving_ids_batch_are_dynamic(self) -> None:
        """Batch call: all 8 moving IDs at once must all be DYNAMIC."""
        ids = np.array(self.MOVING_IDS, dtype=np.uint16)
        result = to_superclass(ids)
        not_dynamic = [
            (self.MOVING_NAMES[raw_id], int(sc))
            for raw_id, sc in zip(self.MOVING_IDS, result)
            if sc != DYNAMIC
        ]
        assert not_dynamic == [], (
            f"These moving IDs did not map to DYNAMIC: {not_dynamic}"
        )

    def test_moving_ids_not_static_obstacle(self) -> None:
        """Moving variants must NOT be STATIC_OBSTACLE."""
        ids = np.array(self.MOVING_IDS, dtype=np.uint16)
        result = to_superclass(ids)
        wrong = [
            (self.MOVING_NAMES[raw_id], int(sc))
            for raw_id, sc in zip(self.MOVING_IDS, result)
            if sc == STATIC_OBSTACLE
        ]
        assert wrong == [], f"Moving IDs wrongly classified as STATIC_OBSTACLE: {wrong}"

    def test_moving_ids_not_drivable(self) -> None:
        """Moving variants must NOT be DRIVABLE."""
        ids = np.array(self.MOVING_IDS, dtype=np.uint16)
        result = to_superclass(ids)
        wrong = [
            (self.MOVING_NAMES[raw_id], int(sc))
            for raw_id, sc in zip(self.MOVING_IDS, result)
            if sc == DRIVABLE
        ]
        assert wrong == [], f"Moving IDs wrongly classified as DRIVABLE: {wrong}"

    def test_moving_ids_not_unknown(self) -> None:
        """Moving variants must NOT be UNKNOWN."""
        ids = np.array(self.MOVING_IDS, dtype=np.uint16)
        result = to_superclass(ids)
        wrong = [
            (self.MOVING_NAMES[raw_id], int(sc))
            for raw_id, sc in zip(self.MOVING_IDS, result)
            if sc == UNKNOWN
        ]
        assert wrong == [], f"Moving IDs wrongly classified as UNKNOWN: {wrong}"

    def test_moving_vs_static_vehicle_differ(self) -> None:
        """car (10) → STATIC_OBSTACLE; moving-car (252) → DYNAMIC. Must differ."""
        static = int(to_superclass(np.array([10], dtype=np.uint16))[0])
        dynamic = int(to_superclass(np.array([252], dtype=np.uint16))[0])
        assert static == STATIC_OBSTACLE
        assert dynamic == DYNAMIC
        assert static != dynamic, "Parked car and moving car must map to different super-classes"


# ─────────────────────────────────────────────────────────────────────────────
# R4 — 19-class pathway (cls19_to_superclass)
# ─────────────────────────────────────────────────────────────────────────────

_CLS19_GROUND_TRUTH: dict[int, int] = {
    0:  STATIC_OBSTACLE,      # car
    1:  STATIC_OBSTACLE,      # bicycle
    2:  STATIC_OBSTACLE,      # vegetation
    3:  STATIC_OBSTACLE,      # motorcycle
    4:  STATIC_OBSTACLE,      # bus
    5:  STATIC_OBSTACLE,      # on-rails
    6:  STATIC_OBSTACLE,      # truck
    7:  STATIC_OBSTACLE,      # other-vehicle
    8:  DYNAMIC,              # person
    9:  DYNAMIC,              # bicyclist
    10: DYNAMIC,              # motorcyclist
    11: DRIVABLE,             # road
    12: DRIVABLE,             # parking
    13: NON_DRIVABLE_TERRAIN, # sidewalk
    14: NON_DRIVABLE_TERRAIN, # other-ground
    15: DRIVABLE,             # lane-marking
    16: NON_DRIVABLE_TERRAIN, # terrain
    17: STATIC_OBSTACLE,      # building
    18: STATIC_OBSTACLE,      # fence
    19: STATIC_OBSTACLE,      # trunk
    20: STATIC_OBSTACLE,      # pole
    21: STATIC_OBSTACLE,      # traffic-sign
}


class TestCls19ToSuperclassExhaustive:
    @pytest.mark.parametrize("cls19,expected_sc", list(_CLS19_GROUND_TRUTH.items()))
    def test_cls19_mapping(self, cls19: int, expected_sc: int) -> None:
        ids = np.array([cls19], dtype=np.uint8)
        result = int(cls19_to_superclass(ids)[0])
        assert result == expected_sc, (
            f"cls19={cls19}: expected {_SC_NAMES[expected_sc]}, "
            f"got {_SC_NAMES.get(result, '?')}={result}"
        )

    def test_cls19_lut_matches_dict(self) -> None:
        """LUT values match the _CLS19_TO_SUPER dict for all defined 19-class IDs."""
        for cls19, expected_sc in _CLS19_TO_SUPER.items():
            ids = np.array([cls19], dtype=np.uint8)
            result = int(cls19_to_superclass(ids)[0])
            assert result == expected_sc, (
                f"LUT mismatch at cls19={cls19}: dict={expected_sc}, lut={result}"
            )

    def test_unknown_cls19_255_is_unknown(self) -> None:
        ids = np.array([255], dtype=np.uint8)
        assert int(cls19_to_superclass(ids)[0]) == UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# R5 — to_19class spot checks
# ─────────────────────────────────────────────────────────────────────────────

class TestTo19Class:
    @pytest.mark.parametrize("raw_id,expected_cls19", [
        (10, 0),   # car
        (11, 1),   # bicycle
        (30, 8),   # person
        (31, 9),   # bicyclist
        (32, 10),  # motorcyclist
        (40, 11),  # road
        (44, 12),  # parking
        (48, 13),  # sidewalk
        (49, 14),  # other-ground
        (60, 15),  # lane-marking
        (72, 16),  # terrain
        (50, 17),  # building
        (51, 18),  # fence
        (70, 2),   # vegetation
        (80, 20),  # pole
        (81, 21),  # traffic-sign
        # moving variants merge into their static 19-class counterpart
        (252, 0),  # moving-car → car (class 0)
        (253, 9),  # moving-bicyclist → bicyclist (class 9)
        (254, 8),  # moving-person → person (class 8)
        (255, 10), # moving-motorcyclist
        (256, 5),  # moving-on-rails
        (257, 4),  # moving-bus
        (258, 6),  # moving-truck
        (259, 7),  # moving-other-vehicle
    ])
    def test_raw_to_19class(self, raw_id: int, expected_cls19: int) -> None:
        ids = np.array([raw_id], dtype=np.uint16)
        result = int(to_19class(ids)[0])
        assert result == expected_cls19, (
            f"raw_id={raw_id}: expected cls19={expected_cls19}, got {result}"
        )

    def test_ignore_ids_return_255(self) -> None:
        for raw_id in [0, 1]:
            ids = np.array([raw_id], dtype=np.uint16)
            assert int(to_19class(ids)[0]) == 255, f"raw_id={raw_id} should be 255"

    def test_output_dtype_uint8(self) -> None:
        ids = np.array([40], dtype=np.uint16)
        assert to_19class(ids).dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# R6 — superclass_histogram
# ─────────────────────────────────────────────────────────────────────────────

class TestSuperclassHistogram:
    def test_counts_sum_to_N(self) -> None:
        ids = np.array([0, 1, 2, 3, 255, 0, 0], dtype=np.uint8)
        hist = superclass_histogram(ids)
        assert sum(hist.values()) == len(ids)

    def test_exact_counts(self) -> None:
        ids = np.array([
            DRIVABLE, DRIVABLE, DRIVABLE,
            NON_DRIVABLE_TERRAIN, NON_DRIVABLE_TERRAIN,
            STATIC_OBSTACLE,
            DYNAMIC,
            UNKNOWN,
        ], dtype=np.uint8)
        hist = superclass_histogram(ids)
        assert hist["DRIVABLE"] == 3
        assert hist["NON_DRIVABLE_TERRAIN"] == 2
        assert hist["STATIC_OBSTACLE"] == 1
        assert hist["DYNAMIC"] == 1
        assert hist["UNKNOWN"] == 1

    def test_keys_are_exactly_five(self) -> None:
        ids = np.zeros(10, dtype=np.uint8)
        hist = superclass_histogram(ids)
        expected_keys = {"DRIVABLE", "NON_DRIVABLE_TERRAIN", "STATIC_OBSTACLE", "DYNAMIC", "UNKNOWN"}
        assert set(hist.keys()) == expected_keys

    def test_empty_input(self) -> None:
        ids = np.array([], dtype=np.uint8)
        hist = superclass_histogram(ids)
        assert sum(hist.values()) == 0

    def test_all_unknown(self) -> None:
        ids = np.full(50, UNKNOWN, dtype=np.uint8)
        hist = superclass_histogram(ids)
        assert hist["UNKNOWN"] == 50
        assert hist["DRIVABLE"] == 0

    def test_values_are_python_ints(self) -> None:
        """Histogram values must be plain Python ints, not numpy scalars."""
        ids = np.array([DRIVABLE, DYNAMIC], dtype=np.uint8)
        hist = superclass_histogram(ids)
        for v in hist.values():
            assert isinstance(v, int), f"Expected int, got {type(v)}"
