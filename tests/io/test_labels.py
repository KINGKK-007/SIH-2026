"""T3.3: raw -> super-class table (README 6.1, Appendix A) and its cross-check."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from foveamap.io import labels as lbl
from foveamap.io.labels import (
    DRIVABLE,
    DYNAMIC,
    NON_DRIVABLE_TERRAIN,
    STATIC_OBSTACLE,
    UNKNOWN,
    is_safety_critical,
    learning_to_raw,
    raw_to_learning,
    raw_to_super,
)

# Appendix A restated independently of the implementation's table.
EXPECTED_SUPER = {
    0: UNKNOWN, 1: UNKNOWN,
    10: STATIC_OBSTACLE, 11: STATIC_OBSTACLE, 13: STATIC_OBSTACLE, 15: STATIC_OBSTACLE,
    16: STATIC_OBSTACLE, 18: STATIC_OBSTACLE, 20: STATIC_OBSTACLE,
    30: DYNAMIC, 31: DYNAMIC, 32: DYNAMIC,
    40: DRIVABLE, 44: DRIVABLE, 60: DRIVABLE,
    48: NON_DRIVABLE_TERRAIN, 49: NON_DRIVABLE_TERRAIN, 72: NON_DRIVABLE_TERRAIN,
    50: STATIC_OBSTACLE, 51: STATIC_OBSTACLE, 52: STATIC_OBSTACLE, 70: STATIC_OBSTACLE,
    71: STATIC_OBSTACLE, 80: STATIC_OBSTACLE, 81: STATIC_OBSTACLE, 99: STATIC_OBSTACLE,
    252: DYNAMIC, 253: DYNAMIC, 254: DYNAMIC, 255: DYNAMIC,
    256: DYNAMIC, 257: DYNAMIC, 258: DYNAMIC, 259: DYNAMIC,
}  # fmt: skip
MOVING_IDS = set(range(252, 260))


def test_every_appendix_a_id_maps_as_specified() -> None:
    ids = np.array(sorted(EXPECTED_SUPER), dtype=np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning for known IDs
        supers, moving = raw_to_super(ids)
    assert supers.dtype == np.uint8 and moving.dtype == np.bool_
    assert dict(zip(ids.tolist(), supers.tolist(), strict=True)) == EXPECTED_SUPER
    assert {int(i) for i, m in zip(ids, moving, strict=True) if m} == MOVING_IDS


def test_table_covers_exactly_appendix_a() -> None:
    assert set(lbl.LABELS) == set(EXPECTED_SUPER)


def test_moving_ids_are_dynamic_and_moving() -> None:
    supers, moving = raw_to_super(np.arange(252, 260))
    assert (supers == DYNAMIC).all() and moving.all()


def test_persons_and_cyclists_always_dynamic_not_moving() -> None:
    supers, moving = raw_to_super(np.array([30, 31, 32]))
    assert (supers == DYNAMIC).all() and not moving.any()
    assert is_safety_critical(np.array([30, 31, 32, 253, 254, 255])).all()
    assert not is_safety_critical(np.array([10, 252, 40, 80])).any()


def test_parked_vehicles_static_until_promoted() -> None:
    supers, moving = raw_to_super(np.array([10, 11, 13, 15, 16, 18, 20]))
    assert (supers == STATIC_OBSTACLE).all() and not moving.any()


def test_full_uint32_labels_use_semantic_bits() -> None:
    raw = np.array([(12 << 16) | 252, (3 << 16) | 40, (0xFFFF << 16) | 30], dtype=np.uint32)
    supers, moving = raw_to_super(raw)
    np.testing.assert_array_equal(supers, [DYNAMIC, DRIVABLE, DYNAMIC])
    np.testing.assert_array_equal(moving, [True, False, False])


def test_unknown_ids_map_to_unknown_with_a_single_warning() -> None:
    raw = np.array([40] * 5 + [7] * 1000 + [300] * 1000 + [65535], dtype=np.uint32)
    with pytest.warns(UserWarning) as record:
        supers, moving = raw_to_super(raw)
    assert len(record) == 1
    message = str(record[0].message)
    assert "2001 points" in message and "7" in message and "300" in message
    assert (supers[5:] == UNKNOWN).all() and not moving.any()
    assert (supers[:5] == DRIVABLE).all()


def test_rejects_non_integer_input() -> None:
    with pytest.raises(TypeError):
        raw_to_super(np.array([40.0]))


def test_learning_maps_round_trip() -> None:
    learning = np.arange(20)
    raw = learning_to_raw(learning)
    assert raw.dtype == np.uint16
    # README 6.1 list
    assert raw.tolist() == [0, 10, 11, 15, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 70, 71, 72, 80, 81]
    np.testing.assert_array_equal(raw_to_learning(raw), learning)


def test_moving_ids_fold_into_static_learning_classes() -> None:
    np.testing.assert_array_equal(
        raw_to_learning(np.array([252, 253, 254, 255, 256, 257, 258, 259, 52, 99, 60])),
        [1, 7, 6, 8, 5, 5, 4, 5, 0, 0, 9],
    )


def test_learning_to_raw_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        learning_to_raw(np.array([20]))


# ── cross-check against semantic-kitti-api (T3.3) ──────────────────────────


def test_cross_check_names_against_semantic_kitti_yaml() -> None:
    cfg = lbl.semantic_kitti_config()
    assert set(cfg["labels"]) == set(lbl.LABELS)
    for raw_id, name in cfg["labels"].items():
        assert lbl.LABELS[raw_id].name == name


def test_cross_check_learning_map_against_semantic_kitti_yaml() -> None:
    cfg = lbl.semantic_kitti_config()
    for raw_id, learning_id in cfg["learning_map"].items():
        assert lbl.LABELS[raw_id].learning_id == learning_id, raw_id


def test_cross_check_learning_map_inv_against_semantic_kitti_yaml() -> None:
    assert lbl.semantic_kitti_config()["learning_map_inv"] == lbl.LEARNING_MAP_INV


def test_raw_colors_are_rgb() -> None:
    assert lbl.raw_color_rgb(40) == (255, 0, 255)  # road: BGR [255, 0, 255]
    assert lbl.raw_color_rgb(10) == (100, 150, 245)  # car: BGR [245, 150, 100]
    assert lbl.raw_color_rgb(12345) == (0, 0, 0)
