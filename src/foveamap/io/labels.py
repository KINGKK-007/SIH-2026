"""Canonical raw SemanticKITTI ID -> FoveaMap super-class mapping (README 6.1, Appendix A, task T3.3).

``raw_to_super`` is the single function used by both the oracle and the model path. It accepts full
uint32 labels (instance in the high 16 bits) or bare semantic IDs. The table below is Appendix A; it is
cross-checked against the vendored ``semantic-kitti.yaml`` (semantic-kitti-api) by the test suite.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

UNKNOWN = 0
DRIVABLE = 1
NON_DRIVABLE_TERRAIN = 2
STATIC_OBSTACLE = 3
DYNAMIC = 4

SUPER_CLASS_NAMES = ("UNKNOWN", "DRIVABLE", "NON_DRIVABLE_TERRAIN", "STATIC_OBSTACLE", "DYNAMIC")
GROUND_CLASSES = (DRIVABLE, NON_DRIVABLE_TERRAIN)
OBSTACLE_CLASSES = (STATIC_OBSTACLE, DYNAMIC)

SEMANTIC_KITTI_YAML = Path(__file__).with_name("semantic-kitti.yaml")


@dataclass(frozen=True)
class LabelInfo:
    raw_id: int
    name: str
    learning_id: int  # 19-class learning map; 0 = ignored
    super_cls: int
    moving: bool = False
    safety_critical: bool = False


_S, _D = STATIC_OBSTACLE, DYNAMIC
# README Appendix A, row by row.
LABEL_TABLE: tuple[LabelInfo, ...] = (
    LabelInfo(0, "unlabeled", 0, UNKNOWN),
    LabelInfo(1, "outlier", 0, UNKNOWN),
    LabelInfo(10, "car", 1, _S),
    LabelInfo(11, "bicycle", 2, _S),
    LabelInfo(13, "bus", 5, _S),
    LabelInfo(15, "motorcycle", 3, _S),
    LabelInfo(16, "on-rails", 5, _S),
    LabelInfo(18, "truck", 4, _S),
    LabelInfo(20, "other-vehicle", 5, _S),
    LabelInfo(30, "person", 6, _D, safety_critical=True),
    LabelInfo(31, "bicyclist", 7, _D, safety_critical=True),
    LabelInfo(32, "motorcyclist", 8, _D, safety_critical=True),
    LabelInfo(40, "road", 9, DRIVABLE),
    LabelInfo(44, "parking", 10, DRIVABLE),
    LabelInfo(48, "sidewalk", 11, NON_DRIVABLE_TERRAIN),
    LabelInfo(49, "other-ground", 12, NON_DRIVABLE_TERRAIN),
    LabelInfo(50, "building", 13, _S),
    LabelInfo(51, "fence", 14, _S),
    LabelInfo(52, "other-structure", 0, _S),
    LabelInfo(60, "lane-marking", 9, DRIVABLE),
    LabelInfo(70, "vegetation", 15, _S),
    LabelInfo(71, "trunk", 16, _S),
    LabelInfo(72, "terrain", 17, NON_DRIVABLE_TERRAIN),
    LabelInfo(80, "pole", 18, _S),
    LabelInfo(81, "traffic-sign", 19, _S),
    LabelInfo(99, "other-object", 0, _S),
    LabelInfo(252, "moving-car", 1, _D, moving=True),
    LabelInfo(253, "moving-bicyclist", 7, _D, moving=True, safety_critical=True),
    LabelInfo(254, "moving-person", 6, _D, moving=True, safety_critical=True),
    LabelInfo(255, "moving-motorcyclist", 8, _D, moving=True, safety_critical=True),
    LabelInfo(256, "moving-on-rails", 5, _D, moving=True),
    LabelInfo(257, "moving-bus", 5, _D, moving=True),
    LabelInfo(258, "moving-truck", 4, _D, moving=True),
    LabelInfo(259, "moving-other-vehicle", 5, _D, moving=True),
)
LABELS: dict[int, LabelInfo] = {info.raw_id: info for info in LABEL_TABLE}

# 19-class learning ID -> raw ID (checkpoint ``learning_map_inv``, README 6.1).
LEARNING_MAP_INV: dict[int, int] = {
    0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40,
    10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81,
}  # fmt: skip

# Raw IDs whose super-class is excluded when comparing model to oracle (README 12.2).
EXCLUDED_FROM_MODEL_COMPARISON = (0, 1, 52, 99)


def _lut(values: dict[int, int], fill: int, dtype: Any) -> np.ndarray:
    table = np.full(0x10000, fill, dtype=dtype)
    for key, value in values.items():
        table[key] = value
    table.setflags(write=False)
    return table


_KNOWN = _lut({i.raw_id: 1 for i in LABEL_TABLE}, 0, np.bool_)
_SUPER = _lut({i.raw_id: i.super_cls for i in LABEL_TABLE}, UNKNOWN, np.uint8)
_MOVING = _lut({i.raw_id: int(i.moving) for i in LABEL_TABLE}, 0, np.bool_)
_SAFETY = _lut({i.raw_id: int(i.safety_critical) for i in LABEL_TABLE}, 0, np.bool_)
_LEARNING = _lut({i.raw_id: i.learning_id for i in LABEL_TABLE}, 0, np.uint8)
_LEARNING_INV = np.array([LEARNING_MAP_INV[k] for k in range(20)], dtype=np.uint16)


def semantic_ids(raw: np.ndarray) -> np.ndarray:
    """Semantic part of full labels or bare IDs, as uint16 (``raw & 0xFFFF``)."""
    raw = np.asarray(raw)
    if raw.dtype.kind not in "ui":
        raise TypeError(f"label IDs must be integers, got dtype {raw.dtype}")
    return (raw.astype(np.int64) & 0xFFFF).astype(np.uint16)


def raw_to_super(raw_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map raw IDs to ``(super_ids uint8, moving bool)``.

    IDs not in Appendix A map to ``UNKNOWN`` (not moving) and trigger one warning per call listing them.
    """
    sem = semantic_ids(raw_ids)
    unknown = ~_KNOWN[sem]
    if unknown.any():
        ids = np.unique(sem[unknown])
        shown = ", ".join(map(str, ids[:10])) + (" ..." if ids.size > 10 else "")
        warnings.warn(
            f"{int(unknown.sum())} points with raw label IDs not in Appendix A ({shown}) mapped to UNKNOWN",
            stacklevel=2,
        )
    return _SUPER[sem], _MOVING[sem]


def is_safety_critical(raw_ids: np.ndarray) -> np.ndarray:
    """Persons, bicyclists and motorcyclists (static or moving) as a bool mask."""
    return _SAFETY[semantic_ids(raw_ids)]


def raw_to_learning(raw_ids: np.ndarray) -> np.ndarray:
    """Raw IDs -> 19-class learning IDs (uint8, 0 = ignored); moving IDs fold into static classes."""
    return _LEARNING[semantic_ids(raw_ids)]


def learning_to_raw(learning_ids: np.ndarray) -> np.ndarray:
    """19-class learning IDs -> raw IDs (uint16) via ``learning_map_inv``."""
    learning = np.asarray(learning_ids)
    if learning.size and (learning.min() < 0 or learning.max() > 19):
        raise ValueError(f"learning IDs must be in 0..19, got range {learning.min()}..{learning.max()}")
    return _LEARNING_INV[learning.astype(np.intp)]


@lru_cache(maxsize=1)
def semantic_kitti_config() -> dict[str, Any]:
    """The vendored ``semantic-kitti.yaml`` (labels, color_map (BGR), learning maps)."""
    return yaml.safe_load(SEMANTIC_KITTI_YAML.read_text(encoding="utf-8"))


def raw_color_rgb(raw_id: int) -> tuple[int, int, int]:
    """SemanticKITTI display colour for a raw ID as RGB (the yaml stores BGR); black if unknown."""
    b, g, r = semantic_kitti_config()["color_map"].get(int(raw_id), [0, 0, 0])
    return int(r), int(g), int(b)


def raw_name(raw_id: int) -> str:
    info = LABELS.get(int(raw_id))
    return info.name if info else f"unknown-{int(raw_id)}"
