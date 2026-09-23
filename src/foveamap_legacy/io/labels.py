"""foveamap.io.labels — SemanticKITTI label taxonomy and super-class mappings.

Label file format
-----------------
Each label is a ``uint32`` value where:
  - Bits 0–15 (low word)  : semantic class ID
  - Bits 16–31 (high word): instance ID

Raw → 19-class mapping
-----------------------
The official SemanticKITTI benchmark defines 19 classes by merging the full
label set.  We follow the mapping published in the ``semantic-kitti-api``
repository (``config/semantic-kitti-all.yaml``).

Super-class mapping (PS-aligned)
---------------------------------
ID  Name                    Colour
 0  DRIVABLE                teal   #1FB6A6
 1  NON_DRIVABLE_TERRAIN    slate  #5B6B7A
 2  STATIC_OBSTACLE         amber  #F5A524
 3  DYNAMIC                 red    #FF4D6D
255 UNKNOWN                 dark   #111111

Decision H6 (§10): standing pedestrians are always DYNAMIC (safety-critical),
parking/lane-marking are DRIVABLE, vegetation is STATIC_OBSTACLE.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Super-class constants
# ─────────────────────────────────────────────────────────────────────────────

DRIVABLE: int = 0
NON_DRIVABLE_TERRAIN: int = 1
STATIC_OBSTACLE: int = 2
DYNAMIC: int = 3
UNKNOWN: int = 255

SUPERCLASS_NAMES: dict[int, str] = {
    DRIVABLE: "DRIVABLE",
    NON_DRIVABLE_TERRAIN: "NON_DRIVABLE_TERRAIN",
    STATIC_OBSTACLE: "STATIC_OBSTACLE",
    DYNAMIC: "DYNAMIC",
    UNKNOWN: "UNKNOWN",
}

SUPERCLASS_COLORS_HEX: dict[int, str] = {
    DRIVABLE: "#1FB6A6",
    NON_DRIVABLE_TERRAIN: "#5B6B7A",
    STATIC_OBSTACLE: "#F5A524",
    DYNAMIC: "#FF4D6D",
    UNKNOWN: "#111111",
}

# ─────────────────────────────────────────────────────────────────────────────
# Raw SemanticKITTI IDs → 19-class benchmark IDs
# Source: semantic-kitti-api config/semantic-kitti-all.yaml  [VERIFY]
# ─────────────────────────────────────────────────────────────────────────────

# Maps raw semantic ID to 19-class benchmark ID (0-indexed).
# Raw IDs not listed here map to 255 (UNKNOWN in 19-class space).
_RAW_TO_19: dict[int, int] = {
    # ignore / outlier → 255
    0: 255,   # unlabeled
    1: 255,   # outlier
    # vehicles (static labels in 19-class mapping)
    10: 0,    # car
    11: 1,    # bicycle
    13: 4,    # bus
    15: 3,    # motorcycle
    16: 5,    # on-rails
    18: 6,    # truck
    20: 7,    # other-vehicle
    # vulnerable road users
    30: 8,    # person
    31: 9,    # bicyclist
    32: 10,   # motorcyclist
    # ground
    40: 11,   # road
    44: 12,   # parking
    48: 13,   # sidewalk
    49: 14,   # other-ground
    60: 15,   # lane-marking
    72: 16,   # terrain
    # structures & objects
    50: 17,   # building
    51: 18,   # fence
    52: 255,  # other-structure → unmapped (treat as unknown)
    70: 2,    # vegetation
    71: 19,   # trunk  (class 19 when used separately; merged with vegetation in some configs)
    80: 20,   # pole
    81: 21,   # traffic-sign
    99: 255,  # other-object → unknown
    # moving variants — merge with their static counterparts in 19-class
    252: 0,   # moving-car     → car
    253: 9,   # moving-bicyclist → bicyclist
    254: 8,   # moving-person  → person
    255: 10,  # moving-motorcyclist → motorcyclist
    256: 5,   # moving-on-rails
    257: 4,   # moving-bus
    258: 6,   # moving-truck
    259: 7,   # moving-other-vehicle
}

# Build lookup table (0..65535) for O(1) vectorised lookup
_RAW_TO_19_LUT: NDArray[np.uint8] = np.full(65536, 255, dtype=np.uint8)
for _raw, _cls in _RAW_TO_19.items():
    _RAW_TO_19_LUT[_raw] = _cls

# ─────────────────────────────────────────────────────────────────────────────
# Raw SemanticKITTI IDs → Super-class
# ─────────────────────────────────────────────────────────────────────────────

# Mapping defined per master plan §5.1, decision H6 applied as default.
# Moving variants of vehicles are DYNAMIC regardless of 19-class merging.
_RAW_TO_SUPER: dict[int, int] = {
    # ── Ignore / outlier ──────────────────────────────────────────────────
    0:   UNKNOWN,
    1:   UNKNOWN,
    # ── DRIVABLE ──────────────────────────────────────────────────────────
    40:  DRIVABLE,   # road
    44:  DRIVABLE,   # parking
    60:  DRIVABLE,   # lane-marking
    # ── NON_DRIVABLE_TERRAIN ──────────────────────────────────────────────
    48:  NON_DRIVABLE_TERRAIN,   # sidewalk
    49:  NON_DRIVABLE_TERRAIN,   # other-ground
    72:  NON_DRIVABLE_TERRAIN,   # terrain
    # ── STATIC_OBSTACLE ───────────────────────────────────────────────────
    10:  STATIC_OBSTACLE,   # car (not moving)
    11:  STATIC_OBSTACLE,   # bicycle (not moving)
    13:  STATIC_OBSTACLE,   # bus (not moving)
    15:  STATIC_OBSTACLE,   # motorcycle (not moving)
    16:  STATIC_OBSTACLE,   # on-rails (not moving)
    18:  STATIC_OBSTACLE,   # truck (not moving)
    20:  STATIC_OBSTACLE,   # other-vehicle (not moving)
    50:  STATIC_OBSTACLE,   # building
    51:  STATIC_OBSTACLE,   # fence
    52:  STATIC_OBSTACLE,   # other-structure
    70:  STATIC_OBSTACLE,   # vegetation
    71:  STATIC_OBSTACLE,   # trunk
    80:  STATIC_OBSTACLE,   # pole
    81:  STATIC_OBSTACLE,   # traffic-sign
    99:  STATIC_OBSTACLE,   # other-object
    # ── DYNAMIC ───────────────────────────────────────────────────────────
    # Persons / VRUs are always DYNAMIC (standing pedestrian still safety-critical)
    30:  DYNAMIC,   # person
    31:  DYNAMIC,   # bicyclist
    32:  DYNAMIC,   # motorcyclist
    # Moving vehicle variants
    252: DYNAMIC,   # moving-car
    253: DYNAMIC,   # moving-bicyclist
    254: DYNAMIC,   # moving-person
    255: DYNAMIC,   # moving-motorcyclist
    256: DYNAMIC,   # moving-on-rails
    257: DYNAMIC,   # moving-bus
    258: DYNAMIC,   # moving-truck
    259: DYNAMIC,   # moving-other-vehicle
}

# Build lookup table (0..65535)
_RAW_TO_SUPER_LUT: NDArray[np.uint8] = np.full(65536, UNKNOWN, dtype=np.uint8)
for _raw, _sc in _RAW_TO_SUPER.items():
    _RAW_TO_SUPER_LUT[_raw] = _sc

# ─────────────────────────────────────────────────────────────────────────────
# 19-class ID → Super-class  (for use when working with cached 19-class labels)
# ─────────────────────────────────────────────────────────────────────────────

_CLS19_TO_SUPER: dict[int, int] = {
    0:   STATIC_OBSTACLE,       # car
    1:   STATIC_OBSTACLE,       # bicycle
    2:   STATIC_OBSTACLE,       # vegetation
    3:   STATIC_OBSTACLE,       # motorcycle
    4:   STATIC_OBSTACLE,       # bus
    5:   STATIC_OBSTACLE,       # on-rails
    6:   STATIC_OBSTACLE,       # truck
    7:   STATIC_OBSTACLE,       # other-vehicle
    8:   DYNAMIC,               # person (always)
    9:   DYNAMIC,               # bicyclist (always)
    10:  DYNAMIC,               # motorcyclist (always)
    11:  DRIVABLE,              # road
    12:  DRIVABLE,              # parking
    13:  NON_DRIVABLE_TERRAIN,  # sidewalk
    14:  NON_DRIVABLE_TERRAIN,  # other-ground
    15:  DRIVABLE,              # lane-marking
    16:  NON_DRIVABLE_TERRAIN,  # terrain
    17:  STATIC_OBSTACLE,       # building
    18:  STATIC_OBSTACLE,       # fence
    19:  STATIC_OBSTACLE,       # trunk
    20:  STATIC_OBSTACLE,       # pole
    21:  STATIC_OBSTACLE,       # traffic-sign
    255: UNKNOWN,
}

_CLS19_TO_SUPER_LUT: NDArray[np.uint8] = np.full(256, UNKNOWN, dtype=np.uint8)
for _c19, _sc in _CLS19_TO_SUPER.items():
    _CLS19_TO_SUPER_LUT[_c19] = _sc


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def unpack_kitti_labels(raw_labels: NDArray[np.uint32]) -> tuple[NDArray[np.uint16], NDArray[np.uint16]]:
    """Split raw uint32 SemanticKITTI labels into semantic and instance IDs.

    Parameters
    ----------
    raw_labels:
        Array of shape ``(N,)`` with ``uint32`` values as stored in ``.label``
        files.  Bit layout: ``[instance_id(16) | semantic_id(16)]``.

    Returns
    -------
    semantic_ids : NDArray[uint16]
        Lower 16-bit semantic class IDs, shape ``(N,)``.
    instance_ids : NDArray[uint16]
        Upper 16-bit instance IDs, shape ``(N,)``.

    Examples
    --------
    >>> import numpy as np
    >>> raw = np.array([0x000A0028], dtype=np.uint32)  # semantic=40 (road), instance=10
    >>> sem, inst = unpack_kitti_labels(raw)
    >>> int(sem[0]), int(inst[0])
    (40, 10)
    """
    raw_labels = np.asarray(raw_labels, dtype=np.uint32)
    semantic_ids = (raw_labels & 0xFFFF).astype(np.uint16)
    instance_ids = (raw_labels >> 16).astype(np.uint16)
    return semantic_ids, instance_ids


def to_19class(semantic_ids: NDArray[np.uint16]) -> NDArray[np.uint8]:
    """Map raw SemanticKITTI semantic IDs to the 19-class benchmark label set.

    Parameters
    ----------
    semantic_ids:
        Array of shape ``(N,)`` with ``uint16`` raw semantic IDs (lower 16
        bits of the label file).

    Returns
    -------
    cls19 : NDArray[uint8]
        19-class IDs, shape ``(N,)``.  Unlabelled / unknown points → 255.
    """
    ids = np.asarray(semantic_ids, dtype=np.uint16)
    return _RAW_TO_19_LUT[ids]


def to_superclass(semantic_ids: NDArray[np.uint16]) -> NDArray[np.uint8]:
    """Map raw SemanticKITTI semantic IDs to the 4 PS super-classes.

    Uses the default mapping from master-plan §5.1 (decision H6).

    Parameters
    ----------
    semantic_ids:
        Array of shape ``(N,)`` with ``uint16`` raw semantic IDs.

    Returns
    -------
    super_ids : NDArray[uint8]
        Super-class IDs, shape ``(N,)``.
        Values: 0=DRIVABLE, 1=NON_DRIVABLE_TERRAIN, 2=STATIC_OBSTACLE,
        3=DYNAMIC, 255=UNKNOWN.
    """
    ids = np.asarray(semantic_ids, dtype=np.uint16)
    return _RAW_TO_SUPER_LUT[ids]


def cls19_to_superclass(cls19_ids: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Map 19-class benchmark IDs to super-class IDs.

    Useful when working with cached predictions from a standard pretrained
    segmentation network that outputs 19-class labels.

    Parameters
    ----------
    cls19_ids:
        Array of shape ``(N,)`` with 19-class IDs (uint8).

    Returns
    -------
    super_ids : NDArray[uint8]
        Super-class IDs, shape ``(N,)``.
    """
    ids = np.asarray(cls19_ids, dtype=np.uint8)
    return _CLS19_TO_SUPER_LUT[ids]


def superclass_histogram(super_ids: NDArray[np.uint8]) -> dict[str, int]:
    """Return a histogram of super-class label counts.

    Parameters
    ----------
    super_ids:
        Array of super-class IDs (uint8), shape ``(N,)``.

    Returns
    -------
    hist : dict[str, int]
        Mapping from super-class name → point count.
    """
    ids = np.asarray(super_ids, dtype=np.uint8)
    hist: dict[str, int] = {}
    for sc_id, name in SUPERCLASS_NAMES.items():
        hist[name] = int(np.sum(ids == sc_id))
    return hist
