"""Shared records passed between stages (README 9.1). These dataclasses are the inter-phase contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Scan:
    seq: str
    idx: int
    xyz: np.ndarray  # (N,3) float32, metres, Velodyne frame
    remission: np.ndarray  # (N,) float32
    raw_labels: np.ndarray | None  # (N,) uint32 or None
    pose: np.ndarray  # (4,4) camera-0 pose from poses.txt
    timestamp: float


@dataclass
class Prediction:
    raw_ids: np.ndarray  # (N,) uint8, static raw SemanticKITTI IDs (learning_map_inv applied)
    conf: np.ndarray  # (N,) uint8, 0..255


@dataclass
class ObjectBox:
    id: int
    cls_name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float
    n_points: int
    mean_conf: float
    moving: bool
    vote_frac: float
    speed_mps: float | None
    safety_critical: bool


@dataclass
class ClassifiedScan:
    scan: Scan
    super_cls: np.ndarray  # (N,) uint8 in {0..4}
    moving: np.ndarray  # (N,) bool
    conf: np.ndarray  # (N,) uint8
    objects: list[ObjectBox] = field(default_factory=list)
