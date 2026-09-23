"""Canonical raw SemanticKITTI ID -> FoveaMap super-class mapping (README 6.1, Appendix A, task T3.3)."""

from __future__ import annotations

import numpy as np

UNKNOWN = 0
DRIVABLE = 1
NON_DRIVABLE_TERRAIN = 2
STATIC_OBSTACLE = 3
DYNAMIC = 4

SUPER_CLASS_NAMES = ("UNKNOWN", "DRIVABLE", "NON_DRIVABLE_TERRAIN", "STATIC_OBSTACLE", "DYNAMIC")


def raw_to_super(raw_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map raw semantic IDs to ``(super_ids uint8, moving bool)``. Unknown IDs map to ``UNKNOWN``."""
    raise NotImplementedError("Implemented in Phase 3, T3.3 (docs/PHASES.md).")
