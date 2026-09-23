"""Moving IoU, object recall/precision, stationary-vehicle FP rate (README 12.6)."""

from __future__ import annotations

import numpy as np  # noqa: F401


def motion_metrics(pred_objects: list, scan_gt: object) -> dict:
    raise NotImplementedError("Implemented in Phase 10, T10.6 (docs/PHASES.md).")
