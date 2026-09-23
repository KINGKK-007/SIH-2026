"""IoU/accuracy formulas and Wilson intervals (README 12.1-12.2)."""

from __future__ import annotations

import numpy as np  # noqa: F401


def confusion_iou(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> dict:
    raise NotImplementedError("Implemented in Phase 9, T9.3 (docs/PHASES.md).")
