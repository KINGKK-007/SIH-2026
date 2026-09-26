"""IoU/accuracy formulas (README 12.1-12.2)."""

from __future__ import annotations

import numpy as np


def confusion_iou(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> dict:
    """Return a confusion matrix and IoU metrics for integer class arrays.

    Rows are ground truth and columns are predictions. Values outside
    ``[0, n_classes)`` are rejected instead of being silently discarded.
    Classes absent from both arrays receive ``None`` IoU and are excluded from mIoU.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes differ: {pred.shape} != {gt.shape}")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    if pred.size and (
        pred.min() < 0 or pred.max() >= n_classes or gt.min() < 0 or gt.max() >= n_classes
    ):
        raise ValueError(f"class IDs must be in [0, {n_classes})")

    matrix = np.bincount(
        gt.astype(np.int64).ravel() * n_classes + pred.astype(np.int64).ravel(),
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    tp = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.full(n_classes, np.nan), where=union > 0)
    accuracy = float(tp.sum() / matrix.sum()) if matrix.sum() else None
    return {
        "confusion_matrix": matrix.tolist(),
        "support": support.tolist(),
        "iou": [None if np.isnan(value) else float(value) for value in iou],
        "miou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else None,
        "accuracy": accuracy,
        "n": int(matrix.sum()),
    }
