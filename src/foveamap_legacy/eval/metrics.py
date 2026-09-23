"""foveamap.eval.metrics — Distance-binned accuracy and IoU evaluation metrics.

Implements range-stratified segmentation accuracy and intersection-over-union (IoU)
metrics matching the FoveaMap ring boundaries (0-10m, 10-30m, 30-60m, 60-100m).

Math & Strategy
---------------
1. **Range Binning:**
   LiDAR point density decreases quadratically with distance. Evaluating overall
   mIoU without range stratification hides poor performance at long range.
   Metrics are binned into half-open intervals :math:`[r_{min}, r_{max})` based on
   Chebyshev distance :math:`\\max(|x|, |y|)` (matching square clipmap rings) or
   Euclidean distance :math:`\\sqrt{x^2 + y^2}`.

2. **Metrics:**
   - **Per-range Accuracy:**
     .. math::
         \\text{Acc} = \\frac{\\sum_{i} \\mathbb{I}(y_i = \\hat{y}_i)}{N_{eval}}
   - **Per-class IoU:**
     .. math::
         \\text{IoU}_c = \\frac{\\text{TP}_c}{\\text{TP}_c + \\text{FP}_c + \\text{FN}_c}
   - **Mean IoU (mIoU):** Average of :math:`\\text{IoU}_c` over classes present in the ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from foveamap_legacy.io.labels import (
    SUPERCLASS_NAMES,
    UNKNOWN,
    cls19_to_superclass,
    to_superclass,
    unpack_kitti_labels,
)

DEFAULT_RING_BOUNDARIES: list[tuple[float, float]] = [
    (0.0, 10.0),
    (10.0, 30.0),
    (30.0, 60.0),
    (60.0, 100.0),
]


def compute_point_ranges(
    points: NDArray[np.floating],
    metric: str = "chebyshev",
) -> NDArray[np.float32]:
    """Calculate distance for each LiDAR point from the sensor origin.

    Parameters
    ----------
    points : NDArray[floating]
        Shape ``(N, 3)`` or ``(N, 4)`` [x, y, z, ...].
    metric : {"chebyshev", "euclidean"}, default "chebyshev"
        Distance metric. "chebyshev" computes :math:`\\max(|x|, |y|)` matching
        the square nested clipmaps of FoveaMap. "euclidean" computes :math:`\\sqrt{x^2 + y^2}`.

    Returns
    -------
    ranges : NDArray[float32]
        Distance in metres for each point, shape ``(N,)``.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.empty(0, dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]

    if metric == "chebyshev":
        return np.maximum(np.abs(x), np.abs(y))
    elif metric == "euclidean":
        return np.hypot(x, y)
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Expected 'chebyshev' or 'euclidean'.")


def _standardize_labels(
    labels: NDArray,
    to_superclasses: bool = True,
) -> NDArray[np.uint8]:
    """Standardize label array to super-classes if requested or appropriate."""
    arr = np.asarray(labels)
    if arr.size == 0:
        return np.empty(0, dtype=np.uint8)

    if not to_superclasses:
        return arr.astype(np.uint16)

    # Check format
    if arr.dtype == np.uint32 or np.max(arr) > 255:
        # Raw KITTI label with instance bits
        sem, _ = unpack_kitti_labels(arr.astype(np.uint32))
        return to_superclass(sem)

    max_v = int(np.max(arr))
    # If already super-classes (0..3, 255)
    if max_v <= 3 or (max_v == UNKNOWN and np.all((arr <= 3) | (arr == UNKNOWN))):
        return arr.astype(np.uint8)

    # If 19-class benchmark (max <= 21)
    if max_v <= 21 or (max_v == UNKNOWN and np.all((arr <= 21) | (arr == UNKNOWN))):
        return cls19_to_superclass(arr.astype(np.uint8))

    # Fallback to raw ID to super-class
    return to_superclass(arr.astype(np.uint16))


def compute_accuracy_by_range(
    gt_labels: NDArray,
    pred_labels: NDArray,
    points: NDArray[np.floating],
    ring_boundaries: Sequence[tuple[float, float]] | None = None,
    distance_metric: str = "chebyshev",
    ignore_label: int = UNKNOWN,
    to_superclasses: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compute classification accuracy partitioned into distance intervals.

    Parameters
    ----------
    gt_labels : NDArray
        Ground truth labels, shape ``(N,)``.
    pred_labels : NDArray
        Predicted labels, shape ``(N,)``.
    points : NDArray[floating]
        Points array, shape ``(N, 3)`` or ``(N, 4)``.
    ring_boundaries : sequence of (float, float), optional
        List of ``(r_min, r_max)`` intervals in metres. Defaults to
        ``[(0, 10), (10, 30), (30, 60), (60, 100)]``.
    distance_metric : {"chebyshev", "euclidean"}, default "chebyshev"
        Distance metric used for range binning.
    ignore_label : int, default 255
        Label ID to exclude from evaluation (e.g. UNKNOWN / outlier).
    to_superclasses : bool, default True
        Whether to map inputs to the 4 super-classes before comparison.

    Returns
    -------
    results : dict[str, dict[str, Any]]
        Dictionary keyed by range bucket name (e.g. ``"0-10m"``) and ``"overall"``,
        with fields:
        - ``"accuracy"``: float in [0, 1]
        - ``"n_points"``: total points in bucket
        - ``"n_eval"``: evaluated points (excluding ignore_label)
        - ``"n_correct"``: count of correctly classified points
    """
    boundaries = ring_boundaries if ring_boundaries is not None else DEFAULT_RING_BOUNDARIES
    gt = _standardize_labels(gt_labels, to_superclasses=to_superclasses)
    pred = _standardize_labels(pred_labels, to_superclasses=to_superclasses)
    ranges = compute_point_ranges(points, metric=distance_metric)

    if len(gt) != len(pred) or len(gt) != len(ranges):
        raise ValueError(
            f"Length mismatch: gt ({len(gt)}), pred ({len(pred)}), points ({len(ranges)})"
        )

    results: dict[str, dict[str, Any]] = {}

    for r_min, r_max in boundaries:
        key = f"{int(r_min)}-{int(r_max)}m"
        in_range = (ranges >= r_min) & (ranges < r_max)
        n_total = int(np.sum(in_range))

        valid = in_range & (gt != ignore_label)
        n_eval = int(np.sum(valid))

        if n_eval > 0:
            n_correct = int(np.sum(gt[valid] == pred[valid]))
            acc = float(n_correct / n_eval)
        else:
            n_correct = 0
            acc = 0.0

        results[key] = {
            "accuracy": acc,
            "n_points": n_total,
            "n_eval": n_eval,
            "n_correct": n_correct,
            "r_min": float(r_min),
            "r_max": float(r_max),
        }

    # Overall metrics across all evaluated ranges
    all_valid = (gt != ignore_label)
    n_all_eval = int(np.sum(all_valid))
    if n_all_eval > 0:
        n_all_correct = int(np.sum(gt[all_valid] == pred[all_valid]))
        overall_acc = float(n_all_correct / n_all_eval)
    else:
        n_all_correct = 0
        overall_acc = 0.0

    results["overall"] = {
        "accuracy": overall_acc,
        "n_points": len(gt),
        "n_eval": n_all_eval,
        "n_correct": n_all_correct,
        "r_min": 0.0,
        "r_max": float(boundaries[-1][1]) if boundaries else 100.0,
    }

    return results


def compute_iou_by_range(
    gt_labels: NDArray,
    pred_labels: NDArray,
    points: NDArray[np.floating],
    ring_boundaries: Sequence[tuple[float, float]] | None = None,
    distance_metric: str = "chebyshev",
    ignore_label: int = UNKNOWN,
    to_superclasses: bool = True,
    classes: Sequence[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-class IoU and mIoU partitioned into distance intervals.

    Parameters
    ----------
    gt_labels : NDArray
        Ground truth labels, shape ``(N,)``.
    pred_labels : NDArray
        Predicted labels, shape ``(N,)``.
    points : NDArray[floating]
        Points array, shape ``(N, 3)`` or ``(N, 4)``.
    ring_boundaries : sequence of (float, float), optional
        List of ``(r_min, r_max)`` intervals in metres.
    distance_metric : {"chebyshev", "euclidean"}, default "chebyshev"
        Distance metric used for range binning.
    ignore_label : int, default 255
        Label ID to exclude from evaluation.
    to_superclasses : bool, default True
        Whether to map inputs to the 4 super-classes.
    classes : sequence of int, optional
        Classes to evaluate. Defaults to ``[0, 1, 2, 3]`` for super-classes.

    Returns
    -------
    results : dict[str, dict[str, Any]]
        Dictionary keyed by range bucket name (e.g. ``"0-10m"``) and ``"overall"``,
        with fields:
        - ``"mIoU"``: float in [0, 1]
        - ``"class_iou"``: dict[int, float]
        - ``"class_names_iou"``: dict[str, float]
        - ``"n_points"``: total points in bucket
        - ``"n_eval"``: evaluated points
    """
    boundaries = ring_boundaries if ring_boundaries is not None else DEFAULT_RING_BOUNDARIES
    gt = _standardize_labels(gt_labels, to_superclasses=to_superclasses)
    pred = _standardize_labels(pred_labels, to_superclasses=to_superclasses)
    ranges = compute_point_ranges(points, metric=distance_metric)

    if len(gt) != len(pred) or len(gt) != len(ranges):
        raise ValueError(
            f"Length mismatch: gt ({len(gt)}), pred ({len(pred)}), points ({len(ranges)})"
        )

    eval_classes = list(classes) if classes is not None else ([0, 1, 2, 3] if to_superclasses else sorted(set(gt[gt != ignore_label])))

    results: dict[str, dict[str, Any]] = {}

    def _eval_subset(mask: NDArray[np.bool_]) -> dict[str, Any]:
        sub_gt = gt[mask]
        sub_pred = pred[mask]

        class_iou: dict[int, float] = {}
        class_names_iou: dict[str, float] = {}
        valid_ious: list[float] = []

        for c in eval_classes:
            gt_c = (sub_gt == c)
            pred_c = (sub_pred == c)

            tp = int(np.sum(gt_c & pred_c))
            fp = int(np.sum(~gt_c & pred_c))
            fn = int(np.sum(gt_c & ~pred_c))

            denom = tp + fp + fn
            if denom > 0:
                iou = float(tp / denom)
                valid_ious.append(iou)
            else:
                # Class not present in GT or pred in this range
                iou = float("nan")

            class_iou[c] = iou
            c_name = SUPERCLASS_NAMES.get(c, str(c))
            class_names_iou[c_name] = iou

        miou = float(np.nanmean(valid_ious)) if valid_ious else 0.0

        return {
            "mIoU": miou,
            "class_iou": class_iou,
            "class_names_iou": class_names_iou,
            "n_points": int(len(sub_gt)),
        }

    for r_min, r_max in boundaries:
        key = f"{int(r_min)}-{int(r_max)}m"
        in_range = (ranges >= r_min) & (ranges < r_max) & (gt != ignore_label)
        bucket_res = _eval_subset(in_range)
        bucket_res["r_min"] = float(r_min)
        bucket_res["r_max"] = float(r_max)
        results[key] = bucket_res

    overall_mask = (gt != ignore_label)
    overall_res = _eval_subset(overall_mask)
    overall_res["r_min"] = 0.0
    overall_res["r_max"] = float(boundaries[-1][1]) if boundaries else 100.0
    results["overall"] = overall_res

    return results


def format_metrics_table(
    accuracy_dict: dict[str, dict[str, Any]],
    iou_dict: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Format accuracy and IoU results as a clean ASCII console table.

    Parameters
    ----------
    accuracy_dict : dict
        Output of :func:`compute_accuracy_by_range`.
    iou_dict : dict, optional
        Output of :func:`compute_iou_by_range`.

    Returns
    -------
    table_str : str
        Ready-to-print formatted table.
    """
    lines = []
    lines.append("=" * 72)
    lines.append(f"{'Range':<12} {'Points':>10} {'Eval Pts':>10} {'Accuracy':>10} {'mIoU':>10}")
    lines.append("-" * 72)

    for key, acc_info in accuracy_dict.items():
        n_pts = acc_info.get("n_points", 0)
        n_eval = acc_info.get("n_eval", 0)
        acc_pct = acc_info.get("accuracy", 0.0) * 100.0

        if iou_dict and key in iou_dict:
            miou_val = iou_dict[key].get("mIoU", 0.0) * 100.0
            miou_str = f"{miou_val:6.2f}%"
        else:
            miou_str = "   N/A   "

        lines.append(
            f"{key:<12} {n_pts:>10,} {n_eval:>10,} {acc_pct:>9.2f}% {miou_str:>10}"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


def metrics_to_dataframe(
    accuracy_dict: dict[str, dict[str, Any]],
    iou_dict: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Convert metrics dictionaries into a pandas DataFrame (if pandas is installed).

    Parameters
    ----------
    accuracy_dict : dict
        Output from :func:`compute_accuracy_by_range`.
    iou_dict : dict, optional
        Output from :func:`compute_iou_by_range`.

    Returns
    -------
    df : pandas.DataFrame or dict
        DataFrame if pandas is available, otherwise a structured list of row dicts.
    """
    rows = []
    for key, acc_info in accuracy_dict.items():
        row = {
            "range": key,
            "n_points": acc_info.get("n_points", 0),
            "n_eval": acc_info.get("n_eval", 0),
            "accuracy": acc_info.get("accuracy", 0.0),
        }
        if iou_dict and key in iou_dict:
            row["mIoU"] = iou_dict[key].get("mIoU", 0.0)
            for c_name, c_val in iou_dict[key].get("class_names_iou", {}).items():
                row[f"IoU_{c_name}"] = c_val
        rows.append(row)

    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows
