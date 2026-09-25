"""Evaluation of point-level moving-object segmentation and 3D object detection (tasks T10.6, T10.7)."""

from __future__ import annotations

from typing import Any

import numpy as np

from foveamap.pipeline.records import ObjectBox

DEFAULT_DISTANCE_BUCKETS: list[tuple[float, float, str]] = [
    (0.0, 10.0, "0-10m"),
    (10.0, 30.0, "10-30m"),
    (30.0, 60.0, "30-60m"),
    (60.0, 100.0, "60-100m"),
]

STATIONARY_VEHICLE_CLASSES = {10, 13, 18, 20}  # car, bus, truck, other-vehicle


def compute_point_motion_metrics(
    pred_moving: np.ndarray,
    gt_moving: np.ndarray,
    distances: np.ndarray,
    raw_labels: np.ndarray | None = None,
    buckets: list[tuple[float, float, str]] = DEFAULT_DISTANCE_BUCKETS,
) -> dict[str, Any]:
    """Calculate point-level moving IoU and stationary vehicle false-positive rate.

    Args:
        pred_moving: (N,) bool array of predicted moving flags.
        gt_moving: (N,) bool array of ground-truth moving flags (classes 252-259).
        distances: (N,) float array of point distances from sensor.
        raw_labels: (N,) optional raw semantic/instance IDs to calculate FPR on parked vehicles.
        buckets: List of (r_min, r_max, label) range intervals.

    Returns:
        Dictionary with overall and per-bucket TP, FP, FN, IoU, and stationary vehicle FPR.
    """
    pred = np.asarray(pred_moving, dtype=bool)
    gt = np.asarray(gt_moving, dtype=bool)
    dist = np.asarray(distances, dtype=np.float32)

    total_tp = int(np.sum(pred & gt))
    total_fp = int(np.sum(pred & ~gt))
    total_fn = int(np.sum(~pred & gt))
    total_tn = int(np.sum(~pred & ~gt))
    denom = total_tp + total_fp + total_fn
    total_iou = float(total_tp / denom) if denom > 0 else 0.0

    stat_veh_fpr = 0.0
    stat_veh_count = 0
    if raw_labels is not None:
        sem_ids = (raw_labels & 0xFFFF).astype(np.int32)
        stat_veh_mask = np.isin(sem_ids, list(STATIONARY_VEHICLE_CLASSES)) & ~gt
        stat_veh_count = int(np.sum(stat_veh_mask))
        if stat_veh_count > 0:
            stat_veh_fp = int(np.sum(pred & stat_veh_mask))
            stat_veh_fpr = float(stat_veh_fp / stat_veh_count)

    bucket_metrics: dict[str, dict[str, Any]] = {}
    for r_min, r_max, name in buckets:
        in_bucket = (dist >= r_min) & (dist < r_max)
        b_n = int(np.sum(in_bucket))
        if b_n == 0:
            bucket_metrics[name] = {
                "n_points": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "iou": 0.0,
                "stat_veh_fpr": 0.0,
            }
            continue

        b_pred = pred[in_bucket]
        b_gt = gt[in_bucket]
        tp = int(np.sum(b_pred & b_gt))
        fp = int(np.sum(b_pred & ~b_gt))
        fn = int(np.sum(~b_pred & b_gt))
        d = tp + fp + fn
        iou = float(tp / d) if d > 0 else 0.0

        b_fpr = 0.0
        if raw_labels is not None:
            b_raw = (raw_labels[in_bucket] & 0xFFFF).astype(np.int32)
            b_stat = np.isin(b_raw, list(STATIONARY_VEHICLE_CLASSES)) & ~b_gt
            n_stat = int(np.sum(b_stat))
            if n_stat > 0:
                b_fpr = float(np.sum(b_pred & b_stat) / n_stat)

        bucket_metrics[name] = {
            "n_points": b_n,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": round(iou, 4),
            "stat_veh_fpr": round(b_fpr, 4),
        }

    return {
        "n_total": len(pred),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "moving_iou": round(total_iou, 4),
        "stationary_vehicle_fpr": round(stat_veh_fpr, 4),
        "stationary_vehicle_count": stat_veh_count,
        "by_distance": bucket_metrics,
    }


def compute_object_level_metrics(
    pred_objects: list[ObjectBox],
    gt_instances: np.ndarray,
    point_cluster_ids: np.ndarray,
    distances: np.ndarray,
    buckets: list[tuple[float, float, str]] = DEFAULT_DISTANCE_BUCKETS,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Calculate object-level recall and precision.

    A predicted cluster matches a GT instance if >= 50% of the predicted cluster's
    points belong to that GT instance (README Section 6.4 and Phase 10 T10.6).

    Args:
        pred_objects: List of predicted ObjectBox instances.
        gt_instances: (N,) int32 ground truth instance IDs for all points (0 = background).
        point_cluster_ids: (N,) int32 cluster IDs assigned to points (-1 = unassigned/noise).
        distances: (N,) float32 range per point.
        buckets: Distance intervals.
        iou_threshold: Minimum point overlap fraction to declare a match (default 0.50).

    Returns:
        Dictionary with total and per-distance recall, precision, and F1.
    """
    valid_gt_mask = gt_instances > 0
    unique_gts = np.unique(gt_instances[valid_gt_mask])
    n_gt = len(unique_gts)
    n_pred = len(pred_objects)

    if n_gt == 0 and n_pred == 0:
        return {
            "n_gt_objects": 0,
            "n_pred_objects": 0,
            "matched": 0,
            "recall": 1.0,
            "precision": 1.0,
            "f1": 1.0,
            "by_distance": {},
        }

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    # Determine match for each predicted object
    for obj in pred_objects:
        obj_pts_mask = point_cluster_ids == obj.id
        n_obj_pts = np.sum(obj_pts_mask)
        if n_obj_pts == 0:
            continue

        in_cluster_gt = gt_instances[obj_pts_mask]
        in_cluster_gt_nonzero = in_cluster_gt[in_cluster_gt > 0]
        if len(in_cluster_gt_nonzero) == 0:
            continue

        # Find most frequent GT instance in this predicted cluster
        counts = np.bincount(in_cluster_gt_nonzero)
        best_gt = int(counts.argmax())
        overlap_frac = counts[best_gt] / n_obj_pts

        if overlap_frac >= iou_threshold:
            matched_gt.add(best_gt)
            matched_pred.add(obj.id)

    total_matched = len(matched_gt)
    recall = float(total_matched / n_gt) if n_gt > 0 else 0.0
    precision = float(len(matched_pred) / n_pred) if n_pred > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Precompute GT object centroid ranges once
    gt_mean_dists = {}
    for g_id in unique_gts:
        m = gt_instances == g_id
        gt_mean_dists[g_id] = float(np.mean(distances[m]))

    # Bucket metrics by object centroid distance
    bucket_metrics: dict[str, dict[str, Any]] = {}
    for r_min, r_max, name in buckets:
        # GT objects whose points average within range
        b_gt_count = sum(1 for d in gt_mean_dists.values() if r_min <= d < r_max)


        b_pred_count = 0
        b_matched_count = 0
        for obj in pred_objects:
            d = float(np.linalg.norm(obj.center[:2]))
            if r_min <= d < r_max:
                b_pred_count += 1
                if obj.id in matched_pred:
                    b_matched_count += 1

        b_rec = float(b_matched_count / b_gt_count) if b_gt_count > 0 else 0.0
        b_prec = float(b_matched_count / b_pred_count) if b_pred_count > 0 else 0.0
        b_f1 = float(2 * b_prec * b_rec / (b_prec + b_rec)) if (b_prec + b_rec) > 0 else 0.0

        bucket_metrics[name] = {
            "n_gt": b_gt_count,
            "n_pred": b_pred_count,
            "matched": b_matched_count,
            "recall": round(b_rec, 4),
            "precision": round(b_prec, 4),
            "f1": round(b_f1, 4),
        }

    return {
        "n_gt_objects": n_gt,
        "n_pred_objects": n_pred,
        "matched": total_matched,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "by_distance": bucket_metrics,
    }
