"""Streaming Sequence 08 semantic evaluation from cached model predictions."""

from __future__ import annotations

import json
from collections.abc import Sequence as SeqType
from pathlib import Path
from typing import Any

import numpy as np

from foveamap.eval.buckets import bucket_labels, bucket_of_range, horizontal_range
from foveamap.eval.metrics import confusion_iou
from foveamap.io.labels import SUPER_CLASS_NAMES, raw_to_learning, raw_to_super
from foveamap.io.sequence import Sequence
from foveamap.models.cache import CachedModel

EVAL_EDGES_M: tuple[float, ...] = (0.0, 10.0, 30.0, 50.0, 60.0, 80.0, 100.0)
LEARNING_CLASS_NAMES: tuple[str, ...] = (
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
    "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence",
    "vegetation", "trunk", "terrain", "pole", "traffic-sign",
)


def _matrix_metrics(matrix: np.ndarray, names: SeqType[str], class_offset: int = 0) -> dict[str, Any]:
    """Calculate metrics directly from an accumulated confusion matrix."""
    indices = np.arange(class_offset, class_offset + len(names))
    tp = np.diag(matrix)[indices]
    support = matrix.sum(axis=1)[indices]
    predicted = matrix.sum(axis=0)[indices]
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.full(len(names), np.nan), where=union > 0)
    return {
        "n": int(matrix.sum()),
        "accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else None,
        "miou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else None,
        "per_class": {
            name: {"iou": None if np.isnan(iou[i]) else float(iou[i]), "support": int(support[i])}
            for i, name in enumerate(names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_arrays(
    pred_raw: np.ndarray,
    gt_raw: np.ndarray,
    xyz: np.ndarray,
    edges_m: SeqType[float] = EVAL_EDGES_M,
) -> dict[str, Any]:
    """Evaluate one or more concatenated scans with explicit point accounting."""
    pred_raw = np.asarray(pred_raw)
    gt_raw = np.asarray(gt_raw)
    xyz = np.asarray(xyz)
    if pred_raw.shape != gt_raw.shape or pred_raw.shape != (len(xyz),):
        raise ValueError("pred_raw, gt_raw and xyz must describe the same number of points")

    gt_learning = raw_to_learning(gt_raw)
    pred_learning = raw_to_learning(pred_raw)
    gt_super, _ = raw_to_super(gt_raw)
    pred_super, _ = raw_to_super(pred_raw)
    buckets = bucket_of_range(horizontal_range(xyz), edges_m)
    labels = bucket_labels(edges_m)
    valid_semantic = gt_learning > 0

    learning_matrix = np.asarray(
        confusion_iou(pred_learning[valid_semantic], gt_learning[valid_semantic], 20)["confusion_matrix"]
    )
    learning = _matrix_metrics(learning_matrix, LEARNING_CLASS_NAMES, class_offset=1)
    super_valid = gt_super > 0
    super_matrix = np.asarray(
        confusion_iou(pred_super[super_valid], gt_super[super_valid], 5)["confusion_matrix"]
    )
    superclass = _matrix_metrics(super_matrix, SUPER_CLASS_NAMES[1:], class_offset=1)

    by_distance: dict[str, Any] = {}
    for idx, label in enumerate(labels):
        in_bucket = buckets == idx
        valid = in_bucket & valid_semantic
        matrix = np.asarray(
            confusion_iou(pred_learning[valid], gt_learning[valid], 20)["confusion_matrix"]
        )
        metrics = _matrix_metrics(matrix, LEARNING_CLASS_NAMES, class_offset=1)
        by_distance[label] = {
            "points": int(in_bucket.sum()),
            "evaluated_points": int(valid.sum()),
            "ignored_gt_points": int((in_bucket & ~valid_semantic).sum()),
            "unknown_prediction_points": int((in_bucket & (pred_learning == 0)).sum()),
            "unknown_prediction_rate": (
                float((in_bucket & (pred_learning == 0)).sum() / in_bucket.sum())
                if in_bucket.any() else None
            ),
            "miou": metrics["miou"],
            "accuracy": metrics["accuracy"],
        }

    return {
        "point_accounting": {
            "input_points": int(len(xyz)),
            "within_100m": int((buckets >= 0).sum()),
            "beyond_100m_or_invalid": int((buckets < 0).sum()),
            "evaluated_19_class_points": int(valid_semantic.sum()),
            "ignored_gt_points": int((~valid_semantic).sum()),
            "prediction_length_matches": True,
        },
        "semantic_19_class": learning,
        "superclass_4_class": superclass,
        "by_distance": by_distance,
    }


def evaluate_cached_sequence(
    data_root: str | Path,
    cache_root: str | Path,
    model_name: str,
    sequence: str = "08",
    start: int = 1271,
    stop: int = 4071,
    stride: int = 1,
) -> dict[str, Any]:
    """Evaluate cached predictions over a half-open frame interval."""
    seq = Sequence(data_root, sequence)
    model = CachedModel(model_name, cache_root)
    if stride <= 0 or start < 0 or stop <= start or stop > len(seq):
        raise ValueError(f"invalid frame interval start={start}, stop={stop}, stride={stride}")

    matrices = np.zeros((20, 20), dtype=np.int64)
    super_matrices = np.zeros((5, 5), dtype=np.int64)
    bucket_matrices = np.zeros((len(EVAL_EDGES_M) - 1, 20, 20), dtype=np.int64)
    bucket_points = np.zeros(len(EVAL_EDGES_M) - 1, dtype=np.int64)
    bucket_ignored = np.zeros_like(bucket_points)
    bucket_unknown = np.zeros_like(bucket_points)
    accounting = {"input_points": 0, "within_100m": 0, "beyond_100m_or_invalid": 0,
                  "evaluated_19_class_points": 0, "ignored_gt_points": 0}
    frames = list(range(start, stop, stride))

    for frame_idx in frames:
        scan = seq.load_frame(frame_idx)
        pred = model.predict(scan)
        one = evaluate_arrays(pred.raw_ids, scan.raw_labels, scan.xyz)
        matrices += np.asarray(one["semantic_19_class"]["confusion_matrix"], dtype=np.int64)
        super_matrices += np.asarray(one["superclass_4_class"]["confusion_matrix"], dtype=np.int64)
        for key in accounting:
            accounting[key] += one["point_accounting"][key]
        for idx, label in enumerate(bucket_labels(EVAL_EDGES_M)):
            bucket_points[idx] += one["by_distance"][label]["points"]
            bucket_ignored[idx] += one["by_distance"][label]["ignored_gt_points"]
            bucket_unknown[idx] += one["by_distance"][label]["unknown_prediction_points"]
            gt_learning = raw_to_learning(scan.raw_labels)
            pred_learning = raw_to_learning(pred.raw_ids)
            bucket = bucket_of_range(horizontal_range(scan.xyz), EVAL_EDGES_M)
            valid = (bucket == idx) & (gt_learning > 0)
            bucket_matrices[idx] += np.asarray(
                confusion_iou(pred_learning[valid], gt_learning[valid], 20)["confusion_matrix"]
            )

    by_distance = {}
    for idx, label in enumerate(bucket_labels(EVAL_EDGES_M)):
        metrics = _matrix_metrics(bucket_matrices[idx], LEARNING_CLASS_NAMES, class_offset=1)
        by_distance[label] = {
            "points": int(bucket_points[idx]),
            "evaluated_points": metrics["n"],
            "ignored_gt_points": int(bucket_ignored[idx]),
            "unknown_prediction_points": int(bucket_unknown[idx]),
            "unknown_prediction_rate": (
                float(bucket_unknown[idx] / bucket_points[idx]) if bucket_points[idx] else None
            ),
            "miou": metrics["miou"],
            "accuracy": metrics["accuracy"],
        }

    meta_path = Path(cache_root) / model_name / sequence / "meta.json"
    provenance = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
    return {
        "schema_version": 1,
        "evaluation": "sequence-08 cached semantic checkpoint reproduction",
        "sequence": sequence,
        "frame_interval": {"start_inclusive": start, "stop_exclusive": stop, "stride": stride,
                           "n_frames": len(frames)},
        "distance_edges_m": list(EVAL_EDGES_M),
        "cache_provenance": provenance,
        "provenance_status": "present" if provenance is not None else "missing",
        "point_accounting": accounting | {"prediction_length_matches": True},
        "semantic_19_class": _matrix_metrics(matrices, LEARNING_CLASS_NAMES, class_offset=1),
        "superclass_4_class": _matrix_metrics(super_matrices, SUPER_CLASS_NAMES[1:], class_offset=1),
        "by_distance": by_distance,
    }


def semantic_report_md(report: dict[str, Any]) -> str:
    """Render the core semantic evidence as a reviewable Markdown table."""
    overall = report["semantic_19_class"]
    lines = [
        "# Sequence 08 semantic evaluation", "",
        f"Frames: {report['frame_interval']['start_inclusive']}-{report['frame_interval']['stop_exclusive'] - 1} "
        f"({report['frame_interval']['n_frames']} evaluated)", "",
        f"19-class mIoU: **{overall['miou'] * 100:.2f}%**" if overall["miou"] is not None else "19-class mIoU: N/A",
        f"Point accuracy: **{overall['accuracy'] * 100:.2f}%**" if overall["accuracy"] is not None else "Point accuracy: N/A",
        f"Cache provenance: **{report['provenance_status'].upper()}**", "",
        "| Distance | Points | Evaluated | Unknown prediction | mIoU | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in report["by_distance"].items():
        unknown = "N/A" if row["unknown_prediction_rate"] is None else f"{row['unknown_prediction_rate'] * 100:.2f}%"
        miou = "N/A" if row["miou"] is None else f"{row['miou'] * 100:.2f}%"
        accuracy = "N/A" if row["accuracy"] is None else f"{row['accuracy'] * 100:.2f}%"
        lines.append(f"| {label} | {row['points']:,} | {row['evaluated_points']:,} | {unknown} | {miou} | {accuracy} |")
    return "\n".join(lines) + "\n"
