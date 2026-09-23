"""tests/test_metrics.py — Verification suite for range-binned evaluation metrics.

QA Requirements Addressed
-------------------------
E1  Range-Binned Accuracy:
    - Points explicitly positioned in each of the 4 distance buckets:
      * Ring 1: 0–10 m (e.g. at 5 m)
      * Ring 2: 10–30 m (e.g. at 20 m)
      * Ring 3: 30–60 m (e.g. at 45 m)
      * Ring 4: 60–100 m (e.g. at 80 m)
    - Synthetic labels with strictly known error rates:
      * 100% accuracy in Ring 1 (e.g. 100/100 correct -> 1.0)
      * 80% accuracy in Ring 2 (e.g. 80/100 correct -> 0.80)
      * 50% accuracy in Ring 3 (e.g. 50/100 correct -> 0.50)
      * 25% accuracy in Ring 4 (e.g. 25/100 correct -> 0.25)
    - Assert compute_accuracy_by_range yields exact mathematical expectations.
E2  Intersection-over-Union (IoU) by Range:
    - Known confusion matrix (TP, FP, FN) evaluated per bucket.
    - Assert per-class IoU = TP / (TP + FP + FN) and mIoU match manual derivations.
E3  Ignore Labels:
    - Points labeled 255 (UNKNOWN / unlabeled) are excluded from evaluation counts.
E4  Distance Metrics:
    - Verify Chebyshev (max(|x|, |y|)) vs Euclidean (sqrt(x^2 + y^2)) range binning.
E5  Formatting & DataFrames:
    - Verify format_metrics_table() produces readable ASCII table.
    - Verify metrics_to_dataframe() converts dictionary correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from foveamap_legacy.eval.metrics import (
    compute_accuracy_by_range,
    compute_iou_by_range,
    compute_point_ranges,
    format_metrics_table,
    metrics_to_dataframe,
)
from foveamap_legacy.io.labels import (
    DRIVABLE,
    DYNAMIC,
    NON_DRIVABLE_TERRAIN,
    STATIC_OBSTACLE,
    UNKNOWN,
)


class TestRangeStratifiedMetrics:
    """Mathematical verification of distance-binned accuracy and IoU."""

    def test_compute_point_ranges_chebyshev_vs_euclidean(self) -> None:
        """Point at (3.0, 4.0): Chebyshev = 4.0, Euclidean = 5.0."""
        pt = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)

        r_cheb = compute_point_ranges(pt, metric="chebyshev")
        assert r_cheb[0] == pytest.approx(4.0)

        r_eucl = compute_point_ranges(pt, metric="euclidean")
        assert r_eucl[0] == pytest.approx(5.0)

    def test_exact_accuracy_by_range(self) -> None:
        """Points explicitly placed in 4 buckets with known accuracy: 100%, 80%, 50%, 25%."""
        # 100 points per bucket
        n_per_bucket = 100

        # Bucket 1: d = 5m (0-10m)
        pts_r1 = np.column_stack([np.full(n_per_bucket, 5.0), np.zeros(n_per_bucket), np.zeros(n_per_bucket)])
        gt_r1 = np.full(n_per_bucket, DRIVABLE, dtype=np.uint8)
        pred_r1 = gt_r1.copy()  # 100% correct (100/100)

        # Bucket 2: d = 20m (10-30m)
        pts_r2 = np.column_stack([np.full(n_per_bucket, 20.0), np.zeros(n_per_bucket), np.zeros(n_per_bucket)])
        gt_r2 = np.full(n_per_bucket, NON_DRIVABLE_TERRAIN, dtype=np.uint8)
        pred_r2 = gt_r2.copy()
        pred_r2[80:] = STATIC_OBSTACLE  # 80 correct, 20 wrong -> 80%

        # Bucket 3: d = 45m (30-60m)
        pts_r3 = np.column_stack([np.full(n_per_bucket, 45.0), np.zeros(n_per_bucket), np.zeros(n_per_bucket)])
        gt_r3 = np.full(n_per_bucket, STATIC_OBSTACLE, dtype=np.uint8)
        pred_r3 = gt_r3.copy()
        pred_r3[50:] = DRIVABLE  # 50 correct, 50 wrong -> 50%

        # Bucket 4: d = 80m (60-100m)
        pts_r4 = np.column_stack([np.full(n_per_bucket, 80.0), np.zeros(n_per_bucket), np.zeros(n_per_bucket)])
        gt_r4 = np.full(n_per_bucket, DYNAMIC, dtype=np.uint8)
        pred_r4 = gt_r4.copy()
        pred_r4[25:] = NON_DRIVABLE_TERRAIN  # 25 correct, 75 wrong -> 25%

        # Assemble full point cloud (400 points total)
        points = np.vstack([pts_r1, pts_r2, pts_r3, pts_r4]).astype(np.float32)
        gt = np.concatenate([gt_r1, gt_r2, gt_r3, gt_r4])
        pred = np.concatenate([pred_r1, pred_r2, pred_r3, pred_r4])

        results = compute_accuracy_by_range(gt, pred, points, to_superclasses=False)

        # Assert exact mathematical expectations per range bucket
        assert results["0-10m"]["accuracy"] == pytest.approx(1.000)
        assert results["0-10m"]["n_points"] == 100
        assert results["0-10m"]["n_correct"] == 100

        assert results["10-30m"]["accuracy"] == pytest.approx(0.800)
        assert results["10-30m"]["n_points"] == 100
        assert results["10-30m"]["n_correct"] == 80

        assert results["30-60m"]["accuracy"] == pytest.approx(0.500)
        assert results["30-60m"]["n_points"] == 100
        assert results["30-60m"]["n_correct"] == 50

        assert results["60-100m"]["accuracy"] == pytest.approx(0.250)
        assert results["60-100m"]["n_points"] == 100
        assert results["60-100m"]["n_correct"] == 25

        # Overall: (100 + 80 + 50 + 25) / 400 = 255 / 400 = 0.6375 (63.75%)
        assert results["overall"]["accuracy"] == pytest.approx(255.0 / 400.0)
        assert results["overall"]["n_points"] == 400
        assert results["overall"]["n_correct"] == 255

    def test_ignore_unknown_label_in_accuracy(self) -> None:
        """Points with label UNKNOWN (255) are excluded from accuracy denominator."""
        pts = np.array([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=np.float32)

        # Point 0: GT=DRIVABLE, Pred=DRIVABLE (correct)
        # Point 1: GT=DRIVABLE, Pred=STATIC_OBSTACLE (wrong)
        # Point 2: GT=UNKNOWN (255) -> must be ignored!
        gt = np.array([DRIVABLE, DRIVABLE, UNKNOWN], dtype=np.uint8)
        pred = np.array([DRIVABLE, STATIC_OBSTACLE, DRIVABLE], dtype=np.uint8)

        res = compute_accuracy_by_range(gt, pred, pts, to_superclasses=False)

        # Evaluated points = 2 (not 3); correct = 1 -> accuracy = 0.5
        assert res["0-10m"]["n_points"] == 3
        assert res["0-10m"]["n_eval"] == 2
        assert res["0-10m"]["n_correct"] == 1
        assert res["0-10m"]["accuracy"] == pytest.approx(0.5)

    def test_compute_iou_by_range_exact(self) -> None:
        """Verify IoU = TP / (TP + FP + FN) on controlled class distributions."""
        # 100 points at range 5m (Bucket 0-10m)
        pts = np.column_stack([np.full(100, 5.0), np.zeros(100), np.zeros(100)]).astype(np.float32)

        # For Class DRIVABLE (0):
        # 60 true positives: GT=0, Pred=0
        # 20 false positives: GT=1, Pred=0
        # 20 false negatives: GT=0, Pred=1
        # IoU(0) = 60 / (60 + 20 + 20) = 60 / 100 = 0.60
        # For Class NON_DRIVABLE (1):
        # 20 false negatives: GT=1, Pred=0
        # 20 false positives: GT=0, Pred=1
        # 0 true positives: IoU(1) = 0 / 40 = 0.0
        gt = np.array([DRIVABLE] * 60 + [NON_DRIVABLE_TERRAIN] * 20 + [DRIVABLE] * 20, dtype=np.uint8)
        pred = np.array([DRIVABLE] * 60 + [DRIVABLE] * 20 + [NON_DRIVABLE_TERRAIN] * 20, dtype=np.uint8)

        iou_res = compute_iou_by_range(gt, pred, pts, to_superclasses=False, classes=[DRIVABLE, NON_DRIVABLE_TERRAIN])

        r1 = iou_res["0-10m"]
        assert r1["class_iou"][DRIVABLE] == pytest.approx(0.60)
        assert r1["class_iou"][NON_DRIVABLE_TERRAIN] == pytest.approx(0.00)
        assert r1["mIoU"] == pytest.approx(0.30)  # (0.60 + 0.00) / 2

    def test_points_beyond_outer_boundary(self) -> None:
        """Points at 120m exceed the 100m outer boundary and are excluded from 0-100m buckets."""
        pts = np.array([[120.0, 0.0, 0.0]], dtype=np.float32)
        gt = np.array([DRIVABLE], dtype=np.uint8)
        pred = np.array([DRIVABLE], dtype=np.uint8)

        res = compute_accuracy_by_range(gt, pred, pts, to_superclasses=False)

        assert res["0-10m"]["n_points"] == 0
        assert res["10-30m"]["n_points"] == 0
        assert res["30-60m"]["n_points"] == 0
        assert res["60-100m"]["n_points"] == 0
        # Overall includes all evaluated points
        assert res["overall"]["n_points"] == 1

    def test_format_metrics_table(self) -> None:
        """ASCII metrics table contains standard column headers and rows."""
        pts = np.array([[5.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)
        gt = np.array([DRIVABLE, NON_DRIVABLE_TERRAIN], dtype=np.uint8)
        pred = gt.copy()

        acc_dict = compute_accuracy_by_range(gt, pred, pts, to_superclasses=False)
        iou_dict = compute_iou_by_range(gt, pred, pts, to_superclasses=False)

        table_str = format_metrics_table(acc_dict, iou_dict)
        assert "Range" in table_str
        assert "Accuracy" in table_str
        assert "mIoU" in table_str
        assert "0-10m" in table_str
        assert "10-30m" in table_str
        assert "overall" in table_str

    def test_metrics_to_dataframe(self) -> None:
        """Converting metrics to dataframe returns structured row entries."""
        pts = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        gt = np.array([DRIVABLE], dtype=np.uint8)
        pred = np.array([DRIVABLE], dtype=np.uint8)

        acc_dict = compute_accuracy_by_range(gt, pred, pts, to_superclasses=False)
        iou_dict = compute_iou_by_range(gt, pred, pts, to_superclasses=False)

        df = metrics_to_dataframe(acc_dict, iou_dict)
        # Can be DataFrame or list of dicts depending on pandas installation
        if hasattr(df, "columns"):
            assert "range" in df.columns
            assert "accuracy" in df.columns
        else:
            assert isinstance(df, list)
            assert len(df) > 0
            assert "range" in df[0]
