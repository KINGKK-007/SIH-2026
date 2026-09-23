"""foveamap.eval — Evaluation metrics and experiment runners."""

from foveamap_legacy.eval.metrics import (
    DEFAULT_RING_BOUNDARIES,
    compute_accuracy_by_range,
    compute_iou_by_range,
    compute_point_ranges,
    format_metrics_table,
    metrics_to_dataframe,
)

__all__ = [
    "DEFAULT_RING_BOUNDARIES",
    "compute_point_ranges",
    "compute_accuracy_by_range",
    "compute_iou_by_range",
    "format_metrics_table",
    "metrics_to_dataframe",
]
