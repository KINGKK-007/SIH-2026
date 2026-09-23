"""The ``SegmentationModel`` protocol (README 9.3)."""

from __future__ import annotations

from typing import Protocol

from foveamap.pipeline.records import Prediction, Scan


class SegmentationModel(Protocol):
    """Per-point semantic segmentation. ``raw_ids`` are already mapped through ``learning_map_inv``."""

    name: str

    def predict(self, scan: Scan) -> Prediction: ...
