"""Wrapper around the selected pretrained range-view network (README 6.3, task T8.2)."""

from __future__ import annotations

from foveamap.pipeline.records import Prediction, Scan


class RangeViewModel:
    """Reuses the reference repo's own preprocessing and kNN post-processing."""

    def __init__(self, cfg: object) -> None:
        raise NotImplementedError("Implemented in Phase 8, T8.2 (docs/PHASES.md).")

    def predict(self, scan: Scan) -> Prediction:
        raise NotImplementedError("Implemented in Phase 8, T8.2 (docs/PHASES.md).")
