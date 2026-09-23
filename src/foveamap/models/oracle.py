"""Ground-truth labels as a model (README 9.3, R12)."""

from __future__ import annotations

from foveamap.pipeline.records import Prediction, Scan


class OracleModel:
    """Returns the scan's ground-truth semantic IDs with ``conf = 255``; moving IDs are preserved."""

    name = "oracle"

    def predict(self, scan: Scan) -> Prediction:
        raise NotImplementedError("Implemented in Phase 7, T7.2 (docs/PHASES.md).")
