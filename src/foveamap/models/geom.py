"""Degraded fallback: ground vs non-ground only; always badged 'GEOM (degraded)' (README 6.3)."""

from __future__ import annotations

from foveamap.pipeline.records import Prediction, Scan


class GeomModel:
    """RANSAC/patchwise ground fit + Euclidean clustering. Never presented as deep-learning output."""

    name = "geom"

    def predict(self, scan: Scan) -> Prediction:
        raise NotImplementedError("Implemented in Phase 8 fallback (docs/PHASES.md).")
