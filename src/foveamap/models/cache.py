"""Cached predictions read from ``data/cache/pred/<model>/<seq>/<idx>.npz`` (README 6.3, R13)."""

from __future__ import annotations

from pathlib import Path

from foveamap.pipeline.records import Prediction, Scan


class CachedModel:
    """Reads cached ``raw_ids``/``conf`` arrays; needs no GPU."""

    def __init__(self, name: str, cache_dir: str | Path) -> None:
        raise NotImplementedError("Implemented in Phase 9, T9.2 (docs/PHASES.md).")

    def predict(self, scan: Scan) -> Prediction:
        raise NotImplementedError("Implemented in Phase 9, T9.2 (docs/PHASES.md).")
