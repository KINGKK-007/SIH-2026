"""Cached predictions read from ``data/cache/pred/<model>/<seq>/<idx>.npz`` (README 6.3, R13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from foveamap.pipeline.records import Prediction, Scan


class CachedModel:
    """Reads cached ``raw_ids``/``conf`` arrays written by ``scripts/cache_predictions.py``; needs no GPU.

    On disk ``raw_ids`` is ``uint8`` (every value LSK3DNet or a range-view model can produce is a static
    class, raw ID <= 99); :class:`~foveamap.pipeline.records.Prediction` widens it to ``uint16`` (D-014)
    so it shares a dtype with the oracle path, which can carry moving IDs 252-259.
    """

    def __init__(self, name: str, cache_dir: str | Path) -> None:
        self.name = name
        self.cache_dir = Path(cache_dir) / name

    def _path(self, scan: Scan) -> Path:
        return self.cache_dir / scan.seq / f"{scan.idx:06d}.npz"

    def predict(self, scan: Scan) -> Prediction:
        path = self._path(scan)
        if not path.is_file():
            raise FileNotFoundError(
                f"no cached prediction at {path}. Run scripts/cache_predictions.py "
                f"--model {self.name} --sequence {scan.seq} first (R13: predictions are cached, not "
                "recomputed live)."
            )
        with np.load(path) as npz:
            raw_ids = npz["raw_ids"]
            conf = npz["conf"]
        if raw_ids.shape != conf.shape or raw_ids.shape != (len(scan.xyz),):
            raise ValueError(
                f"{path}: cached arrays (raw_ids {raw_ids.shape}, conf {conf.shape}) do not match the "
                f"scan's point count ({len(scan.xyz)}); the cache is stale or the wrong sequence/frame."
            )
        return Prediction(raw_ids=raw_ids.astype(np.uint16), conf=conf.astype(np.uint8))
