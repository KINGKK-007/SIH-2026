"""Packed 12-byte per-cell layers derived from accumulators (README 6.6, task T6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foveamap.grid.accumulators import GridAccumulators

BYTES_PER_CELL = 12  # L10; asserted against the packed dtype in Phase 6


@dataclass
class GridLayers:
    """Per-ring packed layers: ground_z, top_z, clearance, cls, moving_frac, count, conf, flags."""

    rings: list[dict[str, np.ndarray]] = field(default_factory=list)


def finalize(acc: GridAccumulators, cfg: object) -> GridLayers:
    raise NotImplementedError("Implemented in Phase 6, T6.2 (docs/PHASES.md).")
