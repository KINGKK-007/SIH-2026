"""Central-difference slope, masked where neighbours lack ground (README 6.7, task T11.2)."""

from __future__ import annotations

from foveamap.grid.layers import GridLayers


def compute_slope(layers: GridLayers, cfg: object) -> GridLayers:
    raise NotImplementedError("Implemented in Phase 11, T11.2 (docs/PHASES.md).")
