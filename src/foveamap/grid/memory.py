"""Four-representation memory accounting (README 6.9, task T7.1)."""

from __future__ import annotations

from dataclasses import dataclass

from foveamap.grid.layers import GridLayers
from foveamap.grid.presets import GridSpec


@dataclass
class MemoryReport:
    dense3d_bytes: int
    sparse3d_bytes: int | None
    uniform25d_bytes: int
    fovea_bytes: int
    basis: str  # "logical" | "allocated"


def memory_report(spec: GridSpec, layers: GridLayers | None, cfg: object) -> MemoryReport:
    raise NotImplementedError("Implemented in Phase 7, T7.1 (docs/PHASES.md).")
