"""Server-side ring textures (README 13.3 texture convention, task T13.1)."""

from __future__ import annotations

import numpy as np

from foveamap.grid.layers import GridLayers


def ring_textures(layers: GridLayers, layer: str) -> list[np.ndarray]:
    """One RGBA image per ring; ``row = N_k - 1 - ix``, ``col = N_k - 1 - iy``."""
    raise NotImplementedError("Implemented in Phase 13, T13.1 (docs/PHASES.md).")
