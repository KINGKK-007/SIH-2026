"""Reference NumPy rasteriser, the correctness oracle (README 6.5.4, 6.5.7, task T5.3)."""

from __future__ import annotations

import numpy as np

from foveamap.grid.accumulators import (
    FrameCounters,
    GridAccumulators,
    make_ring,
    reduce_points,
)
from foveamap.grid.engine import locate
from foveamap.grid.presets import GridSpec


class NumpyBackend:
    """Accumulates already-valid, quantised points (``xyz_mm`` int32, z within int16) into every ring.

    Fills ``n_in_grid`` and ``n_out_of_grid``; the caller (``engine.rasterize``) owns ``n_raw``,
    ``n_invalid`` and ``n_z_saturated``.
    """

    name = "numpy"

    def rasterize(
        self,
        spec: GridSpec,
        xyz_mm: np.ndarray,
        super_cls: np.ndarray,
        moving: np.ndarray,
        conf: np.ndarray,
    ) -> GridAccumulators:
        xyz_mm = np.asarray(xyz_mm)
        n = len(xyz_mm)
        if not (len(super_cls) == len(moving) == len(conf) == n):
            raise ValueError(
                f"length mismatch: xyz {n}, super_cls {len(super_cls)}, moving {len(moving)}, conf {len(conf)}"
            )
        ring, ix, iy = locate(spec, xyz_mm[:, 0], xyz_mm[:, 1])
        inside = ring >= 0
        # One global key per point (ring offset + flat index) -> a single sort and reduction for all rings.
        sides = np.array([r.side for r in spec.rings], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(sides * sides)])
        k = ring[inside].astype(np.intp)
        keys = offsets[k] + iy[inside] * sides[k] + ix[inside]
        cells, reduced = reduce_points(
            keys, xyz_mm[inside, 2], super_cls[inside], moving[inside], conf[inside]
        )
        bounds = np.searchsorted(cells, offsets)
        rings = []
        for j, rs in enumerate(spec.rings):
            lo, hi = bounds[j], bounds[j + 1]
            part = {name: v[lo:hi] for name, v in reduced.items()}
            rings.append(make_ring(j, rs.cell_mm, rs.side, cells[lo:hi] - offsets[j], part))
        in_grid = int(inside.sum())
        return GridAccumulators(rings, FrameCounters(n_in_grid=in_grid, n_out_of_grid=n - in_grid), spec)
