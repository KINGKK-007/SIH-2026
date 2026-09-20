"""foveamap.grid.clipmap — ClipmapGrid: the nested foveated 2.5D map.

``ClipmapGrid`` is the central data structure of FoveaMap.  It holds one dense
square array per ring, each ring forming an annulus that tiles the plane with
no overlap and no gap (invariants I1–I5, §5.5).

Usage (oracle mode)
-------------------
>>> import numpy as np
>>> from foveamap.grid.spec import GridSpec, Ring
>>> from foveamap.grid.clipmap import ClipmapGrid
>>> rings = (Ring(cell_m=0.05, half_extent_m=10.0),
...          Ring(cell_m=0.10, half_extent_m=30.0))
>>> spec = GridSpec(rings=rings, z_range_m=(-3.0, 5.0))
>>> pts = np.zeros((100, 4), dtype=np.float32)
>>> sc  = np.zeros(100, dtype=np.uint8)           # all DRIVABLE
>>> mv  = np.zeros(100, dtype=np.bool_)
>>> vru = np.zeros(100, dtype=np.bool_)
>>> cf  = np.ones(100, dtype=np.float16)
>>> grid = ClipmapGrid.build(pts, sc, mv, vru, cf, spec)
>>> grid.layer("count", 0).shape
(400, 400)
>>> s = grid.stats()
>>> s["allocated_cells"]
200000
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from foveamap.grid.spec import GridSpec, Ring
from foveamap.grid.layers import (
    LAYER_NAMES,
    LAYER_DTYPES,
    BYTES_PER_CELL,
    BYTES_PER_CELL_TARGET,
    make_ring_arrays,
    make_accum_arrays,
    apply_safety_priority,
    apply_majority,
)
from foveamap.grid.aggregate import scatter_reduce


# ─────────────────────────────────────────────────────────────────────────────
# ClipmapGrid
# ─────────────────────────────────────────────────────────────────────────────

class ClipmapGrid:
    """Nested foveated 2.5D map, built from a single classified LiDAR scan.

    This class stores one set of layer arrays per ring and exposes a clean
    accessor API.  It is intentionally *read-only* after construction; the
    factory method :meth:`build` is the only way to create one.

    Attributes
    ----------
    spec : GridSpec
        The grid specification used to build this map.
    n_rings : int
        Number of rings.
    n_points_assigned : int
        Total points assigned to any ring (invariant T1 check).
    n_points_dropped : int
        Points outside the grid's max_range or z_range.
    build_time_ms : float
        Time taken to build the grid (ms), excluding Numba JIT compilation.
    """

    def __init__(
        self,
        spec: GridSpec,
        ring_layer_arrays: list[dict[str, NDArray]],
        n_points_assigned: int,
        n_points_dropped: int,
        build_time_ms: float,
    ) -> None:
        self.spec = spec
        self._rings: list[dict[str, NDArray]] = ring_layer_arrays
        self.n_rings: int = len(ring_layer_arrays)
        self.n_points_assigned: int = n_points_assigned
        self.n_points_dropped: int = n_points_dropped
        self.build_time_ms: float = build_time_ms

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        points: NDArray[np.float32],
        super_cls: NDArray[np.uint8],
        moving: NDArray[np.bool_],
        vru: NDArray[np.bool_],
        conf: NDArray[np.float16],
        spec: GridSpec,
    ) -> "ClipmapGrid":
        """Bin a classified point cloud into the nested grid.

        Parameters
        ----------
        points : (N, 4) float32
            x, y, z, intensity in the Velodyne frame.
        super_cls : (N,) uint8
            Per-point super-class IDs (from :func:`~foveamap.io.labels.to_superclass`).
        moving : (N,) bool
            Per-point moving flag (raw ID ∈ 252–259 or motion-module output).
        vru : (N,) bool
            Per-point VRU flag (raw ID ∈ {30, 31, 32}).
        conf : (N,) float16
            Per-point confidence in [0, 1].
        spec : GridSpec
            Grid specification (rings, z_range, aggregation rule, thresholds).

        Returns
        -------
        ClipmapGrid

        Notes
        -----
        - Points outside ``spec.max_range_m`` (Chebyshev) or ``spec.z_range_m``
          are silently dropped.
        - Build time is measured and stored in :attr:`build_time_ms`.
        """
        t0 = time.perf_counter()

        points   = np.asarray(points,   dtype=np.float32)
        super_cls = np.asarray(super_cls, dtype=np.uint8)
        moving   = np.asarray(moving,   dtype=np.bool_)
        vru      = np.asarray(vru,      dtype=np.bool_)
        conf     = np.asarray(conf,     dtype=np.float16)

        z_min, z_max = spec.z_range_m
        rings = spec.rings
        n_rings = len(rings)

        # Allocate one set of layer + accumulation arrays per ring
        ring_layer_list: list[dict[str, NDArray]] = []
        ring_accum_list: list[dict[str, NDArray]] = []
        sides: list[int] = []

        for ring in rings:
            side = round(2.0 * ring.half_extent_m / ring.cell_m)
            sides.append(side)
            ring_layer_list.append(make_ring_arrays(side))
            ring_accum_list.append(make_accum_arrays(side))

        # Scatter-reduce each ring
        total_assigned = 0
        for r_idx, ring in enumerate(rings):
            inner_half = rings[r_idx - 1].half_extent_m if r_idx > 0 else 0.0
            n = scatter_reduce(
                points, super_cls, moving, vru, conf,
                ring_layer_list[r_idx],
                ring_accum_list[r_idx],
                ring.half_extent_m,
                ring.cell_m,
                inner_half,
                z_min, z_max,
                spec.min_dyn_points,
                spec.min_obs_points,
            )
            total_assigned += n

        # Apply aggregation rule
        for r_idx in range(n_rings):
            la = ring_layer_list[r_idx]
            ac = ring_accum_list[r_idx]
            # Merge accum into la for aggregation functions
            la.update(ac)
            if spec.aggregation == "majority":
                apply_majority(la)
            else:
                apply_safety_priority(
                    la,
                    min_dyn_points=spec.min_dyn_points,
                    min_obs_points=spec.min_obs_points,
                )
            # Remove internal accum keys from public array dict
            for key in list(la.keys()):
                if key.startswith("_"):
                    del la[key]

        n_total = len(points)
        n_dropped = n_total - total_assigned
        build_ms = (time.perf_counter() - t0) * 1000.0

        return cls(
            spec=spec,
            ring_layer_arrays=ring_layer_list,
            n_points_assigned=total_assigned,
            n_points_dropped=n_dropped,
            build_time_ms=build_ms,
        )

    # ── Layer accessor ────────────────────────────────────────────────────────

    def layer(self, name: str, ring: int = 0) -> NDArray:
        """Retrieve a 2-D layer array for a specific ring.

        Parameters
        ----------
        name : str
            One of: ``ground_z``, ``top_z``, ``overhang_z``, ``cls``,
            ``dyn_frac``, ``count``, ``conf``, ``flags``.
        ring : int
            Ring index (0 = innermost/finest).

        Returns
        -------
        NDArray
            Shape ``(side, side)`` where ``side = 2 × half_extent / cell_m``.

        Raises
        ------
        ValueError
            If ``name`` is unknown or ``ring`` is out of range.
        """
        if ring < 0 or ring >= self.n_rings:
            raise ValueError(f"Ring index {ring} out of range [0, {self.n_rings})")
        if name not in LAYER_NAMES:
            raise ValueError(f"Unknown layer {name!r}. Valid: {LAYER_NAMES}")
        return self._rings[ring][name]

    # ── Memory accounting (§5.7) ──────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Compute memory accounting statistics for this grid (E3).

        Returns
        -------
        dict with keys:
        - ``allocated_cells``  : int — total cells across all full squares
        - ``active_cells``     : int — cells in annular regions (no inner overlap)
        - ``allocated_bytes``  : int — allocated × BYTES_PER_CELL (actual)
        - ``active_bytes``     : int — active × BYTES_PER_CELL (actual)
        - ``allocated_bytes_target`` : int — allocated × 8 (master-plan target)
        - ``active_bytes_target``    : int — active × 8 (master-plan target)
        - ``allocated_mb``     : float — allocated_bytes / 1024²
        - ``active_mb``        : float — active_bytes / 1024²
        - ``rings``            : list[dict] — per-ring breakdown
        - ``build_time_ms``    : float
        - ``n_points_assigned``: int
        - ``n_points_dropped`` : int
        - ``bytes_per_cell``   : int — BYTES_PER_CELL (actual)
        """
        rings_info = []
        alloc_total = 0
        active_total = 0

        for r_idx, ring in enumerate(self.spec.rings):
            side = round(2.0 * ring.half_extent_m / ring.cell_m)
            alloc = side * side

            if r_idx == 0:
                active = alloc  # innermost: full square is active
            else:
                inner_half = self.spec.rings[r_idx - 1].half_extent_m
                inner_cells = round(2.0 * inner_half / ring.cell_m)
                active = alloc - inner_cells * inner_cells

            alloc_total  += alloc
            active_total += active

            # Count observed cells
            obs_mask = (self._rings[r_idx]["flags"] & 0b00000100) > 0
            if r_idx > 0:
                # Mask inner region (not physically populated)
                inner_half = self.spec.rings[r_idx - 1].half_extent_m
                inner_cells = round(2.0 * inner_half / ring.cell_m)
                pad = (side - inner_cells) // 2
                obs_mask[pad:pad + inner_cells, pad:pad + inner_cells] = False
            observed = int(np.sum(obs_mask))

            rings_info.append({
                "ring": r_idx,
                "cell_m": ring.cell_m,
                "half_extent_m": ring.half_extent_m,
                "side": side,
                "allocated_cells": alloc,
                "active_cells": active,
                "observed_cells": observed,
                "fill_rate": observed / active if active > 0 else 0.0,
            })

        return {
            "allocated_cells":       alloc_total,
            "active_cells":          active_total,
            "allocated_bytes":       alloc_total  * BYTES_PER_CELL,
            "active_bytes":          active_total * BYTES_PER_CELL,
            "allocated_bytes_target": alloc_total  * BYTES_PER_CELL_TARGET,
            "active_bytes_target":    active_total * BYTES_PER_CELL_TARGET,
            "allocated_mb":          alloc_total  * BYTES_PER_CELL / 1024**2,
            "active_mb":             active_total * BYTES_PER_CELL / 1024**2,
            "bytes_per_cell":        BYTES_PER_CELL,
            "bytes_per_cell_target": BYTES_PER_CELL_TARGET,
            "rings":                 rings_info,
            "build_time_ms":         self.build_time_ms,
            "n_points_assigned":     self.n_points_assigned,
            "n_points_dropped":      self.n_points_dropped,
        }

    # ── Convenience ──────────────────────────────────────────────────────────

    def ring_spec(self, ring: int) -> Ring:
        """Return the Ring specification for ring index ``ring``."""
        return self.spec.rings[ring]

    def side(self, ring: int) -> int:
        """Return the side length (cells) of ring ``ring``'s square."""
        r = self.spec.rings[ring]
        return round(2.0 * r.half_extent_m / r.cell_m)

    def cell_centres(self, ring: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return (X, Y) meshgrid arrays of cell centres in world coordinates.

        Parameters
        ----------
        ring : int

        Returns
        -------
        X, Y : NDArray[float32]
            Both shape ``(side, side)``.  X[row, col] is the world x-coordinate
            of the cell at row/col; Y likewise.
        """
        r = self.spec.rings[ring]
        side = self.side(ring)
        half = r.half_extent_m
        cell = r.cell_m
        coords = np.linspace(-half + 0.5 * cell, half - 0.5 * cell, side, dtype=np.float32)
        X, Y = np.meshgrid(coords, coords)
        return X, Y

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ClipmapGrid(rings={self.n_rings}, "
            f"active_cells={s['active_cells']:,}, "
            f"active_mb={s['active_mb']:.2f}, "
            f"assigned={self.n_points_assigned:,}, "
            f"build={self.build_time_ms:.1f}ms)"
        )

    # ── Uniform baseline factory ──────────────────────────────────────────────

    @classmethod
    def build_uniform(
        cls,
        points: NDArray[np.float32],
        super_cls: NDArray[np.uint8],
        moving: NDArray[np.bool_],
        vru: NDArray[np.bool_],
        conf: NDArray[np.float16],
        cell_m: float,
        half_extent_m: float,
        z_range_m: tuple[float, float] = (-3.0, 5.0),
        aggregation: str = "safety_priority",
    ) -> "ClipmapGrid":
        """Build a single-ring uniform baseline grid.

        Uses exactly the same engine as the foveated multi-ring build, so
        comparisons are like-for-like (master plan §5.5).

        Parameters
        ----------
        cell_m : float
            Cell side length in metres.
        half_extent_m : float
            Half-extent of the uniform square.
        z_range_m : tuple[float, float]
        aggregation : str
        """
        spec = GridSpec(
            rings=(Ring(cell_m=cell_m, half_extent_m=half_extent_m),),
            z_range_m=z_range_m,
            aggregation=aggregation,
        )
        return cls.build(points, super_cls, moving, vru, conf, spec)
