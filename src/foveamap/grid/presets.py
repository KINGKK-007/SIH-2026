"""Ring/grid specs, preset loading and V1-V5 validation (README 6.5.1-6.5.2, task T5.1).

All lengths are integer millimetres (L6). Ring ``k`` covers the half-open square ``[-R_k, R_k)^2`` minus
the inner square ``[-R_{k-1}, R_{k-1})^2`` and is stored as a full ``N_k x N_k`` array, ``N_k = 2 R_k / s_k``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from foveamap.config import GridConfig


class PresetError(ValueError):
    """Raised when a preset violates one of the rules V1-V5 (the message starts with the rule id)."""


@dataclass(frozen=True)
class RingSpec:
    r_min_mm: int
    r_max_mm: int
    cell_mm: int

    @property
    def side(self) -> int:
        """``N_k = 2 R_k / s_k``: cells per side of the ring's full square array."""
        return 2 * self.r_max_mm // self.cell_mm

    @property
    def offset(self) -> int:
        """``R_k / s_k``: index of the cell whose corner is the origin."""
        return self.r_max_mm // self.cell_mm


@dataclass(frozen=True)
class GridSpec:
    name: str
    rings: tuple[RingSpec, ...]

    @property
    def extent_mm(self) -> int:
        """``R_last``: the grid covers ``[-R_last, R_last)^2`` (L4)."""
        return self.rings[-1].r_max_mm

    @property
    def finest_cell_mm(self) -> int:
        return self.rings[0].cell_mm

    def logical_cells_per_ring(self) -> list[int]:
        """Cells in each annulus only: ``((2 R_k)^2 - (2 R_{k-1})^2) / s_k^2``."""
        return [
            ((2 * r.r_max_mm) ** 2 - (2 * r.r_min_mm) ** 2) // (r.cell_mm * r.cell_mm) for r in self.rings
        ]

    def allocated_cells_per_ring(self) -> list[int]:
        """Cells allocated when each ring is a full square array (hole included): ``(2 R_k / s_k)^2``."""
        return [r.side * r.side for r in self.rings]

    def logical_cells(self) -> int:
        return sum(self.logical_cells_per_ring())

    def allocated_cells(self) -> int:
        return sum(self.allocated_cells_per_ring())

    def per_ring_shape(self) -> list[tuple[int, int]]:
        return [(r.side, r.side) for r in self.rings]


def spec_from_rings(name: str, rings: Sequence[tuple[int, int]]) -> GridSpec:
    """Build a spec from ``(R_k, s_k)`` pairs in mm; inner radii follow from the previous ring."""
    out, inner = [], 0
    for r_max, cell in rings:
        out.append(RingSpec(inner, r_max, cell))
        inner = r_max
    return GridSpec(name, tuple(out))


def _is_int(v: object) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def validate_preset(spec: GridSpec, expected_extent_mm: int | None = None) -> None:
    """Raise :class:`PresetError` naming the first violated rule (V1-V5, README 6.5.2)."""
    rings = spec.rings
    where = f"preset {spec.name!r}"
    # V2 first: the other rules do integer arithmetic.
    for k, r in enumerate(rings):
        for field in ("r_min_mm", "r_max_mm", "cell_mm"):
            if not _is_int(getattr(r, field)):
                raise PresetError(
                    f"V2: {where} ring {k} {field}={getattr(r, field)!r} is not an integer (mm)"
                )
        if r.cell_mm <= 0:
            raise PresetError(f"V2: {where} ring {k} cell size must be a positive number of mm")
    if not rings:
        raise PresetError(f"V1: {where} has no rings")
    if rings[0].r_min_mm != 0:
        raise PresetError(f"V1: {where} ring 0 must start at 0, got r_min={rings[0].r_min_mm}")
    for k, r in enumerate(rings):
        if r.r_max_mm <= r.r_min_mm:
            raise PresetError(f"V1: {where} ring {k} is empty or out of order ({r.r_min_mm}..{r.r_max_mm})")
        if k and r.r_min_mm != rings[k - 1].r_max_mm:
            raise PresetError(
                f"V1: {where} ring {k} starts at {r.r_min_mm}, previous ends at {rings[k - 1].r_max_mm}"
            )
    for k in range(1, len(rings)):
        if rings[k].cell_mm % rings[k - 1].cell_mm:
            raise PresetError(
                f"V3: {where} ring {k} cell {rings[k].cell_mm} mm is not a multiple of ring {k - 1} cell "
                f"{rings[k - 1].cell_mm} mm"
            )
    for k, r in enumerate(rings):
        if r.r_max_mm % r.cell_mm:
            raise PresetError(f"V4: {where} R_{k}={r.r_max_mm} mm is not a multiple of s_{k}={r.cell_mm} mm")
        if r.r_min_mm % r.cell_mm:
            raise PresetError(
                f"V4: {where} inner boundary {r.r_min_mm} mm of ring {k} is not a multiple of "
                f"s_{k}={r.cell_mm} mm"
            )
    if expected_extent_mm is not None and spec.extent_mm != expected_extent_mm:
        raise PresetError(
            f"V5: {where} extent R_last={spec.extent_mm} mm != configured extent {expected_extent_mm} mm"
        )
    # V5 (origin on a cell corner) follows from V4: every R_k / s_k is an integer offset.


def load_preset(name: str, cfg: GridConfig) -> GridSpec:
    """Build a :class:`GridSpec` from ``configs/grid.yaml`` and validate it (V5 extent if active)."""
    if name not in cfg.presets:
        raise KeyError(f"unknown grid preset {name!r}; known: {sorted(cfg.presets)}")
    spec = spec_from_rings(name, [(ring.r_mm, ring.cell_mm) for ring in cfg.presets[name].rings])
    validate_preset(spec, expected_extent_mm=cfg.extent_mm if name == cfg.active_preset else None)
    return spec
