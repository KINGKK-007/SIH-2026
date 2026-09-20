"""foveamap.grid.spec — GridSpec and Ring dataclasses.

These are the frozen, validated configuration objects for the variable-
resolution grid engine.  They are constructed from YAML config files
(``configs/grids/*.yaml``) and passed to the clipmap builder (Phase 2).

Alignment invariants (master plan §5.5)
----------------------------------------
I1. Integer nesting: each coarser ring's ``cell_m`` must be an exact integer
    multiple of every finer ring's ``cell_m``.  Validated at construction.
I2. Ring boundaries (``half_extent_m``) must be integer multiples of the
    outer ring's ``cell_m`` so no coarse cell straddles two rings.
    Validated at construction.

Examples
--------
>>> from foveamap.grid.spec import Ring, GridSpec
>>> rings = (
...     Ring(cell_m=0.05, half_extent_m=10.0),
...     Ring(cell_m=0.10, half_extent_m=30.0),
... )
>>> spec = GridSpec(rings=rings, z_range_m=(-3.0, 5.0))
>>> spec.total_cells()
200000
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Ring
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ring:
    """Specification of one ring in the nested clipmap grid.

    Parameters
    ----------
    cell_m : float
        Cell side length in metres (e.g. ``0.05`` for 5 cm).
    half_extent_m : float
        **Outer** half-extent (Chebyshev distance) of this ring in metres.
        The **inner** half-extent is the ``half_extent_m`` of the previous
        (finer) ring, or 0 for the innermost ring.

    Notes
    -----
    - Both values must be strictly positive.
    - ``half_extent_m / cell_m`` must be an integer — validated by
      :class:`GridSpec`.
    """

    cell_m: float
    half_extent_m: float

    def __post_init__(self) -> None:
        if self.cell_m <= 0:
            raise ValueError(f"Ring.cell_m must be > 0, got {self.cell_m}")
        if self.half_extent_m <= 0:
            raise ValueError(f"Ring.half_extent_m must be > 0, got {self.half_extent_m}")

    @property
    def side_cells(self) -> int:
        """Full side length of the bounding square in cells (2 × half-extent / cell)."""
        n = round(2.0 * self.half_extent_m / self.cell_m)
        return n

    def cells_in_square(self) -> int:
        """Total cells in the full bounding square (not the annulus)."""
        s = round(2.0 * self.half_extent_m / self.cell_m)
        return s * s


# ─────────────────────────────────────────────────────────────────────────────
# GridSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridSpec:
    """Complete specification of a nested foveated 2.5D grid.

    Parameters
    ----------
    rings : tuple[Ring, ...]
        Ordered from finest (innermost) to coarsest (outermost).
    z_range_m : tuple[float, float]
        Velodyne-frame z extent (min, max) used for height layer clipping.
    aggregation : str
        Cell class aggregation rule: ``"safety_priority"`` or ``"majority"``.
    max_range_m : float
        Maximum range considered; should equal the outermost ``half_extent_m``.
    min_obs_points : int
        Minimum points to declare a cell a STATIC_OBSTACLE (safety_priority).
    min_dyn_points : int
        Minimum points to declare a cell DYNAMIC (safety_priority).
    preset : str
        Human-readable preset name from the YAML file.
    ring_shape : str
        ``"square"`` (default) or ``"circular"``.

    Raises
    ------
    ValueError
        If any alignment invariant (I1 or I2) is violated.
    """

    rings: tuple[Ring, ...]
    z_range_m: tuple[float, float] = (-3.0, 5.0)
    aggregation: str = "safety_priority"
    max_range_m: float = 100.0
    min_obs_points: int = 2
    min_dyn_points: int = 2
    preset: str = "custom"
    ring_shape: str = "square"

    def __post_init__(self) -> None:
        if len(self.rings) == 0:
            raise ValueError("GridSpec requires at least one ring.")
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """Enforce alignment invariants I1 and I2."""
        rings = self.rings

        for i, ring in enumerate(rings):
            # I2: half_extent_m must be a multiple of cell_m for this ring
            ratio = ring.half_extent_m / ring.cell_m
            if abs(ratio - round(ratio)) > 1e-6:
                raise ValueError(
                    f"Invariant I2 violated for ring {i}: "
                    f"half_extent_m={ring.half_extent_m} is not an integer "
                    f"multiple of cell_m={ring.cell_m} "
                    f"(ratio={ratio:.6f})"
                )

        # I1: each coarser ring's cell must be an integer multiple of all finer ones
        finest_cell = rings[0].cell_m
        for i, ring in enumerate(rings[1:], start=1):
            ratio = ring.cell_m / finest_cell
            if abs(ratio - round(ratio)) > 1e-6:
                raise ValueError(
                    f"Invariant I1 violated for ring {i}: "
                    f"cell_m={ring.cell_m} is not an integer multiple of "
                    f"finest cell_m={finest_cell} "
                    f"(ratio={ratio:.6f})"
                )

        # I2 extension: ring boundary must be multiple of outer ring's cell
        for i in range(len(rings) - 1):
            boundary = rings[i].half_extent_m
            outer_cell = rings[i + 1].cell_m
            ratio = boundary / outer_cell
            if abs(ratio - round(ratio)) > 1e-6:
                raise ValueError(
                    f"Invariant I2 (boundary) violated between ring {i} and {i+1}: "
                    f"boundary={boundary} m is not a multiple of outer cell "
                    f"{outer_cell} m (ratio={ratio:.6f})"
                )

    # ── Cell count helpers ────────────────────────────────────────────────────

    def cells_in_ring(self, ring_idx: int) -> int:
        """Number of cells in the annulus of ring ``ring_idx``.

        For the innermost ring (idx=0) this equals the full square.
        For outer rings this is the full square minus the inner square.
        """
        ring = self.rings[ring_idx]
        outer_sq = ring.cells_in_square()
        if ring_idx == 0:
            return outer_sq
        inner_half = self.rings[ring_idx - 1].half_extent_m
        inner_n = round(2.0 * inner_half / ring.cell_m)
        return outer_sq - inner_n * inner_n

    def total_cells(self) -> int:
        """Total allocated cells across all rings."""
        return sum(self.cells_in_ring(i) for i in range(len(self.rings)))

    def memory_bytes(self, bytes_per_cell: int = 8) -> int:
        """Allocated memory for all rings at ``bytes_per_cell`` B/cell."""
        return self.total_cells() * bytes_per_cell

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GridSpec":
        """Build a :class:`GridSpec` from a parsed YAML dictionary.

        Parameters
        ----------
        d : dict
            Parsed contents of a ``configs/grids/*.yaml`` file.

        Returns
        -------
        GridSpec
        """
        rings = tuple(
            Ring(cell_m=float(r["cell_m"]), half_extent_m=float(r["half_extent_m"]))
            for r in d["rings"]
        )
        z_min, z_max = d.get("z_range_m", [-3.0, 5.0])
        return cls(
            rings=rings,
            z_range_m=(float(z_min), float(z_max)),
            aggregation=d.get("aggregation", "safety_priority"),
            max_range_m=float(d.get("max_range_m", rings[-1].half_extent_m)),
            min_obs_points=int(d.get("min_obs_points", 2)),
            min_dyn_points=int(d.get("min_dyn_points", 2)),
            preset=str(d.get("preset", "custom")),
            ring_shape=str(d.get("ring_shape", "square")),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "GridSpec":
        """Load a :class:`GridSpec` from a YAML config file.

        Parameters
        ----------
        yaml_path : str | Path
            Path to a ``configs/grids/*.yaml`` file.

        Returns
        -------
        GridSpec

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Grid config not found: {yaml_path}")
        with yaml_path.open() as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Return a human-readable summary table of the grid spec."""
        lines = [
            f"GridSpec preset={self.preset!r}  rings={len(self.rings)}  "
            f"total_cells={self.total_cells():,}  "
            f"memory={self.memory_bytes()/1024**2:.1f} MB @ 8 B/cell",
            f"  {'Ring':<5} {'cell_m':>8} {'half_extent_m':>14} {'cells_in_ring':>14}",
            f"  {'----':<5} {'------':>8} {'-------------':>14} {'-------------':>14}",
        ]
        for i, ring in enumerate(self.rings):
            lines.append(
                f"  {i:<5} {ring.cell_m:>8.2f} {ring.half_extent_m:>14.1f} "
                f"{self.cells_in_ring(i):>14,}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader: named presets
# ─────────────────────────────────────────────────────────────────────────────

#: Hard-coded fallback specs for every named preset.
#: Used when the YAML config files are not found (e.g., during testing).
_PRESET_DEFAULTS: dict[str, "GridSpec"] = {}  # populated below after class definition


def load_spec_from_preset(preset: str) -> "GridSpec":
    """Load a :class:`GridSpec` by preset name.

    Searches for ``configs/grids/<preset>.yaml`` relative to the current
    working directory and the package root. Falls back to hard-coded defaults
    if the file is not found.

    Parameters
    ----------
    preset : str
        One of: ``fovea_4ring``, ``ps_literal_2ring``, ``uniform_5cm``,
        ``uniform_20cm``.

    Returns
    -------
    GridSpec

    Raises
    ------
    ValueError
        If ``preset`` is unknown and no file is found.
    """
    # Search for YAML
    candidates = [
        Path("configs/grids") / f"{preset}.yaml",
        Path(__file__).parent.parent.parent.parent / "configs" / "grids" / f"{preset}.yaml",
    ]
    for c in candidates:
        if c.exists():
            return GridSpec.from_yaml(c)

    # Fall back to hard-coded defaults
    if not _PRESET_DEFAULTS:
        _init_preset_defaults()
    if preset in _PRESET_DEFAULTS:
        return _PRESET_DEFAULTS[preset]

    raise ValueError(
        f"Unknown preset {preset!r}. Valid: {list(_PRESET_DEFAULTS.keys())}. "
        "Also checked: " + ", ".join(str(c) for c in candidates)
    )


def _init_preset_defaults() -> None:
    """Populate _PRESET_DEFAULTS with hard-coded GridSpec objects."""
    _PRESET_DEFAULTS["fovea_4ring"] = GridSpec(
        rings=(
            Ring(cell_m=0.05, half_extent_m=10.0),
            Ring(cell_m=0.10, half_extent_m=30.0),
            Ring(cell_m=0.20, half_extent_m=60.0),
            Ring(cell_m=0.40, half_extent_m=100.0),
        ),
        z_range_m=(-3.0, 5.0),
        aggregation="safety_priority",
        preset="fovea_4ring",
    )
    _PRESET_DEFAULTS["ps_literal_2ring"] = GridSpec(
        rings=(
            Ring(cell_m=0.05, half_extent_m=10.0),
            Ring(cell_m=0.50, half_extent_m=100.0),
        ),
        z_range_m=(-3.0, 5.0),
        aggregation="safety_priority",
        preset="ps_literal_2ring",
    )
    _PRESET_DEFAULTS["uniform_5cm"] = GridSpec(
        rings=(Ring(cell_m=0.05, half_extent_m=100.0),),
        z_range_m=(-3.0, 5.0),
        aggregation="safety_priority",
        preset="uniform_5cm",
    )
    _PRESET_DEFAULTS["uniform_20cm"] = GridSpec(
        rings=(Ring(cell_m=0.20, half_extent_m=100.0),),
        z_range_m=(-3.0, 5.0),
        aggregation="safety_priority",
        preset="uniform_20cm",
    )

