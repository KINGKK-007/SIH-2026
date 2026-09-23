"""Ring/grid specs, preset loading and V1-V5 validation (README 6.5.1-6.5.2, task T5.1)."""

from __future__ import annotations

from dataclasses import dataclass


class PresetError(ValueError):
    """Raised when a preset violates one of the rules V1-V5."""


@dataclass(frozen=True)
class RingSpec:
    r_min_mm: int
    r_max_mm: int
    cell_mm: int


@dataclass(frozen=True)
class GridSpec:
    name: str
    rings: tuple[RingSpec, ...]

    def logical_cells(self) -> int:
        raise NotImplementedError("Implemented in Phase 5, T5.1 (docs/PHASES.md).")

    def allocated_cells(self) -> int:
        raise NotImplementedError("Implemented in Phase 5, T5.1 (docs/PHASES.md).")

    def per_ring_shape(self) -> list[tuple[int, int]]:
        raise NotImplementedError("Implemented in Phase 5, T5.1 (docs/PHASES.md).")


def load_preset(name: str, cfg: object) -> GridSpec:
    """Build a :class:`GridSpec` from ``configs/grid.yaml`` and validate it."""
    raise NotImplementedError("Implemented in Phase 5, T5.1 (docs/PHASES.md).")


def validate_preset(spec: GridSpec) -> None:
    """Raise :class:`PresetError` naming the first violated rule (V1-V5)."""
    raise NotImplementedError("Implemented in Phase 5, T5.1 (docs/PHASES.md).")
