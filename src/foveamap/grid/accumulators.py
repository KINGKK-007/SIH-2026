"""Integer-exact per-cell accumulators and fine->coarse reduction (README 6.5.4, tasks T5.3, T6.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameCounters:
    n_raw: int = 0
    n_invalid: int = 0
    n_in_grid: int = 0
    n_out_of_grid: int = 0
    n_z_saturated: int = 0


@dataclass
class RingAccumulators:
    """Dense arrays over one ring's full square (fields per README 6.5.4)."""

    ring: int
    cell_mm: int
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class GridAccumulators:
    rings: list[RingAccumulators]
    counters: FrameCounters


def reduce_block(acc: RingAccumulators, factor: int) -> RingAccumulators:
    """Sum counts/sums and take min/max over each ``factor x factor`` block."""
    raise NotImplementedError("Implemented in Phase 6, T6.1 (docs/PHASES.md).")
