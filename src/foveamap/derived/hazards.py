"""Seeded synthetic pothole/kerb/overhang injection (README 6.8, task T11.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foveamap.pipeline.records import ClassifiedScan


@dataclass
class Hazard:
    type: str
    center_xy_m: tuple[float, float]
    params: dict[str, float]


def inject_hazards(scan: ClassifiedScan, cfg: object, rng: np.random.Generator) -> tuple[ClassifiedScan, list[Hazard]]:
    raise NotImplementedError("Implemented in Phase 11, T11.7 (docs/PHASES.md).")
