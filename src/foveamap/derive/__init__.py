"""foveamap.derive — Derived geometric safety layers and synthetic hazard injection."""

from foveamap.derive.hazards import (
    HazardReport,
    inject_synthetic_hazards,
)
from foveamap.derive.layers import (
    DerivedLayers,
    compute_clearance,
    compute_derived_layers,
    compute_slope,
    compute_step_height,
    compute_traversability,
)

__all__ = [
    "DerivedLayers",
    "compute_slope",
    "compute_step_height",
    "compute_clearance",
    "compute_traversability",
    "compute_derived_layers",
    "HazardReport",
    "inject_synthetic_hazards",
]
