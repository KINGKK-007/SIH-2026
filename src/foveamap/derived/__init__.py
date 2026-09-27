"""Derived geometric layers and synthetic hazards (README 6.7, 6.8, 9.6)."""

from __future__ import annotations

from foveamap.derived.clearance import compute_clearance
from foveamap.derived.halo import compute_halo, extract_padded_ring
from foveamap.derived.slope import compute_slope, compute_slope_deg
from foveamap.derived.step import compute_step, compute_step_height_mm
from foveamap.derived.traversability import (
    NON_TRAVERSABLE_CLASS,
    TRAVERSABLE_CLASS,
    UNKNOWN_CLASS,
    compute_traversability,
    traversability_map,
)
from foveamap.derived.zoom import render_zoom
from foveamap.grid.layers import GridLayers


def compute_derived_layers(layers: GridLayers, cfg: object) -> GridLayers:
    """Run all derived geometry stages in order: halo -> slope -> step -> clearance -> traversability.

    When ``cfg.fast_mode`` is True, halo/slope/clearance are skipped.  Step detection
    (kerbs + potholes) still runs — it is a pure neighbour-difference over int arrays
    and takes < 2 ms, but provides FLAG_KERB and FLAG_POTHOLE that are judicial
    requirements (problem statement: 'curbs, potholes, or overhanging obstacles').
    """
    fast = getattr(cfg, "fast_mode", False)
    if not fast:
        layers = compute_halo(layers, cfg)
        layers = compute_slope(layers, cfg)
        layers = compute_clearance(layers, cfg)
    # Step/kerb + pothole always runs — fast even without halo padding
    layers = compute_step(layers, cfg)
    layers = compute_traversability(layers, cfg)
    return layers


__all__ = [
    "compute_derived_layers",
    "compute_halo",
    "extract_padded_ring",
    "compute_slope",
    "compute_slope_deg",
    "compute_step",
    "compute_step_height_mm",
    "compute_clearance",
    "compute_traversability",
    "traversability_map",
    "render_zoom",
    "TRAVERSABLE_CLASS",
    "NON_TRAVERSABLE_CLASS",
    "UNKNOWN_CLASS",
]
