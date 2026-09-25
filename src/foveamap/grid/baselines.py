"""Uniform baselines produced by the same engine as single-ring presets (README 6.5, task T7.1).

All baselines use ``spec_from_rings`` and the same ``rasterize`` / ``finalize`` path as the fovea preset;
they produce identical ``GridLayers`` objects and can be compared directly.
"""

from __future__ import annotations

from foveamap.grid.presets import GridSpec, spec_from_rings, validate_preset


def uniform_spec(cell_mm: int, extent_mm: int) -> GridSpec:
    """Build a single-ring :class:`GridSpec` with uniform ``cell_mm`` resolution up to ``extent_mm``.

    Both arguments must be positive integers and ``extent_mm`` must be divisible by ``cell_mm``
    (V4); these constraints are checked by :func:`validate_preset`.

    Args:
        cell_mm: Cell side length in millimetres (e.g. 50 for 5 cm, 200 for 20 cm).
        extent_mm: Half-side of the square grid in millimetres (e.g. 100_000 for ±100 m).

    Returns:
        A validated single-ring :class:`GridSpec` whose name is
        ``f"uniform_{cell_mm}mm"``.
    """
    if cell_mm <= 0:
        raise ValueError(f"cell_mm must be positive, got {cell_mm}")
    if extent_mm <= 0:
        raise ValueError(f"extent_mm must be positive, got {extent_mm}")
    name = f"uniform_{cell_mm}mm"
    spec = spec_from_rings(name, [(extent_mm, cell_mm)])
    validate_preset(spec)
    return spec
