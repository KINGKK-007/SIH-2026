"""Four-representation memory accounting (README 6.9, task T7.1).

The four representations are:

1. **Dense 3D** (theoretical): voxel grid over ``[-R, R)^2 × [z_min, z_max)`` at the finest cell
   size, each voxel holding ``cfg.bytes_per_voxel`` bytes.  Never allocated in practice.
2. **Sparse 3D** (measured when layers are provided, else theoretical): only occupied cells from
   the finest ring, each key 8 bytes (int64) + 1 byte value = 9 bytes per unique voxel.
3. **Uniform 2.5D** (theoretical): ``uniform_5cm`` (``cell_mm = finest_cell_mm``) × 12 bytes per
   cell (``BYTES_PER_CELL``), using ``allocated_cells`` (the logical footprint for single-ring).
4. **FoveaMap** (measured when ``layers`` is provided, else theoretical ``allocated_cells × 12``):
   the packed ``GridLayers`` as returned by ``layers.nbytes()`` when layers are given.

None of the numbers are hard-coded (R1): everything flows from ``spec``, ``cfg``, and ``BYTES_PER_CELL``.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil

from foveamap.grid.baselines import uniform_spec
from foveamap.grid.layers import BYTES_PER_CELL, GridLayers
from foveamap.grid.presets import GridSpec

_SPARSE_KEY_BYTES = 8  # int64 flat index
_SPARSE_VAL_BYTES = 1  # minimal payload (e.g. occupancy flag)
_SPARSE_ENTRY_BYTES = _SPARSE_KEY_BYTES + _SPARSE_VAL_BYTES


@dataclass
class MemoryReport:
    """Memory in bytes for each of the four representations.

    Attributes:
        dense3d_bytes: Theoretical dense 3D voxel grid (never allocated in practice).
        sparse3d_bytes: Measured occupied voxels × 9 B when ``layers`` was provided;
            else a theoretical estimate using ``spec.rings[0]``'s logical cells.
        uniform25d_bytes: Theoretical uniform 2.5D at the finest cell size × 12 B.
        fovea_bytes: Measured ``layers.nbytes()`` when layers are provided; else
            ``spec.allocated_cells() × BYTES_PER_CELL``.
        basis: ``"allocated"`` when ``layers`` was supplied (measured),
            ``"theoretical"`` otherwise.
        rss_delta_bytes: RSS memory delta in bytes measured during ``memory_report``
            (always 0 in the theoretical path; may be 0 if psutil can't measure accurately).
    """

    dense3d_bytes: int
    sparse3d_bytes: int
    uniform25d_bytes: int
    fovea_bytes: int
    basis: str  # "theoretical" | "allocated"
    rss_delta_bytes: int = 0


def memory_report(spec: GridSpec, layers: GridLayers | None, cfg: object) -> MemoryReport:
    """Compute the four-representation memory report (README 6.9, invariant I9).

    Args:
        spec: The active :class:`GridSpec` (fovea or a baseline).
        layers: Packed ``GridLayers`` from :func:`~foveamap.grid.layers.finalize`; when provided
            the FoveaMap number is measured (``layers.nbytes()``), otherwise theoretical.
        cfg: ``GridConfig`` (provides ``z_range_mm``, ``bytes_per_voxel``,
            ``finest_cell_mm``-equivalent via ``spec``).

    Returns:
        A :class:`MemoryReport` with all four representations filled in.
    """
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    finest_cell_mm = spec.finest_cell_mm  # s_0, the innermost ring's cell size
    extent_mm = spec.extent_mm  # R_last, ±R in x and y

    # z range in mm (from cfg if GridConfig, else fall back to ±3 m / +5 m)
    try:
        z_min_mm, z_max_mm = cfg.z_range_mm  # type: ignore[union-attr]
        bytes_per_voxel = cfg.bytes_per_voxel  # type: ignore[union-attr]
    except AttributeError:
        z_min_mm, z_max_mm = -3_000, 5_000
        bytes_per_voxel = 1

    z_range_mm = z_max_mm - z_min_mm

    # 1. Dense 3D: (2R / s_0)^2 × (z_range / s_0) × bytes_per_voxel
    side_cells = (2 * extent_mm) // finest_cell_mm
    z_cells = z_range_mm // finest_cell_mm
    dense3d_bytes = int(side_cells) * int(side_cells) * int(z_cells) * int(bytes_per_voxel)

    # 2. Sparse 3D: theoretical = logical_cells(ring 0) * _SPARSE_ENTRY_BYTES
    #    (upper bound: every logical cell occupied by at least one point)
    #    If layers are provided we use ring 0's occupied count (the sparse set that was kept).
    if layers is not None:
        # ring 0 cells with non-zero count: count field > 0 after finalize
        ring0 = layers.rings[0]
        n_occupied = int((ring0["count"] > 0).sum())
        sparse3d_bytes = n_occupied * _SPARSE_ENTRY_BYTES
    else:
        # theoretical upper bound: all logical cells in ring 0 occupied
        sparse3d_bytes = spec.logical_cells_per_ring()[0] * _SPARSE_ENTRY_BYTES

    # 3. Uniform 2.5D: same extent, same finest cell → uniform_spec at finest_cell_mm
    uni_spec = uniform_spec(finest_cell_mm, extent_mm)
    uniform25d_bytes = uni_spec.allocated_cells() * BYTES_PER_CELL

    # 4. FoveaMap
    if layers is not None:
        fovea_bytes = layers.nbytes()
        basis = "allocated"
    else:
        fovea_bytes = spec.allocated_cells() * BYTES_PER_CELL
        basis = "theoretical"

    rss_after = proc.memory_info().rss
    rss_delta = max(0, rss_after - rss_before)

    return MemoryReport(
        dense3d_bytes=dense3d_bytes,
        sparse3d_bytes=sparse3d_bytes,
        uniform25d_bytes=uniform25d_bytes,
        fovea_bytes=fovea_bytes,
        basis=basis,
        rss_delta_bytes=rss_delta,
    )
