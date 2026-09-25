"""One-cell halo from adjacent rings (README 6.7, task T11.1).

Transfers boundary data between adjacent rings:
- finer -> coarser: via exact block reduction into the coarser ring's inner hole.
- coarser -> finer: via nearest-cell replication into the finer ring's outer boundary.
"""

from __future__ import annotations

import numpy as np

from foveamap.grid.layers import (
    EMPTY_CELL,
    FLAG_HAS_GROUND,
    FLAG_HAS_OBSTACLE,
    INT16_MIN,
    INT16_MAX,
    LAYER_DTYPE,
    GridLayers,
)
from foveamap.grid.presets import GridSpec


def _reduce_fine_grid(fine_grid: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a (H, W) fine ring array by an integer factor into (H // f, W // f)."""
    h, w = fine_grid.shape
    new_h = h // factor
    new_w = w // factor
    reshaped = (
        fine_grid.reshape(new_h, factor, new_w, factor)
        .swapaxes(1, 2)
        .reshape(new_h, new_w, factor * factor)
    )

    out = np.empty((new_h, new_w), dtype=LAYER_DTYPE)
    out[:] = EMPTY_CELL

    valid_ground = (reshaped["flags"] & FLAG_HAS_GROUND) != 0
    ground_cnt = valid_ground.sum(axis=-1)
    has_g = ground_cnt > 0
    safe_cnt = np.where(has_g, ground_cnt, 1)

    gz_sum = np.where(valid_ground, reshaped["ground_z"].astype(np.int64), 0).sum(axis=-1)
    reduced_gz = np.where(has_g, np.floor_divide(gz_sum + safe_cnt // 2, safe_cnt), INT16_MIN)
    out["ground_z"] = np.clip(reduced_gz, INT16_MIN, INT16_MAX).astype(np.int16)

    valid_obs = (reshaped["flags"] & FLAG_HAS_OBSTACLE) != 0
    has_obs = valid_obs.any(axis=-1)
    top_z_max = np.where(valid_obs, reshaped["top_z"].astype(np.int64), INT16_MIN).max(axis=-1)
    out["top_z"] = np.where(has_obs, np.clip(top_z_max, INT16_MIN, INT16_MAX), INT16_MIN).astype(np.int16)

    # Class: take the center cell or most common
    center_idx = (factor * factor) // 2
    out["cls"] = reshaped["cls"][:, :, center_idx]
    total_cnt = reshaped["count"].astype(np.int64).sum(axis=-1)
    out["count"] = np.minimum(total_cnt, 65535).astype(np.uint16)
    out["conf"] = reshaped["conf"][:, :, center_idx]

    flags = np.zeros((new_h, new_w), dtype=np.uint8)
    flags |= (has_g * FLAG_HAS_GROUND).astype(np.uint8)
    flags |= (has_obs * FLAG_HAS_OBSTACLE).astype(np.uint8)
    out["flags"] = flags

    return out


def compute_halo(layers: GridLayers, cfg: object | None = None) -> GridLayers:
    """Populate inner holes of coarser rings from adjacent finer rings (fine -> coarse)."""
    if cfg is not None and hasattr(cfg, "halo") and not cfg.halo:
        return layers

    spec: GridSpec | None = layers.spec
    if spec is None or len(layers.rings) <= 1:
        return layers

    new_rings = [r.copy() for r in layers.rings]

    for k in range(1, len(new_rings)):
        r_fine = spec.rings[k - 1]
        r_coarse = spec.rings[k]
        factor = r_coarse.cell_mm // r_fine.cell_mm

        reduced = _reduce_fine_grid(new_rings[k - 1], factor)

        hole_x0 = r_coarse.offset - r_fine.r_max_mm // r_coarse.cell_mm
        hole_x1 = r_coarse.offset + r_fine.r_max_mm // r_coarse.cell_mm
        hole_y0 = r_coarse.offset - r_fine.r_max_mm // r_coarse.cell_mm
        hole_y1 = r_coarse.offset + r_fine.r_max_mm // r_coarse.cell_mm

        # Only overwrite cells that do not already have ground
        coarse_target = new_rings[k][hole_y0:hole_y1, hole_x0:hole_x1]
        mask_to_fill = (coarse_target["flags"] & FLAG_HAS_GROUND) == 0
        coarse_target[mask_to_fill] = reduced[mask_to_fill]

    return GridLayers(spec=spec, rings=new_rings)


def extract_padded_ring(
    layers: GridLayers, ring_idx: int, cfg: object | None = None
) -> np.ndarray:
    """Return an (N_k + 2, N_k + 2) array of ring_idx with outer border filled from ring_idx + 1."""
    ring = layers.rings[ring_idx]
    side = ring.shape[0]
    padded = np.empty((side + 2, side + 2), dtype=LAYER_DTYPE)
    padded[:] = EMPTY_CELL
    padded[1:-1, 1:-1] = ring

    use_halo = True if cfg is None or not hasattr(cfg, "halo") else cfg.halo
    spec = layers.spec

    if use_halo and spec is not None and ring_idx + 1 < len(layers.rings):
        r_curr = spec.rings[ring_idx]
        r_next = spec.rings[ring_idx + 1]
        next_ring = layers.rings[ring_idx + 1]

        # Calculate coordinates for padded array indices [0..side+1]
        xs_mm = (np.arange(side + 2, dtype=np.int64) - 1 - r_curr.offset) * r_curr.cell_mm + r_curr.cell_mm // 2
        ys_mm = (np.arange(side + 2, dtype=np.int64) - 1 - r_curr.offset) * r_curr.cell_mm + r_curr.cell_mm // 2

        ix_next_all = np.floor_divide(xs_mm, r_next.cell_mm) + r_next.offset  # (side+2,)
        iy_next_all = np.floor_divide(ys_mm, r_next.cell_mm) + r_next.offset  # (side+2,)
        valid_ix = (ix_next_all >= 0) & (ix_next_all < r_next.side)
        valid_iy = (iy_next_all >= 0) & (iy_next_all < r_next.side)

        # Outer border rows (0 and side + 1) — vectorised over px
        for py in (0, side + 1):
            y_mm = ys_mm[py]
            iy_next = int(np.floor_divide(y_mm, r_next.cell_mm) + r_next.offset)
            if 0 <= iy_next < r_next.side:
                px_valid = np.where(valid_ix)[0]
                if len(px_valid):
                    padded[py, px_valid] = next_ring[iy_next, ix_next_all[px_valid]]

        # Outer border columns (0 and side + 1) — vectorised over py
        for px in (0, side + 1):
            x_mm = xs_mm[px]
            ix_next = int(np.floor_divide(x_mm, r_next.cell_mm) + r_next.offset)
            if 0 <= ix_next < r_next.side:
                py_valid = np.where(valid_iy)[0]
                if len(py_valid):
                    padded[py_valid, px] = next_ring[iy_next_all[py_valid], ix_next]

    return padded
