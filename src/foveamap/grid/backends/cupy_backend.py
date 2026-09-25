"""GPU-accelerated rasteriser using CuPy (README 6.5.7, ``foveamap.grid.backends.cupy_backend``).

Algorithm mirrors ``NumpyBackend.rasterize`` + ``engine.locate`` + ``accumulators.reduce_points`` but
runs the heavy work on the GPU:

  1. Upload xyz_mm, super_cls, moving, conf to GPU once.
  2. ``locate``: searchsorted over the 3-element radius table + floor_divide → ring / ix / iy.
  3. ``reduce_points``: argsort keys → argsort-based group, then bincount / reduceat for all fields.
  4. Download the *sparse* per-ring result vectors back to CPU.
  5. Build standard ``GridAccumulators`` (CPU numpy) as the NumpyBackend would.

The net effect is that only the dense point cloud crosses the PCIe bus (once, on the way up) and only
the small sparse accumulator arrays come back — a much smaller payload.

Requires CuPy ≥ 9 (reduceat on all ufuncs, flatnonzero, searchsorted, bincount on int64 keys).
``is_available()`` gates the import so the backend is silently skipped when CuPy is absent.
"""

from __future__ import annotations

import numpy as np

from foveamap.gpu_utils import is_cupy_available, to_numpy
from foveamap.grid.accumulators import (
    GROUND_CLASSES,
    INT32_MAX,
    INT32_MIN,
    N_CLASSES,
    OBSTACLE_CLASSES,
    FrameCounters,
    GridAccumulators,
    RingAccumulators,
    make_ring,
)
from foveamap.grid.presets import GridSpec


def is_available() -> bool:
    """True if CuPy is installed and a CUDA device is present."""
    return is_cupy_available()


class CupyBackend:
    """GPU-accelerated rasteriser; implements the ``GridBackend`` protocol."""

    name = "cupy"

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def rasterize(
        self,
        spec: GridSpec,
        xyz_mm: np.ndarray,
        super_cls: np.ndarray,
        moving: np.ndarray,
        conf: np.ndarray,
    ) -> GridAccumulators:
        """Upload → rasterise on GPU → download sparse results → return ``GridAccumulators``."""
        import cupy as cp

        n_pts = len(xyz_mm)

        # ── 1. Upload ──────────────────────────────────────────────────
        xyz_gpu = cp.asarray(xyz_mm, dtype=cp.int64)   # (N, 3)
        cls_gpu = cp.asarray(super_cls, dtype=cp.intp)
        mov_gpu = cp.asarray(np.asarray(moving, dtype=np.bool_), dtype=cp.bool_)
        conf_gpu = cp.asarray(conf, dtype=cp.float32)  # float64 is 32× slower on RTX 4050

        x, y, z_gpu = xyz_gpu[:, 0], xyz_gpu[:, 1], xyz_gpu[:, 2]

        # ── 2. locate: ring / ix / iy ──────────────────────────────────
        radii   = cp.array([r.r_max_mm for r in spec.rings], dtype=cp.int64)
        cell_sz = cp.array([r.cell_mm  for r in spec.rings], dtype=cp.int64)
        offsets = cp.array([r.offset   for r in spec.rings], dtype=cp.int64)

        # Smallest ring containing (x, y): searchsorted of max(|x|, |y|) over ring radii.
        a  = cp.maximum(cp.maximum(x, -x - 1), cp.maximum(y, -y - 1))
        k  = cp.searchsorted(radii, a, side="right")   # (N,) int64
        inside = k < len(radii)

        kk = cp.where(inside, k, 0).astype(cp.intp)
        ring_gpu = cp.where(inside, k, -1).astype(cp.int8)
        ix_gpu = cp.where(inside, cp.floor_divide(x, cell_sz[kk]) + offsets[kk], -1)
        iy_gpu = cp.where(inside, cp.floor_divide(y, cell_sz[kk]) + offsets[kk], -1)

        # ── 3. Global flat key per point ───────────────────────────────
        sides_gpu   = cp.array([r.side for r in spec.rings], dtype=cp.int64)
        ring_offsets = cp.concatenate([cp.zeros(1, dtype=cp.int64),
                                       cp.cumsum(sides_gpu * sides_gpu)])

        in_mask  = ring_gpu >= 0
        n_in     = int(in_mask.sum())
        n_out    = n_pts - n_in

        if n_in == 0:
            rings_cpu = self._empty_rings(spec)
            return GridAccumulators(rings_cpu, FrameCounters(n_in_grid=0, n_out_of_grid=n_pts), spec)

        kin     = ring_gpu[in_mask].astype(cp.intp)
        keys    = ring_offsets[kin] + iy_gpu[in_mask] * sides_gpu[kin] + ix_gpu[in_mask]
        z_in    = z_gpu[in_mask]
        cls_in  = cls_gpu[in_mask]
        mov_in  = mov_gpu[in_mask]
        conf_in = conf_gpu[in_mask]

        # ── 4. reduce_points on GPU ────────────────────────────────────
        cells_sorted, reduced = self._reduce_points_gpu(
            cp, keys, z_in, cls_in, mov_in, conf_in
        )

        # ── 5. Download sparse results → build rings ───────────────────
        cells_np   = to_numpy(cells_sorted)
        offsets_np = to_numpy(ring_offsets)      # (n_rings + 1,) on CPU
        reduced_np = {k: to_numpy(v) for k, v in reduced.items()}

        bounds = np.searchsorted(cells_np, offsets_np)
        rings_cpu = []
        for j, rs in enumerate(spec.rings):
            lo, hi = int(bounds[j]), int(bounds[j + 1])
            part = {name: v[lo:hi] for name, v in reduced_np.items()}
            rings_cpu.append(
                make_ring(j, rs.cell_mm, rs.side, cells_np[lo:hi] - offsets_np[j], part)
            )

        return GridAccumulators(
            rings_cpu,
            FrameCounters(n_in_grid=n_in, n_out_of_grid=n_out),
            spec,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _reduce_points_gpu(cp, keys, z_in, cls_in, mov_in, conf_in):
        """GPU equivalent of ``accumulators.reduce_points``.

        Returns ``(sorted_unique_keys, reduced_fields)`` — all CuPy arrays.
        ``reduced_fields`` values are int64 (same as the CPU version before ``make_ring`` casts them).
        """
        n = len(keys)
        if n == 0:
            empty = {
                "n_total": cp.zeros(0, dtype=cp.int64),
                "n_cls":   cp.zeros((0, N_CLASSES), dtype=cp.int64),
                "n_moving": cp.zeros(0, dtype=cp.int64),
                "g_sum":   cp.zeros(0, dtype=cp.int64),
                "g_min":   cp.zeros(0, dtype=cp.int64),
                "g_max":   cp.zeros(0, dtype=cp.int64),
                "o_min":   cp.zeros(0, dtype=cp.int64),
                "o_max":   cp.zeros(0, dtype=cp.int64),
                "conf_sum": cp.zeros(0, dtype=cp.int64),
            }
            return cp.zeros(0, dtype=cp.int64), empty

        # Sort by key
        order       = cp.argsort(keys)  # CuPy argsort is stable by default
        sorted_keys = keys[order]

        # Group boundary mask
        new_group = cp.concatenate([
            cp.ones(1, dtype=cp.bool_),
            sorted_keys[1:] != sorted_keys[:-1],
        ])
        starts_gpu  = cp.flatnonzero(new_group)                      # (m,)
        m           = int(starts_gpu.shape[0])

        # Group index for each element (same logic as the numpy path)
        group       = cp.empty(n, dtype=cp.intp)
        group[order] = cp.cumsum(new_group) - 1

        # Class mask shortcuts
        z_i32    = z_in.astype(cp.int32)
        ground   = (cls_in == GROUND_CLASSES[0]) | (cls_in == GROUND_CLASSES[1])
        obstacle = (cls_in == OBSTACLE_CLASSES[0]) | (cls_in == OBSTACLE_CLASSES[1])

        # Reorder for reduceat (must be in sorted-key order)
        zs  = z_i32[order]
        gs  = ground[order]
        obs = obstacle[order]

        # RTX 4050 (Ada Lovelace) has 32:1 FP32/FP64 ratio — use float32 everywhere.
        # Weighted bincount with float32 weights is ~32× faster than float64 on this GPU.
        def _bsum(weights):
            return cp.rint(
                cp.bincount(group, weights=weights.astype(cp.float32), minlength=m)
            ).astype(cp.int64)

        n_total  = cp.diff(cp.concatenate([starts_gpu, cp.array([n])])).astype(cp.int64)

        n_cls = (
            cp.bincount(group * N_CLASSES + cls_in, minlength=m * N_CLASSES)
            .reshape(m, N_CLASSES)
            .astype(cp.int64)
        )

        n_moving  = _bsum(mov_in.astype(cp.float32))
        g_sum     = _bsum(cp.where(ground, z_in.astype(cp.float32), cp.float32(0.0)))
        conf_sum  = _bsum(conf_in.astype(cp.float32))

        # min/max per group: CPU reduceat is faster than GPU lexsort chains for this
        # problem size (123K pts, ~50K groups) — download is ~0.1ms, reduceat is <1ms.
        starts_cpu  = to_numpy(starts_gpu)
        zs_cpu  = to_numpy(zs.astype(cp.int64))
        gs_cpu  = to_numpy(gs)
        obs_cpu = to_numpy(obs)
        g_min = np.minimum.reduceat(np.where(gs_cpu,  zs_cpu, INT32_MAX), starts_cpu)
        g_max = np.maximum.reduceat(np.where(gs_cpu,  zs_cpu, INT32_MIN), starts_cpu)
        o_min = np.minimum.reduceat(np.where(obs_cpu, zs_cpu, INT32_MAX), starts_cpu)
        o_max = np.maximum.reduceat(np.where(obs_cpu, zs_cpu, INT32_MIN), starts_cpu)

        reduced = {
            "n_total":  n_total,
            "n_cls":    n_cls,
            "n_moving": n_moving,
            "g_sum":    g_sum,
            "g_min":    g_min,
            "g_max":    g_max,
            "o_min":    o_min,
            "o_max":    o_max,
            "conf_sum": conf_sum,
        }
        return sorted_keys[starts_gpu], reduced

    @staticmethod
    def _empty_rings(spec: GridSpec) -> list[RingAccumulators]:
        from foveamap.grid.accumulators import DTYPES, FIELDS

        rings = []
        for j, rs in enumerate(spec.rings):
            values = {
                name: np.zeros((0, N_CLASSES) if name == "n_cls" else (0,), dtype=DTYPES[name])
                for name in FIELDS
            }
            rings.append(
                RingAccumulators(j, rs.cell_mm, rs.side, np.zeros(0, dtype=np.int64), **values)
            )
        return rings
