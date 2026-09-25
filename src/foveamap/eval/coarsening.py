"""Preliminary cost-of-coarsening prototype (README 12.4, task T7.3).

For each preset, we rasterise sampled scans of the oracle-labelled sequence with ``locate``
to back-project each point to its cell centre, then record the horizontal distance error.
This is done per distance bucket (eval.buckets) so we can see how coarsening degrades
accuracy at close vs far range.

This is the evaluation *plumbing* only; full benchmark numbers come in Phase 12.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from foveamap.eval.buckets import bucket_of_range
from foveamap.grid.baselines import uniform_spec
from foveamap.grid.engine import cell_to_center_mm, locate, rasterize
from foveamap.grid.layers import finalize
from foveamap.grid.presets import GridSpec, load_preset
from foveamap.io.labels import raw_to_super
from foveamap.io.sequence import Sequence
from foveamap.models.oracle import OracleModel

_BUCKET_LABELS = ("0-10 m", "10-30 m", "30-60 m", "60-100 m")


def _back_projection_error_mm(
    spec: GridSpec, xyz_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point to its cell centre and return (error_mm, bucket).

    The error is the Euclidean horizontal distance (x-y plane only) between the point
    and the centre of the cell it was placed in.  Out-of-grid points get error=-1.

    Args:
        spec: The :class:`GridSpec` to use.
        xyz_mm: ``(N, 3)`` int32 millimetre coordinates.

    Returns:
        ``(error_mm, bucket)`` each ``(N,)`` float32; error=-1 for out-of-grid points.
    """
    x_mm = xyz_mm[:, 0].astype(np.int64)
    y_mm = xyz_mm[:, 1].astype(np.int64)
    ring_idx, ix, iy = locate(spec, x_mm, y_mm)

    in_grid = ring_idx >= 0
    error_mm = np.full(len(xyz_mm), -1.0, dtype=np.float32)

    for k, rs in enumerate(spec.rings):
        mask_k = in_grid & (ring_idx == k)
        if not mask_k.any():
            continue
        ixk = ix[mask_k].astype(np.intp)
        iyk = iy[mask_k].astype(np.intp)
        # Vectorised: cell_to_center_mm for all cells in this ring
        # cx = (ixk - rs.offset) * rs.cell_mm + rs.cell_mm // 2
        cx = (ixk - rs.offset) * rs.cell_mm + rs.cell_mm // 2
        cy = (iyk - rs.offset) * rs.cell_mm + rs.cell_mm // 2
        dx = (x_mm[mask_k] - cx).astype(np.float32)
        dy = (y_mm[mask_k] - cy).astype(np.float32)
        error_mm[mask_k] = np.sqrt(dx * dx + dy * dy)

    # Bucket by horizontal range in metres
    r_m = np.sqrt((x_mm / 1000.0) ** 2 + (y_mm / 1000.0) ** 2).astype(np.float32)
    bucket = bucket_of_range(r_m)

    return error_mm, bucket


def cost_of_coarsening(
    data_root: str | Path,
    sequences: list[str],
    cfg: object,
    preset_names: list[str] | None = None,
    n_frames: int = 50,
    seed: int = 1337,
) -> dict:
    """Back-projection accuracy per distance bucket for several presets (T7.3).

    Computes the mean, median, and p90 horizontal cell-centre error (mm) for
    ``n_frames`` randomly sampled frames from each sequence.

    Args:
        data_root: Path to the SemanticKITTI dataset root.
        sequences: Sequence IDs to sample from (e.g. ``["08"]``).
        cfg: :class:`~foveamap.config.GridConfig` or :class:`~foveamap.config.FoveaConfig`.
        preset_names: Preset names to evaluate. Defaults to
            ``["fovea_default", "ps_literal", "uniform_5cm", "uniform_20cm"]``.
        n_frames: Number of frames to sample per sequence.
        seed: Random seed for frame sampling.

    Returns:
        A JSON-serialisable dict keyed by preset name → bucket label → statistics.
    """
    try:
        grid_cfg = cfg.grid  # type: ignore[union-attr]
    except AttributeError:
        grid_cfg = cfg

    if preset_names is None:
        preset_names = ["fovea_default", "ps_literal", "uniform_5cm", "uniform_20cm"]

    specs: dict[str, GridSpec] = {}
    for name in preset_names:
        try:
            specs[name] = load_preset(name, grid_cfg)
        except KeyError:
            if name.startswith("uniform_"):
                # e.g. "uniform_5cm" → cell_mm=50
                parts = name.replace("uniform_", "").replace("cm", "")
                cell_mm = int(float(parts) * 10)
                specs[name] = uniform_spec(cell_mm, grid_cfg.extent_mm)
            else:
                raise

    model = OracleModel()
    rng = np.random.default_rng(seed)

    # Accumulate errors per preset per bucket
    buckets_data: dict[str, dict] = {name: {} for name in specs}

    for seq_id in sequences:
        try:
            seq = Sequence(data_root, seq_id)
        except FileNotFoundError:
            continue
        n_total = seq.n_frames_total
        chosen = rng.choice(n_total, size=min(n_frames, n_total), replace=False).tolist()

        for frame_idx in chosen:
            scan = seq.load_frame(int(frame_idx))
            pred = model.predict(scan)

            # Quantise to mm (same as rasterize does internally)
            xyz_mm = np.rint(np.asarray(scan.xyz, dtype=np.float64) * 1000.0).astype(np.int32)

            for preset_name, spec in specs.items():
                err_mm, bucket = _back_projection_error_mm(spec, xyz_mm)
                valid = err_mm >= 0
                for b in range(4):
                    b_mask = valid & (bucket == b)
                    if not b_mask.any():
                        continue
                    bucket_key = _BUCKET_LABELS[b]
                    stats = buckets_data[preset_name].setdefault(
                        bucket_key, {"n": 0, "sum_mm": 0.0, "_sample": []}
                    )
                    errs = err_mm[b_mask]
                    stats["n"] += int(len(errs))
                    stats["sum_mm"] += float(errs.sum())
                    # Keep a capped sample for percentile computation
                    cap = 100_000
                    have = len(stats["_sample"])
                    if have < cap:
                        stats["_sample"].extend(errs[: cap - have].tolist())

    # Summarise
    out: dict[str, dict] = {}
    for preset_name, bdata in buckets_data.items():
        out[preset_name] = {}
        for bucket_key, stats in bdata.items():
            n = stats["n"]
            mean_mm = stats["sum_mm"] / n if n > 0 else float("nan")
            arr = np.array(stats["_sample"], dtype=np.float32) if stats["_sample"] else np.array([])
            median_mm = float(np.median(arr)) if arr.size > 0 else float("nan")
            p90_mm = float(np.percentile(arr, 90)) if arr.size > 0 else float("nan")
            out[preset_name][bucket_key] = {
                "n": n,
                "mean_mm": round(mean_mm, 3),
                "median_mm": round(median_mm, 3),
                "p90_mm": round(p90_mm, 3),
            }

    return out


def latency_report_grid(
    data_root: str | Path,
    sequence: str,
    cfg: object,
    preset_name: str | None = None,
    n_frames: int = 100,
    seed: int = 42,
) -> dict:
    """Grid-stage wall-clock timing (Gate 7: results written to ``results/latency_grid_prelim.json``).

    Measures ``rasterize`` + ``finalize`` time over ``n_frames`` frames.

    Args:
        data_root: SemanticKITTI root.
        sequence: Sequence ID.
        cfg: GridConfig or FoveaConfig.
        preset_name: Grid preset to measure; defaults to ``active_preset``.
        n_frames: Number of frames to time.
        seed: Random seed for frame selection.

    Returns:
        JSON-serialisable dict with mean/p50/p95/p99/max latencies and a real-time verdict.
    """
    try:
        grid_cfg = cfg.grid  # type: ignore[union-attr]
    except AttributeError:
        grid_cfg = cfg

    pname = preset_name or grid_cfg.active_preset
    spec = load_preset(pname, grid_cfg)
    model = OracleModel()
    seq = Sequence(data_root, sequence)
    rng = np.random.default_rng(seed)
    n_total = seq.n_frames_total
    chosen = rng.choice(n_total, size=min(n_frames, n_total), replace=False).tolist()

    timings_ms = []
    for frame_idx in chosen:
        scan = seq.load_frame(int(frame_idx))
        pred = model.predict(scan)
        super_cls, moving = raw_to_super(pred.raw_ids)
        conf = pred.conf

        t0 = time.perf_counter_ns()
        acc = rasterize(spec, scan.xyz, super_cls, moving, conf, min_range_mm=grid_cfg.min_range_mm)
        finalize(acc, grid_cfg, spec)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        timings_ms.append(elapsed_ms)

    arr = np.array(timings_ms, dtype=np.float64)
    sensor_period_ms = 100.0  # 10 Hz (L3)
    p95 = float(np.percentile(arr, 95))
    return {
        "preset": pname,
        "sequence": sequence,
        "n_frames": len(timings_ms),
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(arr.max()), 2),
        "sensor_period_ms": sensor_period_ms,
        "real_time_verdict": "PASS" if p95 < sensor_period_ms else "FAIL (p95 > 100 ms; see T7.5)",
    }
