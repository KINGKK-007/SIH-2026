"""Frame-alignment verification (PHASES T4.1): does ``relative_transform`` register consecutive scans?

For random consecutive pairs, scan ``i-1`` is transformed into the Velodyne frame of scan ``i`` with
``T_vel_i_from_vel_(i-1)`` (README 5.3) and compared with scan ``i`` by nearest-neighbour distance.

Three measurements (D-018):

* ``spec`` — static structure as written in T4.1 (building, road, vegetation, < 30 m). Kept and gated,
  but on its own it is *not* discriminative: flat-road rings move with the sensor and facades run
  parallel to the motion, so even "no transform at all" scores a few centimetres.
* ``compact`` — poles, trunks and traffic signs (< 30 m). Compact in x-y, so any translation or rotation
  error shows up at full size. This is the measurement that actually tests the transform.
* ``identity control`` — the compact measurement with no transform, on pairs where the vehicle moved
  at least ``min_motion_m``. It must fail the threshold, proving the check can detect an error.

Gate 4 passes only if spec median < 0.15 m, compact median < 0.15 m and the control fails.
"""

from __future__ import annotations

from collections.abc import Sequence as SeqType
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from foveamap.eval.buckets import horizontal_range
from foveamap.io.labels import semantic_ids
from foveamap.io.poses import transform_points
from foveamap.io.sequence import Sequence
from foveamap.pipeline.records import Scan

SPEC_CLASSES_RAW = (50, 40, 70)  # building, road, vegetation (T4.1 as written)
COMPACT_CLASSES_RAW = (80, 71, 81)  # pole, trunk, traffic-sign (D-018)
GATE_MEDIAN_M = 0.15


def _static_points(scan: Scan, classes: SeqType[int], max_range_m: float) -> np.ndarray:
    if scan.raw_labels is None:
        raise ValueError("alignment check needs labels")
    keep = np.isin(semantic_ids(scan.raw_labels), classes) & (horizontal_range(scan.xyz) < max_range_m)
    return scan.xyz[keep].astype(np.float64)


def nn_distances(
    prev: Scan, cur: Scan, T: np.ndarray, classes: SeqType[int], max_range_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NN distances from ``prev`` (moved by ``T`` into ``cur``'s frame) to ``cur``, restricted to ``classes``.

    Returns ``(distances, prev_in_cur, cur_points)``.
    """
    prev_pts = transform_points(T, _static_points(prev, classes, max_range_m))
    cur_pts = _static_points(cur, classes, max_range_m)
    if len(prev_pts) == 0 or len(cur_pts) == 0:
        return np.zeros(0), prev_pts, cur_pts
    distances, _ = cKDTree(cur_pts).query(prev_pts, k=1)
    return distances, prev_pts, cur_pts


def pair_distances(
    seq: Sequence,
    i: int,
    classes: SeqType[int] = COMPACT_CLASSES_RAW,
    max_range_m: float = 30.0,
    transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`nn_distances` for the pair ``(i-1, i)``; ``transform`` overrides the pose-derived one."""
    T = seq.relative_transform(i, i - 1) if transform is None else transform
    return nn_distances(seq.load_frame(i - 1), seq.load_frame(i), T, classes, max_range_m)


def _summary(d: np.ndarray) -> dict[str, float]:
    if len(d) == 0:
        return {"median_m": float("nan"), "p90_m": float("nan"), "n": 0}
    return {"median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)), "n": int(len(d))}


def alignment_check(
    root: str | Path,
    sequences: SeqType[str] = ("04", "08"),
    n_pairs: int = 50,
    seed: int = 1337,
    max_range_m: float = 30.0,
    threshold_m: float = GATE_MEDIAN_M,
    min_motion_m: float = 0.3,
    plot_dir: str | Path | None = None,
    n_plots: int = 2,
) -> dict[str, Any]:
    """Run T4.1 over random consecutive pairs; returns the ``results/frame_alignment.json`` dict."""
    rng = np.random.default_rng(seed)
    pairs: list[dict[str, Any]] = []
    spec_all: list[np.ndarray] = []
    compact_all: list[np.ndarray] = []
    control_all: list[np.ndarray] = []
    for name in sequences:
        seq = Sequence(root, name)
        n = seq.n_frames_total
        frames = np.sort(rng.choice(np.arange(1, n), size=min(n_pairs, n - 1), replace=False))
        for k, i in enumerate(frames.tolist()):
            prev, cur = seq.load_frame(i - 1), seq.load_frame(i)
            T = seq.relative_transform(i, i - 1)
            motion = float(np.linalg.norm(T[:3, 3]))
            d_spec, _, _ = nn_distances(prev, cur, T, SPEC_CLASSES_RAW, max_range_m)
            d_compact, prev_pts, cur_pts = nn_distances(prev, cur, T, COMPACT_CLASSES_RAW, max_range_m)
            d_control, _, _ = nn_distances(prev, cur, np.eye(4), COMPACT_CLASSES_RAW, max_range_m)
            spec_all.append(d_spec)
            compact_all.append(d_compact)
            if motion >= min_motion_m:
                control_all.append(d_control)
            pairs.append(
                {
                    "sequence": name,
                    "frame": i,
                    "ego_translation_m": motion,
                    "spec_median_m": _summary(d_spec)["median_m"],
                    "compact_median_m": _summary(d_compact)["median_m"],
                    "compact_n": int(len(d_compact)),
                    "identity_compact_median_m": _summary(d_control)["median_m"],
                }
            )
            if plot_dir is not None and k < n_plots:
                spec_prev = transform_points(T, _static_points(prev, SPEC_CLASSES_RAW, max_range_m))
                plot_alignment(
                    np.vstack([spec_prev, prev_pts]),
                    np.vstack([_static_points(cur, SPEC_CLASSES_RAW, max_range_m), cur_pts]),
                    Path(plot_dir) / f"frame_alignment_{name}_{i:06d}.png",
                    f"seq {name}: scan {i - 1} -> {i}  compact median {np.median(d_compact):.3f} m",
                )

    def pooled(parts: list[np.ndarray]) -> dict[str, float]:
        return _summary(np.concatenate(parts) if parts else np.zeros(0))

    spec, compact, control = pooled(spec_all), pooled(compact_all), pooled(control_all)
    spec_ok = spec["n"] > 0 and spec["median_m"] < threshold_m
    compact_ok = compact["n"] > 0 and compact["median_m"] < threshold_m
    sensitive = control["n"] > 0 and control["median_m"] >= threshold_m
    verdict = "PASS" if (spec_ok and compact_ok and sensitive) else "FAIL"
    if spec_ok and compact_ok and not sensitive:
        verdict = "INCONCLUSIVE"
    return {
        "experiment": "frame_alignment",
        "criterion": (
            f"pooled median NN < {threshold_m} m on spec classes {list(SPEC_CLASSES_RAW)} and on "
            f"compact classes {list(COMPACT_CLASSES_RAW)}; identity control on moving pairs "
            f"must be >= {threshold_m} m"
        ),
        "threshold_m": threshold_m,
        "max_range_m": max_range_m,
        "min_motion_m": min_motion_m,
        "sequences": list(sequences),
        "n_pairs": len(pairs),
        "seed": seed,
        "spec": spec,
        "compact": compact,
        "identity_control": control,
        "worst_pair_compact_median_m": float(np.nanmax([p["compact_median_m"] for p in pairs] or [np.nan])),
        "verdict": verdict,
        "passed": verdict == "PASS",
        "pairs": pairs,
    }


def plot_alignment(prev_pts: np.ndarray, cur_pts: np.ndarray, out_path: Path, title: str) -> Path:
    """Top-down overlay: current scan (grey) and the transformed previous scan (orange)."""
    from foveamap.viz.figures import BACKGROUND, _dark_axes, plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110, facecolor=BACKGROUND)
    _dark_axes(ax)
    ax.scatter(cur_pts[:, 1], cur_pts[:, 0], s=0.4, c="#8b949e", linewidths=0, label="scan i")
    ax.scatter(prev_pts[:, 1], prev_pts[:, 0], s=0.4, c="#E8833A", linewidths=0, label="scan i-1 -> i")
    lim = max(float(np.abs(cur_pts[:, :2]).max(initial=1.0)), 1.0)
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m, left +)")
    ax.set_ylabel("x (m, forward +)")
    ax.set_title(title)
    ax.legend(loc="lower left", markerscale=20)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BACKGROUND)
    plt.close(fig)
    return out_path
