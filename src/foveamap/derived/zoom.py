"""Zoom-lens data path: fine vs coarse resolution comparison for kerb detection (README 6.7, task T11.9)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from foveamap.config import DerivedConfig
from foveamap.derived.step import compute_step, compute_step_height_mm
from foveamap.grid.engine import rasterize
from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_KERB,
    finalize,
)
from foveamap.grid.presets import GridSpec, spec_from_rings
from foveamap.io.labels import raw_to_super
from foveamap.pipeline.records import ClassifiedScan, Scan


class _DefaultGridCfg:
    class_rule = "safety"
    ground_estimator = "mean"
    contact_height_mm = 300
    min_range_mm = 0

    class class_thresholds:
        dyn_min_points = 2
        dyn_min_frac = 0.3
        obs_min_points = 2
        obs_min_frac = 0.3


def render_zoom(
    scan_or_classified: Scan | ClassifiedScan | None = None,
    world_box: tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0),
    preset_fine: GridSpec | None = None,
    preset_coarse: GridSpec | None = None,
    cfg: DerivedConfig | object | None = None,
    out_path: Path | str | None = None,
    *,
    scan: Scan | ClassifiedScan | None = None,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Render a world-coordinate window at fine and coarse resolutions.

    Parameters
    ----------
    scan_or_classified : Scan or ClassifiedScan
        LiDAR scan containing points and labels.
    world_box : (xmin, xmax, ymin, ymax) in metres
        Bounding box of interest in the sensor frame.
    preset_fine : GridSpec, optional
        Fine grid spec (default: 5cm cells).
    preset_coarse : GridSpec, optional
        Coarse grid spec (default: 20cm cells).
    cfg : DerivedConfig, optional
        Derived layer configuration.
    out_path : Path or str, optional
        Destination path for side-by-side PNG plot.

    Returns
    -------
    (step_fine, step_coarse, kerb_detected_fine, kerb_detected_coarse)
    """
    target_scan = scan if scan is not None else scan_or_classified
    if target_scan is None:
        raise ValueError("Must provide scan or scan_or_classified")
    if isinstance(target_scan, ClassifiedScan):
        pts = target_scan.scan.xyz
        super_cls = target_scan.super_cls
        moving = target_scan.moving
        conf = target_scan.conf
    else:
        pts = target_scan.xyz
        if target_scan.raw_labels is not None:
            super_cls, moving = raw_to_super(target_scan.raw_labels)
        else:
            super_cls = np.ones(len(pts), dtype=np.uint8)
            moving = np.zeros(len(pts), dtype=np.bool_)
        conf = np.full(len(pts), 255, dtype=np.uint8)

    xmin, xmax, ymin, ymax = world_box
    in_box = (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
    )

    box_pts = pts[in_box].copy()
    box_cls = super_cls[in_box].copy()
    box_mv = moving[in_box].copy()
    box_cf = conf[in_box].copy()

    # Center points relative to window center for uniform local rasterisation
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    box_pts[:, 0] -= cx
    box_pts[:, 1] -= cy

    extent_m = max(abs(xmax - xmin), abs(ymax - ymin)) / 2.0 + 0.5
    extent_mm = int(np.ceil(extent_m * 1000.0))

    if preset_fine is None:
        preset_fine = spec_from_rings("zoom_fine", [(extent_mm, 50)])
    if preset_coarse is None:
        preset_coarse = spec_from_rings("zoom_coarse", [(extent_mm, 200)])

    derived_cfg = cfg or DerivedConfig(
        max_slope_deg=15.0,
        kerb_min_m=0.06,
        kerb_max_m=0.25,
        obstacle_step_m=0.30,
        min_points_step=2,
        vehicle_height_m=2.0,
        clearance_margin_m=0.2,
        halo=True,
        fill_missing_ground=False,
    )

    # 1. Fine Rasterisation
    acc_fine = rasterize(preset_fine, box_pts, box_cls, box_mv, box_cf)
    layers_fine = finalize(acc_fine, _DefaultGridCfg(), preset_fine)
    layers_fine = compute_step(layers_fine, derived_cfg)
    steps_fine_mm = compute_step_height_mm(layers_fine, derived_cfg)[0]
    kerb_fine = bool((layers_fine.rings[0]["flags"] & FLAG_KERB).any())

    # 2. Coarse Rasterisation
    acc_coarse = rasterize(preset_coarse, box_pts, box_cls, box_mv, box_cf)
    layers_coarse = finalize(acc_coarse, _DefaultGridCfg(), preset_coarse)
    layers_coarse = compute_step(layers_coarse, derived_cfg)
    steps_coarse_mm = compute_step_height_mm(layers_coarse, derived_cfg)[0]
    kerb_coarse = bool((layers_coarse.rings[0]["flags"] & FLAG_KERB).any())

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        # Fine plot
        im1 = ax1.imshow(
            steps_fine_mm / 1000.0,
            cmap="inferno",
            origin="lower",
            vmin=0.0,
            vmax=0.35,
        )
        ax1.set_title(f"Fine Resolution (5 cm cells)\nKerb Detected: {'YES (PASS)' if kerb_fine else 'NO'}")
        ax1.set_xlabel("Local X cells")
        ax1.set_ylabel("Local Y cells")
        fig.colorbar(im1, ax=ax1, label="Step Height (m)")

        # Coarse plot
        im2 = ax2.imshow(
            steps_coarse_mm / 1000.0,
            cmap="inferno",
            origin="lower",
            vmin=0.0,
            vmax=0.35,
        )
        ax2.set_title(f"Coarse Resolution (20 cm cells)\nKerb Detected: {'YES' if kerb_coarse else 'DEGRADED/MISSED'}")
        ax2.set_xlabel("Local X cells")
        ax2.set_ylabel("Local Y cells")
        fig.colorbar(im2, ax=ax2, label="Step Height (m)")

        fig.suptitle(f"Zoom Lens: Road/Sidewalk Kerb at ({xmin:.1f}..{xmax:.1f} m, {ymin:.1f}..{ymax:.1f} m)", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    return steps_fine_mm, steps_coarse_mm, kerb_fine, kerb_coarse
