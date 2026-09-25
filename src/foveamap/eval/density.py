"""Point density and cell occupancy vs range with sensor-physics justification (task T12.6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from foveamap.config import load_config
from foveamap.eval.buckets import horizontal_range
from foveamap.grid.presets import load_preset
from foveamap.io.sequence import Sequence


def density_eval(
    data_root: str | Path = "data/dataset",
    seq_name: str = "08",
    n_frames: int = 30,
    out_dir: str | Path = "results",
) -> dict[str, Any]:
    """Calculate point density and cell occupancy vs range across frames.

    Generates:
        - results/density_benchmark.json
        - results/plots/density_vs_range.png
        - results/plots/occupancy_vs_range.png
    """
    out_path = Path(out_dir)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    seq = Sequence(data_root, seq_name, with_labels=False)
    fovea_cfg = load_config()
    preset = load_preset("fovea_default", fovea_cfg.grid)

    bin_m = 2.0
    max_m = 100.0
    edges = np.arange(0.0, max_m + bin_m / 2, bin_m)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    annulus_areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

    indices = np.linspace(0, seq.n_frames_total - 1, n_frames, dtype=int)
    total_hist = np.zeros(len(edges) - 1, dtype=np.float64)

    for idx in indices:
        scan = seq.load_frame(idx)
        r = horizontal_range(scan.xyz)
        h, _ = np.histogram(r, bins=edges)
        total_hist += h

    mean_points_per_bin = total_hist / len(indices)
    density_per_m2 = mean_points_per_bin / annulus_areas

    # Cell occupancy per ring
    ring_radii = [r.r_max_mm / 1000.0 for r in preset.rings]
    ring_cell_sizes = [r.cell_mm / 1000.0 for r in preset.rings]

    # Plot 1: Point density vs range
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(bin_centers, density_per_m2, color="#2980b9", linewidth=2.2, label="Observed Point Density")
    ax.set_yscale("log")
    ax.set_xlabel("Horizontal Range (m)", fontweight="bold")
    ax.set_ylabel("Point Density (pts / m², log scale)", fontweight="bold")
    ax.set_title("LiDAR Point Density Falloff vs Range (Sensor-Physics Justification)", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Overlay ring boundaries
    ring_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    prev_r = 0.0
    for k, (r_max, s, col) in enumerate(zip(ring_radii, ring_cell_sizes, ring_colors)):
        ax.axvline(r_max, color=col, linestyle=":", linewidth=1.5)
        mid_r = (prev_r + r_max) / 2.0
        ax.text(mid_r, ax.get_ylim()[1] * 0.4, f"Ring {k}\n{int(s*100)}cm", ha="center", fontsize=8.5, fontweight="bold", color=col)
        prev_r = r_max

    plt.tight_layout()
    density_plot = plots_path / "density_vs_range.png"
    plt.savefig(density_plot)
    plt.close()

    # Plot 2: Cell occupancy vs range (points per cell)
    # In each range bin, points per cell = density_per_m2 * cell_area
    points_per_cell = np.zeros_like(density_per_m2)
    for i, r_c in enumerate(bin_centers):
        # Find which ring this range falls into
        for r_max, s in zip(ring_radii, ring_cell_sizes):
            if r_c <= r_max:
                cell_area = s * s
                points_per_cell[i] = density_per_m2[i] * cell_area
                break

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(bin_centers, points_per_cell, color="#27ae60", linewidth=2.2, label="FoveaMap Points per Cell")
    ax.axhline(1.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="1 point/cell threshold")
    ax.set_xlabel("Horizontal Range (m)", fontweight="bold")
    ax.set_ylabel("Average Points per Allocated Cell", fontweight="bold")
    ax.set_title("Cell Occupancy Stability Across Foveated Rings", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    prev_r = 0.0
    for k, (r_max, s, col) in enumerate(zip(ring_radii, ring_cell_sizes, ring_colors)):
        ax.axvline(r_max, color=col, linestyle=":", linewidth=1.5)
        prev_r = r_max

    plt.tight_layout()
    occupancy_plot = plots_path / "occupancy_vs_range.png"
    plt.savefig(occupancy_plot)
    plt.close()

    report = {
        "sequence": seq_name,
        "n_frames_analyzed": len(indices),
        "bin_width_m": bin_m,
        "max_range_m": max_m,
        "rings": [
            {"ring": k, "r_max_m": r_max, "cell_size_m": s}
            for k, (r_max, s) in enumerate(zip(ring_radii, ring_cell_sizes))
        ],
        "density_10m": float(density_per_m2[int(10 / bin_m)]),
        "density_30m": float(density_per_m2[int(30 / bin_m)]),
        "density_60m": float(density_per_m2[int(60 / bin_m)]),
        "density_90m": float(density_per_m2[int(90 / bin_m)]),
    }

    json_path = out_path / "density_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Density evaluation complete. Saved {json_path}, {density_plot}, and {occupancy_plot}")
    return report
