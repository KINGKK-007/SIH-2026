"""Memory benchmark over frames: theoretical vs measured across 4 representations (task T12.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from foveamap.config import load_config
from foveamap.grid.engine import rasterize
from foveamap.grid.layers import finalize
from foveamap.grid.memory import memory_report
from foveamap.grid.presets import load_preset
from foveamap.io.labels import raw_to_super
from foveamap.io.sequence import Sequence
from foveamap.models.oracle import OracleModel


def memory_eval(
    data_root: str | Path = "data/dataset",
    seq_name: str = "08",
    n_frames: int = 50,
    out_dir: str | Path = "results",
) -> dict[str, Any]:
    """Benchmark memory across 4 representations over multiple frames.

    Generates:
        - results/memory_benchmark.json
        - results/plots/memory_comparison.png (log-scale bar chart)
    """
    out_path = Path(out_dir)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    seq = Sequence(data_root, seq_name, with_labels=True)
    fovea_cfg = load_config()
    preset = load_preset("fovea_default", fovea_cfg.grid)
    model = OracleModel()

    indices = np.linspace(0, seq.n_frames_total - 1, n_frames, dtype=int)

    measured_dense3d: list[int] = []
    measured_sparse3d: list[int] = []
    measured_uniform25d: list[int] = []
    measured_foveamap: list[int] = []

    theo_dense3d = 0
    theo_sparse3d = 0
    theo_uniform25d = 0
    theo_foveamap = 0

    for idx in indices:
        scan = seq.load_frame(idx)
        pred = model.predict(scan)
        super_cls, moving = raw_to_super(pred.raw_ids)

        acc = rasterize(
            preset,
            scan.xyz,
            super_cls,
            moving,
            pred.conf,
            min_range_mm=fovea_cfg.grid.min_range_mm,
        )
        layers = finalize(acc, fovea_cfg.grid, preset)
        mem = memory_report(preset, layers, fovea_cfg.grid)

        # Record measured
        measured_dense3d.append(mem.dense3d_bytes)
        measured_sparse3d.append(mem.sparse3d_bytes)
        measured_uniform25d.append(mem.uniform25d_bytes)
        measured_foveamap.append(mem.fovea_bytes)

        # Theoretical (constant across frames for preset)
        theo_dense3d = mem.dense3d_bytes
        theo_uniform25d = mem.uniform25d_bytes
        theo_sparse3d = int(np.mean([theo_sparse3d, mem.sparse3d_bytes])) if theo_sparse3d else mem.sparse3d_bytes
        theo_foveamap = mem.fovea_bytes

    def stats(arr: list[int]) -> dict[str, float]:
        a = np.array(arr, dtype=np.float64)
        return {
            "mean_bytes": float(np.mean(a)),
            "mean_mb": round(float(np.mean(a)) / (1024 * 1024), 2),
            "p95_bytes": float(np.percentile(a, 95)),
            "p95_mb": round(float(np.percentile(a, 95)) / (1024 * 1024), 2),
            "min_bytes": float(np.min(a)),
            "max_bytes": float(np.max(a)),
        }

    rep_dense = stats(measured_dense3d)
    rep_sparse = stats(measured_sparse3d)
    rep_uniform = stats(measured_uniform25d)
    rep_fovea = stats(measured_foveamap)

    # Reduction factors
    red_vs_dense = rep_dense["mean_bytes"] / max(1.0, rep_fovea["mean_bytes"])
    red_vs_sparse = rep_sparse["mean_bytes"] / max(1.0, rep_fovea["mean_bytes"])
    red_vs_uniform = rep_uniform["mean_bytes"] / max(1.0, rep_fovea["mean_bytes"])

    report = {
        "sequence": seq_name,
        "n_frames_evaluated": len(indices),
        "representations": {
            "dense_3d": {
                "theoretical_mb": round(theo_dense3d / (1024 * 1024), 2),
                **rep_dense,
            },
            "sparse_3d": {
                "theoretical_mb": round(theo_sparse3d / (1024 * 1024), 2),
                **rep_sparse,
            },
            "uniform_25d_5cm": {
                "theoretical_mb": round(theo_uniform25d / (1024 * 1024), 2),
                **rep_uniform,
            },
            "foveamap_25d": {
                "theoretical_mb": round(theo_foveamap / (1024 * 1024), 2),
                **rep_fovea,
            },
        },
        "reduction_ratios": {
            "vs_dense_3d": round(red_vs_dense, 1),
            "vs_sparse_3d": round(red_vs_sparse, 1),
            "vs_uniform_25d": round(red_vs_uniform, 1),
        },
    }

    # Save JSON
    json_path = out_path / "memory_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Plot publication log-scale comparison
    labels = ["Dense 3D\n(5cm voxel)", "Sparse 3D\n(hash/tree)", "Uniform 2.5D\n(5cm baseline)", "FoveaMap\n(Nested 2.5D)"]
    means_mb = [
        rep_dense["mean_mb"],
        rep_sparse["mean_mb"],
        rep_uniform["mean_mb"],
        rep_fovea["mean_mb"],
    ]
    colors = ["#95a5a6", "#34495e", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(labels, means_mb, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Memory Footprint (MB, Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("Memory Footprint: Four Representations over Sequence 08", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Bar labels
    for bar, mb in zip(bars, means_mb):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 1.15,
            f"{mb:,.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # Annotate reduction badge
    ax.annotate(
        f"{red_vs_uniform:.1f}× Reduction vs Uniform 5cm\n{red_vs_dense:.0f}× vs Dense 3D",
        xy=(3, rep_fovea["mean_mb"]),
        xytext=(2.2, rep_uniform["mean_mb"] * 0.4),
        arrowprops=dict(facecolor="#2ecc71", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f8f5", edgecolor="#2ecc71", linewidth=1.5),
        fontsize=9.5,
        fontweight="bold",
    )

    plt.tight_layout()
    plot_file = plots_path / "memory_comparison.png"
    plt.savefig(plot_file)
    plt.close()

    print(f"Memory evaluation complete. Saved {json_path} and {plot_file}")
    return report
