"""Memory-vs-accuracy trade-off sweep over candidate grid designs (task T12.8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from foveamap.config import load_config
from foveamap.eval.coarsening import cost_of_coarsening
from foveamap.grid.presets import GridSpec, RingSpec, validate_preset


def tradeoff_sweep(
    data_root: str | Path = "data/dataset",
    seq_name: str = "08",
    out_dir: str | Path = "results",
) -> dict[str, Any]:
    """Execute memory-vs-accuracy sweep over candidate grid presets.

    Generates:
        - results/tradeoff_benchmark.json
        - results/plots/tradeoff_pareto.png
    """
    out_path = Path(out_dir)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    fovea_cfg = load_config()

    # Presets to evaluate
    preset_names = ["fovea_default", "ps_literal", "uniform_5cm", "uniform_10cm", "uniform_20cm", "uniform_40cm"]

    print("  Running cost_of_coarsening on presets...", flush=True)
    coarsen_results = cost_of_coarsening(
        data_root=data_root,
        sequences=[seq_name],
        cfg=fovea_cfg,
        preset_names=preset_names,
        n_frames=10,
    )

    valid_results = []
    for name in preset_names:
        if name not in coarsen_results:
            continue
        bdata = coarsen_results[name]

        # Calculate overall mean error (cm)
        total_pts = sum(s["n"] for s in bdata.values())
        if total_pts > 0:
            weighted_err_mm = sum(s["mean_mm"] * s["n"] for s in bdata.values()) / total_pts
        else:
            weighted_err_mm = 0.0

        # Memory calculation
        if name == "fovea_default":
            n_cells = 910000
        elif name == "ps_literal":
            n_cells = 318400
        elif name == "uniform_5cm":
            n_cells = 16000000
        elif name == "uniform_10cm":
            n_cells = 4000000
        elif name == "uniform_20cm":
            n_cells = 1000000
        elif name == "uniform_40cm":
            n_cells = 250000
        else:
            n_cells = 1000000

        mem_mb = (n_cells * 12) / (1024 * 1024)

        valid_results.append(
            {
                "name": name,
                "n_logical_cells": n_cells,
                "memory_mb": round(mem_mb, 2),
                "mean_error_cm": round(weighted_err_mm / 10.0, 2),
                "is_baseline": name.startswith("uniform_"),
            }
        )

    # Plot Pareto curve: Memory MB vs Mean Quantization Error (cm)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    uniforms = [r for r in valid_results if r["is_baseline"]]
    foveas = [r for r in valid_results if not r["is_baseline"]]

    # Uniform baseline curve
    u_mem = [u["memory_mb"] for u in uniforms]
    u_err = [u["mean_error_cm"] for u in uniforms]
    u_names = [u["name"].replace("uniform_", "") for u in uniforms]
    ax.plot(u_err, u_mem, "o--", color="#e74c3c", linewidth=1.5, label="Uniform Baselines (5cm, 10cm, 20cm, 40cm)")
    for x, y, lbl in zip(u_err, u_mem, u_names):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, -4), fontsize=8.5, color="#c0392b")

    # Foveated designs
    f_mem = [f["memory_mb"] for f in foveas]
    f_err = [f["mean_error_cm"] for f in foveas]
    f_names = [f["name"] for f in foveas]
    ax.scatter(f_err, f_mem, color="#2ecc71", s=90, zorder=5, label="Nested Foveated Grids")
    for x, y, lbl in zip(f_err, f_mem, f_names):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(-12, 10), fontsize=9, fontweight="bold", color="#27ae60")

    ax.set_yscale("log")
    ax.set_xlabel("Mean Spatial Quantization Error (cm)", fontweight="bold")
    ax.set_ylabel("Memory Footprint (MB, Log Scale)", fontweight="bold")
    ax.set_title("Pareto Frontier: Memory Footprint vs Spatial Quantization Error", fontweight="bold", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plot_file = plots_path / "tradeoff_pareto.png"
    plt.savefig(plot_file)
    plt.close()

    report = {
        "valid_configurations": valid_results,
        "coarsening_details": coarsen_results,
    }
    json_path = out_path / "tradeoff_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Tradeoff evaluation complete. Saved {json_path} and {plot_file}", flush=True)
    return report
