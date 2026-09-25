"""End-to-end latency profiling with per-stage percentiles and real-time budget verification (task T12.3)."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from foveamap.config import load_config
from foveamap.grid.presets import load_preset
from foveamap.io.sequence import Sequence
from foveamap.models.oracle import OracleModel
from foveamap.pipeline.runner import PipelineRunner


def latency_eval(
    data_root: str | Path = "data/dataset",
    seq_name: str = "08",
    n_frames: int = 50,
    warmup: int = 5,
    budget_ms: float = 100.0,
    out_dir: str | Path = "results",
) -> dict[str, Any]:
    """Profile per-stage and end-to-end pipeline latency.

    Generates:
        - results/latency_benchmark.json
        - results/plots/latency_breakdown.png
    """
    out_path = Path(out_dir)
    plots_path = out_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    seq = Sequence(data_root, seq_name, with_labels=True)
    fovea_cfg = load_config()
    preset = load_preset("fovea_default", fovea_cfg.grid)
    model = OracleModel()

    runner = PipelineRunner(mode="oracle", model=model, preset=preset, cfgs=fovea_cfg)

    # Warmup
    for i in range(min(warmup, seq.n_frames_total)):
        runner.process(seq, idx=i)

    # Profiling frames
    indices = np.linspace(warmup, seq.n_frames_total - 1, n_frames, dtype=int)
    stage_times: dict[str, list[float]] = {}
    total_times: list[float] = []

    for idx in indices:
        t0 = perf_counter_ns()
        res = runner.process(seq, idx=idx)
        tot_ms = (perf_counter_ns() - t0) / 1e6
        total_times.append(tot_ms)

        for stage, ms in res.timings_ms.items():
            stage_times.setdefault(stage, []).append(ms)

    def pct(values: list[float]) -> dict[str, float]:
        v = np.array(values)
        return {
            "mean_ms": round(float(np.mean(v)), 2),
            "p50_ms": round(float(np.percentile(v, 50)), 2),
            "p95_ms": round(float(np.percentile(v, 95)), 2),
            "p99_ms": round(float(np.percentile(v, 99)), 2),
            "min_ms": round(float(np.min(v)), 2),
            "max_ms": round(float(np.max(v)), 2),
        }

    stages_summary = {stage: pct(times) for stage, times in stage_times.items()}
    total_summary = pct(total_times)

    pipeline_fps = round(1000.0 / total_summary["mean_ms"], 1) if total_summary["mean_ms"] > 0 else 0.0
    is_realtime = total_summary["p95_ms"] <= budget_ms

    report = {
        "sequence": seq_name,
        "n_frames_profiled": len(indices),
        "budget_ms": budget_ms,
        "target_frequency_hz": 10.0,
        "is_realtime": is_realtime,
        "pipeline_fps": pipeline_fps,
        "total": total_summary,
        "stages": stages_summary,
    }

    json_path = out_path / "latency_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Plot stacked bar chart of stage latencies
    stage_names = list(stage_times.keys())
    stage_p50s = [stages_summary[s]["p50_ms"] for s in stage_names]
    stage_p95s = [stages_summary[s]["p95_ms"] for s in stage_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # Left: Per-stage p50 vs p95 horizontal bar
    y_pos = np.arange(len(stage_names))
    width = 0.35
    ax1.barh(y_pos - width / 2, stage_p50s, width, label="p50 (median)", color="#3498db")
    ax1.barh(y_pos + width / 2, stage_p95s, width, label="p95", color="#e67e22")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([s.replace("_ms", "") for s in stage_names], fontsize=9)
    ax1.set_xlabel("Latency (ms)", fontweight="bold")
    ax1.set_title("Per-Stage Latency Distribution", fontweight="bold", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    # Right: Total latency vs 100ms budget
    tot_p50 = total_summary["p50_ms"]
    tot_p95 = total_summary["p95_ms"]
    ax2.bar(["p50 (median)", "p95"], [tot_p50, tot_p95], color=["#2ecc71", "#27ae60"], width=0.45)
    ax2.axhline(budget_ms, color="#e74c3c", linestyle="--", linewidth=1.5, label="10 Hz Budget (100 ms)")
    ax2.set_ylabel("Total Latency (ms)", fontweight="bold")
    ax2.set_title(f"Total Pipeline: {pipeline_fps:.1f} FPS (REAL-TIME: {'YES' if is_realtime else 'NO'})", fontweight="bold", fontsize=11)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate([tot_p50, tot_p95]):
        ax2.text(i, v + 2, f"{v:.1f} ms", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plot_file = plots_path / "latency_breakdown.png"
    plt.savefig(plot_file)
    plt.close()

    print(f"Latency evaluation complete. Saved {json_path} and {plot_file}")
    return report
