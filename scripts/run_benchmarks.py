"""Master benchmark orchestrator: runs all Phase 12 evaluations and writes SUMMARY.md (task T12.1)."""

from __future__ import annotations

import sys
from pathlib import Path

from foveamap.eval.density import density_eval
from foveamap.eval.latency import latency_eval
from foveamap.eval.memory_eval import memory_eval
from foveamap.eval.report import generate_summary_report
from foveamap.eval.tradeoff import tradeoff_sweep


def main() -> None:
    data_root = Path("data/dataset")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================", flush=True)
    print("  FOVEAMAP MASTER BENCHMARK SUITE (PHASE 12)", flush=True)
    print("==================================================", flush=True)

    # 1. Memory Benchmark (T12.4)
    print("\n[1/5] Running Memory Benchmark across representations...", flush=True)
    memory_eval(data_root=data_root, seq_name="08", n_frames=30, out_dir=results_dir)

    # 2. Latency Profiling (T12.3)
    print("\n[2/5] Running Latency Profiling vs 100 ms real-time budget...", flush=True)
    latency_eval(data_root=data_root, seq_name="08", n_frames=30, warmup=5, budget_ms=100.0, out_dir=results_dir)

    # 3. Density & Occupancy Sensor-Physics Justification (T12.6)
    print("\n[3/5] Evaluating Point Density & Cell Occupancy vs Range...", flush=True)
    density_eval(data_root=data_root, seq_name="08", n_frames=20, out_dir=results_dir)

    # 4. Memory-vs-Accuracy Trade-off Sweep (T12.8)
    print("\n[4/5] Executing Memory-vs-Accuracy Pareto Sweep...", flush=True)
    tradeoff_sweep(data_root=data_root, seq_name="08", out_dir=results_dir)

    # 5. Master Report Generation (T12.9)
    print("\n[5/5] Generating results/SUMMARY.md...", flush=True)
    generate_summary_report(results_dir=results_dir)

    print("\n==================================================", flush=True)
    print("  ALL BENCHMARKS COMPLETED SUCCESSFULLY", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()
