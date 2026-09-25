"""Hazard detection rate per ring/bucket with Wilson CIs (README 6.8, task T11.8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% confidence z-score
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    spread = (z / denom) * np.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
    lower = max(0.0, float(center - spread))
    upper = min(1.0, float(center + spread))
    return lower, upper


def hazard_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate hazard detection results into detection rates and Wilson CIs.

    Parameters
    ----------
    results : list of dicts with keys:
        'type': str ('pothole', 'kerb', 'overhang')
        'ring': int
        'distance_m': float
        'detected': bool
    """
    by_type: dict[str, list[bool]] = {}
    by_ring: dict[int, list[bool]] = {}
    by_bucket: dict[str, list[bool]] = {}

    buckets = [(0.0, 10.0, "0-10m"), (10.0, 30.0, "10-30m"), (30.0, 60.0, "30-60m"), (60.0, 100.0, "60-100m")]

    for r in results:
        t = r["type"]
        by_type.setdefault(t, []).append(r["detected"])

        k = r["ring"]
        by_ring.setdefault(k, []).append(r["detected"])

        d = r["distance_m"]
        for d0, d1, name in buckets:
            if d0 <= d < d1:
                by_bucket.setdefault(name, []).append(r["detected"])
                break

    def summarize_rates(group: dict[Any, list[bool]]) -> dict[str, Any]:
        out = {}
        for key, vals in group.items():
            tot = len(vals)
            succ = sum(vals)
            rate = succ / tot if tot > 0 else 0.0
            ci_low, ci_high = wilson_score_interval(succ, tot)
            out[str(key)] = {
                "n": tot,
                "detected": succ,
                "rate": round(rate, 4),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            }
        return out

    total_n = len(results)
    total_det = sum(1 for r in results if r["detected"])
    total_low, total_high = wilson_score_interval(total_det, total_n)

    return {
        "total": {
            "n": total_n,
            "detected": total_det,
            "rate": round(total_det / total_n, 4) if total_n > 0 else 0.0,
            "ci_95": [round(total_low, 4), round(total_high, 4)],
        },
        "by_type": summarize_rates(by_type),
        "by_ring": summarize_rates(by_ring),
        "by_bucket": summarize_rates(by_bucket),
    }


def save_hazard_evaluation(
    metrics: dict[str, Any],
    out_json: Path | str,
    out_plot: Path | str | None = None,
) -> None:
    """Save metrics to JSON and plot detection rate by ring."""
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if out_plot:
        plot_path = Path(out_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.5))
            by_ring = metrics.get("by_ring", {})
            rings = sorted(by_ring.keys())
            if rings:
                rates = [by_ring[r]["rate"] for r in rings]
                lows = [by_ring[r]["rate"] - by_ring[r]["ci_95"][0] for r in rings]
                highs = [by_ring[r]["ci_95"][1] - by_ring[r]["rate"] for r in rings]
                ax.errorbar(
                    [f"Ring {r}" for r in rings],
                    rates,
                    yerr=[lows, highs],
                    fmt="o-",
                    capsize=5,
                    color="#e74c3c",
                    lw=2,
                )
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel("Detection Rate")
                ax.set_title("Hazard Detection Rate by Ring (Wilson 95% CI)")
                ax.grid(True, linestyle="--", alpha=0.6)
                fig.tight_layout()
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
        except Exception:
            pass
