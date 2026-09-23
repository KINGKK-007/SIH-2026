"""Matplotlib figures for benchmark plots and oracle renders (README 12.3, tasks T2.3, T7.2).

Map orientation everywhere (README 13.2): +x (forward) points up, +y (left) points left.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from foveamap.io.labels import raw_color_rgb, raw_name, semantic_ids  # noqa: E402

BACKGROUND = "#0d1117"
FOREGROUND = "#c9d1d9"


def _dark_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=FOREGROUND)
    ax.xaxis.label.set_color(FOREGROUND)
    ax.yaxis.label.set_color(FOREGROUND)
    ax.title.set_color(FOREGROUND)


def plot_scan_bev(
    xyz: np.ndarray, raw_labels: np.ndarray | None, out_path: str | Path, title: str, extent_m: float = 50.0
) -> Path:
    """Bird's-eye scatter of one scan coloured by raw SemanticKITTI class (T2.3)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = (np.abs(xyz[:, 0]) <= extent_m) & (np.abs(xyz[:, 1]) <= extent_m)
    pts = xyz[keep]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=120, facecolor=BACKGROUND)
    _dark_axes(ax)
    if raw_labels is None:
        ax.scatter(pts[:, 1], pts[:, 0], s=0.2, c=FOREGROUND, linewidths=0)
    else:
        sem = semantic_ids(raw_labels)[keep]
        ids, counts = np.unique(sem, return_counts=True)
        palette = np.array([raw_color_rgb(int(i)) for i in range(max(int(ids.max()) + 1, 1))]) / 255.0
        ax.scatter(pts[:, 1], pts[:, 0], s=0.2, c=palette[sem], linewidths=0)
        order = np.argsort(-counts)[:12]
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=palette[ids[k]], label=raw_name(int(ids[k])))
            for k in order
        ]
        legend = ax.legend(
            handles=handles, loc="lower left", fontsize=7, facecolor=BACKGROUND, framealpha=0.8
        )
        for text in legend.get_texts():
            text.set_color(FOREGROUND)
    ax.plot([0], [0], marker="^", color="white", markersize=8)  # sensor, facing +x (up)
    ax.set_xlim(extent_m, -extent_m)  # +y (left) on the left
    ax.set_ylim(-extent_m, extent_m)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m, left +)")
    ax.set_ylabel("x (m, forward +)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BACKGROUND)
    plt.close(fig)
    return out_path
