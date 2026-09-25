"""Matplotlib figures for benchmark plots and oracle renders (README 12.3, tasks T2.3, T7.2).

Map orientation everywhere (README 13.2): +x (forward) points up, +y (left) points left.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from foveamap.io.labels import raw_color_rgb, raw_name, semantic_ids  # noqa: E402

if TYPE_CHECKING:
    from foveamap.grid.layers import GridLayers
    from foveamap.grid.presets import GridSpec

BACKGROUND = "#0d1117"
FOREGROUND = "#c9d1d9"

# Super-class palette for the grid render (README §13.8 class-colour mapping).
# Index = super-class ID: 0=UNKNOWN, 1=DRIVABLE, 2=NON_DRIVABLE, 3=STATIC_OBSTACLE, 4=DYNAMIC
_SUPER_COLORS = np.array(
    [
        [80, 80, 80],       # 0 UNKNOWN        – dark grey
        [100, 200, 100],    # 1 DRIVABLE        – green
        [160, 120, 80],     # 2 NON_DRIVABLE    – brown
        [200, 100, 80],     # 3 STATIC_OBSTACLE – orange-red
        [220, 50, 50],      # 4 DYNAMIC         – red
    ],
    dtype=np.float32,
) / 255.0


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


def plot_grid_bev(
    spec: GridSpec,
    layers: GridLayers,
    out_path: str | Path,
    title: str,
) -> Path:
    """Top-down 2.5D map render: height-shaded, class-coloured, ring boundaries and cell-size
    labels overlaid (T7.2).

    Orientation: +x (forward) points up, +y (left) points left (README 13.2).

    Args:
        spec: The active :class:`~foveamap.grid.presets.GridSpec`.
        layers: Packed :class:`~foveamap.grid.layers.GridLayers` from ``finalize()``.
        out_path: Destination ``.png`` path; parent directories are created automatically.
        title: Figure title shown at the top.

    Returns:
        Absolute path to the written PNG.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extent_m = spec.extent_mm / 1000.0

    fig, ax = plt.subplots(figsize=(9, 9), dpi=120, facecolor=BACKGROUND)
    _dark_axes(ax)

    # Build RGBA image at the finest-ring resolution; coarser rings fill their annuli.
    finest = spec.rings[0]
    N = finest.side
    img_rgba = np.zeros((N, N, 4), dtype=np.float32)  # [iy, ix]

    from foveamap.grid.layers import INT16_MIN

    # Composite from coarsest to finest so finer data overwrites
    for k in range(len(spec.rings) - 1, -1, -1):
        ring = spec.rings[k]
        ring_layer = layers.rings[k]  # (side, side) structured array
        factor = ring.side
        scale = N // factor  # pixels per cell

        ground_z = ring_layer["ground_z"].astype(np.float32)
        cls = ring_layer["cls"].astype(np.int32)
        count = ring_layer["count"]
        has_data = count > 0

        # Height shading: normalise ground_z to [0.35, 1.0] brightness
        z_valid = ground_z.copy()
        z_valid[~has_data] = np.nan
        with np.errstate(invalid="ignore"):
            z_min = float(np.nanmin(z_valid)) if has_data.any() else 0.0
            z_max = float(np.nanmax(z_valid)) if has_data.any() else 1.0
        z_range = max(z_max - z_min, 1.0)
        brightness = np.where(has_data, 0.35 + 0.65 * (ground_z - z_min) / z_range, 0.0)

        cls_clamped = np.clip(cls, 0, 4)
        base_rgb = _SUPER_COLORS[cls_clamped]  # (factor, factor, 3)

        rgb = base_rgb * brightness[:, :, np.newaxis]
        alpha = has_data.astype(np.float32)

        # Upsample to finest resolution via pixel repetition
        rgb_up = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        alpha_up = np.repeat(np.repeat(alpha, scale, axis=0), scale, axis=1)

        mask = alpha_up > 0
        img_rgba[mask, :3] = rgb_up[mask]
        img_rgba[mask, 3] = 1.0

    # Display: flip vertically so iy=N-1 (world forward) is at top, then flip horizontally
    # so ix=0 (world left) is on the left.
    disp = img_rgba[::-1, ::-1, :]

    ax.imshow(
        disp,
        origin="upper",
        extent=[-extent_m, extent_m, -extent_m, extent_m],
        aspect="equal",
    )

    # Ring boundary arcs and cell-size annotations
    theta = np.linspace(0, 2 * np.pi, 360)
    for ring in spec.rings:
        r_m = ring.r_max_mm / 1000.0
        ax.plot(r_m * np.sin(theta), r_m * np.cos(theta), color="white", lw=0.6, ls="--", alpha=0.5)
        ax.text(
            r_m * 0.71,
            r_m * 0.71,
            f"s={ring.cell_mm / 10:.0f} cm",
            color=FOREGROUND,
            fontsize=6,
            alpha=0.85,
        )

    ax.plot(0, 0, marker="^", color="white", markersize=8, zorder=10)

    ax.set_xlim(extent_m, -extent_m)  # +y (left) on left
    ax.set_ylim(-extent_m, extent_m)
    ax.set_xlabel("y (m, left +)")
    ax.set_ylabel("x (m, forward +)")
    ax.set_title(title, color=FOREGROUND)

    _SUPER_NAMES = ["UNKNOWN", "DRIVABLE", "NON_DRIVABLE", "STATIC_OBSTACLE", "DYNAMIC"]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_SUPER_COLORS[i], label=_SUPER_NAMES[i])
        for i in range(5)
    ]
    legend = ax.legend(legend_handles, _SUPER_NAMES, loc="lower left", fontsize=7,
                       facecolor=BACKGROUND, framealpha=0.8)
    for text in legend.get_texts():
        text.set_color(FOREGROUND)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BACKGROUND)
    plt.close(fig)
    return out_path
