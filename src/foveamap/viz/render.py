"""foveamap.viz.render — Top-down foveated map renderer.

``render_top_down`` produces a matplotlib figure showing the class-coloured
foveated map with ring boundary overlays, cell-size text labels, an ego-vehicle
marker, and an optional traversability overlay.

The function is designed to work with or without a display (uses
``matplotlib.use('Agg')`` when no $DISPLAY is available), so it runs in headless
CI and during batch frame export.

Usage
-----
>>> from foveamap.viz.render import render_top_down, save_png
>>> fig = render_top_down(grid)
>>> save_png(fig, "results/frame_0000.png")
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Headless backend guard (must come before any other matplotlib import)
if os.environ.get("DISPLAY") is None and os.environ.get("MPLBACKEND") is None:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from foveamap.grid.clipmap import ClipmapGrid
from foveamap.grid.layers import FLAG_OBSERVED, FLAG_TRAVERSABLE
from foveamap.viz.palette import (
    HEX,
    RGB_FLOAT,
    SUPERCLASS_LABEL,
    cls_to_rgb,
    alpha_composite_unobserved,
    PALETTE_LUT,
)
from foveamap.io.labels import DRIVABLE, DYNAMIC, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, UNKNOWN

# ─────────────────────────────────────────────────────────────────────────────
# Ring-boundary style
# ─────────────────────────────────────────────────────────────────────────────

_RING_EDGE_COLOR = "#FFFFFF"
_RING_EDGE_ALPHA = 0.55
_RING_EDGE_LW    = 1.0
_RING_LABEL_COLOR = "#E0E0E0"
_RING_LABEL_FS    = 7.5
_BG_COLOR         = "#0D0D0D"

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _composite_rings(grid: ClipmapGrid, shade_ground: bool) -> tuple[NDArray[np.uint8], float]:
    """Compose all rings into a single top-down RGBA image in world coordinates.

    Rings are composited from coarsest to finest so the fine inner ring paints
    over the coarse centre.  World extent is determined by the outermost ring.

    Returns
    -------
    image : (H, W, 3) uint8
    world_half : float — half-extent of the outermost ring in metres.
    """
    outer_ring = grid.spec.rings[-1]
    world_half = outer_ring.half_extent_m
    coarse_cell = outer_ring.cell_m
    coarse_side = grid.side(grid.n_rings - 1)

    # Canvas in coarsest ring's pixel grid
    canvas = np.full((coarse_side, coarse_side, 3), 20, dtype=np.uint8)

    for r_idx in range(grid.n_rings - 1, -1, -1):
        ring = grid.spec.rings[r_idx]
        side = grid.side(r_idx)
        cls_arr  = grid.layer("cls", r_idx)           # (side, side)
        flags    = grid.layer("flags", r_idx)
        observed = (flags & FLAG_OBSERVED) > 0

        rgb = cls_to_rgb(cls_arr)

        if shade_ground:
            gz = grid.layer("ground_z", r_idx)
            has_gz = ~np.isnan(gz) & observed
            if has_gz.any():
                gz_safe = np.where(has_gz, gz, 0.0)
                gz_min = float(gz_safe[has_gz].min())
                gz_max = float(gz_safe[has_gz].max())
                gz_norm = (gz_safe - gz_min) / max(gz_max - gz_min, 0.01)
                shade = (gz_norm * 40 - 20).astype(np.int16)
                rgb = rgb.astype(np.int16)
                rgb[has_gz] = np.clip(rgb[has_gz] + shade[has_gz, None], 0, 255)
                rgb = rgb.astype(np.uint8)

        rgb = alpha_composite_unobserved(rgb, observed)

        # Scale this ring's image onto the coarsest canvas
        scale = coarse_side / side
        if scale == 1.0:
            canvas = rgb
        else:
            scale_int = int(round(scale))
            # Nearest-neighbour upscale using stride tricks
            canvas_h = min(scale_int * side, coarse_side)
            canvas_w = min(scale_int * side, coarse_side)
            region = np.repeat(np.repeat(rgb, scale_int, axis=0), scale_int, axis=1)
            region = region[:canvas_h, :canvas_w]
            canvas[:canvas_h, :canvas_w] = region

    return canvas, world_half


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def render_top_down(
    grid: ClipmapGrid,
    *,
    shade_ground: bool = True,
    show_traversable: bool = False,
    figsize: tuple[float, float] = (9.0, 9.0),
    dpi: int = 120,
    title: str | None = None,
    show_stats: bool = True,
) -> Figure:
    """Render a top-down foveated map with ring overlays.

    Parameters
    ----------
    grid : ClipmapGrid
        The built foveated grid.
    shade_ground : bool
        If True, hill-shade cells by ground_z to give depth cues.
    show_traversable : bool
        Overlay traversable cells with a semi-transparent green tint.
    figsize : tuple[float, float]
        Figure size in inches.
    dpi : int
        Output resolution.
    title : str or None
        Figure title; defaults to "FoveaMap — {preset_name}".
    show_stats : bool
        Annotate the figure with memory and build-time stats.

    Returns
    -------
    matplotlib.figure.Figure
    """
    canvas, world_half = _composite_rings(grid, shade_ground)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # Main image (imshow with world-coordinate extent)
    extent = [-world_half, world_half, -world_half, world_half]
    ax.imshow(
        canvas,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=0,
    )

    # ── Ring boundary overlays ────────────────────────────────────────────
    for r_idx, ring in enumerate(grid.spec.rings):
        he = ring.half_extent_m
        cell_label = (
            f"{int(ring.cell_m * 100)} cm"
            if ring.cell_m < 1.0 else f"{ring.cell_m:.1f} m"
        )
        rect = mpatches.Rectangle(
            (-he, -he), 2 * he, 2 * he,
            linewidth=_RING_EDGE_LW,
            edgecolor=_RING_EDGE_COLOR,
            facecolor="none",
            alpha=_RING_EDGE_ALPHA,
            zorder=2,
        )
        ax.add_patch(rect)
        # Cell-size label at top-right corner of ring
        ax.text(
            he - 0.5, he - 0.5,
            cell_label,
            color=_RING_LABEL_COLOR,
            fontsize=_RING_LABEL_FS,
            ha="right", va="top",
            fontfamily="monospace",
            zorder=3,
            bbox=dict(facecolor="#000000", alpha=0.45, pad=1.5, linewidth=0),
        )
        # Half-extent label on left edge
        ax.text(
            -he + 0.5, 0,
            f"±{he:.0f} m",
            color=_RING_LABEL_COLOR,
            fontsize=_RING_LABEL_FS - 0.5,
            ha="left", va="center",
            fontfamily="monospace",
            zorder=3,
            bbox=dict(facecolor="#000000", alpha=0.35, pad=1.0, linewidth=0),
        )

    # ── Traversable overlay ───────────────────────────────────────────────
    if show_traversable and grid.n_rings > 0:
        for r_idx in range(grid.n_rings):
            trav_flags = grid.layer("flags", r_idx) & FLAG_TRAVERSABLE
            trav_mask = trav_flags > 0
            if trav_mask.any():
                ring = grid.spec.rings[r_idx]
                he   = ring.half_extent_m
                side = grid.side(r_idx)
                trav_rgba = np.zeros((*trav_mask.shape, 4), dtype=np.float32)
                trav_rgba[trav_mask] = [0.2, 0.9, 0.3, 0.30]
                ax.imshow(
                    trav_rgba,
                    extent=[-he, he, -he, he],
                    origin="lower",
                    interpolation="nearest",
                    aspect="equal",
                    zorder=1,
                )

    # ── Ego vehicle marker ────────────────────────────────────────────────
    ax.plot(0, 0, marker="o", color="#FFFFFF", markersize=6, zorder=4)
    ax.annotate(
        "ego", (0, 0),
        xytext=(0.8, 0.8),
        color="#DDDDDD",
        fontsize=7,
        zorder=5,
    )

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xlabel("x → (forward, m)", color="#AAAAAA", fontsize=8)
    ax.set_ylabel("y → (left, m)", color="#AAAAAA", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_xlim(-world_half, world_half)
    ax.set_ylim(-world_half, world_half)

    # ── Title ─────────────────────────────────────────────────────────────
    if title is None:
        ring_desc = " / ".join(
            f"{int(r.cell_m * 100)} cm" if r.cell_m < 1.0 else f"{r.cell_m:.1f} m"
            for r in grid.spec.rings
        )
        title = f"FoveaMap  [{ring_desc}]  ·  {grid.spec.aggregation}"
    fig.suptitle(title, color="#FFFFFF", fontsize=10, y=0.98, fontfamily="monospace")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_entries = [
        mpatches.Patch(facecolor=HEX[sc], label=SUPERCLASS_LABEL[sc])
        for sc in [DRIVABLE, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, DYNAMIC, UNKNOWN]
    ]
    ax.legend(
        handles=legend_entries,
        loc="lower right",
        fontsize=6.5,
        framealpha=0.6,
        facecolor="#1A1A1A",
        edgecolor="#444444",
        labelcolor="#CCCCCC",
    )

    # ── Stats annotation ──────────────────────────────────────────────────
    if show_stats:
        s = grid.stats()
        stats_text = (
            f"active {s['active_cells']:,} cells  "
            f"({s['active_mb']:.1f} MB actual / "
            f"{s['active_cells'] * s['bytes_per_cell_target'] / 1024**2:.1f} MB target)\n"
            f"build {s['build_time_ms']:.1f} ms  "
            f"assigned {s['n_points_assigned']:,}  dropped {s['n_points_dropped']:,}"
        )
        fig.text(
            0.01, 0.01, stats_text,
            color="#888888", fontsize=6.5, fontfamily="monospace",
            va="bottom",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def render_layer(
    grid: ClipmapGrid,
    layer_name: str,
    ring: int = 0,
    *,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (7.0, 7.0),
    dpi: int = 100,
) -> Figure:
    """Render a single scalar layer (e.g., ground_z, count) as a heatmap.

    Parameters
    ----------
    grid : ClipmapGrid
    layer_name : str
    ring : int
    cmap : str
        Matplotlib colormap name.
    figsize, dpi : styling parameters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = grid.layer(layer_name, ring).astype(np.float32)
    r = grid.ring_spec(ring)
    he = r.half_extent_m

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    im = ax.imshow(
        data,
        extent=[-he, he, -he, he],
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        f"{layer_name}  ring={ring}  cell={r.cell_m * 100:.0f} cm",
        color="#CCCCCC", fontsize=9,
    )
    ax.set_xlabel("x (m)", color="#AAAAAA", fontsize=8)
    ax.set_ylabel("y (m)", color="#AAAAAA", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_png(fig: Figure, path: str | Path, *, close: bool = True) -> Path:
    """Save figure to PNG and optionally close it.

    Parameters
    ----------
    fig : Figure
    path : str or Path
    close : bool
        If True, call ``plt.close(fig)`` after saving (default).

    Returns
    -------
    Path — the saved file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    if close:
        plt.close(fig)
    return out


def fig_to_numpy(fig: Figure) -> NDArray[np.uint8]:
    """Render a matplotlib figure to a (H, W, 3) uint8 NumPy array (RGB)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    import struct
    data = buf.read()
    buf.close()
    # Re-open as array using matplotlib
    import matplotlib.image as mpimg
    buf2 = io.BytesIO(data)
    arr = mpimg.imread(buf2)  # float32 RGBA
    return (arr[:, :, :3] * 255).astype(np.uint8)
