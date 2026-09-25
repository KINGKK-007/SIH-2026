"""Server-side ring textures (README 13.3 texture convention, task T13.1)."""

from __future__ import annotations

import numpy as np

from foveamap.grid.layers import (
    DRIVABLE,
    DYNAMIC,
    NON_DRIVABLE_TERRAIN,
    STATIC_OBSTACLE,
    UNKNOWN,
    GridLayers,
)

# RGBA color palettes: [R, G, B, A]
CLASS_PALETTE = np.array(
    [
        [100, 100, 110, 140],  # 0: UNKNOWN (subtle grey)
        [46, 204, 113, 230],   # 1: DRIVABLE (green)
        [180, 110, 50, 220],   # 2: NON_DRIVABLE_TERRAIN (earthy brown)
        [231, 76, 60, 240],    # 3: STATIC_OBSTACLE (red)
        [52, 152, 219, 255],   # 4: DYNAMIC (cyan/blue)
    ],
    dtype=np.uint8,
)

TRAVERSABILITY_PALETTE = np.array(
    [
        [100, 100, 110, 120],  # 0: UNKNOWN
        [46, 204, 113, 230],   # 1: DRIVABLE -> Traversable
        [231, 76, 60, 230],    # 2: NON_DRIVABLE -> Non-traversable
        [231, 76, 60, 240],    # 3: STATIC_OBSTACLE -> Non-traversable
        [231, 76, 60, 255],    # 4: DYNAMIC -> Non-traversable
    ],
    dtype=np.uint8,
)


def _height_to_rgba(height_mm: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """Map height (-2.5 m to +3.5 m) to a vibrant viridis-like colormap."""
    h = height_mm.astype(np.float32) / 1000.0  # metres
    norm = np.clip((h + 2.5) / 6.0, 0.0, 1.0)  # 0.0 to 1.0

    # Simplified viridis approximation: purple (0.0) -> teal (0.5) -> yellow (1.0)
    r = np.clip(np.where(norm < 0.5, norm * 2.0 * 30.0, 30.0 + (norm - 0.5) * 2.0 * 225.0), 0, 255).astype(np.uint8)
    g = np.clip(np.where(norm < 0.5, norm * 2.0 * 180.0, 180.0 + (norm - 0.5) * 2.0 * 50.0), 0, 255).astype(np.uint8)
    b = np.clip(np.where(norm < 0.5, 120.0 + norm * 2.0 * 60.0, 180.0 - (norm - 0.5) * 2.0 * 150.0), 0, 255).astype(np.uint8)
    a = np.where(occupied, 220, 0).astype(np.uint8)

    return np.stack([r, g, b, a], axis=-1)


def ring_textures(layers: GridLayers, layer: str = "class") -> list[np.ndarray]:
    """Generate one RGBA image per ring.

    Texture convention: ``row = N_k - 1 - ix``, ``col = N_k - 1 - iy``
    so that +x (forward) is up (row 0) and +y (left) is left (col 0).

    Args:
        layers: Packed ``GridLayers`` from the grid finalize stage.
        layer: Layer to visualize: ``"class"``, ``"height"``, ``"traversability"``,
            ``"moving"``, or ``"confidence"``.

    Returns:
        List of ``(N_k, N_k, 4)`` uint8 RGBA images, one per ring.
    """
    textures: list[np.ndarray] = []
    layer_name = layer.lower()

    for dense_ring in layers.rings:
        # Re-orient grid so [iy, ix] maps to [row = N - 1 - ix, col = N - 1 - iy]
        ring_oriented = dense_ring.T[::-1, ::-1]
        occupied = ring_oriented["count"] > 0
        n_side = ring_oriented.shape[0]

        rgba = np.zeros((n_side, n_side, 4), dtype=np.uint8)

        if layer_name == "class":
            cls_ids = np.clip(ring_oriented["cls"], 0, 4)
            colors = CLASS_PALETTE[cls_ids]
            colors[~occupied] = [0, 0, 0, 0]
            rgba = colors

        elif layer_name == "height":
            # Use top_z if obstacle present, else ground_z
            z = np.where(ring_oriented["top_z"] != -32768, ring_oriented["top_z"], ring_oriented["ground_z"])
            rgba = _height_to_rgba(z, occupied)

        elif layer_name == "traversability":
            cls_ids = np.clip(ring_oriented["cls"], 0, 4)
            colors = TRAVERSABILITY_PALETTE[cls_ids]
            colors[~occupied] = [0, 0, 0, 0]
            rgba = colors

        elif layer_name == "moving":
            # Blue-to-yellow heatmap based on moving fraction
            frac = ring_oriented["moving_frac"].astype(np.float32) / 255.0
            r = (frac * 255.0).astype(np.uint8)
            g = ((1.0 - np.abs(frac - 0.5) * 2.0) * 200.0).astype(np.uint8)
            b = ((1.0 - frac) * 240.0).astype(np.uint8)
            a = np.where(occupied, np.maximum((frac * 200.0 + 55.0).astype(np.uint8), 80), 0)
            rgba = np.stack([r, g, b, a], axis=-1)

        elif layer_name == "confidence":
            # Confidence grayscale to turquoise
            conf = ring_oriented["conf"].astype(np.float32) / 255.0
            r = ((1.0 - conf) * 50.0).astype(np.uint8)
            g = (conf * 220.0).astype(np.uint8)
            b = (conf * 200.0 + 55.0).astype(np.uint8)
            a = np.where(occupied, 220, 0).astype(np.uint8)
            rgba = np.stack([r, g, b, a], axis=-1)

        else:
            # Fallback to class
            cls_ids = np.clip(ring_oriented["cls"], 0, 4)
            colors = CLASS_PALETTE[cls_ids]
            colors[~occupied] = [0, 0, 0, 0]
            rgba = colors

        textures.append(rgba)

    return textures
