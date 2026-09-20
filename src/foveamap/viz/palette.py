"""foveamap.viz.palette — Super-class colour definitions (dark theme).

All colours follow the master plan §5.1 spec.  Both RGB float (0–1) and
uint8 (0–255) forms are provided for matplotlib and OpenCV compatibility.

Colour table
------------
| Super-class          | Hex       | Rationale                       |
|----------------------|-----------|---------------------------------|
| DRIVABLE             | #1FB6A6   | Teal — safe path                |
| NON_DRIVABLE_TERRAIN | #5B6B7A   | Slate — passable but not road   |
| STATIC_OBSTACLE      | #F5A524   | Amber — caution                 |
| DYNAMIC              | #FF4D6D   | Red — danger / moving objects   |
| UNKNOWN              | #111111   | Near-black — unobserved         |
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from foveamap.io.labels import DRIVABLE, DYNAMIC, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, UNKNOWN

# ─────────────────────────────────────────────────────────────────────────────
# Hex colours (from master plan §5.1)
# ─────────────────────────────────────────────────────────────────────────────

HEX: Final[dict[int, str]] = {
    DRIVABLE:             "#1FB6A6",
    NON_DRIVABLE_TERRAIN: "#5B6B7A",
    STATIC_OBSTACLE:      "#F5A524",
    DYNAMIC:              "#FF4D6D",
    UNKNOWN:              "#111111",
}

SUPERCLASS_LABEL: Final[dict[int, str]] = {
    DRIVABLE:             "DRIVABLE",
    NON_DRIVABLE_TERRAIN: "NON_DRIVABLE_TERRAIN",
    STATIC_OBSTACLE:      "STATIC_OBSTACLE",
    DYNAMIC:              "DYNAMIC",
    UNKNOWN:              "UNKNOWN",
}

# ─────────────────────────────────────────────────────────────────────────────
# RGB float (0-1) and uint8 (0-255)
# ─────────────────────────────────────────────────────────────────────────────

def _hex_to_rgb_f(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r / 255.0, g / 255.0, b / 255.0


def _hex_to_rgb_u8(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


#: Mapping from super-class ID → (R, G, B) floats in [0, 1] for matplotlib.
RGB_FLOAT: Final[dict[int, tuple[float, float, float]]] = {
    sc: _hex_to_rgb_f(hx) for sc, hx in HEX.items()
}

#: Mapping from super-class ID → (R, G, B) uint8 for OpenCV / numpy images.
RGB_UINT8: Final[dict[int, tuple[int, int, int]]] = {
    sc: _hex_to_rgb_u8(hx) for sc, hx in HEX.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# LUT: uint8 super-class ID → RGB image row
# ─────────────────────────────────────────────────────────────────────────────

# 256-row LUT indexed by uint8 super-class ID.
# Index 255 = UNKNOWN; unrecognised IDs also map to UNKNOWN colour.
_UNKNOWN_RGB_U8: tuple[int, int, int] = RGB_UINT8[UNKNOWN]
PALETTE_LUT: NDArray[np.uint8] = np.tile(
    np.array(_UNKNOWN_RGB_U8, dtype=np.uint8), (256, 1)
)
for _sc, _rgb in RGB_UINT8.items():
    PALETTE_LUT[_sc] = np.array(_rgb, dtype=np.uint8)


def cls_to_rgb(cls_array: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Map a 2-D super-class array to an RGB image.

    Parameters
    ----------
    cls_array : NDArray[uint8]
        Shape ``(H, W)`` — super-class IDs.

    Returns
    -------
    NDArray[uint8]
        Shape ``(H, W, 3)`` — RGB image, values 0–255.
    """
    return PALETTE_LUT[cls_array]


def alpha_composite_unobserved(
    rgb: NDArray[np.uint8],
    observed_mask: NDArray[np.bool_],
    bg_color: tuple[int, int, int] = (20, 20, 20),
) -> NDArray[np.uint8]:
    """Dim unobserved cells by blending with the background colour.

    Parameters
    ----------
    rgb : (H, W, 3) uint8
    observed_mask : (H, W) bool — True where the cell has been observed.
    bg_color : (R, G, B) uint8 background (dark default).

    Returns
    -------
    (H, W, 3) uint8
    """
    out = rgb.copy()
    bg = np.array(bg_color, dtype=np.uint8)
    out[~observed_mask] = bg
    return out
