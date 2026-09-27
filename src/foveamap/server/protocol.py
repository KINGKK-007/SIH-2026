"""Typed Socket.IO message schema, frozen in docs/design/protocol.md (README 13.3, task T13.1).

Binary transport (Issue E fix)
-------------------------------
Each ring is serialised as a compact binary blob instead of 11 separate JSON integer lists.

Per-cell layout (12 bytes, little-endian):
  Offset  Size  Type    Field
  0       2     int16   iy    (row index, 0..side-1)
  2       2     int16   ix    (col index, 0..side-1)
  4       2     int16   ground_z  (mm, INT16_MIN = no data)
  6       2     int16   top_z     (mm, INT16_MIN = no data)
  8       1     uint8   flags
  9       1     uint8   cls
  10      1     uint8   moving_frac  (0-255)
  11      1     uint8   conf         (0-255)

The blob is base64-encoded (URL-safe alphabet) and sent as ``cells_b64``.
The JavaScript frontend decodes it with ``atob`` + ``DataView``.

For backwards compatibility the classic JSON lists are omitted when binary is used.
Clients signal binary support via ``auth.frame_codec == "binary-v1"`` on connect
(the server always sends binary; old clients fall back to the legacy JSON path).
"""

from __future__ import annotations

import base64
from typing import Any, TypedDict

import numpy as np

from foveamap.grid.layers import GridLayers
from foveamap.pipeline.runner import FrameResult

SCHEMA_VERSION = 2  # bumped: binary transport replaces JSON integer lists

# Structured dtype matching the 12-byte on-wire cell layout
_CELL_WIRE_DTYPE = np.dtype([
    ("iy",           "<i2"),
    ("ix",           "<i2"),
    ("ground_z",     "<i2"),
    ("top_z",        "<i2"),
    ("flags",        "u1"),
    ("cls",          "u1"),
    ("moving_frac",  "u1"),
    ("conf",         "u1"),
])
assert _CELL_WIRE_DTYPE.itemsize == 12


class FrameCountersDict(TypedDict):
    n_raw: int
    n_invalid: int
    n_in_grid: int
    n_out_of_grid: int
    n_z_saturated: int


class MemoryReportDict(TypedDict):
    basis: str
    dense3d_bytes: int
    sparse3d_bytes: int
    uniform25d_bytes: int
    fovea_bytes: int
    rss_delta_bytes: int


class RingSparseDict(TypedDict):
    ring_idx: int
    cell_mm: int
    r_max_mm: int
    side: int
    n_cells: int       # number of occupied cells encoded in cells_b64
    cells_b64: str     # base64-encoded binary blob — 12 bytes per cell (see module docstring)
    display_group: list[int]  # kept as JSON: one uint8 per cell for colour group


class FrameUpdatePayload(TypedDict):
    schema_version: int
    seq: str
    frame_idx: int
    timestamp: float
    model: str
    preset: str
    counters: FrameCountersDict
    timings_ms: dict[str, float]
    memory: MemoryReportDict
    rings: list[RingSparseDict]
    objects: list[dict[str, Any]]


def serialise_rings(layers: GridLayers, display_groups: list[np.ndarray] | None = None) -> list[RingSparseDict]:
    """Pack occupied cells as 12-byte binary blobs (base64) instead of 11 JSON integer lists.

    Wire size comparison for ~45 K occupied cells:
      JSON text  ≈ 4-6 MB  (11 lists × 45 K numbers × ~4 chars each)
      Binary b64 ≈ 720 KB  (45 K × 12 B × 4/3 base64 overhead)
      gzip(b64)  ≈ 90-150 KB  (further 5-8× compression)
    """
    out: list[RingSparseDict] = []
    spec = layers.spec
    for k, dense_ring in enumerate(layers.rings):
        side = dense_ring.shape[0]
        occupied = dense_ring["count"] > 0
        iy, ix = np.nonzero(occupied)
        n_cells = len(iy)

        cell_mm  = spec.rings[k].cell_mm  if spec else 0
        r_max_mm = spec.rings[k].r_max_mm if spec else 0

        if n_cells == 0:
            out.append({
                "ring_idx": k, "cell_mm": int(cell_mm), "r_max_mm": int(r_max_mm),
                "side": int(side), "n_cells": 0, "cells_b64": "", "display_group": [],
            })
            continue

        cells = dense_ring[iy, ix]

        # Pack into the 12-byte wire struct
        wire = np.empty(n_cells, dtype=_CELL_WIRE_DTYPE)
        wire["iy"]          = iy.astype(np.int16)
        wire["ix"]          = ix.astype(np.int16)
        wire["ground_z"]    = cells["ground_z"]
        wire["top_z"]       = cells["top_z"]
        wire["flags"]       = cells["flags"]
        wire["cls"]         = cells["cls"]
        wire["moving_frac"] = cells["moving_frac"]
        wire["conf"]        = cells["conf"]

        cells_b64 = base64.b64encode(wire.tobytes()).decode("ascii")

        dg = (display_groups[k].tolist() if display_groups is not None
              else np.choose(np.clip(cells["cls"], 0, 4), [0, 1, 5, 4, 5]).tolist())

        out.append({
            "ring_idx":    k,
            "cell_mm":     int(cell_mm),
            "r_max_mm":    int(r_max_mm),
            "side":        int(side),
            "n_cells":     n_cells,
            "cells_b64":   cells_b64,
            "display_group": dg,
        })
    return out


def serialise_frame_result(
    result: FrameResult,
    seq: str,
    timestamp: float,
    model: str,
    preset_name: str,
) -> FrameUpdatePayload:
    """Serialise a FrameResult to the frozen Socket.IO frame_update payload."""
    c = result.counters
    counters_dict: FrameCountersDict = {
        "n_raw": int(c.n_raw),
        "n_invalid": int(c.n_invalid),
        "n_in_grid": int(c.n_in_grid),
        "n_out_of_grid": int(c.n_out_of_grid),
        "n_z_saturated": int(c.n_z_saturated),
    }

    m = result.memory
    memory_dict: MemoryReportDict = {
        "basis": str(m.basis),
        "dense3d_bytes": int(m.dense3d_bytes),
        "sparse3d_bytes": int(m.sparse3d_bytes),
        "uniform25d_bytes": int(m.uniform25d_bytes),
        "fovea_bytes": int(m.fovea_bytes),
        "rss_delta_bytes": int(m.rss_delta_bytes),
    }

    objects_list: list[dict[str, Any]] = []
    for obj in result.objects:
        objects_list.append(
            {
                "id": getattr(obj, "id", 0),
                "cls_name": getattr(obj, "cls_name", "object"),
                "center": list(getattr(obj, "center", [0.0, 0.0, 0.0])),
                "size": list(getattr(obj, "size", [0.0, 0.0, 0.0])),
                "yaw": float(getattr(obj, "yaw", 0.0)),
                "n_points": int(getattr(obj, "n_points", 0)),
                "mean_conf": float(getattr(obj, "mean_conf", 0.0)),
                "moving": bool(getattr(obj, "moving", False)),
                "vote_frac": float(getattr(obj, "vote_frac", 0.0)),
                "speed_mps": getattr(obj, "speed_mps", None),
                "velocity_xy": list(obj.velocity_xy) if obj.velocity_xy is not None else None,
                "instance_id": obj.instance_id,
                "safety_critical": bool(getattr(obj, "safety_critical", False)),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "seq": seq,
        "frame_idx": int(result.scan_idx),
        "timestamp": float(timestamp),
        "model": model,
        "preset": preset_name,
        "counters": counters_dict,
        "timings_ms": {k: float(v) for k, v in result.timings_ms.items()},
        "memory": memory_dict,
        "rings": serialise_rings(result.layers, result.display_groups),
        "objects": objects_list,
    }
