"""Typed Socket.IO message schema, frozen in docs/design/protocol.md (README 13.3, task T13.1)."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from foveamap.grid.layers import GridLayers
from foveamap.pipeline.runner import FrameResult

SCHEMA_VERSION = 1


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
    iy: list[int]
    ix: list[int]
    ground_z: list[int]
    top_z: list[int]
    clearance: list[int]
    cls: list[int]
    moving_frac: list[int]
    count: list[int]
    conf: list[int]
    flags: list[int]


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


def serialise_rings(layers: GridLayers) -> list[RingSparseDict]:
    """Extract sparse representation of occupied cells per ring for transport efficiency."""
    out: list[RingSparseDict] = []
    spec = layers.spec
    for k, dense_ring in enumerate(layers.rings):
        side = dense_ring.shape[0]
        occupied = dense_ring["count"] > 0
        iy, ix = np.nonzero(occupied)
        cells = dense_ring[iy, ix]

        cell_mm = spec.rings[k].cell_mm if spec else 0
        r_max_mm = spec.rings[k].r_max_mm if spec else 0

        out.append(
            {
                "ring_idx": k,
                "cell_mm": int(cell_mm),
                "r_max_mm": int(r_max_mm),
                "side": int(side),
                "iy": iy.astype(int).tolist(),
                "ix": ix.astype(int).tolist(),
                "ground_z": cells["ground_z"].astype(int).tolist(),
                "top_z": cells["top_z"].astype(int).tolist(),
                "clearance": cells["clearance"].astype(int).tolist(),
                "cls": cells["cls"].astype(int).tolist(),
                "moving_frac": cells["moving_frac"].astype(int).tolist(),
                "count": cells["count"].astype(int).tolist(),
                "conf": cells["conf"].astype(int).tolist(),
                "flags": cells["flags"].astype(int).tolist(),
            }
        )
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
        "rings": serialise_rings(result.layers),
        "objects": objects_list,
    }
