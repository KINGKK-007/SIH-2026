# FoveaMap Dashboard Message Protocol (T7.4)

**Status:** FROZEN — do not change field names or types without a `D-###` decision log entry.  
**Frozen:** Phase 7 · T7.4  
**Unblocks:** Phase 13 (dashboard backend & frontend scaffold).

---

## Overview

The pipeline backend (FastAPI + Socket.IO) pushes one `frame_update` event per processed frame.
The React/Vite frontend subscribes to this event and updates all panels.

Transport: **Socket.IO** over WebSocket (fallback: polling).  
Namespace: `/fovea`  
Event name: `frame_update`  
Direction: server → client (push only; the client sends `request_mode` and `seek_frame` commands).

---

## 1. `frame_update` event payload (JSON)

```jsonc
{
  // ── Identity ────────────────────────────────────────────────────────────────
  "seq":       "08",            // string: SemanticKITTI sequence id
  "frame_idx": 0,               // int: original frame index (not strided)
  "timestamp": 0.0,             // float: Velodyne capture time in seconds (from times.txt)
  "model":     "oracle",        // string: "oracle" | "cached/<name>" | "live/<name>"
  "preset":    "fovea_default", // string: active GridSpec name

  // ── Point accounting (FrameCounters) ───────────────────────────────────────
  "counters": {
    "n_raw":         123389,   // int: total points from the sensor
    "n_invalid":     150,      // int: dropped (non-finite or below min_range)
    "n_in_grid":     120000,   // int: placed in a cell
    "n_out_of_grid": 3239,     // int: beyond the grid extent
    "n_z_saturated": 0         // int: z clamped to ±32767 mm
  },

  // ── Per-stage wall-clock timings ───────────────────────────────────────────
  // Keys are stable stage names; values are milliseconds (float, perf_counter_ns / 1e6).
  "timings_ms": {
    "io_ms":       1.2,    // scan load from disk
    "model_ms":    0.3,    // segmentation model (oracle: label copy)
    "label_ms":    0.5,    // raw_to_super mapping
    "grid_ms":     55.0,   // rasterise → GridAccumulators
    "finalize_ms": 3.1,    // finalize → GridLayers
    "memory_ms":   0.2     // memory_report
  },

  // ── Memory report ─────────────────────────────────────────────────────────
  "memory": {
    "basis":            "allocated",  // "theoretical" | "allocated"
    "dense3d_bytes":    9830400000,   // int
    "sparse3d_bytes":   123389,       // int (measured when basis="allocated")
    "uniform25d_bytes": 480000000,    // int
    "fovea_bytes":      2400000,      // int
    "rss_delta_bytes":  0             // int
  },

  // ── Grid layers (per-ring, sparse encoding for transport efficiency) ────────
  // Each ring sends only cells with count > 0 to avoid sending 16 M entries for uniform_5cm.
  // The frontend reconstructs the full square from (iy, ix, data).
  "rings": [
    {
      "ring_idx":   0,       // int: ring number in the GridSpec
      "cell_mm":    50,      // int: cell size in millimetres
      "r_max_mm":   10000,   // int: outer radius in millimetres
      "side":       400,     // int: N_k = 2 R_k / s_k (full square side)
      // Sparse arrays — all the same length m (number of occupied cells):
      "iy":         [0, 1, 2],         // list[int]: row indices (0..side-1)
      "ix":         [0, 3, 7],         // list[int]: column indices
      "ground_z":   [-50, -48, -46],   // list[int]: mm, INT16_MIN = no data
      "top_z":      [-50, 200, -50],   // list[int]: mm, INT16_MIN = no data
      "clearance":  [0, 350, 0],       // list[int]: mm
      "cls":        [1, 3, 2],         // list[int]: super-class 0..4
      "moving_frac":[0, 0, 0],         // list[int]: 0..255
      "count":      [5, 12, 3],        // list[int]: points in cell
      "conf":       [255, 220, 255],   // list[int]: 0..255
      "flags":      [1, 3, 1]          // list[int]: bitmask (see grid/layers.py)
    }
    // … one entry per ring in the GridSpec
  ],

  // ── Detected objects (Phase 10+) ──────────────────────────────────────────
  // Empty list until Phase 10; frontend must handle [].
  "objects": [
    {
      "id":             1,
      "cls_name":       "moving-car",
      "center":         [5.1, -2.3, 0.4],    // [x, y, z] metres, Velodyne frame
      "size":           [4.2, 1.9, 1.5],      // [l, w, h] metres
      "yaw":            0.31,                  // radians
      "n_points":       87,
      "mean_conf":      0.92,
      "moving":         true,
      "vote_frac":      0.85,
      "speed_mps":      null,                  // null until Phase 10 fills it
      "safety_critical":true
    }
  ]
}
```

---

## 2. Client → server commands

```jsonc
// Change pipeline mode:
{ "event": "request_mode", "data": { "mode": "oracle" | "cached" | "live", "model": "salsanext" } }

// Jump to a specific frame (replay / scrubbing):
{ "event": "seek_frame", "data": { "seq": "08", "frame_idx": 500 } }
```

---

## 3. Versioning and extension policy

- Add **new** top-level keys freely (backwards-compatible).
- **Never** rename or remove existing keys without bumping the schema version and logging a `D-###` decision.
- Add a `"schema_version": 1` key to the payload when Phase 13 is implemented; increment on breaking changes.

---

## 4. Implementation notes for Phase 13

- The backend serialises `FrameResult` to this schema in `server/serialise.py`.
- Ring layers are encoded sparse (only cells with `count > 0`) to keep the payload ≪ 10 MB per frame.
- `grid_ms` is the critical latency path; its p95 target is < 15 ms (T7.5 trigger, see HANDOFF.md §6).
- The frontend renders rings in the `<canvas>` element using the sparse cell list directly (no full array reconstruction needed).
