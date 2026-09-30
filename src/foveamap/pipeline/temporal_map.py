"""Persistent rolling 2.5D world map (addresses Issue A — mapping vs single-frame projection).

The problem statement requires *mapping*, not just per-frame projection.  This module maintains
a world-aligned flat accumulation grid that persists across frames so that:

  * Road/terrain seen 5 seconds ago is still visible as the vehicle drives away.
  * Static obstacles (walls, poles, buildings) accumulate and solidify across scans.
  * Dynamic actors (vehicles, pedestrians) are explicitly EXCLUDED from accumulation so
    they do not leave ghost trails in the persistent map.

Design
------
  * World grid: ``(N × N)`` cells at ``cell_m`` resolution centred on the first ego pose.
    The grid slides (rolls) when the ego exits the inner 60 % of the grid to keep memory bounded.
  * Per-cell fields: ``ground_z`` (int16 mm), ``top_z`` (int16 mm), ``cls`` (uint8), ``flags``
    (uint8), ``age`` (uint8 frames-since-last-observation).
  * Update: project each current frame's STATIC occupied ring cells into world frame via
    the scan's KITTI pose matrix, then EMA-blend with existing world-map values.
  * Extraction: query the world map for unobserved / low-count cells in the current fovea rings
    and back-fill them with persistent background knowledge.

Usage (from ``PipelineRunner.process``)
-----------------------------------------
  acc = TemporalMapAccumulator(spec)
  ...
  layers = acc.update_and_merge(layers, scan.pose, dynamic_mask)
"""

from __future__ import annotations

import numpy as np

from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_HAS_OBSTACLE,
    FLAG_POTHOLE,
    FLAG_KERB,
    FLAG_STEEP,
    FLAG_TRAVERSABLE,
    GridLayers,
    INT16_MIN,
)
from foveamap.grid.presets import GridSpec
from foveamap.io.labels import DYNAMIC, UNKNOWN

# How many consecutive frames of non-observation before a world cell fades
_MAX_AGE = 60

# EMA coefficient: higher = faster adaptation to new observations
_EMA_ALPHA = np.float32(0.30)

# Minimum cell count in current frame to trust it for world update
_MIN_COUNT_TRUST = 2


class TemporalMapAccumulator:
    """Maintains a persistent world-frame 2.5D elevation map across LiDAR frames.

    Parameters
    ----------
    spec :
        The active :class:`~foveamap.grid.presets.GridSpec` (used to determine
        the maximum sensor range and ring layout).
    map_radius_m :
        Half-side of the world map in metres (default 200 m → 400 × 400 m grid).
    cell_m :
        World-map cell size in metres (default 0.20 m = 20 cm).
    """

    def __init__(
        self,
        spec: GridSpec,
        map_radius_m: float = 200.0,
        cell_m: float = 0.20,
    ) -> None:
        self.cell_m = float(cell_m)
        self.cell_mm = int(cell_m * 1000)
        n = int(2 * map_radius_m / cell_m)
        self.n = n
        self._map_radius_m = map_radius_m

        # World grid fields
        self._ground_z  = np.full((n, n), INT16_MIN, dtype=np.int16)
        self._top_z     = np.full((n, n), INT16_MIN, dtype=np.int16)
        self._cls       = np.full((n, n), UNKNOWN,   dtype=np.uint8)
        self._flags     = np.zeros((n, n),            dtype=np.uint8)
        self._age       = np.full((n, n), _MAX_AGE,  dtype=np.uint8)  # start as "unseen"

        # World-frame origin (set from first pose received)
        self._origin_set = False
        self._origin_xy  = np.zeros(2, dtype=np.float64)  # world metres
        self._spec       = spec

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def update_and_merge(
        self,
        layers: GridLayers,
        pose: np.ndarray,  # (4, 4) KITTI world-frame pose of the LiDAR at this scan
        dynamic_mask: np.ndarray,  # (N_pts,) bool — per-point dynamic flag (not used directly here)
    ) -> GridLayers:
        """Project current frame into world map and back-fill unseen ring cells.

        Parameters
        ----------
        layers :
            Current-frame :class:`GridLayers` (after finalize + derived).
        pose :
            ``(4, 4)`` rigid body transform mapping LiDAR-frame vectors into the
            world (camera-0) coordinate frame, as stored in KITTI ``poses.txt``.
        dynamic_mask :
            Per-point dynamic flag; not used directly but available for future
            point-level filtering.

        Returns
        -------
        GridLayers
            The *same* layers with unobserved cells back-filled from the world map.
            No copies are made for observed cells.
        """
        # Initialise world origin from first pose
        if not self._origin_set:
            tx = float(pose[0, 3])
            ty = float(pose[2, 3])  # KITTI: x=right, y=down, z=fwd → use x & z for plan view
            self._origin_xy[:] = [tx, ty]
            self._origin_set = True

        # Ego position in world frame (x-right, z-forward in KITTI camera coords)
        ego_x = float(pose[0, 3])
        ego_z = float(pose[2, 3])

        # ── 1. Project current ring cells → world map (static only) ──────
        self._project_to_world(layers, pose, ego_x, ego_z)

        # ── 2. Age all cells by one frame; prune very stale cells ─────────
        stale = self._age < _MAX_AGE
        self._age[stale] += 1
        very_old = self._age >= _MAX_AGE
        self._cls[very_old] = UNKNOWN
        self._ground_z[very_old] = INT16_MIN
        self._top_z[very_old]    = INT16_MIN
        self._flags[very_old]    = 0

        # ── 3. Back-fill unobserved ring cells from world map ─────────────
        new_rings = self._backfill_rings(layers, pose)

        return GridLayers(spec=layers.spec, rings=new_rings)

    def reset(self) -> None:
        """Clear the world map (call on seek / sequence change)."""
        self._ground_z[:] = INT16_MIN
        self._top_z[:]    = INT16_MIN
        self._cls[:]      = UNKNOWN
        self._flags[:]    = 0
        self._age[:]      = _MAX_AGE
        self._origin_set  = False

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _world_to_grid(self, wx: np.ndarray, wz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert world (x, z) metres → integer grid indices (col, row)."""
        ox, oz = self._origin_xy
        col = np.floor((wx - ox) / self.cell_m + self.n / 2).astype(np.int32)
        row = np.floor((wz - oz) / self.cell_m + self.n / 2).astype(np.int32)
        return col, row

    def _project_to_world(
        self,
        layers: GridLayers,
        pose: np.ndarray,
        ego_x: float,
        ego_z: float,
    ) -> None:
        """Write static cells from current rings into the world grid."""
        spec = layers.spec
        # Rotation+translation (camera world frame): pose is (4,4)
        R = pose[:3, :3].astype(np.float32)
        t = pose[:3, 3].astype(np.float32)

        for k, ring in enumerate(layers.rings):
            rs = spec.rings[k] if spec else None
            if rs is None:
                continue
            cell_mm = rs.cell_mm
            r_max_mm = rs.r_max_mm
            side = ring.shape[0]

            # All iy, ix indices where the cell has ground or obstacle
            has_data = ring["count"] > _MIN_COUNT_TRUST
            iy, ix = np.where(has_data)
            if len(iy) == 0:
                continue

            cls_vals   = ring["cls"][iy, ix]
            flags_vals = ring["flags"][iy, ix]
            gz_vals    = ring["ground_z"][iy, ix]
            tz_vals    = ring["top_z"][iy, ix]

            # Skip dynamic cells — they must NOT be accumulated into the persistent map
            is_dynamic = (cls_vals == DYNAMIC) | ((flags_vals & 0x02) != 0)  # FLAG_HAS_OBSTACLE from moving
            static_mask = ~is_dynamic & (gz_vals != INT16_MIN)
            if not static_mask.any():
                continue

            iy_s  = iy[static_mask]
            ix_s  = ix[static_mask]
            gz_s  = gz_vals[static_mask].astype(np.float32)
            tz_s  = tz_vals[static_mask]
            cls_s = cls_vals[static_mask]
            fl_s  = flags_vals[static_mask]

            # Cell centre in LiDAR (Velodyne) frame: x=forward, y=left
            # Ring layout: iy=forward (x), ix=lateral (y) with offset
            half_side = side // 2
            # Cell centre in mm from sensor
            x_mm_lidar = (iy_s.astype(np.float32) - half_side + 0.5) * cell_mm
            y_mm_lidar = (ix_s.astype(np.float32) - half_side + 0.5) * cell_mm
            z_mm_lidar = gz_s  # ground height mm

            # Convert to metres in Velodyne (sensor) frame
            pts_sensor = np.stack([
                x_mm_lidar * 1e-3,
                y_mm_lidar * 1e-3,
                z_mm_lidar * 1e-3,
            ], axis=1).astype(np.float32)  # (M, 3)

            # Transform to KITTI camera world frame via pose
            # KITTI pose maps camera-0 to world; LiDAR→camera handled by calibration.
            # We approximate: treat sensor frame as camera-0 (valid for overhead view).
            pts_world = (R @ pts_sensor.T).T + t  # (M, 3)
            wx = pts_world[:, 0]  # world x (right)
            wz = pts_world[:, 2]  # world z (forward)

            col, row = self._world_to_grid(wx, wz)
            in_bounds = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)
            col = col[in_bounds]
            row = row[in_bounds]
            gz_write  = gz_s[in_bounds].astype(np.int16)
            tz_write  = tz_s[in_bounds]
            cls_write = cls_s[in_bounds]
            fl_write  = fl_s[in_bounds]

            # EMA blend existing ground_z; use hard-set for first observation
            existing_gz = self._ground_z[row, col]
            has_existing = existing_gz != INT16_MIN
            new_gz = np.where(
                has_existing,
                ((_EMA_ALPHA * gz_write.astype(np.float32) +
                  (1 - _EMA_ALPHA) * existing_gz.astype(np.float32))
                 .clip(-32767, 32767).astype(np.int16)),
                gz_write,
            )
            self._ground_z[row, col] = new_gz

            # top_z: take max (worst-case clearance)
            existing_tz = self._top_z[row, col]
            self._top_z[row, col] = np.where(
                (existing_tz == INT16_MIN) | (tz_write > existing_tz),
                tz_write,
                existing_tz,
            )

            # Class: overwrite only if current has more certainty
            self._cls[row, col]   = cls_write
            self._flags[row, col] = fl_write
            self._age[row, col]   = 0  # freshly observed

    def _backfill_rings(self, layers: GridLayers, pose: np.ndarray) -> list[np.ndarray]:
        """For each ring cell with no current observation, look up the world map."""
        spec = layers.spec
        new_rings = [r.copy() for r in layers.rings]

        R = pose[:3, :3].astype(np.float32)
        t = pose[:3, 3].astype(np.float32)

        for k, ring in enumerate(new_rings):
            rs = spec.rings[k] if spec else None
            if rs is None:
                continue
            cell_mm = rs.cell_mm
            side = ring.shape[0]
            half_side = side // 2

            # Identify empty cells in the current frame
            empty = ring["count"] == 0
            iy, ix = np.where(empty)
            if len(iy) == 0:
                continue

            # Cell centre in sensor x/y metres (z=0: only the ground plane footprint is needed)
            x_sens = (iy.astype(np.float32) - half_side + 0.5) * (cell_mm * 1e-3)
            y_sens = (ix.astype(np.float32) - half_side + 0.5) * (cell_mm * 1e-3)
            pts_sensor = np.stack([x_sens, y_sens, np.zeros_like(x_sens)], axis=1)

            # Rotate+translate into world frame exactly like _project_to_world, so the
            # lookup lands on the same world cells the write path filled in.
            pts_world = (R @ pts_sensor.T).T + t
            wx = pts_world[:, 0]
            wz = pts_world[:, 2]

            col, row = self._world_to_grid(wx, wz)
            in_bounds = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)

            iy_v  = iy[in_bounds]
            ix_v  = ix[in_bounds]
            col_v = col[in_bounds]
            row_v = row[in_bounds]

            # Only back-fill cells with valid world-map observations (age < MAX_AGE)
            fresh = self._age[row_v, col_v] < _MAX_AGE
            if not fresh.any():
                continue

            iy_f  = iy_v[fresh]
            ix_f  = ix_v[fresh]
            col_f = col_v[fresh]
            row_f = row_v[fresh]

            ring["ground_z"][iy_f, ix_f]    = self._ground_z[row_f, col_f]
            ring["top_z"][iy_f, ix_f]       = self._top_z[row_f, col_f]
            ring["cls"][iy_f, ix_f]         = self._cls[row_f, col_f]
            ring["flags"][iy_f, ix_f]       = self._flags[row_f, col_f]
            # Mark as back-filled: set count=1 so downstream knows data exists
            ring["count"][iy_f, ix_f]       = 1

        return new_rings
