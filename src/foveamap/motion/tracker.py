"""Optional temporal hysteresis (README 6.4 step 6, task T10.4)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from foveamap.io.poses import transform_points
from foveamap.pipeline.records import ObjectBox

VULNERABLE_ROAD_USERS = {
    "person",
    "bicyclist",
    "motorcyclist",
    "moving-person",
    "moving-bicyclist",
    "moving-motorcyclist",
}


@dataclass
class Track:
    track_id: int
    cls_name: str
    last_center: np.ndarray  # (3,) in previous Velodyne frame
    history: deque[bool] = field(default_factory=deque)  # True if moving vote in that frame
    hits: int = 1
    missed: int = 0
    speed_mps: float | None = None


class ClusterTracker:
    """Associates clusters across frames by ego-compensated centroid with gating."""

    def __init__(self, cfg: Any = None) -> None:
        hysteresis_cfg = getattr(cfg, "hysteresis", None)
        self.enabled = bool(getattr(hysteresis_cfg, "enabled", True)) if hysteresis_cfg else True
        self.window = int(getattr(hysteresis_cfg, "window", 3)) if hysteresis_cfg else 3
        self.min_hits = int(getattr(hysteresis_cfg, "min_hits", 2)) if hysteresis_cfg else 2
        self.gate_m = float(getattr(hysteresis_cfg, "gate_m", 2.0)) if hysteresis_cfg else 2.0
        self.v_min_mps = float(getattr(hysteresis_cfg, "v_min_mps", 0.5)) if hysteresis_cfg else 0.5

        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Clear all active tracks."""
        self.tracks.clear()
        self._next_id = 1

    def update(
        self,
        objects: list[ObjectBox],
        T_prev_to_cur: np.ndarray | None = None,
        dt: float = 0.1,
    ) -> list[ObjectBox]:
        """Associate current objects with active tracks and update motion states.

        Args:
            objects: Detected objects in the current scan.
            T_prev_to_cur: (4, 4) rigid transform mapping previous scan's Velodyne frame into current frame.
            dt: Time delta between scans in seconds (default 0.1 s).

        Returns:
            Updated objects with tracked IDs, velocity, and hysteresis-smoothed moving state.
        """
        if not objects:
            # Mark all tracks missed
            for trk in self.tracks:
                trk.missed += 1
            self.tracks = [trk for trk in self.tracks if trk.missed <= 2]
            return []

        if not self.enabled or T_prev_to_cur is None or len(self.tracks) == 0:
            # Initialize or update tracks directly
            updated: list[ObjectBox] = []
            for obj in objects:
                center_arr = np.array(obj.center, dtype=np.float64)
                hist = deque([obj.moving], maxlen=self.window)
                trk = Track(
                    track_id=self._next_id,
                    cls_name=obj.cls_name,
                    last_center=center_arr,
                    history=hist,
                )
                self._next_id += 1
                self.tracks.append(trk)

                is_vru = obj.cls_name in VULNERABLE_ROAD_USERS
                updated.append(
                    ObjectBox(
                        id=trk.track_id,
                        cls_name=obj.cls_name,
                        center=obj.center,
                        size=obj.size,
                        yaw=obj.yaw,
                        n_points=obj.n_points,
                        mean_conf=obj.mean_conf,
                        moving=bool(obj.moving or is_vru),
                        vote_frac=obj.vote_frac,
                        speed_mps=None,
                        safety_critical=bool(obj.safety_critical or is_vru or obj.moving),
                    )
                )
            return updated

        # Transform previous track centers into current Velodyne frame
        prev_centers = np.stack([trk.last_center for trk in self.tracks], axis=0)  # (M, 3)
        comp_centers = transform_points(T_prev_to_cur, prev_centers)  # (M, 3)

        cur_centers = np.array([obj.center for obj in objects], dtype=np.float64)  # (N, 3)

        # Distance matrix in 3D (or ground xy plane)
        dist_matrix = np.linalg.norm(cur_centers[:, None, :] - comp_centers[None, :, :], axis=2)  # (N, M)

        matched_tracks: set[int] = set()
        matched_objects: dict[int, int] = {}  # obj_idx -> trk_idx

        # Greedy association below gate threshold
        if dist_matrix.size > 0:
            unassigned_objs = set(range(len(objects)))
            unassigned_trks = set(range(len(self.tracks)))

            flat_order = np.argsort(dist_matrix.ravel())
            for idx in flat_order:
                i = int(idx // len(self.tracks))
                j = int(idx % len(self.tracks))
                d = dist_matrix[i, j]
                if d > self.gate_m:
                    break
                if i in unassigned_objs and j in unassigned_trks:
                    matched_objects[i] = j
                    matched_tracks.add(j)
                    unassigned_objs.remove(i)
                    unassigned_trks.remove(j)

        updated_objects: list[ObjectBox] = []

        for i, obj in enumerate(objects):
            center_arr = cur_centers[i]
            is_vru = obj.cls_name in VULNERABLE_ROAD_USERS

            if i in matched_objects:
                trk_idx = matched_objects[i]
                trk = self.tracks[trk_idx]
                displacement = float(np.linalg.norm(center_arr - comp_centers[trk_idx]))
                speed_mps = displacement / max(1e-4, dt)

                trk.history.append(obj.moving)
                trk.last_center = center_arr
                trk.hits += 1
                trk.missed = 0
                trk.speed_mps = speed_mps

                moving_hits = sum(1 for m in trk.history if m)
                hysteresis_moving = (moving_hits >= self.min_hits) or (speed_mps >= self.v_min_mps)
                final_moving = bool(hysteresis_moving or obj.moving or is_vru)

                updated_objects.append(
                    ObjectBox(
                        id=trk.track_id,
                        cls_name=obj.cls_name,
                        center=obj.center,
                        size=obj.size,
                        yaw=obj.yaw,
                        n_points=obj.n_points,
                        mean_conf=obj.mean_conf,
                        moving=final_moving,
                        vote_frac=obj.vote_frac,
                        speed_mps=float(speed_mps),
                        safety_critical=bool(obj.safety_critical or is_vru or final_moving),
                    )
                )
            else:
                # New track
                hist = deque([obj.moving], maxlen=self.window)
                trk = Track(
                    track_id=self._next_id,
                    cls_name=obj.cls_name,
                    last_center=center_arr,
                    history=hist,
                )
                self._next_id += 1
                self.tracks.append(trk)

                updated_objects.append(
                    ObjectBox(
                        id=trk.track_id,
                        cls_name=obj.cls_name,
                        center=obj.center,
                        size=obj.size,
                        yaw=obj.yaw,
                        n_points=obj.n_points,
                        mean_conf=obj.mean_conf,
                        moving=bool(obj.moving or is_vru),
                        vote_frac=obj.vote_frac,
                        speed_mps=None,
                        safety_critical=bool(obj.safety_critical or is_vru or obj.moving),
                    )
                )

        # Increment missed count for unmatched tracks and prune
        for j, trk in enumerate(self.tracks):
            if j not in matched_tracks:
                trk.missed += 1
        self.tracks = [trk for trk in self.tracks if trk.missed <= 2]

        return updated_objects
