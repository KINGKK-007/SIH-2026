"""``estimate_motion``: README 6.4 steps 1-5, 7-8 (task T10.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from foveamap.io.labels import DYNAMIC, LABELS, STATIC_OBSTACLE
from foveamap.io.poses import transform_points
from foveamap.motion.boxes import oriented_box
from foveamap.motion.cluster import cluster_points
from foveamap.motion.residual import range_residual_votes
from foveamap.pipeline.records import ClassifiedScan, ObjectBox, Prediction, Scan

VULNERABLE_ROAD_USERS = {
    "person",
    "bicyclist",
    "motorcyclist",
    "moving-person",
    "moving-bicyclist",
    "moving-motorcyclist",
}


def estimate_motion(
    cur: ClassifiedScan,
    prev_scans: list[tuple[Scan, Prediction]],
    transforms: list[np.ndarray],
    cfg: Any,
    use_oracle: bool = False,
) -> ClassifiedScan:
    """Return ``cur`` with ``moving``/``super_cls`` promoted and ``objects`` filled.

    In the oracle path (or when use_oracle=True), motion comes from raw IDs 252-259
    and objects are formed directly from ground-truth instance IDs.
    In the geometric motion path, range-image residuals and range-scaled Euclidean
    clustering identify dynamic clusters and propagate state.

    Args:
        cur: Current classified scan.
        prev_scans: History of previous scans and their predictions.
        transforms: Relative transforms mapping each previous scan's Velodyne frame into current frame.
        cfg: MotionConfig or dict/object with motion settings.
        use_oracle: If True and ground-truth labels exist, compute motion and objects from GT.

    Returns:
        Updated ClassifiedScan with populated objects and updated moving/super_cls arrays.
    """
    cur_xyz = cur.scan.xyz
    n_pts = len(cur_xyz)
    super_cls = np.array(cur.super_cls, copy=True)
    moving = np.array(cur.moving, copy=True)
    conf = cur.conf
    raw_labels = cur.scan.raw_labels

    box_cfg = getattr(cfg, "box", None)
    angle_step = float(getattr(box_cfg, "angle_step_deg", 1.0)) if box_cfg else 1.0

    cluster_cfg = getattr(cfg, "cluster", None)
    min_points = int(getattr(cluster_cfg, "min_points", 5)) if cluster_cfg else 5

    # ── Oracle Path ─────────────────────────────────────────────────────────
    if use_oracle and raw_labels is not None:
        sem_ids = (raw_labels & 0xFFFF).astype(np.int32)
        inst_ids = (raw_labels >> 16).astype(np.int32)

        # Ground-truth moving points: raw IDs 252-259
        moving_mask = (sem_ids >= 252) & (sem_ids <= 259)
        moving[moving_mask] = True
        super_cls[moving_mask] = DYNAMIC

        objects: list[ObjectBox] = []
        unique_insts = np.unique(inst_ids[inst_ids > 0])

        for inst in unique_insts:
            inst_mask = inst_ids == inst
            n_inst_pts = int(np.sum(inst_mask))
            if n_inst_pts < min_points:
                continue

            pts = cur_xyz[inst_mask]
            center, size, yaw = oriented_box(pts, angle_step_deg=angle_step)

            # Majority semantic class
            inst_sems = sem_ids[inst_mask]
            maj_id = int(np.bincount(inst_sems).argmax())
            info = LABELS.get(maj_id)
            cls_name = info.name if info else f"class_{maj_id}"

            is_inst_moving = bool(np.any(moving_mask[inst_mask]))
            vote_frac = float(np.mean(moving_mask[inst_mask]))
            is_vru = cls_name in VULNERABLE_ROAD_USERS
            safety_crit = bool(info.safety_critical if info else False) or is_inst_moving or is_vru

            mean_conf = float(np.mean(conf[inst_mask])) if len(conf) == n_pts else 255.0

            objects.append(
                ObjectBox(
                    id=int(inst),
                    cls_name=cls_name,
                    center=center,
                    size=size,
                    yaw=yaw,
                    n_points=n_inst_pts,
                    mean_conf=mean_conf,
                    moving=is_inst_moving or is_vru,
                    vote_frac=vote_frac,
                    speed_mps=None,
                    safety_critical=safety_crit,
                )
            )

        return ClassifiedScan(
            scan=cur.scan,
            super_cls=super_cls,
            moving=moving,
            conf=conf,
            objects=objects,
            raw_ids=cur.raw_ids,
        )

    # ── Geometric Motion Path ───────────────────────────────────────────────
    enabled = getattr(cfg, "enabled", True)
    if not enabled:
        return cur

    # 1. Range-image residual votes
    if prev_scans and transforms:
        votes_list: list[np.ndarray] = []
        for (prev_scan, _), T_prev_to_cur in zip(prev_scans, transforms):
            prev_xyz_in_cur = transform_points(T_prev_to_cur, prev_scan.xyz)
            v = range_residual_votes(cur_xyz, prev_xyz_in_cur, cfg)
            votes_list.append(v)

        if len(votes_list) == 1:
            point_votes = votes_list[0]
        else:
            # Multi-gap aggregation: point votes moving if any gap votes moving
            point_votes = np.maximum.reduce(votes_list)
    else:
        point_votes = np.full(n_pts, -1, dtype=np.int8)

    # 2. Identify candidate movable points
    movable_classes = getattr(
        cfg, "movable_classes", [10, 11, 13, 15, 16, 18, 20, 30, 31, 32]
    )

    if cur.raw_ids is not None:
        cand_mask = np.isin(cur.raw_ids, movable_classes)
    elif raw_labels is not None:
        cand_mask = np.isin(raw_labels & 0xFFFF, movable_classes)
    else:
        cand_mask = (super_cls == STATIC_OBSTACLE) | (super_cls == DYNAMIC)

    cand_indices = np.flatnonzero(cand_mask)
    objects = []

    if len(cand_indices) >= min_points:
        cand_xyz = cur_xyz[cand_indices]
        cluster_ids = cluster_points(cand_xyz, getattr(cfg, "cluster", None))

        vote_cfg = getattr(cfg, "vote", None)
        vote_frac_thresh = float(getattr(vote_cfg, "vote_frac", 0.30)) if vote_cfg else 0.30
        vote_min_points = int(getattr(vote_cfg, "vote_min_points", 3)) if vote_cfg else 3

        unique_clusters = np.unique(cluster_ids[cluster_ids >= 0])

        for c_id in unique_clusters:
            in_cluster = cluster_ids == c_id
            cluster_pt_indices = cand_indices[in_cluster]
            n_cluster_pts = len(cluster_pt_indices)

            cluster_votes = point_votes[cluster_pt_indices]
            n_moving_votes = int(np.sum(cluster_votes == 1))
            frac_moving = float(n_moving_votes / max(1, n_cluster_pts))

            is_cluster_moving = (frac_moving >= vote_frac_thresh) and (n_moving_votes >= vote_min_points)

            pts = cur_xyz[cluster_pt_indices]
            center, size, yaw = oriented_box(pts, angle_step_deg=angle_step)

            # Class determination
            if cur.raw_ids is not None:
                maj_raw = int(np.bincount(cur.raw_ids[cluster_pt_indices]).argmax())
                info = LABELS.get(maj_raw)
                cls_name = info.name if info else f"class_{maj_raw}"
                is_safety = bool(info.safety_critical if info else False)
            elif raw_labels is not None:
                raw_sems = (raw_labels[cluster_pt_indices] & 0xFFFF).astype(np.int32)
                maj_raw = int(np.bincount(raw_sems).argmax())
                info = LABELS.get(maj_raw)
                cls_name = info.name if info else f"class_{maj_raw}"
                is_safety = bool(info.safety_critical if info else False)
            else:
                cls_name = "vehicle"
                is_safety = False

            is_vru = cls_name in VULNERABLE_ROAD_USERS
            final_moving = is_cluster_moving or is_vru
            final_safety = is_safety or is_vru or final_moving

            mean_conf = float(np.mean(conf[cluster_pt_indices])) if len(conf) == n_pts else 255.0

            # Propagate to points
            if final_moving:
                moving[cluster_pt_indices] = True
                super_cls[cluster_pt_indices] = DYNAMIC
            else:
                moving[cluster_pt_indices] = False
                # If not VRU, ensure static obstacle
                if not is_vru:
                    super_cls[cluster_pt_indices] = STATIC_OBSTACLE

            objects.append(
                ObjectBox(
                    id=int(c_id),
                    cls_name=cls_name,
                    center=center,
                    size=size,
                    yaw=yaw,
                    n_points=n_cluster_pts,
                    mean_conf=mean_conf,
                    moving=final_moving,
                    vote_frac=frac_moving,
                    speed_mps=None,
                    safety_critical=final_safety,
                )
            )

    return ClassifiedScan(
        scan=cur.scan,
        super_cls=super_cls,
        moving=moving,
        conf=conf,
        objects=objects,
        raw_ids=cur.raw_ids,
    )
