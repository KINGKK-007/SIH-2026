"""Seeded synthetic pothole/kerb/overhang injection (README 6.8, task T11.7).

Injects calibrated synthetic hazards into real or synthetic scans:
- Pothole: drops ground z by depth inside circular radius.
- Kerb: raises ground z by height along a linear strip.
- Overhang: inserts horizontal slab of synthetic points at headway height, labelled STATIC_OBSTACLE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_KERB,
    FLAG_LOW_CLEARANCE,
    FLAG_TRAVERSABLE,
    GridLayers,
)
from foveamap.io.labels import DRIVABLE, STATIC_OBSTACLE
from foveamap.pipeline.records import ClassifiedScan, Scan


@dataclass
class Hazard:
    type: str  # "pothole" | "kerb" | "overhang"
    center_xy_m: tuple[float, float]
    params: dict[str, float]


def inject_hazards(
    scan: ClassifiedScan, cfg: object, rng: np.random.Generator
) -> tuple[ClassifiedScan, list[Hazard]]:
    """Inject pothole, kerb, and overhang hazards into candidate drivable areas."""
    pts = scan.scan.xyz.copy()
    super_cls = scan.super_cls.copy()
    moving = scan.moving.copy()
    conf = scan.conf.copy()

    drivable_idx = np.where(super_cls == DRIVABLE)[0]
    if len(drivable_idx) < 10:
        return scan, []

    hazards: list[Hazard] = []

    # 1. Pothole
    pothole_cfg = getattr(cfg, "pothole", None)
    if pothole_cfg is not None:
        idx0 = rng.choice(drivable_idx)
        cx, cy = float(pts[idx0, 0]), float(pts[idx0, 1])
        pr = float(rng.uniform(pothole_cfg.radius_m[0], pothole_cfg.radius_m[1]))
        pd = float(rng.uniform(pothole_cfg.depth_m[0], pothole_cfg.depth_m[1]))

        dist_sq = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        in_pothole = (dist_sq <= pr * pr) & (super_cls == DRIVABLE)
        pts[in_pothole, 2] -= pd

        hazards.append(
            Hazard(
                type="pothole",
                center_xy_m=(cx, cy),
                params={"radius_m": pr, "depth_m": pd, "detect_frac": getattr(pothole_cfg, "detect_frac", 0.5)},
            )
        )

    # 2. Kerb
    kerb_cfg = getattr(cfg, "kerb", None)
    if kerb_cfg is not None:
        idx1 = rng.choice(drivable_idx)
        cx, cy = float(pts[idx1, 0]), float(pts[idx1, 1])
        kh = float(rng.uniform(kerb_cfg.height_m[0], kerb_cfg.height_m[1]))
        kw = float(getattr(kerb_cfg, "width_m", 0.3))
        kl = float(getattr(kerb_cfg, "length_m", 4.0))

        # Oriented box strip
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        in_kerb = (np.abs(dx) <= kl / 2.0) & (np.abs(dy) <= kw / 2.0) & (super_cls == DRIVABLE)
        pts[in_kerb, 2] += kh

        hazards.append(
            Hazard(
                type="kerb",
                center_xy_m=(cx, cy),
                params={"height_m": kh, "width_m": kw, "length_m": kl},
            )
        )

    # 3. Overhang
    overhang_cfg = getattr(cfg, "overhang", None)
    if overhang_cfg is not None:
        idx2 = rng.choice(drivable_idx)
        cx, cy = float(pts[idx2, 0]), float(pts[idx2, 1])
        base_z = float(pts[idx2, 2])
        oh_h = float(rng.uniform(overhang_cfg.height_above_ground_m[0], overhang_cfg.height_above_ground_m[1]))
        target_z = base_z + oh_h

        # Slab of synthetic obstacle points (3.0 x 1.0 m)
        num_pts = 100
        ox = rng.uniform(cx - 1.5, cx + 1.5, num_pts).astype(np.float32)
        oy = rng.uniform(cy - 0.5, cy + 0.5, num_pts).astype(np.float32)
        oz = np.full(num_pts, target_z, dtype=np.float32)
        new_pts = np.column_stack([ox, oy, oz])

        pts = np.vstack([pts, new_pts])
        super_cls = np.concatenate([super_cls, np.full(num_pts, STATIC_OBSTACLE, dtype=np.uint8)])
        moving = np.concatenate([moving, np.zeros(num_pts, dtype=np.bool_)])
        conf = np.concatenate([conf, np.full(num_pts, 255, dtype=np.uint8)])

        hazards.append(
            Hazard(
                type="overhang",
                center_xy_m=(cx, cy),
                params={"height_above_ground_m": oh_h, "length_m": 3.0, "width_m": 1.0},
            )
        )

    new_scan = Scan(
        seq=scan.scan.seq,
        idx=scan.scan.idx,
        xyz=pts,
        remission=np.zeros(len(pts), dtype=np.float32),
        raw_labels=None,
        pose=scan.scan.pose,
        timestamp=scan.scan.timestamp,
    )

    classified = ClassifiedScan(
        scan=new_scan,
        super_cls=super_cls,
        moving=moving,
        conf=conf,
        objects=scan.objects,
    )

    return classified, hazards


def check_hazard_detected(hazard: Hazard, layers: GridLayers) -> bool:
    """Evaluate whether an injected hazard was detected in layers."""
    cx, cy = hazard.center_xy_m
    spec = layers.spec
    if spec is None:
        return False

    # Find the ring that contains (cx, cy)
    from foveamap.grid.engine import world_to_cell

    cell_info = world_to_cell(spec, int(cx * 1000), int(cy * 1000))
    if cell_info is None:
        return False
    r_idx, ix, iy = cell_info

    ring = layers.rings[r_idx]
    side = ring.shape[0]

    if hazard.type == "kerb":
        # Check if FLAG_KERB is set within 1m of the strip
        r_curr = spec.rings[r_idx]
        delta_cells = max(1, int(2000 // r_curr.cell_mm))
        y0 = max(0, iy - delta_cells)
        y1 = min(side, iy + delta_cells + 1)
        x0 = max(0, ix - delta_cells)
        x1 = min(side, ix + delta_cells + 1)
        flags = ring["flags"][y0:y1, x0:x1]
        return bool((flags & FLAG_KERB).any())

    elif hazard.type == "overhang":
        # Check if FLAG_LOW_CLEARANCE or non-traversable is set under slab
        r_curr = spec.rings[r_idx]
        delta_cells = max(1, int(1500 // r_curr.cell_mm))
        y0 = max(0, iy - delta_cells)
        y1 = min(side, iy + delta_cells + 1)
        x0 = max(0, ix - delta_cells)
        x1 = min(side, ix + delta_cells + 1)
        flags = ring["flags"][y0:y1, x0:x1]
        is_low_clr = (flags & FLAG_LOW_CLEARANCE).any()
        not_trav = ((flags & FLAG_HAS_GROUND) & ((flags & FLAG_TRAVERSABLE) == 0)).any()
        return bool(is_low_clr or not_trav)

    elif hazard.type == "pothole":
        # Check if ground height is lower than surrounding by detect_frac * depth
        depth_mm = int(hazard.params.get("depth_m", 0.1) * 1000)
        frac = hazard.params.get("detect_frac", 0.5)
        thresh_mm = int(depth_mm * frac)

        gz_center = ring["ground_z"][iy, ix]
        if not (ring["flags"][iy, ix] & FLAG_HAS_GROUND):
            return False

        # Compare with surrounding ring cells
        surrounding = []
        for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            ny, nx = iy + dy, ix + dx
            if 0 <= ny < side and 0 <= nx < side:
                if ring["flags"][ny, nx] & FLAG_HAS_GROUND:
                    surrounding.append(int(ring["ground_z"][ny, nx]))

        if not surrounding:
            return False
        surrounding_mean = sum(surrounding) / len(surrounding)
        drop = surrounding_mean - float(gz_center)
        return bool(drop >= thresh_mm)

    return False
