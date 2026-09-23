"""Seeded synthetic SemanticKITTI-format sequences (README 11.1 test fixtures; D-015).

A 64-beam sensor (HDL-64E-like vertical FOV) is ray-cast through an analytic street scene: road with a
12 cm kerb, sidewalks, terrain, buildings, poles, trees with overhanging canopies, parked cars, a moving
car and a walking pedestrian. The vehicle drives a gently weaving path, so consecutive poses contain
real rotations. Files are written exactly in the KITTI/SemanticKITTI formats (README 5.1-5.2), including
camera-0 poses and a non-identity Velodyne->camera ``Tr``, so every loader, the pose maths and the
alignment check can be exercised without the real dataset.

This is test/demo data only. It is never used for any reported number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SENSOR_HEIGHT_M = 1.73
ROAD_Z = -SENSOR_HEIGHT_M
KERB_HEIGHT_M = 0.12
SIDEWALK_Z = ROAD_Z + KERB_HEIGHT_M
TERRAIN_Z = ROAD_Z + 0.05
ROAD_HALF_WIDTH = 5.0
SIDEWALK_OUTER = 8.0

# Raw SemanticKITTI IDs used by the scene.
ROAD, LANE_MARKING, SIDEWALK, TERRAIN = 40, 60, 48, 72
BUILDING, POLE, VEGETATION, TRUNK = 50, 80, 70, 71
CAR, MOVING_CAR, PERSON, MOVING_PERSON = 10, 252, 30, 254

REMISSION = {
    ROAD: 0.25, LANE_MARKING: 0.8, SIDEWALK: 0.3, TERRAIN: 0.2, BUILDING: 0.35, POLE: 0.5,
    VEGETATION: 0.15, TRUNK: 0.2, CAR: 0.6, MOVING_CAR: 0.6, PERSON: 0.3, MOVING_PERSON: 0.3,
}  # fmt: skip


@dataclass(frozen=True)
class SensorModel:
    n_beams: int = 64
    n_azimuth: int = 1024
    fov_up_deg: float = 2.0
    fov_down_deg: float = -24.8
    min_range_m: float = 2.5
    max_range_m: float = 120.0
    range_noise_m: float = 0.01


@dataclass
class Box:
    """Axis-aligned box in the world (Velodyne frame of scan 0); ``velocity`` in m/s along world x/y."""

    lo: np.ndarray
    hi: np.ndarray
    label: int
    instance: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self.lo + self.velocity * t, self.hi + self.velocity * t


def kitti_like_tr() -> np.ndarray:
    """Velodyne -> camera-0: axis swap (x fwd -> z, y left -> -x, z up -> -y) plus a small rotation/offset."""
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    axis = np.array([0.3, 1.0, -0.2]) / np.linalg.norm([0.3, 1.0, -0.2])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    small = np.eye(3) + np.sin(0.01) * k + (1 - np.cos(0.01)) * (k @ k)
    T = np.eye(4)
    T[:3, :3] = small @ base
    T[:3, 3] = [-0.004, -0.076, -0.272]
    return T


def vehicle_trajectory(n_frames: int, dt: float = 0.1, speed_mps: float = 8.0) -> np.ndarray:
    """(n,4,4) poses of the Velodyne frame in the world; a gentle weave (yaw up to ~2 deg)."""
    poses = np.zeros((n_frames, 4, 4))
    x = y = 0.0
    for k in range(n_frames):
        t = k * dt
        yaw = 0.035 * np.sin(0.8 * t)
        c, s = np.cos(yaw), np.sin(yaw)
        poses[k] = [[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]
        x += speed_mps * dt * c
        y += speed_mps * dt * s
    return poses


def camera_poses(vehicle_poses: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    """KITTI ``poses.txt`` convention: camera-0 pose in the camera-0 frame of scan 0 (``Tr V inv(Tr)``)."""
    return Tr @ vehicle_poses @ np.linalg.inv(Tr)


def build_scene(seed: int, length_m: float = 400.0) -> list[Box]:
    rng = np.random.default_rng(seed)
    boxes: list[Box] = []

    def add(lo: tuple[float, float, float], hi: tuple[float, float, float], label: int, **kw: object) -> None:
        boxes.append(Box(np.array(lo, float), np.array(hi, float), label, **kw))  # type: ignore[arg-type]

    x_lo, x_hi = -120.0, length_m
    for side in (-1.0, 1.0):
        # 12 cm kerb face between road and sidewalk (labelled sidewalk, as in SemanticKITTI)
        y0, y1 = sorted((side * ROAD_HALF_WIDTH, side * (ROAD_HALF_WIDTH + 0.02)))
        add((x_lo, y0, ROAD_Z), (x_hi, y1, SIDEWALK_Z), SIDEWALK)
        x = x_lo + rng.uniform(0, 10)
        while x < x_hi:  # buildings with gaps
            w = rng.uniform(10, 22)
            y_near = side * rng.uniform(12, 15)
            y0, y1 = sorted((y_near, y_near + side * rng.uniform(6, 12)))
            add((x, y0, ROAD_Z), (x + w, y1, rng.uniform(5, 14)), BUILDING)
            x += w + rng.uniform(4, 12)
        x = x_lo + rng.uniform(0, 12)
        while x < x_hi:  # poles on the sidewalk
            add((x, side * 6.5 - 0.1, SIDEWALK_Z), (x + 0.2, side * 6.5 + 0.1, 4.5), POLE)
            x += rng.uniform(10, 16)
        x = x_lo + rng.uniform(0, 20)
        while x < x_hi:  # trees: trunk + raised canopy (an overhang)
            yc = side * 9.5
            add((x - 0.15, yc - 0.15, TERRAIN_Z), (x + 0.15, yc + 0.15, 0.6), TRUNK)
            add((x - 1.5, yc - 1.5, 0.6), (x + 1.5, yc + 1.5, 3.0), VEGETATION)
            x += rng.uniform(15, 30)
    instance = 1
    x = 8.0
    while x < x_hi:  # parked cars along the right lane edge
        add((x, -4.7, ROAD_Z), (x + 4.2, -2.9, ROAD_Z + 1.5), CAR, instance=instance)
        instance += 1
        x += rng.uniform(9, 20)
    add((20.0, 1.2, ROAD_Z), (24.4, 3.0, ROAD_Z + 1.5), MOVING_CAR, instance=instance,
        velocity=np.array([13.0, 0.0, 0.0]))  # fmt: skip
    add((12.0, 5.8, SIDEWALK_Z), (12.5, 6.3, SIDEWALK_Z + 1.75), MOVING_PERSON, instance=instance + 1,
        velocity=np.array([1.4, 0.0, 0.0]))  # fmt: skip
    add((30.0, -6.3, SIDEWALK_Z), (30.5, -5.8, SIDEWALK_Z + 1.75), PERSON, instance=instance + 2)
    return boxes


def _ground_label(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ay = np.abs(y)
    z = np.where(ay < ROAD_HALF_WIDTH, ROAD_Z, np.where(ay < SIDEWALK_OUTER, SIDEWALK_Z, TERRAIN_Z))
    label = np.where(ay < ROAD_HALF_WIDTH, ROAD, np.where(ay < SIDEWALK_OUTER, SIDEWALK, TERRAIN))
    label = np.where((ay > 1.95) & (ay < 2.1), LANE_MARKING, label)
    return z, label


def _candidate_rays(
    lo: np.ndarray, hi: np.ndarray, pose: np.ndarray, ray_grid: np.ndarray, az_step: float
) -> np.ndarray:
    """Indices of rays whose azimuth column can hit the box (all rays if the box surrounds the sensor)."""
    corners = np.array([[x, y, 0.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    local = (corners - pose[:3, 3]) @ pose[:3, :3]
    az = np.arctan2(local[:, 1], local[:, 0])
    ref = az[0]
    rel = (az - ref + np.pi) % (2 * np.pi) - np.pi  # unwrap around the first corner
    if rel.max() - rel.min() >= np.pi:
        return ray_grid.ravel()
    n_az = ray_grid.shape[1]
    first = int(np.floor((ref + rel.min() + np.pi) / az_step)) - 1
    last = int(np.ceil((ref + rel.max() + np.pi) / az_step)) + 1
    cols = np.arange(first, last + 1) % n_az
    return ray_grid[:, cols].ravel()


def render_scan(
    boxes: list[Box], pose: np.ndarray, t: float, rng: np.random.Generator, sensor: SensorModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-cast one scan. Returns xyz (N,3) float32 in the Velodyne frame, remission, raw uint32 labels."""
    elev = np.deg2rad(np.linspace(sensor.fov_up_deg, sensor.fov_down_deg, sensor.n_beams))
    azim = np.linspace(-np.pi, np.pi, sensor.n_azimuth, endpoint=False)
    ee, aa = np.meshgrid(elev, azim, indexing="ij")
    d_local = np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], axis=-1).reshape(-1, 3)
    d = d_local @ pose[:3, :3].T
    o = pose[:3, 3]
    best = np.full(len(d), np.inf)
    label = np.zeros(len(d), dtype=np.uint32)

    # ground: three horizontal planes, each valid only over its own y band
    for z_plane in (ROAD_Z, SIDEWALK_Z, TERRAIN_Z):
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (z_plane - o[2]) / d[:, 2]
        hit = (tt > 0) & (tt < best)
        y_hit = o[1] + tt * d[:, 1]
        z_expected, lab = _ground_label(y_hit)
        hit &= np.isclose(z_expected, z_plane)
        best = np.where(hit, tt, best)
        label = np.where(hit, lab.astype(np.uint32), label)

    reach = sensor.max_range_m + 10.0
    inv_d = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    ray_grid = np.arange(len(d)).reshape(sensor.n_beams, sensor.n_azimuth)
    az_step = 2 * np.pi / sensor.n_azimuth
    for box in boxes:
        lo, hi = box.at(t)
        if np.any(np.maximum(lo - o, o - hi)[:2] > reach):
            continue
        rays = _candidate_rays(lo, hi, pose, ray_grid, az_step)
        t1, t2 = (lo - o) * inv_d[rays], (hi - o) * inv_d[rays]
        t_near = np.max(np.minimum(t1, t2), axis=1)
        t_far = np.min(np.maximum(t1, t2), axis=1)
        hit = (t_near <= t_far) & (t_near > 0) & (t_near < best[rays])
        best[rays[hit]] = t_near[hit]
        label[rays[hit]] = np.uint32(box.label | (box.instance << 16))

    valid = np.isfinite(best) & (best >= sensor.min_range_m) & (best <= sensor.max_range_m)
    r = best[valid] + rng.normal(scale=sensor.range_noise_m, size=int(valid.sum()))
    xyz = (d_local[valid] * r[:, None]).astype(np.float32)
    raw = label[valid]
    base = np.array([REMISSION.get(int(s), 0.3) for s in (raw & 0xFFFF)], dtype=np.float32)
    remission = np.clip(base + rng.normal(scale=0.03, size=len(base)), 0, 1).astype(np.float32)
    return xyz, remission, raw


@dataclass(frozen=True)
class SyntheticSequence:
    root: Path
    seq: str
    Tr: np.ndarray
    vehicle_poses: np.ndarray  # Velodyne-frame poses in the world (ground truth)
    n_frames: int


def _fmt(matrix: np.ndarray) -> str:
    return " ".join(f"{v:.12e}" for v in matrix.ravel())


def write_kitti_sequence(
    root: str | Path,
    seq: str = "08",
    n_frames: int = 10,
    seed: int = 1337,
    poses_in_poses_dir: bool = False,
    sensor: SensorModel | None = None,
) -> SyntheticSequence:
    """Write a SemanticKITTI-layout sequence under ``root/sequences/<seq>`` (README 5.1).

    With ``poses_in_poses_dir`` the poses go to ``root/poses/<seq>.txt`` as shipped by
    ``data_odometry_poses.zip`` instead of ``sequences/<seq>/poses.txt``.
    """
    sensor = sensor or SensorModel()
    root = Path(root)
    seq_dir = root / "sequences" / seq
    (seq_dir / "velodyne").mkdir(parents=True, exist_ok=True)
    (seq_dir / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Tr = kitti_like_tr()
    vehicle = vehicle_trajectory(n_frames)
    boxes = build_scene(seed)
    for k in range(n_frames):
        xyz, remission, raw = render_scan(boxes, vehicle[k], k * 0.1, rng, sensor)
        np.hstack([xyz, remission[:, None]]).astype("<f4").tofile(seq_dir / "velodyne" / f"{k:06d}.bin")
        raw.astype("<u4").tofile(seq_dir / "labels" / f"{k:06d}.label")
    projection = np.array([[718.856, 0, 607.1928, 0], [0, 718.856, 185.2157, 0], [0, 0, 1, 0]])
    calib = [f"P{i}: {_fmt(projection)}" for i in range(4)] + [f"Tr: {_fmt(Tr[:3])}"]
    (seq_dir / "calib.txt").write_text("\n".join(calib) + "\n", encoding="utf-8")
    (seq_dir / "times.txt").write_text("".join(f"{k * 0.1:.6e}\n" for k in range(n_frames)), encoding="utf-8")
    pose_lines = "".join(_fmt(P[:3]) + "\n" for P in camera_poses(vehicle, Tr))
    if poses_in_poses_dir:
        (root / "poses").mkdir(parents=True, exist_ok=True)
        (root / "poses" / f"{seq}.txt").write_text(pose_lines, encoding="utf-8")
    else:
        (seq_dir / "poses.txt").write_text(pose_lines, encoding="utf-8")
    return SyntheticSequence(root=root, seq=seq, Tr=Tr, vehicle_poses=vehicle, n_frames=n_frames)
