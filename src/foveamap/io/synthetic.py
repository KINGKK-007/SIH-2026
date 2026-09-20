"""foveamap.io.synthetic — Offline scan generator for no-data operation.

Produces a synthetic Velodyne-like point cloud that mimics the structure of a
real HDL-64E scan without requiring any downloaded data.  The generator is
deterministic given a seed and is used by:

- ``foveamap check-data`` when the real dataset is absent (G2 fallback).
- Unit tests that need point clouds without heavy fixtures.
- Offline CI runs on synthetic data.

Scene composition
-----------------
The synthetic scene (Velodyne frame, z-up) contains:

Ground plane
    Flat road between roughly −8 m and +40 m along x, ±6 m in y.
    Ground points sit at ``z ≈ −1.73 m`` (sensor height), with ±1 cm noise.
    Labelled: road (raw ID 40) → DRIVABLE.

Sidewalks
    Two strips at ±7 m in y, slightly raised (0.12 m step).
    Labelled: sidewalk (raw ID 48) → NON_DRIVABLE_TERRAIN.

Buildings
    Three rectangular façades at varying distances (6–30 m range).
    Labelled: building (raw ID 50) → STATIC_OBSTACLE.

Parked cars
    Two axis-aligned boxes at ∼5 m and ∼15 m range.
    Labelled: car (raw ID 10) → STATIC_OBSTACLE.

Moving car
    One box at ∼8 m range.
    Labelled: moving-car (raw ID 252) → DYNAMIC.

Pedestrian
    A small vertical cluster at ∼6 m range.
    Labelled: person (raw ID 30) → DYNAMIC.

Terrain
    Random open-ground points beyond the road boundary.
    Labelled: terrain (raw ID 72) → NON_DRIVABLE_TERRAIN.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Sensor mount height above flat ground (Velodyne frame)
_SENSOR_HEIGHT_M: float = 1.73


def _box_points(
    rng: np.random.Generator,
    cx: float,
    cy: float,
    cz_bot: float,
    length: float,
    width: float,
    height: float,
    n: int,
) -> NDArray[np.float32]:
    """Sample ``n`` points uniformly on the outer surface of an axis-aligned box."""
    # Distribute n points across 6 faces proportional to face area
    faces = np.array([
        length * width,    # bottom
        length * width,    # top
        length * height,   # front
        length * height,   # back
        width * height,    # left
        width * height,    # right
    ])
    faces_normed = faces / faces.sum()
    counts = rng.multinomial(n, faces_normed)

    pts: list[NDArray[np.float32]] = []

    def _face(axis_fixed: int, val: float, axes: tuple[int, int], lo: tuple[float, float], hi: tuple[float, float], k: int) -> NDArray[np.float32]:
        a = rng.uniform(lo[0], hi[0], k).astype(np.float32)
        b = rng.uniform(lo[1], hi[1], k).astype(np.float32)
        p = np.zeros((k, 3), dtype=np.float32)
        p[:, axis_fixed] = val
        p[:, axes[0]] = a
        p[:, axes[1]] = b
        return p

    half_l, half_w = length / 2, width / 2
    # bottom / top
    for z_val, cnt in [(cz_bot, counts[0]), (cz_bot + height, counts[1])]:
        pts.append(_face(2, z_val, (0, 1), (cx - half_l, cy - half_w), (cx + half_l, cy + half_w), cnt))
    # front / back
    for x_val, cnt in [(cx + half_l, counts[2]), (cx - half_l, counts[3])]:
        pts.append(_face(0, x_val, (1, 2), (cy - half_w, cz_bot), (cy + half_w, cz_bot + height), cnt))
    # left / right
    for y_val, cnt in [(cy + half_w, counts[4]), (cy - half_w, counts[5])]:
        pts.append(_face(1, y_val, (0, 2), (cx - half_l, cz_bot), (cx + half_l, cz_bot + height), cnt))

    return np.vstack(pts)


def generate_synthetic_scan(
    num_points: int = 120_000,
    seed: int = 42,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Generate a synthetic Velodyne-like point cloud with ground-truth labels.

    The scan is in the **Velodyne frame** (x forward, y left, z up) with the
    sensor at height ≈ 1.73 m above the ground plane.

    Parameters
    ----------
    num_points:
        Total number of points to generate.  Defaults to 120,000, matching a
        typical Velodyne HDL-64E scan.
    seed:
        Random seed for reproducibility.  Defaults to 42.

    Returns
    -------
    points : NDArray[float32]
        Shape ``(N, 4)`` — ``[x, y, z, intensity]`` in the Velodyne frame.
        Range is approximately 0–80 m; intensity in ``[0, 1]``.
    labels : NDArray[uint32]
        Shape ``(N,)`` — raw ``uint32`` SemanticKITTI labels.
        Bit layout: ``[0x0000 | semantic_id]`` (instance IDs are zero).

    Notes
    -----
    - Points are generated component-by-component rather than via full
      ray-casting, so the density distribution is not exactly HDL-64E.
      The purpose is offline functional testing, not geometric fidelity.
    - The intensity column is drawn from ``Uniform(0.1, 0.9)`` per region.
    """
    rng = np.random.default_rng(seed)
    gz = -_SENSOR_HEIGHT_M  # ground z in Velodyne frame

    # ── Component budgets (fractions of total) ───────────────────────────────
    budgets = {
        "road":      0.30,
        "sidewalk":  0.10,
        "terrain":   0.10,
        "building":  0.20,
        "car_static":0.12,
        "car_moving":0.06,
        "pedestrian":0.03,
        "scatter":   0.09,   # background noise / vegetation
    }
    assert abs(sum(budgets.values()) - 1.0) < 1e-6, "Budgets must sum to 1"
    counts = {k: int(v * num_points) for k, v in budgets.items()}
    # Distribute any rounding deficit into the largest bucket so total == num_points
    deficit = num_points - sum(counts.values())
    largest_key = max(counts, key=lambda k: counts[k])
    counts[largest_key] += deficit

    xyz_parts: list[NDArray[np.float32]] = []
    inten_parts: list[NDArray[np.float32]] = []
    label_parts: list[NDArray[np.uint32]] = []

    def _add(xyz: NDArray[np.float32], inten: NDArray[np.float32], raw_id: int) -> None:
        xyz_parts.append(xyz)
        inten_parts.append(inten)
        label_parts.append(np.full(len(xyz), raw_id, dtype=np.uint32))

    # ── Road ─────────────────────────────────────────────────────────────────
    n = counts["road"]
    x = rng.uniform(-10.0, 50.0, n).astype(np.float32)
    y = rng.uniform(-5.5, 5.5, n).astype(np.float32)
    z = np.full(n, gz, dtype=np.float32) + rng.normal(0, 0.01, n).astype(np.float32)
    _add(np.column_stack([x, y, z]), rng.uniform(0.2, 0.6, n).astype(np.float32), 40)

    # ── Sidewalks (±7 m) ─────────────────────────────────────────────────────
    n = counts["sidewalk"]
    n_side = n // 2
    for y_sign in [1, -1]:
        x_s = rng.uniform(-10.0, 50.0, n_side).astype(np.float32)
        y_s = (y_sign * rng.uniform(5.8, 8.5, n_side)).astype(np.float32)
        z_s = np.full(n_side, gz + 0.12, dtype=np.float32) + rng.normal(0, 0.01, n_side).astype(np.float32)
        _add(np.column_stack([x_s, y_s, z_s]), rng.uniform(0.15, 0.45, n_side).astype(np.float32), 48)

    # ── Terrain (open ground) ─────────────────────────────────────────────────
    n = counts["terrain"]
    x_t = rng.uniform(-20.0, 80.0, n).astype(np.float32)
    y_t = (np.sign(rng.uniform(-1, 1, n)) * rng.uniform(9.0, 30.0, n)).astype(np.float32)
    z_t = np.full(n, gz, dtype=np.float32) + rng.normal(0, 0.05, n).astype(np.float32)
    _add(np.column_stack([x_t, y_t, z_t]), rng.uniform(0.1, 0.4, n).astype(np.float32), 72)

    # ── Buildings ─────────────────────────────────────────────────────────────
    n_per_building = counts["building"] // 3
    building_specs = [
        (25.0, 10.0, gz, 15.0, 4.0, 8.0),
        (15.0, -12.0, gz, 20.0, 6.0, 6.0),
        (40.0, 0.0, gz, 8.0, 20.0, 10.0),
    ]
    for cx, cy, cz_bot, l, w, h in building_specs:
        pts = _box_points(rng, cx, cy, cz_bot, l, w, h, n_per_building)
        _add(pts.astype(np.float32), rng.uniform(0.3, 0.8, len(pts)).astype(np.float32), 50)

    # ── Parked cars ───────────────────────────────────────────────────────────
    n_car = counts["car_static"] // 2
    for cx, cy in [(5.0, 7.5), (15.0, -7.0)]:
        pts = _box_points(rng, cx, cy, gz, 4.5, 2.0, 1.5, n_car)
        _add(pts.astype(np.float32), rng.uniform(0.4, 0.9, len(pts)).astype(np.float32), 10)

    # ── Moving car ────────────────────────────────────────────────────────────
    n = counts["car_moving"]
    pts = _box_points(rng, 8.0, -3.0, gz, 4.5, 2.0, 1.5, n)
    _add(pts.astype(np.float32), rng.uniform(0.4, 0.9, n).astype(np.float32), 252)

    # ── Pedestrian ────────────────────────────────────────────────────────────
    n = counts["pedestrian"]
    x_p = rng.normal(6.0, 0.2, n).astype(np.float32)
    y_p = rng.normal(3.5, 0.2, n).astype(np.float32)
    z_p = rng.uniform(gz, gz + 1.8, n).astype(np.float32)
    _add(np.column_stack([x_p, y_p, z_p]), rng.uniform(0.2, 0.6, n).astype(np.float32), 30)

    # ── Background scatter (vegetation / outliers) ────────────────────────────
    n = counts["scatter"]
    x_sc = rng.uniform(-20.0, 80.0, n).astype(np.float32)
    y_sc = rng.uniform(-30.0, 30.0, n).astype(np.float32)
    z_sc = rng.uniform(gz, gz + 3.0, n).astype(np.float32)
    _add(np.column_stack([x_sc, y_sc, z_sc]), rng.uniform(0.05, 0.3, n).astype(np.float32), 70)

    # ── Assemble ──────────────────────────────────────────────────────────────
    xyz = np.vstack(xyz_parts)
    inten = np.concatenate(inten_parts)
    labels = np.concatenate(label_parts)

    # Trim or pad to exactly num_points (integer division rounding in component
    # budgets can leave a small deficit or surplus)
    current_n = len(labels)
    if current_n > num_points:
        xyz = xyz[:num_points]
        inten = inten[:num_points]
        labels = labels[:num_points]
    elif current_n < num_points:
        deficit = num_points - current_n
        # Pad with additional road points
        x_pad = rng.uniform(-10.0, 50.0, deficit).astype(np.float32)
        y_pad = rng.uniform(-5.5, 5.5, deficit).astype(np.float32)
        z_pad = np.full(deficit, gz, dtype=np.float32) + rng.normal(0, 0.01, deficit).astype(np.float32)
        xyz = np.vstack([xyz, np.column_stack([x_pad, y_pad, z_pad])])
        inten = np.concatenate([inten, rng.uniform(0.2, 0.6, deficit).astype(np.float32)])
        labels = np.concatenate([labels, np.full(deficit, 40, dtype=np.uint32)])

    # Shuffle so point order doesn't encode structure
    perm = rng.permutation(num_points)
    points = np.column_stack([xyz[perm], inten[perm]]).astype(np.float32)
    labels = labels[perm]

    return points, labels
