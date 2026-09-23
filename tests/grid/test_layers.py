"""T6.2: finalize -> packed 12-byte layers (README 6.6), every branch on hand-built cells."""

from __future__ import annotations

import numpy as np
import pytest

from foveamap.config import GridConfig
from foveamap.grid.accumulators import INT32_MAX, INT32_MIN, GridAccumulators, RingAccumulators
from foveamap.grid.engine import rasterize
from foveamap.grid.layers import (
    FLAG_HAS_GROUND,
    FLAG_HAS_OBSTACLE,
    FLAG_HAS_OVERHANG,
    INT16_MIN,
    LAYER_DTYPE,
    cell_layers,
    finalize,
)
from foveamap.grid.presets import GridSpec
from foveamap.io.labels import DRIVABLE, DYNAMIC, NON_DRIVABLE_TERRAIN, STATIC_OBSTACLE, UNKNOWN

from .conftest import labelled_points


def cell(
    n_cls: tuple[int, int, int, int, int],
    ground: tuple[int, int, int] | None = None,  # (sum, min, max) of ground z in mm
    obstacle: tuple[int, int] | None = None,  # (min, max) of obstacle z in mm
    moving: int = 0,
    conf_sum: int | None = None,
) -> dict:
    n = sum(n_cls)
    g_sum, g_min, g_max = ground if ground else (0, INT32_MAX, INT32_MIN)
    o_min, o_max = obstacle if obstacle else (INT32_MAX, INT32_MIN)
    return {"n_total": n, "n_cls": list(n_cls), "n_moving": moving, "g_sum": g_sum, "g_min": g_min,
            "g_max": g_max, "o_min": o_min, "o_max": o_max, "conf_sum": 255 * n if conf_sum is None else conf_sum}  # fmt: skip


def ring_of(cells: list[dict], side: int = 16) -> RingAccumulators:
    from foveamap.grid.accumulators import DTYPES

    m = len(cells)
    values = {k: np.array([c[k] for c in cells], dtype=DTYPES[k]) for k in cells[0]} if cells else {}
    if not cells:
        return RingAccumulators.empty(0, 50, side)
    values["n_cls"] = values["n_cls"].reshape(m, 5)
    return RingAccumulators(0, 50, side, np.arange(m, dtype=np.int64), **values)


def one(c: dict, cfg: GridConfig) -> np.void:
    return cell_layers(ring_of([c]), cfg)[0]


def with_rule(cfg: GridConfig, **changes: object) -> GridConfig:
    return cfg.model_copy(update=changes)


def test_layout_is_12_bytes() -> None:
    assert LAYER_DTYPE.itemsize == 12
    assert LAYER_DTYPE.names == (
        "ground_z",
        "top_z",
        "clearance",
        "cls",
        "moving_frac",
        "count",
        "conf",
        "flags",
    )


def test_grounded_obstacle(grid_cfg: GridConfig) -> None:
    # ground at -1.700 m, obstacle bottom 100 mm above it (< 300 mm contact height): grounded, no overhang
    v = one(cell((0, 2, 0, 3, 0), ground=(-3400, -1700, -1700), obstacle=(-1600, 400)), grid_cfg)
    assert (v["ground_z"], v["top_z"], v["clearance"]) == (-1700, 400, 0)
    assert v["cls"] == STATIC_OBSTACLE
    assert v["flags"] == FLAG_HAS_GROUND | FLAG_HAS_OBSTACLE


def test_overhang(grid_cfg: GridConfig) -> None:
    # branch 2.1 m above the road: clearance = gap, has_overhang
    v = one(cell((0, 6, 0, 2, 0), ground=(-10200, -1710, -1690), obstacle=(400, 900)), grid_cfg)
    assert v["ground_z"] == -1700 and v["top_z"] == 900
    assert v["clearance"] == 400 - (-1700)
    assert v["flags"] == FLAG_HAS_GROUND | FLAG_HAS_OBSTACLE | FLAG_HAS_OVERHANG


def test_contact_height_boundary(grid_cfg: GridConfig) -> None:
    exactly = one(cell((0, 1, 0, 2, 0), ground=(-1700, -1700, -1700), obstacle=(-1400, 0)), grid_cfg)
    below = one(cell((0, 1, 0, 2, 0), ground=(-1700, -1700, -1700), obstacle=(-1401, 0)), grid_cfg)
    assert exactly["clearance"] == 300 and exactly["flags"] & FLAG_HAS_OVERHANG  # gap == contact -> overhang
    assert below["clearance"] == 0 and not below["flags"] & FLAG_HAS_OVERHANG


def test_obstacle_without_ground(grid_cfg: GridConfig) -> None:
    v = one(cell((0, 0, 0, 4, 0), obstacle=(-1000, 3000)), grid_cfg)
    assert v["ground_z"] == INT16_MIN and v["clearance"] == INT16_MIN and v["top_z"] == 3000
    assert v["flags"] == FLAG_HAS_OBSTACLE


def test_ground_without_obstacle(grid_cfg: GridConfig) -> None:
    v = one(cell((0, 3, 0, 0, 0), ground=(-5100, -1710, -1690)), grid_cfg)
    assert (v["ground_z"], v["top_z"], v["clearance"], v["cls"]) == (-1700, INT16_MIN, 0, DRIVABLE)


def test_integer_rounding(grid_cfg: GridConfig) -> None:
    # (g_sum + n // 2) // n: -3401 / 2 -> -1700 (-1700.5 rounds half up); -3403 / 2 -> -1701
    assert one(cell((0, 2, 0, 0, 0), ground=(-3401, -1701, -1700)), grid_cfg)["ground_z"] == -1700
    assert one(cell((0, 2, 0, 0, 0), ground=(-3403, -1702, -1701)), grid_cfg)["ground_z"] == -1701
    v = one(cell((0, 2, 1, 0, 0), ground=(-5100, -1700, -1700), moving=1, conf_sum=200), grid_cfg)
    assert v["moving_frac"] == 85  # round(255 / 3) = 85
    assert v["conf"] == 67  # round(200 / 3) = 66.67 -> 67


def test_ground_estimator_min(grid_cfg: GridConfig) -> None:
    cfg = with_rule(grid_cfg, ground_estimator="min")
    assert one(cell((0, 3, 0, 0, 0), ground=(-5100, -1750, -1650)), cfg)["ground_z"] == -1750


def test_clearance_saturates_int16(grid_cfg: GridConfig) -> None:
    v = one(cell((0, 1, 0, 1, 0), ground=(-32767, -32767, -32767), obstacle=(32767, 32767)), grid_cfg)
    assert v["clearance"] == 32767  # gap 65534 mm saturates, never wraps


# ── class rule (safety, default) ────────────────────────────────────────────


def test_dynamic_priority(grid_cfg: GridConfig) -> None:
    # 2 dynamic of 8 points: 2 >= max(2, 0.2 * 8 = 1.6) -> DYNAMIC even though ground dominates
    assert (
        one(cell((0, 6, 0, 0, 2), ground=(-10200, -1700, -1700), obstacle=(-1500, 0)), grid_cfg)["cls"]
        == DYNAMIC
    )


def test_dynamic_below_threshold(grid_cfg: GridConfig) -> None:
    # 2 dynamic of 12: 2 < 0.2 * 12 = 2.4 -> not DYNAMIC; 0 static -> ground argmax
    v = one(cell((0, 10, 0, 0, 2), ground=(-17000, -1700, -1700), obstacle=(-1500, 0)), grid_cfg)
    assert v["cls"] == DRIVABLE


def test_single_dynamic_point_not_enough(grid_cfg: GridConfig) -> None:
    assert (
        one(cell((0, 0, 0, 0, 1), obstacle=(0, 0)), grid_cfg)["cls"] == NON_DRIVABLE_TERRAIN
    )  # all-zero tie


def test_static_obstacle_threshold(grid_cfg: GridConfig) -> None:
    assert (
        one(cell((0, 8, 0, 2, 0), ground=(-13600, -1700, -1700), obstacle=(-1600, 0)), grid_cfg)["cls"]
        == STATIC_OBSTACLE
    )
    assert (
        one(cell((0, 9, 0, 2, 0), ground=(-15300, -1700, -1700), obstacle=(-1600, 0)), grid_cfg)["cls"]
        == DRIVABLE
    )


def test_fraction_threshold_is_exact(grid_cfg: GridConfig) -> None:
    # exactly 20 %: 3 of 15 -> meets 0.20 exactly (no float rounding surprises)
    assert (
        one(cell((0, 12, 0, 3, 0), ground=(-20400, -1700, -1700), obstacle=(-1600, 0)), grid_cfg)["cls"]
        == STATIC_OBSTACLE
    )


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((0, 2, 2, 0, 0), NON_DRIVABLE_TERRAIN),  # drivable vs non-drivable tie -> non-drivable
        ((2, 2, 0, 0, 0), UNKNOWN),  # unknown vs drivable tie -> unknown (unknown is not free)
        ((2, 0, 2, 0, 0), NON_DRIVABLE_TERRAIN),
        ((1, 3, 2, 0, 0), DRIVABLE),
        ((3, 1, 1, 0, 0), UNKNOWN),
    ],
)
def test_ground_argmax_ties_are_conservative(grid_cfg: GridConfig, counts: tuple, expected: int) -> None:
    n_ground = counts[1] + counts[2]
    ground = (-1700 * n_ground, -1700, -1700) if n_ground else None
    assert one(cell(counts, ground=ground), grid_cfg)["cls"] == expected


# ── class rule: majority (ablation) ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((0, 6, 0, 0, 2), DRIVABLE),  # plain argmax, no dynamic priority
        ((0, 2, 0, 0, 2), DYNAMIC),  # ties -> most conservative
        ((0, 2, 0, 2, 0), STATIC_OBSTACLE),
        ((0, 2, 2, 0, 0), NON_DRIVABLE_TERRAIN),
        ((2, 2, 0, 0, 0), UNKNOWN),
    ],
)
def test_majority_rule(grid_cfg: GridConfig, counts: tuple, expected: int) -> None:
    cfg = with_rule(grid_cfg, class_rule="majority")
    n_ground, n_obs = counts[1] + counts[2], counts[3] + counts[4]
    ground = (-1700 * n_ground, -1700, -1700) if n_ground else None
    obstacle = (-1600, 0) if n_obs else None
    assert one(cell(counts, ground=ground, obstacle=obstacle), cfg)["cls"] == expected


def test_count_saturates(grid_cfg: GridConfig) -> None:
    v = one(cell((0, 60_000, 10_000, 0, 0), ground=(-1700 * 70_000, -1700, -1700)), grid_cfg)
    assert v["count"] == 65535


# ── finalize on real accumulators ───────────────────────────────────────────


def test_finalize_dense_layout(presets: dict[str, GridSpec], grid_cfg: GridConfig) -> None:
    spec = presets["tiny_test"]
    rng = np.random.default_rng(9)
    xyz_mm, cls, moving, conf = labelled_points(rng, 4_000, spec.extent_mm)
    acc = rasterize(spec, xyz_mm / 1000.0, cls, moving, conf)
    layers = finalize(acc, grid_cfg)
    assert layers.spec is spec
    for k, (grid, r) in enumerate(zip(layers.rings, acc.rings, strict=True)):
        assert grid.dtype == LAYER_DTYPE and grid.shape == (r.side, r.side)
        flat = grid.reshape(-1)
        empty = np.ones(r.side * r.side, dtype=bool)
        empty[r.cells] = False
        assert (flat["count"][empty] == 0).all() and (flat["cls"][empty] == UNKNOWN).all()
        assert (flat["ground_z"][empty] == INT16_MIN).all() and (flat["top_z"][empty] == INT16_MIN).all()
        assert (flat["clearance"][empty] == INT16_MIN).all() and (flat["flags"][empty] == 0).all()
        np.testing.assert_array_equal(flat["count"][r.cells], np.minimum(r.n_total, 65535))
        np.testing.assert_array_equal(flat[r.cells], cell_layers(r, grid_cfg))
        assert layers.layer("count", k).shape == (r.side, r.side)
    assert layers.nbytes() == spec.allocated_cells() * 12
    assert int(sum(g["count"].sum(dtype=np.int64) for g in layers.rings)) == acc.counters.n_in_grid


def test_finalize_empty_grid(presets: dict[str, GridSpec], grid_cfg: GridConfig) -> None:
    spec = presets["tiny_ps_test"]
    acc = GridAccumulators([RingAccumulators.empty(k, r.cell_mm, r.side) for k, r in enumerate(spec.rings)])
    layers = finalize(acc, grid_cfg, spec)
    assert all((g["cls"] == UNKNOWN).all() and (g["ground_z"] == INT16_MIN).all() for g in layers.rings)
