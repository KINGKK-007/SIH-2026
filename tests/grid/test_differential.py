"""T6.1: I7 differential test against a deliberately naive dict-based implementation (README 6.5.5).

The naive version uses only Python ints, one point at a time, and restates README 6.5.3/6.5.4 directly.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from foveamap.grid.engine import rasterize
from foveamap.grid.presets import GridSpec

from .test_consistency import points_strategy, to_arrays

SETTINGS = settings(
    max_examples=200, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
BIG, SMALL = 2**31 - 1, -(2**31)


def naive_rasterize(spec: GridSpec, pts: list[tuple[int, int, int, int, bool, int]]) -> dict:
    cells: dict[tuple[int, int, int], dict] = {}
    out = 0
    for x, y, z, cls, moving, conf in pts:
        z = max(-32767, min(32767, z))
        ring = None
        for k, rs in enumerate(spec.rings):
            if -rs.r_max_mm <= x < rs.r_max_mm and -rs.r_max_mm <= y < rs.r_max_mm:
                ring = k
                break
        if ring is None:
            out += 1
            continue
        rs = spec.rings[ring]
        key = (ring, math.floor(x / rs.cell_mm) + rs.r_max_mm // rs.cell_mm,
               math.floor(y / rs.cell_mm) + rs.r_max_mm // rs.cell_mm)  # fmt: skip
        c = cells.setdefault(key, {"n": 0, "cls": [0] * 5, "mov": 0, "gs": 0, "gmin": BIG, "gmax": SMALL,
                                   "omin": BIG, "omax": SMALL, "conf": 0})  # fmt: skip
        c["n"] += 1
        c["cls"][cls] += 1
        c["mov"] += int(moving)
        c["conf"] += conf
        if cls in (1, 2):
            c["gs"] += z
            c["gmin"], c["gmax"] = min(c["gmin"], z), max(c["gmax"], z)
        if cls in (3, 4):
            c["omin"], c["omax"] = min(c["omin"], z), max(c["omax"], z)
    return {"cells": cells, "out": out}


def compare(spec: GridSpec, pts: list) -> None:
    naive = naive_rasterize(spec, pts)
    xyz_m, cls, moving, conf = to_arrays(pts)
    acc = rasterize(spec, xyz_m, cls, moving, conf)
    assert acc.counters.n_out_of_grid == naive["out"]
    for k, r in enumerate(acc.rings):
        expected = sorted(
            ((iy * r.side + ix), c) for (ring, ix, iy), c in naive["cells"].items() if ring == k
        )
        assert r.cells.tolist() == [flat for flat, _ in expected]
        for i, (_, c) in enumerate(expected):
            got = (int(r.n_total[i]), r.n_cls[i].tolist(), int(r.n_moving[i]), int(r.g_sum[i]), int(r.g_min[i]),
                   int(r.g_max[i]), int(r.o_min[i]), int(r.o_max[i]), int(r.conf_sum[i]))  # fmt: skip
            want = (
                c["n"],
                c["cls"],
                c["mov"],
                c["gs"],
                c["gmin"],
                c["gmax"],
                c["omin"],
                c["omax"],
                c["conf"],
            )
            assert got == want


@SETTINGS
@given(st.data())
def test_i7_tiny_test(presets: dict[str, GridSpec], data: st.DataObject) -> None:
    spec = presets["tiny_test"]
    compare(spec, data.draw(points_strategy(spec)))


@SETTINGS
@given(st.data())
def test_i7_tiny_ps_test(presets: dict[str, GridSpec], data: st.DataObject) -> None:
    spec = presets["tiny_ps_test"]
    compare(spec, data.draw(points_strategy(spec)))


@SETTINGS
@given(st.data())
def test_i7_uniform_single_ring(presets: dict[str, GridSpec], data: st.DataObject) -> None:
    spec = presets["uniform_40cm"]
    compare(spec, data.draw(points_strategy(spec, max_n=300)))


@SETTINGS
@given(st.data())
def test_i7_fovea_default(presets: dict[str, GridSpec], data: st.DataObject) -> None:
    spec = presets["fovea_default"]
    compare(spec, data.draw(points_strategy(spec, max_n=300)))


def test_i7_clustered_points(presets: dict[str, GridSpec]) -> None:
    rng = np.random.default_rng(3)
    pts = [(int(x), int(y), int(z), int(c), bool(m), int(f)) for x, y, z, c, m, f in zip(
        rng.integers(-500, 500, 3_000), rng.integers(-500, 500, 3_000), rng.integers(-40_000, 40_000, 3_000),
        rng.integers(0, 5, 3_000), rng.random(3_000) < 0.5, rng.integers(0, 256, 3_000), strict=True)]  # fmt: skip
    for name in ("tiny_test", "tiny_ps_test", "uniform_5cm"):
        compare(presets[name], pts)
