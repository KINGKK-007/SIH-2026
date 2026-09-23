"""T6.1: I4 fine->coarse bit-identity and I6 permutation invariance / determinism (README 6.5.5)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from foveamap.grid.accumulators import GridAccumulators, RingAccumulators, reduce_block
from foveamap.grid.engine import locate, rasterize
from foveamap.grid.presets import GridSpec, RingSpec

SETTINGS = settings(
    max_examples=200, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


def points_strategy(spec: GridSpec, max_n: int = 400) -> st.SearchStrategy:
    """Labelled integer-mm points that stress the grid: clusters (many points per cell and per coarse block)
    plus exact boundary coordinates (+-R_k, +-R_k -+ 1 mm, multiples of every cell size) and uniform noise.

    Hypothesis draws the structure (seed, size, cluster count and spread, mix); NumPy generates the points,
    which keeps 200 examples per property fast.
    """
    extent = spec.extent_mm
    boundary = np.array(sorted({v for r in spec.rings for base in (r.r_max_mm, r.cell_mm, 2 * r.cell_mm)
                                for v in (base, -base, base - 1, -base + 1, base + 1, -base - 1)} | {0, -1}))  # fmt: skip
    lim = extent + extent // 5

    @st.composite
    def build(draw: st.DrawFn) -> list[tuple[int, int, int, int, bool, int]]:
        rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
        n = draw(st.integers(0, max_n))
        n_centres = draw(st.integers(1, 6))
        spread = draw(st.sampled_from([50, 300, 1_500]))
        frac_boundary = draw(st.sampled_from([0.0, 0.2, 0.5]))
        centres = rng.integers(-lim, lim, size=(n_centres, 2))
        xy = centres[rng.integers(0, n_centres, n)] + rng.integers(-spread, spread + 1, size=(n, 2))
        mode = rng.random(n)
        on_boundary = mode < frac_boundary
        xy[on_boundary, 0] = rng.choice(boundary, int(on_boundary.sum()))
        both = on_boundary & (rng.random(n) < 0.5)
        xy[both, 1] = rng.choice(boundary, int(both.sum()))
        noise = mode > 0.9
        xy[noise] = rng.integers(-lim, lim, size=(int(noise.sum()), 2))
        z = rng.integers(-40_000, 40_001, n)
        cls, mov, conf = rng.integers(0, 5, n), rng.random(n) < 0.3, rng.integers(0, 256, n)
        return [(int(a), int(b), int(c), int(d), bool(e), int(f))
                for a, b, c, d, e, f in zip(xy[:, 0], xy[:, 1], z, cls, mov, conf, strict=True)]  # fmt: skip

    return build()


def to_arrays(pts: list[tuple[int, int, int, int, bool, int]]) -> tuple[np.ndarray, ...]:
    arr = np.array(pts, dtype=np.int64).reshape(-1, 6)
    xyz_m = arr[:, :3].astype(np.float64) / 1000.0
    return xyz_m, arr[:, 3].astype(np.uint8), arr[:, 4].astype(bool), arr[:, 5].astype(np.uint8)


def same_values(a: RingAccumulators, b: RingAccumulators) -> None:
    """Bit-identical cells and fields (ring id aside)."""
    assert (a.cell_mm, a.side) == (b.cell_mm, b.side)
    np.testing.assert_array_equal(a.cells, b.cells)
    for (name, x), y in zip(a.fields().items(), b.fields().values(), strict=True):
        assert x.dtype == y.dtype, name
        np.testing.assert_array_equal(x, y, err_msg=name)


def assert_identical(a: GridAccumulators, b: GridAccumulators) -> None:
    assert a.counters == b.counters
    assert len(a.rings) == len(b.rings)
    for ra, rb in zip(a.rings, b.rings, strict=True):
        assert ra.equals(rb)


def check_fine_to_coarse(spec: GridSpec, pts: list) -> None:
    xyz_m, cls, moving, conf = to_arrays(pts)
    direct = rasterize(spec, xyz_m, cls, moving, conf)
    q = np.rint(xyz_m * 1000.0).astype(np.int64)
    ring, _, _ = locate(spec, q[:, 0], q[:, 1])
    s1 = spec.finest_cell_mm
    for k, rs in enumerate(spec.rings):
        sel = ring == k
        fine_spec = GridSpec(f"fine_{k}", (RingSpec(0, rs.r_max_mm, s1),))
        fine = rasterize(fine_spec, xyz_m[sel], cls[sel], moving[sel], conf[sel]).rings[0]
        coarse = reduce_block(fine, rs.cell_mm // s1)
        same_values(coarse, direct.rings[k])


@SETTINGS
@given(st.data())
def test_i4_fine_to_coarse_tiny_test(presets: dict[str, GridSpec], data: st.DataObject) -> None:
    spec = presets["tiny_test"]
    check_fine_to_coarse(spec, data.draw(points_strategy(spec)))


@SETTINGS
@given(st.data())
def test_i4_fine_to_coarse_tiny_ps_test(presets: dict[str, GridSpec], data: st.DataObject) -> None:
    spec = presets["tiny_ps_test"]
    check_fine_to_coarse(spec, data.draw(points_strategy(spec)))


def test_i4_dense_clusters(presets: dict[str, GridSpec]) -> None:
    """Many points per cell (the case random sparse points rarely hit)."""
    rng = np.random.default_rng(7)
    centres = rng.integers(-12_000, 12_000, size=(40, 2))
    pts = []
    for cx, cy in centres:
        for _ in range(60):
            pts.append((int(cx + rng.integers(-300, 300)), int(cy + rng.integers(-300, 300)),
                        int(rng.integers(-3_000, 3_000)), int(rng.integers(0, 5)), bool(rng.random() < 0.3),
                        int(rng.integers(0, 256))))  # fmt: skip
    for name in ("tiny_test", "tiny_ps_test"):
        check_fine_to_coarse(presets[name], pts)


def test_reduce_block_rejects_bad_factor(presets: dict[str, GridSpec]) -> None:
    acc = rasterize(presets["tiny_test"], np.zeros((1, 3)), np.zeros(1, np.uint8), np.zeros(1, bool),
                    np.zeros(1, np.uint8)).rings[0]  # fmt: skip
    with pytest.raises(ValueError):
        reduce_block(acc, 3)  # side 80 is not divisible by 3


@SETTINGS
@given(st.data(), st.randoms(use_true_random=False))
def test_i6_permutation_invariance(presets: dict[str, GridSpec], data: st.DataObject, rnd: object) -> None:
    spec = presets["tiny_test"]
    pts = data.draw(points_strategy(spec, max_n=300))
    xyz_m, cls, moving, conf = to_arrays(pts)
    perm = np.arange(len(pts))
    rnd.shuffle(perm)  # type: ignore[attr-defined]
    a = rasterize(spec, xyz_m, cls, moving, conf)
    b = rasterize(spec, xyz_m[perm], cls[perm], moving[perm], conf[perm])
    assert_identical(a, b)


def test_i6_repeated_runs_identical(presets: dict[str, GridSpec]) -> None:
    rng = np.random.default_rng(11)
    n = 50_000
    xyz = np.column_stack([rng.uniform(-110, 110, (n, 2)), rng.uniform(-3, 5, n)])
    cls = rng.integers(0, 5, n).astype(np.uint8)
    moving, conf = rng.random(n) < 0.1, rng.integers(0, 256, n).astype(np.uint8)
    for name in ("fovea_default", "ps_literal", "uniform_20cm"):
        runs = [rasterize(presets[name], xyz, cls, moving, conf, 1_000) for _ in range(3)]
        assert_identical(runs[0], runs[1])
        assert_identical(runs[0], runs[2])
