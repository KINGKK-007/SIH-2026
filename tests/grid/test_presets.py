"""T5.1: preset specs, V1-V5 validation and cell counts equal to the closed forms (README 6.5.1-6.5.2)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from foveamap.config import GridConfig
from foveamap.grid.presets import (
    GridSpec,
    PresetError,
    RingSpec,
    load_preset,
    spec_from_rings,
    validate_preset,
)


def closed_form_logical(rings: list[tuple[int, int]]) -> int:
    """README 6.5.1: ((2 R_k)^2 - (2 R_{k-1})^2) / s_k^2 summed over rings."""
    total, inner = 0, 0
    for r, s in rings:
        total += ((2 * r) ** 2 - (2 * inner) ** 2) // (s * s)
        inner = r
    return total


def closed_form_allocated(rings: list[tuple[int, int]]) -> int:
    """README 6.5.1: (2 R_k / s_k)^2 summed over rings."""
    return sum((2 * r // s) ** 2 for r, s in rings)


def test_every_config_preset_is_valid_and_counts_match_closed_forms(presets: dict[str, GridSpec]) -> None:
    for name, spec in presets.items():
        rings = [(r.r_max_mm, r.cell_mm) for r in spec.rings]
        assert spec.logical_cells() == closed_form_logical(rings), name
        assert spec.allocated_cells() == closed_form_allocated(rings), name
        assert spec.logical_cells() <= spec.allocated_cells()
        assert spec.per_ring_shape() == [(2 * r // s, 2 * r // s) for r, s in rings]


def test_counts_equal_readme_illustrative_values(presets: dict[str, GridSpec]) -> None:
    # README 6.5.1 table ("asserted equal to the program's output by a unit test").
    expected = {
        "fovea_default": (910_000, 1_130_000),
        "ps_literal": (318_400, 320_000),
        "uniform_5cm": (16_000_000, 16_000_000),
        "uniform_20cm": (1_000_000, 1_000_000),
    }
    for name, (logical, allocated) in expected.items():
        assert (presets[name].logical_cells(), presets[name].allocated_cells()) == (logical, allocated), name


def test_per_ring_logical_counts(presets: dict[str, GridSpec]) -> None:
    assert presets["fovea_default"].logical_cells_per_ring() == [160_000, 320_000, 270_000, 160_000]
    assert presets["fovea_default"].allocated_cells_per_ring() == [160_000, 360_000, 360_000, 250_000]


def test_load_preset_converts_metres_to_mm(presets: dict[str, GridSpec]) -> None:
    spec = presets["fovea_default"]
    assert spec.name == "fovea_default"
    assert spec.rings == (
        RingSpec(0, 10_000, 50),
        RingSpec(10_000, 30_000, 100),
        RingSpec(30_000, 60_000, 200),
        RingSpec(60_000, 100_000, 400),
    )
    assert spec.extent_mm == 100_000 and spec.finest_cell_mm == 50


def test_unknown_preset(grid_cfg: GridConfig) -> None:
    with pytest.raises(KeyError, match="no_such_preset"):
        load_preset("no_such_preset", grid_cfg)


def test_active_preset_extent_must_match_config(grid_cfg: GridConfig) -> None:
    spec = load_preset(grid_cfg.active_preset, grid_cfg)
    assert spec.extent_mm == grid_cfg.extent_mm


# ── one failing preset per rule ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rings", "rule"),
    [
        ([RingSpec(5, 10_000, 50)], "V1"),  # first ring must start at 0
        ([RingSpec(0, 10_000, 50), RingSpec(12_000, 30_000, 100)], "V1"),  # gap between rings
        ([RingSpec(0, 10_000, 50), RingSpec(10_000, 10_000, 100)], "V1"),  # empty ring
        ([RingSpec(0, 10_000, 50), RingSpec(10_000, 5_000, 100)], "V1"),  # out of order
        ([], "V1"),  # no rings
        ([RingSpec(0, 10_000, 0)], "V2"),  # zero cell
        ([RingSpec(0, 10_000.0, 50)], "V2"),  # not an integer (float)  # type: ignore[arg-type]
        ([RingSpec(0, 10_000, True)], "V2"),  # bool is not a length  # type: ignore[arg-type]
        ([RingSpec(0, 10_000, 50), RingSpec(10_000, 30_000, 75)], "V3"),  # 75 not a multiple of 50
        ([RingSpec(0, 10_000, 200), RingSpec(10_000, 30_000, 100)], "V3"),  # coarse -> fine
        ([RingSpec(0, 10_010, 50)], "V4"),  # R_k not a multiple of s_k
        (
            [RingSpec(0, 10_000, 50), RingSpec(10_000, 30_000, 300)],
            "V4",
        ),  # R_{k-1} = 10 m not a multiple of 0.3 m
    ],
)
def test_each_rule_rejects(rings: list[RingSpec], rule: str) -> None:
    with pytest.raises(PresetError, match=rule):
        validate_preset(GridSpec("bad", tuple(rings)))


def test_v5_extent_mismatch(grid_cfg: GridConfig) -> None:
    spec = spec_from_rings("x", [(10_000, 50), (30_000, 100)])
    with pytest.raises(PresetError, match="V5"):
        validate_preset(spec, expected_extent_mm=100_000)
    validate_preset(spec, expected_extent_mm=30_000)


def test_ps_example_passes_and_inconsistent_mix_fails() -> None:
    validate_preset(spec_from_rings("ps", [(10_000, 50), (100_000, 500)]))
    with pytest.raises(PresetError, match="V3"):  # 5/10/20/50 cm: 50 is not a multiple of 20
        validate_preset(spec_from_rings("bad", [(10_000, 50), (30_000, 100), (60_000, 200), (100_000, 500)]))


# ── property test: validator agrees with an independent checker (README 11.3) ─


def independent_ok(rings: list[tuple[int, int, int]]) -> bool:
    if not rings or rings[0][0] != 0:
        return False
    for k, (r_min, r_max, s) in enumerate(rings):
        if s <= 0 or r_max <= r_min or r_max % s:
            return False
        if k and (r_min != rings[k - 1][1] or s % rings[k - 1][2] or r_min % s):
            return False
    return True


ring_lists = st.lists(
    st.tuples(st.sampled_from([0, 2_000, 5_000, 6_000, 10_000, 12_000]),
              st.sampled_from([2_000, 6_000, 10_000, 12_000, 30_000, 60_000, 100_000]),
              st.sampled_from([25, 50, 75, 100, 200, 400, 500, 1_000])),
    min_size=1, max_size=4,
)  # fmt: skip


@settings(max_examples=400, derandomize=True)
@given(ring_lists)
def test_validator_matches_independent_checker(rings: list[tuple[int, int, int]]) -> None:
    spec = GridSpec("prop", tuple(RingSpec(*r) for r in rings))
    try:
        validate_preset(spec)
        accepted = True
    except PresetError:
        accepted = False
    assert accepted == independent_ok(rings)


@settings(max_examples=200, derandomize=True)
@given(
    st.lists(st.integers(1, 5), min_size=1, max_size=4),  # cell multipliers relative to the previous ring
    st.lists(st.integers(1, 6), min_size=4, max_size=4),  # radius steps in units of the ring's cell size
)
def test_constructed_valid_presets_pass(mults: list[int], steps: list[int]) -> None:
    rings: list[tuple[int, int]] = []
    cell, inner = 50, 0
    for k, m in enumerate(mults):
        cell = cell * m if k else cell
        inner_r = -(-inner // cell) * cell  # next multiple of this ring's cell
        if inner_r != inner:
            return  # inner boundary not aligned to this cell size; skip (covered by the checker test)
        r = inner + steps[k] * cell * 10
        rings.append((r, cell))
        inner = r
    validate_preset(spec_from_rings("constructed", rings))
