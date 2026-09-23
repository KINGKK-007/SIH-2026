"""Config schema tests (README 11.2): defaults load, unknown keys rejected, metre->mm conversion exact."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from foveamap.config import (
    CONFIG_FILES,
    ConfigError,
    FoveaConfig,
    GridConfig,
    load_config,
    load_yaml_config,
    m_to_mm,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _raw(key: str) -> dict[str, Any]:
    filename, _ = CONFIG_FILES[key]
    return yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))


def _all_raw() -> dict[str, dict[str, Any]]:
    return {key: _raw(key) for key in CONFIG_FILES}


# ── defaults ────────────────────────────────────────────────────────────────


def test_every_config_file_exists() -> None:
    for filename, _ in CONFIG_FILES.values():
        assert (CONFIG_DIR / filename).is_file(), filename


@pytest.mark.parametrize("key", sorted(CONFIG_FILES))
def test_each_default_file_validates(key: str) -> None:
    filename, model = CONFIG_FILES[key]
    assert isinstance(load_yaml_config(CONFIG_DIR / filename, model), model)


def test_load_config_defaults() -> None:
    cfg = load_config(CONFIG_DIR)
    assert cfg.grid.active_preset == "fovea_default"
    assert cfg.benchmark.sequence == "08"
    assert cfg.model.seed == 1337 and cfg.hazards.seed == 1337  # R4


def test_grid_default_rings_in_mm() -> None:
    grid = load_config(CONFIG_DIR).grid
    rings = [(r.r_mm, r.cell_mm) for r in grid.presets["fovea_default"].rings]
    assert rings == [(10_000, 50), (30_000, 100), (60_000, 200), (100_000, 400)]
    assert [(r.r_mm, r.cell_mm) for r in grid.presets["ps_literal"].rings] == [(10_000, 50), (100_000, 500)]
    assert grid.extent_mm == 100_000
    assert grid.min_range_mm == 1_000
    assert grid.z_range_mm == (-3_000, 5_000)
    assert grid.contact_height_mm == 300


def test_other_lengths_in_mm() -> None:
    cfg = load_config(CONFIG_DIR)
    assert cfg.benchmark.buckets_mm == [0, 10_000, 30_000, 60_000, 100_000]
    assert (cfg.derived.kerb_min_mm, cfg.derived.kerb_max_mm, cfg.derived.obstacle_step_mm) == (60, 250, 300)
    assert cfg.derived.vehicle_height_mm == 2_000 and cfg.derived.clearance_margin_mm == 200
    assert cfg.motion.tau0_mm == 300 and cfg.motion.cluster.eps0_mm == 500
    assert cfg.motion.hysteresis.gate_mm == 2_000 and cfg.motion.same_object_radius_mm == 5_000
    assert cfg.hazards.pothole.radius_mm == (300, 600) and cfg.hazards.pothole.depth_mm == (80, 150)
    assert cfg.hazards.kerb.height_mm == (100, 200)
    assert cfg.hazards.overhang.height_above_ground_mm == (1_600, 2_400)


# ── unknown keys ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(CONFIG_FILES))
def test_unknown_top_level_key_rejected(key: str) -> None:
    _, model = CONFIG_FILES[key]
    data = _raw(key)
    data["not_a_real_key"] = 1
    with pytest.raises(ValidationError, match="not_a_real_key"):
        model.model_validate(data)


@pytest.mark.parametrize(
    ("key", "path"),
    [
        ("grid", ("class_thresholds",)),
        ("grid", ("presets", "fovea_default")),
        ("model", ("input",)),
        ("model", ("finetune",)),
        ("motion", ("cluster",)),
        ("motion", ("hysteresis",)),
        ("hazards", ("pothole",)),
        ("benchmark", ("latency",)),
        ("benchmark", ("sweep",)),
    ],
)
def test_unknown_nested_key_rejected(key: str, path: tuple[str, ...]) -> None:
    _, model = CONFIG_FILES[key]
    data = _raw(key)
    node = data
    for part in path:
        node = node[part]
    node["typo_key"] = 0
    with pytest.raises(ValidationError, match="typo_key"):
        model.model_validate(data)


def test_unknown_key_inside_ring_rejected() -> None:
    data = _raw("grid")
    data["presets"]["tiny_test"]["rings"][0]["radius"] = 2
    with pytest.raises(ValidationError, match="radius"):
        GridConfig.model_validate(data)


def test_load_yaml_config_names_the_file(tmp_path: Path) -> None:
    data = _raw("dashboard")
    data["colour"] = "blue"
    bad = tmp_path / "dashboard.yaml"
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    _, model = CONFIG_FILES["dashboard"]
    with pytest.raises(ConfigError, match="dashboard.yaml"):
        load_yaml_config(bad, model)


def test_missing_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path)


# ── metre -> millimetre conversion ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("metres", "mm"),
    [
        (0.05, 50),
        (0.1, 100),
        (0.2, 200),
        (0.4, 400),
        (0.5, 500),
        (0.3, 300),
        (0.001, 1),
        (10, 10_000),
        (100.0, 100_000),
        (-3.0, -3_000),
        (0, 0),
        (0.58, 580),
        (1.001, 1_001),
        (1.005, 1_005),
        (4.35, 4_350),  # int(x * 1000) gets 1.001 and 1.005 wrong
    ],
)
def test_m_to_mm_exact(metres: float, mm: int) -> None:
    assert m_to_mm(metres) == mm
    assert isinstance(m_to_mm(metres), int)


def test_naive_float_conversion_would_be_wrong() -> None:
    # The reason m_to_mm exists (L6): plain float arithmetic truncates these.
    assert int(1.005 * 1000) == 1004
    assert m_to_mm(1.005) == 1005


@pytest.mark.parametrize("metres", [0.0505, 1e-4, 0.12345, 2.0000001])
def test_m_to_mm_rejects_sub_millimetre(metres: float) -> None:
    with pytest.raises(ValueError, match="whole number of millimetres"):
        m_to_mm(metres)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_m_to_mm_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        m_to_mm(bad)


@pytest.mark.parametrize("bad", [True, "0.05", None])
def test_m_to_mm_rejects_non_numbers(bad: object) -> None:
    with pytest.raises(ValueError, match="number of metres"):
        m_to_mm(bad)  # type: ignore[arg-type]


@settings(max_examples=500, derandomize=True)
@given(st.integers(min_value=-10_000_000, max_value=10_000_000))
def test_m_to_mm_round_trips_every_millimetre(mm: int) -> None:
    assert m_to_mm(mm / 1000) == mm


def test_sub_millimetre_cell_rejected_by_schema() -> None:
    data = _raw("grid")
    data["presets"]["tiny_test"]["rings"][0]["cell"] = 0.0501
    with pytest.raises(ValidationError, match="whole number of millimetres"):
        GridConfig.model_validate(data)


# ── field and cross-field validation ────────────────────────────────────────


def _expect_invalid(model: type[BaseModel], data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        model.model_validate(data)


def test_active_preset_must_exist() -> None:
    data = _raw("grid")
    data["active_preset"] = "fovea_missing"
    _expect_invalid(GridConfig, data, "active_preset")


def test_extent_must_match_active_preset() -> None:
    data = _raw("grid")
    data["extent_m"] = 120.0
    _expect_invalid(GridConfig, data, "extent_m")


def test_extent_follows_active_preset() -> None:
    data = _raw("grid")
    data["active_preset"] = "tiny_test"
    data["extent_m"] = 12.0
    assert GridConfig.model_validate(data).extent_mm == 12_000


def test_z_range_must_increase() -> None:
    data = _raw("grid")
    data["z_range_m"] = [5.0, -3.0]
    _expect_invalid(GridConfig, data, "z_range_m")


def test_backend_choices() -> None:
    data = _raw("grid")
    data["backend"] = "cuda"
    _expect_invalid(GridConfig, data, "backend")


@pytest.mark.parametrize("seq", [8, "8", "008"])
def test_sequence_must_be_two_digit_string(seq: object) -> None:
    _, model = CONFIG_FILES["benchmark"]
    data = _raw("benchmark")
    data["sequence"] = seq
    _expect_invalid(model, data, "sequence")


def test_buckets_must_increase_from_zero() -> None:
    _, model = CONFIG_FILES["benchmark"]
    data = _raw("benchmark")
    data["buckets_m"] = [0, 30, 10, 100]
    _expect_invalid(model, data, "buckets_m")
    data["buckets_m"] = [5, 10, 30, 60, 100]
    _expect_invalid(model, data, "buckets_m")


def test_hazard_range_order() -> None:
    _, model = CONFIG_FILES["hazards"]
    data = _raw("hazards")
    data["pothole"]["depth_m"] = [0.15, 0.08]
    _expect_invalid(model, data, "depth_m")


def test_kerb_thresholds_ordered() -> None:
    _, model = CONFIG_FILES["derived"]
    data = _raw("derived")
    data["kerb_max_m"] = 0.40
    _expect_invalid(model, data, "kerb_max_m")


def test_motion_window_odd_and_hysteresis_consistent() -> None:
    _, model = CONFIG_FILES["motion"]
    data = _raw("motion")
    data["window"] = 4
    _expect_invalid(model, data, "window")
    data = _raw("motion")
    data["hysteresis"]["min_hits"] = 5
    _expect_invalid(model, data, "min_hits")


def test_benchmark_presets_must_exist_in_grid() -> None:
    raw = _all_raw()
    raw["benchmark"]["presets"] = ["fovea_default", "uniform_1cm"]
    with pytest.raises(ValidationError, match="uniform_1cm"):
        FoveaConfig.model_validate(raw)


def test_buckets_must_end_at_grid_extent() -> None:
    raw = _all_raw()
    raw["benchmark"]["buckets_m"] = [0, 10, 30, 60, 80]
    with pytest.raises(ValidationError, match="extent_m"):
        FoveaConfig.model_validate(raw)


def test_configs_are_immutable() -> None:
    cfg = load_config(CONFIG_DIR)
    with pytest.raises(ValidationError):
        cfg.grid.active_preset = "ps_literal"  # type: ignore[misc]


def test_raw_data_not_mutated_by_validation() -> None:
    raw = _all_raw()
    before = copy.deepcopy(raw)
    FoveaConfig.model_validate(raw)
    assert raw == before
