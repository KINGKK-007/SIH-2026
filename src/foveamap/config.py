"""Pydantic models for every ``configs/*.yaml`` file (README Section 8).

Rules enforced here:

* Unknown keys are errors, at every nesting level.
* Lengths are written in metres in YAML and must be a whole number of millimetres (L6, V2).
  :func:`m_to_mm` converts them exactly via the shortest decimal representation, so ``1.005`` becomes
  ``1005`` (``int(1.005 * 1000)`` gives ``1004``). Every length field has an ``*_mm`` accessor.
* Cross-file consistency (benchmark presets exist in ``grid.yaml``, buckets end at the grid extent)
  is checked when all files are loaded together with :func:`load_config`.

Grid-geometry rules V1-V5 are *not* checked here; they belong to ``grid.presets.validate_preset``.
"""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

__all__ = [
    "BenchmarkConfig",
    "ConfigError",
    "DashboardConfig",
    "DerivedConfig",
    "FoveaConfig",
    "GridConfig",
    "HazardConfig",
    "LSK3DNetConfig",
    "ModelConfig",
    "MotionConfig",
    "load_config",
    "load_yaml_config",
    "m_to_mm",
]


class ConfigError(ValueError):
    """A config file is missing, unreadable, or fails validation."""


def m_to_mm(value: float, *, name: str = "value") -> int:
    """Convert metres to integer millimetres exactly; raise ``ValueError`` if not a whole number of mm."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number of metres, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    mm = Decimal(str(value)) * 1000
    if mm != mm.to_integral_value():
        raise ValueError(f"{name}={value!r} m is not a whole number of millimetres")
    return int(mm)


def _check_metres(value: float) -> float:
    m_to_mm(value)
    return value


Metres = Annotated[float, AfterValidator(_check_metres)]
"""A length in metres that must be an exact number of millimetres."""

PositiveMetres = Annotated[float, Field(gt=0), AfterValidator(_check_metres)]
NonNegativeMetres = Annotated[float, Field(ge=0), AfterValidator(_check_metres)]
Fraction = Annotated[float, Field(ge=0, le=1)]
SequenceId = Annotated[str, StringConstraints(pattern=r"^\d{2}$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _ordered_pair(pair: tuple[float, float], what: str) -> None:
    if pair[0] > pair[1]:
        raise ValueError(f"{what}: lower bound {pair[0]} exceeds upper bound {pair[1]}")


# ── grid.yaml (README 8.1) ──────────────────────────────────────────────────


class RingConfig(_Strict):
    r: PositiveMetres = Field(description="outer radius R_k in metres")
    cell: PositiveMetres = Field(description="cell size s_k in metres")

    @property
    def r_mm(self) -> int:
        return m_to_mm(self.r)

    @property
    def cell_mm(self) -> int:
        return m_to_mm(self.cell)


class PresetConfig(_Strict):
    rings: list[RingConfig] = Field(min_length=1)


class ClassThresholds(_Strict):
    dyn_min_points: int = Field(ge=0)
    dyn_min_frac: Fraction
    obs_min_points: int = Field(ge=0)
    obs_min_frac: Fraction


class GridConfig(_Strict):
    backend: Literal["auto", "numpy", "cpp", "cupy"] = "auto"
    extent_m: PositiveMetres
    min_range_m: NonNegativeMetres
    z_range_m: tuple[Metres, Metres]
    bytes_per_voxel: int = Field(ge=1)
    class_rule: Literal["safety", "majority"] = "safety"
    class_thresholds: ClassThresholds
    contact_height_m: NonNegativeMetres
    ground_estimator: Literal["mean", "min"] = "mean"
    presets: dict[str, PresetConfig] = Field(min_length=1)
    active_preset: str

    @model_validator(mode="after")
    def _consistent(self) -> GridConfig:
        if self.z_range_m[0] >= self.z_range_m[1]:
            raise ValueError(f"z_range_m must be increasing, got {list(self.z_range_m)}")
        if self.active_preset not in self.presets:
            raise ValueError(f"active_preset {self.active_preset!r} is not one of {sorted(self.presets)}")
        last_r = self.presets[self.active_preset].rings[-1].r_mm
        if last_r != self.extent_mm:
            raise ValueError(
                f"extent_m ({self.extent_m}) must equal the outer radius of the active preset's last ring "
                f"({last_r / 1000})"
            )
        return self

    @property
    def extent_mm(self) -> int:
        return m_to_mm(self.extent_m)

    @property
    def min_range_mm(self) -> int:
        return m_to_mm(self.min_range_m)

    @property
    def z_range_mm(self) -> tuple[int, int]:
        return (m_to_mm(self.z_range_m[0]), m_to_mm(self.z_range_m[1]))

    @property
    def contact_height_mm(self) -> int:
        return m_to_mm(self.contact_height_m)


# ── model.yaml (README 8.2) ─────────────────────────────────────────────────


class ModelInputConfig(_Strict):
    """Copied from the checkpoint's own config in Phase 8; never guessed (README 3.1)."""

    height: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    fov_up_deg: float | None = None
    fov_down_deg: float | None = None
    mean: list[float] | None = None
    std: list[float] | None = None


class FinetuneConfig(_Strict):
    enabled: bool = False
    distance_weight_alpha: float = Field(default=1.0, ge=0)


class LSK3DNetConfig(_Strict):
    """Sparse large-kernel network settings (docs/DECISIONS.md D-022 overrides README L11).

    Not part of README 8.2; this section only exists because the production model family was switched
    from range-view to sparse-voxel. Values not covered here (voxel spatial shape, volume-space crop,
    class count, kernel size) come from the checkpoint's own ``config_path`` and must not be duplicated
    or guessed here (README 3.1).
    """

    repo_dir: str = Field(description="vendored LSK3DNet checkout, path relative to the repo root")
    config_path: str = Field(description="the checkpoint's own training/eval YAML, read as-is")
    half_precision: bool = Field(
        default=True, description="torch.autocast fp16 forward pass; needed to fit a 6 GB laptop GPU"
    )
    vram_budget_gb: float = Field(gt=0, description="target GPU VRAM; drives batch-of-one, cache-clearing")
    empty_cache_every_scan: bool = True


class ModelConfig(_Strict):
    name: str | None = None
    family: Literal["rangeview", "sparse_voxel"] = "rangeview"
    checkpoint: str | None = None
    device: Literal["cuda", "cpu"] = "cuda"
    input: ModelInputConfig
    knn_postprocess: bool = True
    cache_dir: str
    seed: int
    finetune: FinetuneConfig
    lsk3dnet: LSK3DNetConfig | None = None

    @model_validator(mode="after")
    def _family_needs_its_section(self) -> ModelConfig:
        if self.family == "sparse_voxel" and self.lsk3dnet is None:
            raise ValueError("model.yaml: family 'sparse_voxel' requires an 'lsk3dnet:' section")
        return self


# ── motion.yaml (README 8.3) ────────────────────────────────────────────────


class MotionClusterConfig(_Strict):
    eps0_m: PositiveMetres
    eps1_rel: float = Field(ge=0)
    min_points: int = Field(ge=1)

    @property
    def eps0_mm(self) -> int:
        return m_to_mm(self.eps0_m)


class MotionVoteConfig(_Strict):
    vote_frac: Fraction
    vote_min_points: int = Field(ge=0)


class HysteresisConfig(_Strict):
    enabled: bool
    window: int = Field(ge=1)
    min_hits: int = Field(ge=1)
    gate_m: PositiveMetres
    v_min_mps: float = Field(ge=0)

    @model_validator(mode="after")
    def _hits_fit_window(self) -> HysteresisConfig:
        if self.min_hits > self.window:
            raise ValueError(f"min_hits ({self.min_hits}) cannot exceed window ({self.window})")
        return self

    @property
    def gate_mm(self) -> int:
        return m_to_mm(self.gate_m)


class BoxConfig(_Strict):
    angle_step_deg: float = Field(gt=0, le=90)


class MotionConfig(_Strict):
    enabled: bool
    frame_gaps: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    window: int = Field(ge=1)
    tau0_m: NonNegativeMetres
    tau1_rel: float = Field(ge=0)
    occlusion_rule: bool
    same_object_radius_m: PositiveMetres
    movable_classes: list[Annotated[int, Field(ge=0, le=0xFFFF)]] = Field(min_length=1)
    cluster: MotionClusterConfig
    vote: MotionVoteConfig
    hysteresis: HysteresisConfig
    box: BoxConfig

    @model_validator(mode="after")
    def _odd_window(self) -> MotionConfig:
        if self.window % 2 == 0:
            raise ValueError(f"window must be odd so it has a centre pixel, got {self.window}")
        return self

    @property
    def tau0_mm(self) -> int:
        return m_to_mm(self.tau0_m)

    @property
    def same_object_radius_mm(self) -> int:
        return m_to_mm(self.same_object_radius_m)


# ── derived.yaml (README 8.4) ───────────────────────────────────────────────


class DerivedConfig(_Strict):
    max_slope_deg: float = Field(gt=0, lt=90)
    kerb_min_m: PositiveMetres
    kerb_max_m: PositiveMetres
    obstacle_step_m: PositiveMetres
    min_points_step: int = Field(ge=1)
    vehicle_height_m: PositiveMetres
    clearance_margin_m: NonNegativeMetres
    halo: bool
    fill_missing_ground: bool
    fast_mode: bool = False

    @model_validator(mode="after")
    def _ordered_steps(self) -> DerivedConfig:
        if not self.kerb_min_m < self.kerb_max_m < self.obstacle_step_m:
            raise ValueError(
                "need kerb_min_m < kerb_max_m < obstacle_step_m, got "
                f"{self.kerb_min_m}, {self.kerb_max_m}, {self.obstacle_step_m}"
            )
        return self

    @property
    def kerb_min_mm(self) -> int:
        return m_to_mm(self.kerb_min_m)

    @property
    def kerb_max_mm(self) -> int:
        return m_to_mm(self.kerb_max_m)

    @property
    def obstacle_step_mm(self) -> int:
        return m_to_mm(self.obstacle_step_m)

    @property
    def vehicle_height_mm(self) -> int:
        return m_to_mm(self.vehicle_height_m)

    @property
    def clearance_margin_mm(self) -> int:
        return m_to_mm(self.clearance_margin_m)


# ── hazards.yaml (README 8.5) ───────────────────────────────────────────────

MetreRange = tuple[PositiveMetres, PositiveMetres]


def _range_mm(pair: tuple[float, float]) -> tuple[int, int]:
    return (m_to_mm(pair[0]), m_to_mm(pair[1]))


class PotholeConfig(_Strict):
    radius_m: MetreRange
    depth_m: MetreRange
    detect_frac: Annotated[float, Field(gt=0, le=1)]

    @model_validator(mode="after")
    def _ranges(self) -> PotholeConfig:
        _ordered_pair(self.radius_m, "radius_m")
        _ordered_pair(self.depth_m, "depth_m")
        return self

    @property
    def radius_mm(self) -> tuple[int, int]:
        return _range_mm(self.radius_m)

    @property
    def depth_mm(self) -> tuple[int, int]:
        return _range_mm(self.depth_m)


class KerbHazardConfig(_Strict):
    width_m: PositiveMetres
    length_m: PositiveMetres
    height_m: MetreRange

    @model_validator(mode="after")
    def _ranges(self) -> KerbHazardConfig:
        _ordered_pair(self.height_m, "height_m")
        return self

    @property
    def height_mm(self) -> tuple[int, int]:
        return _range_mm(self.height_m)


class OverhangConfig(_Strict):
    size_m: tuple[PositiveMetres, PositiveMetres]
    height_above_ground_m: MetreRange

    @model_validator(mode="after")
    def _ranges(self) -> OverhangConfig:
        _ordered_pair(self.height_above_ground_m, "height_above_ground_m")
        return self

    @property
    def height_above_ground_mm(self) -> tuple[int, int]:
        return _range_mm(self.height_above_ground_m)


class HazardConfig(_Strict):
    seed: int
    per_bucket_injections: int = Field(ge=1)
    min_ground_points: int = Field(ge=0)
    pothole: PotholeConfig
    kerb: KerbHazardConfig
    overhang: OverhangConfig


# ── benchmark.yaml (README 8.6) ─────────────────────────────────────────────


class LatencyConfig(_Strict):
    warmup: int = Field(ge=0)
    frames: int = Field(ge=1)
    budget_ms: float = Field(gt=0)


class SweepConfig(_Strict):
    ring_radii_m: list[list[PositiveMetres]] = Field(min_length=1)
    cell_sizes_m: list[list[PositiveMetres]] = Field(min_length=1)

    @model_validator(mode="after")
    def _increasing_radii(self) -> SweepConfig:
        for radii in self.ring_radii_m:
            if any(a >= b for a, b in zip(radii, radii[1:], strict=False)):
                raise ValueError(f"ring_radii_m entries must be strictly increasing, got {radii}")
        return self


class BenchmarkConfig(_Strict):
    sequence: SequenceId
    frame_stride: int = Field(ge=1)
    latency: LatencyConfig
    buckets_m: list[NonNegativeMetres] = Field(min_length=2)
    presets: list[str] = Field(min_length=1)
    sweep: SweepConfig
    output_dir: str

    @model_validator(mode="after")
    def _buckets(self) -> BenchmarkConfig:
        if self.buckets_m[0] != 0:
            raise ValueError(f"buckets_m must start at 0, got {self.buckets_m[0]}")
        if any(a >= b for a, b in zip(self.buckets_m, self.buckets_m[1:], strict=False)):
            raise ValueError(f"buckets_m must be strictly increasing, got {self.buckets_m}")
        return self

    @property
    def buckets_mm(self) -> list[int]:
        return [m_to_mm(b) for b in self.buckets_m]


# ── dashboard.yaml (README 8.7) ─────────────────────────────────────────────


class DashboardConfig(_Strict):
    host: str
    port: int = Field(ge=1, le=65535)
    target_display_fps: float = Field(gt=0)
    playback_fps: float = Field(gt=0)
    texture_format: Literal["webp", "png"] = "webp"
    show_gt_overlay: bool = False
    theme: Literal["dark", "light"] = "dark"


# ── loading ─────────────────────────────────────────────────────────────────

M = TypeVar("M", bound=BaseModel)


class FoveaConfig(_Strict):
    """All seven config files, validated together."""

    grid: GridConfig
    model: ModelConfig
    motion: MotionConfig
    derived: DerivedConfig
    hazards: HazardConfig
    benchmark: BenchmarkConfig
    dashboard: DashboardConfig

    @model_validator(mode="after")
    def _cross_file(self) -> FoveaConfig:
        unknown = [p for p in self.benchmark.presets if p not in self.grid.presets]
        if unknown:
            raise ValueError(f"benchmark.yaml presets {unknown} are not defined in grid.yaml")
        if self.benchmark.buckets_mm[-1] != self.grid.extent_mm:
            raise ValueError(
                f"benchmark.yaml buckets_m must end at grid.yaml extent_m ({self.grid.extent_m}), "
                f"got {self.benchmark.buckets_m[-1]}"
            )
        return self


CONFIG_FILES: dict[str, tuple[str, type[BaseModel]]] = {
    "grid": ("grid.yaml", GridConfig),
    "model": ("model.yaml", ModelConfig),
    "motion": ("motion.yaml", MotionConfig),
    "derived": ("derived.yaml", DerivedConfig),
    "hazards": ("hazards.yaml", HazardConfig),
    "benchmark": ("benchmark.yaml", BenchmarkConfig),
    "dashboard": ("dashboard.yaml", DashboardConfig),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def load_yaml_config(path: str | Path, model: type[M]) -> M:
    """Load and validate one YAML file against ``model``."""
    path = Path(path)
    try:
        return model.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def load_config(config_dir: str | Path = "configs") -> FoveaConfig:
    """Load and validate all seven config files from ``config_dir``."""
    config_dir = Path(config_dir)
    raw = {key: _read_yaml(config_dir / filename) for key, (filename, _) in CONFIG_FILES.items()}
    try:
        return FoveaConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{config_dir}: {exc}") from exc
