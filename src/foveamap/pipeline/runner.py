"""``PipelineRunner``: stage orchestration with per-stage timers (README 6.10, 9.7, task T7.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from foveamap.grid.accumulators import FrameCounters
from foveamap.grid.layers import GridLayers
from foveamap.grid.memory import MemoryReport
from foveamap.grid.presets import GridSpec
from foveamap.io.sequence import Sequence
from foveamap.pipeline.records import ObjectBox


@dataclass
class FrameResult:
    scan_idx: int
    layers: GridLayers
    objects: list[ObjectBox]
    counters: FrameCounters
    timings_ms: dict[str, float]
    memory: MemoryReport


class PipelineRunner:
    def __init__(self, mode: Literal["oracle", "cached", "live"], model: object, preset: GridSpec, cfgs: object) -> None:
        raise NotImplementedError("Implemented in Phase 7, T7.2 (docs/PHASES.md).")

    def process(self, seq: Sequence, idx: int) -> FrameResult:
        raise NotImplementedError("Implemented in Phase 7, T7.2 (docs/PHASES.md).")
