"""``PipelineRunner``: stage orchestration with per-stage timers (README 6.10, 9.7, task T7.2).

Oracle mode (Phase 7):

    scan → model.predict → raw_to_super → rasterize → finalize → memory_report → FrameResult

Every stage is wrapped with ``perf_counter_ns`` timers; the ``timings_ms`` dict keys match the
stage names so the dashboard (Phase 13) can display them without modification.

The ``cached`` and ``live`` modes are stubs filled in Phases 9 and 13 respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Literal

from foveamap.derived import compute_derived_layers
from foveamap.grid.accumulators import FrameCounters
from foveamap.grid.engine import rasterize
from foveamap.grid.layers import GridLayers, finalize
from foveamap.grid.memory import MemoryReport, memory_report
from foveamap.grid.presets import GridSpec
from foveamap.io.labels import raw_to_super
from foveamap.io.sequence import Sequence
from foveamap.pipeline.records import ObjectBox


@dataclass
class FrameResult:
    """Per-frame output of the full pipeline.

    Attributes:
        scan_idx: Original frame index in the sequence.
        layers: Packed 12-byte ``GridLayers``.
        objects: Detected object boxes (empty in oracle / Phase 7).
        counters: Frame-level point accounting.
        timings_ms: Wall-clock milliseconds per stage (``perf_counter_ns`` based).
        memory: Four-representation memory report.
    """

    scan_idx: int
    layers: GridLayers
    objects: list[ObjectBox]
    counters: FrameCounters
    timings_ms: dict[str, float]
    memory: MemoryReport


def _ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


class PipelineRunner:
    """Runs the full FoveaMap pipeline for one frame with per-stage timing.

    Args:
        mode: ``"oracle"`` uses ground-truth labels as the segmentation model;
            ``"cached"`` reads predictions from disk (Phase 9);
            ``"live"`` runs the neural network in real time (Phase 13).
        model: Any object implementing the
            :class:`~foveamap.models.base.SegmentationModel` protocol.
        preset: The active :class:`~foveamap.grid.presets.GridSpec`.
        cfgs: The loaded :class:`~foveamap.config.FoveaConfig` (or its ``.grid`` / ``.model``
            components); only ``cfgs.grid`` is used in Phase 7.
    """

    def __init__(
        self,
        mode: Literal["oracle", "cached", "live"],
        model: object,
        preset: GridSpec,
        cfgs: object,
    ) -> None:
        if mode not in ("oracle", "cached", "live"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "live":
            raise NotImplementedError("mode='live' is implemented in Phase 13 (docs/PHASES.md).")
        self.mode = mode
        self.model = model
        self.preset = preset
        # Accept either a FoveaConfig or bare GridConfig
        try:
            self.grid_cfg = cfgs.grid  # type: ignore[union-attr]
        except AttributeError:
            self.grid_cfg = cfgs
        self.derived_cfg = getattr(cfgs, "derived", None)

    def process(self, seq: Sequence, idx: int) -> FrameResult:
        """Run the pipeline for one frame and return a :class:`FrameResult`.

        Args:
            seq: An open :class:`~foveamap.io.sequence.Sequence`.
            idx: Original frame index (not strided).

        Returns:
            A fully populated :class:`FrameResult`.
        """
        timings_ns: dict[str, int] = {}

        # Stage 1: I/O — load the scan
        t0 = perf_counter_ns()
        scan = seq.load_frame(idx)
        timings_ns["io_ms"] = perf_counter_ns() - t0

        # Stage 2: Segmentation model (oracle: ground-truth labels)
        t0 = perf_counter_ns()
        prediction = self.model.predict(scan)  # type: ignore[attr-defined]
        timings_ns["model_ms"] = perf_counter_ns() - t0

        # Stage 3: Super-class mapping
        t0 = perf_counter_ns()
        super_cls, moving = raw_to_super(prediction.raw_ids)
        conf = prediction.conf
        timings_ns["label_ms"] = perf_counter_ns() - t0

        # Stage 4: Grid rasterisation
        t0 = perf_counter_ns()
        acc = rasterize(
            self.preset,
            scan.xyz,
            super_cls,
            moving,
            conf,
            min_range_mm=self.grid_cfg.min_range_mm,
        )
        timings_ns["grid_ms"] = perf_counter_ns() - t0

        # Stage 5: Finalize → packed layers
        t0 = perf_counter_ns()
        layers = finalize(acc, self.grid_cfg, self.preset)
        timings_ns["finalize_ms"] = perf_counter_ns() - t0

        # Stage 5.5: Derived geometric layers (Phase 11)
        if getattr(self, "derived_cfg", None) is not None:
            t0 = perf_counter_ns()
            layers = compute_derived_layers(layers, self.derived_cfg)
            timings_ns["derived_ms"] = perf_counter_ns() - t0

        # Stage 6: Memory accounting
        t0 = perf_counter_ns()
        mem = memory_report(self.preset, layers, self.grid_cfg)
        timings_ns["memory_ms"] = perf_counter_ns() - t0

        timings_ms = {k: _ns_to_ms(v) for k, v in timings_ns.items()}

        return FrameResult(
            scan_idx=idx,
            layers=layers,
            objects=[],  # populated by Phase 10 (motion/object detection)
            counters=acc.counters,
            timings_ms=timings_ms,
            memory=mem,
        )
