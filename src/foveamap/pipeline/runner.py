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
from foveamap.pipeline.records import ClassifiedScan, ObjectBox, Prediction, Scan


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
        self.motion_cfg = getattr(cfgs, "motion", None)
        from foveamap.motion.tracker import ClusterTracker
        self.tracker = ClusterTracker(self.motion_cfg)

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

        # Stage 3.5: Motion detection & Object tracking (Phase 10)
        t0 = perf_counter_ns()
        from foveamap.io.poses import relative_transform
        from foveamap.motion.pipeline import estimate_motion

        cur_classified = ClassifiedScan(
            scan=scan,
            super_cls=super_cls,
            moving=moving,
            conf=conf,
            objects=[],
            raw_ids=prediction.raw_ids,
        )

        if self.mode == "oracle":
            classified = estimate_motion(
                cur_classified,
                prev_scans=[],
                transforms=[],
                cfg=self.motion_cfg,
                use_oracle=True,
            )
            objects = classified.objects
            super_cls = classified.super_cls
            moving = classified.moving
        elif self.motion_cfg is not None and getattr(self.motion_cfg, "enabled", True):
            frame_gaps = getattr(self.motion_cfg, "frame_gaps", [2])
            prev_scans: list[tuple[Scan, Prediction]] = []
            transforms: list[np.ndarray] = []
            for gap in frame_gaps:
                prev_idx = idx - gap
                if 0 <= prev_idx < seq.n_frames_total:
                    try:
                        p_scan = seq.load_frame(prev_idx)
                        p_pred = self.model.predict(p_scan)
                        T = relative_transform(seq.calib, seq.poses, idx, prev_idx)
                        prev_scans.append((p_scan, p_pred))
                        transforms.append(T)
                    except Exception:
                        pass

            classified = estimate_motion(
                cur_classified,
                prev_scans=prev_scans,
                transforms=transforms,
                cfg=self.motion_cfg,
                use_oracle=False,
            )
            T_prev = transforms[0] if transforms else None
            objects = self.tracker.update(classified.objects, T_prev)
            super_cls = classified.super_cls
            moving = classified.moving
        else:
            objects = []

        timings_ns["motion_ms"] = perf_counter_ns() - t0

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
            objects=objects,
            counters=acc.counters,
            timings_ms=timings_ms,
            memory=mem,
        )
