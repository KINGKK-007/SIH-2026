"""FastAPI app factory with the Socket.IO ASGI mount (README 9.9, task T13.2)."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
import socketio
from pydantic import BaseModel, Field

from foveamap.config import load_config
from foveamap.grid.presets import PresetError, load_preset, spec_from_rings, validate_preset
from foveamap.io.sequence import Sequence
from foveamap.models.oracle import OracleModel
from foveamap.pipeline.runner import PipelineRunner
from foveamap.server.protocol import serialise_frame_result
from foveamap.server.sockets import (
    PlaybackManager,
    get_playback_manager,
    set_playback_manager,
    sio,
)


class ZoomRequest(BaseModel):
    frame_idx: int = Field(ge=0)
    x_m: float = Field(ge=-97, le=97)
    y_m: float = Field(ge=-97, le=97)


class FoveaRequest(BaseModel):
    frame_idx: int = Field(ge=0)
    ring0_m: int = Field(ge=5, le=20)
    ring1_m: int = Field(ge=20, le=40)
    cell0_cm: int = Field(default=5, ge=5, le=10)
    cell1_cm: int = Field(default=10, ge=10, le=20)


class HazardRequest(BaseModel):
    frame_idx: int = Field(ge=0)
    kind: str
    x_m: float = Field(ge=-95, le=95)
    y_m: float = Field(ge=-95, le=95)


def create_app(
    runner: PipelineRunner | None = None,
    seq: Sequence | None = None,
    cfgs: Any = None,
    data_root: str = "data/dataset",
    config_dir: str = "configs",
    sequence_id: str = "08",
) -> Any:
    """Create FastAPI application with mounted Socket.IO server.

    If runner, seq, or cfgs are not supplied, default oracle pipeline on
    sequence_id is initialized automatically.
    """
    if cfgs is None:
        cfgs = load_config(config_dir)

    if seq is None:
        seq = Sequence(data_root, sequence_id)

    if runner is None:
        preset_name = cfgs.grid.active_preset
        preset = load_preset(preset_name, cfgs.grid)
        model = OracleModel()
        runner = PipelineRunner(mode="oracle", model=model, preset=preset, cfgs=cfgs)

    # Initialize playback manager
    manager = PlaybackManager(runner=runner, seq=seq, cfgs=cfgs, model_name="oracle")
    set_playback_manager(manager)

    app = FastAPI(title="FoveaMap Dashboard Server", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "app": "foveamap"}

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        return get_playback_manager().get_state()

    @app.get("/api/presets")
    async def list_presets() -> dict[str, Any]:
        mgr = get_playback_manager()
        active = mgr.cfgs.grid.active_preset if hasattr(mgr.cfgs, "grid") else "fovea_default"
        presets = list(mgr.cfgs.grid.presets.keys()) if hasattr(mgr.cfgs, "grid") else ["fovea_default"]
        return {"active": active, "available": presets}

    @app.post("/api/zoom")
    async def zoom_window(request: ZoomRequest) -> dict[str, Any]:
        """Compare the same real scan window at 5 cm and 50 cm resolution."""
        if request.frame_idx >= len(seq):
            raise HTTPException(404, "Frame is outside this sequence")

        def compute() -> dict[str, Any]:
            from foveamap.derived.zoom import render_zoom

            scan = seq.load_frame(request.frame_idx)
            box = (request.x_m - 2.5, request.x_m + 2.5,
                   request.y_m - 2.5, request.y_m + 2.5)
            fine, coarse, fine_kerb, coarse_kerb = render_zoom(
                scan, box,
                spec_from_rings("lens_5cm", [(3000, 50)]),
                spec_from_rings("lens_50cm", [(3000, 500)]),
                cfg=cfgs.derived,
            )
            _, medium, _, medium_kerb = render_zoom(
                scan, box,
                spec_from_rings("lens_5cm", [(3000, 50)]),
                spec_from_rings("lens_20cm", [(3000, 200)]),
                cfg=cfgs.derived,
            )
            _, far, _, far_kerb = render_zoom(
                scan, box,
                spec_from_rings("lens_5cm", [(3200, 50)]),
                spec_from_rings("lens_40cm", [(3200, 400)]),
                cfg=cfgs.derived,
            )
            return {
                "frame_idx": request.frame_idx,
                "center_m": [request.x_m, request.y_m],
                "source": "ground_truth_labels" if scan.raw_labels is not None else "unlabelled_scan",
                "fine_step_mm": fine.tolist(),
                "coarse_step_mm": coarse.tolist(),
                "medium_step_mm": medium.tolist(),
                "far_step_mm": far.tolist(),
                "fine_kerb": fine_kerb,
                "medium_kerb": medium_kerb,
                "far_kerb": far_kerb,
                "coarse_kerb": coarse_kerb,
            }

        return await run_in_threadpool(compute)

    @app.post("/api/fovea-preview")
    async def fovea_preview(request: FoveaRequest) -> dict[str, Any]:
        """Reprocess one frame with a validated, temporary ring geometry."""
        if request.frame_idx >= len(seq):
            raise HTTPException(404, "Frame is outside this sequence")
        if request.ring1_m <= request.ring0_m:
            raise HTTPException(422, "Ring 1 must extend beyond Ring 0")
        if request.cell0_cm not in {5, 10} or request.cell1_cm not in {10, 20}:
            raise HTTPException(422, "Choose 5 or 10 cm for Ring 0 and 10 or 20 cm for Ring 1")
        spec = spec_from_rings("interactive_preview", [
            (request.ring0_m * 1000, request.cell0_cm * 10),
            (request.ring1_m * 1000, request.cell1_cm * 10),
            (60000, 200), (100000, 400),
        ])
        try:
            validate_preset(spec, expected_extent_mm=cfgs.grid.extent_mm)
        except PresetError as exc:
            raise HTTPException(422, str(exc)) from exc

        def compute() -> dict[str, Any]:
            preview_runner = PipelineRunner(
                mode=runner.mode, model=runner.model, preset=spec,
                cfgs=cfgs, device=runner.device,
            )
            started = perf_counter()
            result = preview_runner.process(seq, request.frame_idx)
            elapsed_ms = (perf_counter() - started) * 1000
            return {
                "recompute_ms": round(elapsed_ms, 1),
                "frame": serialise_frame_result(
                    result, seq.seq, float(seq.times[request.frame_idx]),
                    runner.mode, spec.name,
                ),
            }

        return await run_in_threadpool(compute)

    @app.post("/api/hazard-preview")
    async def hazard_preview(request: HazardRequest) -> dict[str, Any]:
        """Inject one synthetic hazard into a copy of an oracle scan and re-run the pipeline."""
        if runner.mode != "oracle":
            raise HTTPException(409, "Hazard preview requires oracle mode")
        if request.frame_idx >= len(seq):
            raise HTTPException(404, "Frame is outside this sequence")
        if request.kind not in {"kerb", "pothole", "overhang"}:
            raise HTTPException(422, "Unknown hazard type")

        def compute() -> dict[str, Any]:
            from foveamap.derived.hazards import Hazard, check_hazard_detected
            from foveamap.grid.engine import world_to_cell
            from foveamap.io.labels import DRIVABLE, raw_to_super

            scan = seq.load_frame(request.frame_idx)
            if scan.raw_labels is None:
                raise ValueError("Ground-truth labels are required")
            classes, _ = raw_to_super(scan.raw_labels)
            dist2 = ((scan.xyz[:, 0] - request.x_m) ** 2
                     + (scan.xyz[:, 1] - request.y_m) ** 2)
            ground = (classes == DRIVABLE) & (dist2 < 1.0)
            if not np.any(ground):
                raise ValueError("Choose a point on observed drivable ground")
            base_z = float(np.median(scan.xyz[ground, 2]))
            xyz = scan.xyz.copy()
            labels = scan.raw_labels.copy()
            remission = scan.remission.copy()
            params: dict[str, float] = {}
            if request.kind == "pothole":
                affected = (classes == DRIVABLE) & (dist2 <= 0.5 ** 2)
                xyz[affected, 2] -= 0.18
                params = {"depth_m": 0.18, "radius_m": 0.5, "detect_frac": 0.5}
            elif request.kind == "kerb":
                affected = (classes == DRIVABLE) & (np.abs(xyz[:, 0] - request.x_m) <= 1.5) & (np.abs(xyz[:, 1] - request.y_m) <= 0.25)
                xyz[affected, 2] += 0.16
                params = {"height_m": 0.16, "length_m": 3.0, "width_m": 0.5}
            else:
                gx, gy = np.meshgrid(
                    np.linspace(request.x_m - 1.5, request.x_m + 1.5, 30),
                    np.linspace(request.y_m - 0.5, request.y_m + 0.5, 12),
                )
                slab = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, base_z + 1.5))).astype(np.float32)
                xyz = np.vstack((xyz, slab))
                labels = np.concatenate((labels, np.full(len(slab), 50, dtype=labels.dtype)))
                remission = np.concatenate((remission, np.zeros(len(slab), dtype=remission.dtype)))
                params = {"height_above_ground_m": 1.5, "length_m": 3.0, "width_m": 1.0}

            injected_scan = replace(scan, xyz=xyz, raw_labels=labels, remission=remission)

            class PreviewSequence:
                def load_frame(self, _: int) -> Any:
                    return injected_scan

            started = perf_counter()
            result = runner.process(PreviewSequence(), request.frame_idx)  # type: ignore[arg-type]
            elapsed_ms = (perf_counter() - started) * 1000
            hazard = Hazard(request.kind, (request.x_m, request.y_m), params)
            cell = world_to_cell(runner.preset, int(request.x_m * 1000), int(request.y_m * 1000))
            return {
                "kind": request.kind,
                "center_m": [request.x_m, request.y_m],
                "ring_idx": cell[0] if cell else None,
                "detected": check_hazard_detected(hazard, result.layers),
                "recompute_ms": round(elapsed_ms, 1),
                "frame": serialise_frame_result(
                    result, seq.seq, float(seq.times[request.frame_idx]),
                    "oracle + synthetic hazard", runner.preset.name,
                ),
            }

        try:
            return await run_in_threadpool(compute)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/play")
    async def api_play() -> dict[str, Any]:
        mgr = get_playback_manager()
        await mgr.play()
        return mgr.get_state()

    @app.post("/api/pause")
    async def api_pause() -> dict[str, Any]:
        mgr = get_playback_manager()
        await mgr.pause()
        return mgr.get_state()

    @app.post("/api/seek/{idx}")
    async def api_seek(idx: int) -> dict[str, Any]:
        mgr = get_playback_manager()
        await mgr.seek(idx)
        return mgr.get_state()

    @app.post("/api/step/{delta}")
    async def api_step(delta: int) -> dict[str, Any]:
        mgr = get_playback_manager()
        await mgr.step(delta)
        return mgr.get_state()

    # Serve built React frontend if dist exists
    dashboard_dist = Path(__file__).resolve().parents[2] / "dashboard" / "dist"
    if dashboard_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=str(dashboard_dist / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file_path = dashboard_dist / full_path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(dashboard_dist / "index.html")

    # Wrap FastAPI with Socket.IO ASGI
    asgi_app = socketio.ASGIApp(
        sio,
        other_asgi_app=app,
        socketio_path="socket.io",
    )

    return asgi_app
