"""FastAPI app factory with the Socket.IO ASGI mount (README 9.9, task T13.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import socketio

from foveamap.config import load_config
from foveamap.grid.presets import load_preset
from foveamap.io.sequence import Sequence
from foveamap.models.oracle import OracleModel
from foveamap.pipeline.runner import PipelineRunner
from foveamap.server.sockets import (
    PlaybackManager,
    get_playback_manager,
    set_playback_manager,
    sio,
)


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
