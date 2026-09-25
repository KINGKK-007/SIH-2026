"""End-to-end tests for the FastAPI + Socket.IO server (T13.4)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from foveamap.config import load_config
from foveamap.grid.presets import load_preset
from foveamap.io.sequence import Sequence
from foveamap.models.oracle import OracleModel
from foveamap.pipeline.runner import PipelineRunner
from foveamap.server.app import create_app
from foveamap.server.sockets import get_playback_manager


def test_api_endpoints(synthetic_root: Path) -> None:
    cfgs = load_config()
    preset = load_preset("fovea_default", cfgs.grid)
    runner = PipelineRunner(mode="oracle", model=OracleModel(), preset=preset, cfgs=cfgs)
    seq = Sequence(synthetic_root, "08")

    asgi_app = create_app(runner=runner, seq=seq, cfgs=cfgs)

    # asgi_app is socketio.ASGIApp; other_asgi_app is the FastAPI instance
    fastapi_app = asgi_app.other_asgi_app
    client = TestClient(fastapi_app)

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "app": "foveamap"}

    # Initial state
    res = client.get("/api/state")
    assert res.status_code == 200
    state = res.json()
    assert state["seq"] == "08"
    assert state["mode"] == "oracle"
    assert state["is_playing"] is False

    # Presets
    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.json()
    assert "fovea_default" in presets["available"]

    # Step
    res = client.post("/api/step/1")
    assert res.status_code == 200
    assert res.json()["frame_idx"] == 1

    # Seek
    res = client.post("/api/seek/0")
    assert res.status_code == 200
    assert res.json()["frame_idx"] == 0


@pytest.mark.anyio
async def test_playback_loop_and_emit(synthetic_root: Path) -> None:
    cfgs = load_config()
    preset = load_preset("fovea_default", cfgs.grid)
    runner = PipelineRunner(mode="oracle", model=OracleModel(), preset=preset, cfgs=cfgs)
    seq = Sequence(synthetic_root, "08")

    create_app(runner=runner, seq=seq, cfgs=cfgs)
    mgr = get_playback_manager()

    payload = await mgr.emit_current_frame()
    assert payload["schema_version"] == 1
    assert payload["seq"] == "08"
    assert payload["frame_idx"] == 0
    assert "counters" in payload
    assert "timings_ms" in payload
    assert "memory" in payload
    assert "rings" in payload

    # Test seek
    payload = await mgr.seek(1)
    assert payload["frame_idx"] == 1

    # Test step
    payload = await mgr.step(-1)
    assert payload["frame_idx"] == 0
