"""Socket.IO event handlers and playback loop (README 13.3, task T13.2)."""

from __future__ import annotations

import asyncio
from typing import Any

import socketio

from foveamap.io.sequence import Sequence
from foveamap.pipeline.runner import PipelineRunner
from foveamap.server.protocol import serialise_frame_result

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")


class PlaybackManager:
    """Manages playback state and the background push loop."""

    def __init__(
        self,
        runner: PipelineRunner,
        seq: Sequence,
        cfgs: Any,
        model_name: str = "oracle",
    ) -> None:
        self.runner = runner
        self.seq = seq
        self.cfgs = cfgs
        self.model_name = model_name
        self.current_idx = 0
        self.total_frames = len(seq)
        self.is_playing = False
        self.speed = 1.0  # multiplier (1.0 = 10 fps)
        self._loop_task: asyncio.Task[None] | None = None

    def get_state(self) -> dict[str, Any]:
        return {
            "seq": self.seq.seq,
            "frame_idx": self.current_idx,
            "total_frames": self.total_frames,
            "mode": self.runner.mode,
            "model": self.model_name,
            "preset": self.cfgs.grid.active_preset if hasattr(self.cfgs, "grid") else "fovea_default",
            "is_playing": self.is_playing,
            "speed": self.speed,
        }

    async def emit_current_frame(self) -> dict[str, Any]:
        """Process and emit the current frame result."""
        frame_idx = self.seq.frame_index(self.current_idx)
        result = self.runner.process(self.seq, frame_idx)

        # Get timestamp if available
        timestamp = 0.0
        if hasattr(self.seq, "times") and self.current_idx < len(self.seq.times):
            timestamp = float(self.seq.times[self.current_idx])

        preset_name = (
            self.cfgs.grid.active_preset if hasattr(self.cfgs, "grid") else "fovea_default"
        )
        payload = serialise_frame_result(
            result=result,
            seq=self.seq.seq,
            timestamp=timestamp,
            model=f"{self.runner.mode}/{self.model_name}" if self.runner.mode != "oracle" else "oracle",
            preset_name=preset_name,
        )

        await sio.emit("frame_update", payload, namespace="/fovea")
        return payload

    async def play(self) -> None:
        if self.is_playing:
            return
        self.is_playing = True
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._playback_loop())

    async def pause(self) -> None:
        self.is_playing = False
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loop_task = None

    async def seek(self, frame_idx: int) -> dict[str, Any]:
        self.current_idx = max(0, min(frame_idx, self.total_frames - 1))
        return await self.emit_current_frame()

    async def step(self, delta: int = 1) -> dict[str, Any]:
        self.current_idx = (self.current_idx + delta) % max(1, self.total_frames)
        return await self.emit_current_frame()

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, min(speed, 10.0))

    async def _playback_loop(self) -> None:
        while self.is_playing:
            try:
                await self.emit_current_frame()
                self.current_idx = (self.current_idx + 1) % max(1, self.total_frames)
                interval = max(0.01, 0.1 / self.speed)  # 10 Hz base sensor rate
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"Error in playback loop: {exc}")
                await asyncio.sleep(0.5)


_manager: PlaybackManager | None = None


def set_playback_manager(manager: PlaybackManager) -> None:
    global _manager
    _manager = manager


def get_playback_manager() -> PlaybackManager:
    if _manager is None:
        raise RuntimeError("PlaybackManager not initialized")
    return _manager


@sio.on("connect", namespace="/fovea")
async def on_connect(sid: str, environ: dict) -> None:
    if _manager:
        await sio.emit("state_update", _manager.get_state(), to=sid, namespace="/fovea")
        # Emit initial frame upon connection
        await _manager.emit_current_frame()


@sio.on("seek_frame", namespace="/fovea")
async def on_seek_frame(sid: str, data: dict) -> None:
    if _manager and "frame_idx" in data:
        await _manager.seek(int(data["frame_idx"]))


@sio.on("play", namespace="/fovea")
async def on_play(sid: str, data: Any = None) -> None:
    if _manager:
        await _manager.play()


@sio.on("pause", namespace="/fovea")
async def on_pause(sid: str, data: Any = None) -> None:
    if _manager:
        await _manager.pause()


@sio.on("step", namespace="/fovea")
async def on_step(sid: str, data: dict = None) -> None:
    if _manager:
        delta = int(data.get("delta", 1)) if data else 1
        await _manager.step(delta)


@sio.on("set_speed", namespace="/fovea")
async def on_set_speed(sid: str, data: dict) -> None:
    if _manager and "speed" in data:
        _manager.set_speed(float(data["speed"]))
