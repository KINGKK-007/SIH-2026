import React, { useEffect, useState, useRef } from "react";
import { socket } from "./socket";
import type { ActiveLayer, FrameUpdatePayload, PlaybackState } from "./types";
import { Header } from "./components/Header";
import { MapView } from "./components/MapView";
import { PlaybackControls } from "./components/PlaybackControls";
import { MemoryMeter } from "./components/MemoryMeter";
import { LatencyPanel } from "./components/LatencyPanel";

export const App: React.FC = () => {
  const [connected, setConnected] = useState<boolean>(socket.connected);
  const [frame, setFrame] = useState<FrameUpdatePayload | null>(null);
  const [playbackState, setPlaybackState] = useState<PlaybackState | null>(null);
  const [activeLayer, setActiveLayer] = useState<ActiveLayer>("class");
  const [fps, setFps] = useState<number>(0);

  // FPS tracking
  const frameTimesRef = useRef<number[]>([]);

  useEffect(() => {
    function onConnect() {
      setConnected(true);
    }

    function onDisconnect() {
      setConnected(false);
    }

    function onStateUpdate(state: PlaybackState) {
      setPlaybackState(state);
    }

    function onFrameUpdate(payload: FrameUpdatePayload) {
      setFrame(payload);

      // Track display FPS
      const now = performance.now();
      frameTimesRef.current.push(now);
      while (frameTimesRef.current.length > 0 && frameTimesRef.current[0] < now - 1000) {
        frameTimesRef.current.shift();
      }
      setFps(frameTimesRef.current.length);
    }

    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("state_update", onStateUpdate);
    socket.on("frame_update", onFrameUpdate);

    return () => {
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("state_update", onStateUpdate);
      socket.off("frame_update", onFrameUpdate);
    };
  }, []);

  const currentFrameIdx = frame?.frame_idx ?? playbackState?.frame_idx ?? 0;

  return (
    <div className="foveamap-app">
      <Header
        connected={connected}
        frame={frame}
        state={playbackState}
        fps={fps}
      />

      <main className="dashboard-grid">
        <section className="main-viewport-col">
          <MapView
            frame={frame}
            activeLayer={activeLayer}
            onLayerChange={setActiveLayer}
          />
          <PlaybackControls
            state={playbackState}
            currentFrameIdx={currentFrameIdx}
          />
        </section>

        <aside className="telemetry-sidebar-col">
          <MemoryMeter memory={frame?.memory} />
          <LatencyPanel timings={frame?.timings_ms} counters={frame?.counters} />
        </aside>
      </main>
    </div>
  );
};

export default App;
