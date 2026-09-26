import React, { useEffect, useRef, useState } from "react";
import type { PlaybackState } from "../types";
import { socket } from "../socket";

interface PlaybackControlsProps {
  state: PlaybackState | null;
  currentFrameIdx: number;
}

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({ state, currentFrameIdx }) => {
  const isPlaying = state?.is_playing ?? false;
  const totalFrames = state?.total_frames ?? 0;
  const currentSpeed = state?.speed ?? 1.0;

  // Local slider position during drag — syncs with server frame_idx when not dragging
  const isSeekingRef = useRef(false);
  const [seekValue, setSeekValue] = useState<number>(currentFrameIdx);

  useEffect(() => {
    if (!isSeekingRef.current) {
      setSeekValue(currentFrameIdx);
    }
  }, [currentFrameIdx]);

  const handlePlayPause = () => {
    if (isPlaying) {
      socket.emit("pause");
    } else {
      socket.emit("play");
    }
  };

  const handleStep = (delta: number) => {
    socket.emit("step", { delta });
  };

  const handleSeekChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    isSeekingRef.current = true;
    setSeekValue(parseInt(e.target.value, 10));
  };

  const handleSeekCommit = () => {
    socket.emit("seek_frame", { frame_idx: seekValue });
    isSeekingRef.current = false;
  };

  const handleSpeed = (speed: number) => {
    socket.emit("set_speed", { speed });
  };

  return (
    <div className="playback-panel playback-compact">
      <header><span>Sequence playback</span><small>{isPlaying ? "Playing" : "Paused"}</small></header>
      {/* Row 1: transport controls + speed */}
      <div className="playback-row-top">
        <div className="playback-buttons">
          <button className="play-btn step-btn" onClick={() => handleStep(-1)} title="Step Backward" disabled={!state}>⏮</button>
          <button
            className={`play-btn primary-play-btn ${isPlaying ? "playing" : ""}`}
            onClick={handlePlayPause}
            title={isPlaying ? "Pause" : "Play"}
            disabled={!state}
          >
            {isPlaying ? "⏸" : "▶"}
          </button>
          <button className="play-btn step-btn" onClick={() => handleStep(1)} title="Step Forward" disabled={!state}>⏭</button>
        </div>

        <div className="frame-counter" style={{ flex: 1, justifyContent: "center" }}>
          <span className="current-frame">{String(currentFrameIdx).padStart(6, "0")}</span>
          <span className="divider">/</span>
          <span className="total-frame">{String(totalFrames).padStart(6, "0")}</span>
        </div>

        <div className="speed-selector">
          {[0.5, 1.0, 2.0, 5.0].map((s) => (
            <button
              key={s}
              className={`speed-btn ${Math.abs(currentSpeed - s) < 0.05 ? "speed-active" : ""}`}
              onClick={() => handleSpeed(s)}
              disabled={!state}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* Row 2: scrub slider full width */}
      <input
        type="range"
        aria-label="Seek sequence frame"
        min={0}
        max={Math.max(1, totalFrames - 1)}
        value={seekValue}
        onChange={handleSeekChange}
        onMouseUp={handleSeekCommit}
        onTouchEnd={handleSeekCommit}
        onKeyUp={handleSeekCommit}
        className="scrub-slider scrub-full"
        disabled={!state || totalFrames < 2}
      />
    </div>
  );
};
