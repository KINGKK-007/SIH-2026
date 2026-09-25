import React from "react";
import type { PlaybackState } from "../types";
import { socket } from "../socket";

interface PlaybackControlsProps {
  state: PlaybackState | null;
  currentFrameIdx: number;
}

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({ state, currentFrameIdx }) => {
  const isPlaying = state?.is_playing ?? false;
  const totalFrames = state?.total_frames ?? 4071;
  const currentSpeed = state?.speed ?? 1.0;

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

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const idx = parseInt(e.target.value, 10);
    socket.emit("seek_frame", { frame_idx: idx });
  };

  const handleSpeed = (speed: number) => {
    socket.emit("set_speed", { speed });
  };

  return (
    <div className="playback-panel">
      <div className="playback-buttons">
        <button className="play-btn step-btn" onClick={() => handleStep(-1)} title="Step Backward">
          ⏮
        </button>

        <button
          className={`play-btn primary-play-btn ${isPlaying ? "playing" : ""}`}
          onClick={handlePlayPause}
          title={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>

        <button className="play-btn step-btn" onClick={() => handleStep(1)} title="Step Forward">
          ⏭
        </button>
      </div>

      <div className="slider-wrapper">
        <input
          type="range"
          min={0}
          max={Math.max(1, totalFrames - 1)}
          value={currentFrameIdx}
          onChange={handleSeek}
          className="scrub-slider"
        />
        <div className="frame-counter">
          <span className="current-frame">{String(currentFrameIdx).padStart(6, "0")}</span>
          <span className="divider">/</span>
          <span className="total-frame">{String(totalFrames).padStart(6, "0")}</span>
        </div>
      </div>

      <div className="speed-selector">
        {[0.5, 1.0, 2.0, 5.0].map((s) => (
          <button
            key={s}
            className={`speed-btn ${Math.abs(currentSpeed - s) < 0.05 ? "speed-active" : ""}`}
            onClick={() => handleSpeed(s)}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
};
