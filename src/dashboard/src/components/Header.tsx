import React from "react";
import type { FrameUpdatePayload, PlaybackState } from "../types";

interface HeaderProps {
  connected: boolean;
  frame: FrameUpdatePayload | null;
  state: PlaybackState | null;
  fps: number;
}

export const Header: React.FC<HeaderProps> = ({ connected, frame, state, fps }) => {
  const modelName = frame?.model || state?.model || "oracle";
  const presetName = frame?.preset || state?.preset || "fovea_default";
  const seqName = frame?.seq || state?.seq || "08";

  return (
    <header className="header-container">
      <div className="header-left">
        <div className="brand-logo">
          <div className="logo-ring outer"></div>
          <div className="logo-ring mid"></div>
          <div className="logo-ring inner"></div>
          <span className="brand-title">FoveaMap</span>
        </div>
        <span className="brand-tagline">Multi-Ring 2.5D Semantic LiDAR Map</span>
      </div>

      <div className="header-center">
        <div className="badge badge-seq">
          <span className="badge-label">Sequence</span>
          <span className="badge-value">{seqName}</span>
        </div>

        <div className={`badge badge-mode ${modelName.includes("oracle") ? "mode-oracle" : "mode-model"}`}>
          <span className="badge-dot"></span>
          <span className="badge-value">
            {modelName === "oracle" ? "ORACLE (Ground Truth)" : `MODEL (${modelName})`}
          </span>
        </div>

        <div className="badge badge-preset">
          <span className="badge-label">Preset</span>
          <span className="badge-value">{presetName}</span>
        </div>
      </div>

      <div className="header-right">
        <div className="telemetry-badge">
          <span className="telemetry-label">Display FPS</span>
          <span className="telemetry-val">{fps > 0 ? fps.toFixed(1) : "—"}</span>
        </div>

        <div className={`conn-status ${connected ? "conn-active" : "conn-offline"}`}>
          <span className="conn-dot"></span>
          <span className="conn-text">{connected ? "Connected" : "Reconnecting..."}</span>
        </div>
      </div>
    </header>
  );
};
