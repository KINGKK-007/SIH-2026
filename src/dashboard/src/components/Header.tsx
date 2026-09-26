import React from "react";
import type { FrameUpdatePayload, PlaybackState } from "../types";

interface HeaderProps {
  connected: boolean;
  frame: FrameUpdatePayload | null;
  state: PlaybackState | null;
  title: string;
}

export const Header: React.FC<HeaderProps> = ({ connected, frame, state, title }) => {
  const sequence = frame?.seq || state?.seq || "08";
  const rings = frame?.rings ?? [];
  const gridSummary = rings.length
    ? `${rings.length} resolution zones · ${rings.map((ring) => `${formatCellSize(ring.cell_mm)}`).join(" / ")} · ${formatRange(rings.at(-1)?.r_max_mm ?? 0)}`
    : "Resolution geometry will appear with the first live frame";
  return <header className="console-header">
    <div>
      <p className="header-kicker">SemanticKITTI · Sequence {sequence}</p>
      <h1>{title}</h1>
      <p className="header-description">Adaptive 2.5D mapping · {gridSummary}</p>
    </div>
    <div className="header-statuses">
      <span className={`live-status ${connected ? "online" : "offline"}`}><i />{connected ? state?.is_playing ? "Streaming" : "Connected · paused" : "Reconnecting"}</span>
    </div>
  </header>;
};

const formatCellSize = (cellMm: number) => cellMm % 10 === 0 ? `${cellMm / 10} cm` : `${cellMm} mm`;
const formatRange = (rangeMm: number) => `${rangeMm / 1000} m range`;
