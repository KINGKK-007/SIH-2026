import type { FrameUpdatePayload, PlaybackState } from "../types";

interface HeaderProps {
  connected: boolean;
  frame: FrameUpdatePayload | null;
  state: PlaybackState | null;
  title: string;
}

export function Header({ connected, frame, state, title }: HeaderProps) {
  const sequence = frame?.seq || state?.seq || "08";
  const rings = frame?.rings ?? [];
  const extent = rings.at(-1)?.r_max_mm;
  const source = frame?.model ?? (state ? `${state.mode}/${state.model}` : null);
  const sourceLabel = source === "oracle" ? "ORACLE · GROUND TRUTH" : source ? source.replace("/", " · ").toUpperCase() : "AWAITING DATA";
  return <header className="console-header">
    <div className="header-main">
      <div className="header-breadcrumb">SemanticKITTI <span>/</span> Sequence {sequence} <span>/</span> Frame <strong>{frame ? String(frame.frame_idx).padStart(6, "0") : "——————"}</strong></div>
      <div className="header-title-row"><h1>{title}</h1><span className="source-badge">{sourceLabel}</span></div>
      <div className="header-meta"><span>{extent ? `±${extent / 1000} m extent` : "Extent pending"}</span><i /><span>{rings.length ? `${rings.length} adaptive zones` : "Zones pending"}</span><i /><span>Vehicle centred</span></div>
    </div>
    <div className="header-statuses">
      <span className={`live-status ${connected ? "online" : "offline"}`}><i />{connected ? state?.is_playing ? "Streaming" : "Connected" : "Reconnecting"}</span>
    </div>
  </header>;
}
