import type { FrameUpdatePayload } from "../types";

interface TelemetryStripProps {
  frame: FrameUpdatePayload | null;
  latency: number;
  fps: number;
  activeCells: number;
  displayFps: number;
}

export function TelemetryStrip({ frame, latency, fps, activeCells, displayFps }: TelemetryStripProps) {
  const items = [
    { label: "Pipeline", value: fps ? fps.toFixed(1) : "—", unit: "FPS", detail: "stage estimate" },
    { label: "Latency", value: latency ? latency.toFixed(1) : "—", unit: "ms", detail: "stage total" },
    { label: "Active cells", value: frame ? activeCells.toLocaleString() : "—", unit: "", detail: "observed grid" },
    { label: "Points", value: frame ? frame.counters.n_raw.toLocaleString() : "—", unit: "", detail: "current scan" },
    { label: "Display", value: displayFps ? String(displayFps) : "—", unit: "FPS", detail: "received frames" },
  ];

  return <section className="telemetry-strip" aria-label="Current frame telemetry">
    {items.map((item) => <div className="telemetry-item" key={item.label}>
      <span className="telemetry-label">{item.label}</span>
      <div className="telemetry-value">{item.value}<small>{item.unit}</small></div>
      <span className="telemetry-detail">{item.detail}</span>
    </div>)}
  </section>;
}
