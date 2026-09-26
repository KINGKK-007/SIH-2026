import React from "react";
import type { FrameCounters, TimingsMs } from "../types";

interface LatencyPanelProps {
  timings?: TimingsMs;
  counters?: FrameCounters;
}

export const LatencyPanel: React.FC<LatencyPanelProps> = ({ timings, counters }) => {
  if (!timings) {
    return (
      <div className="card latency-card">
        <h3 className="card-title">Pipeline Latency</h3>
        <div className="empty-state">No latency telemetry</div>
      </div>
    );
  }

  const stages = [
    { key: "io_ms", label: "I/O scan load", color: "#6f7471" },
    { key: "model_ms", label: "Model / oracle", color: "#a88e63" },
    { key: "label_ms", label: "Label mapping", color: "#827b69" },
    { key: "motion_ms", label: "Motion / objects", color: "#a75f59" },
    { key: "grid_ms", label: "Grid rasterise", color: "#c69a52" },
    { key: "finalize_ms", label: "Finalize layers", color: "#9a9d98" },
    { key: "derived_ms", label: "Derived layers", color: "#748e73" },
    { key: "memory_ms", label: "Memory report", color: "#6f6566" },
  ];

  const totalMs: number = Object.values(timings).reduce<number>(
    (acc, v) => (typeof v === "number" ? acc + v : acc),
    0
  );
  const deadlineMs = 100.0; // 100 ms HDL-64E period (10 Hz)
  const isUnderDeadline = totalMs <= deadlineMs;

  return (
    <div className="card latency-card">
      <div className="card-header">
        <h3 className="card-title">Latency Breakdown</h3>
        <div className={`status-badge ${isUnderDeadline ? "status-good" : "status-warn"}`}>
          {totalMs.toFixed(1)} ms / 100 ms
        </div>
      </div>

      <div className="stacked-latency-bar">
        {stages.map((stg) => {
          const val = timings[stg.key] || 0;
          const pct = totalMs > 0 ? (val / totalMs) * 100 : 0;
          return (
            <div
              key={stg.key}
              className="stacked-slice"
              style={{
                width: `${pct}%`,
                backgroundColor: stg.color,
              }}
              title={`${stg.label}: ${val.toFixed(2)} ms (${pct.toFixed(0)}%)`}
            />
          );
        })}
      </div>

      <div className="latency-stages-grid">
        {stages.map((stg) => {
          const val = timings[stg.key] || 0;
          return (
            <div key={stg.key} className="stage-mini-item">
              <span className="stage-indicator" style={{ backgroundColor: stg.color }}></span>
              <span className="stage-name">{stg.label}</span>
              <span className="stage-val">{val.toFixed(1)} ms</span>
            </div>
          );
        })}
      </div>

      {counters && (
        <div className="counters-box">
          <div className="counter-item">
            <span className="counter-num">{counters.n_raw.toLocaleString()}</span>
            <span className="counter-lbl">Raw Points</span>
          </div>
          <div className="counter-item">
            <span className="counter-num counter-good">{counters.n_in_grid.toLocaleString()}</span>
            <span className="counter-lbl">In Grid</span>
          </div>
          <div className="counter-item">
            <span className="counter-num counter-muted">{counters.n_out_of_grid.toLocaleString()}</span>
            <span className="counter-lbl">Out of Bounds</span>
          </div>
          <div className="counter-item">
            <span className="counter-num">{counters.n_invalid.toLocaleString()}</span>
            <span className="counter-lbl">Invalid</span>
          </div>
        </div>
      )}
    </div>
  );
};
