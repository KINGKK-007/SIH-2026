import React from "react";
import type { MemoryReport } from "../types";

interface MemoryMeterProps {
  memory?: MemoryReport;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GiB`;
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

export const MemoryMeter: React.FC<MemoryMeterProps> = ({ memory }) => {
  if (!memory) {
    return (
      <div className="card memory-card">
        <h3 className="card-title">Memory Representation</h3>
        <div className="empty-state">No memory metrics yet</div>
      </div>
    );
  }

  const { dense3d_bytes, uniform25d_bytes, fovea_bytes, sparse3d_bytes } = memory;
  const reductionRatio = uniform25d_bytes > 0 && fovea_bytes > 0 ? (uniform25d_bytes / fovea_bytes).toFixed(1) : "—";
  const denseRatio = dense3d_bytes > 0 && fovea_bytes > 0 ? Math.round(dense3d_bytes / fovea_bytes) : "—";

  // Use logarithmic scale for bar widths because values span from KB to GB
  const maxLog = Math.log10(Math.max(dense3d_bytes, 1));
  const getWidthPercent = (val: number) => {
    if (val <= 0) return "2%";
    const log = Math.log10(val);
    const p = Math.max(4, Math.min(100, (log / maxLog) * 100));
    return `${p.toFixed(1)}%`;
  };

  return (
    <div className="card memory-card">
      <div className="card-header">
        <h3 className="card-title">Memory Accounting</h3>
        <span className="badge-highlight">{reductionRatio}× Reduction</span>
      </div>

      <div className="memory-bars-list">
        <div className="memory-bar-item">
          <div className="bar-labels">
            <span className="label-name">Dense 3D (Theoretical)</span>
            <span className="label-val">{formatBytes(dense3d_bytes)}</span>
          </div>
          <div className="progress-bg">
            <div className="progress-fill fill-dense" style={{ width: getWidthPercent(dense3d_bytes) }}></div>
          </div>
        </div>

        <div className="memory-bar-item">
          <div className="bar-labels">
            <span className="label-name">Uniform 2.5D (finest cell baseline)</span>
            <span className="label-val">{formatBytes(uniform25d_bytes)}</span>
          </div>
          <div className="progress-bg">
            <div className="progress-fill fill-uniform" style={{ width: getWidthPercent(uniform25d_bytes) }}></div>
          </div>
        </div>

        <div className="memory-bar-item highlight-item">
          <div className="bar-labels">
            <span className="label-name">FoveaMap (adaptive grid)</span>
            <span className="label-val highlight-val">{formatBytes(fovea_bytes)}</span>
          </div>
          <div className="progress-bg">
            <div className="progress-fill fill-fovea" style={{ width: getWidthPercent(fovea_bytes) }}></div>
          </div>
        </div>

        <div className="memory-bar-item">
          <div className="bar-labels">
            <span className="label-name">Sparse 3D (Occupied points)</span>
            <span className="label-val">{formatBytes(sparse3d_bytes)}</span>
          </div>
          <div className="progress-bg">
            <div className="progress-fill fill-sparse" style={{ width: getWidthPercent(sparse3d_bytes) }}></div>
          </div>
        </div>
      </div>

      <div className="memory-footer">
        <div className="stat-pill">
          <span className="stat-pill-label">vs Uniform 2.5D:</span>
          <span className="stat-pill-val">{reductionRatio}× smaller</span>
        </div>
        <div className="stat-pill">
          <span className="stat-pill-label">vs Dense 3D:</span>
          <span className="stat-pill-val">{denseRatio}× smaller</span>
        </div>
      </div>
      <p className="memory-scale-note">Bar lengths use a logarithmic scale. Map allocation is not GPU VRAM. Basis: {memory.basis}.</p>
    </div>
  );
};
