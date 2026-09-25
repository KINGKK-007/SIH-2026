import React, { useEffect, useRef, useState } from "react";
import type { ActiveLayer, FrameUpdatePayload } from "../types";

interface MapViewProps {
  frame: FrameUpdatePayload | null;
  activeLayer: ActiveLayer;
  onLayerChange: (layer: ActiveLayer) => void;
}

const CLASS_COLORS: Record<number, string> = {
  0: "rgba(120, 120, 130, 0.4)", // UNKNOWN
  1: "#2ecc71",                  // DRIVABLE (green)
  2: "#d35400",                  // NON_DRIVABLE_TERRAIN (brown)
  3: "#e74c3c",                  // STATIC_OBSTACLE (red)
  4: "#3498db",                  // DYNAMIC (blue/cyan)
};

const TRAV_COLORS: Record<number, string> = {
  0: "rgba(120, 120, 130, 0.3)",
  1: "#2ecc71", // Traversable
  2: "#e74c3c", // Non-traversable
  3: "#e74c3c",
  4: "#e74c3c",
};

export const MapView: React.FC<MapViewProps> = ({ frame, activeLayer, onLayerChange }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState<number>(1.0);
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoverInfo, setHoverInfo] = useState<string | null>(null);

  // Redraw canvas whenever frame, activeLayer, zoom, or offset changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2 + offset.x;
    const centerY = height / 2 + offset.y;

    // Clear background
    ctx.fillStyle = "#0c1017";
    ctx.fillRect(0, 0, width, height);

    // Max extent across all rings (default ±100m = 100,000 mm)
    const maxExtentMm = frame?.rings?.length
      ? Math.max(...frame.rings.map((r) => r.r_max_mm))
      : 100000;

    // Pixels per mm at zoom=1.0: fit maxExtentMm in canvas with margin
    const baseScale = (Math.min(width, height) * 0.44) / maxExtentMm;
    const scale = baseScale * zoom;

    // Draw coordinate grid lines (every 20m)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let r = 20000; r <= maxExtentMm; r += 20000) {
      const px = r * scale;
      ctx.beginPath();
      ctx.arc(centerX, centerY, px, 0, 2 * Math.PI);
      ctx.stroke();

      // Range text
      ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
      ctx.font = "10px Inter, monospace";
      ctx.fillText(`${r / 1000}m`, centerX + 4, centerY - px + 12);
    }

    // Draw axes (+x is forward/up, +y is left)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // Draw ego-vehicle marker at origin
    ctx.fillStyle = "#f39c12";
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - 10);
    ctx.lineTo(centerX - 6, centerY + 6);
    ctx.lineTo(centerX + 6, centerY + 6);
    ctx.closePath();
    ctx.fill();

    if (!frame || !frame.rings || frame.rings.length === 0) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
      ctx.font = "14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for frame stream...", width / 2, height / 2 + 50);
      return;
    }

    // Render rings from outer (coarsest) to inner (finest) so fine rings draw on top
    const sortedRings = [...frame.rings].sort((a, b) => b.ring_idx - a.ring_idx);

    for (const ring of sortedRings) {
      const cellMm = ring.cell_mm;
      const cellPx = Math.max(1.0, cellMm * scale);
      const rMax = ring.r_max_mm;

      // Draw ring square boundary
      const boundaryPx = rMax * scale;
      ctx.strokeStyle = ring.ring_idx === 0 ? "rgba(46, 204, 113, 0.35)" : "rgba(52, 152, 219, 0.2)";
      ctx.lineWidth = 1;
      ctx.strokeRect(centerX - boundaryPx, centerY - boundaryPx, boundaryPx * 2, boundaryPx * 2);

      // Render cells
      const nCells = ring.ix.length;
      for (let i = 0; i < nCells; i++) {
        const ix = ring.ix[i];
        const iy = ring.iy[i];

        // World coordinates from cell index:
        // ix goes from -R to +R (+x forward), iy goes from -R to +R (+y left)
        const cellXmm = -rMax + ix * cellMm;
        const cellYmm = -rMax + iy * cellMm;

        // Canvas coords (+x up = -screenY, +y left = -screenX)
        const screenX = centerX - (cellYmm + cellMm / 2) * scale;
        const screenY = centerY - (cellXmm + cellMm / 2) * scale;

        // Cell color based on active layer
        if (activeLayer === "class") {
          const clsId = ring.cls[i] || 0;
          ctx.fillStyle = CLASS_COLORS[clsId] || CLASS_COLORS[0];
        } else if (activeLayer === "height") {
          const z = ring.top_z[i] !== -32768 ? ring.top_z[i] : ring.ground_z[i];
          const norm = Math.max(0, Math.min(1, (z / 1000 + 2.5) / 6.0));
          const r = Math.round(norm < 0.5 ? norm * 60 : 30 + (norm - 0.5) * 450);
          const g = Math.round(norm < 0.5 ? norm * 360 : 180 + (norm - 0.5) * 100);
          const b = Math.round(norm < 0.5 ? 120 + norm * 120 : 180 - (norm - 0.5) * 300);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
        } else if (activeLayer === "traversability") {
          const clsId = ring.cls[i] || 0;
          ctx.fillStyle = TRAV_COLORS[clsId] || TRAV_COLORS[0];
        } else if (activeLayer === "moving") {
          const frac = (ring.moving_frac[i] || 0) / 255.0;
          ctx.fillStyle = frac > 0.1 ? `rgba(231, 76, 60, ${0.4 + frac * 0.6})` : "rgba(52, 152, 219, 0.5)";
        } else {
          const conf = (ring.conf[i] || 0) / 255.0;
          ctx.fillStyle = `rgba(46, 204, 113, ${0.2 + conf * 0.8})`;
        }

        ctx.fillRect(screenX, screenY, cellPx, cellPx);
      }
    }

    // Draw detected object bounding boxes if present
    if (frame.objects && frame.objects.length > 0) {
      for (const obj of frame.objects) {
        const [ox, oy] = obj.center;
        const [l, w] = obj.size;
        const screenX = centerX - (oy * 1000) * scale;
        const screenY = centerY - (ox * 1000) * scale;
        const lPx = l * 1000 * scale;
        const wPx = w * 1000 * scale;

        ctx.save();
        ctx.translate(screenX, screenY);
        ctx.rotate(-obj.yaw);
        ctx.strokeStyle = obj.moving ? "#e74c3c" : "#f1c40f";
        ctx.lineWidth = 2;
        ctx.strokeRect(-wPx / 2, -lPx / 2, wPx, lPx);
        ctx.restore();
      }
    }
  }, [frame, activeLayer, zoom, offset]);

  // Handle canvas interactions
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2 + offset.x;
    const centerY = height / 2 + offset.y;

    const maxExtentMm = frame?.rings?.length
      ? Math.max(...frame.rings.map((r) => r.r_max_mm))
      : 100000;
    const baseScale = (Math.min(width, height) * 0.44) / maxExtentMm;
    const scale = baseScale * zoom;

    // Convert pixel to ground coordinates (+x forward, +y left)
    const yMm = -(mx - centerX) / scale;
    const xMm = -(my - centerY) / scale;
    const xM = xMm / 1000.0;
    const yM = yMm / 1000.0;
    const distM = Math.sqrt(xM * xM + yM * yM);

    // Identify ring
    let ringIdx = -1;
    let ringRes = "";
    if (frame?.rings) {
      for (const r of frame.rings) {
        if (Math.abs(xMm) <= r.r_max_mm && Math.abs(yMm) <= r.r_max_mm) {
          ringIdx = r.ring_idx;
          ringRes = `${r.cell_mm / 10} cm`;
          break;
        }
      }
    }

    setHoverInfo(`X: ${xM.toFixed(1)}m | Y: ${yM.toFixed(1)}m | Range: ${distM.toFixed(1)}m | Ring ${ringIdx >= 0 ? ringIdx : "Out"} (${ringRes || "—"})`);
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom((z) => Math.max(0.4, Math.min(6.0, z * factor)));
  };

  return (
    <div className="map-view-wrapper">
      <div className="map-toolbar">
        <div className="layer-selector">
          {(["class", "height", "traversability", "moving", "confidence"] as ActiveLayer[]).map((lyr) => (
            <button
              key={lyr}
              className={`layer-btn ${activeLayer === lyr ? "layer-active" : ""}`}
              onClick={() => onLayerChange(lyr)}
            >
              {lyr.toUpperCase()}
            </button>
          ))}
        </div>

        <div className="zoom-controls">
          <button className="ctrl-btn" onClick={() => setZoom((z) => Math.min(6.0, z * 1.2))}>+</button>
          <button className="ctrl-btn" onClick={() => setZoom((z) => Math.max(0.4, z / 1.2))}>−</button>
          <button
            className="ctrl-btn reset-btn"
            onClick={() => {
              setZoom(1.0);
              setOffset({ x: 0, y: 0 });
            }}
          >
            Reset
          </button>
        </div>
      </div>

      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={800}
          height={700}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        />
        {hoverInfo && <div className="map-cursor-readout">{hoverInfo}</div>}
      </div>

      <div className="legend-bar">
        <span className="legend-item"><span className="legend-dot" style={{ background: "#2ecc71" }}></span> Drivable</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: "#e74c3c" }}></span> Obstacle</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: "#d35400" }}></span> Terrain</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: "#3498db" }}></span> Dynamic</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: "#787882" }}></span> Unknown</span>
      </div>
    </div>
  );
};
