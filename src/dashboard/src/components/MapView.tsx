import React, { useEffect, useRef, useState } from "react";
import type { ActiveLayer, AnalysisMode, FrameUpdatePayload } from "../types";

interface MapViewProps {
  frame: FrameUpdatePayload | null;
  activeLayer: ActiveLayer;
  onLayerChange: (layer: ActiveLayer) => void;
  analysisMode: AnalysisMode;
  onAnalysisModeChange: (mode: AnalysisMode) => void;
  selectedRing: number | null;
  onRingSelect: (ring: number | null) => void;
  selectedObject: number | null;
  onObjectSelect: (id: number | null) => void;
}

// Bit flags matching Python grid/layers.py
const FLAG_HAS_GROUND   = 0x01;
const FLAG_KERB         = 0x08;
const FLAG_LOW_CLEAR    = 0x10;
const FLAG_TRAVERSABLE  = 0x20;

const TERRAIN_COLORS: Record<number, [number, number, number, number]> = {
  0: [120, 126, 136, 0.35],
  1: [34, 177, 76, 0.94],
  2: [210, 142, 35, 0.92],
  3: [194, 57, 52, 0.90],
  4: [194, 57, 52, 0.90],
};

const OBJECT_COLORS: Record<number, [number, number, number, number]> = {
  0: [140, 145, 154, 0.20],
  1: [101, 116, 132, 0.18],
  2: [101, 116, 132, 0.20],
  3: [230, 112, 38, 0.95],
  4: [16, 164, 203, 0.98],
};

// Per-ring boundary colors (darker for light bg)
const RING_COLORS = [
  "rgba(40, 160, 80, 0.9)",   // ring 0 — green
  "rgba(30, 100, 200, 0.8)",  // ring 1 — blue
  "rgba(200, 130, 0, 0.75)",  // ring 2 — amber
  "rgba(190, 50, 50, 0.70)",  // ring 3 — red
];

function jetColor(t: number): [number, number, number] {
  // Jet colormap: blue → cyan → green → yellow → red
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * t - 3)))));
  const g = Math.round(Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * t - 2)))));
  const b = Math.round(Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * t - 1)))));
  return [r, g, b];
}

export const MapView: React.FC<MapViewProps> = ({
  frame,
  activeLayer,
  onLayerChange,
  analysisMode,
  selectedRing,
  onRingSelect,
  selectedObject,
  onObjectSelect,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState<number>(2.0);
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoverInfo, setHoverInfo] = useState<string | null>(null);
  const [layersOpen, setLayersOpen] = useState(false);
  const [visibility, setVisibility] = useState({ rings: true, objects: true, vehicle: true });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2 + offset.x;
    const centerY = height / 2 + offset.y;

    // Dark perception-console background
    ctx.fillStyle = "#080c11";
    ctx.fillRect(0, 0, width, height);

    const maxExtentMm = frame?.rings?.length
      ? Math.max(...frame.rings.map((r) => r.r_max_mm))
      : 100000;

    const baseScale = (Math.min(width, height) * 0.44) / maxExtentMm;
    const scale = baseScale * zoom;

    // Light grid circles every 20m
    for (let r = 20000; r <= maxExtentMm; r += 20000) {
      const px = r * scale;
      ctx.strokeStyle = "rgba(150,190,220,0.10)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, px, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = "rgba(180,200,215,0.42)";
      ctx.font = "10px monospace";
      ctx.fillText(`${r / 1000}m`, centerX + 4, centerY - px + 12);
    }

    // Axis lines
    ctx.strokeStyle = "rgba(150,190,220,0.13)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, 0); ctx.lineTo(centerX, height);
    ctx.moveTo(0, centerY); ctx.lineTo(width, centerY);
    ctx.stroke();

    ctx.fillStyle = "rgba(180,200,215,0.48)";
    ctx.font = "11px monospace";
    ctx.fillText("↑ FWD", centerX + 5, Math.max(15, centerY - maxExtentMm * scale - 4));

    if (!frame || !frame.rings || frame.rings.length === 0) {
      ctx.fillStyle = "rgba(210,225,235,0.55)";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for frame data...", width / 2, height / 2 + 40);
      ctx.textAlign = "left";
      return;
    }

    // Outer → inner so fine rings overdraw coarse
    const sortedRings = [...frame.rings].sort((a, b) => b.ring_idx - a.ring_idx);

    for (const ring of sortedRings) {
      const cellMm = ring.cell_mm;
      const cellPx = Math.max(1.0, cellMm * scale);
      // Leave a small gap between cells for point-cloud feel
      const drawPx = Math.max(1.0, cellPx * 0.88);
      const gapOff = (cellPx - drawPx) / 2;
      const rMax = ring.r_max_mm;
      const nCells = ring.ix.length;

      // Cells
      for (let i = 0; i < nCells; i++) {
        const ix = ring.ix[i];
        const iy = ring.iy[i];
        const flags = ring.flags[i] || 0;
        const cellXmm = -rMax + ix * cellMm;
        const cellYmm = -rMax + iy * cellMm;
        const screenX = centerX - (cellYmm + cellMm / 2) * scale;
        const screenY = centerY - (cellXmm + cellMm / 2) * scale;

        let r = 170, g = 170, b = 175, a = 0.35;

        if (activeLayer === "class") {
          const isKerb   = (flags & FLAG_KERB) !== 0;
          const isLowClr = (flags & FLAG_LOW_CLEAR) !== 0;

          if (isLowClr) {
            [r, g, b, a] = [150, 40, 200, 0.92];  // SemanticKITTI motorcyclist violet
          } else if (isKerb) {
            [r, g, b, a] = [250, 170, 30, 0.95];  // SemanticKITTI traffic-sign amber
          } else {
            const clsId = ring.cls[i] || 0;
            const palette = analysisMode === "terrain" ? TERRAIN_COLORS : OBJECT_COLORS;
            [r, g, b, a] = palette[clsId] ?? palette[0];
          }
        } else if (activeLayer === "height") {
          const gz = ring.ground_z[i];
          const tz = ring.top_z[i];
          const z = tz !== -32768 ? tz : gz;
          // -2.5m → 0 → +4m mapped to jet 0→1
          const t = Math.max(0, Math.min(1, (z / 1000 + 2.5) / 6.5));
          [r, g, b] = jetColor(t);
          a = gz !== -32768 ? 0.95 : 0.5;
        } else if (activeLayer === "traversability") {
          const isTrav = (flags & FLAG_TRAVERSABLE) !== 0;
          const hasGnd = (flags & FLAG_HAS_GROUND) !== 0;
          if (!hasGnd)            { [r, g, b, a] = [180, 180, 185, 0.3]; }
          else if (isTrav)        { [r, g, b, a] = [46, 160, 80,  0.90]; }
          else                    { [r, g, b, a] = [200, 50, 50,  0.90]; }
        } else if (activeLayer === "moving") {
          const frac = (ring.moving_frac[i] || 0) / 255.0;
          if (frac > 0.1) { [r, g, b, a] = [200, 30, 30, 0.5 + frac * 0.5]; }
          else             { [r, g, b, a] = [60, 120, 200, 0.65]; }
        } else {
          const conf = (ring.conf[i] || 0) / 255.0;
          [r, g, b, a] = [30, 130, 80, 0.2 + conf * 0.8];
        }

        ctx.fillStyle = `rgba(${r},${g},${b},${a})`;
        ctx.fillRect(screenX + gapOff, screenY + gapOff, drawPx, drawPx);
      }

      if (visibility.rings) {
        const boundaryPx = rMax * scale;
        const borderColor = RING_COLORS[ring.ring_idx % RING_COLORS.length];
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = selectedRing === ring.ring_idx ? 3 : ring.ring_idx === 0 ? 2 : 1.5;
        ctx.setLineDash(ring.ring_idx === 0 ? [] : [5, 3]);
        ctx.strokeRect(centerX - boundaryPx, centerY - boundaryPx, boundaryPx * 2, boundaryPx * 2);
        ctx.setLineDash([]);
        const resCm = cellMm / 10;
        const resLabel = `R${ring.ring_idx}  ${resCm % 1 === 0 ? resCm.toFixed(0) : resCm.toFixed(1)} cm/cell  ±${rMax / 1000}m`;
        ctx.font = "bold 11px monospace";
        const lw = ctx.measureText(resLabel).width;
        const bx = centerX - boundaryPx + 5;
        const by = centerY - boundaryPx + 5;
        ctx.fillStyle = "rgba(8,12,17,0.88)";
        ctx.fillRect(bx, by, lw + 8, 19);
        ctx.fillStyle = borderColor.replace(/[\d.]+\)$/, "1)");
        ctx.fillText(resLabel, bx + 4, by + 14);
      }
    }

    // Object bounding boxes — black rectangle outlines (SemanticKITTI 3D box style)
    // Skip degenerate boxes from clustering artifacts (> 15m = merged cluster, not a single vehicle)
    const MAX_BOX_M = 15.0;
    if (visibility.objects && analysisMode === "objects" && frame.objects && frame.objects.length > 0) {
      for (const obj of frame.objects) {
        const [ox, oy] = obj.center;
        const [l, w] = obj.size;

        // Sanity-check: skip obviously degenerate clusters
        if (l > MAX_BOX_M || w > MAX_BOX_M) continue;

        const screenX = centerX - (oy * 1000) * scale;
        const screenY = centerY - (ox * 1000) * scale;
        const lPx = l * 1000 * scale;
        const wPx = w * 1000 * scale;

        ctx.save();
        ctx.translate(screenX, screenY);
        ctx.rotate(-obj.yaw);

        // Fill: red tint for moving, pale blue for static
        if (obj.moving) {
          ctx.fillStyle = "rgba(220, 20, 60, 0.15)";
        } else {
          ctx.fillStyle = "rgba(30, 60, 140, 0.10)";
        }
        ctx.fillRect(-wPx / 2, -lPx / 2, wPx, lPx);

        // Black rectangle outline — solid black like SemanticKITTI 3D boxes
        ctx.strokeStyle = selectedObject === obj.id ? "#23c7d9" : "#05080b";
        ctx.lineWidth = selectedObject === obj.id ? 3 : 2;
        ctx.strokeRect(-wPx / 2, -lPx / 2, wPx, lPx);

        // Forward direction tick (white over black)
        ctx.strokeStyle = obj.moving ? "rgba(220,20,60,0.9)" : "rgba(30,60,140,0.8)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, -lPx / 2);
        ctx.lineTo(0, -lPx / 2 - 8);
        ctx.stroke();
        ctx.restore();

        // Label pill
        const labelText = obj.moving ? `▶ ${obj.cls_name}` : obj.cls_name;
        ctx.font = "bold 11px monospace";
        ctx.textAlign = "center";
        const tw = ctx.measureText(labelText).width;
        const lx = screenX - tw / 2 - 4;
        const ly = screenY - lPx / 2 - 19;
        // Pill background: black (matches box outline)
        ctx.fillStyle = obj.moving ? "rgba(220,20,60,0.90)" : "rgba(0,0,0,0.80)";
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(lx, ly, tw + 8, 17, 3);
        } else {
          ctx.rect(lx, ly, tw + 8, 17);
        }
        ctx.fill();
        ctx.fillStyle = "#ffffff";
        ctx.fillText(labelText, screenX, ly + 12);
        ctx.textAlign = "left";
      }
    }

    // Ego-vehicle (blue box like SemanticKITTI)
    if (visibility.vehicle) {
      const vehW = Math.max(6, 2000 * scale);
      const vehL = Math.max(9, 4000 * scale);
      ctx.fillStyle = "#1a5276";
      ctx.fillRect(centerX - vehW / 2, centerY - vehL / 2, vehW, vehL);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(centerX - vehW / 2, centerY - vehL / 2, vehW, vehL);
    }

  }, [frame, activeLayer, analysisMode, visibility, zoom, offset, selectedRing, selectedObject]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;
    const resize = () => {
      const rect = container.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width));
      const height = Math.max(360, Math.round(rect.height));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        setOffset((current) => ({ ...current }));
      }
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    return () => observer.disconnect();
  }, []);

  // Interactions
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) setOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const w = canvas.width, h = canvas.height;
    const cx = w / 2 + offset.x, cy = h / 2 + offset.y;
    const maxExtentMm = frame?.rings?.length ? Math.max(...frame.rings.map((r) => r.r_max_mm)) : 100000;
    const scale = (Math.min(w, h) * 0.44 / maxExtentMm) * zoom;
    const yMm = -(mx - cx) / scale;
    const xMm = -(my - cy) / scale;
    const distM = Math.sqrt(xMm * xMm + yMm * yMm) / 1000;

    let ringLabel = "Out of bounds";
    if (frame?.rings) {
      const asc = [...frame.rings].sort((a, b) => a.ring_idx - b.ring_idx);
      for (const r of asc) {
        if (Math.abs(xMm) <= r.r_max_mm && Math.abs(yMm) <= r.r_max_mm) {
          ringLabel = `Ring ${r.ring_idx}  @ ${r.cell_mm / 10} cm/cell`;
          break;
        }
      }
    }
    setHoverInfo(`X ${(xMm/1000).toFixed(1)}m  Y ${(yMm/1000).toFixed(1)}m  |  Dist ${distM.toFixed(1)}m  |  ${ringLabel}`);
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleMapClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (Math.abs(e.clientX - dragStart.x - offset.x) > 4 || Math.abs(e.clientY - dragStart.y - offset.y) > 4) return;
    const canvas = canvasRef.current;
    if (!canvas || !frame?.rings.length) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * canvas.width / rect.width;
    const my = (e.clientY - rect.top) * canvas.height / rect.height;
    const cx = canvas.width / 2 + offset.x, cy = canvas.height / 2 + offset.y;
    const maxExtentMm = Math.max(...frame.rings.map((ring) => ring.r_max_mm));
    const scale = (Math.min(canvas.width, canvas.height) * 0.44 / maxExtentMm) * zoom;
    const xM = -(my - cy) / scale / 1000;
    const yM = -(mx - cx) / scale / 1000;

    if (analysisMode === "objects" && visibility.objects) {
      const nearest = frame.objects
        .map((object) => ({ object, distance: Math.hypot(object.center[0] - xM, object.center[1] - yM) }))
        .filter(({ object }) => object.size[0] <= 15 && object.size[1] <= 15)
        .sort((a, b) => a.distance - b.distance)[0];
      if (nearest && nearest.distance <= Math.max(nearest.object.size[0], nearest.object.size[1], 2)) {
        onObjectSelect(nearest.object.id === selectedObject ? null : nearest.object.id);
        return;
      }
    }
    const distanceMm = Math.max(Math.abs(xM), Math.abs(yM)) * 1000;
    const ring = [...frame.rings].sort((a, b) => a.r_max_mm - b.r_max_mm).find((item) => distanceMm <= item.r_max_mm);
    onRingSelect(ring?.ring_idx === selectedRing ? null : ring?.ring_idx ?? null);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setZoom((z) => Math.max(0.3, Math.min(8.0, z * (e.deltaY < 0 ? 1.15 : 0.87))));
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  const MAX_BOX_DISPLAY_M = 15.0;
  const validObjects = frame?.objects?.filter((o) => {
    const [l, w] = o.size ?? [0, 0];
    return l <= MAX_BOX_DISPLAY_M && w <= MAX_BOX_DISPLAY_M;
  }) ?? [];
  const filteredCount = (frame?.objects?.length ?? 0) - validObjects.length;
  const movingCount = validObjects.filter((o) => o.moving).length;
  const staticCount = validObjects.length - movingCount;

  return (
    <div className="map-view-wrapper">
      <div className="map-toolbar">
        <div className="analysis-controls">
          <span className="viewer-layer-label">MAP MODE</span>
          <div className="overlay-selector" aria-label="Supporting overlay">
            {(["class", "height", "confidence", "traversability"] as ActiveLayer[]).map((lyr) => (
              <button
                key={lyr}
                className={`overlay-btn ${activeLayer === lyr ? "overlay-active" : ""}`}
                onClick={() => onLayerChange(lyr)}
              >
                {lyr === "class" ? "Semantic" : lyr === "height" ? "Elevation"
                 : lyr === "traversability" ? "Traversability" : "Confidence"}
              </button>
            ))}
          </div>
        </div>
        <div className="zoom-controls">
          <button className="ctrl-btn" onClick={() => setZoom((z) => Math.min(8.0, z * 1.25))}>+</button>
          <button className="ctrl-btn" onClick={() => setZoom((z) => Math.max(0.3, z / 1.25))}>−</button>
          <button className="ctrl-btn reset-btn" onClick={() => { setZoom(2.0); setOffset({ x: 0, y: 0 }); }}>Reset</button>
          <button className={`ctrl-btn reset-btn ${layersOpen ? "control-active" : ""}`} onClick={() => setLayersOpen((value) => !value)}>Layers</button>
        </div>
      </div>

      {layersOpen && <div className="layers-popover">
        <strong>VISIBLE OVERLAYS</strong>
        {(["rings", "objects", "vehicle"] as const).map((item) => <label key={item}>
          <input type="checkbox" checked={visibility[item]} onChange={() => setVisibility((current) => ({ ...current, [item]: !current[item] }))} />
          <span>{item === "rings" ? "Resolution rings" : item[0].toUpperCase() + item.slice(1)}</span>
        </label>)}
      </div>}

      <div className="canvas-container" ref={containerRef}>
        <canvas
          ref={canvasRef}
          style={{ cursor: isDragging ? "grabbing" : "crosshair" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleMapClick}
          onMouseLeave={handleMouseUp}
        />
        {hoverInfo && <div className="map-cursor-readout map-cursor-light">{hoverInfo}</div>}
      </div>

      <div className="legend-bar legend-light">
        {activeLayer === "class" && <>
          {analysisMode === "terrain" ? <>
            <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(34,177,76)" }}></span>Drivable</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(210,142,35)" }}></span>Non-drivable</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(194,57,52)" }}></span>Obstruction</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(120,126,136)" }}></span>Unknown</span>
          </> : <>
            <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(230,112,38)" }}></span>Static object</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(16,164,203)" }}></span>Dynamic object</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(140,145,154)" }}></span>Context / unknown</span>
          </>}
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(250,170,30)" }}></span>Kerb</span>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(150,40,200)" }}></span>Low Clear</span>
          {analysisMode === "objects" && <span className="legend-item object-box-key">□ object box</span>}
        </>}
        {activeLayer === "height" && <>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(0,0,200)" }}></span>-2.5m</span>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(0,220,220)" }}></span>0m</span>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(0,220,0)" }}></span>+2m</span>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(255,220,0)" }}></span>+3m</span>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(220,0,0)" }}></span>+4m</span>
        </>}
        {activeLayer === "traversability" && <>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(46,160,80)" }}></span>Traversable</span>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(200,50,50)" }}></span>Blocked</span>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(180,180,185)" }}></span>Unknown</span>
        </>}
        {activeLayer === "moving" && <>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(60,120,200)" }}></span>Static</span>
          <span className="legend-item"><span className="legend-dot" style={{ background: "rgb(200,30,30)" }}></span>Moving</span>
        </>}

        <span className="legend-sep"></span>
        {analysisMode === "objects" && frame?.objects && frame.objects.length > 0 && (
          <span className="legend-item obj-stat">
            <span className="legend-dot" style={{ background: "rgba(220,20,60,0.9)" }}></span>{movingCount} moving
            &nbsp;·&nbsp;
            <span className="legend-dot" style={{ background: "rgba(0,0,0,0.7)" }}></span>{staticCount} static
            {filteredCount > 0 && (
              <span style={{ color: "#999", marginLeft: 6, fontSize: "0.7rem" }}>
                (+{filteredCount} oversized skipped)
              </span>
            )}
          </span>
        )}
      </div>
    </div>
  );
};
