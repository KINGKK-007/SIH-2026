import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IsometricMap } from "./IsometricMap";
import type { ActiveLayer, AnalysisMode, FrameUpdatePayload } from "../types";

interface MapViewProps {
  frame: FrameUpdatePayload | null;
  activeLayer: ActiveLayer;
  analysisMode: AnalysisMode;
  selectedRing: number | null;
  onRingSelect: (ring: number | null) => void;
  selectedObject: number | null;
  onObjectSelect: (id: number | null) => void;
  onProbe?: (xM: number, yM: number) => void;
  probeMode?: boolean;
  probePoint?: { x: number; y: number } | null;
}

// Bit flags matching Python grid/layers.py
const FLAG_HAS_GROUND   = 0x01;
const FLAG_KERB         = 0x10;
const FLAG_LOW_CLEAR    = 0x40;
const FLAG_TRAVERSABLE  = 0x08;

const TERRAIN_COLORS: Record<number, [number, number, number, number]> = {
  0: [100, 113, 128, 0.65],
  1: [8, 145, 160, 0.94],
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

// Per-ring boundaries remain legible over the dark map.
const RING_COLORS = [
  "rgba(245, 189, 91, 0.95)",
  "rgba(116, 205, 216, 0.90)",
  "rgba(159, 167, 229, 0.90)",
  "rgba(174, 153, 186, 0.88)",
];

interface HoverInfo {
  xM: number;
  yM: number;
  zM: number | null;
  ring: number | null;
  occupied: boolean | null;
}

function jetColor(t: number): [number, number, number] {
  // Jet colormap: blue → cyan → green → yellow → red
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * t - 3)))));
  const g = Math.round(Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * t - 2)))));
  const b = Math.round(Math.max(0, Math.min(255, 255 * (1.5 - Math.abs(4 * t - 1)))));
  return [r, g, b];
}

function cellColor(ring: FrameUpdatePayload["rings"][number], i: number, layer: ActiveLayer, mode: AnalysisMode): [number, number, number, number] | null {
  const flags = ring.flags[i] || 0;
  if (layer === "class") {
    if (flags & FLAG_LOW_CLEAR) return [177, 99, 239, 0.98];
    if (flags & FLAG_KERB) return [255, 185, 57, 0.98];
    return (mode === "terrain" ? TERRAIN_COLORS : OBJECT_COLORS)[ring.cls[i] || 0] ?? TERRAIN_COLORS[0];
  }
  if (layer === "height") {
    const ground = ring.ground_z[i], top = ring.top_z[i];
    const z = top !== -32768 ? top : ground;
    if (z === -32768) return null;
    return [...jetColor((z / 1000 + 2.5) / 6.5), ground !== -32768 ? 0.96 : 0.55];
  }
  if (layer === "traversability") {
    if (flags & FLAG_LOW_CLEAR) return [177, 99, 239, 0.96];
    if (flags & FLAG_KERB) return [255, 185, 57, 0.96];
    if (!(flags & FLAG_HAS_GROUND)) return [109, 122, 143, 0.55];
    return flags & FLAG_TRAVERSABLE ? [55, 196, 121, 0.94] : [231, 70, 79, 0.94];
  }
  if (layer === "moving") {
    const moving = (ring.moving_frac[i] || 0) / 255;
    return moving > 0.1 ? [238, 74, 84, 0.55 + moving * 0.45] : [69, 144, 221, 0.7];
  }
  return [45, 201, 154, 0.2 + (ring.conf[i] || 0) / 255 * 0.8];
}

function buildTextures(frame: FrameUpdatePayload | null, layer: ActiveLayer, mode: AnalysisMode): Map<number, HTMLCanvasElement> {
  const textures = new Map<number, HTMLCanvasElement>();
  if (!frame) return textures;
  for (const ring of frame.rings) {
    const side = ring.side;
    const canvas = document.createElement("canvas");
    canvas.width = side;
    canvas.height = side;
    const ctx = canvas.getContext("2d");
    if (!ctx) continue;
    const image = ctx.createImageData(side, side);
    for (let i = 0; i < ring.ix.length; i++) {
      const color = cellColor(ring, i, layer, mode);
      if (!color) continue;
      const pixel = ((side - 1 - ring.ix[i]) * side + side - 1 - ring.iy[i]) * 4;
      image.data[pixel] = color[0];
      image.data[pixel + 1] = color[1];
      image.data[pixel + 2] = color[2];
      image.data[pixel + 3] = Math.round(color[3] * 255);
    }
    ctx.putImageData(image, 0, 0);
    textures.set(ring.ring_idx, canvas);
  }
  return textures;
}

export const MapView: React.FC<MapViewProps> = ({
  frame,
  activeLayer,
  analysisMode,
  selectedRing,
  onRingSelect,
  selectedObject,
  onObjectSelect,
  onProbe,
  probeMode = false,
  probePoint = null,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState<number>(2.0);
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const pressRef = useRef({ x: 0, y: 0, moved: false });
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [layersOpen, setLayersOpen] = useState(false);
  const [visibility, setVisibility] = useState({ rings: true, objects: true, vehicle: true });
  const [perspective, setPerspective] = useState<"top" | "iso">("top");
  const textures = useMemo(() => buildTextures(frame, activeLayer, analysisMode), [frame, activeLayer, analysisMode]);
  const [size, setSize] = useState({ width: 640, height: 520 });
  const hoverIndex = useMemo(() => new Map((frame?.rings ?? []).map((ring) => [ring.ring_idx, new Map(ring.ix.map((ix, i) => [ix * ring.side + ring.iy[i], i]))])), [frame]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || perspective === "iso") return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2 + offset.x;
    const centerY = height / 2 + offset.y;

    ctx.fillStyle = "#0c1017";
    ctx.fillRect(0, 0, width, height);

    const maxExtentMm = frame?.rings?.length
      ? Math.max(...frame.rings.map((r) => r.r_max_mm))
      : 100000;

    const baseScale = (Math.min(width, height) * 0.44) / maxExtentMm;
    const scale = baseScale * zoom;

    // Distance rings every 20m.
    for (let r = 20000; r <= maxExtentMm; r += 20000) {
      const px = r * scale;
      ctx.strokeStyle = "rgba(99,166,191,0.17)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, px, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = "#7895a4";
      ctx.font = "10px monospace";
      ctx.fillText(`${r / 1000}m`, centerX + 4, centerY - px + 12);
    }

    // Axis lines
    ctx.strokeStyle = "rgba(99,166,191,0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, 0); ctx.lineTo(centerX, height);
    ctx.moveTo(0, centerY); ctx.lineTo(width, centerY);
    ctx.stroke();

    ctx.fillStyle = "#9db2bc";
    ctx.font = "11px monospace";
    ctx.fillText("↑ FWD", centerX + 5, Math.max(15, centerY - maxExtentMm * scale - 4));

    if (!frame || !frame.rings || frame.rings.length === 0) {
      ctx.fillStyle = "#9db2bc";
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
      const rMax = ring.r_max_mm;
      const texture = textures.get(ring.ring_idx);
      if (texture) {
        ctx.imageSmoothingEnabled = false;
        const extentPx = rMax * scale;
        ctx.drawImage(texture, centerX - extentPx, centerY - extentPx, extentPx * 2, extentPx * 2);
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
        ctx.fillStyle = "rgba(12,16,23,0.94)";
        ctx.fillRect(bx, by, lw + 8, 19);
        ctx.fillStyle = borderColor;
        ctx.fillText(resLabel, bx + 4, by + 14);
      }
    }

    // Boxes use measured motion fields only; proximity is not a collision prediction.
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

        const rangeM = Math.hypot(ox, oy);
        const nearby = rangeM < 8 && (obj.moving || obj.safety_critical);

        if (obj.moving) {
          ctx.fillStyle = "rgba(231, 70, 79, 0.18)";
        } else {
          ctx.fillStyle = "rgba(66, 170, 216, 0.12)";
        }
        ctx.fillRect(-wPx / 2, -lPx / 2, wPx, lPx);

        ctx.strokeStyle = selectedObject === obj.id ? "#ffce7c" : nearby ? "#ff696d" : obj.moving ? "#f3a35f" : "#60c8e7";
        ctx.lineWidth = selectedObject === obj.id ? 3 : 2;
        ctx.strokeRect(-wPx / 2, -lPx / 2, wPx, lPx);

        // Heading from the tracked box yaw; speed is shown only when supplied.
        ctx.strokeStyle = obj.moving ? "#ff696d" : "#60c8e7";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, -lPx / 2);
        ctx.lineTo(0, -lPx / 2 - 12);
        ctx.moveTo(-4, -lPx / 2 - 8);
        ctx.lineTo(0, -lPx / 2 - 12);
        ctx.lineTo(4, -lPx / 2 - 8);
        ctx.stroke();
        ctx.restore();

        // Label pill
        const speedText = obj.speed_mps == null ? "" : ` · ${(obj.speed_mps * 3.6).toFixed(0)} km/h`;
        const labelText = `${obj.cls_name}${speedText}${nearby ? " · NEARBY" : ""}`;
        ctx.font = "bold 11px monospace";
        ctx.textAlign = "center";
        const tw = ctx.measureText(labelText).width;
        const lx = screenX - tw / 2 - 4;
        const ly = screenY - lPx / 2 - 19;
        ctx.fillStyle = nearby ? "#96394a" : obj.moving ? "#704936" : "#16465e";
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

    if (probePoint) {
      const px = centerX - probePoint.y * 1000 * scale;
      const py = centerY - probePoint.x * 1000 * scale;
      ctx.strokeStyle = "#ffcc73";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, 10, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px - 16, py); ctx.lineTo(px + 16, py); ctx.moveTo(px, py - 16); ctx.lineTo(px, py + 16); ctx.stroke();
    }

  }, [frame, activeLayer, analysisMode, visibility, zoom, offset, selectedRing, selectedObject, textures, perspective, probePoint]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container) return;
    const resize = () => {
      const rect = container.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width));
      const height = Math.max(360, Math.round(rect.height));
      if (canvas && (canvas.width !== width || canvas.height !== height)) {
        canvas.width = width;
        canvas.height = height;
        setOffset((current) => ({ ...current }));
      }
      setSize((current) => current.width === width && current.height === height ? current : { width, height });
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    return () => observer.disconnect();
  }, [perspective]);

  // Interactions
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    pressRef.current = { x: e.clientX, y: e.clientY, moved: false };
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      if (Math.hypot(e.clientX - pressRef.current.x, e.clientY - pressRef.current.y) > 4) pressRef.current.moved = true;
      setOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }

    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * canvas.width / rect.width;
    const my = (e.clientY - rect.top) * canvas.height / rect.height;
    const w = canvas.width, h = canvas.height;
    const cx = w / 2 + offset.x, cy = h / 2 + offset.y;
    const maxExtentMm = frame?.rings?.length ? Math.max(...frame.rings.map((r) => r.r_max_mm)) : 100000;
    const scale = (Math.min(w, h) * 0.44 / maxExtentMm) * zoom;
    const u = (mx - cx) / scale, v = (my - cy) / scale;
    const xMm = perspective === "top" ? -v : ((u / .7) + (-v / .35)) / 2;
    const yMm = perspective === "top" ? -u : ((-v / .35) - (u / .7)) / 2;

    let ringIndex: number | null = null;
    let zM: number | null = null;
    let occupied: boolean | null = null;
    if (frame?.rings) {
      const asc = [...frame.rings].sort((a, b) => a.ring_idx - b.ring_idx);
      for (const r of asc) {
        if (Math.abs(xMm) < r.r_max_mm && Math.abs(yMm) < r.r_max_mm) {
          ringIndex = r.ring_idx;
          const ix = Math.floor((xMm + r.r_max_mm) / r.cell_mm);
          const iy = Math.floor((yMm + r.r_max_mm) / r.cell_mm);
          const cell = hoverIndex.get(r.ring_idx)?.get(ix * r.side + iy) ?? -1;
          occupied = cell >= 0;
          if (cell >= 0) {
            const z = r.ground_z[cell] !== -32768 ? r.ground_z[cell] : r.top_z[cell];
            zM = z !== -32768 ? z / 1000 : null;
          }
          break;
        }
      }
    }
    setHoverInfo({ xM: xMm / 1000, yM: yMm / 1000, zM, ring: ringIndex, occupied });
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleMapClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (pressRef.current.moved) return;
    const canvas = e.currentTarget;
    if (!frame?.rings.length) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * canvas.width / rect.width;
    const my = (e.clientY - rect.top) * canvas.height / rect.height;
    const cx = canvas.width / 2 + offset.x, cy = canvas.height / 2 + offset.y;
    const maxExtentMm = Math.max(...frame.rings.map((ring) => ring.r_max_mm));
    const scale = (Math.min(canvas.width, canvas.height) * 0.44 / maxExtentMm) * zoom;
    const u = (mx - cx) / scale, v = (my - cy) / scale;
    const xM = (perspective === "top" ? -v : ((u / .7) + (-v / .35)) / 2) / 1000;
    const yM = (perspective === "top" ? -u : ((-v / .35) - (u / .7)) / 2) / 1000;

    if (probeMode && onProbe) {
      onProbe(Number(xM.toFixed(2)), Number(yM.toFixed(2)));
      return;
    }

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

  const zoomByWheel = useCallback((deltaY: number) => {
    setZoom((z) => Math.max(0.3, Math.min(8.0, z * (deltaY < 0 ? 1.15 : 0.87))));
  }, []);
  const fallbackToTop = useCallback(() => setPerspective("top"), []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      zoomByWheel(e.deltaY);
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [perspective, zoomByWheel]);

  return (
    <div className="map-view-wrapper">
      {layersOpen && <div className="layers-popover">
        <strong>VISIBLE OVERLAYS</strong>
        {(["rings", "objects", "vehicle"] as const).map((item) => <label key={item}>
          <input type="checkbox" checked={visibility[item]} onChange={() => setVisibility((current) => ({ ...current, [item]: !current[item] }))} />
          <span>{item === "rings" ? "Resolution rings" : item[0].toUpperCase() + item.slice(1)}</span>
        </label>)}
      </div>}

      <div className="canvas-container" ref={containerRef}>
        {perspective === "top" ? <canvas
          ref={canvasRef}
          style={{ cursor: isDragging ? "grabbing" : probeMode ? "crosshair" : "grab" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleMapClick}
          onMouseLeave={() => { handleMouseUp(); setHoverInfo(null); }}
        /> : <IsometricMap frame={frame} layer={activeLayer} mode={analysisMode} width={size.width} height={size.height} zoom={zoom} offset={offset} showRings={visibility.rings} showObjects={visibility.objects} showVehicle={visibility.vehicle} selectedRing={selectedRing} selectedObject={selectedObject} probePoint={probePoint} onZoom={zoomByWheel} onUnavailable={fallbackToTop} colorFor={cellColor} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={() => { handleMouseUp(); setHoverInfo(null); }} onClick={handleMapClick} />}
      </div>

      <div className="map-footer-controls">
        <span className="map-help">{perspective === "top" ? "Top view" : "2.5D elevation relief"} · drag to pan · scroll to zoom</span>
        <div className="zoom-controls">
          <button className={`ctrl-btn reset-btn ${perspective === "top" ? "control-active" : ""}`} onClick={() => setPerspective("top")}>Top-down 2D</button>
          <button className={`ctrl-btn reset-btn ${perspective === "iso" ? "control-active" : ""}`} onClick={() => setPerspective("iso")}>Isometric 2.5D</button>
          <button className="ctrl-btn" onClick={() => setZoom((z) => Math.min(8.0, z * 1.25))}>+</button>
          <button className="ctrl-btn" onClick={() => setZoom((z) => Math.max(0.3, z / 1.25))}>−</button>
          <button className="ctrl-btn reset-btn" onClick={() => { setZoom(2.0); setOffset({ x: 0, y: 0 }); }}>Reset</button>
          <button className={`ctrl-btn reset-btn ${layersOpen ? "control-active" : ""}`} onClick={() => setLayersOpen((value) => !value)}>Layers</button>
        </div>
      </div>
      <div className="frame-status-bar" aria-live="off">
        <span><b>FRAME</b>{frame ? String(frame.frame_idx).padStart(6, "0") : "—"}</span>
        <span><b>X</b>{hoverInfo ? `${hoverInfo.xM.toFixed(2)} m` : "—"}</span>
        <span><b>Y</b>{hoverInfo ? `${hoverInfo.yM.toFixed(2)} m` : "—"}</span>
        <span><b>Z</b>{hoverInfo?.zM != null ? `${hoverInfo.zM.toFixed(2)} m` : "—"}</span>
        <span><b>CELL</b>{hoverInfo?.ring != null ? `R${hoverInfo.ring}` : "—"}</span>
        <span>{hoverInfo?.occupied == null ? "Hover map to inspect" : hoverInfo.occupied ? "Occupied" : "No observation"}</span>
      </div>
    </div>
  );
};
