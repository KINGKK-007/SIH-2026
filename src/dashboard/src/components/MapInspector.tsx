import { ChevronDown } from "lucide-react";
import type { ActiveLayer, AnalysisMode, FrameUpdatePayload } from "../types";

interface MapInspectorProps {
  frame: FrameUpdatePayload | null;
  layer: ActiveLayer;
  mode: AnalysisMode;
  selectedRing: number | null;
  selectedObject: number | null;
  terrain: Array<{ label: string; pct: number; color: string }>;
  onLayerChange: (layer: ActiveLayer) => void;
  onRingSelect: (ring: number | null) => void;
  onObjectSelect: (id: number | null) => void;
}

const formatCm = (mm: number) => `${Number((mm / 10).toFixed(1))} cm`;
const zoneNames = ["Near field", "Intermediate", "Reduced detail", "Far field"];

export function MapInspector({ frame, layer, mode, selectedRing, selectedObject, terrain, onLayerChange, onRingSelect, onObjectSelect }: MapInspectorProps) {
  const rings = [...(frame?.rings ?? [])].sort((a, b) => a.ring_idx - b.ring_idx);
  const objects = (frame?.objects ?? []).filter((object) => object.size[0] <= 15 && object.size[1] <= 15);
  const selected = objects.find((object) => object.id === selectedObject);

  return <aside className="map-inspector" aria-label="Map inspector">
    <div className="inspector-title"><span>INSPECTOR</span><strong>Map context</strong></div>
    <section className="inspector-section">
      <label className="inspector-label" htmlFor="map-mode">Map layer</label>
      <div className="inspector-select-wrap"><select id="map-mode" value={layer} onChange={(event) => onLayerChange(event.target.value as ActiveLayer)}>
        <option value="class">Semantic classes</option>
        <option value="height">Elevation</option>
        <option value="traversability">Traversability</option>
        <option value="moving">Movement</option>
        <option value="confidence">Confidence</option>
      </select><ChevronDown size={15} /></div>
      {layer === "height" && <div className="height-scale"><div className="height-scale-track" /><span>−2.5 m</span><span>+4.0 m</span></div>}
      <p className="inspector-hint">{layer === "height" ? "Top height, falling back to ground height in the LiDAR frame." : layer === "traversability" ? "Green: traversable. Red: blocked. Gray: no ground estimate." : layer === "confidence" ? "Source confidence. Oracle mode uses full confidence." : layer === "moving" ? "Red marks cells with moving point evidence." : "Semantic groups with kerb and low clearance overlays."}</p>
    </section>
    <section className="inspector-section">
      <div className="inspector-section-heading"><span className="inspector-label">Resolution</span><small>{rings.length ? `${rings.length} zones` : "Waiting for data"}</small></div>
      <div className="resolution-list">{rings.map((ring, index) => <button
        key={ring.ring_idx}
        type="button"
        className={`resolution-row ${selectedRing === ring.ring_idx ? "selected" : ""}`}
        onClick={() => onRingSelect(selectedRing === ring.ring_idx ? null : ring.ring_idx)}
        aria-pressed={selectedRing === ring.ring_idx}
      ><i className={`zone-marker zone-${ring.ring_idx}`} /><span className="resolution-name"><strong>R{ring.ring_idx}</strong><small>{zoneNames[index] ?? "Adaptive zone"}</small></span><span className="resolution-values"><strong>{formatCm(ring.cell_mm)}</strong><small>{ring.ix.length.toLocaleString()} cells</small></span></button>)}</div>
      {rings.length > 0 && <p className="inspector-hint">Zones use square distance from the vehicle. Select a zone to highlight its boundary.</p>}
    </section>
    <section className="inspector-section">
      <span className="inspector-label">View</span>
      <div className="inspector-key-values"><span>Reference</span><strong>Vehicle centred</strong><span>Extent</span><strong>{rings.length ? `±${rings.at(-1)!.r_max_mm / 1000} m` : "—"}</strong><span>Preset</span><strong>{frame?.preset ?? "—"}</strong></div>
    </section>
    {mode === "terrain" && <section className="inspector-section">
      <span className="inspector-label">Terrain composition</span>
      <div className="terrain-bar">{terrain.map((row) => <i key={row.label} className={row.color} style={{ width: `${row.pct}%` }} />)}</div>
      <div className="terrain-list">{terrain.map((row) => <div key={row.label}><span><i className={row.color} />{row.label}</span><strong>{frame ? `${row.pct.toFixed(1)}%` : "—"}</strong></div>)}</div>
    </section>}
    {mode === "objects" && <section className="inspector-section object-inspector-section">
      <div className="inspector-section-heading"><span className="inspector-label">Objects</span><small>{objects.length} visible</small></div>
      <div className="inspector-object-list">{objects.map((object) => <button key={object.id} className={selectedObject === object.id ? "selected" : ""} onClick={() => onObjectSelect(selectedObject === object.id ? null : object.id)}><span><i className={object.moving ? "moving" : "static"} />{object.cls_name} #{object.id}</span><small>{Math.hypot(object.center[0], object.center[1]).toFixed(1)} m{object.speed_mps == null ? "" : ` · ${(object.speed_mps * 3.6).toFixed(0)} km/h`}</small></button>)}</div>
      {selected && <div className="selected-object-detail"><strong>{selected.cls_name} #{selected.id}</strong><span>{selected.moving ? "Moving" : "Static"} · {Math.round(selected.mean_conf / 255 * 100)}% source confidence</span><span>{selected.speed_mps == null ? "Speed unavailable" : `${(selected.speed_mps * 3.6).toFixed(1)} km/h speed`} · {Math.hypot(selected.center[0], selected.center[1]).toFixed(1)} m from ego</span><span>{Math.hypot(selected.center[0], selected.center[1]) < 8 && (selected.moving || selected.safety_critical) ? "Nearby object · inspect path" : "Outside 8 m proximity cue"}</span></div>}
      {!objects.length && <p className="inspector-hint">No object boxes in this frame.</p>}
    </section>}
  </aside>;
}
