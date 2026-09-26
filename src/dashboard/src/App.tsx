import React, { useEffect, useMemo, useRef, useState } from "react";
import { Header } from "./components/Header";
import { LatencyPanel } from "./components/LatencyPanel";
import { MapView } from "./components/MapView";
import { MemoryMeter } from "./components/MemoryMeter";
import { PlaybackControls } from "./components/PlaybackControls";
import { socket } from "./socket";
import type { ActiveLayer, AnalysisMode, FrameUpdatePayload, PlaybackState } from "./types";

type View = "overview" | "live" | "semantic" | "objects" | "terrain" | "elevation" | "performance" | "benchmarks" | "settings";
const NAV: Array<{ id: View; icon: string; label: string }> = [
  { id: "overview", icon: "▦", label: "Overview" },
  { id: "live", icon: "◉", label: "Live Perception" },
  { id: "semantic", icon: "◇", label: "Semantic Map" },
  { id: "objects", icon: "▣", label: "Object Detection" },
  { id: "terrain", icon: "△", label: "Terrain Analysis" },
  { id: "elevation", icon: "≋", label: "Elevation Map" },
  { id: "performance", icon: "⌁", label: "Performance" },
  { id: "benchmarks", icon: "↔", label: "Foveated vs Uniform" },
];
const fmtBytes = (bytes = 0) => bytes ? `${(bytes / 1048576).toFixed(1)} MB` : "—";

export const App: React.FC = () => {
  const [connected, setConnected] = useState(socket.connected);
  const [frame, setFrame] = useState<FrameUpdatePayload | null>(null);
  const [state, setState] = useState<PlaybackState | null>(null);
  const [layer, setLayer] = useState<ActiveLayer>("class");
  const [mode, setMode] = useState<AnalysisMode>("terrain");
  const [view, setView] = useState<View>(window.location.pathname === "/dashboard" ? "performance" : "live");
  const [collapsed, setCollapsed] = useState(false);
  const [displayFps, setDisplayFps] = useState(0);
  const [selectedRing, setSelectedRing] = useState<number | null>(null);
  const [selectedObject, setSelectedObject] = useState<number | null>(null);
  const [history, setHistory] = useState<Array<{ latency: number; fps: number; memory: number }>>([]);
  const frameTimes = useRef<number[]>([]);

  useEffect(() => {
    const onFrame = (payload: FrameUpdatePayload) => {
      setFrame(payload);
      const frameLatency = Object.values(payload.timings_ms).reduce<number>((sum, value) => sum + (value ?? 0), 0);
      setHistory((current) => [...current, {
        latency: frameLatency,
        fps: frameLatency ? 1000 / frameLatency : 0,
        memory: payload.memory.fovea_bytes / 1048576,
      }].slice(-60));
      const now = performance.now();
      frameTimes.current.push(now);
      frameTimes.current = frameTimes.current.filter((time) => time >= now - 1000);
      setDisplayFps(frameTimes.current.length);
    };
    const onConnect = () => setConnected(true);
    const onDisconnect = () => setConnected(false);
    socket.on("connect", onConnect);
    socket.on("disconnect", onDisconnect);
    socket.on("state_update", setState);
    socket.on("frame_update", onFrame);
    return () => {
      socket.off("connect", onConnect);
      socket.off("disconnect", onDisconnect);
      socket.off("state_update", setState);
      socket.off("frame_update", onFrame);
    };
  }, []);

  const latency = frame ? Object.values(frame.timings_ms).reduce<number>((sum, value) => sum + (value ?? 0), 0) : 0;
  const fps = latency ? 1000 / latency : 0;
  const activeCells = frame?.rings.reduce((sum, ring) => sum + ring.ix.length, 0) ?? 0;
  const reduction = frame?.memory.uniform25d_bytes && frame.memory.fovea_bytes
    ? 100 * (1 - frame.memory.fovea_bytes / frame.memory.uniform25d_bytes) : 0;
  const terrain = useMemo(() => {
    const counts = [0, 0, 0, 0, 0];
    frame?.rings.forEach((ring) => ring.cls.forEach((cls, i) => { counts[cls] += ring.count[i] ?? 1; }));
    const total = counts.reduce((a, b) => a + b, 0) || 1;
    return [
      { label: "Drivable", count: counts[1], color: "cyan" },
      { label: "Non-drivable", count: counts[2], color: "amber" },
      { label: "Obstacle", count: counts[3] + counts[4], color: "red" },
      { label: "Unknown", count: counts[0], color: "gray" },
    ].map((row) => ({ ...row, pct: row.count / total * 100 }));
  }, [frame]);

  const selectView = (next: View) => {
    setView(next);
    if (next === "objects") { setMode("objects"); setLayer("class"); }
    if (next === "terrain") { setMode("terrain"); setLayer("traversability"); }
    if (next === "elevation") { setMode("terrain"); setLayer("height"); }
    if (next === "semantic" || next === "live" || next === "overview") { setMode("terrain"); setLayer("class"); }
  };
  const title = NAV.find((item) => item.id === view)?.label ?? "Live Perception";
  const taskMapPage = ["live", "objects", "terrain", "elevation"].includes(view);
  const objects = frame?.objects.filter((object) => object.size[0] <= 15 && object.size[1] <= 15) ?? [];
  const ringSummary = frame?.rings.length
    ? `${frame.rings.map((ring) => `${ring.cell_mm / 10} cm`).join(" / ")} · ${(frame.rings.at(-1)?.r_max_mm ?? 0) / 1000} m range`
    : "live grid geometry pending";

  return <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
    <aside className="sidebar">
      <button className="brand" onClick={() => setCollapsed((value) => !value)} aria-label="Toggle sidebar">
        <span className="brand-mark"><i /><i /><i /></span><strong>FoveaMap</strong>
      </button>
      <nav>{NAV.map((item) => <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => selectView(item.id)}><span>{item.icon}</span><b>{item.label}</b></button>)}</nav>
      <div className="sidebar-bottom">
        <button><span>◫</span><b>Sequence 08</b></button>
        <button onClick={() => setView("settings")}><span>⚙</span><b>Settings</b></button>
        <div className="model-foot"><small>MODEL PATH</small><strong>{frame?.model ?? state?.model ?? "oracle"}</strong></div>
      </div>
    </aside>
    <section className="workspace">
      <Header connected={connected} frame={frame} state={state} title={title} />
      <main className="content">
        {view === "overview" && <section className="overview-page">
          <section className="metric-grid">
            <Metric label="Pipeline rate (derived)" value={fps ? fps.toFixed(1) : "—"} unit="FPS" note={`1 ÷ stage total · Display ${displayFps || "—"} FPS`} />
            <Metric label="Pipeline stage total" value={latency ? latency.toFixed(1) : "—"} unit="ms" note="Sum of measured frame stages" />
            <Metric label="Map memory" value={fmtBytes(frame?.memory.fovea_bytes)} note={reduction ? `${reduction.toFixed(1)}% below uniform 5 cm` : "Awaiting measured baseline"} tone="green" />
            <Metric label="Active cells" value={activeCells ? activeCells.toLocaleString() : "—"} note={`${frame?.counters.n_in_grid.toLocaleString() ?? "—"} points in grid`} />
          </section>
          <section className="overview-grid">
            <article className="overview-hero">
              <small>SYSTEM OVERVIEW</small><h2>3D LiDAR → semantic inference → adaptive 2.5D map</h2>
              <p>FoveaMap preserves the finest available detail near the vehicle, then reduces grid density with distance. Current geometry: {ringSummary}.</p>
              <div className="pipeline-flow"><span>RAW LIDAR<br/><b>{frame?.counters.n_raw.toLocaleString() ?? "—"} points</b></span><i>→</i><span>SEMANTICS<br/><b>{frame?.model ?? "Waiting"}</b></span><i>→</i><span>ADAPTIVE GRID<br/><b>{activeCells.toLocaleString()} cells</b></span><i>→</i><span>PERCEPTION<br/><b>{objects.length} objects</b></span></div>
            </article>
            <RingPanel rings={frame?.rings ?? []} selectedRing={selectedRing} onSelect={setSelectedRing} />
            <TerrainPanel rows={terrain} />
            <ObjectPanel objects={objects} selectedObject={selectedObject} onSelect={setSelectedObject} />
          </section>
          <p className="overview-action">Use <b>Live Perception</b> for playback, <b>Semantic Map</b> for class composition, and the task pages for terrain or object analysis.</p>
        </section>}
        {taskMapPage && <>
          <section className="metric-grid">
            <Metric label="Pipeline rate (derived)" value={fps ? fps.toFixed(1) : "—"} unit="FPS" note={`1 ÷ stage total · Display ${displayFps || "—"} FPS`} />
            <Metric label="Pipeline stage total" value={latency ? latency.toFixed(1) : "—"} unit="ms" note="Sum of measured frame stages" />
            <Metric label="Map memory" value={fmtBytes(frame?.memory.fovea_bytes)} note={reduction ? `${reduction.toFixed(1)}% below uniform 5 cm` : "Awaiting measured baseline"} tone="green" />
            <Metric label="Active cells" value={activeCells ? activeCells.toLocaleString() : "—"} note={`${frame?.counters.n_in_grid.toLocaleString() ?? "—"} points in grid`} />
          </section>
          <section className="perception-layout">
            <div className="map-column">
              <div className="section-heading"><div><small>{view === "live" ? "Live perception stream" : view === "objects" ? "Object detection" : view === "terrain" ? "Terrain analysis" : "Elevation map"}</small><h2>{view === "objects" ? "Instances and observed motion" : view === "terrain" ? "Traversability and terrain composition" : view === "elevation" ? "Ground and obstacle height" : "Adaptive perception field"}</h2><p>{view === "terrain" ? "Traversability mode separates driveable ground, blocked cells, and areas without sufficient evidence." : view === "objects" ? "Object mode overlays measured clusters and motion state on semantic cells." : view === "elevation" ? "Elevation mode maps measured cell height from −2.5 m to +4 m." : `Top view · vehicle centred · ${ringSummary}`}</p></div><span>Top view · vehicle centred</span></div>
              <MapView frame={frame} activeLayer={layer} onLayerChange={setLayer} analysisMode={mode} onAnalysisModeChange={setMode} selectedRing={selectedRing} onRingSelect={setSelectedRing} selectedObject={selectedObject} onObjectSelect={setSelectedObject} />
              <PlaybackControls state={state} currentFrameIdx={frame?.frame_idx ?? state?.frame_idx ?? 0} />
            </div>
            <aside className="insight-column">
              <RingPanel rings={frame?.rings ?? []} selectedRing={selectedRing} onSelect={setSelectedRing} />
              {mode === "objects" || view === "objects" ? <ObjectPanel objects={objects} selectedObject={selectedObject} onSelect={setSelectedObject} /> : <TerrainPanel rows={terrain} />}
            </aside>
          </section>
        </>}
        {view === "semantic" && <section className="semantic-page">
          <div className="semantic-map-pane">
            <div className="section-heading"><div><small>SEMANTIC SUPERCLASSES</small><h2>Classified adaptive map</h2></div><span>POINT-WEIGHTED DISTRIBUTION</span></div>
            <MapView frame={frame} activeLayer={layer} onLayerChange={setLayer} analysisMode="terrain" onAnalysisModeChange={setMode} selectedRing={selectedRing} onRingSelect={setSelectedRing} selectedObject={selectedObject} onObjectSelect={setSelectedObject} />
          </div>
          <aside className="semantic-analysis">
            <TerrainPanel rows={terrain} />
            <section className="side-card semantic-notes"><header><span>CLASS INTERPRETATION</span><b>5 GROUPS</b></header><p><b>Drivable</b> combines road, parking and lane markings.</p><p><b>Non-drivable</b> includes sidewalk, terrain and other ground.</p><p><b>Obstacle</b> combines static structures and dynamic-capable classes.</p><p>Use Object Detection for instance boxes and observed motion.</p></section>
          </aside>
        </section>}
        {view === "performance" && <section className="page-stack">
          <section className="metric-grid">
            <Metric label="Pipeline rate (derived)" value={fps ? fps.toFixed(1) : "—"} unit="FPS" note="1 ÷ measured stage total" />
            <Metric label="Pipeline stage total" value={latency ? latency.toFixed(1) : "—"} unit="ms" note="Not inference-only" />
            <Metric label="Map memory" value={fmtBytes(frame?.memory.fovea_bytes)} note="Allocated 2.5D layers" tone="green" />
            <Metric label="Raw points" value={frame?.counters.n_raw.toLocaleString() ?? "—"} note="Current LiDAR scan" tone="blue" />
          </section>
          <section className="analytics-grid"><LatencyPanel timings={frame?.timings_ms} counters={frame?.counters} /><MemoryMeter memory={frame?.memory} /></section>
          <section className="chart-grid">
            <TelemetryChart title="Pipeline FPS over time" values={history.map((point) => point.fps)} color="#23c7d9" unit="FPS" />
            <TelemetryChart title="End-to-end latency over time" values={history.map((point) => point.latency)} color="#f4a340" unit="ms" reference={100} />
            <TelemetryChart title="Map memory over time" values={history.map((point) => point.memory)} color="#39d98a" unit="MB" />
          </section>
          <p className="evidence-note">Live values are generated by the active pipeline. RTX 4050 accuracy, VRAM and percentile reports appear after the acceptance run.</p>
        </section>}
        {view === "benchmarks" && <Benchmark memory={frame?.memory} latency={latency} fps={fps} rings={frame?.rings ?? []} />}
        {view === "settings" && <section className="empty-page"><h2>Runtime configuration</h2><p>Sequence {frame?.seq ?? state?.seq ?? "08"} · {frame?.rings.length ?? "—"} nested zones · {ringSummary} · telemetry supplied by the active backend.</p></section>}
      </main>
    </section>
  </div>;
};

const Metric = ({ label, value, unit, note }: { label: string; value: string; unit?: string; note: string; tone?: string }) =>
  <article className="metric-card"><span>{label}</span><div><strong>{value}</strong>{unit && <em>{unit}</em>}</div><small>{note}</small></article>;

const RingPanel = ({ rings, selectedRing, onSelect }: { rings: FrameUpdatePayload["rings"]; selectedRing: number | null; onSelect: (ring: number | null) => void }) => {
  const ordered = [...rings].sort((a, b) => a.ring_idx - b.ring_idx);
  const roles = ["Near-field safety", "Intermediate detail", "Reduced detail", "Far-field coverage"];
  return <section className="side-card"><header><span>Adaptive resolution</span><b>{ordered.length ? `${ordered.length} zones` : "Live geometry"}</b></header><div className="ring-list">{ordered.map((ring, position) => {
    const innerMm = position ? ordered[position - 1].r_max_mm : 0;
    return <button type="button" key={ring.ring_idx} className={`ring-row r${ring.ring_idx} ${selectedRing === ring.ring_idx ? "selected" : ""}`} onClick={() => onSelect(selectedRing === ring.ring_idx ? null : ring.ring_idx)}><i /><div><strong>R{ring.ring_idx} · {roles[position] ?? "Adaptive zone"}</strong><span>{innerMm / 1000}–{ring.r_max_mm / 1000} m</span></div><b>{ring.cell_mm / 10} cm</b><small>{ring.ix.length.toLocaleString()} active</small></button>;
  })}{!ordered.length && <p className="panel-empty">Waiting for live grid geometry</p>}</div></section>;
};

const ObjectPanel = ({ objects, selectedObject, onSelect }: { objects: FrameUpdatePayload["objects"]; selectedObject: number | null; onSelect: (id: number | null) => void }) =>
  <section className="side-card"><header><span>DETECTED OBJECTS</span><b>{objects.length} TOTAL</b></header><div className="object-list">{objects.slice(0,6).map((object) =>
    <button type="button" key={object.id} className={selectedObject === object.id ? "selected" : ""} onClick={() => onSelect(selectedObject === object.id ? null : object.id)}><i className={object.moving ? "moving" : "static"} /><div><strong>{object.cls_name} #{object.id}</strong><span>{Math.hypot(object.center[0], object.center[1]).toFixed(1)} m · {(object.mean_conf * 100).toFixed(1)}% confidence</span></div><b>{object.moving ? "MOVING" : "STATIC"}</b></button>
  )}{!objects.length && <p className="panel-empty">Waiting for detected objects</p>}</div></section>;

const TerrainPanel = ({ rows }: { rows: Array<{ label: string; pct: number; color: string }> }) =>
  <section className="side-card"><header><span>TERRAIN COMPOSITION</span><b>BY POINTS</b></header><div className="terrain-bar">{rows.map((row) => <i key={row.label} className={row.color} style={{ width: `${row.pct}%` }} />)}</div><div className="terrain-list">{rows.map((row) => <div key={row.label}><span><i className={row.color} />{row.label}</span><b>{row.pct.toFixed(1)}%</b></div>)}</div></section>;

const Benchmark = ({ memory, latency, fps, rings }: { memory?: FrameUpdatePayload["memory"]; latency: number; fps: number; rings: FrameUpdatePayload["rings"] }) => {
  const reduction = memory?.uniform25d_bytes && memory.fovea_bytes ? 100 * (1-memory.fovea_bytes/memory.uniform25d_bytes) : 0;
  const first = rings.at(0), last = rings.at(-1);
  const finest = first ? `${first.cell_mm / 10} cm` : "—", range = last ? `${last.r_max_mm / 1000} m` : "—";
  const geometry = rings.length ? rings.map((ring) => `${ring.cell_mm / 10}`).join("/") + " cm" : "Awaiting live geometry";
  return <section className="benchmark-page"><div className="reduction-callout"><strong>{reduction ? `${reduction.toFixed(1)}%` : "—"}</strong><span>Measured map-memory reduction</span><small>against the backend's uniform finest-cell 2.5D baseline</small></div><div className="compare-grid"><article><small>Uniform baseline</small><h2>{fmtBytes(memory?.uniform25d_bytes)}</h2><p>{finest} cells · {range} extent</p></article><article className="fovea-choice"><small>FoveaMap</small><h2>{fmtBytes(memory?.fovea_bytes)}</h2><p>{geometry} adaptive grid · {range} extent</p></article></div><div className="comparison-table"><div><span>Metric</span><b>Uniform {finest}</b><b>FoveaMap</b></div><div><span>Map memory</span><b>{fmtBytes(memory?.uniform25d_bytes)}</b><b>{fmtBytes(memory?.fovea_bytes)}</b></div><div><span>Pipeline stage total</span><b>Not measured</b><b>{latency ? `${latency.toFixed(1)} ms` : "—"}</b></div><div><span>Derived pipeline rate</span><b>Not measured</b><b>{fps ? `${fps.toFixed(1)} FPS` : "—"}</b></div><div><span>Near-field resolution</span><b>{finest}</b><b>{finest}</b></div><div><span>Maximum range</span><b>{range}</b><b>{range}</b></div></div><p className="evidence-note">Only values supplied by the current backend are shown. Uniform-grid latency and throughput remain unclaimed until measured on the same hardware and frames.</p></section>;
};

const TelemetryChart = ({ title, values, color, unit, reference }: { title: string; values: number[]; color: string; unit: string; reference?: number }) => {
  const width = 420, height = 130, pad = 14;
  const max = Math.max(reference ?? 0, ...values, 1);
  const points = values.map((value, index) => {
    const x = pad + (values.length <= 1 ? 0 : index / (values.length - 1)) * (width - pad * 2);
    const y = height - pad - value / max * (height - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  const latest = values.at(-1);
  return <article className="telemetry-chart"><header><span>{title}</span><b>{latest === undefined ? "—" : `${latest.toFixed(1)} ${unit}`}</b></header><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
    {[.25,.5,.75].map((part) => <line key={part} x1={pad} x2={width-pad} y1={height*part} y2={height*part} className="chart-gridline" />)}
    {reference && <line x1={pad} x2={width-pad} y1={height-pad-reference/max*(height-pad*2)} y2={height-pad-reference/max*(height-pad*2)} className="chart-reference" />}
    {points && <polyline points={points} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />}
  </svg><small>Last {values.length} live frames</small></article>;
};

export default App;
