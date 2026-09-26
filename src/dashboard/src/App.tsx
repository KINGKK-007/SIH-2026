import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AppSidebar, type View } from "./components/AppSidebar";
import { Header } from "./components/Header";
import { LatencyPanel } from "./components/LatencyPanel";
import { MapInspector } from "./components/MapInspector";
import { MapView } from "./components/MapView";
import { MemoryMeter } from "./components/MemoryMeter";
import { PlaybackControls } from "./components/PlaybackControls";
import { PerceptionLab, StepPreview, type ZoomResult } from "./components/PerceptionLab";
import { TelemetryStrip } from "./components/TelemetryStrip";
import { SERVER_URL, socket } from "./socket";
import type { ActiveLayer, AnalysisMode, FrameUpdatePayload, PlaybackState } from "./types";

const TITLES: Record<View, string> = {
  overview: "Perception workspace", live: "Live perception", semantic: "Semantic map",
  objects: "Object Detection", terrain: "Terrain Analysis", elevation: "Elevation Map",
  performance: "Performance", benchmarks: "Foveated vs Uniform", settings: "Settings",
};
const MAP_VIEWS: View[] = ["overview", "live", "semantic", "objects", "terrain", "elevation"];
const VIEWS_WITHOUT_KPIS: View[] = ["objects", "terrain", "elevation"];
const fmtBytes = (bytes = 0) => bytes ? `${(bytes / 1048576).toFixed(1)} MiB` : "—";

export function App() {
  const [connected, setConnected] = useState(socket.connected);
  const [frame, setFrame] = useState<FrameUpdatePayload | null>(null);
  const [previewFrame, setPreviewFrame] = useState<FrameUpdatePayload | null>(null);
  const [probePoint, setProbePoint] = useState<{ x: number; y: number; frameIdx: number } | null>(null);
  const [probeMode, setProbeMode] = useState(false);
  const [state, setState] = useState<PlaybackState | null>(null);
  const [layer, setLayer] = useState<ActiveLayer>("class");
  const [mode, setMode] = useState<AnalysisMode>("terrain");
  const [view, setView] = useState<View>("overview");
  const [collapsed, setCollapsed] = useState(false);
  const [displayFps, setDisplayFps] = useState(0);
  const [selectedRing, setSelectedRing] = useState<number | null>(null);
  const [selectedObject, setSelectedObject] = useState<number | null>(null);
  const [history, setHistory] = useState<Array<{ latency: number; fps: number; memory: number }>>([]);
  const frameTimes = useRef<number[]>([]);

  useEffect(() => {
    const onFrame = (payload: FrameUpdatePayload) => {
      setFrame(payload);
      setPreviewFrame(null);
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

  const selectView = (next: View) => {
    setView(next);
    window.history.replaceState(null, "", `#${next}`);
    if (next === "objects") { setMode("objects"); setLayer("class"); }
    if (next === "terrain") { setMode("terrain"); setLayer("traversability"); }
    if (next === "elevation") { setMode("terrain"); setLayer("height"); }
    if (next === "semantic" || next === "live" || next === "overview") { setMode("terrain"); setLayer("class"); }
  };

  useEffect(() => {
    const sync = () => {
      const next = window.location.hash.slice(1) as View;
      if (next in TITLES) selectView(next);
    };
    sync();
    window.addEventListener("hashchange", sync);
    const timer = window.setInterval(() => {
      frameTimes.current = frameTimes.current.filter((time) => time >= performance.now() - 1000);
      setDisplayFps(frameTimes.current.length);
    }, 1000);
    return () => { window.removeEventListener("hashchange", sync); window.clearInterval(timer); };
  }, []);

  const latency = frame ? Object.values(frame.timings_ms).reduce<number>((sum, value) => sum + (value ?? 0), 0) : 0;
  const fps = latency ? 1000 / latency : 0;
  const activeCells = frame?.rings.reduce((sum, ring) => sum + ring.ix.length, 0) ?? 0;
  const visibleFrame = previewFrame ?? frame;
  const setPreview = useCallback((next: FrameUpdatePayload | null) => setPreviewFrame(next), []);
  const terrain = useMemo(() => {
    const counts = [0, 0, 0, 0, 0];
    visibleFrame?.rings.forEach((ring) => ring.cls.forEach((cls, i) => { counts[cls] += ring.count[i] ?? 1; }));
    const total = counts.reduce((a, b) => a + b, 0) || 1;
    return [
      { label: "Drivable", count: counts[1], color: "drivable" },
      { label: "Other terrain", count: counts[2], color: "other-terrain" },
      { label: "Obstacle", count: counts[3] + counts[4], color: "obstacle" },
      { label: "Unknown", count: counts[0], color: "unknown" },
    ].map((row) => ({ ...row, pct: row.count / total * 100 }));
  }, [visibleFrame]);
  const isMapView = MAP_VIEWS.includes(view);

  return <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
    <AppSidebar view={view} collapsed={collapsed} connected={connected} sequence={frame?.seq ?? state?.seq ?? "08"} onSelect={selectView} onToggle={() => setCollapsed((value) => !value)} />
    <section className="workspace">
      <Header connected={connected} frame={frame} state={state} title={TITLES[view]} />
      <main className="content">
        {!VIEWS_WITHOUT_KPIS.includes(view) && <TelemetryStrip frame={frame} latency={latency} fps={fps} activeCells={activeCells} displayFps={displayFps} />}
        {isMapView && <section className="map-workspace" aria-label="LiDAR map workspace">
          <div className="map-workspace-heading"><div><span className="section-eyebrow">LIVE SPATIAL VIEW</span><h2>{view === "overview" ? "Adaptive 2.5D map" : TITLES[view]}</h2></div><span className="map-workspace-subtitle">{visibleFrame?.preset ?? "Awaiting grid"} · LiDAR frame{previewFrame ? " · preview" : ""}</span></div>
          <div className="perception-layout">
            <div className="map-column"><MapView frame={visibleFrame} activeLayer={layer} analysisMode={mode} selectedRing={selectedRing} onRingSelect={setSelectedRing} selectedObject={selectedObject} onObjectSelect={setSelectedObject} probeMode={probeMode} probePoint={probePoint} onProbe={(x, y) => { setProbePoint({ x, y, frameIdx: frame?.frame_idx ?? 0 }); setProbeMode(false); }} /></div>
            <MapInspector frame={visibleFrame} layer={layer} mode={mode} terrain={terrain} selectedRing={selectedRing} selectedObject={selectedObject} onLayerChange={setLayer} onRingSelect={setSelectedRing} onObjectSelect={setSelectedObject} />
          </div>
          <PerceptionLab frame={frame} point={probePoint} probeMode={probeMode} onProbeMode={setProbeMode} onPreview={setPreview} />
          <PlaybackControls state={state} currentFrameIdx={frame?.frame_idx ?? state?.frame_idx ?? 0} />
          <div className="workspace-footnote"><span>Source: {visibleFrame?.model === "oracle" ? "ground-truth replay" : visibleFrame?.model ?? "awaiting backend"}</span><span>Stage telemetry excludes transfer and browser rendering. No real-time claim is inferred from this view.</span></div>
        </section>}
        {view === "performance" && <section className="page-stack">
          <div className="page-heading"><span className="section-eyebrow">ANALYSIS</span><h2>Pipeline performance</h2><p>Current frame telemetry and the last 60 received frames.</p></div>
          <section className="analytics-grid"><LatencyPanel timings={frame?.timings_ms} counters={frame?.counters} /><MemoryMeter memory={frame?.memory} /></section>
          <section className="chart-grid">
            <TelemetryChart title="Pipeline FPS over time" values={history.map((point) => point.fps)} color="#C69A52" unit="FPS" />
            <TelemetryChart title="Pipeline stage total" values={history.map((point) => point.latency)} color="#BD8151" unit="ms" reference={100} />
            <TelemetryChart title="Map memory over time" values={history.map((point) => point.memory)} color="#748E73" unit="MiB" />
          </section>
          <p className="evidence-note">Stage totals exclude network transfer and browser rendering. Display rate counts received frames. Full real-time acceptance requires a separately measured end-to-end p95 below 100 ms.</p>
        </section>}
        {view === "benchmarks" && <Benchmark frame={frame} latency={latency} fps={fps} />}
        {view === "settings" && <section className="page-heading"><span className="section-eyebrow">WORKSPACE</span><h2>Settings</h2><p>Map layers and playback controls are available in the live map workspace.</p></section>}
      </main>
    </section>
  </div>;
}

function chooseFarWindow(frame: FrameUpdatePayload): { x: number; y: number } {
  const ring = frame.rings.at(-1);
  const inner = frame.rings.at(-2)?.r_max_mm ?? 60000;
  if (!ring) return { x: 70, y: 0 };
  let best = -1, bestCount = -1;
  for (let i = 0; i < ring.ix.length; i++) {
    const x = (-ring.r_max_mm + (ring.ix[i] + .5) * ring.cell_mm) / 1000;
    const y = (-ring.r_max_mm + (ring.iy[i] + .5) * ring.cell_mm) / 1000;
    const range = Math.max(Math.abs(x), Math.abs(y));
    if (range < inner / 1000 + 3 || range > ring.r_max_mm / 1000 - 3) continue;
    if (ring.ground_z[i] === -32768) continue;
    if (ring.count[i] > bestCount) { best = i; bestCount = ring.count[i]; }
  }
  if (best < 0) return { x: 70, y: 0 };
  return {
    x: Number(((-ring.r_max_mm + (ring.ix[best] + .5) * ring.cell_mm) / 1000).toFixed(2)),
    y: Number(((-ring.r_max_mm + (ring.iy[best] + .5) * ring.cell_mm) / 1000).toFixed(2)),
  };
}

function Benchmark({ frame, latency, fps }: { frame: FrameUpdatePayload | null; latency: number; fps: number }) {
  const [location, setLocation] = useState({ x: 5, y: 0 });
  const [target, setTarget] = useState<{ x: number; y: number; farX: number; farY: number; frameIdx: number } | null>(null);
  const [comparison, setComparison] = useState<ZoomResult | null>(null);
  const [farComparison, setFarComparison] = useState<ZoomResult | null>(null);
  const [error, setError] = useState("");
  const memory = frame?.memory;
  const rings = frame?.rings ?? [];
  const first = rings.at(0), last = rings.at(-1);
  const extentMm = last?.r_max_mm ?? 100000;
  const uniform20Bytes = Math.pow(2 * extentMm / 200, 2) * 12;
  const reduction = memory?.uniform25d_bytes && memory.fovea_bytes ? 100 * (1 - memory.fovea_bytes / memory.uniform25d_bytes) : 0;
  const finest = first ? `${first.cell_mm / 10} cm` : "—";
  const nearLimit = (first?.r_max_mm ?? 10000) / 1000 - 2.5;
  const range = last ? `±${last.r_max_mm / 1000} m` : "—";
  const geometry = rings.length ? rings.map((ring) => `${ring.cell_mm / 10}`).join(" / ") + " cm" : "Awaiting live geometry";

  useEffect(() => {
    if (!target) return;
    const abort = new AbortController();
    const request = async (x: number, y: number) => {
      const response = await fetch(`${SERVER_URL}/api/zoom`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame_idx: target.frameIdx, x_m: x, y_m: y }),
      signal: abort.signal,
      });
      if (!response.ok) throw new Error(`Window request failed (${response.status})`);
      return response.json() as Promise<ZoomResult>;
    };
    Promise.all([request(target.x, target.y), request(target.farX, target.farY)])
      .then(([near, far]) => { setComparison(near); setFarComparison(far); setError(""); })
      .catch((reason: Error) => { if (reason.name !== "AbortError") setError(reason.message); });
    return () => abort.abort();
  }, [target]);

  return <section className="benchmark-page">
    <div className="page-heading"><span className="section-eyebrow">ANALYSIS</span><h2>Foveated vs Uniform</h2><p>Full-grid allocation and a real 5 m scan window at three resolutions.</p></div>
    <div className="reduction-callout"><strong>{reduction ? `${reduction.toFixed(1)}%` : "—"}</strong><span>less allocated map memory than uniform 5 cm</span><small>Current frame · same ±{extentMm / 1000} m extent · 12 bytes per cell</small></div>
    <div className="compare-grid compare-three"><article><small>UNIFORM 5 CM · THEORETICAL</small><h2>{fmtBytes(memory?.uniform25d_bytes)}</h2><p>5 cm everywhere · {range}</p></article><article className="fovea-choice"><small>FOVEAMAP · ALLOCATED</small><h2>{fmtBytes(memory?.fovea_bytes)}</h2><p>{geometry} rings · {range}</p></article><article><small>UNIFORM 20 CM · THEORETICAL</small><h2>{frame ? fmtBytes(uniform20Bytes) : "—"}</h2><p>20 cm everywhere · {range}</p></article></div>
    <div className="comparison-table compare-four"><div><span>Metric</span><b>Uniform 5 cm</b><b>FoveaMap</b><b>Uniform 20 cm</b></div><div><span>Allocated memory</span><b>{fmtBytes(memory?.uniform25d_bytes)}</b><b>{fmtBytes(memory?.fovea_bytes)}</b><b>{frame ? fmtBytes(uniform20Bytes) : "—"}</b></div><div><span>Near resolution</span><b>5 cm</b><b>{finest}</b><b>20 cm</b></div><div><span>Far resolution</span><b>5 cm</b><b>{last ? `${last.cell_mm / 10} cm` : "—"}</b><b>20 cm</b></div></div>
    <div className="benchmark-visual"><header><div><span className="section-eyebrow">REAL SCAN DETAIL</span><h3>Same windows, different cell sizes</h3><p>Choose a near window. A far window with observed ground is selected from the same frame.</p></div><form onSubmit={(event) => { event.preventDefault(); if (Math.max(Math.abs(location.x), Math.abs(location.y)) > nearLimit) { setError(`Choose a center within ±${nearLimit.toFixed(1)} m to keep the 5 m window in the near ring.`); return; } if (frame) { const far = chooseFarWindow(frame); setTarget({ ...location, farX: far.x, farY: far.y, frameIdx: frame.frame_idx }); } }}><label>X <input type="number" min={-nearLimit} max={nearLimit} step="0.1" value={location.x} onChange={(event) => setLocation({ ...location, x: Number(event.target.value) })} /></label><label>Y <input type="number" min={-nearLimit} max={nearLimit} step="0.1" value={location.y} onChange={(event) => setLocation({ ...location, y: Number(event.target.value) })} /></label><button type="submit" disabled={!frame}>Inspect</button></form></header>
      {comparison && <><h4>Near field · ({comparison.center_m[0].toFixed(1)}, {comparison.center_m[1].toFixed(1)}) m</h4><div className="benchmark-visual-grid"><StepPreview steps={comparison.fine_step_mm} label="Uniform 5 cm" detected={comparison.fine_kerb} /><StepPreview steps={comparison.fine_step_mm} label="Fovea near resolution · 5 cm" detected={comparison.fine_kerb} /><StepPreview steps={comparison.medium_step_mm} label="Uniform 20 cm" detected={comparison.medium_kerb} /></div></>}
      {farComparison && <><h4>Far field · ({farComparison.center_m[0].toFixed(1)}, {farComparison.center_m[1].toFixed(1)}) m</h4><div className="benchmark-visual-grid"><StepPreview steps={farComparison.fine_step_mm} label="Uniform 5 cm" detected={farComparison.fine_kerb} /><StepPreview steps={farComparison.far_step_mm} label="Fovea far resolution · 40 cm" detected={farComparison.far_kerb} /><StepPreview steps={farComparison.medium_step_mm} label="Uniform 20 cm" detected={farComparison.medium_kerb} /></div></>}
      {error && <p className="lab-error">{error}</p>}
      <small>Frame {comparison?.frame_idx ?? "—"} · ground-truth-labelled LiDAR · both centered on real 5 m windows with grid padding. Each panel is a local reraster at the labeled cell size; it is not a replay of the full-grid state. Step height and kerb flags are recomputed by the backend.</small>
    </div>
    <p className="evidence-note">Current FoveaMap stage total: {latency ? `${latency.toFixed(1)} ms` : "—"} ({fps ? `${fps.toFixed(1)} derived FPS` : "—"}). Uniform-grid pipeline timings have not been measured on the same frame and hardware, so no throughput comparison is shown.</p>
  </section>;
}

function TelemetryChart({ title, values, color, unit, reference }: { title: string; values: number[]; color: string; unit: string; reference?: number }) {
  const width = 420, height = 130, pad = 14;
  const max = Math.max(reference ?? 0, ...values, 1);
  const points = values.map((value, index) => {
    const x = pad + (values.length <= 1 ? 0 : index / (values.length - 1)) * (width - pad * 2);
    const y = height - pad - value / max * (height - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  const latest = values.at(-1);
  return <article className="telemetry-chart"><header><span>{title}</span><b>{latest === undefined ? "—" : `${latest.toFixed(1)} ${unit}`}</b></header><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
    {[.25, .5, .75].map((part) => <line key={part} x1={pad} x2={width - pad} y1={height * part} y2={height * part} className="chart-gridline" />)}
    {reference && <line x1={pad} x2={width - pad} y1={height - pad - reference / max * (height - pad * 2)} y2={height - pad - reference / max * (height - pad * 2)} className="chart-reference" />}
    {points && <polyline points={points} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" />}
  </svg><small>Last {values.length} received frames · scale 0–{max.toFixed(1)} {unit}{reference ? ` · dashed: ${reference} ${unit}` : ""}</small></article>;
}

export default App;
