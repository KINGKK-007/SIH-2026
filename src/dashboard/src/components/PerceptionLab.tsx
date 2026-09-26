import { useEffect, useRef, useState } from "react";
import type { FrameUpdatePayload } from "../types";
import { SERVER_URL } from "../socket";

type Point = { x: number; y: number; frameIdx: number };
export type ZoomResult = {
  frame_idx: number;
  center_m: [number, number];
  source: string;
  fine_step_mm: number[][];
  coarse_step_mm: number[][];
  medium_step_mm: number[][];
  far_step_mm: number[][];
  fine_kerb: boolean;
  medium_kerb: boolean;
  far_kerb: boolean;
  coarse_kerb: boolean;
};
type HazardResult = {
  kind: string;
  ring_idx: number | null;
  detected: boolean;
  recompute_ms: number;
  frame: FrameUpdatePayload;
};

async function postJSON<T>(path: string, body: object, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${SERVER_URL}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body), signal,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function StepPreview({ steps, label, detected }: { steps: number[][]; label: string; detected: boolean }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current, context = canvas?.getContext("2d");
    if (!canvas || !context || !steps.length) return;
    const rows = steps.length, cols = steps[0].length;
    canvas.width = cols; canvas.height = rows;
    const image = context.createImageData(cols, rows);
    for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
      const step = steps[rows - 1 - y][cols - 1 - x] ?? 0;
      const t = Math.min(1, Math.max(0, step / 250));
      const at = (y * cols + x) * 4;
      image.data[at] = Math.round(25 + t * 230);
      image.data[at + 1] = Math.round(46 + t * 101);
      image.data[at + 2] = Math.round(63 - t * 30);
      image.data[at + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [steps]);
  return <div className="lens-tile"><div className="lens-tile-head"><strong>{label}</strong><span className={detected ? "detected" : "missed"}>{detected ? "Kerb flag present" : "No kerb flag"}</span></div><canvas ref={ref} aria-label={`${label} step height map`} /><small>Step height · navy 0 mm → amber ≥250 mm</small></div>;
}

export function PerceptionLab({ frame, point, probeMode, onProbeMode, onPreview }: {
  frame: FrameUpdatePayload | null;
  point: Point | null;
  probeMode: boolean;
  onProbeMode: (value: boolean) => void;
  onPreview: (frame: FrameUpdatePayload | null) => void;
}) {
  const [lens, setLens] = useState<ZoomResult | null>(null);
  const [lensBusy, setLensBusy] = useState(false);
  const [lensError, setLensError] = useState("");
  const [radius0, setRadius0] = useState(10);
  const [radius1, setRadius1] = useState(30);
  const [cell0, setCell0] = useState(5);
  const [cell1, setCell1] = useState(10);
  const [playground, setPlayground] = useState(false);
  const [recomputeMs, setRecomputeMs] = useState<number | null>(null);
  const [previewMemory, setPreviewMemory] = useState<number | null>(null);
  const [previewCells, setPreviewCells] = useState<number | null>(null);
  const [playgroundError, setPlaygroundError] = useState("");
  const [hazardBusy, setHazardBusy] = useState(false);
  const [hazardResult, setHazardResult] = useState<HazardResult | null>(null);
  const [hazardError, setHazardError] = useState("");
  const frameIdx = frame?.frame_idx;
  const lensFrameIdx = point?.frameIdx;
  const pointX = point?.x, pointY = point?.y;
  const currentLens = lens && point && lens.frame_idx === point.frameIdx && lens.center_m[0] === point.x && lens.center_m[1] === point.y ? lens : null;

  useEffect(() => {
    if (lensFrameIdx == null || pointX == null || pointY == null) return;
    const abort = new AbortController();
    const timer = window.setTimeout(() => {
      setLensBusy(true); setLensError("");
      postJSON<ZoomResult>("/api/zoom", { frame_idx: lensFrameIdx, x_m: pointX, y_m: pointY }, abort.signal)
        .then(setLens)
        .catch((error: Error) => { if (error.name !== "AbortError") setLensError(error.message); })
        .finally(() => { if (!abort.signal.aborted) setLensBusy(false); });
    }, 120);
    return () => { abort.abort(); window.clearTimeout(timer); };
  }, [lensFrameIdx, pointX, pointY]);

  useEffect(() => {
    if (!playground || frameIdx == null) return;
    const abort = new AbortController();
    const timer = window.setTimeout(() => {
      setPlaygroundError("");
      postJSON<{ frame: FrameUpdatePayload; recompute_ms: number }>("/api/fovea-preview", {
        frame_idx: frameIdx, ring0_m: radius0, ring1_m: radius1,
        cell0_cm: cell0, cell1_cm: cell1,
      }, abort.signal).then((response) => {
        onPreview(response.frame);
        setRecomputeMs(response.recompute_ms);
        setPreviewMemory(response.frame.memory.fovea_bytes / 1048576);
        setPreviewCells(response.frame.rings.reduce((sum, ring) => sum + ring.ix.length, 0));
      }).catch((error: Error) => { if (error.name !== "AbortError") setPlaygroundError(error.message); });
    }, 150);
    return () => { abort.abort(); window.clearTimeout(timer); };
  }, [playground, frameIdx, radius0, radius1, cell0, cell1, onPreview]);

  const enablePlayground = async () => {
    await postJSON("/api/pause", {}).catch(() => undefined);
    onPreview(null);
    setPlayground(true);
    setHazardResult(null);
  };

  const inject = async (kind: string) => {
    if (!frame || !point) return;
    setHazardBusy(true); setHazardError(""); setHazardResult(null);
    await postJSON("/api/pause", {}).catch(() => undefined);
    try {
      const result = await postJSON<HazardResult>("/api/hazard-preview", {
        frame_idx: point.frameIdx, kind, x_m: point.x, y_m: point.y,
      });
      setHazardResult(result);
      onPreview(result.frame);
      setPlayground(false);
    } catch (error) {
      setHazardError(error instanceof Error ? error.message : "Hazard preview failed");
    } finally { setHazardBusy(false); }
  };

  return <section className="perception-lab" aria-label="Interactive perception tools">
    <div className="lab-title"><div><span className="section-eyebrow">INTERACTIVE DEMO</span><h3>Inspect the evidence</h3></div><small>All previews use the current scan · source data is unchanged</small></div>
    <div className="lab-grid">
      <article className="lab-card lens-card">
        <div className="lab-card-head"><strong>Resolution lens</strong><button className={probeMode ? "control-active" : ""} onClick={() => { if (!probeMode) void postJSON("/api/pause", {}).catch(() => undefined); onProbeMode(!probeMode); }}>{probeMode ? "Click a map location" : "Pick a map location"}</button></div>
        <p>Compare one 5 m window at 5 cm and 50 cm. The server rerasterizes the real scan and runs the kerb detector at both resolutions.</p>
        <div className="lab-location">{point ? `X ${point.x.toFixed(2)} m · Y ${point.y.toFixed(2)} m` : "Select a location on the map"}{lensBusy && " · computing…"}</div>
        {currentLens && <><div className="lens-pair"><StepPreview steps={currentLens.fine_step_mm} label="Fine · 5 cm" detected={currentLens.fine_kerb} /><StepPreview steps={currentLens.coarse_step_mm} label="Coarse · 50 cm" detected={currentLens.coarse_kerb} /></div><small className="lab-source">Frame {currentLens.frame_idx} · {currentLens.source.replaceAll("_", " ")}. A missing flag is reported as observed, without assuming a kerb is present.</small></>}
        {lensError && <p className="lab-error">{lensError}</p>}
      </article>

      <article className="lab-card">
        <div className="lab-card-head"><strong>Fovea playground</strong><button onClick={() => { if (playground) { setPlayground(false); onPreview(null); } else void enablePlayground(); }}>{playground ? "Close preview" : "Adjust rings"}</button></div>
        <p>Rerasterize the current frame with valid ring radii and cell sizes. Outer rings remain at 20 / 40 cm and the extent stays at 100 m.</p>
        {playground && <div className="lab-sliders"><label>Near field · R0 <strong>{radius0} m</strong><input type="range" min="5" max="20" step="1" value={radius0} onChange={(event) => setRadius0(Math.min(Number(event.target.value), radius1 - 1))} /></label><label>Middle field · R1 <strong>{radius1} m</strong><input type="range" min="20" max="40" step="1" value={radius1} onChange={(event) => setRadius1(Math.max(Number(event.target.value), radius0 + 1))} /></label><label>Near cell size <select value={cell0} onChange={(event) => setCell0(Number(event.target.value))}><option value="5">5 cm</option><option value="10">10 cm</option></select></label><label>Middle cell size <select value={cell1} onChange={(event) => setCell1(Number(event.target.value))}><option value="10">10 cm</option><option value="20">20 cm</option></select></label></div>}
        {playground && <div className="lab-metrics"><div><span>Map allocation</span><strong>{previewMemory == null ? "…" : `${previewMemory.toFixed(1)} MiB`}</strong></div><div><span>Observed cells</span><strong>{previewCells == null ? "…" : previewCells.toLocaleString()}</strong></div><div><span>Recompute</span><strong>{recomputeMs == null ? "…" : `${recomputeMs.toFixed(0)} ms`}</strong></div></div>}
        {playgroundError && <p className="lab-error">{playgroundError}</p>}
      </article>

      <article className="lab-card">
        <div className="lab-card-head"><strong>Hazard injection</strong><span className="lab-demo-tag">SYNTHETIC PREVIEW</span></div>
        <p>Pick observed drivable ground, then inject one hazard into a copy of this oracle scan. The map and detector rerun without editing the dataset.</p>
        <div className="hazard-buttons">{(["kerb", "pothole", "overhang"] as const).map((kind) => <button key={kind} disabled={!point || !frame || hazardBusy} onClick={() => void inject(kind)}>{kind === "overhang" ? "Low clearance" : kind[0].toUpperCase() + kind.slice(1)}</button>)}</div>
        {hazardBusy && <p className="lab-status">Reprocessing frame…</p>}
        {hazardResult && <><p className={`lab-result ${hazardResult.detected ? "detected" : "missed"}`}>R{hazardResult.ring_idx ?? "?"} · {hazardResult.detected ? "Detector flagged the injected hazard" : "Detector missed the injected hazard"} · {hazardResult.recompute_ms.toFixed(0)} ms</p><button className="clear-preview" onClick={() => { onPreview(null); setHazardResult(null); }}>Clear preview</button></>}
        {hazardError && <p className="lab-error">{hazardError}</p>}
      </article>
    </div>
  </section>;
}
