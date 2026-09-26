import { useEffect, useMemo, useRef, type MouseEventHandler } from "react";
import type { ActiveLayer, AnalysisMode, FrameUpdatePayload, RingSparse } from "../types";

type Paint = [number, number, number, number] | null;

interface Props {
  frame: FrameUpdatePayload | null;
  layer: ActiveLayer;
  mode: AnalysisMode;
  width: number;
  height: number;
  zoom: number;
  offset: { x: number; y: number };
  showRings: boolean;
  showObjects: boolean;
  showVehicle: boolean;
  selectedRing: number | null;
  selectedObject: number | null;
  probePoint: { x: number; y: number } | null;
  onZoom: (deltaY: number) => void;
  onUnavailable: () => void;
  colorFor: (ring: RingSparse, index: number, layer: ActiveLayer, mode: AnalysisMode) => Paint;
  onMouseDown: MouseEventHandler<HTMLCanvasElement>;
  onMouseMove: MouseEventHandler<HTMLCanvasElement>;
  onMouseUp: MouseEventHandler<HTMLCanvasElement>;
  onMouseLeave: MouseEventHandler<HTMLCanvasElement>;
  onClick: MouseEventHandler<HTMLCanvasElement>;
}

const VERTEX = `#version 300 es
precision highp float;
layout(location=0) in vec4 aCell;
layout(location=1) in vec4 aPaint;
layout(location=2) in float aSize;
uniform vec2 uViewport;
uniform vec2 uCenter;
uniform float uScale;
out vec4 vPaint;
void main() {
  int face = gl_VertexID / 6;
  int corner = gl_VertexID % 6;
  vec2 tile;
  float z;
  if (face == 0) {
    tile = corner == 0 || corner == 3 || corner == 5 ? vec2(0.0, 0.0) :
           corner == 1 ? vec2(1.0, 0.0) :
           corner == 2 || corner == 4 ? vec2(1.0, 1.0) : vec2(0.0, 1.0);
    // Triangle two starts at (0,0), (1,1), (0,1).
    if (corner == 5) tile = vec2(0.0, 1.0);
    z = aCell.w;
  } else if (face == 1) {
    tile = corner == 0 || corner == 3 || corner == 5 ? vec2(1.0, 0.0) :
           corner == 1 ? vec2(1.0, 1.0) :
           corner == 2 || corner == 4 ? vec2(1.0, 1.0) : vec2(1.0, 0.0);
    z = corner == 0 || corner == 1 || corner == 3 ? aCell.w : aCell.z;
  } else {
    tile = corner == 0 || corner == 3 || corner == 5 ? vec2(0.0, 1.0) :
           corner == 1 ? vec2(1.0, 1.0) :
           corner == 2 || corner == 4 ? vec2(1.0, 1.0) : vec2(0.0, 1.0);
    z = corner == 0 || corner == 1 || corner == 3 ? aCell.w : aCell.z;
  }
  vec2 world = aCell.xy + tile * aSize;
  vec2 projected = vec2((world.x - world.y) * 0.70, (-world.x - world.y) * 0.35 - z * 0.9);
  vec2 pixel = uCenter + projected * uScale;
  gl_Position = vec4(pixel.x / uViewport.x * 2.0 - 1.0, 1.0 - pixel.y / uViewport.y * 2.0, 0.0, 1.0);
  float shade = face == 0 ? 1.0 : face == 1 ? 0.63 : 0.76;
  vPaint = vec4(aPaint.rgb * shade, aPaint.a);
}`;

const FRAGMENT = `#version 300 es
precision mediump float;
in vec4 vPaint;
out vec4 outputColor;
void main() { outputColor = vPaint; }`;

function shader(gl: WebGL2RenderingContext, type: number, source: string) {
  const result = gl.createShader(type);
  if (!result) return null;
  gl.shaderSource(result, source);
  gl.compileShader(result);
  if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(result));
    gl.deleteShader(result);
    return null;
  }
  return result;
}

function project(x: number, y: number, z: number, cx: number, cy: number, scale: number) {
  return { x: cx + (x - y) * .7 * scale, y: cy + (-x - y) * .35 * scale - z * .9 * scale };
}

export function IsometricMap(props: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlay = useRef<HTMLCanvasElement | null>(null);
  const renderer = useRef<{ gl: WebGL2RenderingContext; program: WebGLProgram; buffer: WebGLBuffer; vao: WebGLVertexArrayObject; vert: WebGLShader; frag: WebGLShader } | null>(null);
  const uploaded = useRef<Float32Array | null>(null);
  const { frame, layer, mode, width, height, zoom, offset, colorFor, onZoom, onUnavailable } = props;
  const maxExtent = frame?.rings.length ? Math.max(...frame.rings.map((ring) => ring.r_max_mm)) / 1000 : 100;
  const scale = Math.min(width, height) * .44 / maxExtent * zoom;
  const cx = width / 2 + offset.x, cy = height / 2 + offset.y;

  const cells = useMemo(() => {
    if (!frame) return new Float32Array(0);
    const values: number[] = [];
    for (const ring of [...frame.rings].sort((a, b) => b.ring_idx - a.ring_idx)) {
      for (let i = 0; i < ring.ix.length; i++) {
        const paint = colorFor(ring, i, layer, mode);
        if (!paint) continue;
        const ground = ring.ground_z[i] !== -32768 ? ring.ground_z[i] / 1000 : 0;
        const top = ring.top_z[i] !== -32768 ? ring.top_z[i] / 1000 : ground;
        values.push(
          -ring.r_max_mm / 1000 + ring.ix[i] * ring.cell_mm / 1000,
          -ring.r_max_mm / 1000 + ring.iy[i] * ring.cell_mm / 1000,
          ground - 0.16, Math.max(ground, top),
          paint[0] / 255, paint[1] / 255, paint[2] / 255, paint[3],
          ring.cell_mm / 1000,
        );
      }
    }
    return new Float32Array(values);
  }, [frame, layer, mode, colorFor]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas?.getContext("webgl2", { alpha: false, antialias: true });
    if (!gl || !canvas) { onUnavailable(); return; }
    const vert = shader(gl, gl.VERTEX_SHADER, VERTEX);
    const frag = shader(gl, gl.FRAGMENT_SHADER, FRAGMENT);
    if (!vert || !frag) return;
    const program = gl.createProgram();
    const buffer = gl.createBuffer();
    const vao = gl.createVertexArray();
    if (!program || !buffer || !vao) return;
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
      return;
    }
    renderer.current = { gl, program, buffer, vao, vert, frag };
    uploaded.current = null;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    for (const [index, length, start] of [[0, 4, 0], [1, 4, 4], [2, 1, 8]]) {
      gl.enableVertexAttribArray(index);
      gl.vertexAttribPointer(index, length, gl.FLOAT, false, 36, start * 4);
      gl.vertexAttribDivisor(index, 1);
    }
    gl.bindVertexArray(null);
    return () => {
      renderer.current = null;
      uploaded.current = null;
      gl.deleteBuffer(buffer);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
      gl.deleteShader(vert);
      gl.deleteShader(frag);
    };
  }, [onUnavailable]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => { event.preventDefault(); onZoom(event.deltaY); };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [onZoom]);

  useEffect(() => {
    const resources = renderer.current;
    if (!resources) return;
    const { gl, program, buffer, vao } = resources;
    gl.viewport(0, 0, width, height);
    gl.clearColor(.047, .063, .09, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);
    gl.uniform2f(gl.getUniformLocation(program, "uViewport"), width, height);
    gl.uniform2f(gl.getUniformLocation(program, "uCenter"), cx, cy);
    gl.uniform1f(gl.getUniformLocation(program, "uScale"), scale);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    if (uploaded.current !== cells) {
      gl.bufferData(gl.ARRAY_BUFFER, cells, gl.DYNAMIC_DRAW);
      uploaded.current = cells;
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 18, cells.length / 9);
    gl.bindVertexArray(null);
  }, [cells, width, height, cx, cy, scale]);

  useEffect(() => {
    const canvas = overlay.current, ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (props.showRings && frame) {
      for (const ring of frame.rings) {
        const r = ring.r_max_mm / 1000;
        const corners = [[-r, -r], [r, -r], [r, r], [-r, r]];
        ctx.strokeStyle = props.selectedRing === ring.ring_idx ? "#f6bf66" : "rgba(120,191,207,.56)";
        ctx.lineWidth = props.selectedRing === ring.ring_idx ? 2 : 1;
        ctx.setLineDash(ring.ring_idx ? [5, 4] : []);
        ctx.beginPath();
        corners.forEach(([x, y], index) => {
          const p = project(x, y, 0, cx, cy, scale);
          if (index) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
        });
        ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
        const label = project(r, -r, 0, cx, cy, scale);
        ctx.fillStyle = "#9cc3d1"; ctx.font = "11px monospace";
        ctx.fillText(`R${ring.ring_idx} · ${ring.cell_mm / 10} cm`, label.x + 6, label.y);
      }
    }
    if (props.showObjects && mode === "objects" && frame) {
      for (const obj of frame.objects) {
        if (obj.size[0] > 15 || obj.size[1] > 15) continue;
        const p = project(obj.center[0], obj.center[1], obj.center[2], cx, cy, scale);
        ctx.strokeStyle = props.selectedObject === obj.id ? "#f6bf66" : obj.moving ? "#f05a65" : "#58bdde";
        ctx.lineWidth = 2;
        ctx.strokeRect(p.x - 7, p.y - 7, 14, 14);
        ctx.font = "10px monospace";
        ctx.fillStyle = ctx.strokeStyle;
        const speed = obj.speed_mps == null ? "" : ` ${(obj.speed_mps * 3.6).toFixed(0)} km/h`;
        ctx.fillText(`${obj.cls_name}${speed}`, p.x + 11, p.y - 9);
      }
    }
    if (props.showVehicle) {
      const p = project(0, 0, 0, cx, cy, scale);
      ctx.fillStyle = "#64c6e6";
      ctx.beginPath(); ctx.moveTo(p.x, p.y - 8); ctx.lineTo(p.x - 6, p.y + 6); ctx.lineTo(p.x + 6, p.y + 6); ctx.closePath(); ctx.fill();
    }
    if (props.probePoint) {
      const p = project(props.probePoint.x, props.probePoint.y, 0, cx, cy, scale);
      ctx.strokeStyle = "#ffcc73"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p.x - 13, p.y); ctx.lineTo(p.x + 13, p.y); ctx.moveTo(p.x, p.y - 13); ctx.lineTo(p.x, p.y + 13); ctx.stroke();
    }
  }, [frame, props.showRings, props.showObjects, props.showVehicle, props.selectedRing, props.selectedObject, props.probePoint, mode, width, height, cx, cy, scale]);

  return <div className="iso-stack">
    <canvas ref={canvasRef} width={width} height={height} aria-label="Isometric 2.5D map" onMouseDown={props.onMouseDown} onMouseMove={props.onMouseMove} onMouseUp={props.onMouseUp} onMouseLeave={props.onMouseLeave} onClick={props.onClick} />
    <canvas ref={overlay} width={width} height={height} className="iso-overlay" aria-hidden="true" />
    {!frame && <span className="iso-empty">Waiting for frame data…</span>}
  </div>;
}
