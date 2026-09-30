export interface FrameCounters {
  n_raw: number;
  n_invalid: number;
  n_in_grid: number;
  n_out_of_grid: number;
  n_z_saturated: number;
}

export interface TimingsMs {
  io_ms?: number;
  model_ms?: number;
  label_ms?: number;
  grid_ms?: number;
  finalize_ms?: number;
  memory_ms?: number;
  [key: string]: number | undefined;
}

export interface MemoryReport {
  basis: string;
  dense3d_bytes: number;
  sparse3d_bytes: number;
  uniform25d_bytes: number;
  fovea_bytes: number;
  rss_delta_bytes: number;
}

/** Wire format — server sends cells_b64 (binary v2) or legacy JSON arrays (v1). */
export interface RingSparseWire {
  ring_idx: number;
  cell_mm: number;
  r_max_mm: number;
  side: number;
  // Binary v2 (schema_version >= 2)
  n_cells?: number;
  cells_b64?: string;
  display_group?: number[];
  // Legacy JSON v1 (schema_version 1) — kept for backwards compat
  iy?: number[];
  ix?: number[];
  ground_z?: number[];
  top_z?: number[];
  clearance?: number[];
  cls?: number[];
  moving_frac?: number[];
  count?: number[];
  conf?: number[];
  flags?: number[];
}

/** Decoded, ready-to-render ring — always has typed arrays after decodeRing(). */
export interface RingSparse {
  ring_idx: number;
  cell_mm: number;
  r_max_mm: number;
  side: number;
  iy: Int16Array;
  ix: Int16Array;
  ground_z: Int16Array;
  top_z: Int16Array;
  flags: Uint8Array;
  cls: Uint8Array;
  moving_frac: Uint8Array;
  conf: Uint8Array;
  display_group?: number[];
}

/** Decode a wire ring (binary v2 or legacy JSON) into a typed-array RingSparse. */
export function decodeRing(wire: RingSparseWire): RingSparse {
  const meta = {
    ring_idx: wire.ring_idx,
    cell_mm: wire.cell_mm,
    r_max_mm: wire.r_max_mm,
    side: wire.side,
    display_group: wire.display_group,
  };

  if (wire.cells_b64 !== undefined && wire.cells_b64 !== "") {
    // Binary path: 12 bytes per cell, little-endian
    const raw = Uint8Array.from(atob(wire.cells_b64), c => c.charCodeAt(0));
    const n = raw.byteLength / 12;
    const dv = new DataView(raw.buffer);
    const iy = new Int16Array(n);
    const ix = new Int16Array(n);
    const gz = new Int16Array(n);
    const tz = new Int16Array(n);
    const flags = new Uint8Array(n);
    const cls = new Uint8Array(n);
    const mf = new Uint8Array(n);
    const conf = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const o = i * 12;
      iy[i]    = dv.getInt16(o,     true);
      ix[i]    = dv.getInt16(o + 2, true);
      gz[i]    = dv.getInt16(o + 4, true);
      tz[i]    = dv.getInt16(o + 6, true);
      flags[i] = dv.getUint8(o + 8);
      cls[i]   = dv.getUint8(o + 9);
      mf[i]    = dv.getUint8(o + 10);
      conf[i]  = dv.getUint8(o + 11);
    }
    return { ...meta, iy, ix, ground_z: gz, top_z: tz, flags, cls, moving_frac: mf, conf };
  }

  // Legacy JSON fallback
  const toI16 = (a?: number[]) => Int16Array.from(a ?? []);
  const toU8  = (a?: number[]) => Uint8Array.from(a ?? []);
  return {
    ...meta,
    iy: toI16(wire.iy), ix: toI16(wire.ix),
    ground_z: toI16(wire.ground_z), top_z: toI16(wire.top_z),
    flags: toU8(wire.flags), cls: toU8(wire.cls),
    moving_frac: toU8(wire.moving_frac), conf: toU8(wire.conf),
  };
}

export interface ObjectBox {
  id: number;
  cls_name: string;
  center: [number, number, number];
  size: [number, number, number];
  yaw: number;
  n_points: number;
  mean_conf: number;
  moving: boolean;
  vote_frac: number;
  speed_mps: number | null;
  velocity_xy: [number, number] | null;
  instance_id: number | null;
  safety_critical: boolean;
}

export interface FrameUpdatePayload {
  schema_version: number;
  seq: string;
  frame_idx: number;
  timestamp: number;
  model: string;
  preset: string;
  counters: FrameCounters;
  timings_ms: TimingsMs;
  memory: MemoryReport;
  rings: RingSparse[];   // decoded — always typed arrays after socket handler runs decodeRing
  objects: ObjectBox[];
}

export interface PlaybackState {
  seq: string;
  frame_idx: number;
  total_frames: number;
  mode: string;
  model: string;
  preset: string;
  is_playing: boolean;
  speed: number;
}

export type ActiveLayer = "class" | "height" | "traversability" | "moving" | "confidence";
export type AnalysisMode = "terrain" | "objects";
