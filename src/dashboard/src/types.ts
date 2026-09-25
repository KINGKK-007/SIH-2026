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

export interface RingSparse {
  ring_idx: number;
  cell_mm: number;
  r_max_mm: number;
  side: number;
  iy: number[];
  ix: number[];
  ground_z: number[];
  top_z: number[];
  clearance: number[];
  cls: number[];
  moving_frac: number[];
  count: number[];
  conf: number[];
  flags: number[];
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
  rings: RingSparse[];
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
