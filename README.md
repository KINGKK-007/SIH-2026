# FoveaMap

### Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

**Smart India Hackathon 2026 · Problem Statement SIH26053**

> Foveated 2.5D LiDAR mapping: sharp where a mistake hurts, cheap where it doesn't.
> FoveaMap turns raw 3D LiDAR point clouds into a variable-resolution 2.5D elevation-and-semantics map —
> 5 cm detail next to the vehicle, coarsening out to 40 cm at 100 m — the way human vision keeps the
> centre of gaze sharp and the periphery coarse.

[![License: MIT](https://img.shields.io/badge/license-MIT-informational)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Node 18+](https://img.shields.io/badge/node-18%2B-339933)](src/dashboard/package.json)
[![Stack](https://img.shields.io/badge/stack-FastAPI%20%2B%20Socket.IO%20%2B%20React%2019-informational)](#architecture)
[![Dataset](https://img.shields.io/badge/dataset-SemanticKITTI%20seq.%2008-orange)](#5-dataset)
[![Model](https://img.shields.io/badge/model-LSK3DNet%20(CVPR%202024)-blueviolet)](#6-segmentation-model--lsk3dnet)

---

## Demo video

<a href="https://youtu.be/_Xer_w2xR28" target="_blank">
  <img src="https://img.youtube.com/vi/_Xer_w2xR28/maxresdefault.jpg" alt="FoveaMap demo video" width="720">
  <br>
  <img src="https://img.shields.io/badge/%E2%96%B6-Watch%20the%20demo%20on%20YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch on YouTube">
</a>

**[youtu.be/_Xer_w2xR28](https://youtu.be/_Xer_w2xR28)** — click the thumbnail above to play the demo.

---

## Table of contents

1. [What is FoveaMap](#1-what-is-foveamap)
2. [Why a foveated grid](#2-why-a-foveated-grid)
3. [Architecture](#3-architecture)
4. [Repository structure](#4-repository-structure)
5. [Dataset](#5-dataset)
6. [Segmentation model — LSK3DNet](#6-segmentation-model--lsk3dnet)
7. [The variable-resolution grid engine](#7-the-variable-resolution-grid-engine)
8. [Motion & object detection](#8-motion--object-detection)
9. [Derived layers & hazards](#9-derived-layers--hazards)
10. [Dashboard](#10-dashboard)
11. [Installation](#11-installation)
12. [Running the project](#12-running-the-project)
13. [CLI reference](#13-cli-reference)
14. [Configuration reference](#14-configuration-reference)
15. [Measured results](#15-measured-results)
16. [Testing & quality gates](#16-testing--quality-gates)
17. [Known limitations](#17-known-limitations)
18. [Roadmap](#18-roadmap)
19. [Engineering decisions](#19-engineering-decisions)
20. [Citations & acknowledgements](#20-citations--acknowledgements)
21. [License](#21-license)

---

## 1. What is FoveaMap

Autonomous vehicles need rich 3D LiDAR perception, but a full-resolution 3D representation is too large and
too slow to process every frame, while a flat 2D occupancy grid throws away the height information needed to
see curbs, potholes, and overhanging obstacles. FoveaMap resolves that trade-off with a **foveated 2.5D grid**:
cell size grows with distance from the sensor, matching the physical point-spacing of a rotating LiDAR, so the
region that matters most for collision safety stays at native resolution while far-field memory and compute
collapse by over an order of magnitude.

The system ingests raw SemanticKITTI LiDAR scans and produces, per frame:

- **Terrain analysis** — drivable vs. non-drivable vs. unknown surface, ground/obstacle elevation, slope,
  kerb/step detection, overhang clearance, and a 3-state traversability map.
- **Object detection** — semantic class (vehicle, pedestrian, pole, structure, …), oriented bounding box,
  distance, confidence, and an independently-computed motion state (`STATIC` / `MOVING` / `UNCERTAIN`).
- **A live dashboard** — a real-time Socket.IO-streamed 2.5D map with resolution-ring overlays, an object
  panel, a memory meter, and a foveated-vs-uniform benchmark comparison.

Everything is built **oracle-first**: the grid engine, invariants, and dashboard were validated against
ground-truth SemanticKITTI labels before the deep-learning model was wired in, so grid-coarsening error and
model error are never conflated.

---

## 2. Why a foveated grid

A rotating multi-beam LiDAR (KITTI's Velodyne HDL-64E) samples the world along concentric arcs — point
spacing is a few centimetres at 10 m but tens of centimetres by 100 m. A uniform fine grid wastes memory on
cells the sensor can never densely fill at range; a uniform coarse grid throws away the fine detail needed for
safety right next to the vehicle. FoveaMap's ring sizes are chosen to track this physical sampling limit, not
picked arbitrarily — and the memory/accuracy trade-off is measured, not asserted (see
[§15 Measured results](#15-measured-results)).

| Ring | Range | Cell size | Role |
|---|---|---:|---|
| R0 | 0 – 10 m | **5 cm** | Near-field safety zone — kerbs, potholes, close obstacles |
| R1 | 10 – 30 m | **10 cm** | Intermediate detail |
| R2 | 30 – 60 m | **20 cm** | Reduced detail |
| R3 | 60 – 100 m | **40 cm** | Far-field situational awareness |

Each ring is an exact square annulus nested inside the next (every coarse cell boundary coincides with a fine
cell boundary — see [Grid invariants](#7-the-variable-resolution-grid-engine)), so there is no overlap, no gap,
and no alignment error between resolutions.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                   Data layer                                          │
│  SemanticKITTI scans + poses + calib → pose alignment → range bucketing → labels      │
└───────────────────────────────────┬────────────────────────────────────────────────────┘
                                     │  per-point (x, y, z, intensity)
                 ┌───────────────────┴────────────────────┐
                 ▼                                          ▼
   ┌─────────────────────────────┐          ┌──────────────────────────────────┐
   │   LSK3DNet (sparse voxel     │          │   Oracle mode (SemanticKITTI      │
   │   CNN, CVPR 2024) — per-     │          │   ground-truth labels) — isolates │
   │   point semantic class +     │          │   grid error from model error     │
   │   confidence, 100 m extended │          │                                    │
   │   coverage                   │          │                                    │
   └───────────────┬───────────────┘          └─────────────────┬──────────────────┘
                   └───────────────────┬───────────────────────┘
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │   Geometric motion estimation            │
                   │   ego-compensated residual → range-scaled│
                   │   DBSCAN clustering → oriented boxes →   │
                   │   temporal hysteresis tracking           │
                   └───────────────────┬───────────────────────┘
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │   Foveated grid engine (NumPy backend,   │
                   │   optional C++/pybind11)                 │
                   │   integer-millimetre addressing, 4 nested│
                   │   square rings, sparse accumulators,     │
                   │   12-byte packed cell layers             │
                   └───────────────────┬───────────────────────┘
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │   Derived layers                         │
                   │   slope · kerb/step · overhang clearance │
                   │   · 3-state traversability · synthetic   │
                   │   hazard injection (pothole/kerb/overhang)│
                   └───────────────────┬───────────────────────┘
                                        ▼
            ┌─────────────────────────────────────────────────────────┐
            │   Benchmarking                                          │
            │   memory (4 representations) · latency (per-stage p50/  │
            │   p95/p99) · accuracy-by-distance · cost of coarsening  │
            └────────────────────────────┬─────────────────────────────┘
                                          ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │   Real-time dashboard                                                 │
   │   FastAPI + python-socketio backend (gzip/binary frame transport) ──▶ │
   │   React 18 + Vite + TypeScript frontend (Canvas 2D/2.5D renderer,     │
   │   resolution rings, object panel, memory meter, benchmarks view)      │
   └───────────────────────────────────────────────────────────────────────┘
```

**Backend (Python, `src/foveamap/`)** — a staged, per-frame pipeline (`pipeline/runner.py`) that times every
stage independently (`load → preprocess → inference → motion → grid → derived → serialize`), oracle/cached/live
model modes, and a FastAPI + Socket.IO server that streams frame results to any number of connected dashboard
clients.

**Frontend (TypeScript, `src/dashboard/`)** — React 18 + Vite, a Canvas-based renderer that composites one
texture per ring directly from the server's packed cell arrays (no client-side re-rasterisation), with
top-down 2D and isometric 2.5D perspectives, object/terrain panels, and a live telemetry strip.

---

## 4. Repository structure

```text
SIH-2026/
├── src/
│   ├── main.py                    # thin shim: `python src/main.py ...` → foveamap.cli.main()
│   ├── foveamap/                  # the production Python package
│   │   ├── cli.py                 # `foveamap <command>` entry point (inspect/render/memory/align/stats/evaluate/serve)
│   │   ├── config.py              # pydantic config models, loads configs/*.yaml
│   │   ├── io/                    # SemanticKITTI I/O: scans, labels, poses, calib, sequence iterator, verification
│   │   ├── grid/                  # the foveated grid engine: presets, integer addressing, accumulators, packed layers
│   │   │   └── backends/          # NumpyBackend (reference/oracle) + optional C++ backend hook
│   │   ├── models/                # segmentation model wrappers: lsk3dnet.py (production), oracle.py, cache.py
│   │   ├── motion/                # geometric motion: residual.py, cluster.py, boxes.py, pipeline.py, tracker.py
│   │   ├── derived/                # slope, step/kerb, clearance, traversability, hazard injection
│   │   ├── eval/                  # buckets, density, alignment, data_stats, memory/latency/motion/hazard evaluators
│   │   ├── pipeline/               # PipelineRunner (stage orchestration + timers), temporal_map, display grouping
│   │   ├── server/                # FastAPI app, Socket.IO handlers, wire protocol, server-side render helpers
│   │   └── viz/                   # matplotlib figure helpers (top-down renders, video)
│   ├── foveamap_legacy/            # v0 reference implementation (motion/derived/hazard prototypes; not production)
│   └── dashboard/                 # React 18 + Vite + TypeScript frontend
│       └── src/
│           ├── App.tsx             # hash-routed view switcher (Overview/Object Detection/Terrain/Elevation/…)
│           ├── socket.ts           # Socket.IO client, gzip/binary frame decoding
│           └── components/         # MapView, IsometricMap, AppSidebar, Header, MemoryMeter, LatencyPanel, …
├── LSK3DNet-main/                  # vendored upstream LSK3DNet checkout (segmentation model source + weights config)
├── configs/                        # grid.yaml, model.yaml, motion.yaml, derived.yaml, hazards.yaml, benchmark.yaml, dashboard.yaml
├── data/                           # SemanticKITTI sequence 08 (git-ignored) + prediction/grid cache
├── docs/                           # DECISIONS.md, PHASES.md, PROGRESS.md, MODEL_CARD.md, LIMITATIONS.md, LOCKED_IMPLEMENTATION_PLAN.md, DEMO_SCRIPT.md
├── results/                        # generated metrics JSON, tables, plots (never hand-edited)
├── scripts/                        # doctor.py, verify_data.py, cache_predictions.py, run_benchmarks.py, …
├── tests/                          # 45+ test modules: grid, io, motion, derived, eval, models, server, e2e, legacy
├── Makefile                        # make doctor/install/test/lint/demo/benchmark/report/video/…
└── requirements.txt / pyproject.toml
```

---

## 5. Dataset

FoveaMap is built on **[SemanticKITTI](http://www.semantic-kitti.org/)** (per-point semantic labels) over the
**[KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)** benchmark's raw Velodyne scans,
poses and calibration. The current evaluation uses **sequence 08 only** (the standard SemanticKITTI validation
sequence), split chronologically with guard gaps to reduce leakage between tuning and reported numbers:

| Region | Frames | Purpose |
|---|---:|---|
| Integration / development | 0000 – 0814 | Debugging, visual design |
| Guard gap | 0815 – 0864 | — |
| Calibration | 0865 – 1220 | Threshold selection |
| Guard gap | 1221 – 1270 | — |
| **Locked system evaluation** | **1271 – 4070** | All reported numbers |

Because the pretrained LSK3DNet checkpoint may itself have used sequence 08 for validation, reported semantic
mIoU is described as **checkpoint reproduction**, not evidence of unseen-domain generalisation — the final
region is a temporally held-out *system* evaluation (pipeline + grid + motion), not a model-generalisation claim.

**Frame/pose math** follows the exact SemanticKITTI convention: `T_vel_i_from_vel_j = inv(Tr) · inv(P_i) · P_j · Tr`,
where `Tr` is the Velodyne→camera-0 calibration transform and `P_i` is the camera-0 world pose. All grid
arithmetic is done on **integer millimetres** (never floating-point division on a cell boundary) to eliminate
rounding ambiguity at ring edges.

---

## 6. Segmentation model — LSK3DNet

FoveaMap's production semantic segmentation model is **LSK3DNet**, a large-sparse-kernel 3D sparse convolution
network (CVPR 2024) — see [Citations](#20-citations--acknowledgements). It replaces the originally scoped
range-view family (SalsaNext/CENet/RangeNet++) because a pretrained checkpoint was available directly and a
sparse-voxel network reaches materially higher accuracy per the paper's published benchmark.

- **Input:** raw points `(x, y, z, intensity)` plus a per-point surface normal, computed via a vendored
  pybind11/C++ extension (`c_gen_normal_map`) from a 64×900 range-image projection.
- **Output:** `predict(points) → (raw_ids: uint8, conf: uint8)`, mapped back to raw SemanticKITTI label IDs
  through the checkpoint's `learning_map_inv`.
- **Coverage:** the native checkpoint is trained on a ±50 m / [-4, 2] m crop at 5 cm voxels. FoveaMap's
  `extended_100m` coverage profile (the production default) expands the sparse coordinate space to ±100 m
  (4000×4000×120 voxels) **while preserving the native 5 cm voxel size**, so the model sees the full grid
  extent instead of returning `UNKNOWN` beyond 50 m.
- **Caching:** inference runs once per frame and is cached to `data/cache/pred/<model>/<seq>/<frame>.npz`
  with full provenance (checkpoint SHA-256, repo commit, PyTorch/CUDA/driver versions, GPU) — everything
  downstream of caching runs with **no GPU required**.
- **Known, documented gap:** points outside the (extended) training crop are returned as `UNKNOWN, conf=0`
  rather than fed through the network (which would produce out-of-range voxel indices), which mechanically
  caps the 60–100 m accuracy bucket. This is reported, not hidden — see [§17](#17-known-limitations).

Full provenance, environment-setup notes (the upstream `torch==1.11.0+cu113` pin predates both development
GPUs and must be replaced with current CUDA-matched wheels), and the model-selection rationale live in
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## 7. The variable-resolution grid engine

The grid is a **nested clipmap of square rings**, anchored at the sensor, where every baseline (uniform grid)
is produced by the *same* engine as the foveated preset — so every comparison in this README uses identical
code paths, not a hand-rolled baseline.

- **Integer-only addressing** (`grid/engine.py`): all coordinates are quantised to millimetres once, and every
  subsequent cell lookup is floor-division on integers — no floating-point division ever decides which cell a
  boundary point falls into.
- **Exact partition, provably.** Ten invariants (I1–I10) are property-tested with `hypothesis` (≥200 examples
  each, including boundary-adversarial points at exact ring radii and cell-size multiples): point conservation,
  exact plane partition, nesting/alignment, fine→coarse bit-identical reduction, permutation invariance, a
  differential test against a naive dict-based reference, and sparse/dense accounting consistency.
- **Sparse accumulators** (`grid/accumulators.py`): occupied cells only (sorted flat indices + per-field
  arrays) — a dense `uniform_5cm` accumulator set would cost ~700 MB per frame; the sparse form never exceeds
  the frame's point count.
- **12-byte packed cells** (`grid/layers.py`): `ground_z`, `top_z`, `clearance` (int16, mm) · `cls` (uint8) ·
  `moving_frac` (uint8) · `count` (uint16) · `conf` (uint8) · `flags` (uint8: has-ground / has-obstacle /
  overhang / traversable / kerb / steep / low-clearance).
- **Backends:** a pure-NumPy reference backend is always the correctness oracle; an optional C++/pybind11
  backend (triggered only if NumPy exceeds its latency budget) must match it bit-for-bit (invariant I8).

| Preset | Rings | Logical cells | Allocated cells |
|---|---|---:|---:|
| `fovea_default` (production) | 5/10/20/40 cm @ 10/30/60/100 m | 910,000 | 1,130,000 |
| `ps_literal` | 5/50 cm @ 10/100 m | 318,400 | 320,000 |
| `uniform_5cm` | 5 cm, 0–100 m | 16,000,000 | 16,000,000 |
| `uniform_20cm` (iso-memory baseline) | 20 cm, 0–100 m | 1,000,000 | 1,000,000 |

---

## 8. Motion & object detection

Motion state is deliberately **not** inferred from the segmentation network — it is computed by a separate,
fully geometric, GPU-free module (`foveamap/motion/`) so the system degrades gracefully without a GPU and so
motion and semantics stay independently verifiable:

1. **Ego-motion compensation** — the previous scan is transformed into the current frame via the exact
   pose/calibration math (§5).
2. **Range-image residual** — a windowed `min |r − r_prev|` test per point against a range-scaled threshold
   `τ(r) = τ0 + τ1·r`, with disocclusion handling for receding objects.
3. **Range-scaled Euclidean clustering** (`scipy.spatial.cKDTree`) with `eps(r) = eps0 + eps1·r`.
4. **Cluster voting** — a cluster is `MOVING` if ≥ 30% of its points voted moving (≥ 3 points).
5. **Temporal hysteresis tracking** (`motion/tracker.py`) — nearest-neighbour association across frames with
   velocity gating, so object identity persists across short occlusions instead of flickering frame to frame.
6. **Output per object:** class, oriented bounding box (minimum-area rectangle via an angle sweep), centre,
   size, yaw, confidence, `moving: bool`, `speed_mps`, `safety_critical: bool` (pedestrians/cyclists/
   motorcyclists are always flagged regardless of instantaneous motion state).

This runs in pure NumPy/SciPy at roughly 5–15 ms per frame with 0 MB of GPU VRAM.

---

## 9. Derived layers & hazards

Computed per ring with a one-cell halo from the adjacent ring (continuous across resolution boundaries):

- **Slope** — central-difference gradient of ground height, masked where neighbours lack ground.
- **Kerb/step detector** — max ground-height discontinuity to 4-neighbours; separate thresholds distinguish a
  kerb from an obstacle-height step.
- **Overhang clearance** — vertical gap between ground and the lowest obstacle return; flags `low_clearance`.
- **3-state traversability** — `TRAVERSABLE` / `NON_TRAVERSABLE` / `UNKNOWN` (unobserved is never treated as
  free space).
- **Synthetic hazard injection** — since no public LiDAR dataset ships ground-truth potholes/kerbs/overhangs,
  FoveaMap injects them into real scans at a seeded `(range, bearing)` (pothole: lowered ground footprint;
  kerb: raised strip; overhang: synthetic slab at vehicle-clearance height) and measures detection rate per
  ring/distance/size with Wilson 95% confidence intervals — clearly labelled as an approximation everywhere
  it's reported (see [§15](#15-measured-results) for the measured rates, including where this fails).

---

## 10. Dashboard

A single-page React application (hash-routed, no page reloads) streamed live from the FastAPI/Socket.IO
backend:

| View | Shows |
|---|---|
| **Overview** | Live 2.5D spatial view (top-down & isometric), pipeline FPS/latency/active-cells/points KPIs |
| **Object Detection** | Detected objects list — class, distance, confidence, static/dynamic badge, velocity arrows |
| **Terrain Analysis** | Drivable/non-drivable/obstacle/unknown composition, traversability overlay |
| **Elevation Map** | Height-shaded ground/obstacle layer |
| **Performance** | Per-stage latency breakdown, FPS over time, memory over time |
| **Foveated vs Uniform** | Side-by-side memory/latency/cell-count comparison against the uniform baselines |
| **Settings** | Sequence/preset/mode selection, playback controls |

The map renderer composites one `<canvas>` texture per ring directly from the server's packed cell arrays
(`MapView.tsx`), with resolution-ring boundaries, per-cell inspection on hover, and object/vehicle overlays
drawn in a vehicle-centred, +x-forward-up convention matching SemanticKITTI.

---

## 11. Installation

### Prerequisites

| Requirement | Version | Needed for |
|---|---|---|
| Python | 3.10+ | Pipeline, tests |
| Node.js | 18+ | Dashboard |
| NVIDIA GPU + CUDA | current driver-matched PyTorch/`spconv`/`torch-scatter` | Live LSK3DNet inference only — grid/eval/dashboard run from the oracle or prediction cache with **no GPU** |
| CMake + a C++11 compiler | — | LSK3DNet's `c_gen_normal_map` extension (mandatory for live/cached inference); optional C++ grid backend |
| ~10 GB disk | — | SemanticKITTI sequence 08 (velodyne + labels + calib + poses) |

### 1. Clone and set up the Python environment

```bash
git clone https://github.com/KINGKK-007/SIH-2026.git
cd SIH-2026
python -m venv venv
# Windows:  venv\Scripts\activate       macOS/Linux:  source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # installs the `foveamap` package from src/
```

### 2. Set up the dashboard

```bash
cd src/dashboard
npm install
cd ../..
```

### 3. Verify the environment

```bash
python -m scripts.doctor          # or: make doctor
```

Prints exactly what's missing (Python/Node versions, data layout, weights, optional GPU/C++ backend) without
failing the parts that don't apply to what you're about to run.

### 4. Get the data

Register and download from [semantic-kitti.org](http://www.semantic-kitti.org/dataset.html) (labels) and the
[KITTI Odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (velodyne/calib/poses —
sequence **08** is enough for this project's scope). Extract into `data/dataset/sequences/08/` so the layout
matches:

```text
data/dataset/sequences/08/
├── velodyne/000000.bin …     # float32 x,y,z,remission
├── labels/000000.label …     # uint32 (lower 16 bits = semantic class)
├── calib.txt
├── times.txt
└── poses.txt
```

then verify:

```bash
python scripts/verify_data.py --sequences 08
```

### 5. (Optional, for live/cached model inference) LSK3DNet weights

Download the checkpoint referenced in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) into `configs/weights/`, and
build the vendored `c_gen_normal_map` C++ extension (see that doc for exact CUDA-version-matched wheel
commands — the upstream LSK3DNet `requirements.txt` pins a 2022-era `torch`/`spconv` build that predates
current GPUs, so install current wheels instead). **Oracle mode needs none of this** — it runs entirely from
ground-truth SemanticKITTI labels.

---

## 12. Running the project

### Backend + dashboard (recommended path — no GPU needed)

```bash
# Terminal 1 — backend (FastAPI + Socket.IO), oracle mode (ground-truth labels)
python -m foveamap.cli serve --mode oracle --sequence 08 --host 127.0.0.1 --port 8000

# Terminal 2 — frontend dev server
cd src/dashboard && npm run dev
```

Open the printed Vite URL (default `http://localhost:5173`). The backend also serves the production-built
dashboard directly from `/` if you run `npm run build` first.

### Model-backed modes

```bash
# From a prediction cache (no GPU needed once cached)
python -m foveamap.cli serve --mode cached --sequence 08 --model lsk3dnet

# Live inference (GPU required)
python -m foveamap.cli serve --mode live --sequence 08 --model lsk3dnet
```

### One-shot Makefile targets

```bash
make demo        # start backend + dashboard
make render       # render top-down PNG maps for sample frames
make benchmark     # regenerate every metric in results/
make test          # full test suite
```

---

## 13. CLI reference

Every command is available both as `python -m foveamap.cli <command>` and, once installed, as `foveamap <command>`.

| Command | Purpose |
|---|---|
| `inspect --sequence 08 --idx 0` | Print one scan's stats (point count, class histogram, pose) and write a bird's-eye scatter |
| `align --sequences 08 --pairs 50` | Frame-alignment verification of the pose/calibration math (median NN distance gate) |
| `stats --sequences 08` | Points and per-class counts per distance bucket |
| `render --mode oracle --sequence 08 --frames 0 1 2 --preset fovea_default` | Top-down 2.5D PNG render with the fovea overlay, ring boundaries, and cell-size labels |
| `memory --sequence 08 --idx 0` | Four-representation memory report (dense 3D, sparse 3D, uniform 2.5D, FoveaMap) |
| `evaluate --model lsk3dnet` | Streaming-confusion-matrix semantic evaluation of cached predictions on the locked sequence-08 interval |
| `serve --mode {oracle,cached,live} --sequence 08 --host --port` | Launch the FastAPI + Socket.IO dashboard server |

---

## 14. Configuration reference

Every threshold, radius, cell size and path is config-driven — never a magic number in code (`configs/*.yaml`):

| File | Governs |
|---|---|
| `grid.yaml` | Ring presets, `active_preset`, class-assignment thresholds, backend selection, `min_range_m` (ego-body exclusion) |
| `model.yaml` | Segmentation model family, LSK3DNet checkpoint/coverage-profile settings |
| `motion.yaml` | Frame gaps, residual thresholds (`τ0, τ1`), clustering radii, vote fractions — frozen after tuning on dev frames only |
| `derived.yaml` | Slope/kerb/clearance thresholds |
| `hazards.yaml` | Synthetic pothole/kerb/overhang injection parameters |
| `benchmark.yaml` | Experiment matrix for `make benchmark` |
| `dashboard.yaml` | Server host/port, playback FPS, texture format, theme |

---

## 15. Measured results

These numbers are generated by the repository's own evaluators (`results/*.json`), not hand-typed — regenerate
them with `make benchmark`. They are reported **as measured**, including where the system currently falls
short, per the project's own honest-reporting rule: negative results are not hidden.

### Memory (oracle mode, sequence 08, 30-frame sample)

| Representation | Measured size | vs. FoveaMap |
|---|---:|---:|
| Dense 3D voxel grid (theoretical) | 2,441 MB | 188.8× larger |
| Sparse occupied-only 3D | 0.26 MB | *smaller than FoveaMap* (occupied-voxel lower bound; no 2.5D semantic layers) |
| Uniform 2.5D @ 5 cm | 183 MB | 14.2× larger |
| **FoveaMap (`fovea_default`)** | **12.9 MB** | — |

The sparse-occupied-3D case is reported deliberately rather than hidden: it is a smaller *lower bound* because
it carries no packed semantic/height/confidence layers, only occupancy — FoveaMap's 12-byte-per-cell payload is
the fair comparison against the other three.

### Latency (oracle mode, CPU, 30-frame sample)

| Metric | Value |
|---|---:|
| Mean end-to-end | 876 ms |
| p95 end-to-end | 1,002 ms |
| Pipeline FPS | 1.1 |
| **Real-time (p95 < 100 ms)?** | **No** — reported honestly, not claimed |

This is a CPU, oracle-mode measurement (no GPU inference in the critical path); the motion and derived-layer
stages dominate. See [§17](#17-known-limitations) for the breakdown and what would close the gap.

### Object detection (20-frame sample)

| Metric | Value |
|---|---:|
| Recall | 78.3% |
| Precision | 83.8% |
| F1 | 81.0% |

### Synthetic hazard detection (750 injections, Wilson 95% CI)

| Hazard | Detection rate |
|---|---:|
| Kerb | 99.6% (CI 97.8–99.9%) |
| Overhang | — see `results/hazard_metrics.json` |
| **Pothole** | **1.6%** (CI 0.6–4.0%) — a known, reported weak spot |

### Motion (sequence-08 sample)

Point-level moving-class IoU measured **0.0** on this sample (zero true positives) — flagged here rather than
omitted; see [§17](#17-known-limitations) and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the geometric
motion detector's documented failure modes (radial/creeping motion, disocclusion artefacts, pose sensitivity).

---

## 16. Testing & quality gates

```bash
make test          # full suite
make test-fast      # quick subset, < 60 s, excludes @data tests
make lint            # ruff
make format          # black
make typecheck       # mypy (io/, grid/)
```

45+ test modules across the production package: `tests/grid` (property-based invariant proofs, ≥200 examples
each via `hypothesis`), `tests/io`, `tests/motion`, `tests/derived`, `tests/eval`, `tests/models`,
`tests/server` (Socket.IO protocol end-to-end), plus `tests/legacy` for the v0 reference implementation and
top-level config/doctor/smoke tests.

Frontend: `npm run typecheck`, `npm run lint` (oxlint), `npm run build`.

---

## 17. Known limitations

Reported honestly and in full in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md); the headlines:

- **Not real-time on this CPU-only dev setup** (§15) — the motion (DBSCAN) and derived-layer (step/kerb)
  stages dominate per-frame cost; a GPU-backed motion/derived path or the optional C++ grid backend are the
  documented paths to close this gap.
- **Geometric motion detector misses strictly-radial or creeping motion** (sub-~0.2 m/s), and is sensitive to
  ego-pose accuracy — a learned multi-scan moving-object-segmentation network is the documented upgrade path.
- **Pothole detection is weak** (1.6% on the synthetic-injection benchmark) — ground-return point density at
  typical pothole footprint sizes is the limiting factor; kerb and overhang detection are strong by contrast.
- **LSK3DNet accuracy beyond its native 50 m crop is model-limited**, not grid-limited — the 60–100 m bucket
  is mechanically capped by how much of that region the checkpoint was trained to see (§6).
- **Sequence 08 only** — no cross-sequence generalisation claim is made; the locked evaluation region is a
  held-out *system* test, not a held-out *model* test (§5).
- **No dedicated "wall" class in SemanticKITTI** — building/fence/vertical-structure returns are shown under
  the honest superclass `STATIC_OBSTACLE` rather than inventing a class the dataset doesn't have.

---

## 18. Roadmap

- GPU-accelerated / C++ grid and motion stages to close the real-time latency gap.
- Learned multi-scan moving-object-segmentation network to replace the purely geometric motion detector.
- Full `make benchmark && make report` orchestration regenerating every number in this README automatically
  (no hand-typed results, ever).
- Offline, deterministic demo video renderer (`viz/video.py`) independent of the live server/GPU/network.
- SLAM, path planning, closed-loop control, and multi-sensor fusion are explicitly **out of scope** — this
  project targets perception only.

---

## 19. Engineering decisions

Every deviation from the original plan is logged with a date, reason, and impact in
[`docs/DECISIONS.md`](docs/DECISIONS.md) (34 entries at time of writing) — including why LSK3DNet replaced the
originally scoped range-view model family, why the grid uses integer-millimetre addressing, why sequence 08
is split the way it is, and the frontend's two-mode (Terrain Analysis / Object Detection) information
architecture. The approved scope and decision baseline is
[`docs/LOCKED_IMPLEMENTATION_PLAN.md`](docs/LOCKED_IMPLEMENTATION_PLAN.md); day-by-day task status is in
[`docs/PHASES.md`](docs/PHASES.md) and [`docs/PROGRESS.md`](docs/PROGRESS.md).

---

## 20. Citations & acknowledgements

**Dataset**

```bibtex
@inproceedings{behley2019semantickitti,
  author    = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
  title     = {{SemanticKITTI}: A Dataset for Semantic Scene Understanding of LiDAR Sequences},
  booktitle = {Proc. of the IEEE/CVF International Conf. on Computer Vision (ICCV)},
  year      = {2019}
}
@inproceedings{geiger2012kitti,
  author    = {A. Geiger and P. Lenz and R. Urtasun},
  title     = {Are we ready for Autonomous Driving? The {KITTI} Vision Benchmark Suite},
  booktitle = {Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2012}
}
```

SemanticKITTI and KITTI Odometry are licensed **CC BY-NC-SA** (non-commercial research use); both are
downloaded separately per their own licence terms and are **not** redistributed in this repository.

**Segmentation model**

```bibtex
@inproceedings{feng2024lsk3dnet,
  title     = {{LSK3DNet}: Towards Effective and Efficient {3D} Perception with Large Sparse Kernels},
  author    = {Feng, Tuo and Wang, Wenguan and Ma, Fan and Yang, Yi},
  booktitle = {Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}
```

Source: [github.com/FengZicai/LSK3DNet](https://github.com/FengZicai/LSK3DNet) (MIT licence), vendored at the
repository root as `LSK3DNet-main/`. The label mapping in `foveamap/io/labels.py` is cross-checked against
[PRBonn/semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) (MIT), vendored at
`src/foveamap/io/semantic-kitti.yaml`.

**Core open-source dependencies**

NumPy · SciPy · PyTorch · FastAPI · `python-socketio` · `hypothesis` · `scikit-learn` · pydantic (backend) ·
React · Vite · TypeScript · `socket.io-client` · Lucide icons (frontend) — see `requirements.txt` and
`src/dashboard/package.json` for the full, versioned list.

---

## 21. License

FoveaMap's own code is released under the **[MIT License](LICENSE)**. Datasets (SemanticKITTI, KITTI Odometry)
and the vendored LSK3DNet checkpoint/source are governed by their respective upstream licences (see
[§20](#20-citations--acknowledgements) and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)) and are not
redistributed here.
