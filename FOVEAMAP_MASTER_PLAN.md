# FoveaMap — Master Plan & Agent Instructions

> **Foveated 2.5D LiDAR mapping: sharp where a mistake hurts, cheap where it doesn't.**
> Version 1.0 · 2026-09-20 · IIIT-B Sem 5 · GenAI class assignment 1

---

## 0. How to read this document (agents start here)

This file is self-contained: context, the problem statement, our proposed solution, how to build it in phases, the decisions a human must make, everything we are likely to miss, and future work. Read it fully before writing code.

**Tags used throughout**

| Tag | Meaning |
|---|---|
| **[MUST]** | Required for the repo to count as done |
| **[SHOULD]** | Core quality; do it unless time is gone |
| **[STRETCH]** | Only after all MUST/SHOULD are done |
| **[VERIFY]** | Came from web research or memory, not yet tested by us. Confirm before relying on it |
| **[HUMAN]** | Needs a human decision. See §10 |
| **[ASSUMPTION]** | We chose a value with no hard evidence; revisit if measurement disagrees |

**Ground rules**

1. **Do not push to any remote, create GitHub repos, publish pages/releases, or upload data without an explicit human "yes".** Local git init/commit is fine.
2. **Do not create accounts or type credentials.** KITTI/SemanticKITTI downloads require a login. The human does registration and downloads; the agent writes the scripts and instructions.
3. **Never hard-code result numbers in the README or docs.** Generate them from `results/*.json` so nothing goes stale.
4. **Never claim "real-time"** unless measured end-to-end p95 latency < 100 ms on stated hardware (the sensor runs at 10 Hz).
5. **Time-box every unknown** (model install, dataset access). When a time-box expires, take the documented fallback.
6. **Log every deviation** from this plan in `docs/DECISIONS.md` with a one-line reason.
7. Do not present validation-set numbers as leaderboard numbers.

**Startup protocol**

1. Read this whole file.
2. Ask the human the *blocking* decisions from §10 in **one batched question** (recommended default listed first for each). Use the default for anything unanswered.
3. Record answers in `docs/DECISIONS.md`.
4. Start Phase 0 (§9). Ask the remaining §10 decisions just-in-time at the phase where they apply, and stop at each **Human checkpoint** listed in §9.

---

## 1. Context

- **Setting:** class assignment for the GenAI course at IIIT Bangalore (semester 5). The deliverable is a **GitHub repo containing a working prototype** for the problem statement below.
- **Audience:** the people who wrote the problem statement (PS) and will grade or review the repo. They see many submissions. Ours must (a) run, (b) look good in the first 30 seconds, (c) make technical sense when questioned.
- **Known constraints:** limited time (exact deadline unknown, **[HUMAN]** H1); dev machine is Windows 11 with no confirmed NVIDIA GPU (**[HUMAN]** H2); prototype, not production.
- **Not known:** the grading rubric. §3 infers it.
- **Working directory when this was written:** `C:\Users\SPARS\Desktop\IIIT B\sem5\GenAI\classAssign1` (not a git repo yet; contains unrelated files from other assignments, leave them alone).

---

## 2. Problem statement (verbatim, encoding repaired)

**Background:**

Autonomous navigation depends on the ability of a vehicle to perceive its surroundings with high precision. While 3D Lidar point clouds provide rich spatial data, processing millions of points in real-time creates immense computational bottlenecks and memory latency. Conversely, standard 2D occupancy grids lose critical height information necessary for detecting curbs, potholes, or overhanging obstacles. To balance precision and performance, there is a need for a 'foveated' mapping approach—similar to human vision—where the immediate vicinity is rendered in high detail for safety, and distant areas are simplified to reduce the processing load.

**Description:**

The goal is to build a deep learning pipeline that transforms raw Lidar point clouds into a variable resolution 2.5D grid (an elevation map with semantic layers). The system must perform three primary tasks:

1. **Terrain Analysis:** Distinguish between drivable surfaces and non-drivable terrain.
2. **Object Detection:** Identify and classify static obstacles (walls, poles) and dynamic objects (pedestrians, other vehicles).
3. **Adaptive Spatial Representation:** Implement a non-uniform grid where the cell size increases as the distance from the sensor increases. This requires a sophisticated data structure that can handle variable resolution without causing alignment errors or data loss during the projection from 3D to 2.5D.

**Expected Solution:** A software framework consisting of:

- **A Deep Learning Model:** A network (e.g., PointNet++ or a Sparse Convolutional Neural Network) capable of semantic segmentation of point clouds into terrain, static obstacles, and moving objects.
- **Variable Resolution Grid Engine:** An algorithm that projects classified 3D points into a 2.5D grid where the resolution is high (e.g., 5cm cells) within a 10m radius and decreases (e.g., 50cm cells) up to a 100m radius.
- **Real-time Visualization:** A dashboard showing the 2.5D map with distinct color-coding for terrain and objects, demonstrating a significant reduction in memory usage compared to a uniform high-resolution 3D map.
- **Performance Metrics:** Evidence of low latency (high FPS) and high accuracy in object classification across varying distances.

---

## 3. What the PS setters most likely expect

| PS sentence | What a reviewer probably looks for | Our answer |
|---|---|---|
| "deep learning pipeline… raw point clouds → variable-resolution 2.5D grid" | One command from raw `.bin` scans to a map, with a real network in the loop | `foveamap run` on SemanticKITTI; pretrained SOTA range-view net |
| Terrain analysis | Drivable vs non-drivable, visible on the map | Class taxonomy (§5.1) + derived traversability layer (§5.6) |
| Object detection, static vs dynamic | Classes *and* motion state; objects, not just coloured dots | Semantic labels + motion flag + clustered object boxes |
| Adaptive grid "without alignment errors or data loss" | Proof, not a claim | Integer-multiple nested cells, cell-based ring assignment, conservation and nesting **unit tests** (§5.5) |
| Network "e.g. PointNet++ or Sparse CNN" | A recognised architecture and a sensible choice | Range-view net (FRNet family); justified against PointNet++/sparse CNN in §5.3 |
| 5 cm within 10 m, 50 cm to 100 m | Our config reproduces (or improves on) this | `ps_literal_2ring` preset is always evaluated; default is a 4-ring sensor-matched variant |
| Dashboard, colour coding, memory reduction vs uniform 3D | Visual, interactive, with a memory number | Live dashboard with memory meter against four baselines (§7) |
| Latency/FPS, accuracy across distances | Tables and plots split **by range bucket**, honest latency method | §6 experiments E1–E8 |

**Unstated but nearly certain:** a README quickstart that works; reproducible numbers; a visible limitations section (reviewers trust honest work); clean structure; a demo asset (GIF/video) at the top of the README.

---

## 4. Proposed solution — FoveaMap

### 4.1 Pitch

FoveaMap turns each LiDAR scan into a **foveated 2.5D map**: a set of nested, aligned grids whose cell size grows with distance (5 cm → 10 → 20 → 40 cm). Every cell stores ground height, obstacle height, overhang clearance, semantic class, a moving flag and confidence. A **pretrained range-view network** labels every point; a light **motion module** marks movers; the **grid engine** projects labelled points into the nested grid with provable alignment; a **dashboard** shows the map live next to a memory meter, a latency bar and an accuracy-vs-distance curve.

Central insight: **the fovea is not only cheaper, it matches the sensor.** A 64-beam LiDAR's point spacing at 10 m is a few cm; at 100 m it is roughly 0.3 m horizontally and 0.7 m vertically **[VERIFY]**. 5 cm cells at long range are mostly empty; 40–50 cm cells sit near the physical sampling limit. We pick ring sizes from that physics and show it in a plot (E7).

### 4.2 Pipeline

```
 SemanticKITTI scans (.bin, 10 Hz) + poses + calib
        │
        ▼
 [P1] Data layer ── poses→velodyne frame, GT labels, range buckets
        │
        ├──────────────────────────────┐
        ▼                              ▼
 [P3] Segmenter (pretrained)      [oracle mode] GT labels
   FRNet / CENet / SalsaNext        (isolates grid loss from model error)
        │   per-point class + confidence
        ▼
 [P4] Motion flag (scan-to-scan residual, cluster vote) + object clustering → boxes
        │
        ▼
 [P2] Foveated grid engine (nested clipmap, 4 rings, square, aligned)
        │   layers: ground_z, top_z, overhang_z, class, dyn_frac, count, conf
        ▼
 [P5] Derived layers: slope, step (kerb), clearance, traversable
        │
        ▼
 [P6/P7] Benchmarks (memory · latency · accuracy-by-range) + Dashboard + demo video
```

### 4.3 Four design ideas that make it credible

1. **Oracle-first modular build.** Every stage speaks the same interfaces (§8). The grid engine and dashboard are built and tested on **ground-truth labels first**, so a working demo exists on day 2 before any network is installed. Swapping in the network later changes one line. It also separates "loss caused by coarsening" from "loss caused by the model".
2. **Predictions are cached to disk** in SemanticKITTI `.label` format. A GPU is needed once (a batch job on Colab/Kaggle/any GPU); grid engine, evaluation and dashboard then run on any laptop. GT and predicted labels share one file format, so one evaluator handles both.
3. **Pretrained SOTA, not from-scratch training.** LiDAR semantic segmentation on SemanticKITTI is a saturated benchmark. Effort goes to the part with no off-the-shelf solution (the foveated grid) and, optionally, one small genuine training contribution (range-aware fine-tune, S1).
4. **A trade-off explorer answers the PS's hidden question:** "does blurring the distance hurt accuracy?" We sweep grid presets, plot memory against accuracy-per-range, and report the answer whichever way it comes out.

### 4.4 The "attention" list (what the demo shows)

1. **Fovea overlay:** ring boundaries drawn on the map, each labelled with its cell size; toggle to show real cell edges.
2. **Memory meter:** log-scale bars for dense 3D 5 cm / sparse 3D 5 cm / uniform 2.5D 5 cm / FoveaMap, updating per frame.
3. **Latency bar** by stage against a 100 ms "real-time @ 10 Hz" line, plus live FPS.
4. **Accuracy-vs-distance curve:** mIoU per range bucket, uniform-5 cm vs foveated.
5. **Fovea slider** (SHOULD): drag ring radii/cell sizes; map, memory and accuracy update instantly.
6. **Zoom lens:** the same kerb rendered at 5 cm and at 40 cm, showing why the fovea sits near the car.
7. **Object boxes** with class chips; movers get a velocity arrow (STRETCH: arrows need tracking).
8. **Hazard injection:** a button inserts a synthetic pothole/kerb/overhang; show which ring detects it.
9. **Zero-install proof:** README hero GIF, pre-rendered demo video, and (STRETCH) Colab notebook / static replay page.

---

## 5. Technical design

### 5.1 Data and class taxonomy

**Dataset:** SemanticKITTI (KITTI Odometry velodyne + SemanticKITTI labels). Real Velodyne HDL-64E scans at 10 Hz, 360°, ~120k points/scan, sequences 00–10 labelled (08 = standard validation), 11–21 test with hidden labels. Provides per-point semantic labels, instance ids, moving/static variants, and per-sequence poses. **[VERIFY]** exact scan counts (seq 08 has 4071 scans as far as we know).

**Label file format:** `uint32` per point: low 16 bits = semantic id, high 16 bits = instance id.

**Raw SemanticKITTI ids we use** (confirm against `config/semantic-kitti-all.yaml` in [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) **[VERIFY]**):

| Group | Raw ids |
|---|---|
| ignore | 0 unlabeled, 1 outlier |
| vehicles (static) | 10 car, 11 bicycle, 13 bus, 15 motorcycle, 16 on-rails, 18 truck, 20 other-vehicle |
| vulnerable road users | 30 person, 31 bicyclist, 32 motorcyclist |
| ground | 40 road, 44 parking, 48 sidewalk, 49 other-ground, 60 lane-marking, 72 terrain |
| structures/objects | 50 building, 51 fence, 52 other-structure, 70 vegetation, 71 trunk, 80 pole, 81 traffic-sign, 99 other-object |
| **moving** variants | 252 moving-car, 253 moving-bicyclist, 254 moving-person, 255 moving-motorcyclist, 256 moving-on-rails, 257 moving-bus, 258 moving-truck, 259 moving-other-vehicle |

The official 19-class benchmark **merges moving into static** classes. So a stock pretrained segmentation checkpoint does *not* output "moving". Motion must come from the motion module (§5.4). **[VERIFY]** the chosen checkpoint's label space.

**PS-aligned super-classes (default mapping, [HUMAN] H6):**

| Super-class | Colour (dark theme) | Members |
|---|---|---|
| `DRIVABLE` | teal `#1FB6A6` | road, parking, lane-marking |
| `NON_DRIVABLE_TERRAIN` | slate `#5B6B7A` | sidewalk, terrain, other-ground |
| `STATIC_OBSTACLE` | amber `#F5A524` | building, fence, other-structure, vegetation, trunk, pole, traffic-sign, other-object, **non-moving** vehicles |
| `DYNAMIC` | red `#FF4D6D` | moving-* vehicles, and person/bicyclist/motorcyclist always (a standing pedestrian is still safety-critical) |
| `UNKNOWN` | near-black | unobserved cells, ignore labels |

Every cell also carries a boolean `moving` and (SHOULD) a `vru` flag so the dashboard can distinguish "moving car" from "standing pedestrian".

### 5.2 Frames, poses, calibration (a real trap)

- Scans are in the **Velodyne frame** (x forward, y left, z up). Sensor height ≈ 1.73 m, so flat ground sits near z ≈ −1.73 **[VERIFY]**.
- `poses.txt` is in the **camera-0 frame**. To get Velodyne-frame poses: `T_velo(t) = Tr⁻¹ · pose(t) · Tr`, with `Tr` from `calib.txt`. Skipping this silently produces wrong motion residuals.
- Ego-motion comes from the shipped `poses.txt`. KISS-ICP (`pip install kiss-icp`, MIT) is only needed if we later run on unlabelled/own data.
- **Sanity test:** transform scan t−1 into frame t; static structure (walls, poles) must overlap visually and by nearest-neighbour residual.

### 5.3 Segmenter (the deep-learning model)

**Choice: a range-view network.** Evidence that it is the only family that meets real-time on embedded hardware ([Are We Ready for Real-Time LiDAR Semantic Segmentation?, arXiv 2410.08365](https://arxiv.org/abs/2410.08365)):

| Model (family) | SemanticKITTI mIoU | Latency RTX 4090 / Jetson AGX Orin |
|---|---|---|
| SalsaNext 1024 (range-view) | 54.4 | 30 / 84 ms |
| SalsaNext 2048 (range-view) | 55.9 | 36 / 109 ms |
| MinkNet (sparse conv) | 64.3 | 71 / 211 ms |
| SPVCNN (voxel+point) | 65.3 | 85 / 252 ms |
| WaffleIron 48-256 (point) | 65.8 | 370 / 1847 ms |

(Their numbers use their own training protocol; ours will differ. Sparse-conv methods are far from real time on embedded hardware; preprocessing is often the hidden cost.)

**Candidate models and time-boxed fallback chain** (spike **first**, at most 3 h per candidate; record outcome in `docs/DECISIONS.md`):

| Order | Model | Why | Notes [VERIFY all] |
|---|---|---|---|
| 1 | [FRNet](https://github.com/Xiangxu-0103/FRNet) (Apache-2.0) | Best accuracy/speed range-view net (paper: 73.3 mIoU SemanticKITTI test); checkpoints on Google Drive per README; README table lists 29.1 FPS (FRNet) / 33.8 FPS (Fast-FRNet), GPU unspecified | Built on MMDetection3D → version-pinning pain, Linux-friendly, hard on native Windows |
| 2 | [CENet](https://github.com/huixiancheng/CENet) (MIT) | Plain PyTorch, pretrained weights on Google Drive, `infer.py` provided | Needs kNN post-processing (time it!); lower mIoU than FRNet |
| 3 | [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext) | Classic, simple | Check weights are public |
| 4 | MMDetection3D model zoo (Cylinder3D / MinkUNet / SPVCNN) | Fallback accuracy | Slow, not real-time; reported only as "accuracy ceiling" |
| 5 | Oracle (GT labels) | Never blocks the project | Must be labelled "oracle" everywhere it is used |

**Why not PointNet++ (the PS's example)?** It is a classic baseline but scores roughly 20 mIoU on SemanticKITTI in the original benchmark paper **[VERIFY]** and does not scale to 120k-point scans in real time. We state this in the README and use it as the reason for choosing a modern architecture.

**Checkpoint-leakage check [MUST]:** confirm the chosen checkpoint was **not** trained on sequence 08. If it was, evaluate on a different labelled sequence or clearly flag the numbers as optimistic.

**Optional teacher [STRETCH]:** [Point Transformer V3](https://arxiv.org/pdf/2312.10035) via Pointcept (~75 mIoU SemanticKITTI reported) as an accuracy ceiling or distillation teacher.

**Optional training contribution [STRETCH, H11]:** *range-aware fine-tune*: continue training the range-view net with a per-point loss weight that grows with distance, and compare per-range mIoU before/after. Needs SemanticKITTI train sequences (large download) and a GPU for a few hours.

### 5.4 Motion state and object instances

**Default (MVP/Core) — geometric, no extra network:**

1. Pose-compensate scan t−k (k = 1 and 3–5, i.e. 0.1–0.5 s back) into frame t.
2. Build range images; per-point **range residual** `(r_prev_projected − r_cur) / r_cur` (the LMNet residual-image idea).
3. Candidate mover = point of a *movable class* (vehicle, VRU) with residual above threshold.
4. **Cluster vote:** Euclidean/DBSCAN clustering of movable-class points (`scipy.spatial.cKDTree` connected components, eps ≈ 0.5–0.8 m). A cluster is `moving` if > X % of its points are candidates. This suppresses ground/edge noise.
5. Output per object: class (majority vote), BEV oriented box (`cv2.minAreaRect`), height, moving flag.

**Known limits (document them):** slow pedestrians (≈1.4 m/s → 14 cm between scans) are hard to separate from noise beyond ~20 m at k = 1; use larger k for far objects. Evaluate honestly against GT `moving-*` labels (E5).

**Upgrade [STRETCH, H5]:** [4DMOS](https://github.com/PRBonn/4DMOS) (MIT, pretrained SemanticKITTI weights, multi-scan sparse conv). Requires **MinkowskiEngine** (effectively Linux-only) → run in Colab/WSL2 and cache predictions.

**Tracking for velocity arrows [STRETCH]:** nearest-neighbour association of cluster centroids across frames (after ego-compensation), gate 2 m.

### 5.5 Variable-resolution grid engine (the core deliverable)

**Structure: nested "clipmap" of square rings, all anchored at the ego origin.**

Default preset `fovea_4ring`:

| Ring | Covers (half-extent, Chebyshev) | Cell | Cells in ring |
|---|---|---|---|
| 1 | 0–10 m | 5 cm | 160,000 |
| 2 | 10–30 m | 10 cm | 320,000 |
| 3 | 30–60 m | 20 cm | 270,000 |
| 4 | 60–100 m | 40 cm | 160,000 |
| **Total** | | | **910,000** |

Uniform 5 cm over the same 200 m × 200 m square = **16,000,000 cells** → **17.6× fewer cells** (≈ 128 MB → 7.3 MB at 8 B/cell). Circular rings give the same ratio. The number is computed by code from the config, never typed by hand.

**Presets to implement and always evaluate:**

| Preset | Rings | Purpose |
|---|---|---|
| `uniform_5cm` | single 5 cm | The "high-resolution baseline" the PS compares against |
| `uniform_20cm` | single 20 cm | Cheap uniform baseline |
| `ps_literal_2ring` | 5 cm ≤ 10 m, 50 cm to 100 m | The PS example, verbatim |
| `fovea_4ring` (**default**) | 5/10/20/40 cm | Sensor-matched, power-of-two nesting |

**Alignment invariants (this is the "no alignment errors / no data loss" answer):**

- **I1. Integer nesting:** each coarser cell size is an exact integer multiple of every finer one (5→10→20→40 ✔; `ps_literal_2ring` 5→50 ✔; **5/10/20/50 ✘** because 50/20 is not an integer, so avoid it).
- **I2. Cell-based ring assignment, not point-based:** a point belongs to the ring whose square annulus contains it, using half-open intervals `[a, b)`. Ring boundaries are multiples of the *outer* ring's cell size (10 m /0.10 ✔, 30 m /0.20 ✔, 60 m /0.40 ✔, 100 m /0.40 ✔), so no coarse cell straddles two rings.
- **I3. One point → exactly one cell.** Index with `floor((x + H) / cell)` in float64 with a rounding guard for exact multiples.
- **I4. Additive statistics reduce exactly:** aggregate counts/sums at the finest resolution and block-reduce upward, or bin directly; both must agree bit-for-bit for count, sum, min, max.

**Required tests [MUST]** (`tests/test_grid_invariants.py`):

| Test | Assertion |
|---|---|
| T1 conservation | Σ counts over all rings == number of in-range points; no point counted twice |
| T2 nesting | 5 cm binning block-reduced to 10/20/40 cm == direct binning at those sizes (count, sum, min, max) |
| T3 boundary | Points at exactly ±10.0, ±30.0, ±60.0, ±100.0 m and at −0.0 land in exactly one deterministic cell |
| T4 round trip | cell centre → world → cell index returns the same cell, every ring |
| T5 partition | Ring masks tile the plane with no overlap and no gap |
| T6 (STRETCH) scroll | Shifting ego by an exact multiple of the coarsest cell shifts the map without changing cell contents |

**Layers per cell** (default 8 bytes/cell; dtypes are part of the memory metric):

| Layer | Definition |
|---|---|
| `ground_z` | Median z of points labelled ground (drivable/non-drivable terrain); NaN if unobserved |
| `top_z` | Max z of non-ground points in the cell |
| `overhang_z` | Lowest z of points more than 0.3 m above local ground (gives bridge/branch clearance) |
| `cls` | Super-class (uint8), aggregation rule below |
| `dyn_frac` | Fraction of points flagged moving/VRU (uint8, 0–255) |
| `count` | Point count (uint8, saturating) |
| `conf` | Mean network confidence (uint8) |
| `flags` | bit0 moving, bit1 vru, bit2 observed, bit3 traversable |

**Class aggregation rule ([HUMAN] H8):**
- `safety_priority` (default): `DYNAMIC` if ≥ `min_dyn` points; else `STATIC_OBSTACLE` if ≥ `min_obs` points rise > 0.15 m above local ground; else majority of ground classes; else `UNKNOWN`. Small hazards (a 15 cm pole in a 40 cm cell) survive coarsening.
- `majority`: plain mode. Report both in E4.

**Implementation:** numpy + `numba` (`@njit(parallel=True)` scatter-reduce) first; optional torch/CuPy GPU path [STRETCH]. Target ≤ 15 ms per frame for ~120k points on CPU **[ASSUMPTION]**. Reference designs to borrow ideas from: [elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) (layer model, overhang handling), geometry clipmaps (nested rings), [wavemap](https://arxiv.org/pdf/2306.01279) (multi-resolution mapping), [NUC-Net](https://arxiv.org/abs/2505.24634) (non-uniform radial partition inside a network). We use square rings by default; `ring_shape: circular` (cell-centre rule) is an option **[HUMAN]** H7.

**Uniform baselines use the same engine with one ring**, so comparisons are like-for-like.

### 5.6 Derived layers (plain geometry, no learning)

- `slope`: gradient of `ground_z` over 3×3 neighbours (per ring, using that ring's cell size).
- `step`: max |`ground_z` difference| to 8 neighbours → **kerb detector** (a ~10–15 cm kerb needs the 5–10 cm rings; at 40 cm cells the step is averaged away, which is exactly what the zoom lens shows).
- `clearance`: `overhang_z − ground_z`; passable if greater than the vehicle height parameter (default 2.0 m, [ASSUMPTION]).
- `traversable`: drivable class ∧ slope < θ ∧ step < s ∧ no obstacle ∧ clearance ok. Thresholds in config.
- **Ring-seam handling:** neighbour differences across a ring boundary use the finer ring resampled by block-mean to the coarser cell size (test T2 guarantees consistency).

**Hazards the dataset cannot validate:** SemanticKITTI has no pothole/kerb-height ground truth. Use **synthetic hazard injection**: lower ground-point z by 8–15 cm inside a disc (pothole), add a 12 cm step along a line (kerb), add a slab at 1.5 m height (overhang), then measure detection rate per ring and per distance. State plainly that this is an approximation (points keep their x,y; no ray re-simulation).

### 5.7 Memory accounting (E3) — four baselines, no strawmen

| Representation | How measured |
|---|---|
| Dense 3D uniform 5 cm | Analytic: (200 m / 0.05)² × (z_range / 0.05) voxels × 1 B. ≈ 1.9 × 10⁹ voxels ≈ 1.9 GB for a 6 m z-range **[ASSUMPTION]** |
| Sparse 3D uniform 5 cm | **Measured** per scan: occupied voxels × (12 B coords + 1 B label) |
| Uniform 2.5D 5 cm | `uniform_5cm` preset: cells × bytes/cell (dense array) |
| **FoveaMap** | `fovea_4ring`: allocated **and** active cells × bytes/cell |

Report both *allocated* and *active* cells for FoveaMap (dense per-ring arrays allocate the full square; masked inner regions are inactive). Also show **raw point cloud size** (~2 MB/scan) for honesty: a single foveated map is not smaller than one raw scan; its value is a **fixed-size, planner-ready, time-aggregatable** structure, whereas accumulating raw points grows without bound (see FAQ §13).

### 5.8 Latency budget and measurement protocol (E2)

**Target [ASSUMPTION]:** end-to-end p95 ≤ 100 ms (sensor period). Stage budgets to design toward: preprocess ≤ 10, network ≤ 40, post-process ≤ 10, motion ≤ 10, clustering ≤ 5, grid ≤ 15, derived layers ≤ 5.

**Protocol [MUST]:** batch size 1; 50 warm-up frames discarded; `torch.cuda.synchronize()` around GPU timing; report p50/p95/p99 per stage; **include preprocessing and post-processing** (kNN, range projection); report disk I/O separately; exclude rendering; state CPU/GPU model, driver, torch version. Also report throughput (max FPS) separately from real-time latency.

---

## 6. Evaluation plan

Default eval subset: **every 5th scan of sequence 08** (~800 frames) **[HUMAN]** H17. Latency runs use all frames of a contiguous 500-frame clip.

**Range buckets (aligned with the fovea rings): [0,10) [10,30) [30,60) [60,100] m.** Range = √(x²+y²) in the Velodyne frame. Every metric below is reported per bucket **with its support** (points and GT instances in that bucket) so thin far-range numbers are not over-read.

| ID | Experiment | Metric | Output artifact |
|---|---|---|---|
| E1 | Point-level semantic accuracy | mIoU / per-class IoU, 4 super-classes and 19 classes, per range bucket | `results/e1_semantic.json`, plot |
| E2 | Latency and FPS | per-stage p50/p95/p99, e2e FPS, on each hardware tried | `results/e2_latency.json`, stacked-bar plot |
| E3 | Memory | cells, allocated/active bytes for 4 baselines, per-scan (mean ± sd) | `results/e3_memory.json`, bar plot |
| E4 | **Cost of coarsening** (oracle labels through each preset) | cell-level class agreement vs uniform-5 cm; ground-height RMSE vs uniform-5 cm; small-object survival rate (poles/pedestrians remain non-empty obstacle cells); fill rate (% observed cells) per ring | `results/e4_coarsening.json`, plot |
| E5 | Motion | moving-object IoU vs GT `moving-*` labels, per range; false-move rate on static vehicles | `results/e5_motion.json` |
| E6 | Instance recall/precision | GT instance detected if ≥ 50 % of its points get the right super-class and it forms a cluster; per range | `results/e6_instances.json` |
| E7 | Sensor-physics justification | point density per m² vs range, with ring cell sizes overlaid | `docs/figs/density_vs_range.png` |
| E8 | Hazard injection | detection rate of synthetic pothole/kerb/overhang by ring and distance | `results/e8_hazards.json` |
| E9 | Pareto sweep | memory vs (E1/E4 accuracy per range) across presets and ring-radius grid | `results/e9_pareto.json`, plot |

**Interpretation rules:** E1 mixes network error with distance sparsity; E4 uses oracle labels so it isolates the grid's contribution. Present both, side by side.

---

## 7. Dashboard and visual spec

**Layout (dark theme, 16:9):**

```
┌───────────────────────────────────────────────┬───────────────────────────┐
│  FOVEA MAP (top-down 2.5D)                    │  MEMORY (log bars)        │
│  hill-shaded by ground_z, class-coloured,     │  dense3D ▮▮▮▮▮▮▮▮ 1.9 GB  │
│  ring outlines + labels 5/10/20/40 cm,        │  sparse3D ▮▮▮▮ 21 MB      │
│  object boxes + chips, ego car marker,        │  uniform2.5D ▮▮▮ 128 MB   │
│  [x] cell edges  [x] traversable overlay      │  FoveaMap ▮ 7.3 MB        │
│                                               ├───────────────────────────┤
│                                               │  LATENCY (stacked) | 100ms│
│                                               │  FPS: 24.7  p95: 71 ms    │
├───────────────────────────┬───────────────────┼───────────────────────────┤
│  ZOOM LENS: kerb 5cm vs   │  3D VIEW (optional)│  mIoU vs DISTANCE         │
│  same kerb at 40cm        │  extruded obstacles│  uniform vs foveated      │
├───────────────────────────┴───────────────────┴───────────────────────────┤
│  Controls: ▶ play  frame slider  fovea slider  hazard-inject  model/preset  │
└───────────────────────────────────────────────────────────────────────────┘
```
(All numbers in the sketch are placeholders; the app reads real values from the pipeline.)

**Colour semantics:** as §5.1. Unobserved cells stay dark, never interpolated silently; provide an "interpolate for display" toggle that is clearly labelled.

**Tech options ([HUMAN] H9):**

| Option | Effort | Look | Notes |
|---|---|---|---|
| **A. Streamlit + Plotly + rendered map images** (default) | Low | Good with custom CSS | Python-only team; `st.empty()` loop reaches ~10 fps display. Display FPS ≠ pipeline FPS; report the headless number |
| B. Rerun (rerun.io) | Low | Excellent 3D | Viewer rather than a dashboard; limited custom panels |
| C. FastAPI + WebSocket + JS canvas/Chart.js | Medium-high | Best | Needs a front-end-capable teammate |
| D. Pre-rendered MP4/GIF only | Lowest | Fixed | Always produce this as the safety net regardless |

**Always ship [MUST]:** an offline renderer (`foveamap render`) that writes the composed dashboard frames to an MP4 and a short GIF for the README, so the demo survives any live-demo failure.

**Demo storyboard (60–90 s clip):** (1) scan as raw dots, (2) labels appear, (3) map builds with ring overlay, (4) memory meter drops from GB to MB, (5) pedestrian/car boxes, mover turns red, (6) zoom-lens kerb, (7) accuracy-vs-distance curve, (8) FPS/latency bar. Pick a sequence segment that has moving cars, a pedestrian and a kerb (auto-select by scanning GT for those, then confirm with the human, H10).

---

## 8. Repository layout, interfaces, config, CLI

```
foveamap/
├─ README.md                    # hero GIF, TL;DR table (auto-generated), quickstart, limitations, FAQ
├─ LICENSE                      # human decides (H13); note KITTI data licence separately
├─ NOTICE.md                    # data + model attributions and licences
├─ pyproject.toml
├─ configs/
│  ├─ default.yaml
│  └─ grids/{uniform_5cm,uniform_20cm,ps_literal_2ring,fovea_4ring}.yaml
├─ data/                        # .gitignored; README with download steps
├─ src/foveamap/
│  ├─ io/{kitti.py,labels.py,poses.py}
│  ├─ models/{base.py,frnet.py,cenet.py,oracle.py,cache.py}
│  ├─ motion/{residual.py,cluster.py}
│  ├─ grid/{spec.py,clipmap.py,aggregate.py,layers.py,baselines.py}
│  ├─ derive/{traversability.py,hazards.py}
│  ├─ eval/{metrics.py,ranges.py,latency.py,memory.py}
│  ├─ viz/{palette.py,render.py,app.py}
│  └─ cli.py
├─ scripts/{check_data.py,cache_predictions.py,run_benchmarks.py,make_readme_tables.py}
├─ tests/                       # test_grid_invariants.py, test_labels.py, test_poses.py, test_memory.py
├─ notebooks/                   # exploration + Colab entry (STRETCH)
├─ docs/{DECISIONS.md,design.md,figs/}
├─ results/                     # committed: small JSON/PNG/MP4/GIF only
└─ .github/workflows/ci.yml     # ruff + pytest (CPU, synthetic data)
```

**Key interfaces** (fix these on day 1 so tracks can work in parallel):

```python
@dataclass
class SegOutput:
    label: np.ndarray        # (N,) uint16 raw SemanticKITTI-style ids (or 19-class ids + mapping)
    conf: np.ndarray         # (N,) float16, 0-1

class Segmenter(Protocol):
    name: str
    def __call__(self, points: np.ndarray) -> SegOutput: ...   # points (N,4): x,y,z,intensity, velodyne frame

@dataclass(frozen=True)
class Ring:
    cell_m: float
    half_extent_m: float     # outer bound of this square ring; inner bound = previous ring's

@dataclass(frozen=True)
class GridSpec:
    rings: tuple[Ring, ...]
    z_range_m: tuple[float, float]
    aggregation: str         # "safety_priority" | "majority"

class ClipmapGrid:
    @classmethod
    def build(cls, points, super_class, moving, conf, spec) -> "ClipmapGrid": ...
    def layer(self, name: str, ring: int) -> np.ndarray: ...
    def stats(self) -> dict:      # allocated_cells, active_cells, bytes
```

**Example config**

```yaml
grid:
  preset: fovea_4ring
  rings:                      # square, nested, anchored at ego origin
    - {cell_m: 0.05, half_extent_m: 10}
    - {cell_m: 0.10, half_extent_m: 30}
    - {cell_m: 0.20, half_extent_m: 60}
    - {cell_m: 0.40, half_extent_m: 100}
  z_range_m: [-3.0, 5.0]      # velodyne frame; ground near -1.73
  max_range_m: 100
  aggregation: safety_priority
  min_obs_points: 2
  min_dyn_points: 2
segmenter: {name: frnet, checkpoint: ckpt/frnet_semkitti.pth, cache_dir: preds/frnet}
motion: {enabled: true, k_scans: [1, 3], residual_thresh: 0.1, cluster_eps_m: 0.6, vote_frac: 0.3}
vehicle: {height_m: 2.0, max_slope_deg: 15, max_step_m: 0.10}
```
(Threshold values above are starting points [ASSUMPTION]; tune on a few training-sequence frames, never on the eval subset.)

**CLI**

```
foveamap check-data   --root data/semkitti
foveamap cache        --seq 08 --model frnet          # GPU job → preds/
foveamap run          --seq 08 --start 0 --n 300 --labels {gt,pred} --grid fovea_4ring
foveamap bench        --seq 08 --stride 5 --presets all   # writes results/*.json
foveamap render       --seq 08 --start 1200 --n 600 --out results/demo.mp4
foveamap dashboard    # Streamlit app
```

---

## 9. Implementation phases

Parallel tracks for a team of 2–3 (interfaces in §8 are frozen on day 1): **A** = P1, P2, P5 · **B** = P3, P4 · **C** = P6 scaffolding, P7. Integration checkpoint after P2 + P3.

| Phase | Work | Acceptance criteria | Est. (one person) |
|---|---|---|---|
| **P0 Setup & decisions** | Local `git init`; venv; repo skeleton; `docs/DECISIONS.md`; batched §10 decisions; human downloads data; **start the model spike immediately** (highest install risk) | `foveamap check-data` loads a scan + labels from seq 08 | 0.5 d |
| **P1 Data layer** [MUST] | `.bin` reader, label split (semantic/instance), raw→super-class maps, poses→velodyne frame via `Tr`, range util, quick BEV plot coloured by GT | Unit tests on label maps; consecutive-scan alignment check passes | 0.5 d |
| **P2 Grid engine on oracle labels** [MUST] | `GridSpec`, clipmap binning, layers (§5.5), uniform baselines via single ring, memory accounting, tests T1–T5, top-down renderer with ring overlay | Tests green; ≤ 15 ms/frame CPU; first hero image | 1–1.5 d |
| **P3 Segmenter** [MUST] | Model spike (fallback chain §5.3) → wrapper → **cache predictions** as `.label` → E1 by range → E2 latency | Cached preds for eval subset; per-range mIoU table; leakage check recorded | 1 d (time-boxed) |
| **P4 Motion + objects** [SHOULD] | Residual motion, cluster vote, boxes; E5, E6 | Moving-IoU reported honestly; boxes render | 0.5–1 d |
| **P5 Derived layers + hazards** [SHOULD] | slope/step/clearance/traversable; hazard injection; E8 | Kerb visible in step layer at ≤ 10 cm rings; detection table | 0.5 d |
| **P6 Benchmarks & ablations** [MUST for E1–E4, SHOULD for E5–E9] | `foveamap bench` writes all `results/*.json` + plots; README tables auto-generated | One command regenerates every number/figure | 1 d |
| **P7 Dashboard & demo** [MUST offline video, SHOULD live app] | Composed dashboard renderer → MP4/GIF; Streamlit app with overlays, meters, fovea slider | Demo storyboard (§7) runs; GIF in README | 1–1.5 d |
| **P8 Polish** [MUST] | README, docs/design.md, limitations, FAQ, NOTICE, LICENSE, CI (ruff+pytest on synthetic data), tag `v0.1` | Fresh-clone quickstart works on a clean machine; human review **before any push** | 0.5–1 d |

**Human checkpoints:** after P0 (env + data OK?) · after P2 (approve palette and ring config from the hero image) · after P3 (approve model + accuracy story) · after P7 (approve dashboard look) · before any push/publish.

**Scope tiers (choose by time, H1)**

| Tier | Contents | Roughly |
|---|---|---|
| **MVP** (~3 days, 1 person) | P0–P3 (one model, cached preds), P6 for E1–E4, offline video, basic README | Passes every explicit PS item |
| **Core** (~1 week) | + P4, P5, live Streamlit app with fovea slider, E5–E9, tests + CI | The version that stands out |
| **Stretch** (2 weeks) | S1–S7 in §12 | Only if Core is finished and polished |

---

## 10. Human decision points ([HUMAN])

**Blocking — ask together at P0.** Recommended default first.

| ID | Decision | Options | Default | Impact |
|---|---|---|---|---|
| H1 | Time budget and team size | 3 d solo / 1 wk 2–3 people / 2 wk | 1 wk, 2–3 people | Sets tier and track split |
| H2 | Compute for the network | (a) Colab/Kaggle GPU for one batch caching job, everything else local CPU · (b) local NVIDIA GPU under WSL2/Linux · (c) CPU only | (a) | MMDetection3D/MinkowskiEngine are painful on native Windows; caching makes the GPU a one-off |
| H3 | Data acquisition | (a) human downloads the KITTI odometry velodyne zip (a single file of roughly 80 GB **[VERIFY]**; cannot fetch one sequence alone) + SemanticKITTI labels (~0.2 GB) + calib/poses · (b) find a trusted mirror of individual sequences **[VERIFY licence/trust]** · (c) start with a small subset only | (b) if available, else (a) | Needs 08 at minimum; more only for S1. **Human registers/downloads; agent must not** |
| H4 | Segmenter | FRNet / CENet / SalsaNext / oracle-only | FRNet with fallback chain | Accuracy, speed, install pain |
| H12 | Other deliverables required by the course | report / slides / demo video / viva; page limits | ask | Decide whether to also produce a short report and slides |
| H13 | Repo identity | name (`foveamap`), location (`classAssign1/foveamap/` vs new folder), licence (MIT vs Apache-2.0), visibility (private until reviewed) | `foveamap`, MIT, private | Also whether/when pushing is allowed |

**Just-in-time decisions**

| ID | Phase | Decision | Options | Default |
|---|---|---|---|---|
| H5 | P4 | Motion source | geometric residual / 4DMOS / both | geometric; 4DMOS as stretch |
| H6 | P1 | Class policy: are parking, sidewalk, lane-marking, other-ground "drivable"? Standing pedestrians dynamic? Vegetation an obstacle? | table §5.1 / custom | §5.1 mapping |
| H7 | P2 | Headline grid | `fovea_4ring` / `ps_literal_2ring` / both; square vs circular rings | `fovea_4ring` headline, PS-literal always in tables, square rings |
| H8 | P2 | Cell class rule | `safety_priority` / `majority` | `safety_priority`, report both |
| H9 | P7 | Dashboard tech | Streamlit / Rerun / custom web / video-only | Streamlit + always the MP4 |
| H10 | P7 | Demo clip | auto-pick / human picks | auto-pick then confirm |
| H11 | P3+ | Training ambition | none / range-aware fine-tune (S1) / distillation | none unless time remains; a real training result helps a GenAI course grade |
| H14 | P7 | Look and branding | dark "sensor" theme / light; name and logo | dark, FoveaMap wordmark |
| H15 | P2 | Persistent world-frame map with pose accumulation in scope? | no / yes | no (per-frame ego-centric map); stretch S6 |
| H16 | P5 | Add Patchwork++ ground segmentation as baseline/robustness check? | no / yes | optional ablation only |
| H17 | P6 | Eval subset | every 5th frame / every 2nd / all | every 5th of seq 08 |
| H18 | P6 | Second dataset (nuScenes) for generalisation | no / yes | no (stretch S5) |

**How to ask:** one `AskUserQuestion` call per batch, at most 4 questions per call, recommended option first labelled "(Recommended)". Do not stop the whole project waiting on a just-in-time decision: proceed on the default, note it in `docs/DECISIONS.md`, and surface it at the next checkpoint.

---

## 11. Gaps and risks we might otherwise miss

| # | Gap / risk | Mitigation |
|---|---|---|
| G1 | GPU unknown; Windows dev machine; MMDetection3D / MinkowskiEngine are Linux-first | H2 default: Colab/Kaggle for caching. Keep everything else CPU-only. Record exact env |
| G2 | KITTI velodyne data is one huge zip; account required | H3; provide `scripts/check_data.py` and clear steps; oracle mode needs only seq 08 |
| G3 | Data licence (KITTI/SemanticKITTI are CC BY-NC-SA 3.0 **[VERIFY]**); model licences differ (FRNet Apache-2.0, CENet MIT, 4DMOS MIT, KISS-ICP MIT **[VERIFY]**) | `NOTICE.md`; never commit raw scans; commit only small derived assets; non-commercial framing |
| G4 | Pretrained checkpoint may have trained on seq 08 (leakage) | Check model card/config; flag or switch eval sequence |
| G5 | Stock checkpoint has 19 classes, no "moving" | Motion module (§5.4); do not pretend the net outputs motion |
| G6 | Poses are in camera frame | `Tr` conversion + alignment test (§5.2) |
| G7 | No elevation/pothole ground truth exists | Oracle-vs-uniform comparison (E4) + synthetic hazards (E8); state the limitation |
| G8 | Far-range metrics have tiny support (few pedestrians at 80 m) | Report support counts; bootstrap CIs if time; avoid over-claiming |
| G9 | Latency numbers are easy to fake (no warm-up, excluded preprocessing, async GPU) | Protocol §5.8; report hardware; separate throughput from latency |
| G10 | "Real-time" display in Streamlit ≠ pipeline speed | Report headless pipeline FPS; state display FPS separately |
| G11 | Reviewer asks "raw points are smaller than your grid" | FAQ answer §13; show measured raw size honestly |
| G12 | 2.5D loses vertical structure (bridges, trees) | `overhang_z` + clearance layer; name the limit in README |
| G13 | Float rounding at ring boundaries causes off-by-one cells | Rounding guard + test T3 |
| G14 | Ring sizes not integer multiples → seams | Invariant I1; validate spec at load time and raise |
| G15 | Circular "radius" in PS vs square rings | Square by default (exact alignment); explain in docs; circular is an option |
| G16 | Coarse cells erase small hazards | `safety_priority` aggregation; E4 small-object survival metric |
| G17 | Thresholds tuned on the eval subset | Tune on train-sequence frames only |
| G18 | Numbers in README drift from results | Auto-generated tables from `results/*.json` |
| G19 | Demo fails live | Pre-rendered MP4 + GIF + cached predictions; run without GPU |
| G20 | Non-determinism | Seeds, pinned `requirements.txt`, config-driven runs, results record git hash + config |
| G21 | Repo hygiene (large files, secrets, `.ipynb` outputs) | `.gitignore` for data/ckpt/preds; commit only small results; strip notebook outputs |
| G22 | Over-scoping | Tiers §9; MVP first; stretch only after Core is polished |
| G23 | Course may expect *some* training | H11; S1 range-aware fine-tune is the smallest credible training contribution |
| G24 | Possible unclear deliverables beyond repo (report/slides/viva) | H12 |
| G25 | Datasets label at 19 vs 25 classes are easy to mix up | One `labels.py` owns all id maps with tests |

---

## 12. Future work (README "Roadmap" + honest scope statement)

| ID | Item | Why it matters |
|---|---|---|
| S1 | **Range-aware fine-tuning** (distance-weighted loss) of the range-view net; before/after per-range mIoU | Directly targets the PS's "accuracy across distances" |
| S2 | 4DMOS / learned moving-object segmentation replacing the geometric motion module | Better mover recall at range |
| S3 | Distillation from PTv3 into a small range-view net; TensorRT FP16/INT8; Jetson Orin benchmarks | Real embedded-deployment story |
| S4 | GitHub Pages static replay + Colab one-click demo | Zero-install judging |
| S5 | nuScenes (32-beam) generalisation and a second sensor geometry | Robustness |
| S6 | Persistent world-frame foveated map with pose accumulation, toroidal scrolling, Bayesian height fusion | Denser far range; a real map, not a per-scan snapshot |
| S7 | CARLA closed-loop clip with exact ground-truth geometry (validates elevation error quantitatively) | Answers the "no elevation GT" gap |
| S8 | **Gaze-controlled fovea:** move/steepen the fine region toward the planned path, turns, or detected pedestrians | Foveation that actually follows attention |
| S9 | Non-uniform partition *inside* the network (NUC-Net-style radial partition) so network and map share one structure | Removes resampling, cuts network cost |
| S10 | Free-space raycasting layer and uncertainty layer; learned traversability | Planner-grade map |
| S11 | ROS 2 `grid_map` export/adapter | Drop-in for robotics stacks |
| S12 | Camera fusion for far-range semantics | Helps sparse far LiDAR |
| S13 | Wavelet/octree compression of the fovea (wavemap-style) | Further memory reduction |

**Deliberately out of scope:** SLAM, path planning, closed-loop control, multi-sensor calibration, production hardening.

---

## 13. README outline, talking points, FAQ

**README outline:** (1) hero GIF + one-line pitch; (2) TL;DR results table (auto-generated: memory ×, p95 latency/FPS, mIoU by range); (3) architecture diagram; (4) the fovea idea with the density-vs-range figure; (5) quickstart (3 commands, oracle mode needs no GPU); (6) reproduce results; (7) results and honest limitations; (8) repo map; (9) roadmap; (10) data/model attributions and licences.

**Anticipated questions and short answers**

| Question | Answer |
|---|---|
| Why not PointNet++ as in the PS? | It is a baseline; on SemanticKITTI it reaches roughly 20 mIoU **[VERIFY]** and does not scale to full scans in real time. We use a modern range-view network chosen on measured accuracy/latency evidence. |
| Did you train the model? | We use released checkpoints (no leakage on the eval sequence, checked). [If S1 done:] we fine-tuned with a range-aware loss and report before/after. |
| How do you know the grid has no alignment error or data loss? | Integer-multiple nesting, cell-based ring assignment, and tests T1–T5: conservation, nesting equality, boundary determinism, round trip, partition. |
| Does coarsening hurt accuracy at range? | E4 (oracle labels) isolates it; E9 shows the memory/accuracy trade-off. We report whichever way it comes out. |
| Why square rings when the PS says radius? | Square rings align exactly with nested cells; the square contains the 100 m circle. Circular rings are supported via a cell-centre rule. |
| A raw scan is only ~2 MB. Why a grid? | The grid is fixed-size, planner-ready and time-aggregatable; raw accumulation grows unbounded. The PS baseline is a uniform high-res 3D map, which we compare against, and we also show measured sparse-3D and raw sizes. |
| 2.5D can't represent bridges | `overhang_z` and clearance layers capture passable/impassable overheads; still a limit for multi-level structures (stated). |
| Is it really real-time? | We report end-to-end p50/p95/p99 with preprocessing on named hardware, against the 100 ms sensor period. |
| How are potholes validated? | SemanticKITTI has no such ground truth, so we inject synthetic hazards and report detection by ring, and say so. |
| Embedded deployment? | Range-view nets were the only family real-time on Jetson Orin in the cited study; TensorRT deployment is roadmap S3. |

---

## 14. Definition of done

- [ ] `foveamap run` works from a fresh clone on oracle labels with no GPU
- [ ] Cached predictions from a pretrained network run through the same pipeline; checkpoint leakage check documented
- [ ] Grid invariants T1–T5 pass in CI
- [ ] `results/` contains E1–E4 (MUST) and as many of E5–E9 as time allows, all regenerated by one command
- [ ] README numbers are auto-generated; limitations section present
- [ ] Dashboard (live or pre-rendered) shows: colour-coded map with ring overlay, memory meter vs baselines, latency/FPS, accuracy-vs-distance
- [ ] MP4 + GIF demo committed (small)
- [ ] `NOTICE.md` lists data/model licences; no raw data or checkpoints in git
- [ ] `docs/DECISIONS.md` records every §10 answer and every deviation
- [ ] Human reviewed the repo; **push only after explicit approval**

---

## 15. References

- SemanticKITTI dataset: https://semantic-kitti.org/dataset.html · API/configs: https://github.com/PRBonn/semantic-kitti-api
- FRNet: https://github.com/Xiangxu-0103/FRNet · paper https://arxiv.org/abs/2312.04484
- CENet: https://github.com/huixiancheng/CENet · SalsaNext: https://github.com/TiagoCortinhal/SalsaNext
- Real-time/embedded LiDAR segmentation study: https://arxiv.org/abs/2410.08365
- Point Transformer V3: https://arxiv.org/pdf/2312.10035 · MMDetection3D model zoo: https://mmdetection3d.readthedocs.io/en/latest/model_zoo.html
- 4DMOS: https://github.com/PRBonn/4DMOS · Awesome-LiDAR-MOS: https://github.com/neng-wang/Awesome-LiDAR-MOS
- KISS-ICP: https://github.com/PRBonn/kiss-icp · Patchwork++: https://arxiv.org/abs/2207.11919
- elevation_mapping_cupy: https://github.com/leggedrobotics/elevation_mapping_cupy · wavemap: https://arxiv.org/pdf/2306.01279
- NUC-Net (non-uniform cylindrical partition): https://arxiv.org/abs/2505.24634 · PolarNet: https://arxiv.org/pdf/2003.14032
