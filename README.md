# FoveaMap — Adaptive Variable-Resolution 2.5D LiDAR Mapping

**Smart India Hackathon 2026 · Problem Statement SIH26053**
**Title:** Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception

> **Foveated 2.5D LiDAR mapping: sharp where a mistake hurts, cheap where it doesn't.**
> A deep-learning pipeline that turns raw LiDAR point clouds into a foveated, variable-resolution 2.5D elevation-and-semantics map — inspired by how human vision keeps the centre of gaze sharp and the periphery coarse.

---

## 1. Problem Statement

**Background.** Autonomous vehicles need to perceive their surroundings with high precision. Raw 3D LiDAR point clouds are information-rich but expensive to process in real time; plain 2D occupancy grids are cheap but throw away the height information needed to detect curbs, potholes, or overhanging obstacles. The problem calls for a **foveated mapping approach** — like human vision — where the area immediately around the vehicle is rendered in high detail for safety, while distant regions are simplified to save compute and memory.

**What must be built.** A system that:

1. **Terrain Analysis** — separates drivable surfaces from non-drivable terrain.
2. **Object Detection** — classifies static obstacles (walls, poles, buildings) and dynamic objects (pedestrians, vehicles), including their motion state.
3. **Adaptive Spatial Representation** — a non-uniform grid whose cell size grows with distance from the sensor, implemented without alignment errors or data loss when projecting 3D points into 2.5D.

**Expected solution components** (as specified in the PS):

| Component                       | Requirement                                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Deep learning model             | Semantic segmentation of point clouds into terrain / static obstacles / moving objects (e.g. PointNet++ or a Sparse CNN)   |
| Variable-resolution grid engine | Projects classified 3D points into a 2.5D grid: fine cells (e.g. 5 cm) within 10 m, coarsening (e.g. 50 cm) out to 100 m   |
| Real-time visualization         | Dashboard with colour-coded terrain/objects and a demonstrated memory-usage reduction vs. a uniform high-resolution 3D map |
| Performance metrics             | Low latency / high FPS and classification accuracy reported across distance                                                |

---

## 2. Our Idea — FoveaMap

FoveaMap converts every LiDAR scan into a **foveated 2.5D map**: a set of nested, precisely-aligned grids whose cell size grows outward from the sensor (5 cm → 10 cm → 20 cm → 40 cm). Every cell stores ground height, obstacle height, overhang clearance, a semantic class, a "moving" flag, and a confidence score.

**Central insight — the fovea matches the sensor, not just the budget.** A rotating LiDAR's point spacing is a few centimetres at 10 m but tens of centimetres by 100 m. Fine cells far away are mostly empty; the ring sizes are chosen from this physical sampling limit, not picked arbitrarily — and we prove that with a density-vs-range plot.

### 2.1 Pipeline at a glance

```
Raw LiDAR scans + poses + calibration
        │
        ▼
Data layer  ──  pose alignment, ground-truth/label handling, range bucketing
        │
        ├───────────────────────────────┐
        ▼                                ▼
Semantic Segmentation Model         Oracle mode (ground-truth labels)
(pretrained range-view network)     — isolates grid error from model error
        │  per-point class + confidence
        ▼
Motion Estimation  ──  scan-to-scan residual + cluster voting → moving/static + object boxes
        │
        ▼
Foveated Grid Engine  ──  nested rings, square, alignment-verified
        │  layers: ground height, top height, overhang clearance, class, motion, count, confidence
        ▼
Derived Layers  ──  slope, kerb/step detection, clearance, traversability
        │
        ▼
Benchmarking (memory · latency · accuracy-by-range)  +  Real-Time Dashboard
```

### 2.2 What makes this approach credible

1. **Oracle-first, modular build.** The grid engine and dashboard are built and validated against ground-truth labels first, so a working demo exists before the deep-learning model is even wired in. This also separates "error caused by grid coarsening" from "error caused by the model" — a distinction most naive solutions never make.
2. **Predictions are cached, not recomputed live.** Model inference runs once on GPU; the grid engine, evaluation, and dashboard then run on any laptop from cached predictions.
3. **A modern, real-time-capable segmentation network**, chosen on measured accuracy/latency evidence rather than defaulting to PointNet++, which does not scale to full scans in real time. This choice is justified quantitatively (§4.1).
4. **A trade-off explorer** directly answers the hidden question behind the problem statement: *does coarsening the far field actually hurt accuracy or safety?* We sweep grid presets and report memory-vs-accuracy honestly, whichever way it comes out.

### 2.3 What the live demo shows

- **Fovea overlay** — ring boundaries drawn on the map with their cell sizes labelled.
- **Memory meter** — live comparison of dense 3D, sparse 3D, uniform 2.5D, and FoveaMap memory footprints.
- **Latency/FPS bar** against the sensor's real-time budget.
- **Accuracy-vs-distance curve** — uniform grid vs. foveated grid, per range bucket.
- **Fovea slider** — drag ring radii/cell sizes and watch the map, memory, and accuracy update live.
- **Zoom lens** — the same kerb rendered at fine and coarse resolution, showing exactly why the fovea sits near the vehicle.
- **Object boxes** with class labels; moving objects are visually distinguished.
- **Synthetic hazard injection** — a pothole/kerb/overhang is inserted on demand to show which ring detects it.

---

## 3. Technical Design

### 3.1 Semantic classes

Points are grouped into safety-oriented super-classes, each carried per grid cell along with a `moving` flag:

| Super-class              | Meaning                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `DRIVABLE`             | Road, parking areas, lane markings                                                                                       |
| `NON_DRIVABLE_TERRAIN` | Sidewalk, natural terrain, other ground                                                                                  |
| `STATIC_OBSTACLE`      | Buildings, fences, vegetation, poles, signs, parked vehicles                                                             |
| `DYNAMIC`              | Moving vehicles, and pedestrians/cyclists/motorcyclists (always flagged safety-critical, even if momentarily stationary) |
| `UNKNOWN`              | Unobserved or unlabelled cells                                                                                           |

### 3.2 Semantic segmentation model

A **range-view convolutional network** is used for per-point semantic segmentation — this family is the only one shown to meet real-time constraints on embedded automotive-grade hardware, versus sparse-convolution or point-based networks which are far more accurate but far too slow for real-time use. This choice is documented and benchmarked against PointNet++/Sparse-CNN alternatives explicitly named in the problem statement, with reasoning based on measured accuracy-vs-latency trade-offs rather than convenience.

A pretrained checkpoint is used rather than training from scratch, since LiDAR semantic segmentation is a saturated research problem; our engineering effort is concentrated on the part with no existing off-the-shelf solution — the **adaptive resolution grid engine** — plus, time permitting, a targeted fine-tuning pass that reweights the training loss by distance, directly improving the accuracy-vs-range story the problem statement asks for.

Because released segmentation checkpoints label only *static* semantic classes, motion state is **not** assumed to come from the segmentation network — it is computed by a separate module (§3.3), and this separation is stated explicitly rather than glossed over.

### 3.3 Motion / dynamic object detection

Moving objects are identified geometrically, without requiring a second heavy network:

1. A previous scan is pose-compensated (ego-motion removed) into the current frame.
2. Per-point range residuals are computed — points that shifted more than expected are motion candidates.
3. Candidate points of movable classes (vehicles, pedestrians, cyclists) are clustered; a cluster is marked **moving** if enough of its points vote for motion, filtering out sensor noise.
4. Each object is output with a class, a bounding box, height, and a moving/static flag.

This keeps the system light enough for real-time operation, with a documented upgrade path to a learned multi-scan motion segmentation network if time allows.

### 3.4 Variable-resolution grid engine — the core deliverable

The grid is a **nested "clipmap" of square rings**, anchored at the vehicle. The default configuration:

| Ring            | Range from sensor | Cell size | Approx. cell count   |
| --------------- | ----------------- | --------- | -------------------- |
| 1               | 0 – 10 m         | 5 cm      | 160,000              |
| 2               | 10 – 30 m        | 10 cm     | 320,000              |
| 3               | 30 – 60 m        | 20 cm     | 270,000              |
| 4               | 60 – 100 m       | 40 cm     | 160,000              |
| **Total** |                   |           | **≈ 910,000** |

A uniform 5 cm grid over the same 200 m × 200 m area needs **≈ 16,000,000 cells** — roughly **17.6× more** than the foveated grid, i.e. a **~94% reduction in cell count** (and a proportional reduction in memory footprint). This number is computed programmatically from the grid configuration, never hard-coded, so it can never go stale.

The problem statement's own literal example (5 cm within 10 m, 50 cm out to 100 m) is also implemented and evaluated as an alternate preset, alongside two uniform-resolution baselines used purely for fair comparison.

**Guaranteeing "no alignment errors, no data loss" — this is treated as something to *prove*, not just claim:**

- Every coarser cell size is an exact integer multiple of the finer ones, so cell boundaries never straddle rings.
- Ring assignment is done per-cell (not per-point) using half-open boundary intervals, so every point maps to **exactly one** cell, with no double-counting or gaps.
- Aggregated per-cell statistics (counts, sums, min/max) are verified to match bit-for-bit whether computed directly at a coarse resolution or by reducing from the finest resolution upward.
- An automated test suite checks: point conservation, correct nesting, deterministic boundary handling, round-trip cell↔world consistency, and full-plane partitioning with no overlaps or gaps.

**Per-cell data layers:** ground height, obstacle top height, overhang clearance (for bridges/branches), semantic class, fraction of moving points, point count, and mean model confidence — compact enough to keep memory low while remaining rich enough for downstream planning.

### 3.5 Derived layers (plain geometry, safety-focused)

- **Slope** — local gradient of ground height, used to flag steep or unsafe terrain.
- **Step / kerb detector** — height discontinuity between neighbouring cells; this specifically needs the fine rings near the vehicle to be reliable, which is visually demonstrated in the zoom-lens view.
- **Clearance** — vertical gap between the ground and any overhang, to determine if a tall vehicle can pass underneath.
- **Traversability** — a combined flag: drivable class, acceptable slope, acceptable step height, sufficient clearance, no obstacle.

Since public LiDAR datasets don't include ground-truth potholes or kerb heights, synthetic hazards (a shallow pothole, a raised kerb, a low overhang) are injected into real scans so hazard-detection rate can still be measured honestly per ring and per distance.

### 3.6 Memory accounting

Memory usage is compared across **four representations**, not a single strawman baseline:

1. Dense uniform 3D voxel grid at fine resolution (the naive baseline the PS references).
2. Sparse (occupied-only) 3D voxel grid at the same resolution.
3. Uniform 2.5D grid at fine resolution.
4. **FoveaMap** — the nested variable-resolution grid.

Both the theoretical footprint and the actually measured, per-scan memory usage are reported, so the demonstrated reduction is evidence-based rather than asserted.

### 3.7 Latency and real-time performance

Latency is measured end-to-end (including pre/post-processing, not just network inference), reported as p50/p95/p99 across a representative run, with hardware and software versions stated. The system is only described as "real-time" if the measured p95 latency beats the sensor's scan period — never asserted without a number behind it. Live display frame-rate and underlying pipeline frame-rate are reported separately, since a dashboard's refresh rate is not the same thing as processing speed.

---

## 4. Evaluation Plan

All metrics are reported **per distance bucket** aligned to the grid rings (0–10 m, 10–30 m, 30–60 m, 60–100 m), with the number of supporting points/objects shown alongside each number so sparse far-range results aren't over-interpreted.

| Experiment                   | What it measures                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Semantic accuracy            | Per-class and per-super-class accuracy, split by distance                                                           |
| Latency & FPS                | Per-stage and end-to-end timing against the sensor's real-time budget                                               |
| Memory                       | Cell counts and measured memory across all four baselines                                                           |
| Cost of coarsening           | Using ground-truth labels, isolates exactly how much accuracy the adaptive grid trades away vs. a uniform fine grid |
| Motion accuracy              | Moving-object detection accuracy vs. ground truth, and false-positive rate on stationary vehicles                   |
| Object recall/precision      | How reliably objects are detected and correctly classified, by distance                                             |
| Sensor-physics justification | Point density vs. distance, overlaid with the chosen ring sizes, to show the grid design is physically motivated    |
| Hazard detection             | Detection rate of injected potholes/kerbs/overhangs, by ring and distance                                           |
| Memory-vs-accuracy trade-off | A sweep across grid configurations, answering "how much does foveation cost us?"                                    |

---

## 5. Real-Time Visualization Dashboard

A dark-themed, single-screen dashboard shows, side by side:

- The top-down 2.5D map, shaded by height and coloured by class, with ring boundaries and object boxes overlaid.
- A live memory-usage comparison against the baseline representations (log scale).
- A latency breakdown against the real-time budget, with current FPS.
- An accuracy-vs-distance curve comparing the foveated grid to a uniform grid.
- A zoom-lens panel comparing the same real-world feature (e.g. a kerb) at fine vs. coarse resolution.
- Controls to play back a scan sequence, adjust ring configuration live, and inject synthetic hazards.

An **offline pre-rendered video is always produced as well**, so the demonstration does not depend on a live environment working correctly during judging.

---

## 6. Implementation Plan

### 6.1 Architecture / repository structure

```
foveamap/
├─ README.md              # this file (auto-generated tables of results)
├─ configs/                # grid presets, model, motion, vehicle parameters
├─ data/                   # datasets (not committed; download scripts only)
├─ src/foveamap/
│  ├─ io/                  # scan/label/pose loading and coordinate handling
│  ├─ models/              # segmentation model wrappers + oracle mode
│  ├─ motion/               # motion residual + clustering + object boxes
│  ├─ grid/                # variable-resolution grid engine + invariant tests
│  ├─ derived/              # slope, step, clearance, traversability
│  ├─ eval/                # benchmarking scripts
│  └─ dashboard/            # visualization app
├─ tests/                  # grid-invariant unit tests
├─ results/                # auto-generated benchmark outputs (json + plots)
└─ docs/                   # design notes, decisions log, limitations
```

### 6.2 Phased build plan

| Phase                                   | Focus                                                                                            | Outcome                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| **Phase 0 — Setup**              | Repo skeleton, environment, data access verified                                                 | Pipeline loads one real scan with labels          |
| **Phase 1 — Data layer**         | Point cloud + pose + label loading, coordinate-frame handling                                    | Verified frame alignment across consecutive scans |
| **Phase 2 — Grid engine**        | Nested variable-resolution grid, alignment invariants, memory accounting, baseline uniform grids | Passing invariant tests; first rendered map       |
| **Phase 3 — Segmentation model** | Integrate pretrained model, cache predictions, evaluate accuracy by distance                     | Per-distance accuracy table                       |
| **Phase 4 — Motion & objects**   | Motion residual + clustering, object boxes                                                       | Motion accuracy reported honestly                 |
| **Phase 5 — Derived layers**     | Slope, kerb/step, clearance, traversability, synthetic hazard injection                          | Kerb visibly detected at fine resolution          |
| **Phase 6 — Benchmarking**       | One command regenerates every metric and plot                                                    | Reproducible, no hand-typed numbers               |
| **Phase 7 — Dashboard & demo**   | Live dashboard + offline rendered video                                                          | Full demo storyboard runs end-to-end              |
| **Phase 8 — Polish**             | Documentation, limitations section, licensing/attribution, clean-up                              | Fresh clone runs the quickstart successfully      |

### 6.3 Build philosophy

- Build and validate the grid engine against ground-truth labels **before** the deep-learning model is wired in, so model error and grid error are never conflated.
- Cache model predictions once so the rest of the system runs without requiring a GPU.
- Never hard-code a result number — every figure in this document and its final report is regenerated from benchmark output.
- Time-box every uncertain step (data access, model installation) with a documented fallback so no single blocker stalls the whole project.
- Log every deviation from the plan with a one-line reason for full traceability.

---

## 7. Honest Limitations

- Public LiDAR benchmarks have no ground truth for potholes, kerb heights, or overhangs — these are validated using synthetic hazard injection, which is clearly labelled as an approximation.
- A 2.5D representation cannot fully capture true multi-level structures (e.g. bridges with traffic beneath and above) — an overhang-clearance layer partially addresses this but the fundamental limitation is stated plainly rather than hidden.
- Accuracy at very long range has limited supporting data (few pedestrians/objects appear at 80–100 m in any single sequence), so far-range numbers are reported with their sample size attached.
- "Real-time" is only claimed where end-to-end latency, including pre/post-processing, has actually been measured against the sensor's real update rate.

---

## 8. Roadmap (Future Work)

- Distance-weighted fine-tuning of the segmentation model to directly target far-range accuracy.
- Replacing the geometric motion module with a learned multi-scan motion-segmentation network.
- Model compression/quantization for embedded deployment (e.g. automotive-grade edge hardware).
- A persistent, world-frame version of the map that accumulates observations across time rather than resetting every frame.
- Gaze-controlled foveation — steering the high-resolution region toward the planned path, an upcoming turn, or a detected pedestrian, rather than keeping it fixed around the vehicle.
- Fusing camera data to improve semantics at long range, where LiDAR points are naturally sparse.
- Exporting the map in a standard robotics format for direct use by downstream planning modules.

**Deliberately out of scope for this submission:** full SLAM, path planning, closed-loop vehicle control, and multi-sensor calibration.

---

## 9. Why This Solution Answers the Problem Statement

| Problem statement requirement                                     | How FoveaMap addresses it                                                                                                                               |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Deep-learning point-cloud pipeline                                | Pretrained semantic segmentation network in the loop, end to end from raw scan to map                                                                   |
| Terrain analysis                                                  | Drivable vs. non-drivable super-classes, plus a derived traversability layer                                                                            |
| Object detection, static and dynamic                              | Per-point semantic classes combined with an explicit motion module and clustered object boxes                                                           |
| Adaptive spatial representation, no alignment errors or data loss | Integer-multiple nested grid with a dedicated automated test suite proving conservation, correct nesting, and full coverage                             |
| Fine cells near the sensor, coarse cells far away                 | Configurable nested rings; the problem statement's literal example is implemented and evaluated alongside our recommended, sensor-matched configuration |
| Real-time visualization dashboard with memory reduction shown     | Live dashboard with a memory meter comparing against three baseline representations                                                                     |
| Low latency / high FPS and accuracy across distance               | End-to-end latency measured honestly (not just model inference), and every accuracy metric is reported broken down by distance                          |