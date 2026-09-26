# FoveaMap Locked Implementation Plan

**Status:** approved implementation baseline  
**Date:** 2026-09-26  
**Primary requirement:** Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception  
**Dataset:** SemanticKITTI sequence 08 only  

This document records the decisions agreed before further implementation. It is the reference for scope,
technical choices, evaluation claims, frontend behaviour, and phase order. Changes require a dated entry in
`docs/DECISIONS.md`; measured evidence may refine thresholds but must not silently change the intent below.

## 1. Product interpretation

FoveaMap converts raw LiDAR scans into a variable-resolution 2.5D elevation and semantic map. The final user
experience has two primary workflows:

1. **Terrain Analysis** — drivable, non-drivable and unknown terrain, supported by elevation, slope,
   kerb/step, clearance, traversability and confidence.
2. **Object Detection** — semantic object class, bounding box, confidence, distance and independently
   estimated motion state (`STATIC`, `MOVING`, or `UNCERTAIN`).

Adaptive spatial representation is not a third mode. Ring boundaries, range bands and cell resolution are
visible in both workflows.

## 2. Production model

- **LSK3DNet is the production semantic-segmentation model.** No switch to a lower-accuracy model is planned.
- The exact checkpoint, SHA-256, upstream configuration, PyTorch/CUDA versions, `spconv` version,
  `torch-scatter` version and GPU are recorded with every prediction cache.
- The paper's best reported mIoU and the exact selected checkpoint's measured result are reported separately.
- LSK3DNet supplies semantic class. It does not by itself prove that an object is moving.
- Observed movement is estimated from pose-compensated consecutive scans, residual voting, clustering and
  optional temporal tracking.

### Required production acceptance tests

1. **Reference parity:** wrapper and upstream inference produce exactly equal per-point labels on three native
   scans. FP16 confidence may differ slightly, but label equality remains required.
2. **Frontend identity:** the live model path displays `MODEL: LSK3DNet`; oracle and cached modes remain clearly
   labelled.
3. **RTX 4050 stability:** at least 100 consecutive scans, batch one, no OOM, no CPU fallback, no growing VRAM
   leak, output length equal to input length, with peak allocated/reserved VRAM and p50/p95/p99 latency saved.
4. **Provenance:** checkpoint hash and software/hardware metadata accompany results and caches.

## 3. 100 m semantic coverage

The implementation target is semantic and geometric coverage to a 100 m radius. The first approach expands
LSK3DNet's sparse coordinate space while preserving the checkpoint's 5 cm voxel size:

| Setting | Native | Extended target |
|---|---:|---:|
| X bounds | -50 to +50 m | -100 to +100 m |
| Y bounds | -50 to +50 m | -100 to +100 m |
| Z bounds | -4 to +2 m | -4 to +2 m |
| Spatial shape | 2000 x 2000 x 120 | 4000 x 4000 x 120 |
| Voxel size | 5 cm | 5 cm |

The extended configuration is accepted only if it passes:

- Point accounting through 100 m with no unexplained loss.
- At least 99.9% label agreement between native and extended configurations for points inside 45 m.
- Accuracy and unknown-rate reporting for 0-10, 10-30, 30-50, 50-60, 60-80 and 80-100 m.
- RTX 4050 peak-VRAM and latency comparison against the native configuration.
- Boundary tests at negative coordinates, exact voxels, ring boundaries and +/-100 m.

If direct sparse extension fails, the fallback order is overlapping tiled LSK3DNet inference, then a documented
hybrid long-range path. Returning every point beyond 50 m as unknown is the final fallback, not the intended
submission configuration.

## 4. Adaptive grid

The production grid uses exact hierarchical nesting:

| Ring | Range | Cell size |
|---|---:|---:|
| 1 | 0-10 m | 5 cm |
| 2 | 10-30 m | 10 cm |
| 3 | 30-60 m | 20 cm |
| 4 | 60-100 m | 40 cm |

The problem statement's 50 cm value is an example. Forty centimetres is selected because it is an exact
multiple of 20 cm, preserving deterministic fine-to-coarse reduction and boundary alignment.

"No data loss" means no accidental loss during projection: every valid point is assigned exactly once or
explicitly counted as invalid/out of range. Spatial coarsening intentionally reduces detail, and that loss is
measured through map-cell, back-projection and hazard metrics.

## 5. Sequence 08 data policy

Only SemanticKITTI sequence 08 is available. Use chronological regions with guard gaps:

| Purpose | Frames |
|---|---:|
| Integration/development | 0000-0814 |
| Guard gap | 0815-0864 |
| Calibration | 0865-1220 |
| Guard gap | 1221-1270 |
| Locked evaluation | 1271-4070 |

- Development frames support debugging and visual design.
- Calibration frames support threshold selection.
- Final evaluation frames are not used for tuning.
- Motion processing starts from the second frame within each region and does not borrow across split boundaries.
- Because the pretrained checkpoint may already have been selected using SemanticKITTI sequence 08, semantic
  mIoU is described as checkpoint reproduction. The final region is a temporally held-out **system** evaluation,
  not evidence of unseen-domain model generalisation.

## 6. Semantic and motion meaning

The application keeps these concepts separate:

- **Semantic class:** car, person, bicycle, pole, building, vegetation and related classes from LSK3DNet.
- **Capability:** inherently static or potentially dynamic class.
- **Observed motion:** `STATIC`, `MOVING`, or `UNCERTAIN`, based on temporal evidence.

SemanticKITTI has no dedicated general wall class. Building/fence and wall-like vertical structures are shown
under the honest display superclass `STATIC STRUCTURE`.

## 7. Frontend contract

The main map exposes only two primary buttons:

- `TERRAIN ANALYSIS`
- `OBJECT DETECTION`

Height, traversability, movement and confidence remain available as contextual overlays, styling and selected-
cell/object details rather than five competing top-level modes.

Both modes permanently show adaptive ring boundaries, range labels, cell resolutions, fine/coarse cell structure
and selected-cell metadata. The header identifies the active data path: `ORACLE (GROUND TRUTH)`,
`MODEL: LSK3DNet`, or `CACHED: LSK3DNet`.

Detailed analytics live at `/dashboard`, opened in a new browser tab through `DASHBOARD` rather than a popup.
The main screen retains a compact strip for model/mode, pipeline FPS, p95 latency, memory and connection state.

## 8. Evaluation contract

Keep these results separate:

1. **Point semantic accuracy:** 19-class mIoU, per-class IoU, superclass IoU and distance buckets.
2. **Map-cell accuracy:** model-generated cells against oracle-generated cells observed in both maps.
3. **Back-projection accuracy:** each point receives its aggregated cell class, measuring coarsening loss.
4. **Object/motion accuracy:** moving-point IoU, object precision/recall, class accuracy and stationary-vehicle FPR.
5. **Hazard accuracy:** detection rate by ring, distance and feature size.

The primary memory claim compares adaptive 2.5D against uniform 5 cm 2.5D. Uniform 20 cm, sparse occupied 3D
and theoretical dense 5 cm 3D remain visible as supporting baselines. Results must state logical versus allocated
memory and must not hide cases where sparse occupied 3D is smaller.

"Real-time" is printed only when measured live end-to-end p95 latency on named hardware is below the 100 ms
sensor period. Model inference, preprocessing, grid, postprocessing, full pipeline, cached playback and display
FPS are reported separately.

## 9. Scope priority

Mandatory terrain outputs are drivable/non-drivable/unknown, ground and top elevation, slope, kerb/step,
overhang clearance and traversability. Pothole detection remains optional if real-data evidence is insufficient.

Mandatory object outputs are semantic category, clustering, bounding box, motion state, confidence and distance.
Tracking IDs, speed, trails and advanced occlusion handling are enhancements and do not block delivery.

## 10. Phase order from the current repository state

Completed work is preserved. The next implementation sequence is:

1. **Sequence 08 assembly and gate:** merge the five download chunks into the canonical path, exclude duplicate
   `(1)` scans, verify 4,071 scan/label pairs, and record the sequence-only split decision.
2. **Real-data regression:** rerun dataset verification, inspection, I/O, alignment, grid, derived and motion tests
   against the canonical dataset.
3. **LSK3DNet native acceptance:** checkpoint provenance, three-scan parity, model-mode frontend proof and RTX
   4050 stability benchmark.
4. **LSK3DNet 100 m extension:** direct sparse-space expansion, near-field consistency, long-range accuracy and
   VRAM/latency tests; use tiled fallback only if needed.
5. **Prediction cache and model evaluation:** cache sequence 08, run semantic metrics and validate no-GPU cached
   mode.
6. **Two-mode frontend:** Terrain Analysis and Object Detection with adaptive representation in both.
7. **Separate analytics dashboard:** `/dashboard`, generated metrics, compact main-screen KPIs.
8. **Benchmark and report:** memory, latency, accuracy, coarsening, density and reproducibility artifacts.
9. **Offline demo and final audit:** video, screenshots, limitations, licences, fresh-clone test and submission.

## 11. Minimum successful submission

- Verified sequence 08 and documented temporal split.
- LSK3DNet model path with provenance and parity evidence.
- Semantic and geometric coverage targeting 100 m.
- Variable-resolution 2.5D map with exact point accounting.
- Terrain Analysis and Object Detection views with adaptive rings visible in both.
- Honest static/moving/uncertain object state.
- Generated memory, accuracy and latency evidence.
- Separate analytics dashboard and cached offline operation.
- Deterministic offline demonstration video and explicit limitations.

