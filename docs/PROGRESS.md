# Progress log

One line per task (README R10): `date · task · commit · outcome`. Gate summaries are appended when a phase completes.

- 2026-09-23 · T1.1 · bced375, 7dc4088 · v0 code moved to `foveamap_legacy` (311 tests kept); README Section 7 skeleton with Section 9 stubs, MIT licence, Makefile, DECISIONS D-001..D-011.
- 2026-09-23 · T1.2 · 9eb56c1 · `venv/` on Python 3.12 (Windows, torch 2.14.0+cpu); `import foveamap, torch, numpy, scipy, fastapi` OK; `requirements.lock` written (platform-specific, D-008).
- 2026-09-23 · T1.3 · 4fe1509 · `scripts/doctor.py` + tests; GPU/C++ warn, data/weights warn until `--require-data`/`--require-weights`.
- 2026-09-23 · T1.4 · 183093b · `foveamap/config.py` + seven default YAMLs; unknown keys rejected; exact metre->mm; 74 config tests pass.
- 2026-09-23 · T1.5 · b150652 · `make test` (460 passed), `make lint`, `make typecheck` clean; D-012.

## Gate 1 — passed 2026-09-23 (Windows 11, Python 3.12.4, mingw32-make)

- `make doctor`: 27 checks, 20 ok, 7 warnings, 0 failures. Warnings: gpu (CUDA not available), cmake (not on PATH, optional C++), data:04/07/08 (not downloaded; Phase 2), weights (no model yet; Phase 8), cpp backend (not built).
- `pytest tests/test_config.py`: 74 passed.
- `make test`: 460 passed (149 new + 311 legacy). `make test-fast`: 460 passed in ~21 s.
- `make lint`: all checks passed. `make typecheck`: no issues in 15 source files.

**Next:** Phase 2 needs the dataset (teammate's machine). Work that does not need data can start on Phases 3 and 5 code/tests with synthetic inputs; their gates run once data is present.

## Phases 2–4 — code complete 2026-09-23; gates need the real dataset

All code below is tested on the synthetic SemanticKITTI-format fixture (D-015). The `@data` tests skip until
`data/dataset` (or `$FOVEAMAP_DATA_ROOT`) holds sequences 04/07/08.

- 2026-09-23 · T3.1 · 92927ea · `load_scan_bin`, `load_label`, `split_label`; real-length check is `@data`.
- 2026-09-23 · T3.2 · e5032de · calib/poses parsing, `relative_transform` exactly per README 5.3; property test proves frame semantics.
- 2026-09-23 · T3.3 · dccc3c4 · `raw_to_super` from Appendix A; cross-check vs semantic-kitti-api yaml: identical (D-013).
- 2026-09-23 · fixture · 5ccbfb8 · synthetic KITTI-format sequence generator (D-015).
- 2026-09-23 · T3.4 · b8174a4 · `Sequence` (lazy, strided, poses in either location, times fallback).
- 2026-09-23 · T3.5 · 53a38fb · `bucket_of_range` (D-016), `points_per_range_bin`.
- 2026-09-23 · T2.2 · 9b89157 · `scripts/verify_data.py` (+ `foveamap.io.verify`); corrupted-copy tests; D-017 (Phase 3 before Phase 2 tools).
- 2026-09-23 · T2.3 · b4c7794 · `foveamap inspect` prints stats/pose and writes the bird's-eye PNG.
- 2026-09-23 · T4.1 · 02bf3a5 · `foveamap align`; gate strengthened with compact structure + identity control (D-018).
- 2026-09-23 · T4.2 · 6a9165a · `foveamap stats` -> `results/data_stats.json`, `results/tables/data_stats.md`.

`make test-fast`: 539 passed, 5 deselected (`@data`).

### To run on the data machine (closes Gates 2, 3 and 4)

```bash
make doctor ARGS="--require-data"
python scripts/verify_data.py --sequences 04 07 08 --json results/data_verification.json
python -m foveamap.cli inspect --sequence 08 --idx 0
python -m pytest tests/io -q
python -m foveamap.cli align --sequences 04 08 --pairs 50
python -m foveamap.cli stats --sequences 04 07 08
python -m pytest -m data -q
```

Then paste the summaries here, commit `results/frame_alignment.json`, `results/data_stats.json`,
`results/tables/data_stats.md` and the plots, and tag `phase-2-complete`, `phase-3-complete`, `phase-4-complete`.
If Gate 4 is not PASS, do not raise the threshold (R2): the transform is wrong; see `foveamap.eval.alignment`.

## Phase 5 — Grid Core

- 2026-09-24 · T5.1 · e145a58 · presets, V1–V5, closed-form counts (fovea_default 910,000 logical / 1,130,000 allocated asserted by test, not typed in code).
- 2026-09-24 · T5.2 · b6f09e8 · integer-only addressing; I2 on boundary-adversarial points, I5 exhaustive on tiny presets.
- 2026-09-24 · T5.3 · 779ad77 · NumPy reference rasteriser, sparse accumulators (D-019); I1, I10. Grid-stage time on this laptop ~57 ms per 120k points (> 15 ms budget → T7.5 fast path).

### Gate 5 — passed 2026-09-24

`pytest tests/grid/test_presets.py tests/grid/test_addressing.py tests/grid/test_conservation.py -q`: 55 passed (V1–V5, I1, I2, I5, I10; code-computed counts equal the closed forms).
Built before Gate 4 (D-020).

## Phase 6 — Grid Invariants & Packed Layers

- 2026-09-24 · T6.1 · ff52cbe · reduce_block; I4 (fine->coarse bit-identity, independent code paths), I6 (permutation, determinism), I7 (naive dict implementation incl. uniform presets); hypothesis 200 examples/property, fixed seed; mutation-checked (min->max in reduce_block and a closed ring boundary are both caught).
- 2026-09-24 · T6.2 · 5e27380 · packed 12-byte layers, safety/majority class rules, overhang clearance, flags; D-021 tie-breaks.

### Gate 6 — passed 2026-09-24

`pytest tests/grid -q --ignore=tests/grid/test_memory.py --ignore=tests/grid/test_cpp_parity.py`: 93 passed (I1–I7, I10, layer tests).

## Phase 7 — Baselines, Memory Accounting & Oracle Render

- 2026-09-25 · T7.1 · 175a972 · `grid/baselines.py` (uniform_spec single-ring through same engine); `grid/memory.py` (memory_report: dense 3D theoretical, sparse 3D measured, uniform 2.5D theoretical, FoveaMap measured); 12 new tests in `tests/grid/test_memory.py` all pass. I9 confirmed: allocated counts == array sizes, logical ≤ allocated.
- 2026-09-25 · T7.2 · bab30b8 · `models/oracle.py` (OracleModel: GT labels + conf=255); `pipeline/runner.py` (PipelineRunner oracle mode with per-stage perf_counter_ns timers); CLI `render` writes top-down PNGs height-shaded + class-coloured + ring boundaries; CLI `memory` prints the four-representation report. Gate 7 render verified: 3 PNGs for frames 0,1,2 written. Memory report: dense=2.38 GB, sparse=261 KB, uniform 2.5D=183 MB, FoveaMap=12.93 MB, reduction=14.2×.
- 2026-09-25 · T7.3 · 0ffa532 · `eval/coarsening.py` (cost_of_coarsening: back-projection error per distance bucket; latency_report_grid: rasterize+finalize timing over 100 frames); `results/latency_grid_prelim.json` written. Grid-stage p95=133 ms > 100 ms → T7.5 triggered.
- 2026-09-25 · T7.4 · 880e185 · `docs/design/protocol.md` frozen (Socket.IO `frame_update` event schema with all fields from FrameResult; client commands; versioning policy). Unblocks Phase 13.

### Gate 7 — partial (T7.5 pending, T7.1-T7.4 verified)

- `pytest tests/grid -q`: **105 passed** (includes new test_memory.py; I1–I7, I9, I10).
- `python -m foveamap.cli render --mode oracle --sequence 08 --frames 0 1 2 --preset fovea_default`: 3 PNGs written ✅
- `python -m foveamap.cli memory --sequence 08 --idx 0`: four-representation report printed ✅
- `results/latency_grid_prelim.json` exists; p95=133 ms → T7.5 triggered (see DECISIONS.md) ✅
- `docs/design/protocol.md` exists ✅

**T7.5 note:** p95 grid latency = 133 ms > 100 ms threshold; numba/C++ fast path to be implemented (T7.5, tier P2). The NumPy backend remains the correctness oracle.

## Phase 8 — Segmentation Model Selection & Integration

- 2026-09-25 · T8.2 · D-022/D-023/D-024/D-025: production model switched to LSK3DNet (sparse-voxel network, CVPR 2024); `configs/model.yaml` + `src/foveamap/config.py` updated with `LSK3DNetConfig` (family: sparse_voxel); `src/foveamap/models/lsk3dnet.py` (`LSK3DNetModel`) implemented with single-scan batching, fp16 autocast, empty_cache, and crop out-of-range protection (D-024); vendored `LSK3DNet-main/` placed at repo root and imported in-place (D-025); `docs/MODEL_CARD.md` and `configs/weights/WEIGHTS.md` created with RTX 4050 target setup guide. 74 config tests pass. `LSK3DNetModel` awaits checkpoint download and GPU stack build on friend's RTX 4050 machine.

## Phase 9 — Prediction Cache & Model Evaluation

- 2026-09-25 · T9.1 + T9.2 · `scripts/cache_predictions.py` implemented (resumable, per-sequence meta.json with SHA-256 and platform provenance); `src/foveamap/models/cache.py` (`CachedModel`) implemented and unit tested (`tests/models/test_cache.py`, 3/3 passed, no GPU needed); `src/foveamap/pipeline/runner.py` and CLI `render` wired to support `--mode cached`. All 653 test suite items pass.

## Phase 13 — Dashboard Backend & Frontend Scaffold

- 2026-09-25 · T13.1 · `src/foveamap/server/protocol.py`: `serialise_frame_result` implemented conforming to frozen Socket.IO `frame_update` schema; `src/foveamap/server/render.py`: `ring_textures` server-side multi-layer rendering (`class`, `height`, `traversability`, `moving`, `confidence`) with orientation convention `row = N_k - 1 - ix`, `col = N_k - 1 - iy`.
- 2026-09-25 · T13.2 · `src/foveamap/server/app.py` & `src/foveamap/server/sockets.py`: FastAPI + `python-socketio` ASGI server; `PlaybackManager` background task supporting `play`, `pause`, `step`, `seek`, `speed` across modes (`oracle`, `cached`, `live`); REST API (`/api/health`, `/api/state`, `/api/presets`, `/api/play`, etc.); CLI `foveamap serve` subcommand added.
- 2026-09-25 · T13.3 · `src/dashboard`: React 18 + Vite + TypeScript application scaffolded; dark glassmorphic UI; components: `Header` (mode badge, display FPS, sequence/preset metadata), `MapView` (multi-ring canvas compositing with pan/zoom, coordinate hover readout, layer selector), `PlaybackControls` (play/pause, frame scrubber, speed selector), `MemoryMeter` (four-representation log-scale comparison with reduction factor badge), `LatencyPanel` (per-stage stacked bar against 100 ms period); `socket.io-client` connected to `/fovea`.
- 2026-09-25 · T13.4 · Python tests `tests/server/test_render_and_protocol.py` (texture shapes & schema serialization) and `tests/server/test_server_e2e.py` (REST endpoints, asyncio Socket.IO playback loop & frame emission) implemented and passing.

### Gate 13 — passed 2026-09-25

- `pytest tests/server -v`: **4 passed** (texture rendering, protocol serialisation, REST endpoints, asyncio playback loop & client frame updates).
- `npm run typecheck`: **0 errors**.
- `npm run build`: **succeeded** (Vite build output in `src/dashboard/dist/`).
- Python test client confirmed `create_app()` serves `src/dashboard/dist/index.html` at `GET /` with HTTP 200.
- Full test suite: **657 passed**, 0 failed.

## Phase 11 — Derived Layers & Synthetic Hazards

- 2026-09-25 · T11.1 · `src/foveamap/derived/halo.py`: `compute_halo` (fine->coarse block reduction into inner holes) and `extract_padded_ring` (coarse->fine nearest-cell replication into boundary pads) implemented.
- 2026-09-25 · T11.2 · `src/foveamap/derived/slope.py`: `compute_slope_deg` and `compute_slope` (central difference in X and Y, degrees, `FLAG_STEEP` bit 5 set when slope > max_slope_deg).
- 2026-09-25 · T11.3 · `src/foveamap/derived/step.py`: `compute_step_height_mm` and `compute_step` (max step across 4-neighbours with >= min_points_step, `FLAG_KERB` bit 4 set in [kerb_min, kerb_max]).
- 2026-09-25 · T11.4 · `src/foveamap/derived/clearance.py`: `compute_clearance` (checks overhang vertical gap against vehicle_height + margin, sets `FLAG_LOW_CLEARANCE` bit 6).
- 2026-09-25 · T11.5 · `src/foveamap/derived/traversability.py`: `compute_traversability` and `traversability_map` (3-state categories: TRAVERSABLE=1, NON_TRAVERSABLE=2, UNKNOWN=0; sets `FLAG_TRAVERSABLE` bit 3).
- 2026-09-25 · T11.6 · `tests/derived/test_synthetic_terrain.py`: synthetic ramp slope recovery, kerb step detection at fine vs coarse, overhang clearance, and empty cell unknown status verified.
- 2026-09-25 · T11.7 · `src/foveamap/derived/hazards.py`: `inject_hazards` (seeded pothole, kerb, and overhang injections into candidate drivable scans) and `check_hazard_detected`.
- 2026-09-25 · T11.8 · `src/foveamap/eval/hazard_eval.py`: Wilson 95% confidence intervals, summary aggregations by type, ring, and range bucket; saved to `results/hazard_metrics.json` and `results/plots/hazard_detection_by_ring.png` (750 injections evaluated across Sequence 08).
- 2026-09-25 · T11.9 · `src/foveamap/derived/zoom.py`: `render_zoom` implemented; real kerb comparison plot saved to `results/plots/zoom_kerb_fine_vs_coarse.png` for Sequence 08 Frame 0.
- 2026-09-25 · T11.10 · Pipeline integration: `PipelineRunner` wired to run `compute_derived_layers` and record `derived_ms`.

### Gate 11 — passed 2026-09-25

- `pytest tests/derived -v`: **13 passed**, 0 failed.
- Kerb visibly detected at fine resolution in `results/plots/zoom_kerb_fine_vs_coarse.png`.
- `results/hazard_metrics.json` exists with n = 750 (250 per hazard type, >> 200 required) and Wilson 95% CIs.
- Rendered traversability map saved to `results/plots/traversability_08_000000.png`.
- Full test suite: **671 passed**, 0 failed.

## Phase 10 — Motion & Object Detection

- 2026-09-25 · T10.1 · `src/foveamap/motion/residual.py`: Spherical range-image projection ($64 \times 1024$, $+3^\circ$ to $-25^\circ$), ego-motion compensated previous scan projection ($T_{\text{vel}, t \leftarrow t-\text{gap}}$), windowed depth residual search ($3 \times 3$), static-consistency threshold $\tau(r) = \tau_0 + \tau_1 \cdot r$, and occlusion/disocclusion handling.
- 2026-09-25 · T10.2 · `src/foveamap/motion/cluster.py`: Range-scaled Euclidean clustering ($\varepsilon(r) = \varepsilon_0 + \varepsilon_1 \cdot r$) on candidate movable classes, accelerated via `scipy.sparse.csgraph.connected_components`. `src/foveamap/motion/boxes.py`: `oriented_box` minimum-area oriented bounding box in ground plane via vectorized $0^\circ$–$90^\circ$ angle sweep.
- 2026-09-25 · T10.3 · `src/foveamap/motion/pipeline.py`: `estimate_motion` supporting both ground-truth oracle path (raw IDs 252-259, GT instance IDs) and geometric motion path; wired into `PipelineRunner` (`src/foveamap/pipeline/runner.py`) to populate `FrameResult.objects` and `timings_ms["motion_ms"]`.
- 2026-09-25 · T10.4 · `src/foveamap/motion/tracker.py`: `ClusterTracker` with multi-frame velocity estimation, ego-compensated centroid tracking, gating ($2.0\text{ m}$), and temporal hysteresis ($2$ hits in last $3$ frames); vulnerable road users (VRUs: person, bicyclist, motorcyclist) elevated to `DYNAMIC` and `safety_critical = True`.
- 2026-09-25 · T10.5 · `scripts/run_phase10_eval.py`: Tuning ablation over `frame_gaps ∈ {1, 2, 3, 5}` on dev sequences written to `results/tables/motion_ablation.md`, confirming gap 2 ($0.20\text{ s}$) as optimal trade-off (FPR on parked vehicles = $0.0000$).
- 2026-09-25 · T10.6 & T10.7 · `src/foveamap/eval/motion_eval.py`: Point-level moving IoU, parked vehicle FPR, and object recall/precision evaluated across 20 real Sequence 08 frames ($2.44\text{M}$ points, $318$ GT objects) and written to `results/motion_metrics.json` and `results/object_metrics.json`.
- 2026-09-25 · `docs/LIMITATIONS.md`: Added honest strengths and weaknesses summary for the geometric motion module.

### Gate 10 — passed 2026-09-25

- `pytest tests/motion -v`: **15 passed**, 0 failed.
- `results/motion_metrics.json` and `results/object_metrics.json` written with sample sizes ($2,444,397$ points, $318$ GT objects).
- Object recall: **78.3%**, precision: **83.8%**, F1: **81.0%** ($100\%$ recall/precision at close range $0$–$10\text{ m}$).
- Stationary vehicle false-positive rate: **0.43%** ($786$ false positives out of $176,519$ parked vehicle points).
- `results/tables/motion_ablation.md` written and frozen config committed.
- Honest strengths/weaknesses summary added to `docs/LIMITATIONS.md`.

## Sequence 08-only implementation refresh

- 2026-09-26 · planning · Added `docs/LOCKED_IMPLEMENTATION_PLAN.md`; production decisions, 100 m LSK3DNet extension tests, two-mode frontend, separate analytics dashboard, evaluation definitions and minimum submission scope are frozen in D-026–D-028.
- 2026-09-26 · data assembly · Merged the five timestamped sequence-08 chunks into `data/dataset/sequences/08` with space-saving NTFS hard links; excluded accidental duplicates `000699(1).bin` and `000711(1).bin`; retained all source chunks.
- 2026-09-26 · data verification · `python scripts/verify_data.py --sequences 08 --json results/data_verification_sequence08.json`: PASS, 4,071 scans, 4,071 labels, 4,071 poses, 4,071 timestamps; 50 sampled scans loaded successfully.
- 2026-09-26 · real-frame inspection · frame 000000 loaded with 123,389 points and wrote `results/plots/inspect_08_000000.png`.
- 2026-09-26 · I/O regression · `python -m pytest tests/io -q`: 62 passed.

### Sequence 08 data gate — passed 2026-09-26

## Sequence 08 real-data regression

- 2026-09-26 · alignment · 50 consecutive-frame pairs: specified static classes median 0.0463 m, compact structures median 0.0355 m, identity control 0.6488 m; threshold 0.15 m; PASS. Wrote `results/frame_alignment_sequence08.json`.
- 2026-09-26 · statistics · Scanned all 4,071 frames and wrote `results/data_stats_sequence08.json` and `results/tables/data_stats_sequence08.md`. The 60–100 m bucket contains 4,290,586 points, of which 4,214,576 are ground-truth `UNKNOWN`; long-range metrics must retain counts and unknown rates.
- 2026-09-26 · environment · Corrected the editable install to the active `(1)` repository and installed declared dependencies `psutil` and `python-socketio` (D-029).
- 2026-09-26 · regression fix · Restored `points_per_range_bin` after the Phase 12 density module had removed the T3.5 API (D-030); `tests/eval/test_buckets.py`: 7 passed; Ruff clean.
- 2026-09-26 · real-data tests · `python -m pytest -m data -q`: 5 passed, 684 deselected.

### Sequence 08 real-data regression gate — passed 2026-09-26

## LSK3DNet 100 m implementation — CPU/config portion

- 2026-09-26 · D-031 · Added explicit `native` and `extended_100m` coverage profiles. Production defaults to bounds [-100,100] m in x/y with `spatial_shape=[4000,4000,120]`, preserving 5 cm voxels; native remains available for three-scan parity.
- 2026-09-26 · geometry safety · `resolve_inference_geometry` deep-copies the upstream YAML, updates both model and dataset sparse shapes, and rejects an extended profile whose voxel size differs from the checkpoint.
- 2026-09-26 · verification · `tests/test_config.py` plus `tests/models/test_lsk3dnet_geometry.py`: 77 passed; Ruff clean. No model inference or GPU benchmark was run on this lower-spec development device.

### LSK3DNet 100 m implementation status — code/config complete; RTX 4050 evidence pending

The next local implementation phase is the two-mode frontend and separate `/dashboard`. Native parity, 100-scan stability, near-field agreement and long-range accuracy remain target-machine acceptance steps.

## Problem-statement frontend alignment

- 2026-09-26 · D-032 · Replaced the five equal-weight layer choices with two primary modes: Terrain Analysis and Object Detection. Supporting semantic, elevation, traversability, motion and confidence views are now contextual overlays within the relevant mode.
- 2026-09-26 · adaptive representation · Kept the four resolution boundaries, per-ring cell-size labels and 100 m coverage visible in both modes; added a compact live summary for adaptive-grid settings, pipeline FPS, latency and map memory.
- 2026-09-26 · analytics separation · Moved detailed memory accounting and latency breakdown to `/dashboard`; the header opens it in a separate tab and provides a return-to-map action.
- 2026-09-26 · verification · Installed the pinned dashboard dependencies with `npm ci`; `npm run typecheck`, `npm run lint` and `npm run build` pass. No model inference, CUDA check or GPU benchmark was run.

### Frontend alignment gate — passed 2026-09-26

The next implementation phase is evaluation and evidence tooling that can be developed locally, followed by the explicitly deferred RTX 4050 acceptance run for LSK3DNet accuracy, VRAM and latency.

## Unified semantic evaluation and evidence

- 2026-09-26 · D-033 · Implemented confusion-matrix IoU/accuracy metrics with absent-class handling and strict shape/class validation.
- 2026-09-26 · cached evaluation · Added streaming sequence evaluation over the locked 1271-4070 interval. It reports 19-class and four-superclass accuracy, six distance bands, per-class support, unknown prediction rates, cache provenance and exact point accounting through 100 m.
- 2026-09-26 · CLI · Added `foveamap evaluate`, producing `results/semantic_evaluation_sequence08.json` and `results/tables/semantic_evaluation_sequence08.md` from cached predictions without loading the model or CUDA.
- 2026-09-26 · verification · `python -m pytest tests/eval -q`: 22 passed; focused Ruff checks pass. Synthetic end-to-end coverage verifies cache loading, provenance, perfect-label mIoU and both output artifacts.

### Unified semantic evaluation gate — passed 2026-09-26

Production command after the RTX 4050 cache is copied into place:

```powershell
python -m foveamap.cli evaluate --sequence 08 --model lsk3dnet --cache-root data/cache/pred --data-root data/dataset --start 1271 --stop 4071 --json results/semantic_evaluation_sequence08.json --table results/tables/semantic_evaluation_sequence08.md
```

Next: implement the RTX 4050 acceptance runner for native parity, extended-100 m consistency, 100-scan stability, VRAM, latency and cache provenance.

## Frontend redesign integration

- 2026-09-26 · D-034 · Integrated the `frontend.md` application shell: collapsible sidebar, simplified live header, four primary KPI cards, hero LiDAR map, adaptive-ring panel, object list, terrain composition and dedicated Performance and Foveated-vs-Uniform views.
- 2026-09-26 · data integrity · Preserved the current Canvas renderer and Socket.IO pipeline. Cards and panels derive values from live frame payloads; missing benchmark values remain explicitly unmeasured. Production rings remain 5/10/20/40 cm per the locked implementation plan.
- 2026-09-26 · visual system · Applied the dark perception-console palette, neutral panels, restrained cyan/green/amber/red accents, responsive desktop/tablet/mobile layouts and a dark map canvas.
- 2026-09-26 · verification · Dashboard TypeScript, Oxlint and Vite production build all pass.

### Frontend redesign foundation — passed 2026-09-26

Remaining frontend polish from `frontend.md`: historical telemetry charts, interactive layer visibility checkboxes, ring/object map selection and ingestion of the final RTX 4050 evaluation artifacts.
