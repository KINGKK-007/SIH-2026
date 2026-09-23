# FoveaMap — Handoff & Current Plan Status

**Last updated:** 2026-09-24 · **Completed through:** Phase 6 (Gates 1, 5, 6 passed; Phases 2–4 code-complete, gates need the dataset)
**Audience:** the next engineer or coding agent (e.g. Claude Code on the teammate's machine that has the SemanticKITTI data).

This file tells you what is done, what is left, and exactly how to continue. The binding spec is still
`README.md`; the task list is `docs/PHASES.md` (15 phases, ticked). This file does not replace them.

---

## 1. Read in this order

1. **This file** (state + next steps).
2. `README.md` — the single source of truth (sections 0, 3, 5, 6, 8, 9, 11, 12, 13 matter most). Its Section 10 (Phases 0–8) is **superseded by `docs/PHASES.md`** (D-003).
3. `docs/PHASES.md` — the 15-phase plan with `[x]` on finished tasks and a status tracker at the bottom.
4. `docs/DECISIONS.md` — every deviation (D-001 … D-021). Read D-004, D-005, D-018, D-019, D-021 before touching related code.
5. `docs/PROGRESS.md` — per-task log with commit hashes and gate summaries, plus the data-machine checklist.

---

## 2. Repository & branch state

| Ref | Contains | State |
|---|---|---|
| `main` | Phase 1 (PR #1, merged) | tag `phase-1-complete` |
| `phase-2-4` | Phases 2, 3, 4 code (on top of `main`) | pushed; gates 2–4 await the dataset |
| `phase-5-7` | **everything in `phase-2-4`** + Phases 5 and 6 | pushed; tags `phase-5-complete`, `phase-6-complete` |

**Merge advice:** open one PR `phase-5-7 → main` (it already includes `phase-2-4`), and merge with a **merge commit**,
not squash/rebase, so the one-commit-per-task history and the phase tags stay valid. Then continue Phase 7 on a
new branch from `main` (e.g. `phase-7`).

**Commit rules (team preference):** commit under the local human git identity. **Do not add AI co-author trailers
(`Co-Authored-By: ...`) or "Generated with ..." lines** to commits or PR descriptions. Message format (README R8):
`<type>(<module>): T<phase>.<n> <summary>`, one task per commit, tag `phase-<N>-complete` when a gate passes.

---

## 3. Phase status (1–15)

Legend: ✅ gate passed · 🟡 code done, gate needs real data/GPU · ⏳ not started

| # | Phase | Status | What exists | What is left |
|---|---|---|---|---|
| 1 | Setup & environment | ✅ | skeleton (README §7), `make doctor`, pydantic config (`foveamap/config.py`, 7 YAMLs), Makefile, CI-style targets | — |
| 2 | Data acquisition & verification | 🟡 | `scripts/verify_data.py` (+ `foveamap/io/verify.py`), `foveamap inspect` | **[HUMAN]** download 04/07/08 (T2.1); run Gate 2 on real data |
| 3 | Data layer | 🟡 | `io/kitti.py`, `io/poses.py` (`relative_transform` exact), `io/labels.py` (Appendix A, cross-checked vs semantic-kitti-api, D-013), `io/sequence.py`, `eval/buckets.py`, `eval/density.py` v1 | run `pytest tests/io -q` with data present (`@data` tests) |
| 4 | Frame alignment & data stats | 🟡 | `foveamap align` (`eval/alignment.py`, strengthened gate D-018), `foveamap stats` (`eval/data_stats.py`) | run on real 04/08; commit `results/frame_alignment.json`, `results/data_stats.json`, `results/tables/data_stats.md` |
| 5 | Grid core | ✅ | `grid/presets.py` (V1–V5), `grid/engine.py` (integer addressing, `locate`, `rasterize`), `grid/accumulators.py` (sparse, D-019), `grid/backends/numpy_backend.py` | — (performance: see §6) |
| 6 | Grid invariants & packed layers | ✅ | `reduce_block`, I4/I6/I7 property suites (mutation-checked), `grid/layers.py` (12-byte layout, class rules, D-021) | — |
| 7 | Baselines, memory, oracle render | ⏳ **next** | stubs: `grid/baselines.py`, `grid/memory.py`, `models/oracle.py`, `pipeline/runner.py`, CLI `render`/`memory`; helpers ready: `GridLayers.nbytes()`, `logical_mask`, `viz/figures.py` | T7.1–T7.4 (P0), T7.5 C++/numba (P2, **triggered**, see §6) |
| 8 | Model selection & integration | ⏳ | stub `models/rangeview.py`, `models/geom.py`; `configs/model.yaml` (nulls); `io/labels.learning_to_raw` | needs GPU + data; T8.1 selection gate (SalsaNext → CENet → RangeNet++) |
| 9 | Prediction cache & model eval | ⏳ | stubs `models/cache.py`, `scripts/cache_predictions.py` | needs GPU (or Colab) once; then CPU-only |
| 10 | Motion & objects | ⏳ | stubs in `motion/`; **reference v0 code** in `src/foveamap_legacy/motion/` (residual + DBSCAN) | port to spec (README 6.4), tune on 04/07 only |
| 11 | Derived layers & hazards | ⏳ | stubs in `derived/`; v0 reference in `src/foveamap_legacy/derive/` (slope, step, clearance, hazards) | port to integer layers + halo (README 6.7–6.8) |
| 12 | Benchmarking | ⏳ | `eval/*` stubs; data stats and alignment already produce JSON | `make benchmark`, `make report` (R1: no hand-typed numbers) |
| 13 | Dashboard backend & scaffold | ⏳ | stubs `server/*`, `src/dashboard/README.md` | FastAPI + Socket.IO + React/Vite; protocol frozen in T7.4 first |
| 14 | Dashboard panels | ⏳ | v0 matplotlib composer in `src/foveamap_legacy/dashboard/app.py` (reference only) | README §13 |
| 15 | Video, tests, polish | ⏳ | stub `viz/video.py` | README §13.7, Appendix B; remove `foveamap_legacy` (D-004) |

Test suite today: `make test-fast` → **632 passed, 5 deselected (`@data`) in ~37 s**; `make lint`, `make typecheck` clean.

---

## 4. Immediate next steps on the data machine (in order)

### 4.1 Environment

```bash
git fetch origin && git checkout phase-5-7     # or main after the PR is merged
make install                                    # venv + requirements.txt + pip install -e .
make doctor                                     # expect 0 failures (GPU/C++ may warn)
```

`requirements.lock` in the repo was produced on Windows/CPU (D-008); on another platform install from
`requirements.txt` and regenerate the lock (`pip freeze --exclude-editable > requirements.lock`) in its own commit.

### 4.2 Data (Phase 2 T2.1 `[HUMAN]`)

Extract sequences **04, 07, 08** (velodyne + labels + calib + poses) into `data/dataset/` per README §5.1
(or point `FOVEAMAP_DATA_ROOT` at the dataset root). Poses may be in `poses/<seq>.txt`; `verify_data.py` links them.

### 4.3 Close Gates 2, 3, 4 (commands are also in `docs/PROGRESS.md`)

```bash
make doctor ARGS="--require-data"
python scripts/verify_data.py --sequences 04 07 08 --json results/data_verification.json
python -m foveamap.cli inspect --sequence 08 --idx 0
python -m pytest tests/io -q
python -m foveamap.cli align --sequences 04 08 --pairs 50
python -m foveamap.cli stats --sequences 04 07 08
python -m pytest -m data -q
```

Gate 4 verdict must be **PASS** (not FAIL/INCONCLUSIVE). If it fails, the transform is wrong — **do not raise the
threshold** (R2); see D-018 and `eval/alignment.py`. Paste summaries into `docs/PROGRESS.md`, commit the results
JSON/tables/plots, tick T2.1 and the gates in `docs/PHASES.md`, tag `phase-2-complete`, `phase-3-complete`, `phase-4-complete`.

### 4.4 Phase 7 (P0) — the next build phase

Spec: `docs/PHASES.md` Phase 7, README §6.5, §6.9, §12.4, §13.3. Build on what exists:

| Task | Build | Use |
|---|---|---|
| T7.1 | `grid/baselines.py` (uniform specs through the same engine: `spec_from_rings(name, [(R, s)])` already works) and `grid/memory.py` `memory_report`: dense 3D (theoretical), sparse 3D (measured unique voxel keys × (8+1) B), uniform 2.5D (`uniform_5cm` × 12 B), FoveaMap (`GridLayers.nbytes()` + logical/allocated), RSS delta via `psutil`; tests `tests/grid/test_memory.py` (I9: allocated counts == array sizes, logical ≤ allocated). Never hard-code the reduction factor (R1). | `GridSpec.logical_cells()/allocated_cells()`, `finalize`, `LAYER_DTYPE` |
| T7.2 | `models/oracle.py` (`OracleModel.predict` → `Prediction(raw_ids uint16, conf=255)`, D-014), minimal `pipeline/runner.py` (oracle mode, per-stage `perf_counter_ns` timers), CLI `render` → top-down PNG per frame: height-shaded, class palette README §13.8, ring boundaries + cell-size labels, +x up / +y left | `io.labels.raw_to_super`, `grid.engine.rasterize`, `grid.layers.finalize`, `grid.engine.logical_mask`, `viz/figures.py` style |
| T7.3 | Preliminary cost of coarsening (oracle, 50 scans of 04/07): point back-projection accuracy per distance bucket for `fovea_default`, `ps_literal`, `uniform_5cm`, `uniform_20cm` | `grid.layers.cell_layers` on **sparse** accumulators (no 16 M-cell dense arrays), `eval.buckets` |
| T7.4 | Freeze `docs/design/protocol.md` from README §13.3 (unblocks Phase 13) | — |
| T7.5 (P2, **triggered**) | Fast grid path (numba first, C++ optional) that is **bit-identical** to `NumpyBackend` (add a parity test reusing `tests/grid/test_differential.py` / `test_consistency.py` strategies) | `grid/accumulators.reduce_points` is the contract |

Gate 7 commands (need data): `pytest tests/grid -q`; `python -m foveamap.cli render --mode oracle --sequence 08 --frames 0 1 2 --preset fovea_default`; `python -m foveamap.cli memory --sequence 08 --idx 0`; grid timing over 100 frames → `results/latency_grid_prelim.json`; `docs/design/protocol.md` exists.

Tip: everything in Phase 7 can be developed and unit-tested **without** real data using the synthetic sequence
(`foveamap.io.synthetic.write_kitti_sequence`, fixture `synthetic_root` in `tests/conftest.py`), then run the gate on real data.

### 4.5 After Phase 7

Track B (Phases 8–9) needs a GPU once; everything after uses the cache (R13). Phases 10–11 can start after Gate 7 +
Gate 9 (10) / Gate 6 (11). Phase 13 can start as soon as T7.4 freezes the protocol. Follow `docs/PHASES.md`.

---

## 5. What is implemented (quick API map)

| Module | Public API |
|---|---|
| `foveamap.config` | `load_config(dir)`, `load_yaml_config`, `m_to_mm` (exact metre→mm), `GridConfig`, `ModelConfig`, `MotionConfig`, `DerivedConfig`, `HazardConfig`, `BenchmarkConfig`, `DashboardConfig` |
| `foveamap.io.kitti` | `load_scan_bin` → `(xyz f32, remission f32)`, `load_label` → uint32, `split_label` |
| `foveamap.io.poses` | `Calib`, `load_calib`, `load_poses` (T,4,4), `relative_transform(calib, poses, i, j)` = `inv(Tr) inv(P_i) P_j Tr`, `transform_points`, `rotation_angle_deg` |
| `foveamap.io.labels` | `raw_to_super` → `(super uint8, moving bool)`, `is_safety_critical`, `raw_to_learning`, `learning_to_raw`, `LABELS`, `raw_name`, `raw_color_rgb`, class constants |
| `foveamap.io.sequence` | `Sequence(root, seq, with_labels, frame_stride)` → `Scan` records; `.relative_transform(i, j)` |
| `foveamap.io.synthetic` | `write_kitti_sequence(root, seq, n_frames, seed, poses_in_poses_dir, speed_mps)` (test/demo data only, D-015) |
| `foveamap.io.verify` | `verify_dataset`, `verify_sequence`, `format_report` |
| `foveamap.eval` | `buckets.bucket_of_range` (D-016), `density.points_per_range_bin`, `alignment.alignment_check`, `data_stats.data_stats` / `stats_table_md` |
| `foveamap.grid.presets` | `RingSpec`, `GridSpec` (`logical_cells`, `allocated_cells`, `per_ring_shape`, …), `spec_from_rings`, `load_preset`, `validate_preset`, `PresetError` |
| `foveamap.grid.engine` | `quantize_mm`, `locate`, `world_to_cell`, `cell_to_corner_mm`, `cell_to_center_mm`, `is_logical_cell`, `logical_mask`, `prepare_points`, `rasterize(spec, xyz_m, super_cls, moving, conf, min_range_mm)` |
| `foveamap.grid.accumulators` | `RingAccumulators` (sparse; `to_dense`, `from_dense`, `equals`), `GridAccumulators`, `FrameCounters`, `reduce_points`, `reduce_cells`, `reduce_block` |
| `foveamap.grid.layers` | `LAYER_DTYPE` (12 B), `cell_layers`, `classify`, `finalize` → `GridLayers` (`layer`, `nbytes`), flag constants |
| CLI (`python -m foveamap.cli …`) | `inspect`, `align`, `stats` work; `render`, `memory`, `--mode …` are Phase 7/9/13 stubs |

---

## 6. Known issues, lessons and gotchas

- **Grid speed (T7.5 trigger):** NumPy reference path ≈ 57 ms per 120 k points on the Windows laptop (> 15 ms budget).
  Keep `NumpyBackend` as the correctness oracle; add a numba/C++ fast path proven bit-identical (I8-style test).
  Report latency honestly (R5/L14): never "real-time" without a measured end-to-end p95 < 100 ms.
- **Alignment gate (D-018):** the README's class set (building/road/vegetation) accepts wrong transforms on flat
  roads; the gate also requires compact structure (pole/trunk/sign) and an identity control that must fail.
- **Property tests must be able to fail:** the first I4/I7 strategy (sparse random points) missed real bugs; the
  current clustered + boundary strategy was mutation-checked. Keep ≥ 200 examples (README 11.3); reduce example
  *size* if slow, never the count below 100.
- **float32 scans:** always widen to float64 before `* 1000` (NumPy 2 keeps float32 with Python scalars).
- **Sparse accumulators (D-019):** never allocate dense `uniform_5cm` accumulators or layers per frame except for
  memory accounting; use `cell_layers` on sparse accumulators for evaluation.
- **Legacy code:** `src/foveamap_legacy` (v0, 311 tests in `tests/legacy`) is reference only (motion, derived
  layers, hazards, video). Port ideas, not code verbatim; delete it in Phase 15 (D-004).
- **Tuning discipline (L2):** tune only on 04 and 07; report only on 08.
- **Tooling:** `ruff` skips E501 (black enforces line length); `make test-fast` must stay < 60 s; `@data`, `@gpu`,
  `@slow`, `@cpp` markers gate heavy tests; `FOVEAMAP_DATA_ROOT` overrides `data/dataset`.
- **Make on Windows:** `mingw32-make` works with the Makefile; flags go through `ARGS="..."` (D-006).

---

## 7. Prompt to start the next agent

```text
You are continuing the FoveaMap repository. Read HANDOFF.md first, then README.md (binding spec),
docs/PHASES.md (task list, 15 phases), docs/DECISIONS.md and docs/PROGRESS.md.
Run `make doctor`. Then: (1) if SemanticKITTI 04/07/08 is present, close Gates 2-4 with the commands in
HANDOFF.md §4.3; (2) start Phase 7 (HANDOFF.md §4.4) on a branch from main. Follow the per-task loop in
docs/PHASES.md 10.0: tests first for grid/io/derived code, one commit per task with
`<type>(<module>): T<phase>.<n> ...`, tick the checkbox, append to docs/PROGRESS.md, log deviations as D-### in
docs/DECISIONS.md. Never hard-code result numbers, never weaken a test or gate, never tune on sequence 08.
Commit under the human's git identity with no AI co-author trailers. Ask the user only for [HUMAN] steps.
```
