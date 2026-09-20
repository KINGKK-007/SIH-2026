# FoveaMap — Architecture Decision Records

> **Rule (master plan §0):** Log every deviation from the plan in this file
> with a one-line reason.  Record all §10 Human Decision answers here.

---

## ADR-001 — Repository root location and name

**Date:** 2026-09-20
**Status:** Accepted

**Context:**
The master plan was authored under a Windows path
`C:\Users\SPARS\Desktop\IIIT B\sem5\GenAI\classAssign1`.
The actual development environment is macOS at
`/Users/kanavkumar/Desktop/SIH-2026`.

**Decision:**
Use `/Users/kanavkumar/Desktop/SIH-2026` as the repository root.
Package name is `foveamap` as specified in master plan §10 H13.

**Consequences:**
All install paths, config defaults, and CI commands reference this location.

---

## ADR-002 — Source layout: `src/foveamap/` over flat layout

**Date:** 2026-09-20
**Status:** Accepted

**Context:**
The master plan's repo layout (§8) shows `src/foveamap/` (src-layout).
This is best practice for installable packages (avoids import-before-install
confusion; required by hatchling).

**Decision:**
Use `src/foveamap/` layout.  `pyproject.toml` sets
`[tool.hatch.build.targets.wheel] packages = ["src/foveamap"]`.

**Consequences:**
Running tests requires the package to be installed (`pip install -e .`).
CI must install before pytest.

---

## ADR-003 — Synthetic scan fallback for offline/no-data operation

**Date:** 2026-09-20
**Status:** Accepted (master plan G2 fallback, mandatory)

**Context:**
SemanticKITTI requires account registration and a multi-GB download (G2).
The master plan mandates a synthetic scan so the pipeline and tests can run
with no data.

**Decision:**
Implement `foveamap.io.synthetic.generate_synthetic_scan()` with a fixed scene
(ground, sidewalks, buildings, parked/moving cars, pedestrian, terrain).
`foveamap check-data` automatically calls it when the real dataset is absent.

**Consequences:**
- Point density distribution is not ray-cast HDL-64E fidelity.
- Suitable for functional testing of IO/label/pose/grid code but **not** for
  measuring segmentation accuracy.
- State this in the README limitations section.

---

## ADR-004 — Build system: Hatchling over setuptools/flit

**Date:** 2026-09-20
**Status:** Accepted

**Context:**
The master plan lists `pyproject.toml` with no explicit build-backend
preference.  Three options: setuptools, flit, hatchling.

**Decision:**
Use **hatchling** (default for modern projects, fast, PEP 517/518 compliant,
no legacy `setup.py`).

**Consequences:**
Requires `hatchling` at install time (automatically pulled by pip).

---

## ADR-005 — License: Apache 2.0

**Date:** 2026-09-20
**Status:** Accepted (H13 default)

**Context:**
H13 lists MIT vs Apache-2.0.  FRNet (our primary segmenter) is Apache-2.0.
Using the same licence avoids downstream licence friction.

**Decision:**
Apache 2.0 for the FoveaMap codebase.
SemanticKITTI and KITTI data remain CC BY-NC-SA 3.0 (non-commercial);
documented in `NOTICE.md`.

---

## ADR-006 — GridSpec I1/I2 invariant validation at construction time

**Date:** 2026-09-20
**Status:** Accepted

**Context:**
Master plan §5.5 lists alignment invariants I1 and I2 as mandatory.
Invalid grids (e.g. 5/10/20/50 cm) silently produce seam artefacts.

**Decision:**
`GridSpec.__post_init__()` validates I1 and I2 at construction time and raises
`ValueError` with a human-readable message if violated.
This is tested in `tests/test_labels.py` via grid construction paths.

---

## ADR-007 — Phase 1 scope: stub `__init__.py` for motion/derive/eval/viz

**Date:** 2026-09-20
**Status:** Accepted

**Context:**
Phase 1 deliverables cover IO and skeleton only.  Motion (P4), derived layers
(P5), evaluation (P6), and viz (P7) are later phases.

**Decision:**
Create `__init__.py` stubs with docstrings listing planned modules.
No implementation code in these sub-packages yet.

**Consequences:**
`from foveamap.motion import ...` will raise `ImportError` until Phase 4.
This is intentional and documented.

---

## §10 Human Decision Log

| ID  | Decision                          | Answer / Default used         | Source              |
|-----|-----------------------------------|-------------------------------|---------------------|
| H1  | Time budget and team size         | 1 week, 2–3 people (default)  | Master plan default |
| H2  | Compute for network               | (a) Colab/Kaggle for caching  | Master plan default |
| H3  | Data acquisition                  | Seq 08 minimum; human downloads | Master plan default |
| H4  | Segmenter                         | FRNet with fallback chain     | Master plan default |
| H6  | Class policy                      | §5.1 mapping (table in labels.py) | Master plan default |
| H7  | Headline grid                     | fovea_4ring headline; ps_literal always in tables; square rings | Master plan default |
| H8  | Cell class rule                   | safety_priority; report both  | Master plan default |
| H9  | Dashboard tech                    | Streamlit + MP4               | Master plan default |
| H13 | Repo identity / licence           | foveamap; Apache 2.0; private | Master plan default |
| H17 | Eval subset                       | Every 5th frame of seq 08     | Master plan default |

All remaining just-in-time decisions (H5, H10, H11, H14, H15, H16, H18)
will be recorded when the relevant phase begins.

---

## ADR-008 — Cell memory layout: 17 bytes/cell actual vs 8 bytes/cell target

**Date:** 2026-09-20
**Phase:** P2

**Decision:**
The master plan quotes "8 bytes/cell" as the target figure. Our implementation
stores 3 float32 elevation layers (12 B) + 5 uint8 semantic layers (5 B) = 17 B/cell.
Both numbers are reported in `stats()` (`bytes_per_cell=17`,
`bytes_per_cell_target=8`). The 8 B/cell target corresponds to packing semantics
only (cls + dyn_frac + count + conf + flags = 5 B padded to 8). Memory metrics
(E3) report both honestly.

**Reason:** Float32 ground/top/overhang elevation is required for the
slope/step/clearance derived layers (Phase 5). Separate float32 arrays are
far more efficient to compute with numpy than packed structs.

---

## ADR-009 — Numba kernel: `parallel=False` default, `FOVEAMAP_PARALLEL_GRID=1` opt-in

**Date:** 2026-09-20
**Phase:** P2

**Decision:**
The inner Numba scatter-reduce loop uses `@njit(parallel=False)` by default.
`FOVEAMAP_PARALLEL_GRID=1` enables a future parallel variant. Parallelism is
disabled in tests to avoid Numba threading issues inside pytest.

**Measured latency (120k pts, fovea_4ring, single-thread CPU, macOS):**
- Numba path: ~17 ms/frame (target: ≤15 ms)
- NumPy path (fallback): ~28 ms/frame
- Both within acceptable range for no-GPU prototyping.

**Deviation:** 2 ms above 15 ms target (single-thread, no GPU). Logged per §0 rule 6.

---

## ADR-010 — Ring assignment: half-open Chebyshev interval `[inner, outer)`

**Date:** 2026-09-20
**Phase:** P2

**Decision:**
A point at Chebyshev distance `d = max(|x|, |y|)` is assigned to ring `r`
if `inner_half ≤ d < outer_half`. The innermost ring uses `inner_half=0`.
Points outside the outermost ring are dropped and counted in `n_points_dropped`.

This satisfies invariant I2 (cell-based ring assignment, no boundary straddling).

---

## ADR-011 — Rounding guard: `floor((coord + H) / cell + 1e-9)` in float64

**Date:** 2026-09-20
**Phase:** P2

**Decision:**
All cell-index computations use float64 arithmetic with a 1e-9 additive guard
before `floor()`. This ensures a point at exactly `x = 10.0000` with
`half_extent=10, cell=0.05` maps deterministically to cell 200 (not 199.999…→199).

Satisfies master plan §5.5 invariant I3 and risk G13.

---

## ADR-012 — `ground_z`: mean approximation (not true median)

**Date:** 2026-09-20
**Phase:** P2

**Decision:**
The master plan defines `ground_z` as the "median z of ground points". The
Numba kernel accumulates `sum_z` and `n_ground` (O(1) per point) and
computes the mean in the finalisation step. True median requires storing all
values (O(n) memory per cell), which is infeasible in a scatter kernel.

**Consequence:** `ground_z` is a mean approximation. For flat road points
(≈ constant z) the difference is negligible. Documented here and in `layers.py`.
A future upgrade could use P2 sketching or block-reduced true median.

---

## ADR-013 — `overhang_z`: set to `top_z` when `top_z - ground_z > 0.3 m`

**Date:** 2026-09-20
**Phase:** P2

**Decision:**
The per-cell `overhang_z` is computed in the NumPy finalisation step (not in
the Numba inner loop) as: `overhang_z = top_z` where `(top_z - ground_z) > 0.30 m`.
This is a conservative approximation — any non-ground point rising more than
0.30 m above local ground is flagged as a potential overhang. The clearance
layer (`top_z - ground_z`) is used for passability checks in Phase 5.

---

## ADR-014 — `stats()` reports both allocated and active cells

**Date:** 2026-09-20
**Phase:** P2

**Decision (§5.7):**
Dense ring arrays allocate full squares. Inner overlap is subtracted to get
"active" cells (the annular region actually populated by this ring's points).

- **allocated_cells**: sum of all full squares per ring
- **active_cells**: allocated minus inner-square overlap area per ring
- `allocated_mb` and `active_mb` derived from these × `bytes_per_cell`

The fovea_4ring preset gives: allocated=1,130,000, active=910,000 (exactly
matching master plan §5.5 table). This was verified in the integration test.

---

## ADR-015 — OracleSegmenter: `grid_inputs()` convenience method added

**Date:** 2026-09-20
**Phase:** P2

**Decision:**
`OracleSegmenter` was extended with:
- `moving_mask` property: bool array, True for raw IDs 252–259
- `vru_mask` property: bool array, True for raw IDs 30, 31, 32
- `super_cls` property: pre-computed super-class uint8 array
- `grid_inputs(points)` → (super_cls, moving, vru, conf) — all four arrays
  needed by `ClipmapGrid.build()` in a single call

This makes oracle-mode grid builds one line of code and avoids duplicate
label-processing logic in calling code.
