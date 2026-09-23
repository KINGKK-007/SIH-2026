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
