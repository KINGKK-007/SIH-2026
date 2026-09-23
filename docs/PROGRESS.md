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
