# NOTICE — FoveaMap

FoveaMap is © 2026 FoveaMap contributors, licensed under the MIT License (see `LICENSE`).

---

## Dataset

**SemanticKITTI**
- Authors: J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, J. Gall
- Paper: "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences", ICCV 2019
- URL: https://semantic-kitti.org/
- Licence: **Creative Commons Attribution-NonCommercial-ShareAlike 3.0** (CC BY-NC-SA 3.0)
- ⚠️  SemanticKITTI data is **not** included in this repository. Download separately after
  accepting the licence at https://semantic-kitti.org/dataset.html

**KITTI Odometry benchmark** (velodyne scans and poses)
- Authors: A. Geiger, P. Lenz, C. Stachniss, R. Urtasun
- URL: https://www.cvlibs.net/datasets/kitti/
- Licence: CC BY-NC-SA 3.0 (non-commercial research use only)

---

## Pretrained Models (to be downloaded separately)

**LSK3DNet** (production segmenter — see `docs/DECISIONS.md` D-022, `docs/MODEL_CARD.md`)
- Authors: Tuo Feng, Wenguan Wang, Fan Ma, Yi Yang
- Paper: "LSK3DNet: Towards Effective and Efficient 3D Perception with Large Sparse Kernels", CVPR 2024 —
  https://arxiv.org/abs/2403.15173
- Repository: https://github.com/FengZicai/LSK3DNet (vendored in-place at `LSK3DNet-main/`)
- Licence: MIT
- Checkpoint: hosted per the upstream README's Model Zoo link; **not** included here — see
  `docs/MODEL_CARD.md` for the exact filename and SHA-256 verification command.

**Range-view candidates** (SalsaNext / CENet / RangeNet++ — documented only as the original Model Selection
Gate's comparison set, superseded by LSK3DNet per D-022; not used in production)
- SalsaNext — https://github.com/TiagoCortinhal/SalsaNext — MIT
- CENet — https://github.com/huixiancheng/CENet — MIT
- RangeNet++ (`lidar-bonnetal`) — https://github.com/PRBonn/lidar-bonnetal — MIT

---

## Python Dependencies (OSS, all permissive)

| Package | Licence |
|---|---|
| numpy | BSD 3-Clause |
| scipy | BSD 3-Clause |
| pyyaml | MIT |
| click | BSD 3-Clause |
| numba | BSD 2-Clause |
| matplotlib | PSF / BSD |
| pytest | MIT |

---

## Inspired by / design references

- `elevation_mapping_cupy` (Apache 2.0) — https://github.com/leggedrobotics/elevation_mapping_cupy
- `wavemap` (BSD 3-Clause) — https://arxiv.org/pdf/2306.01279
- KISS-ICP (MIT) — https://github.com/PRBonn/kiss-icp
- 4DMOS (MIT) — https://github.com/PRBonn/4DMOS
- semantic-kitti-api (MIT) — https://github.com/PRBonn/semantic-kitti-api

No source code from the above projects was copied into FoveaMap; they are referenced for
design inspiration and cited in docs.
