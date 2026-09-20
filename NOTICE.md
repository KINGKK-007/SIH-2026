# NOTICE — FoveaMap

FoveaMap is © 2026 Kanav Kumar, licensed under Apache 2.0 (see `LICENSE`).

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

**FRNet** (primary segmenter, Phase 3)
- Authors: Xiangxu Lin et al.
- Repository: https://github.com/Xiangxu-0103/FRNet
- Paper: https://arxiv.org/abs/2312.04484
- Licence: Apache 2.0
- Checkpoints: hosted on Google Drive per the FRNet README; **not** included here.

**CENet** (fallback segmenter, Phase 3)
- Authors: Huixin Cheng et al.
- Repository: https://github.com/huixiancheng/CENet
- Licence: MIT
- Checkpoints: hosted on Google Drive; **not** included here.

**SalsaNext** (fallback segmenter, Phase 3)
- Authors: Tiago Cortinhal et al.
- Repository: https://github.com/TiagoCortinhal/SalsaNext
- Licence: MIT (check repo for current status)
- Checkpoints: check repository; **not** included here.

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
