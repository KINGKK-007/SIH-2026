# Model card

D-022 (`docs/DECISIONS.md`): the production segmentation model is **LSK3DNet**, not a range-view network
(README L11's original choice). This overrides the README Section 6.3 candidate table for this project;
`docs/PHASES.md` T8.1 (the range-view Model Selection Gate) was not run.

## LSK3DNet

| Field | Value |
|---|---|
| Paper | [LSK3DNet: Towards Effective and Efficient 3D Perception with Large Sparse Kernels](https://arxiv.org/abs/2403.15173) (CVPR 2024) |
| Source repo | https://github.com/FengZicai/LSK3DNet — vendored at the FoveaMap repo root as `LSK3DNet-main/` (a zip snapshot, not a git checkout: there is no commit hash to record) |
| Licence | MIT (`LSK3DNet-main/LICENSE`), compatible with this project's MIT licence |
| Checkpoint | `opensource_9ks_s030_w64_0.pt` (SemanticKITTI val, reported 70.2% mIoU with TTA per the upstream README's Model Zoo table) — **[HUMAN]** download from the upstream README's Model Zoo OneDrive link and place at `configs/weights/opensource_9ks_s030_w64_0.pt`; record its SHA-256 in `configs/weights/WEIGHTS.md` (`python -c "from foveamap.models.lsk3dnet import checkpoint_sha256; print(checkpoint_sha256('configs/weights/opensource_9ks_s030_w64_0.pt'))"`) |
| Architecture config | `LSK3DNet-main/config/lk-semantickitti_sub_tta.yaml` (read as-is by the wrapper; not duplicated into `configs/`, README 3.1) — `input_dims=13`, `hiden_size=64`, `large_kernel_size=[9,9,9]`, `scale_list=[2,4,8,16]`, `num_classes=20`, voxel `spatial_shape=[2000,2000,120]` over `min/max_volume_space` ±50 m (x,y) / [-4, 2] m (z) at 5 cm/voxel |
| Wrapper | `src/foveamap/models/lsk3dnet.py` (`LSK3DNetModel`), single scan, no TTA voting, optional fp16 autocast forward |
| Input, not a range image | The model consumes raw points `(x, y, z, intensity)` plus a per-point surface normal computed from a 64x900 range-image projection (`fov_up=3°`, `fov_down=-25°`, matching the KITTI HDL-64E, L3) — the range image here is only a feature-extraction step, not the segmentation representation itself, so the `input:` block in `configs/model.yaml` (height/width/FOV/mean/std) is unused for this family and left null |
| Known accuracy gap by design (D-024) | Points outside the checkpoint's ±50 m/[-4, 2] m training crop are never scored by the model; they come back `UNKNOWN`, `conf=0`. This mechanically caps the 60-100 m distance bucket in T9.3 — report it, do not paper over it |
| mIoU / per-class IoU / inference latency | **TBD.** Filled in by T9.3 (`results/accuracy_model.json`, `results/tables/accuracy_by_distance.md`) and T9.4 (`results/latency_inference.json`) once the checkpoint is downloaded and `scripts/cache_predictions.py` has been run on the GPU machine (R1: no number is hand-written here) |

### Environment setup (both GPU machines: this dev machine's RTX A500 4 GB, and the RTX 4050 6 GB laptop used for the actual T9.1/T9.3/T9.4 runs, D-023)

LSK3DNet's `requirements.txt` pins `torch==1.11.0+cu113` / `spconv-cu113`, from 2022. Neither the RTX A500
(Ampere, compute capability 8.6) nor an RTX 40-series card (Ada Lovelace, compute capability 8.9) is
supported by a CUDA 11.3 build — **do not** `pip install -r LSK3DNet-main/requirements.txt` as-is. Install
current wheels matching the installed `torch`/CUDA instead:

1. **PyTorch** — a build for the driver's CUDA version already works here (`torch==2.5.1+cu121`, confirmed
   `torch.cuda.is_available()==True` on the RTX A500). On the RTX 4050 machine, install the current stable
   `torch` CUDA 12.x wheel from https://pytorch.org/get-started/locally/ the same way.
2. **spconv** — install the wheel tagged for the closest CUDA version at or below the installed CUDA
   toolkit (e.g. `pip install spconv-cu120` for a CUDA 12.1 `torch`); check
   https://github.com/traveller59/spconv for the currently published `spconv-cuXXX` tags, since the exact
   set changes over time. `cumm-cuXXX` installs automatically as its dependency.
3. **torch-scatter** — install the wheel matching the exact `torch`+CUDA combination from the PyG wheel
   index, e.g. `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html` (substitute
   the installed `torch.__version__` and CUDA tag).
4. **Remaining Python deps** used only by the data-loading/label path, not by inference itself:
   `pip install easydict pyyaml pyquaternion nuscenes-devkit`. (`LSK3DNet-main/requirements.txt` also
   lists packages for training, point-cloud completion (`chamfer`, `emd-ext`) and Jupyter that inference
   does not need — do not install the whole file.)
5. **The `c_gen_normal_map` C++ extension** (`LSK3DNet-main/c_utils`) — mandatory; there is no working
   pure-Python fallback (`LSK3DNet-main/utils/normalmap.py` keeps one only as a commented-out reference).
   Needs `pybind11`, `CMake >= 3.0`, a C++11 compiler with OpenMP, and Python dev headers.
   - **Recommended: WSL2 Ubuntu** (already installed on this dev machine, currently stopped; simplest
     path since the upstream build instructions and published pybind11/CMake tooling target Linux):
     ```bash
     pip install pybind11 cmake
     cd LSK3DNet-main/c_utils
     mkdir build && cd build
     cmake ..
     make -j4
     export PYTHONPATH=$PYTHONPATH:$(pwd)   # add to the shell that runs cache_predictions.py
     ```
     CUDA passthrough works the same way in WSL2 with a current NVIDIA driver, so `torch.cuda.is_available()`
     stays `True` inside the WSL2 environment.
   - **Native Windows** works too (MSVC + CMake, `pybind11` provides its own CMake config), but is more
     fragile in practice (mixing MinGW and MSVC-built Python extensions in the same interpreter causes ABI
     issues) — prefer WSL2 unless there is a specific reason not to.
6. Verify before running `scripts/cache_predictions.py`:
   ```bash
   python -c "import spconv, torch_scatter, c_gen_normal_map; print('ok')"
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

### Why this is a deviation, not a silent swap

`docs/PHASES.md` still lists T8.1 (the range-view Model Selection Gate) and T9.4's alternatives table as
"the optional sparse-conv reference on a small subset" — with LSK3DNet as the production model, that
framing is now backwards. If a lighter range-view network is later benchmarked for comparison, T9.4
should report it as the reference and LSK3DNet as the production result, not the other way around.
