# Setting Up LSK3DNet for FoveaMap (Phase 8)

> Model: **LSK3DNet** (CVPR 2024) — Large Sparse Kernel 3D Neural Network  
> Paper: <https://arxiv.org/abs/2403.15173>  
> Code: <https://github.com/FengZicai/LSK3DNet>

---

## 1. Prerequisites

LSK3DNet uses **spconv v2** (Sparse Convolution) which requires a CUDA-enabled GPU and matching
PyTorch/CUDA versions. The tested combination is:

| Package | Version |
|---------|---------|
| Python  | 3.8–3.10 |
| PyTorch | 1.11.0+cu113 |
| CUDA    | 11.3 |
| spconv  | spconv-cu113==2.1.21 |
| cumm    | cumm-cu113==0.2.9 |
| torch-scatter | 2.1.0 |

> [!WARNING]
> The FoveaMap environment uses Python 3.13. **LSK3DNet inference requires a separate conda
> environment** (Python 3.9, torch 1.11) because spconv-cu113 does not support Python 3.13.
> See Section 4 for the dual-environment setup.

---

## 2. Download the pretrained checkpoint

1. Visit the [LSK3DNet releases page](https://github.com/FengZicai/LSK3DNet/releases) or the
   Google Drive / Baidu Pan link in the README.
2. Download the **SemanticKITTI** checkpoint (filename: `lsk3dnet_semantickitti.pt` or similar).
3. Place it at:
   ```
   configs/weights/lsk3dnet_semantickitti.pt
   ```
   (create the `configs/weights/` directory if it does not exist).

The path must match `checkpoint:` in `configs/model.yaml`.

---

## 3. Clone the LSK3DNet source

The adapter imports `builder`, `network`, and `utils` from the LSK3DNet repository directly
(no separate install step required).

```powershell
# From the FOVEMAP root:
git clone https://github.com/FengZicai/LSK3DNet.git data/lsk3dnet_src
```

Then set the environment variable so the adapter can find it:

```powershell
$env:LSK3DNet_SRC = "data/lsk3dnet_src"
```

Or pass it explicitly:

```python
from foveamap.models.lsk3dnet_adapter import LSK3DNetModel
model = LSK3DNetModel("configs/weights/lsk3dnet_semantickitti.pt",
                      lsk3dnet_src="data/lsk3dnet_src")
```

---

## 4. Dual-environment setup (recommended)

Because spconv-cu113 requires Python ≤ 3.10 and the FoveaMap core uses Python 3.13,
run inference in a dedicated conda environment and cache predictions to disk (Phase 9).

```powershell
# Create the inference environment
conda create -n lsk3dnet python=3.9 -y
conda activate lsk3dnet

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 `
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install spconv-cu113==2.1.21 cumm-cu113==0.2.9
pip install torch-scatter==2.1.0 easydict pyyaml numpy scipy

# Clone LSK3DNet
git clone https://github.com/FengZicai/LSK3DNet.git data/lsk3dnet_src
```

Then run the cache-fill script (Phase 9 T9.1):

```powershell
conda activate lsk3dnet
python scripts/fill_pred_cache.py --sequence 08 --model lsk3dnet
```

The FoveaMap core (in its own environment) then reads the cached predictions without
needing spconv.

---

## 5. Model input/output specification

| Property | Value | Source |
|----------|-------|--------|
| Input format | Sparse voxel tensor | `network/voxel_fea_generator.py` |
| Feature dims | 13 (xyz + intensity + displacement + voxel-offset + normals) | `network/architecture.py` |
| Voxel size | 5 cm isotropic | `config/lk-semantickitti_sub_tta.yaml` |
| Spatial bounds (X, Y) | ±50 m | dataloader/pc_dataset.py |
| Spatial bounds (Z) | −4 m to +2 m | dataloader/pc_dataset.py |
| Grid shape | 2000 × 2000 × 120 | derived from bounds/voxel |
| Range image | 64 × 900, FOV +3° to −25° | dataloader |
| Output classes | 20 (19 valid + 0=ignore) | config |
| Output format | Per-point logits → argmax → `learning_map_inv` → raw ID | `evaluate.py` |
| Checkpoint ext | `.pt` | repo releases |
| Backbone | `largekernelseg` | `network/architecture.py` |

---

## 6. Spatial extent mismatch with FoveaMap grid

- **LSK3DNet** accepts points within ±50 m in X/Y.  
- **FoveaMap grid** extends to ±100 m (configured in `configs/grid.yaml`).

The adapter in `models/lsk3dnet_adapter.py` handles this by:

1. **Clipping** points to ±50 m before voxelisation.
2. **Filling** out-of-range points (50–100 m) with class=`UNKNOWN` (raw_id=0) and conf=0.
3. The grid then uses these UNKNOWN predictions for cells beyond 50 m — their `cls` will
   be `UNKNOWN` unless the oracle provides ground-truth labels.

> [!NOTE]
> The 50 m clip only affects the **`live`** and **`cached`** pipeline modes.  
> The **`oracle`** mode (used in Phase 7) is unaffected — it does not call the model.

---

## 7. Verification (after Phase 9 cache fill)

```powershell
# With the main foveamap environment active and cache filled:
foveamap render --mode cached --sequence 08 --frames 0 --model lsk3dnet
foveamap memory --sequence 08 --idx 0
```

---

## 8. File checklist

```
configs/
  model.yaml                   ← updated (name: lsk3dnet, checkpoint: ..., input: ...)
  weights/
    lsk3dnet_semantickitti.pt  ← you provide this

src/foveamap/models/
  lsk3dnet_adapter.py          ← Phase 8 adapter (wraps LSK3DNet → SegmentationModel)

data/
  lsk3dnet_src/                ← cloned LSK3DNet repo (set LSK3DNet_SRC env var)
```
