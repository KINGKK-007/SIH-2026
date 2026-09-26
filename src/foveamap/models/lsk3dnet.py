"""Wrapper around the pretrained LSK3DNet sparse large-kernel network.

``docs/DECISIONS.md`` D-022 overrides README Locked Decision L11: LSK3DNet (CVPR 2024, a sparse-voxel
network built on SPVCNN + dynamically-pruned large kernels, 75.6% mIoU on the SemanticKITTI val split)
is the production segmentation model instead of a range-view network. It is heavier than SalsaNext/CENet,
so this wrapper is written for a small laptop GPU (README's assumed RTX 4050, 6 GB VRAM): batch of one
scan, no test-time-augmentation voting, an optional ``torch.autocast`` fp16 forward pass, and an explicit
``torch.cuda.empty_cache()`` after every scan.

This module does not vendor LSK3DNet's code (task T8.2 normally asks for a vendored copy or a pinned git
submodule): the upstream checkout already lives at the repo root (``LSK3DNet-main/``, MIT licence, no git
history since it was dropped in as a zip) and is imported in place by adding it to ``sys.path``. See the
module docstring of :func:`_import_lsk3dnet` for the two prerequisites this relies on, neither of which
this wrapper can satisfy on its own:

1. A CUDA build of ``spconv`` and ``torch-scatter`` matching the installed ``torch`` + CUDA version
   (LSK3DNet pins ``torch==1.11.0+cu113`` / ``spconv-cu113``, which predate the Ada Lovelace RTX 40-series
   and will not run on it; install current wheels instead, see ``docs/MODEL_CARD.md``).
2. The vendored ``c_utils`` pybind11 extension (``c_gen_normal_map``) built for the target machine. There
   is no working pure-Python fallback in the vendored repo (``LSK3DNet-main/utils/normalmap.py`` ships the
   fallback only as a commented-out reference implementation), so this is mandatory, not optional.
"""

from __future__ import annotations

import copy
import hashlib
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from foveamap.io.labels import learning_to_raw
from foveamap.pipeline.records import Prediction, Scan

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_inference_geometry(lsk_yaml: dict[str, Any], lsk_cfg: Any) -> dict[str, Any]:
    """Return a model config with native or 100 m sparse geometry applied.

    The extended profile changes coordinate bounds and sparse spatial shape but
    preserves the native voxel size on every axis.  This function is deliberately
    independent of torch/spconv so it can be validated on CPU development machines.
    """
    resolved = copy.deepcopy(lsk_yaml)
    native_min = np.asarray(resolved["dataset_params"]["min_volume_space"], dtype=np.float64)
    native_max = np.asarray(resolved["dataset_params"]["max_volume_space"], dtype=np.float64)
    native_shape = np.asarray(resolved["model_params"]["spatial_shape"], dtype=np.int64)

    if lsk_cfg.coverage_profile == "native":
        return resolved

    extended_min = np.asarray(lsk_cfg.extended_min_volume_space, dtype=np.float64)
    extended_max = np.asarray(lsk_cfg.extended_max_volume_space, dtype=np.float64)
    extended_shape = np.asarray(lsk_cfg.extended_spatial_shape, dtype=np.int64)
    native_voxel = (native_max - native_min) / native_shape
    extended_voxel = (extended_max - extended_min) / extended_shape
    if not np.allclose(native_voxel, extended_voxel, rtol=0.0, atol=1e-12):
        raise ValueError(
            "extended_100m geometry must preserve the checkpoint voxel size: "
            f"native={native_voxel.tolist()}, extended={extended_voxel.tolist()}"
        )

    resolved["dataset_params"]["min_volume_space"] = extended_min.tolist()
    resolved["dataset_params"]["max_volume_space"] = extended_max.tolist()
    resolved["dataset_params"]["spatial_shape"] = extended_shape.tolist()
    resolved["model_params"]["spatial_shape"] = extended_shape.tolist()
    return resolved


def checkpoint_sha256(path: str | Path) -> str:
    """SHA-256 of a checkpoint file, in 1 MiB chunks (README 6.3, T9.1 ``meta.json``)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else (_REPO_ROOT / path)


@lru_cache(maxsize=1)
def _import_lsk3dnet(repo_dir: str) -> Any:
    """Import LSK3DNet's ``network``/``utils``/``config`` top-level packages from ``repo_dir``.

    Cached (``lru_cache``) because it mutates ``sys.path`` process-wide: once imported, LSK3DNet's very
    generically-named top-level modules (``utils``, ``network``, ``config``, ``dataloader``, ``builder``)
    stay in ``sys.modules`` for the lifetime of the process. Run inference in an isolated process (as
    ``scripts/cache_predictions.py`` does, per R13: model inference runs once, its output is cached) so
    this never shares a process with unrelated code that expects different modules of the same names.
    """
    lsk_dir = _repo_path(repo_dir)
    if not lsk_dir.is_dir():
        raise RuntimeError(
            f"LSK3DNet checkout not found at {lsk_dir}. Expected the vendored LSK3DNet-main/ directory "
            "(configs/model.yaml: model_params.lsk3dnet.repo_dir)."
        )
    if str(lsk_dir) not in sys.path:
        sys.path.insert(0, str(lsk_dir))

    missing: list[str] = []
    for pkg in ("spconv", "torch_scatter"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise RuntimeError(
            f"missing packages required by LSK3DNet: {', '.join(missing)}. Install CUDA-version-matched "
            "wheels (a CUDA-11.3 build will not run on an Ada Lovelace RTX 40-series GPU) -- see "
            "docs/MODEL_CARD.md for the exact commands."
        )
    try:
        import c_gen_normal_map  # noqa: F401  (import-only availability check)
    except ImportError as exc:
        raise RuntimeError(
            "the compiled c_gen_normal_map extension (LSK3DNet-main/c_utils) is not importable. LSK3DNet "
            "needs it for its per-point normal feature and ships no working pure-Python fallback. Build it "
            "(see docs/MODEL_CARD.md) and make sure the build output directory is on PYTHONPATH."
        ) from exc

    from network.largekernel_model import get_model_class
    from utils.load_save_util import load_checkpoint_model_mask, load_checkpoint_old
    from utils.load_util import load_yaml
    from utils.normalmap import compute_normals_range

    return {
        "get_model_class": get_model_class,
        "load_checkpoint_model_mask": load_checkpoint_model_mask,
        "load_checkpoint_old": load_checkpoint_old,
        "load_yaml": load_yaml,
        "compute_normals_range": compute_normals_range,
    }


class LSK3DNetModel:
    """Sparse large-kernel network, wrapped behind ``SegmentationModel`` (README 9.3).

    Points outside the checkpoint's training crop (``dataset_params.min/max_volume_space`` in the
    checkpoint's own YAML -- by default +-50 m in x/y and [-4, 2] m in z, narrower than the FoveaMap grid
    extent of +-100 m, README L4) would produce out-of-range sparse-conv voxel indices, so they are never
    fed to the network: they come back as ``UNKNOWN`` with ``conf = 0``, same as an unlabelled point, and
    this range gap must show up honestly in the accuracy-by-distance report (T9.3), not be hidden.
    """

    name = "lsk3dnet"

    def __init__(self, model_cfg: Any) -> None:
        import torch

        lsk_cfg = model_cfg.lsk3dnet
        if lsk_cfg is None:
            raise ValueError("configs/model.yaml: family 'sparse_voxel' needs a 'lsk3dnet:' section")
        if not model_cfg.checkpoint:
            raise ValueError("configs/model.yaml: 'checkpoint' is not set (see configs/weights/WEIGHTS.md)")

        mods = _import_lsk3dnet(lsk_cfg.repo_dir)
        checkpoint_path = _repo_path(model_cfg.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint_path}. This is a [HUMAN] step, see "
                "configs/weights/WEIGHTS.md and the LSK3DNet-main README's Model Zoo link."
            )

        self.device = torch.device(model_cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("configs/model.yaml sets device: cuda but torch.cuda.is_available() is False")

        upstream_yaml = mods["load_yaml"](str(_repo_path(lsk_cfg.config_path)))
        lsk_yaml = resolve_inference_geometry(upstream_yaml, lsk_cfg)
        self._min_volume = np.asarray(lsk_yaml["dataset_params"]["min_volume_space"], dtype=np.float64)
        self._max_volume = np.asarray(lsk_yaml["dataset_params"]["max_volume_space"], dtype=np.float64)
        self.coverage_profile = lsk_cfg.coverage_profile

        # EasyDict so the vendored model code's `config['model_params']['x']` attribute-style access works.
        from easydict import EasyDict

        model = mods["get_model_class"](lsk_yaml["model_params"]["model_architecture"])(EasyDict(lsk_yaml))
        try:
            model, _mask = mods["load_checkpoint_model_mask"](str(checkpoint_path), model, self.device)
        except Exception:
            model = mods["load_checkpoint_old"](str(checkpoint_path), model)
        self.model = model.to(self.device).eval()

        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256(checkpoint_path)
        self.half_precision = bool(lsk_cfg.half_precision) and self.device.type == "cuda"
        self.empty_cache_every_scan = bool(lsk_cfg.empty_cache_every_scan)
        self._compute_normals_range = mods["compute_normals_range"]
        self._torch = torch

    def _in_crop_mask(self, xyz: np.ndarray) -> np.ndarray:
        lo, hi = self._min_volume, self._max_volume
        finite = np.isfinite(xyz).all(axis=1)
        inside = np.logical_and(xyz > lo, xyz < hi).all(axis=1)
        return finite & inside

    def predict(self, scan: Scan) -> Prediction:
        torch = self._torch
        n = len(scan.xyz)
        raw_ids = np.zeros(n, dtype=np.uint16)  # UNKNOWN
        conf = np.zeros(n, dtype=np.uint8)

        mask = self._in_crop_mask(scan.xyz)
        n_in_crop = int(mask.sum())
        if n_in_crop:
            xyz = scan.xyz[mask].astype(np.float32)
            sig = scan.remission[mask].astype(np.float32).reshape(-1, 1)
            feat = np.concatenate([xyz, sig], axis=1)
            normal = self._compute_normals_range(feat).astype(np.float32)

            points_t = torch.from_numpy(feat).to(self.device)
            normal_t = torch.from_numpy(np.ascontiguousarray(normal)).to(self.device)
            batch_idx_t = torch.zeros(n_in_crop, dtype=torch.long, device=self.device)
            data_dict = {
                "points": points_t,
                "normal": normal_t,
                "batch_idx": batch_idx_t,
                "batch_size": 1,
            }

            with torch.no_grad():
                if self.half_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(data_dict)
                else:
                    out = self.model(data_dict)
                probs = torch.softmax(out["logits"].float(), dim=1)
                conf_in, learning_in = probs.max(dim=1)

            learning_np = learning_in.to(torch.uint8).cpu().numpy()
            conf_np = (conf_in.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            raw_ids[mask] = learning_to_raw(learning_np)
            conf[mask] = conf_np

            if self.empty_cache_every_scan and self.device.type == "cuda":
                torch.cuda.empty_cache()

        if n_in_crop < n:
            warnings.warn(
                f"LSK3DNetModel: {n - n_in_crop}/{n} points fall outside the configured inference volume "
                f"({self._min_volume.tolist()} .. {self._max_volume.tolist()} m) and are reported as "
                "UNKNOWN with conf=0",
                stacklevel=2,
            )

        return Prediction(raw_ids=raw_ids, conf=conf)
