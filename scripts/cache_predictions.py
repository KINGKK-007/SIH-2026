"""Write the prediction cache for the selected model (README 6.3, R13, task T9.1).

Usage: python scripts/cache_predictions.py --sequences 04 07 08 [--data-root data/dataset]
                                            [--config-dir configs] [--overwrite] [--dry-run N]

Runs the model named in ``configs/model.yaml`` (``model_params.name``) once per scan and writes
``data/cache/pred/<model>/<seq>/<frame:06d>.npz`` (arrays ``raw_ids`` uint8, ``conf`` uint8) plus one
``data/cache/pred/<model>/<seq>/meta.json`` per sequence (model name, checkpoint SHA-256, FoveaMap and
LSK3DNet-main provenance, GPU, torch version, run timestamp). Resumable: an existing ``<frame>.npz`` is
skipped unless ``--overwrite`` is given, so an interrupted run only re-does the remaining frames.

GPU required (the model runs live here); the resulting cache needs no GPU downstream (R13). This is
normally run on the GPU machine (a teammate's RTX 4050 laptop, D-022) and the ``data/cache/pred/`` tree
copied onto the machine that renders the dashboard or computes benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from foveamap.config import load_config
from foveamap.io.sequence import Sequence

ROOT = Path(__file__).resolve().parents[1]


def _build_model(model_cfg):
    if model_cfg.family == "sparse_voxel":
        from foveamap.models.lsk3dnet import LSK3DNetModel

        return LSK3DNetModel(model_cfg)
    raise NotImplementedError(
        f"family {model_cfg.family!r} has no wrapper yet (models/rangeview.py, task T8.2)."
    )


def _run_meta(model, model_cfg) -> dict:
    import torch

    return {
        "model_name": model_cfg.name,
        "family": model_cfg.family,
        "checkpoint": str(model.checkpoint_path),
        "checkpoint_sha256": model.checkpoint_sha256,
        "lsk3dnet_repo_dir": model_cfg.lsk3dnet.repo_dir if model_cfg.lsk3dnet else None,
        "half_precision": getattr(model, "half_precision", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def cache_sequence(
    model, model_cfg, data_root: str, seq: str, cache_root: Path, overwrite: bool, dry_run: int | None
) -> tuple[int, int]:
    """Returns (n_written, n_total)."""
    sequence = Sequence(data_root, seq, with_labels=False)
    n_total = len(sequence) if dry_run is None else min(dry_run, len(sequence))
    out_dir = cache_root / model_cfg.name / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for i in tqdm(range(n_total), desc=f"sequence {seq}", unit="scan"):
        scan = sequence.load_frame(sequence.frame_index(i))
        out_path = out_dir / f"{scan.idx:06d}.npz"
        if out_path.exists() and not overwrite:
            continue
        pred = model.predict(scan)
        np.savez_compressed(
            out_path,
            raw_ids=pred.raw_ids.astype(np.uint8),
            conf=pred.conf.astype(np.uint8),
        )
        n_written += 1

    meta_path = out_dir / "meta.json"
    meta = _run_meta(model, model_cfg) | {"sequence": seq, "n_frames": n_total}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return n_written, n_total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", nargs="+", default=["04", "07", "08"])
    parser.add_argument(
        "--data-root", default=os.environ.get("FOVEAMAP_DATA_ROOT", str(ROOT / "data" / "dataset"))
    )
    parser.add_argument("--config-dir", default=str(ROOT / "configs"))
    parser.add_argument("--overwrite", action="store_true", help="recompute frames that are already cached")
    parser.add_argument(
        "--dry-run", type=int, default=None, metavar="N", help="cache only the first N frames"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir)
    model_cfg = cfg.model
    if not model_cfg.name:
        print("configs/model.yaml: model_params.name is not set (Gate 8 not done yet)", file=sys.stderr)
        return 2

    model = _build_model(model_cfg)
    cache_root = Path(model_cfg.cache_dir)

    ok = True
    for seq in args.sequences:
        n_written, n_total = cache_sequence(
            model, model_cfg, args.data_root, seq, cache_root, args.overwrite, args.dry_run
        )
        print(f"sequence {seq}: wrote {n_written} new file(s), {n_total} frame(s) total")
        if args.dry_run is None:
            cached = sorted((cache_root / model_cfg.name / seq).glob("*.npz"))
            if len(cached) != n_total:
                print(f"  MISMATCH: {len(cached)} cache files for {n_total} frames", file=sys.stderr)
                ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
