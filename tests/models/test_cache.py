"""Round-trip test for ``CachedModel`` (README 6.3, R13, task T9.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foveamap.io.sequence import Sequence
from foveamap.models.cache import CachedModel
from foveamap.pipeline.records import Prediction


def _write_cache(
    cache_dir: Path, model: str, seq: str, idx: int, raw_ids: np.ndarray, conf: np.ndarray
) -> None:
    out_dir = cache_dir / model / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{idx:06d}.npz", raw_ids=raw_ids, conf=conf)


def test_round_trip(tmp_path: Path, synthetic_root: Path) -> None:
    scan = Sequence(synthetic_root, "08").load_frame(0)
    n = len(scan.xyz)
    rng = np.random.default_rng(1337)
    raw_ids = rng.integers(0, 100, size=n).astype(np.uint8)
    conf = rng.integers(0, 256, size=n).astype(np.uint8)
    _write_cache(tmp_path, "lsk3dnet", scan.seq, scan.idx, raw_ids, conf)

    model = CachedModel("lsk3dnet", tmp_path)
    pred = model.predict(scan)

    assert isinstance(pred, Prediction)
    assert pred.raw_ids.dtype == np.uint16
    assert pred.conf.dtype == np.uint8
    np.testing.assert_array_equal(pred.raw_ids, raw_ids.astype(np.uint16))
    np.testing.assert_array_equal(pred.conf, conf)


def test_missing_cache_file_raises(tmp_path: Path, synthetic_root: Path) -> None:
    scan = Sequence(synthetic_root, "08").load_frame(0)
    model = CachedModel("lsk3dnet", tmp_path)
    with pytest.raises(FileNotFoundError, match="cache_predictions.py"):
        model.predict(scan)


def test_wrong_point_count_raises(tmp_path: Path, synthetic_root: Path) -> None:
    scan = Sequence(synthetic_root, "08").load_frame(0)
    wrong_n = len(scan.xyz) - 1
    _write_cache(
        tmp_path,
        "lsk3dnet",
        scan.seq,
        scan.idx,
        np.zeros(wrong_n, dtype=np.uint8),
        np.zeros(wrong_n, dtype=np.uint8),
    )
    model = CachedModel("lsk3dnet", tmp_path)
    with pytest.raises(ValueError, match="do not match"):
        model.predict(scan)
