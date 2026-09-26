from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from foveamap.cli import main
from foveamap.eval.metrics import confusion_iou
from foveamap.eval.semantic_eval import (
    EVAL_EDGES_M,
    evaluate_arrays,
    evaluate_cached_sequence,
    semantic_report_md,
)
from foveamap.io.labels import semantic_ids
from foveamap.io.sequence import Sequence


def test_confusion_iou_known_values() -> None:
    result = confusion_iou(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]), 2)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["iou"] == pytest.approx([0.5, 2 / 3])
    assert result["miou"] == pytest.approx(7 / 12)


def test_six_locked_distance_buckets_and_accounting() -> None:
    ranges = np.array([5, 20, 40, 55, 70, 90, 101], dtype=np.float32)
    xyz = np.column_stack((ranges, np.zeros_like(ranges), np.zeros_like(ranges)))
    gt = np.full(7, 40, dtype=np.uint32)
    pred = gt.astype(np.uint16)
    pred[5] = 0
    result = evaluate_arrays(pred, gt, xyz)
    assert EVAL_EDGES_M == (0, 10, 30, 50, 60, 80, 100)
    assert result["point_accounting"]["input_points"] == 7
    assert result["point_accounting"]["within_100m"] == 6
    assert result["point_accounting"]["beyond_100m_or_invalid"] == 1
    assert result["by_distance"]["80-100 m"]["unknown_prediction_rate"] == 1.0


def test_report_is_generated_from_metrics() -> None:
    xyz = np.array([[1, 0, 0], [12, 0, 0]], dtype=np.float32)
    result = evaluate_arrays(np.array([40, 40]), np.array([40, 40]), xyz)
    result |= {
        "frame_interval": {"start_inclusive": 0, "stop_exclusive": 1, "n_frames": 1},
        "provenance_status": "present",
    }
    md = semantic_report_md(result)
    assert "19-class mIoU: **100.00%**" in md
    assert "80-100 m" in md


def test_cached_sequence_and_cli_write_evidence(synthetic_root: Path, tmp_path: Path) -> None:
    model_name = "test-model"
    cache_dir = tmp_path / "cache" / model_name / "08"
    cache_dir.mkdir(parents=True)
    seq = Sequence(synthetic_root, "08")
    for frame_idx in range(2):
        scan = seq.load_frame(frame_idx)
        np.savez_compressed(
            cache_dir / f"{frame_idx:06d}.npz",
            raw_ids=semantic_ids(scan.raw_labels).astype(np.uint8),
            conf=np.full(len(scan.xyz), 255, dtype=np.uint8),
        )
    (cache_dir / "meta.json").write_text(
        json.dumps({"model_name": model_name, "checkpoint_sha256": "abc123", "gpu": "fixture"}),
        encoding="utf-8",
    )

    report = evaluate_cached_sequence(synthetic_root, tmp_path / "cache", model_name, start=0, stop=2)
    assert report["provenance_status"] == "present"
    assert report["frame_interval"]["n_frames"] == 2
    assert report["semantic_19_class"]["miou"] == pytest.approx(1.0)

    json_out = tmp_path / "semantic.json"
    table_out = tmp_path / "semantic.md"
    assert main([
        "evaluate", "--sequence", "08", "--data-root", str(synthetic_root),
        "--cache-root", str(tmp_path / "cache"), "--model", model_name,
        "--start", "0", "--stop", "2", "--json", str(json_out), "--table", str(table_out),
    ]) == 0
    assert json_out.is_file()
    assert table_out.is_file()
